"""Predeclared depth-10 taxi-out residual with released ground-traffic context.

The 24 neighbour covariates use released departure AOBT and movement fields;
the eight runway sequence fields use released movement metadata. No departure
block time or taxi-time labels enter these caches. Model choice uses 2025
labels only; ranking predictions are never used for selection.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from catboost import CatBoostRegressor, Pool

import deep_arrival_expert as arrival
import deep_timestamp_expert as deep
from proxy_neighbour_expert import feature_names as neighbour_feature_names
from runway_arrival_features import FEATURES as RUNWAY_FEATURES


FOLDS = deep.FOLDS
WEIGHTS = deep.WEIGHTS
REFERENCE_COLUMNS = ("MVT_ID_mvt", "target", "fold", "airport", "month",
                     "MVT_TIME_UTC_mvt", "a_valid")
EXPECTED_OOF_ROWS = 672428
EXPECTED_RANKING_ROWS = 344841


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def assert_exact_ids(actual: pd.Series, expected: pd.Series, name: str) -> None:
    a, e = pd.Index(actual), pd.Index(expected)
    if (a.has_duplicates or a.isna().any() or e.has_duplicates or e.isna().any()
            or len(a) != len(e) or not a.isin(e).all() or not e.isin(a).all()):
        raise ValueError(f"{name} must cover the exact unique reference IDs")


def freeze_reference(args: argparse.Namespace) -> dict:
    """Freeze the v6 candidate as a small, immutable all-finite OOF reference."""
    source_hash = digest(args.source_reference)
    path = args.output_dir / "frozen_v6_oof_reference.parquet"
    if not path.exists():
        source = pd.read_parquet(args.source_reference)
        if (len(source) != EXPECTED_OOF_ROWS or source.MVT_ID_mvt.duplicated().any()
                or source.MVT_ID_mvt.isna().any()):
            raise ValueError("v6 source reference row universe changed")
        if any(c not in source for c in (*REFERENCE_COLUMNS, "candidate")):
            raise ValueError("v6 source reference lacks required OOF fields")
        if not np.isfinite(source[["target", "candidate"]].to_numpy(dtype=float)).all():
            raise ValueError("v6 candidate and all scoring targets must be finite")
        if set(source.fold.unique()) != set(FOLDS):
            raise ValueError("v6 OOF reference folds differ from the declared two folds")
        frozen = source.loc[:, [*REFERENCE_COLUMNS, "candidate"]].rename(
            columns={"candidate": "selected"})
        # In particular, no source expert column survives this projection.
        path.parent.mkdir(parents=True, exist_ok=True)
        frozen.to_parquet(path, index=False)
    frozen = pd.read_parquet(path)
    if list(frozen) != [*REFERENCE_COLUMNS, "selected"]:
        raise ValueError("Frozen reference must contain only declared reference fields")
    if (len(frozen) != EXPECTED_OOF_ROWS or frozen.MVT_ID_mvt.duplicated().any()
            or frozen.MVT_ID_mvt.isna().any()
            or not np.isfinite(frozen[["target", "selected"]].to_numpy(dtype=float)).all()):
        raise ValueError("Frozen reference is incomplete or nonfinite")
    # The source may be rewritten by another experiment; refuse to silently
    # bind the same frozen reference to different source bytes.
    source = pd.read_parquet(args.source_reference,
                             columns=[*REFERENCE_COLUMNS, "candidate"])
    check = source.rename(columns={"candidate": "selected"})
    if not frozen.equals(check):
        raise ValueError("Frozen v6 candidate differs from current source reference")
    return {"source_reference_sha256": source_hash,
            "frozen_reference_sha256": digest(path)}


def protocol(args: argparse.Namespace) -> dict:
    references = freeze_reference(args)
    neighbour_path = args.neighbour_dir / "training_neighbour_features.parquet"
    neighbour_ranking = args.neighbour_dir / "ranking_neighbour_features.parquet"
    runway_path = args.runway_dir / "training_runway_arrival_features.parquet"
    runway_ranking = args.runway_dir / "ranking_runway_arrival_features.parquet"
    arr_path = args.arrival_dir / "training_arrival_features.parquet"
    prior_oof = args.prior_fresh_oof
    value = {
        "purpose": "2025 local comparison; no leaderboard, upload, or ranking selection",
        "references": references,
        "input_sha256": {
            "training_rows": digest(args.cache_dir / "training_rows.parquet"),
            "arrival_training": digest(arr_path),
            "neighbour_training": digest(neighbour_path),
            "neighbour_ranking": digest(neighbour_ranking),
            "runway_training": digest(runway_path),
            "runway_ranking": digest(runway_ranking),
            "fresh_prior_depth10_arrival_oof": digest(prior_oof),
            "neighbour_feature_protocol": digest(args.neighbour_dir / "protocol.json"),
            "runway_feature_protocol": digest(args.runway_dir / "protocol.json"),
        },
        "source_paths": {
            "v6_oof": str(args.source_reference),
            "frozen_v6_oof": str(args.output_dir / "frozen_v6_oof_reference.parquet"),
            "ranking_reference_if_promoted": str(args.ranking_reference),
            "fresh_prior_oof": str(prior_oof),
        },
        "architecture": {
            "algorithm": "CatBoost GPU residual to own AOBT proxy",
            "depth": 10, "max_iterations": 10000, "random_seed": 2026,
            "other_params": "deep_timestamp_expert.params unchanged",
            "training_labels": "finite 2025 labels in 0..86400 seconds, own proxy 0..7200",
            "heldout_scoring_labels": "all finite, including extreme and negative",
            "features": "deep timestamp + fixed 14 ARR + fixed 24 neighbour + fixed 8 runway sequence covariates",
            "neighbour_feature_names": neighbour_feature_names(),
            "runway_feature_names": list(RUNWAY_FEATURES),
        },
        "selection": {
            "months": [1, 7], "forward_months": [11, 12],
            "weights": list(WEIGHTS),
            "criterion": "minimum seasonal all-finite RMSE, ties to smaller weight",
            "forward": "same selected weight without adjustment",
            "paired_day_bootstrap": "1000 UTC-day resamples, 95% percentile interval",
            "gate": "nonzero weight and CI lower > 0 separately in both folds",
        },
        "fresh_audit": {
            "months": [4, 10],
            "excluded_from_both_fits": [4, 10],
            "prior": "saved depth10 ARR OOF from v6 fresh_new",
            "candidate": "new depth10 ARR+neighbour+runway OOF with the same architecture",
            "weight": "same selected seasonal weight; no retuning",
            "gate": "exact full valid-ID coverage, finite predictions, paired day CI lower > 0",
        },
        "limitations": "Repeated 2025 model comparisons and inherited base OOF training overlap make this a comparison estimate, not untouched 2026 performance.",
    }
    path = args.output_dir / "protocol.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise ValueError("Frozen v7 experiment protocol or an input hash changed")
    else:
        write_json(path, value)
    return value


def load_features(args: argparse.Namespace, ranking: bool = False):
    rows, features = arrival.load_features(args, ranking=ranking)
    path = args.neighbour_dir / (
        "ranking_neighbour_features.parquet" if ranking
        else "training_neighbour_features.parquet")
    neighbour = pd.read_parquet(path)
    if not np.array_equal(neighbour.MVT_ID_mvt.to_numpy(), rows.MVT_ID_mvt.to_numpy()):
        raise ValueError("Neighbour feature cache does not align with departure IDs")
    names = neighbour_feature_names()
    if list(neighbour.columns) != ["MVT_ID_mvt", *names]:
        raise ValueError("The fixed 24-neighbour feature schema changed")
    if len(names) != 24 or len(features.columns.intersection(names)):
        raise ValueError("Neighbour predictors collide with existing predictors")
    features = pd.concat([features.reset_index(drop=True),
                          neighbour[names].reset_index(drop=True)], axis=1)
    runway_path = args.runway_dir / (
        "ranking_runway_arrival_features.parquet" if ranking
        else "training_runway_arrival_features.parquet")
    runway = pd.read_parquet(runway_path)
    if not np.array_equal(runway.MVT_ID_mvt.to_numpy(), rows.MVT_ID_mvt.to_numpy()):
        raise ValueError("Runway feature cache does not align with departure IDs")
    runway_names = list(RUNWAY_FEATURES)
    if len(runway_names) != 8 or any(name not in runway for name in runway_names):
        raise ValueError("Fixed runway feature schema changed")
    if len(features.columns.intersection(runway_names)):
        raise ValueError("Runway predictors collide with existing predictors")
    features = pd.concat([features.reset_index(drop=True),
                          runway[runway_names].reset_index(drop=True)], axis=1)
    if features.columns.duplicated().any():
        raise ValueError("Duplicate predictor names")
    return rows, features


def load_reference(args: argparse.Namespace) -> pd.DataFrame:
    ref = pd.read_parquet(args.output_dir / "frozen_v6_oof_reference.parquet")
    if len(ref) != EXPECTED_OOF_ROWS or set(ref.fold.unique()) != set(FOLDS):
        raise ValueError("Frozen OOF reference row universe changed")
    if not np.isfinite(ref[["target", "selected"]].to_numpy(dtype=float)).all():
        raise ValueError("Frozen OOF reference contains nonfinite score values")
    return ref


def evaluate(args: argparse.Namespace, include_audit: bool = True) -> dict:
    frozen_protocol = protocol(args)
    ref = load_reference(args)
    report = {"protocol": frozen_protocol, "folds": {},
              "selected_weight": None, "existing_folds_passed": False,
              "promoted": False}
    parts = []
    for name in FOLDS:
        held = ref.loc[ref.fold.eq(name)]
        expert = pd.read_parquet(args.output_dir / f"{name}_oof.parquet")
        expected_ids = held.loc[held.a_valid, "MVT_ID_mvt"]
        assert_exact_ids(expert.MVT_ID_mvt, expected_ids, f"{name} expert")
        if not np.isfinite(expert.expert.to_numpy(dtype=float)).all():
            raise ValueError(f"{name} expert contains nonfinite predictions")
        part = held.merge(expert, on="MVT_ID_mvt", how="left",
                          validate="one_to_one", sort=False)
        present = part.expert.notna().to_numpy()
        if not np.array_equal(present, part.a_valid.to_numpy(dtype=bool)):
            raise ValueError(f"{name} valid-AOBT prediction mask differs")
        base = part.selected.to_numpy(dtype=float)
        alt = part.expert.fillna(part.selected).to_numpy(dtype=float)
        y = part.target.to_numpy(dtype=float)
        scores = {str(w): deep.rmse(y, np.maximum(base + w * (alt-base), 0))
                  for w in WEIGHTS}
        if name == "seasonal_jan_jul":
            report["selected_weight"] = min(WEIGHTS, key=lambda w: (scores[str(w)], w))
        weight = report["selected_weight"]
        candidate = np.maximum(base + weight * (alt-base), 0)
        info = json.loads((args.output_dir / f"{name}_validation.json").read_text())
        report["folds"][name] = {
            "n_all_finite": len(part), "n_valid_aobt": int(present.sum()),
            "scores": scores, "trees": info["trees"],
            "fit_seconds": info["fit_seconds"],
            "bootstrap": arrival.bootstrap(part, base, candidate),
            "oof_sha256": digest(args.output_dir / f"{name}_oof.parquet"),
        }
        part["candidate"] = candidate
        parts.append(part)
    report["existing_folds_passed"] = (report["selected_weight"] > 0
        and all(f["bootstrap"]["gain_ci95_sec"][0] > 0
                for f in report["folds"].values()))
    audit_path = args.output_dir / "fresh_audit.json"
    if include_audit and audit_path.exists():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if not audit.get("coverage_verified") or audit["weight"] != report["selected_weight"]:
            raise ValueError("Fresh audit coverage or preselected weight differs")
        for key, path in (
            ("prior_oof_sha256", args.prior_fresh_oof),
            ("new_oof_sha256", args.output_dir / "fresh_new/fresh_apr_oct_oof.parquet"),
            ("paired_predictions_sha256", args.output_dir / "fresh_audit_predictions.parquet"),
        ):
            if audit[key] != digest(path):
                raise ValueError(f"Fresh audit input changed: {key}")
        report["fresh_audit"] = audit
        report["promoted"] = report["existing_folds_passed"] and audit["passed"]
    path = args.output_dir / "validation_predictions.parquet"
    pd.concat(parts, ignore_index=True).to_parquet(path, index=False)
    report["validation_predictions_sha256"] = digest(path)
    write_json(args.output_dir / "validation.json", report)
    print(json.dumps({"selected_weight": report["selected_weight"],
                      "existing_folds_passed": report["existing_folds_passed"],
                      "promoted": report["promoted"],
                      "folds": report["folds"]}, indent=2), flush=True)
    return report


def fit(args: argparse.Namespace) -> None:
    protocol(args)
    rows, features = load_features(args)
    ref = load_reference(args)
    for name, months in FOLDS.items():
        if not (args.output_dir / f"{name}_oof.parquet").exists():
            deep.fit_fold(name, months, rows, features, ref, args)
    del rows, features, ref
    gc.collect()
    evaluate(args)


def fresh_audit(args: argparse.Namespace) -> None:
    """Fit only the new architecture, excluding April/October labels."""
    report = evaluate(args, include_audit=False)
    if not report["existing_folds_passed"]:
        raise ValueError("Existing-fold gates failed before fresh audit")
    rows, features = load_features(args)
    proxy = rows.proxy.to_numpy(dtype=float)
    y = rows.target.to_numpy(dtype=float)
    held = rows.month.isin((4, 10)).to_numpy()
    valid = np.isfinite(proxy) & (proxy >= 0) & (proxy <= 7200) & np.isfinite(y)
    ref = rows.loc[held & valid, ["MVT_ID_mvt", "target", "time"]].copy()
    ref.rename(columns={"time": "MVT_TIME_UTC_mvt"}, inplace=True)
    ref["selected"] = proxy[held & valid]
    ref["fold"] = "fresh_apr_oct"
    expected_ids = ref.MVT_ID_mvt
    if expected_ids.duplicated().any() or expected_ids.isna().any():
        raise ValueError("Fresh held-out reference IDs invalid")
    new_args = argparse.Namespace(**vars(args))
    new_args.output_dir = args.output_dir / "fresh_new"
    new_path = new_args.output_dir / "fresh_apr_oct_oof.parquet"
    if not new_path.exists():
        deep.fit_fold("fresh_apr_oct", (4, 10), rows, features, ref, new_args)
    prior = pd.read_parquet(args.prior_fresh_oof).rename(columns={"expert": "prior"})
    new = pd.read_parquet(new_path).rename(columns={"expert": "new"})
    for name, frame in (("prior", prior), ("new", new)):
        assert_exact_ids(frame.MVT_ID_mvt, expected_ids, f"Fresh {name} OOF")
        if not np.isfinite(frame[name].to_numpy(dtype=float)).all():
            raise ValueError(f"Fresh {name} OOF predictions must be finite")
    paired = ref.merge(prior, on="MVT_ID_mvt", how="left", sort=False,
                       validate="one_to_one").merge(new, on="MVT_ID_mvt",
                                                     how="left", sort=False,
                                                     validate="one_to_one")
    if (len(paired) != len(ref) or
            not np.array_equal(paired.MVT_ID_mvt.to_numpy(), expected_ids.to_numpy()) or
            not np.isfinite(paired[["target", "prior", "new"]].to_numpy(dtype=float)).all()):
        raise ValueError("Fresh paired OOF predictions are incomplete or misaligned")
    base = np.maximum(paired.prior.to_numpy(dtype=float), 0)
    candidate = np.maximum(base + report["selected_weight"] *
                           (paired.new.to_numpy(dtype=float) - base), 0)
    paired["candidate"] = candidate
    path = args.output_dir / "fresh_audit_predictions.parquet"
    paired.to_parquet(path, index=False)
    bootstrap = arrival.bootstrap(paired, base, candidate, seed=20261007)
    audit = {
        "months": [4, 10], "weight": report["selected_weight"],
        "n_valid_aobt": len(paired), "coverage_verified": True,
        "prior_depth10_arrival_rmse": deep.rmse(paired.target.to_numpy(dtype=float), base),
        "blended_depth10_arrival_neighbour_runway_rmse": deep.rmse(
            paired.target.to_numpy(dtype=float), candidate),
        "bootstrap": bootstrap, "passed": bootstrap["gain_ci95_sec"][0] > 0,
        "prior_oof_sha256": digest(args.prior_fresh_oof),
        "new_oof_sha256": digest(new_path),
        "paired_predictions_sha256": digest(path),
        "scope": "Independent paired architecture comparison, not full ensemble OOF",
    }
    write_json(args.output_dir / "fresh_audit.json", audit)
    del rows, features
    gc.collect()
    evaluate(args)


def final_predict(args: argparse.Namespace) -> None:
    """Fit all eligible 2025 labels only after every frozen local gate passes."""
    report = evaluate(args)
    if not report["promoted"]:
        raise ValueError("Combined expert failed the predeclared local gates")
    frozen = pd.read_parquet(args.ranking_reference)
    if (list(frozen) != ["MVT_ID_mvt", "TAXITIME_SEC_mvt"]
            or len(frozen) != EXPECTED_RANKING_ROWS
            or frozen.MVT_ID_mvt.duplicated().any()
            or frozen.MVT_ID_mvt.isna().any()
            or not np.isfinite(frozen.TAXITIME_SEC_mvt.to_numpy(dtype=float)).all()):
        raise ValueError("v6 ranking reference schema, IDs, or predictions changed")
    model_path = args.output_dir / "full_2025.cbm"
    params = deep.params(args)
    params["iterations"] = int(np.median(
        [row["trees"] for row in report["folds"].values()]))
    model = CatBoostRegressor(**params)
    fit_seconds = None
    if not model_path.exists():
        rows, features = load_features(args)
        proxy = rows.proxy.to_numpy(dtype=float)
        target = rows.target.to_numpy(dtype=float)
        core = (np.isfinite(target) & (target >= 0) & (target <= 86400)
                & np.isfinite(proxy) & (proxy >= 0) & (proxy <= 7200))
        cats = features.select_dtypes(include="category").columns.tolist()
        pool = Pool(features.loc[core], label=(target-proxy)[core],
                    cat_features=cats)
        start = time.monotonic()
        model.fit(pool)
        fit_seconds = time.monotonic() - start
        model.save_model(str(model_path))
        del pool, rows, features
        gc.collect()
    else:
        model.load_model(str(model_path))
    rank_rows, rank_features = load_features(args, ranking=True)
    if not np.array_equal(frozen.MVT_ID_mvt.to_numpy(),
                          rank_rows.MVT_ID_mvt.to_numpy()):
        raise ValueError("v6 ranking reference ID order differs from feature cache")
    rank_proxy = rank_rows.proxy.to_numpy(dtype=float)
    valid = (np.isfinite(rank_proxy) & (rank_proxy >= 0)
             & (rank_proxy <= 7200))
    expert_values = np.full(len(rank_rows), np.nan, dtype=float)
    expert_values[valid] = (rank_proxy[valid]
                            + model.predict(rank_features.loc[valid],
                                            thread_count=args.threads))
    if not np.isfinite(expert_values[valid]).all():
        raise ValueError("Eligible ranking expert predictions must be finite")
    raw = pd.DataFrame({"MVT_ID_mvt": rank_rows.MVT_ID_mvt.to_numpy(copy=True),
                        "expert": expert_values})
    raw_path = args.output_dir / "ranking_expert.parquet"
    raw.to_parquet(raw_path, index=False)
    base = frozen.merge(raw, on="MVT_ID_mvt", validate="one_to_one", sort=False)
    present = base.expert.notna().to_numpy()
    if len(base) != EXPECTED_RANKING_ROWS or not np.array_equal(present, valid):
        raise ValueError("Ranking expert coverage differs from the valid-AOBT gate")
    # pandas 3 uses copy-on-write: the mutable blend array must own its storage.
    values = base.TAXITIME_SEC_mvt.to_numpy(dtype=float, copy=True)
    alternate = base.expert.to_numpy(dtype=float, copy=True)
    weight = report["selected_weight"]
    values[present] = np.maximum(values[present]
                                 + weight * (alternate[present]-values[present]), 0)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Final ranking predictions must be finite and nonnegative")
    result = pd.DataFrame({"MVT_ID_mvt": base.MVT_ID_mvt.to_numpy(copy=True),
                           "TAXITIME_SEC_mvt": values})
    result_path = args.output_dir / "predictions.parquet"
    result.to_parquet(result_path, index=False)
    write_json(args.output_dir / "manifest.json", {
        "validation_sha256": digest(args.output_dir / "validation.json"),
        "fresh_audit_sha256": digest(args.output_dir / "fresh_audit.json"),
        "ranking_reference_sha256": digest(args.ranking_reference),
        "ranking_rows_sha256": digest(args.cache_dir / "ranking_rows.parquet"),
        "ranking_arrival_features_sha256": digest(
            args.arrival_dir / "ranking_arrival_features.parquet"),
        "ranking_expert_sha256": digest(raw_path),
        "predictions_sha256": digest(result_path),
        "model_sha256": digest(model_path),
        "rows": len(result), "valid_aobt": int(valid.sum()),
        "blend_weight": weight, "trees": model.tree_count_,
        "full_fit_seconds": fit_seconds,
        "feature_names": list(rank_features),
        "selection": "2025 labels only; all predeclared local gates passed",
    })
    print(json.dumps({"rows": len(result), "valid_aobt": int(valid.sum()),
                      "blend_weight": weight, "trees": model.tree_count_,
                      "full_fit_seconds": fit_seconds}, indent=2), flush=True)


def prepare(args: argparse.Namespace) -> None:
    value = protocol(args)
    print(json.dumps({"protocol_path": str(args.output_dir / "protocol.json"),
                      "frozen_reference_path": str(args.output_dir / "frozen_v6_oof_reference.parquet"),
                      "reference_sha256": value["references"]}, indent=2), flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("prepare", "fit", "evaluate", "fresh-audit",
                                      "final-predict"),
                   default="prepare")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--cache-dir", type=Path, default=Path("artifacts/baseline"))
    p.add_argument("--weather-file", type=Path,
                   default=Path("data/external/weather.parquet"))
    p.add_argument("--arrival-dir", type=Path,
                   default=Path("artifacts/v5-arrival-clean"))
    p.add_argument("--neighbour-dir", type=Path,
                   default=Path("artifacts/v6-neighbour"))
    p.add_argument("--runway-dir", type=Path,
                   default=Path("artifacts/v6-runway-arrival"))
    p.add_argument("--source-reference", type=Path,
                   default=Path("artifacts/v6-deep-arrival/validation_predictions.parquet"))
    p.add_argument("--prior-fresh-oof", type=Path,
                   default=Path("artifacts/v6-deep-arrival/fresh_new/fresh_apr_oct_oof.parquet"))
    p.add_argument("--ranking-reference", type=Path,
                   default=Path("submissions/merry-mushroom_v6.parquet"))
    p.add_argument("--output-dir", type=Path,
                   default=Path("artifacts/v7-runway-traffic"))
    args = p.parse_args()
    # Architecture stays fixed to the earlier depth-10 ARR experiment.
    args.iterations = 10000
    args.depth = 10
    args.threads = 2
    {"prepare": prepare, "fit": fit, "evaluate": evaluate,
     "fresh-audit": fresh_audit,
     "final-predict": final_predict}[args.mode](args)


if __name__ == "__main__":
    main()
