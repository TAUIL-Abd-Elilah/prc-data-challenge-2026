"""Prospective v10 comparison using released same-runway ARR taxi context.

This code is prepared before constructing the ten fixed ARR taxi features.
It reuses the accepted v7 depth-10 CatBoost feature loader and the existing
complementary-month residual trainer. No ranking predictions or final model
can be produced from this experiment without a separately frozen later guard.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import deep_arrival_expert as arrival
import deep_timestamp_expert as deep
import traffic_deep_expert as traffic
from runway_arrival_taxi_features import ARR_COLUMNS, DEP_COLUMNS
from runway_arrival_taxi_features import FEATURES as TAXI_FEATURES
from runway_arrival_taxi_features import require_memory
from solution import _training_files


FOLDS = deep.FOLDS
WEIGHTS = (0.0, 0.1, 0.25, 0.5, 1.0)
FRESH_MONTHS = (4, 10)
REFERENCE_COLUMNS = ("MVT_ID_mvt", "target", "fold", "airport", "month",
                     "MVT_TIME_UTC_mvt", "a_valid", "selected")
EXPECTED_OOF_ROWS = 672428
BOOTSTRAP_SEED = 20261012


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def exact_ids(actual: pd.Series, expected: pd.Series, label: str) -> None:
    left, right = pd.Index(actual), pd.Index(expected)
    if (len(left) != len(right) or left.has_duplicates or right.has_duplicates
            or left.isna().any() or right.isna().any()
            or not left.isin(right).all() or not right.isin(left).all()):
        raise ValueError(f"{label}: exact unique ID coverage failed")


def protocol_spec() -> dict:
    """Fixed before any v10 cache build or label-based model comparison."""
    return {
        "purpose": "One 2025 local comparison of fixed released-ARR taxi covariates",
        "architecture": {
            "base": "v7 depth-10 traffic CatBoost residual to own AOBT proxy",
            "trainer": "deep_timestamp_expert.fit_fold, unchanged",
            "depth": 10, "max_iterations": 10000, "random_seed": 2026,
            "other_params": "deep_timestamp_expert.params, unchanged",
            "additional_features": list(TAXI_FEATURES),
            "additional_feature_count": 10,
            "training_labels": "finite 2025 labels in [0,86400], own AOBT proxy in [0,7200]",
            "scoring_labels": "all finite, including negative/extreme",
            "forbidden_prediction_sources": ["own departure BLOCK", "own departure TAXITIME",
                                             "opaque movement IDs", "ranking labels"],
            "ARR_availability": "released retrospective batch ARR BLOCK/TAXITIME, not real-time forecasting",
        },
        "existing_folds": {
            "selection_months": [1, 7], "forward_months": [11, 12],
            "weights": list(WEIGHTS),
            "selection": "minimum clipped all-finite Jan/Jul RMSE; ties favor smaller weight",
            "forward": "use the same Jan/Jul-selected weight without adjustment",
            "gate": "nonzero selected weight, RMSE improvement and paired UTC-day CI lower >0 in both folds",
            "reference": "frozen v7 candidate all-finite OOF, with invalid-AOBT rows unchanged",
        },
        "fresh_matched_audit": {
            "months": list(FRESH_MONTHS),
            "reference": "saved v7 fresh_audit_predictions candidate; both v7 component models excluded April/October labels",
            "candidate": "new v10 depth-10 residual expert excluding April/October labels from fit and early stopping",
            "weight": "unchanged selected Jan/Jul blend weight",
            "gate": "exact all finite-label valid-AOBT ID/label/time coverage; each month improves RMSE and pooled paired-day CI lower >0",
        },
        "february_august_policy": "Their labels may occur in complementary training but do not select this feature family, architecture, blend or gates",
        "may_september_policy": "Remain reserved; no later guard is specified or run by this script",
        "ranking_status": "No final fit, ranking predictions, upload or submission from this script",
        "limitations": "Architecture comparison after repeated 2025 exploration; not an unbiased complete ensemble estimate or guaranteed ranking performance",
        "leaderboard_use": False,
    }


def freeze_reference(args: argparse.Namespace) -> dict:
    """Keep only the accepted v7 candidate and mandatory OOF metadata."""
    source_path = args.v7_dir / "validation_predictions.parquet"
    report = json.loads((args.v7_dir / "validation.json").read_text(encoding="utf-8"))
    if (not report.get("promoted") or report.get("validation_predictions_sha256")
            != sha256(source_path)):
        raise ValueError("Accepted v7 candidate report or OOF changed")
    source = pd.read_parquet(source_path,
                             columns=[*REFERENCE_COLUMNS[:-1], "candidate"])
    if (len(source) != EXPECTED_OOF_ROWS or source.MVT_ID_mvt.isna().any()
            or source.MVT_ID_mvt.duplicated().any()
            or set(source.fold.unique()) != set(FOLDS)
            or any(not source.loc[source.fold.eq(name), "month"].isin(months).all()
                   for name, months in FOLDS.items())
            or not np.isfinite(source[["target", "candidate"]]
                               .to_numpy(dtype=float)).all()
            or (source.candidate.to_numpy(dtype=float) < 0).any()):
        raise ValueError("v7 candidate all-finite OOF reference is invalid")
    expected = pd.read_parquet(args.cache_dir / "training_rows.parquet",
                               columns=["MVT_ID_mvt", "target", "proxy", "month",
                                        "airport", "time"])
    expected = expected.loc[
        expected.month.isin((1, 7, 11, 12)).to_numpy()
        & np.isfinite(expected.target.to_numpy(dtype=float))]
    exact_ids(source.MVT_ID_mvt, expected.MVT_ID_mvt, "v7 frozen reference")
    aligned = expected.merge(source, on="MVT_ID_mvt", how="left", sort=False,
                             validate="one_to_one", suffixes=("_baseline", ""))
    valid = np.isfinite(aligned.proxy.to_numpy(dtype=float)) & (
        aligned.proxy.to_numpy(dtype=float) >= 0) & (
        aligned.proxy.to_numpy(dtype=float) <= 7200)
    if (not np.array_equal(aligned.target_baseline.to_numpy(dtype=float),
                           aligned.target.to_numpy(dtype=float))
            or not np.array_equal(aligned.month_baseline.to_numpy(),
                                  aligned.month.to_numpy())
            or not np.array_equal(aligned.airport_baseline.astype("string").to_numpy(),
                                  aligned.airport.astype("string").to_numpy())
            or not np.array_equal(aligned.a_valid.to_numpy(dtype=bool), valid)
            or not np.array_equal(pd.to_datetime(aligned.time, utc=True).to_numpy(),
                                  pd.to_datetime(aligned.MVT_TIME_UTC_mvt,
                                                 utc=True).to_numpy())):
        raise ValueError("v7 reference IDs, labels, airport, month or time disagree")
    frozen = source.rename(columns={"candidate": "selected"})
    frozen_path = args.output_dir / "frozen_v7_oof_reference.parquet"
    if frozen_path.exists():
        if not pd.read_parquet(frozen_path).equals(frozen):
            raise ValueError("Previously frozen v7 candidate reference differs")
    else:
        frozen_path.parent.mkdir(parents=True, exist_ok=True)
        frozen.to_parquet(frozen_path, index=False)
    return {"source_reference_sha256": sha256(source_path),
            "frozen_reference_sha256": sha256(frozen_path)}


def protocol(args: argparse.Namespace) -> dict:
    require_memory(max(10.0, args.min_free_gib))
    if len(TAXI_FEATURES) != 10 or len(set(TAXI_FEATURES)) != 10:
        raise ValueError("Fixed runway ARR taxi feature list changed")
    references = freeze_reference(args)
    raw = _training_files(args.data_dir)
    if len(raw) != 12 or len({path.name for path in raw}) != 12:
        raise ValueError("Expected twelve canonical 2025 source files")
    paths = {
        "own_script": Path(__file__).resolve(),
        "v7_loader": Path(traffic.__file__).resolve(),
        "residual_trainer": Path(deep.__file__).resolve(),
        "v7_protocol": args.v7_dir / "protocol.json",
        "v7_validation": args.v7_dir / "validation.json",
        "v7_oof": args.v7_dir / "validation_predictions.parquet",
        "v7_fresh_report": args.v7_dir / "fresh_audit.json",
        "v7_fresh_predictions": args.v7_dir / "fresh_audit_predictions.parquet",
        "v7_fresh_model": args.v7_dir / "fresh_new/fresh_apr_oct.cbm",
        "v6_fresh_model": args.v6_dir / "fresh_new/fresh_apr_oct.cbm",
        "v7_seasonal_model": args.v7_dir / "seasonal_jan_jul.cbm",
        "v7_forward_model": args.v7_dir / "forward_nov_dec.cbm",
        "v6_seasonal_model": args.v6_dir / "seasonal_jan_jul.cbm",
        "v6_forward_model": args.v6_dir / "forward_nov_dec.cbm",
        "baseline_rows": args.cache_dir / "training_rows.parquet",
        "baseline_features": args.cache_dir / "features.parquet",
        "arrival_cache": args.arrival_dir / "training_arrival_features.parquet",
        "neighbour_cache": args.neighbour_dir / "training_neighbour_features.parquet",
        "runway_sequence_cache": args.runway_dir / "training_runway_arrival_features.parquet",
        "runway_taxi_cache": args.taxi_dir / "training_runway_arrival_taxi_features.parquet",
        "runway_taxi_build_protocol": args.taxi_dir / "protocol.json",
        "runway_taxi_build_manifest": args.taxi_dir / "feature_build.json",
        "weather": args.weather_file,
    }
    taxi_build = json.loads((args.taxi_dir / "feature_build.json")
                            .read_text(encoding="utf-8"))
    taxi_protocol = json.loads((args.taxi_dir / "protocol.json")
                               .read_text(encoding="utf-8"))
    taxi_file = paths["runway_taxi_cache"]
    builder_spec = taxi_protocol["spec"]
    if (taxi_build.get("departure_labels_used") is not False
            or taxi_build.get("released_arrival_taxi_used") is not True
            or builder_spec["raw_query"]["phase"] != "DEP"
            or builder_spec["raw_query"]["columns"] != list(DEP_COLUMNS)
            or builder_spec["raw_events"]["phase"] != "ARR"
            or builder_spec["raw_events"]["columns"] != list(ARR_COLUMNS)
            or builder_spec["features"] != list(TAXI_FEATURES)):
        raise ValueError("Runway taxi cache provenance or fixed field policy changed")
    if taxi_build["training"]["output_sha256"] != sha256(taxi_file):
        raise ValueError("Fixed runway taxi feature output differs from builder manifest")
    if taxi_build["protocol_sha256"] != sha256(paths["runway_taxi_build_protocol"]):
        raise ValueError("Runway taxi builder protocol changed")
    value = {
        "spec": protocol_spec(), "references": references,
        "input_sha256": {name: sha256(path) for name, path in paths.items()},
        "raw_training_sha256": {path.name: sha256(path) for path in raw},
    }
    target = args.output_dir / "protocol.json"
    if target.exists():
        if json.loads(target.read_text(encoding="utf-8")) != value:
            raise ValueError("Frozen v10 protocol or source/input hashes changed")
    else:
        write_json(target, value)
    return value


def load_features(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, features = traffic.load_features(args)
    taxi_path = args.taxi_dir / "training_runway_arrival_taxi_features.parquet"
    taxi = pd.read_parquet(taxi_path)
    if (list(taxi) != ["MVT_ID_mvt", *TAXI_FEATURES]
            or len(taxi) != len(rows)
            or not np.array_equal(taxi.MVT_ID_mvt.to_numpy(),
                                  rows.MVT_ID_mvt.to_numpy())
            or any(name in features for name in TAXI_FEATURES)):
        raise ValueError("Ten fixed runway taxi features lack exact ID/schema alignment")
    if np.isinf(taxi[list(TAXI_FEATURES)].to_numpy(dtype=float)).any():
        raise ValueError("Runway taxi feature cache contains infinity")
    features = pd.concat([features.reset_index(drop=True),
                          taxi[list(TAXI_FEATURES)].reset_index(drop=True)], axis=1)
    if (features.columns.duplicated().any() or "MVT_ID_mvt" in features
            or "target" in features or "BLOCK_TIME_UTC_mvt" in features
            or "TAXITIME_SEC_mvt" in features):
        raise ValueError("v10 model predictor schema contains forbidden own fields")
    return rows, features


def load_reference(args: argparse.Namespace) -> pd.DataFrame:
    ref = pd.read_parquet(args.output_dir / "frozen_v7_oof_reference.parquet")
    if (list(ref) != list(REFERENCE_COLUMNS) or len(ref) != EXPECTED_OOF_ROWS
            or ref.MVT_ID_mvt.isna().any() or ref.MVT_ID_mvt.duplicated().any()
            or set(ref.fold.unique()) != set(FOLDS)
            or not np.isfinite(ref[["target", "selected"]]
                               .to_numpy(dtype=float)).all()
            or (ref.selected.to_numpy(dtype=float) < 0).any()):
        raise ValueError("Frozen v7 candidate reference schema or coverage changed")
    return ref


def verify_expert(args: argparse.Namespace, name: str,
                  expected: pd.Series) -> pd.DataFrame:
    path = args.output_dir / f"{name}_oof.parquet"
    expert = pd.read_parquet(path)
    if list(expert) != ["MVT_ID_mvt", "expert"]:
        raise ValueError(f"{name} expert OOF has unexpected schema")
    exact_ids(expert.MVT_ID_mvt, expected, f"{name} v10 expert")
    if not np.isfinite(expert.expert.to_numpy(dtype=float)).all():
        raise ValueError(f"{name} v10 expert has nonfinite predictions")
    return expert


def fit_folds(args: argparse.Namespace) -> None:
    protocol(args)
    require_memory(max(10.0, args.min_free_gib))
    for name in FOLDS:
        if any((args.output_dir / f"{name}{suffix}").exists() for suffix in
               ("_oof.parquet", ".cbm", "_validation.json")):
            raise FileExistsError(f"v10 {name} artifacts already exist; verify before reuse")
    rows, features = load_features(args)
    ref = load_reference(args)
    for name, months in FOLDS.items():
        deep.fit_fold(name, months, rows, features, ref, args)
        held = ref.loc[ref.fold.eq(name)]
        verify_expert(args, name, held.loc[held.a_valid, "MVT_ID_mvt"])
    evaluate(args)


def evaluate(args: argparse.Namespace, include_fresh: bool = True) -> dict:
    fixed = protocol(args)
    ref = load_reference(args)
    report: dict = {"protocol_sha256": sha256(args.output_dir / "protocol.json"),
                    "frozen_reference_sha256": fixed["references"]["frozen_reference_sha256"],
                    "folds": {}, "selected_weight": None,
                    "existing_folds_passed": False, "fresh_audit_passed": False,
                    "ranking_authorized": False}
    pieces = []
    for name in FOLDS:
        held = ref.loc[ref.fold.eq(name)]
        expert = verify_expert(args, name,
                               held.loc[held.a_valid, "MVT_ID_mvt"])
        part = held.merge(expert, on="MVT_ID_mvt", how="left", sort=False,
                          validate="one_to_one")
        present = part.expert.notna().to_numpy()
        if not np.array_equal(present, part.a_valid.to_numpy(dtype=bool)):
            raise ValueError(f"{name} expert is missing valid-AOBT rows")
        y = part.target.to_numpy(dtype=float)
        base = part.selected.to_numpy(dtype=float)
        alternate = part.expert.fillna(part.selected).to_numpy(dtype=float)
        scores = {str(weight): deep.rmse(y, np.maximum(
            base + weight*(alternate-base), 0)) for weight in WEIGHTS}
        if name == "seasonal_jan_jul":
            report["selected_weight"] = min(WEIGHTS,
                key=lambda weight: (scores[str(weight)], weight))
        weight = report["selected_weight"]
        candidate = np.maximum(base + weight*(alternate-base), 0)
        if not np.array_equal(candidate[~present], base[~present]):
            raise ValueError(f"{name} changed invalid-AOBT reference predictions")
        info_path = args.output_dir / f"{name}_validation.json"
        info = json.loads(info_path.read_text(encoding="utf-8"))
        report["folds"][name] = {
            "rows_all_finite": len(part), "rows_valid_aobt": int(present.sum()),
            "scores_all_finite_rmse": scores,
            "day_bootstrap": arrival.bootstrap(part, base, candidate,
                                                seed=BOOTSTRAP_SEED),
            "expert_oof_sha256": sha256(args.output_dir / f"{name}_oof.parquet"),
            "model_sha256": sha256(args.output_dir / f"{name}.cbm"),
            "fit_report_sha256": sha256(info_path),
            "trees": info["trees"],
        }
        part["candidate"] = candidate
        pieces.append(part)
    weight = report["selected_weight"]
    report["existing_folds_passed"] = bool(weight > 0 and all(
        fold["scores_all_finite_rmse"][str(weight)]
        < fold["scores_all_finite_rmse"]["0.0"]
        and fold["day_bootstrap"]["gain_ci95_sec"][0] > 0
        for fold in report["folds"].values()))
    path = args.output_dir / "validation_predictions.parquet"
    pd.concat(pieces, ignore_index=True).to_parquet(path, index=False)
    report["validation_predictions_sha256"] = sha256(path)
    fresh_path = args.output_dir / "fresh_audit.json"
    if include_fresh and fresh_path.exists():
        fresh = json.loads(fresh_path.read_text(encoding="utf-8"))
        if (fresh["weight"] != weight or fresh["protocol_sha256"] !=
                report["protocol_sha256"] or fresh.get("months") !=
                list(FRESH_MONTHS) or not fresh.get("coverage_verified")
                or fresh["v7_paired_sha256"] !=
                   sha256(args.v7_dir / "fresh_audit_predictions.parquet")
                or fresh["v10_expert_oof_sha256"] != sha256(
                    args.output_dir / "fresh_new/fresh_apr_oct_oof.parquet")
                or fresh["v10_model_sha256"] != sha256(
                    args.output_dir / "fresh_new/fresh_apr_oct.cbm")
                or fresh["predictions_sha256"] != sha256(
                    args.output_dir / "fresh_audit_predictions.parquet")):
            raise ValueError("Fresh v10 audit used another source, weight or model")
        audit_gate = all(fresh["scores"][str(month)]["v10_blend_rmse"]
                         < fresh["scores"][str(month)]["v7_rmse"]
                         for month in FRESH_MONTHS) and (
                             fresh["bootstrap"]["gain_ci95_sec"][0] > 0)
        if bool(fresh["passed"]) != bool(audit_gate):
            raise ValueError("Fresh v10 audit pass flag conflicts with fixed gate")
        report["fresh_audit_passed"] = bool(audit_gate)
        report["fresh_audit_sha256"] = sha256(fresh_path)
    write_json(args.output_dir / "validation.json", report)
    print(json.dumps({"selected_weight": weight,
                      "existing_folds_passed": report["existing_folds_passed"],
                      "fresh_audit_passed": report["fresh_audit_passed"],
                      "ranking_authorized": False}, indent=2), flush=True)
    return report


def fresh_audit(args: argparse.Namespace) -> None:
    if (args.output_dir / "fresh_audit.json").exists():
        raise FileExistsError("v10 fresh audit already exists; do not overwrite")
    local = evaluate(args, include_fresh=False)
    if not local["existing_folds_passed"]:
        raise ValueError("Existing-fold gates failed before April/October audit")
    protocol(args)
    require_memory(max(10.0, args.min_free_gib))
    rows, features = load_features(args)
    proxy = rows.proxy.to_numpy(dtype=float)
    y = rows.target.to_numpy(dtype=float)
    held = rows.month.isin(FRESH_MONTHS).to_numpy()
    valid = np.isfinite(proxy) & (proxy >= 0) & (proxy <= 7200) & np.isfinite(y)
    expected = rows.loc[held & valid,
                        ["MVT_ID_mvt", "target", "month", "time"]].copy()
    expected.rename(columns={"time": "MVT_TIME_UTC_mvt"}, inplace=True)
    if expected.MVT_ID_mvt.isna().any() or expected.MVT_ID_mvt.duplicated().any():
        raise ValueError("Fresh valid-AOBT held-out IDs are invalid")
    if (not np.array_equal(pd.to_datetime(expected.MVT_TIME_UTC_mvt, utc=True)
                           .dt.month.to_numpy(), expected.month.to_numpy())
            or set(expected.month.unique()) != set(FRESH_MONTHS)):
        raise ValueError("Fresh held-out movement time and month disagree")
    v7_path = args.v7_dir / "fresh_audit_predictions.parquet"
    v7_report = json.loads((args.v7_dir / "fresh_audit.json")
                           .read_text(encoding="utf-8"))
    if (v7_report.get("months") != list(FRESH_MONTHS)
            or not v7_report.get("passed") or not v7_report.get("coverage_verified")
            or v7_report["paired_predictions_sha256"] != sha256(v7_path)):
        raise ValueError("Saved v7 April/October paired comparator changed")
    prior = pd.read_parquet(v7_path,
                            columns=["MVT_ID_mvt", "target", "MVT_TIME_UTC_mvt",
                                     "candidate"])
    exact_ids(prior.MVT_ID_mvt, expected.MVT_ID_mvt,
              "v7 fresh matched comparator")
    paired = expected.merge(prior, on="MVT_ID_mvt", how="left", sort=False,
                            validate="one_to_one", suffixes=("", "_v7"))
    if (not np.array_equal(paired.target.to_numpy(dtype=float),
                           paired.target_v7.to_numpy(dtype=float))
            or not np.array_equal(
                pd.to_datetime(paired.MVT_TIME_UTC_mvt, utc=True).to_numpy(),
                pd.to_datetime(paired.MVT_TIME_UTC_mvt_v7, utc=True).to_numpy())
            or not np.isfinite(paired.candidate.to_numpy(dtype=float)).all()
            or (paired.candidate.to_numpy(dtype=float) < 0).any()):
        raise ValueError("v7 fresh comparator labels, times or values disagree")
    reference = paired[["MVT_ID_mvt", "target", "month", "MVT_TIME_UTC_mvt",
                        "candidate"]].rename(columns={"candidate": "selected"})
    reference["fold"] = "fresh_apr_oct"
    new_args = argparse.Namespace(**vars(args))
    new_args.output_dir = args.output_dir / "fresh_new"
    if any((new_args.output_dir / f"fresh_apr_oct{suffix}").exists()
           for suffix in ("_oof.parquet", ".cbm", "_validation.json")):
        raise FileExistsError("v10 fresh expert artifacts already exist")
    deep.fit_fold("fresh_apr_oct", FRESH_MONTHS, rows, features, reference,
                  new_args)
    expert = verify_expert(new_args, "fresh_apr_oct", expected.MVT_ID_mvt)
    paired = reference.merge(expert, on="MVT_ID_mvt", how="left", sort=False,
                             validate="one_to_one")
    if (len(paired) != len(expected)
            or not np.array_equal(paired.MVT_ID_mvt.to_numpy(),
                                  expected.MVT_ID_mvt.to_numpy())
            or not np.isfinite(paired[["target", "selected", "expert"]]
                               .to_numpy(dtype=float)).all()):
        raise ValueError("v10 fresh matched expert or comparator coverage failed")
    base = paired.selected.to_numpy(dtype=float)
    candidate = np.maximum(base + local["selected_weight"]*(
        paired.expert.to_numpy(dtype=float)-base), 0)
    paired["v10_blend"] = candidate
    month = paired.month.to_numpy(dtype=int)
    scores = {str(value): {"n": int((month == value).sum()),
                           "v7_rmse": deep.rmse(paired.target.to_numpy(dtype=float)[month == value],
                                                base[month == value]),
                           "v10_blend_rmse": deep.rmse(
                               paired.target.to_numpy(dtype=float)[month == value],
                               candidate[month == value])}
              for value in FRESH_MONTHS}
    bootstrap = arrival.bootstrap(paired, base, candidate,
                                  seed=BOOTSTRAP_SEED)
    passed = (all(scores[str(value)]["v10_blend_rmse"]
                  < scores[str(value)]["v7_rmse"] for value in FRESH_MONTHS)
              and bootstrap["gain_ci95_sec"][0] > 0)
    output_path = args.output_dir / "fresh_audit_predictions.parquet"
    paired.to_parquet(output_path, index=False)
    audit = {
        "months": list(FRESH_MONTHS), "weight": local["selected_weight"],
        "scores": scores, "bootstrap": bootstrap, "passed": bool(passed),
        "coverage_verified": True, "rows_valid_aobt_finite": len(paired),
        "protocol_sha256": sha256(args.output_dir / "protocol.json"),
        "v7_fresh_report_sha256": sha256(args.v7_dir / "fresh_audit.json"),
        "v7_paired_sha256": sha256(v7_path),
        "v7_fresh_model_sha256": sha256(args.v7_dir / "fresh_new/fresh_apr_oct.cbm"),
        "v6_fresh_model_sha256": sha256(args.v6_dir / "fresh_new/fresh_apr_oct.cbm"),
        "v10_expert_oof_sha256": sha256(new_args.output_dir /
                                        "fresh_apr_oct_oof.parquet"),
        "v10_model_sha256": sha256(new_args.output_dir / "fresh_apr_oct.cbm"),
        "predictions_sha256": sha256(output_path),
        "scope": "Matched architecture comparison; not unbiased complete ensemble OOF",
    }
    write_json(args.output_dir / "fresh_audit.json", audit)
    evaluate(args)


def fit_final(_: argparse.Namespace) -> None:
    raise RuntimeError("V10 final training requires a separately frozen later guard")


def final_predict(_: argparse.Namespace) -> None:
    raise RuntimeError("V10 ranking output requires a separately frozen later guard")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("show-spec", "prepare", "fit-folds",
                                          "evaluate", "fresh-audit", "fit-final",
                                          "final-predict"), default="show-spec")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--cache-dir", type=Path,
                        default=Path("artifacts/baseline"))
    parser.add_argument("--weather-file", type=Path,
                        default=Path("data/external/weather.parquet"))
    parser.add_argument("--arrival-dir", type=Path,
                        default=Path("artifacts/v5-arrival-clean"))
    parser.add_argument("--neighbour-dir", type=Path,
                        default=Path("artifacts/v6-neighbour"))
    parser.add_argument("--runway-dir", type=Path,
                        default=Path("artifacts/v6-runway-arrival"))
    parser.add_argument("--taxi-dir", type=Path,
                        default=Path("artifacts/v10-runway-arrival-taxi"))
    parser.add_argument("--v7-dir", type=Path,
                        default=Path("artifacts/v7-runway-traffic"))
    parser.add_argument("--v6-dir", type=Path,
                        default=Path("artifacts/v6-deep-arrival"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("artifacts/v10-runway-taxi"))
    parser.add_argument("--min-free-gib", type=float, default=10.0)
    args = parser.parse_args()
    args.iterations, args.depth, args.threads = 10000, 10, 2
    if args.mode == "show-spec":
        print(json.dumps(protocol_spec(), indent=2))
    elif args.mode == "prepare":
        value = protocol(args)
        print(json.dumps({"protocol": str(args.output_dir / "protocol.json"),
                          "new_features": len(value["spec"]["architecture"]
                                             ["additional_features"])}))
    elif args.mode == "fit-folds":
        fit_folds(args)
    elif args.mode == "evaluate":
        evaluate(args)
    elif args.mode == "fresh-audit":
        fresh_audit(args)
    elif args.mode == "fit-final":
        fit_final(args)
    else:
        final_predict(args)


if __name__ == "__main__":
    main()
