"""Fixed high-capacity CPU LightGBM residual experiment for PRC 2026.

Predictors are the released timestamp, flight, weather, ARR traffic, neighbour
proxy and runway sequence covariates. Departure BLOCK/TAXITIME are never model
inputs. January/July choose a coarse blend against frozen v6; November/December
apply it unchanged. A separately refitted April/October paired audit compares
this architecture to the saved depth-10 ARR expert before promotion.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

import deep_arrival_expert as arrival
import deep_timestamp_expert as deep
import traffic_deep_expert as traffic
from solution import _training_files


FOLDS = {"seasonal_jan_jul": (1, 7), "forward_nov_dec": (11, 12)}
WEIGHTS = (0.0, 0.1, 0.25, 0.5, 1.0)
EXPECTED_OOF_ROWS = 672428
EXPECTED_RANKING_ROWS = 344841
SEED = 2026
MAX_ROUNDS = 2000
EARLY_STOP_ROUNDS = 150
REFERENCE_COLUMNS = ("MVT_ID_mvt", "target", "fold", "airport", "month",
                     "MVT_TIME_UTC_mvt", "a_valid")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def assert_exact_ids(actual: pd.Series, expected: pd.Series, label: str) -> None:
    a, e = pd.Index(actual), pd.Index(expected)
    if (a.has_duplicates or e.has_duplicates or a.isna().any() or e.isna().any()
            or len(a) != len(e) or not a.isin(e).all() or not e.isin(a).all()):
        raise ValueError(f"{label}: exact unique ID coverage failed")


def freeze_reference(args: argparse.Namespace) -> dict:
    source = args.source_reference
    path = args.output_dir / "frozen_v6_oof_reference.parquet"
    if not path.exists():
        frame = pd.read_parquet(source)
        required = [*REFERENCE_COLUMNS, "candidate"]
        if any(col not in frame for col in required) or len(frame) != EXPECTED_OOF_ROWS:
            raise ValueError("v6 source reference schema or row count changed")
        frozen = frame[required].rename(columns={"candidate": "selected"})
        path.parent.mkdir(parents=True, exist_ok=True)
        frozen.to_parquet(path, index=False)
    frozen = pd.read_parquet(path)
    if (list(frozen) != [*REFERENCE_COLUMNS, "selected"]
            or len(frozen) != EXPECTED_OOF_ROWS
            or set(frozen.fold.unique()) != set(FOLDS)
            or frozen.MVT_ID_mvt.isna().any()
            or frozen.MVT_ID_mvt.duplicated().any()
            or not np.isfinite(frozen[["target", "selected"]]
                               .to_numpy(dtype=float)).all()):
        raise ValueError("Frozen v6 reference invalid")
    current = pd.read_parquet(source,
                              columns=[*REFERENCE_COLUMNS, "candidate"])
    if not frozen.equals(current.rename(columns={"candidate": "selected"})):
        raise ValueError("Frozen v6 reference changed relative to source")
    baseline = pd.read_parquet(args.cache_dir / "training_rows.parquet",
                               columns=["MVT_ID_mvt", "target", "proxy", "month"])
    expected = baseline.loc[baseline.month.isin((1, 7, 11, 12))
                            & np.isfinite(baseline.target.to_numpy(dtype=float))]
    assert_exact_ids(frozen.MVT_ID_mvt, expected.MVT_ID_mvt,
                     "All finite January/July/November/December departures")
    aligned = frozen.merge(expected, on="MVT_ID_mvt", how="left",
                           sort=False, validate="one_to_one",
                           suffixes=("", "_baseline"))
    proxy = aligned.proxy.to_numpy(dtype=float)
    valid = np.isfinite(proxy) & (proxy >= 0) & (proxy <= 7200)
    if (len(aligned) != len(frozen)
            or not np.allclose(aligned.target, aligned.target_baseline,
                               rtol=0, atol=1e-6)
            or not np.array_equal(aligned.a_valid.to_numpy(dtype=bool), valid)
            or not np.array_equal(aligned.month.to_numpy(),
                                   aligned.month_baseline.to_numpy())):
        raise ValueError("Frozen OOF labels, months or AOBT gate differ from baseline")
    for name, months in FOLDS.items():
        if not aligned.loc[aligned.fold.eq(name), "month"].isin(months).all():
            raise ValueError(f"{name} contains an unexpected month")
    return {"source_sha256": sha256(source), "frozen_sha256": sha256(path)}


def fixed_params() -> dict:
    return dict(objective="regression", metric="rmse", learning_rate=.03,
                num_leaves=255, min_data_in_leaf=80, lambda_l2=30,
                feature_fraction=.9, bagging_fraction=.8, bagging_freq=1,
                num_threads=3, seed=SEED, feature_fraction_seed=SEED,
                bagging_seed=SEED, deterministic=True, force_col_wise=True,
                verbosity=-1)


def protocol(args: argparse.Namespace) -> dict:
    reference = freeze_reference(args)
    sources = {
        "baseline_training_rows": args.cache_dir / "training_rows.parquet",
        "baseline_features": args.cache_dir / "features.parquet",
        "baseline_ranking_rows": args.cache_dir / "ranking_rows.parquet",
        "baseline_ranking_features": args.cache_dir / "ranking_features.parquet",
        "weather": args.weather_file,
        "arrival_training": args.arrival_dir / "training_arrival_features.parquet",
        "arrival_ranking": args.arrival_dir / "ranking_arrival_features.parquet",
        "neighbour_training": args.neighbour_dir / "training_neighbour_features.parquet",
        "neighbour_ranking": args.neighbour_dir / "ranking_neighbour_features.parquet",
        "runway_training": args.runway_dir / "training_runway_arrival_features.parquet",
        "runway_ranking": args.runway_dir / "ranking_runway_arrival_features.parquet",
        "neighbour_protocol": args.neighbour_dir / "protocol.json",
        "runway_protocol": args.runway_dir / "protocol.json",
        "prior_apr_oct_oof": args.prior_fresh_oof,
        "ranking_reference": args.ranking_reference,
        "submission_template": args.data_dir / "submitting.parquet",
    }
    sources.update({f"raw_training_{path.name}": path
                    for path in _training_files(args.data_dir)})
    sources["raw_ranking"] = args.data_dir / "ranking.parquet"
    value = {
        "purpose": "Independent local 2025 experiment; no leaderboard tuning",
        "frozen_v6_reference": reference,
        "source_sha256": {name: sha256(path) for name, path in sources.items()},
        "predictors": "deep timestamp + 14 ARR + 24 neighbour + 8 runway; exact training/ranking cache ID alignment",
        "forbidden_predictors": ["departure BLOCK_TIME_UTC_mvt",
                                 "departure TAXITIME_SEC_mvt", "opaque flight ID values"],
        "params": fixed_params(),
        "max_rounds": MAX_ROUNDS,
        "early_stop_rounds": EARLY_STOP_ROUNDS,
        "early_stop_days": "eligible complement-month UTC calendar days with (Unix day + 2026) modulo 10 == 0",
        "training_gate": "finite target in [0,86400] and own AOBT proxy in [0,7200]",
        "heldout_gate": "all finite targets with own AOBT proxy in [0,7200]",
        "selection": {"months": [1, 7], "weights": list(WEIGHTS),
                      "metric": "all finite clipped RMSE; ties smaller weight",
                      "forward_months": [11, 12], "forward_weight": "unchanged",
                      "promotion_gate": "nonzero selected weight and positive UTC-day bootstrap 95% gain lower bound in both folds"},
        "fresh_audit": {"months": [4, 10],
                        "train_and_early_stop_exclude": [4, 10],
                        "prior": "saved depth-10 ARR fresh OOF",
                        "candidate": "new fixed LightGBM on all covariates",
                        "blend_weight": "seasonal selected weight unchanged",
                        "gate": "exact valid-ID coverage, finite predictions, paired day bootstrap lower gain > 0"},
        "limitations": "Repeated 2025 comparisons and inherited v6 base OOF overlap limit interpretation; 2026 ranking feedback is not used.",
    }
    path = args.output_dir / "protocol.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise ValueError("Frozen v8 protocol or source hash changed")
    else:
        write_json(path, value)
    return value


def load_features(args: argparse.Namespace, ranking: bool = False):
    rows, features = traffic.load_features(args, ranking=ranking)
    if features.columns.duplicated().any():
        raise ValueError("Duplicate predictor names")
    return rows, features


def masks(rows: pd.DataFrame, held_months: tuple[int, int]) -> tuple[np.ndarray,
                                                                       np.ndarray,
                                                                       np.ndarray]:
    proxy = rows.proxy.to_numpy(dtype=float)
    target = rows.target.to_numpy(dtype=float)
    held = rows.month.isin(held_months).to_numpy()
    proxy_valid = np.isfinite(proxy) & (proxy >= 0) & (proxy <= 7200)
    train = ~held & proxy_valid & np.isfinite(target) & (target >= 0) & (target <= 86400)
    test = held & proxy_valid & np.isfinite(target)
    utc = pd.to_datetime(rows.time, utc=True)
    day = utc.dt.floor("D").dt.as_unit("ns").astype("int64").to_numpy() // (
        86400 * 1_000_000_000)
    early = train & ((day + SEED) % 10 == 0)
    fit = train & ~early
    if (not fit.any() or not early.any() or np.any(held & (fit | early))):
        raise ValueError("Calendar-day training/early-stop split invalid")
    return np.flatnonzero(fit), np.flatnonzero(early), np.flatnonzero(test)


def progress_callback(path: Path, start: float):
    records: list[dict] = []

    def record(env) -> None:
        tree = env.iteration + 1
        if tree % 100:
            return
        metric = {f"{dataset}.{name}": float(value)
                  for dataset, name, value, _ in (env.evaluation_result_list or [])}
        records.append({"tree": tree,
                        "elapsed_seconds": round(time.monotonic() - start, 2),
                        "metrics": metric})
        write_json(path, {"status": "running", "records": records})

    record.order = 20
    return record, records


def fit_fold(name: str, held_months: tuple[int, int], rows: pd.DataFrame,
             features: pd.DataFrame, args: argparse.Namespace) -> dict:
    fit_idx, early_idx, test_idx = masks(rows, held_months)
    target = rows.target.to_numpy(dtype=float)
    proxy = rows.proxy.to_numpy(dtype=float)
    residual = target - proxy
    categorical = features.select_dtypes(include="category").columns.tolist()
    train = lgb.Dataset(features.iloc[fit_idx], label=residual[fit_idx],
                        categorical_feature=categorical, free_raw_data=True)
    early = lgb.Dataset(features.iloc[early_idx], label=residual[early_idx],
                        categorical_feature=categorical, reference=train,
                        free_raw_data=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    callback, progress = progress_callback(args.output_dir / f"{name}_progress.json",
                                           start)
    model = lgb.train(fixed_params(), train, num_boost_round=MAX_ROUNDS,
                      valid_sets=[early], valid_names=["calendar_day_early"],
                      callbacks=[lgb.early_stopping(EARLY_STOP_ROUNDS,
                                                    verbose=False), callback])
    elapsed = time.monotonic() - start
    prediction = proxy[test_idx] + model.predict(
        features.iloc[test_idx], num_threads=3)
    result = pd.DataFrame({"MVT_ID_mvt": rows.MVT_ID_mvt.iloc[test_idx]
                           .to_numpy(), "expert": prediction})
    if (result.MVT_ID_mvt.duplicated().any() or
            not np.isfinite(result.expert.to_numpy(dtype=float)).all()):
        raise ValueError(f"{name} OOF expert has duplicate IDs or nonfinite values")
    path = args.output_dir / f"{name}_oof.parquet"
    result.to_parquet(path, index=False)
    model.save_model(str(args.output_dir / f"{name}.txt"))
    report = {"fold": name, "held_months": list(held_months),
              "n_fit": len(fit_idx), "n_early": len(early_idx),
              "n_test": len(test_idx), "best_iteration": model.best_iteration,
              "fit_seconds": elapsed, "oof_sha256": sha256(path),
              "feature_names": list(features)}
    write_json(args.output_dir / f"{name}_progress.json",
               {"status": "complete", "records": progress,
                "best_iteration": model.best_iteration,
                "fit_seconds": elapsed})
    write_json(args.output_dir / f"{name}_fit.json", report)
    print(json.dumps({k: report[k] for k in ("fold", "n_fit", "n_early",
                                               "n_test", "best_iteration",
                                               "fit_seconds")}, indent=2), flush=True)
    return report


def load_reference(args: argparse.Namespace) -> pd.DataFrame:
    ref = pd.read_parquet(args.output_dir / "frozen_v6_oof_reference.parquet")
    if (len(ref) != EXPECTED_OOF_ROWS or ref.MVT_ID_mvt.duplicated().any()
            or ref.MVT_ID_mvt.isna().any()
            or set(ref.fold.unique()) != set(FOLDS)
            or not np.isfinite(ref[["target", "selected"]]
                               .to_numpy(dtype=float)).all()):
        raise ValueError("Frozen v6 reference invalid")
    return ref


def evaluate(args: argparse.Namespace, include_fresh: bool = True) -> dict:
    fixed = protocol(args)
    ref = load_reference(args)
    report = {"protocol": fixed, "folds": {}, "selected_weight": None,
              "existing_folds_passed": False, "promoted": False}
    predictions = []
    for name in FOLDS:
        held = ref.loc[ref.fold.eq(name)].copy()
        expert = pd.read_parquet(args.output_dir / f"{name}_oof.parquet")
        assert_exact_ids(expert.MVT_ID_mvt,
                         held.loc[held.a_valid, "MVT_ID_mvt"],
                         f"{name} valid-AOBT OOF")
        if not np.isfinite(expert.expert.to_numpy(dtype=float)).all():
            raise ValueError(f"{name} expert is nonfinite")
        merged = held.merge(expert, on="MVT_ID_mvt", how="left",
                            sort=False, validate="one_to_one")
        if (len(merged) != len(held) or
                not np.array_equal(merged.MVT_ID_mvt.to_numpy(),
                                   held.MVT_ID_mvt.to_numpy()) or
                not np.array_equal(merged.expert.notna().to_numpy(),
                                   merged.a_valid.to_numpy(dtype=bool))):
            raise ValueError(f"{name} OOF merge or proxy gate misaligned")
        base = merged.selected.to_numpy(dtype=float)
        alternative = merged.expert.fillna(merged.selected).to_numpy(dtype=float)
        target = merged.target.to_numpy(dtype=float)
        scores = {str(w): deep.rmse(target,
                                   np.maximum(base + w * (alternative-base), 0))
                  for w in WEIGHTS}
        if name == "seasonal_jan_jul":
            report["selected_weight"] = min(WEIGHTS,
                key=lambda w: (scores[str(w)], w))
        weight = report["selected_weight"]
        candidate = np.maximum(base + weight * (alternative-base), 0)
        report["folds"][name] = {
            "n_all_finite": len(merged),
            "n_valid_aobt": int(merged.a_valid.sum()),
            "scores": scores,
            "bootstrap": arrival.bootstrap(merged, base, candidate),
            "oof_sha256": sha256(args.output_dir / f"{name}_oof.parquet"),
            "fit": json.loads((args.output_dir / f"{name}_fit.json")
                              .read_text(encoding="utf-8")),
        }
        merged["candidate"] = candidate
        predictions.append(merged)
    report["existing_folds_passed"] = (
        report["selected_weight"] > 0 and all(
            fold["bootstrap"]["gain_ci95_sec"][0] > 0
            for fold in report["folds"].values()))
    fresh_path = args.output_dir / "fresh_audit.json"
    if include_fresh and fresh_path.exists():
        fresh = json.loads(fresh_path.read_text(encoding="utf-8"))
        if (not fresh.get("coverage_verified") or
                fresh["weight"] != report["selected_weight"]):
            raise ValueError("Fresh audit coverage or weight differs")
        for key, path in (
            ("prior_oof_sha256", args.prior_fresh_oof),
            ("new_oof_sha256", args.output_dir / "fresh_new/fresh_apr_oct_oof.parquet"),
            ("paired_sha256", args.output_dir / "fresh_audit_predictions.parquet"),
        ):
            if fresh[key] != sha256(path):
                raise ValueError(f"Fresh audit input changed: {key}")
        report["fresh_audit"] = fresh
        report["promoted"] = report["existing_folds_passed"] and fresh["passed"]
    all_rows = pd.concat(predictions, ignore_index=True)
    path = args.output_dir / "validation_predictions.parquet"
    all_rows.to_parquet(path, index=False)
    report["validation_predictions_sha256"] = sha256(path)
    write_json(args.output_dir / "validation.json", report)
    print(json.dumps({"selected_weight": report["selected_weight"],
                      "existing_folds_passed": report["existing_folds_passed"],
                      "promoted": report["promoted"],
                      "folds": {k: {"scores": v["scores"],
                                    "bootstrap": v["bootstrap"]}
                                for k, v in report["folds"].items()}},
                     indent=2), flush=True)
    return report


def fit(args: argparse.Namespace) -> None:
    protocol(args)
    rows, features = load_features(args)
    for name, months in FOLDS.items():
        if not (args.output_dir / f"{name}_oof.parquet").exists():
            fit_fold(name, months, rows, features, args)
    del rows, features
    gc.collect()
    evaluate(args)


def fresh_audit(args: argparse.Namespace) -> None:
    report = evaluate(args, include_fresh=False)
    if not report["existing_folds_passed"]:
        raise ValueError("Existing-fold gate failed; fresh audit not launched")
    rows, features = load_features(args)
    proxy = rows.proxy.to_numpy(dtype=float)
    target = rows.target.to_numpy(dtype=float)
    held = rows.month.isin((4, 10)).to_numpy()
    valid = (np.isfinite(proxy) & (proxy >= 0) & (proxy <= 7200)
             & np.isfinite(target))
    reference = rows.loc[held & valid, ["MVT_ID_mvt", "target", "time"]].copy()
    reference.rename(columns={"time": "MVT_TIME_UTC_mvt"}, inplace=True)
    reference["selected"] = proxy[held & valid]
    reference["fold"] = "fresh_apr_oct"
    if (reference.MVT_ID_mvt.duplicated().any() or
            reference.MVT_ID_mvt.isna().any()):
        raise ValueError("Fresh held-out reference IDs invalid")
    new_args = argparse.Namespace(**vars(args))
    new_args.output_dir = args.output_dir / "fresh_new"
    new_path = new_args.output_dir / "fresh_apr_oct_oof.parquet"
    if not new_path.exists():
        fit_fold("fresh_apr_oct", (4, 10), rows, features, new_args)
    prior = pd.read_parquet(args.prior_fresh_oof).rename(
        columns={"expert": "prior"})
    new = pd.read_parquet(new_path).rename(columns={"expert": "new"})
    for label, frame in (("prior", prior), ("new", new)):
        assert_exact_ids(frame.MVT_ID_mvt, reference.MVT_ID_mvt,
                         f"fresh {label} expert")
        if not np.isfinite(frame[label].to_numpy(dtype=float)).all():
            raise ValueError(f"fresh {label} expert nonfinite")
    paired = reference.merge(prior, on="MVT_ID_mvt", how="left",
                             sort=False, validate="one_to_one").merge(
        new, on="MVT_ID_mvt", how="left", sort=False, validate="one_to_one")
    if (len(paired) != len(reference)
            or not np.array_equal(paired.MVT_ID_mvt.to_numpy(),
                                  reference.MVT_ID_mvt.to_numpy())
            or not np.isfinite(paired[["target", "prior", "new"]]
                               .to_numpy(dtype=float)).all()):
        raise ValueError("Fresh paired predictions misaligned or nonfinite")
    base = np.maximum(paired.prior.to_numpy(dtype=float), 0)
    candidate = np.maximum(base + report["selected_weight"] *
                           (paired.new.to_numpy(dtype=float) - base), 0)
    paired["candidate"] = candidate
    path = args.output_dir / "fresh_audit_predictions.parquet"
    paired.to_parquet(path, index=False)
    bootstrap = arrival.bootstrap(paired, base, candidate, seed=20261008)
    audit = {"months": [4, 10], "weight": report["selected_weight"],
             "n_valid_aobt": len(paired), "coverage_verified": True,
             "prior_rmse": deep.rmse(paired.target.to_numpy(dtype=float), base),
             "candidate_rmse": deep.rmse(paired.target.to_numpy(dtype=float),
                                          candidate),
             "bootstrap": bootstrap,
             "passed": bootstrap["gain_ci95_sec"][0] > 0,
             "prior_oof_sha256": sha256(args.prior_fresh_oof),
             "new_oof_sha256": sha256(new_path), "paired_sha256": sha256(path),
             "scope": "paired architecture comparison, not full v6 ensemble OOF"}
    write_json(args.output_dir / "fresh_audit.json", audit)
    del rows, features
    gc.collect()
    evaluate(args)


def feature_schema(features: pd.DataFrame) -> dict:
    kinds = []
    for dtype in features.dtypes:
        if isinstance(dtype, pd.CategoricalDtype):
            kinds.append("category")
        elif pd.api.types.is_numeric_dtype(dtype):
            kinds.append("numeric")
        else:
            raise ValueError(f"Unsupported LightGBM predictor dtype: {dtype}")
    return {"columns": list(features),
            "dtypes": [str(dtype) for dtype in features.dtypes],
            "kinds": kinds,
            "categorical_columns": features.select_dtypes(
                include="category").columns.tolist()}


def full_fit_context(args: argparse.Namespace, report: dict,
                     rounds: int) -> dict:
    return {
        "protocol_sha256": sha256(args.output_dir / "protocol.json"),
        "validation_sha256": sha256(args.output_dir / "validation.json"),
        "fresh_audit_sha256": sha256(args.output_dir / "fresh_audit.json"),
        "ranking_reference_sha256": sha256(args.ranking_reference),
        "rounds": rounds, "weight": report["selected_weight"],
        "params": fixed_params(),
    }


def load_or_fit_full(args: argparse.Namespace, report: dict,
                     rounds: int) -> tuple[lgb.Booster, dict]:
    """Reuse a hash-bound saved fit; require a manifest before trusting it."""
    model_path = args.output_dir / "full_2025.txt"
    manifest_path = args.output_dir / "full_fit_manifest.json"
    context = full_fit_context(args, report, rounds)
    if model_path.exists():
        if not manifest_path.exists():
            raise ValueError("Saved full model lacks its fit manifest")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if any(manifest.get(key) != value for key, value in context.items()):
            raise ValueError("Saved full model was fitted under different inputs or policy")
        schema = manifest.get("train_feature_schema")
        if not isinstance(schema, dict) or not schema.get("columns"):
            raise ValueError("Saved full model has no training feature schema")
        model = lgb.Booster(model_file=str(model_path))
        if (model.num_trees() != rounds
                or model.feature_name() != schema["columns"]
                or manifest.get("n_train", 0) <= 0):
            raise ValueError("Saved full model trees or feature names differ")
        model_hash = sha256(model_path)
        if manifest.get("model_sha256") not in (None, model_hash):
            raise ValueError("Saved full model checksum differs from fit manifest")
        # If the process ended after saving the model and before marking the
        # manifest complete, the prefit manifest and Booster checks recover it.
        if manifest.get("status") == "fitting" and manifest.get("model_sha256") is None:
            manifest["model_sha256"] = model_hash
            manifest["status"] = "complete"
            write_json(manifest_path, manifest)
        elif manifest.get("status") != "complete":
            raise ValueError("Saved full model manifest has an invalid state")
        return model, manifest
    if manifest_path.exists():
        raise ValueError("Full fit manifest exists but its model file is absent")

    rows, features = load_features(args)
    proxy = rows.proxy.to_numpy(dtype=float)
    target = rows.target.to_numpy(dtype=float)
    train = (np.isfinite(proxy) & (proxy >= 0) & (proxy <= 7200)
             & np.isfinite(target) & (target >= 0) & (target <= 86400))
    schema = feature_schema(features)
    manifest = {**context, "status": "fitting", "n_train": int(train.sum()),
                "train_feature_schema": schema, "model_sha256": None}
    write_json(manifest_path, manifest)
    dataset = lgb.Dataset(features.loc[train], label=(target-proxy)[train],
                          categorical_feature=schema["categorical_columns"],
                          free_raw_data=True)
    start = time.monotonic()
    callback, progress = progress_callback(
        args.output_dir / "full_2025_progress.json", start)
    model = lgb.train(fixed_params(), dataset, num_boost_round=rounds,
                      callbacks=[callback])
    elapsed = time.monotonic() - start
    model.save_model(str(model_path))
    if model.num_trees() != rounds or model.feature_name() != schema["columns"]:
        raise ValueError("Full fit trees or feature schema differ from prefit manifest")
    manifest.update(status="complete", model_sha256=sha256(model_path),
                    fit_seconds=elapsed)
    write_json(manifest_path, manifest)
    write_json(args.output_dir / "full_2025_progress.json",
               {"status": "complete", "records": progress,
                "trees": model.num_trees(), "fit_seconds": elapsed})
    del rows, features, dataset
    gc.collect()
    return model, manifest


def final_predict(args: argparse.Namespace) -> None:
    report = evaluate(args)
    if not report["promoted"]:
        raise ValueError("Promotion requires both existing folds and fresh audit")
    rounds = int(np.median([report["folds"][name]["fit"]["best_iteration"]
                            for name in FOLDS]))
    model, manifest = load_or_fit_full(args, report, rounds)
    rank_rows, rank_features = load_features(args, ranking=True)
    ranking_schema = feature_schema(rank_features)
    if (ranking_schema["columns"] != manifest["train_feature_schema"]["columns"]
            or ranking_schema["kinds"] != manifest["train_feature_schema"]["kinds"]
            or ranking_schema["categorical_columns"] !=
               manifest["train_feature_schema"]["categorical_columns"]):
        raise ValueError("Ranking predictors differ from fitted feature schema")
    rank_proxy = rank_rows.proxy.to_numpy(dtype=float)
    valid = np.isfinite(rank_proxy) & (rank_proxy >= 0) & (rank_proxy <= 7200)
    expert = np.full(len(rank_rows), np.nan, dtype=float)
    expert[valid] = rank_proxy[valid] + model.predict(
        rank_features.loc[valid], num_threads=3)
    if not np.isfinite(expert[valid]).all():
        raise ValueError("Valid ranking expert predictions nonfinite")
    raw = pd.DataFrame({"MVT_ID_mvt": rank_rows.MVT_ID_mvt,
                        "expert": expert})
    raw.to_parquet(args.output_dir / "ranking_expert.parquet", index=False)
    base = pd.read_parquet(args.ranking_reference)
    template = pd.read_parquet(args.data_dir / "submitting.parquet",
                               columns=["MVT_ID_mvt"])
    if (list(base) != ["MVT_ID_mvt", "TAXITIME_SEC_mvt"]
            or len(base) != EXPECTED_RANKING_ROWS
            or not base.MVT_ID_mvt.equals(rank_rows.MVT_ID_mvt)
            or not base.MVT_ID_mvt.equals(template.MVT_ID_mvt)
            or base.MVT_ID_mvt.duplicated().any()
            or not np.isfinite(base.TAXITIME_SEC_mvt.to_numpy(dtype=float)).all()):
        raise ValueError("v6 ranking reference, cache and template misaligned")
    values = base.TAXITIME_SEC_mvt.to_numpy(dtype=float, copy=True)
    values[valid] = np.maximum(values[valid] + report["selected_weight"] *
                               (expert[valid] - values[valid]), 0)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Final ranking predictions nonfinite or negative")
    base["TAXITIME_SEC_mvt"] = values
    path = args.output_dir / "predictions.parquet"
    base.to_parquet(path, index=False)
    write_json(args.output_dir / "final_report.json", {
        "rounds": rounds, "n_train": manifest["n_train"],
        "n_ranking_valid_aobt": int(valid.sum()),
        "weight": report["selected_weight"],
        "model_sha256": manifest["model_sha256"],
        "fit_manifest_sha256": sha256(args.output_dir / "full_fit_manifest.json"),
        "validation_sha256": manifest["validation_sha256"],
        "fresh_audit_sha256": manifest["fresh_audit_sha256"],
        "ranking_reference_sha256": manifest["ranking_reference_sha256"],
        "train_feature_schema": manifest["train_feature_schema"],
        "ranking_feature_schema": ranking_schema,
        "output_sha256": sha256(path),
        "ranking_expert_sha256": sha256(args.output_dir / "ranking_expert.parquet"),
    })


def prepare(args: argparse.Namespace) -> None:
    value = protocol(args)
    print(json.dumps({"protocol_path": str(args.output_dir / "protocol.json"),
                      "reference": value["frozen_v6_reference"],
                      "params": value["params"]}, indent=2))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("prepare", "fit", "evaluate",
                                      "fresh-audit", "final-predict"),
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
                   default=Path("artifacts/v8-lightgbm"))
    args = p.parse_args()
    {"prepare": prepare, "fit": fit, "evaluate": evaluate,
     "fresh-audit": fresh_audit, "final-predict": final_predict}[args.mode](args)


if __name__ == "__main__":
    main()
