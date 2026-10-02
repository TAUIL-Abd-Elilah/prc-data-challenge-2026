"""Prepared February/August refit of the accepted v7 valid-AOBT expert.

This script is an isolated comparator for the prospectively reserved 2025
guard. It does not choose features, weights, or routing from the held-out
months, and it never reads ranking labels or official leaderboard results.
Run ``--mode prepare`` to freeze input hashes; run ``--mode fit`` only after
the guard's proposed replacement policy has been frozen.
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
from catboost import CatBoostRegressor, Pool

import deep_timestamp_expert as deep
import traffic_deep_expert as traffic
from solution import _training_files


HELDOUT_MONTHS = (2, 8)
FEATURE_REPORT = Path("artifacts/v7-runway-traffic/fresh_new/fresh_apr_oct_validation.json")
V7_FOLD_REPORTS = (
    Path("artifacts/v7-runway-traffic/seasonal_jan_jul_validation.json"),
    Path("artifacts/v7-runway-traffic/forward_nov_dec_validation.json"),
)
V7_PROTOCOL = Path("artifacts/v7-runway-traffic/protocol.json")
GUARD_PROTOCOL = Path("reports/reserved_guard_protocol.json")
SELECTED_POLICY = Path("artifacts/reserved-valid-guard/selected_policy.json")
OUTPUT_COLUMNS = ("MVT_ID_mvt", "target", "MVT_TIME_UTC_mvt", "expert")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def schema_hash(names: list[str]) -> str:
    value = json.dumps(names, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def expected_features() -> list[str]:
    report = json.loads(FEATURE_REPORT.read_text(encoding="utf-8"))
    names = report["features"]
    if (not isinstance(names, list) or len(names) != len(set(names))
            or len(names) < 100 or not all(isinstance(name, str) for name in names)):
        raise ValueError("Frozen v7 feature report has an invalid schema")
    for path in V7_FOLD_REPORTS:
        if json.loads(path.read_text(encoding="utf-8"))["features"] != names:
            raise ValueError("V7 seasonal, forward and fresh feature schemas disagree")
    return names


def frozen_protocol(args: argparse.Namespace) -> dict:
    v7 = json.loads(V7_PROTOCOL.read_text(encoding="utf-8"))
    architecture = v7["architecture"]
    guard = json.loads(GUARD_PROTOCOL.read_text(encoding="utf-8"))
    selection = json.loads(SELECTED_POLICY.read_text(encoding="utf-8"))
    if (architecture["depth"] != 10 or architecture["max_iterations"] != 10000
            or architecture["random_seed"] != 2026
            or guard["guard_months"] != list(HELDOUT_MONTHS)):
        raise ValueError("Frozen v7 architecture or reserved guard changed")
    if (selection.get("selected_route") not in ("v8_combo", "v9b")
            or selection.get("guard_months_not_scored_for_selection")
               != list(HELDOUT_MONTHS)):
        raise ValueError("A non-v7 valid route must be frozen before comparator preparation")
    names = expected_features()
    raw = _training_files(args.data_dir)
    if len(raw) != 12 or len({p.name for p in raw}) != 12:
        raise ValueError("Expected twelve canonical 2025 training files")
    source_files = {
        "own_script": Path(__file__).resolve(),
        "traffic_loader": Path(traffic.__file__).resolve(),
        "deep_trainer": Path(deep.__file__).resolve(),
        "arrival_loader": Path("deep_arrival_expert.py"),
        "cache_loader": Path("catboost_expert.py"),
        "base_features": Path("solution.py"),
        "weather_loader": Path("weather_model.py"),
        "flight_loader": Path("airport_models.py"),
        "training_rows": args.cache_dir / "training_rows.parquet",
        "training_features": args.cache_dir / "features.parquet",
        "arrival_features": args.arrival_dir / "training_arrival_features.parquet",
        "neighbour_features": args.neighbour_dir / "training_neighbour_features.parquet",
        "runway_features": args.runway_dir / "training_runway_arrival_features.parquet",
        "weather": args.weather_file,
        "v7_protocol": V7_PROTOCOL,
        "v7_feature_report": FEATURE_REPORT,
        "v7_seasonal_report": V7_FOLD_REPORTS[0],
        "v7_forward_report": V7_FOLD_REPORTS[1],
        "reserved_guard_protocol": GUARD_PROTOCOL,
        "selected_valid_policy": SELECTED_POLICY,
    }
    value = {
        "purpose": "Fixed v7 traffic-depth10 comparator for February/August guard",
        "heldout_months": list(HELDOUT_MONTHS),
        "selected_valid_route": selection["selected_route"],
        "training_exclusion": "Both months excluded from fit and internal early stopping",
        "architecture": {
            "model": "CatBoost GPU residual to own AOBT proxy",
            "depth": 10, "iterations": 10000, "seed": 2026,
            "early_stop_rounds": 200,
            "other_params": "deep_timestamp_expert.params, unchanged",
        },
        "core_training_labels": "finite 2025 target in [0,86400] and own proxy in [0,7200]",
        "heldout_rows": "all finite-label February/August rows with own proxy in [0,7200]",
        "feature_names": names,
        "feature_schema_sha256": schema_hash(names),
        "input_sha256": {name: sha256(path) for name, path in source_files.items()},
        "raw_training_sha256": {p.name: sha256(p) for p in raw},
        "official_scores_used": False,
    }
    path = args.output_dir / "protocol.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise ValueError("Reserved comparator protocol or input hashes changed")
    else:
        write_json(path, value)
    return value


def validate_rows_and_features(rows: pd.DataFrame, features: pd.DataFrame,
                               expected: list[str]) -> tuple[np.ndarray, np.ndarray,
                                                                 np.ndarray, np.ndarray]:
    required = {"MVT_ID_mvt", "target", "proxy", "month", "time"}
    if (not required.issubset(rows.columns) or len(rows) != len(features)
            or rows.MVT_ID_mvt.isna().any() or rows.MVT_ID_mvt.duplicated().any()
            or list(features) != expected or features.columns.duplicated().any()):
        raise ValueError("v7 rows, ID coverage, or exact feature schema changed")
    timestamp = pd.to_datetime(rows.time, utc=True, errors="coerce")
    if (timestamp.isna().any() or not timestamp.dt.year.eq(2025).all()
            or not np.array_equal(timestamp.dt.month.to_numpy(),
                                  rows.month.to_numpy())):
        raise ValueError("Cached month and movement timestamp disagree")
    y = rows.target.to_numpy(dtype=float)
    proxy = rows.proxy.to_numpy(dtype=float)
    valid_proxy = np.isfinite(proxy) & (proxy >= 0) & (proxy <= 7200)
    heldout = rows.month.isin(HELDOUT_MONTHS).to_numpy()
    train = np.flatnonzero(~heldout & valid_proxy & np.isfinite(y)
                           & (y >= 0) & (y <= 86400))
    test = np.flatnonzero(heldout & valid_proxy & np.isfinite(y))
    if (len(train) < 100000 or len(test) < 10000
            or not set(rows.month.iloc[test].unique()) == set(HELDOUT_MONTHS)
            or np.intersect1d(train, test).size):
        raise ValueError("Reserved fit or held-out finite-label coverage is invalid")
    return train, test, y, proxy


def fit(args: argparse.Namespace) -> None:
    if not (args.output_dir / "protocol.json").exists():
        raise FileNotFoundError("Run --mode prepare and freeze comparator inputs before fitting")
    protocol = frozen_protocol(args)
    output = args.output_dir
    if any((output / name).exists() for name in (
            "feb_aug_oof.parquet", "feb_aug.cbm", "feb_aug_fit.json",
            "feb_aug_manifest.json")):
        raise FileExistsError("Comparator outputs already exist; do not refit silently")
    rows, features = traffic.load_features(args)
    train, test, y, proxy = validate_rows_and_features(
        rows, features, protocol["feature_names"])
    rng = np.random.default_rng(2026)
    order = rng.permutation(train)
    n_early = max(20000, int(.06 * len(order)))
    early, fit_index = order[:n_early], order[n_early:]
    if (len(fit_index) == 0 or len(early) == 0
            or rows.month.iloc[np.r_[early, fit_index]].isin(HELDOUT_MONTHS).any()):
        raise ValueError("February/August labels entered fitting or early stopping")
    categories = features.select_dtypes(include="category").columns.tolist()
    residual = y - proxy
    model = CatBoostRegressor(**deep.params(args))
    pool = Pool(features.iloc[fit_index], label=residual[fit_index],
                cat_features=categories)
    evaluation = Pool(features.iloc[early], label=residual[early],
                      cat_features=categories)
    start = time.monotonic()
    model.fit(pool, eval_set=evaluation, early_stopping_rounds=200,
              use_best_model=True)
    elapsed = time.monotonic() - start
    del pool, evaluation
    gc.collect()
    prediction = proxy[test] + model.predict(features.iloc[test],
                                              thread_count=args.threads)
    if not np.isfinite(prediction).all():
        raise ValueError("Comparator predictions must be finite")
    heldout = rows.iloc[test]
    oof = pd.DataFrame({
        "MVT_ID_mvt": heldout.MVT_ID_mvt.to_numpy(copy=True),
        "target": y[test],
        "MVT_TIME_UTC_mvt": pd.to_datetime(heldout.time, utc=True)
                              .reset_index(drop=True),
        "expert": prediction,
    })
    if (list(oof) != list(OUTPUT_COLUMNS) or oof.MVT_ID_mvt.isna().any()
            or oof.MVT_ID_mvt.duplicated().any()
            or not np.array_equal(oof.MVT_ID_mvt.to_numpy(),
                                  heldout.MVT_ID_mvt.to_numpy())
            or not np.array_equal(oof.target.to_numpy(dtype=float), y[test])
            or not np.isfinite(oof[["target", "expert"]].to_numpy(dtype=float)).all()
            or pd.to_datetime(oof.MVT_TIME_UTC_mvt, utc=True).isna().any()):
        raise ValueError("Comparator output ID, label, time or prediction mismatch")
    output.mkdir(parents=True, exist_ok=True)
    model_path = output / "feb_aug.cbm"
    model.save_model(str(model_path))
    oof_path = output / "feb_aug_oof.parquet"
    temp = oof_path.with_suffix(".parquet.tmp")
    oof.to_parquet(temp, index=False)
    temp.replace(oof_path)
    report_path = output / "feb_aug_fit.json"
    report = {
        "heldout_months": list(HELDOUT_MONTHS),
        "fit_rows": len(fit_index), "early_stop_rows": len(early),
        "heldout_rows": len(test),
        "heldout_month_counts": {str(month): int((rows.month.iloc[test] == month).sum())
                                 for month in HELDOUT_MONTHS},
        "best_iteration": model.get_best_iteration(),
        "trees": model.tree_count_, "fit_seconds": elapsed,
        "feature_names": list(features),
        "feature_schema_sha256": schema_hash(list(features)),
        "categorical_features": categories,
        "model_params": deep.params(args),
        "fit_recipe": "Exact deep_timestamp_expert.fit_fold core mask, RNG split, residual and early-stop settings",
        "heldout_labels_not_fit": True,
    }
    write_json(report_path, report)
    manifest = {
        "heldout_months": list(HELDOUT_MONTHS),
        "comparator_model_sha256": sha256(model_path),
        "oof_sha256": sha256(oof_path),
        "fit_report_sha256": sha256(report_path),
        "protocol_sha256": sha256(output / "protocol.json"),
        "frozen_v7_protocol_sha256": sha256(V7_PROTOCOL),
        "reserved_guard_protocol_sha256": sha256(GUARD_PROTOCOL),
        "selected_valid_route": protocol["selected_valid_route"],
        "selected_valid_policy_sha256": sha256(SELECTED_POLICY),
        "feature_schema_sha256": schema_hash(list(features)),
        "feature_names": list(features),
        "input_sha256": protocol["input_sha256"],
        "raw_training_sha256": protocol["raw_training_sha256"],
        "oof_columns": list(OUTPUT_COLUMNS),
        "heldout_month_counts": report["heldout_month_counts"],
    }
    write_json(output / "feb_aug_manifest.json", manifest)
    print(json.dumps({"model": str(model_path), "oof": str(oof_path),
                      "heldout_rows": len(oof), "trees": model.tree_count_,
                      "fit_seconds": elapsed}, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "fit"), default="prepare")
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
    parser.add_argument("--output-dir", type=Path,
                        default=Path("artifacts/v8-reserved-v7"))
    args = parser.parse_args()
    args.iterations = 10000
    args.depth = 10
    args.threads = 2
    if args.mode == "prepare":
        protocol = frozen_protocol(args)
        print(json.dumps({"protocol": str(args.output_dir / "protocol.json"),
                          "features": len(protocol["feature_names"]),
                          "heldout_months": list(HELDOUT_MONTHS)}), flush=True)
    else:
        fit(args)


if __name__ == "__main__":
    main()
