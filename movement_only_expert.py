"""Movement-only taxi-out transfer experiment for the no-NM fallback gate.

The model trains on ordinary 2025 departures from every airport. Its predictor
matrix contains movement, schedule, movement-time traffic, NOAA weather, and
released arrival covariates only. NM/flight-table fields are read solely to
identify the predeclared no-NM evaluation gate; they never enter the model.

Protocol: prepare -> fit-fold for each complementary split -> evaluate. A
positive local result is still provisional until an independent April/October
paired architecture audit. This module does not create ranking predictions.
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
import os
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl

from solution import AIRPORT_TZ, _training_files
from weather_model import add_weather


FOLDS = {"seasonal_jan_jul": (1, 7), "forward_nov_dec": (11, 12)}
WEIGHTS = (0.0, 0.25, 0.5, 1.0)
REFERENCE_SHA256 = "036da1502f4ae68238927e31a0479b9cb82a6fb31b7acdbab26aa0657b7a67f8"
SEED = 20261002

# Every cached column here has movement-only provenance in solution.build_features.
# In particular the AOBT backlog, *_flt, NM agreement, and proxy columns are
# absent. The schedule gap below is independently reconstructed from mvt clocks.
SAFE_CACHED_NUMERIC = (
    "dep_prev5", "dep_prev15", "dep_prev30", "dep_next15",
    "arr_prev15", "arr_prev30", "arr_next15", "same_runway_prev15",
    "utc_hour", "utc_weekday", "utc_month", "utc_dayofyear", "is_weekend",
    "local_hour", "local_weekday", "hour_sin", "hour_cos", "year_sin",
    "year_cos", "stand_number", "flight_number",
)
SAFE_CACHED_CATEGORICAL = (
    "ADEP_mvt", "ADES_mvt", "RUNWAY_mvt", "STAND_mvt",
    "AIRCRAFT_TYPE_mvt", "FLIGHT_RULE_mvt", "flight_prefix",
    "stand_zone", "route", "airport_runway", "airport_stand",
    "airport_stand_runway",
)
ARRIVAL_COLUMNS = (
    "arr_taxi_count_15m", "arr_taxi_mean_15m", "arr_taxi_highshare_15m",
    "arr_taxi_count_60m", "arr_taxi_mean_60m", "arr_taxi_highshare_60m",
    "arr_taxi_count_180m", "arr_taxi_mean_180m", "arr_taxi_highshare_180m",
    "arr_taxiing_now", "arr_block_prev_60m",
    "arr_same_stand_last_block_gap", "arr_same_stand_last_taxi",
    "arr_same_stand_blocks_4h",
)
DERIVED_COLUMNS = (
    "flight_name_mvt", "mvt_utc_minute", "mvt_utc_second",
    "schedule_utc_hour", "schedule_utc_minute", "schedule_utc_second",
    "schedule_utc_weekday", "schedule_local_hour", "schedule_local_weekday",
    "mvt_schedule_gap_seconds", "mvt_schedule_day_offset", "schedule_missing",
)
FORBIDDEN_NAME_PARTS = ("_flt", "aobt", "lobt", "iobt", "eobt",
                        "callsign", "flight_id", "mvt_id", "taxitime", "target")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def available_memory_gib() -> float:
    if os.name == "nt":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(MemoryStatus)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise OSError("GlobalMemoryStatusEx failed")
        return status.ullAvailPhys / 2**30
    if Path("/proc/meminfo").exists():
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024 / 2**30
    raise RuntimeError("Could not establish available physical memory")


def require_memory(min_gib: float) -> None:
    available = available_memory_gib()
    if available < min_gib:
        raise MemoryError(f"Need at least {min_gib:g} GiB free; currently {available:.2f} GiB")


def protocol() -> dict:
    return {
        "reference": "Frozen clipped v5 all-finite OOF",
        "reference_sha256": REFERENCE_SHA256,
        "purpose": "Transfer learned movement behavior from ordinary departures to invalid-AOBT no-NM non-LIRF rows",
        "selection_months": [1, 7],
        "corroboration_months": [11, 12],
        "required_fresh_architecture_audit_months": [4, 10],
        "training": "All complementary-month 2025 departures with finite taxi-out target from 0 to 7200 seconds",
        "internal_early_stop": "Calendar day number modulo 11 equals zero within complementary months; no held-out labels",
        "gate": "Invalid AOBT, AOBT_3_flt and LOBT_flt missing, airport != LIRF; gate metadata never enters model",
        "feature_sources": ["movement and schedule", "MVT-time airport/runway traffic",
                            "NOAA GHCNh weather", "released arrival cache"],
        "feature_exclusions": ["all *_flt/NM fields", "NM or movement IDs", "callsign",
                               "off-block proxies", "AOBT-time backlog", "departure BLOCK/TAXITIME"],
        "model": {"family": "LightGBM direct regression", "leaves": 63,
                  "max_rounds": 1200, "early_stop_rounds": 100,
                  "threads": 3, "seed": SEED},
        "candidate_weights": list(WEIGHTS),
        "selection": "Minimum all-finite Jan/Jul RMSE; ties favor smaller weight",
        "promotion": "Selected positive weight improves all-finite RMSE and UTC-day bootstrap 95% lower gain bound >0 on both folds; independent Apr/Oct paired architecture audit also required",
        "fresh_architecture_audit": {
            "reference": "Refit the accepted v5 ordinary CatBoost direct architecture excluding April/October labels",
            "candidate": "Refit movement-only LightGBM on ordinary departures excluding April/October labels",
            "comparison": "On finite no-NM invalid-AOBT non-LIRF April/October rows, compare CatBoost direct against the fixed Jan/Jul weight blend of CatBoost direct and movement-only direct, clipped at zero",
            "pass_rule": "Both individual months improve RMSE and pooled UTC-day bootstrap 95% gain lower bound is positive",
            "no_in_sample_v5": True,
        },
        "leaderboard_use": "No leaderboard feedback enters model or selection",
        "status": "Predeclared protocol; fit and validation artifacts track completion",
    }


def ensure_protocol(out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "protocol.json"
    planned = protocol()
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != planned:
            raise ValueError("Existing movement-only protocol differs from predeclared protocol")
    else:
        path.write_text(json.dumps(planned, indent=2) + "\n", encoding="utf-8")
    return planned


def verify_reference(path: Path) -> None:
    actual = sha256(path)
    if actual != REFERENCE_SHA256:
        raise ValueError(f"Frozen v5 OOF SHA changed: {actual}")


def read_baseline_rows(cache_dir: Path, columns: list[str]) -> pd.DataFrame:
    rows = pd.read_parquet(cache_dir / "training_rows.parquet", columns=columns)
    if rows.MVT_ID_mvt.isna().any() or rows.MVT_ID_mvt.duplicated().any():
        raise ValueError("Baseline movement IDs are missing or duplicated")
    return rows


def assert_safe_matrix(features: pd.DataFrame, names: list[str]) -> None:
    if list(features.columns) != names or len(names) != len(set(names)):
        raise ValueError("Movement-only feature schema differs from the allowlist")
    forbidden = [name for name in names
                 if (any(part in name.lower() for part in FORBIDDEN_NAME_PARTS)
                     or ("proxy" in name.lower() and not name.startswith("wx_")))]
    if forbidden:
        raise ValueError(f"Forbidden predictor names: {forbidden}")
    if any(not (pd.api.types.is_numeric_dtype(features[name]) or
                isinstance(features[name].dtype, pd.CategoricalDtype)) for name in names):
        raise ValueError("Only numeric and categorical features are allowed")


def prepare(args: argparse.Namespace) -> dict:
    require_memory(args.min_free_gib)
    ensure_protocol(args.output_dir)
    verify_reference(args.v5_oof)
    cached_names = list(SAFE_CACHED_NUMERIC + SAFE_CACHED_CATEGORICAL)
    features = pd.read_parquet(args.cache_dir / "features.parquet", columns=cached_names)
    rows = read_baseline_rows(args.cache_dir,
                              ["MVT_ID_mvt", "target", "proxy", "month", "airport", "time"])
    if len(features) != len(rows):
        raise ValueError("Baseline features and row records have different lengths")
    if not features.ADEP_mvt.astype("string").eq(rows.airport.astype("string")).all():
        raise ValueError("Baseline feature/row airports do not align")

    raw = (pl.scan_parquet([str(p) for p in _training_files(args.data_dir)])
           .filter(pl.col("PHASE_mvt") == "DEP")
           .select(["MVT_ID_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
                    "FLIGHT_mvt"]).collect().to_pandas())
    if not np.array_equal(raw.MVT_ID_mvt.to_numpy(), rows.MVT_ID_mvt.to_numpy()):
        raise ValueError("Raw departure order differs from baseline cache IDs")
    mvt = pd.to_datetime(raw.MVT_TIME_UTC_mvt, utc=True, errors="coerce")
    sched = pd.to_datetime(raw.SCHED_TIME_UTC_mvt, utc=True, errors="coerce")
    if not mvt.eq(pd.to_datetime(rows.time, utc=True)).all():
        raise ValueError("Raw movement times differ from baseline row times")
    features["flight_name_mvt"] = (raw.FLIGHT_mvt.astype("string")
                                   .fillna("__MISSING__").astype("category"))
    features["mvt_utc_minute"] = mvt.dt.minute.astype("float32")
    features["mvt_utc_second"] = mvt.dt.second.astype("float32")
    features["schedule_utc_hour"] = sched.dt.hour.astype("float32")
    features["schedule_utc_minute"] = sched.dt.minute.astype("float32")
    features["schedule_utc_second"] = sched.dt.second.astype("float32")
    features["schedule_utc_weekday"] = sched.dt.dayofweek.astype("float32")
    schedule_local_hour = np.full(len(rows), np.nan, dtype=np.float32)
    schedule_local_weekday = np.full(len(rows), np.nan, dtype=np.float32)
    for airport, timezone_name in AIRPORT_TZ.items():
        idx = np.flatnonzero(rows.airport.eq(airport).to_numpy())
        if len(idx):
            local = sched.iloc[idx].dt.tz_convert(timezone_name)
            schedule_local_hour[idx] = local.dt.hour.to_numpy(dtype=np.float32)
            schedule_local_weekday[idx] = local.dt.dayofweek.to_numpy(dtype=np.float32)
    features["schedule_local_hour"] = schedule_local_hour
    features["schedule_local_weekday"] = schedule_local_weekday
    gap = (mvt - sched).dt.total_seconds()
    features["mvt_schedule_gap_seconds"] = gap.clip(-604800, 604800).astype("float32")
    features["mvt_schedule_day_offset"] = np.floor(gap / 86400).clip(-30, 30).astype("float32")
    features["schedule_missing"] = sched.isna().astype("int8")
    del raw, mvt, sched, gap
    gc.collect()

    features = add_weather(features, rows[["airport", "time"]], args.weather_file)
    arrivals = pd.read_parquet(args.arrival_cache,
                               columns=["MVT_ID_mvt", *ARRIVAL_COLUMNS])
    if not np.array_equal(arrivals.MVT_ID_mvt.to_numpy(), rows.MVT_ID_mvt.to_numpy()):
        raise ValueError("Released ARR feature cache differs in movement ID order")
    for name in ARRIVAL_COLUMNS:
        features[name] = pd.to_numeric(arrivals[name], errors="coerce").astype("float32")
    del arrivals
    gc.collect()
    weather_names = [name for name in features if name.startswith("wx_")]
    names = cached_names + list(DERIVED_COLUMNS) + weather_names + list(ARRIVAL_COLUMNS)
    assert_safe_matrix(features, names)
    categorical = [name for name in names
                   if isinstance(features[name].dtype, pd.CategoricalDtype)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    temp = args.output_dir / "features.tmp.parquet"
    features.to_parquet(temp, index=False)
    temp.replace(args.output_dir / "features.parquet")
    pd.DataFrame({"MVT_ID_mvt": rows.MVT_ID_mvt}).to_parquet(
        args.output_dir / "row_ids.parquet", index=False)
    manifest = {"rows": len(rows), "features": names, "categorical": categorical,
                "baseline_rows_sha256": sha256(args.cache_dir / "training_rows.parquet"),
                "baseline_features_sha256": sha256(args.cache_dir / "features.parquet"),
                "arrival_cache_sha256": sha256(args.arrival_cache),
                "weather_file_sha256": sha256(args.weather_file),
                "reference_sha256": REFERENCE_SHA256,
                "weather_source": "NOAA NCEI GHCNh CC0-1.0",
                "arrival_source": "Released ARR movement cache"}
    (args.output_dir / "features_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {"rows": len(rows), "features": len(names), "categorical": len(categorical)}


def load_prepared(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    manifest = json.loads((args.output_dir / "features_manifest.json").read_text(encoding="utf-8"))
    if sha256(args.cache_dir / "training_rows.parquet") != manifest["baseline_rows_sha256"]:
        raise ValueError("Baseline training rows changed after feature preparation")
    if sha256(args.cache_dir / "features.parquet") != manifest["baseline_features_sha256"]:
        raise ValueError("Baseline cached features changed after feature preparation")
    features = pd.read_parquet(args.output_dir / "features.parquet")
    rows = read_baseline_rows(args.cache_dir,
                              ["MVT_ID_mvt", "target", "proxy", "month", "airport", "time"])
    ids = pd.read_parquet(args.output_dir / "row_ids.parquet", columns=["MVT_ID_mvt"])
    if len(features) != len(rows) or not np.array_equal(ids.MVT_ID_mvt, rows.MVT_ID_mvt):
        raise ValueError("Prepared movement features no longer align with row IDs")
    assert_safe_matrix(features, manifest["features"])
    if not features.ADEP_mvt.astype("string").eq(rows.airport.astype("string")).all():
        raise ValueError("Prepared movement features no longer align with airports")
    return features, rows, manifest


def model_params(threads: int) -> dict:
    return {"objective": "regression", "metric": "rmse", "learning_rate": 0.04,
            "num_leaves": 63, "min_data_in_leaf": 90, "lambda_l2": 8.0,
            "feature_fraction": 0.85, "bagging_fraction": 0.85,
            "bagging_freq": 1, "max_cat_threshold": 64, "cat_smooth": 20,
            "verbosity": -1, "num_threads": threads, "seed": SEED,
            "feature_fraction_seed": SEED, "bagging_seed": SEED,
            "deterministic": True, "force_col_wise": True}


def read_gate(cache_dir: Path, rows: pd.DataFrame) -> np.ndarray:
    flags = pd.read_parquet(cache_dir / "features.parquet",
                            columns=["AOBT_3_flt_missing", "LOBT_flt_missing"])
    if len(flags) != len(rows):
        raise ValueError("Gate metadata does not align with baseline rows")
    return (flags.AOBT_3_flt_missing.to_numpy(dtype=bool)
            & flags.LOBT_flt_missing.to_numpy(dtype=bool)
            & ~np.isfinite(rows.proxy.to_numpy(dtype=float))
            & ~rows.airport.eq("LIRF").to_numpy(dtype=bool))


def fit_fold(args: argparse.Namespace) -> dict:
    require_memory(args.min_free_gib)
    ensure_protocol(args.output_dir)
    verify_reference(args.v5_oof)
    features, rows, manifest = load_prepared(args)
    months = FOLDS[args.fold]
    y = rows.target.to_numpy(dtype=np.float32)
    month = rows.month.to_numpy(dtype=np.int16)
    heldout = np.isin(month, months)
    ordinary = np.isfinite(y) & (y >= 0) & (y <= 7200) & ~heldout
    day = pd.to_datetime(rows.time, utc=True, errors="coerce").dt.floor("D")
    if day.isna().any():
        raise ValueError("Every training departure must have a movement date")
    day_number = day.dt.as_unit("ns").astype("int64").to_numpy() // 86_400_000_000_000
    internal = ordinary & (day_number % 11 == 0)
    fit_mask = ordinary & ~internal
    if fit_mask.sum() < 100_000 or internal.sum() < 10_000:
        raise ValueError("Insufficient ordinary fit or internal early-stop rows")
    gate = read_gate(args.cache_dir, rows)
    predict_mask = heldout & gate & np.isfinite(y)
    if predict_mask.sum() < 100:
        raise ValueError("Unexpectedly few held-out no-NM non-LIRF rows")
    cats = manifest["categorical"]
    train_set = lgb.Dataset(features.loc[fit_mask], label=y[fit_mask],
                            categorical_feature=cats, free_raw_data=True)
    early_set = lgb.Dataset(features.loc[internal], label=y[internal],
                            reference=train_set, categorical_feature=cats,
                            free_raw_data=True)
    model = lgb.train(model_params(args.threads), train_set, num_boost_round=1200,
                      valid_sets=[early_set], callbacks=[
                          lgb.early_stopping(100, verbose=True),
                          lgb.log_evaluation(period=100)])
    best = int(model.best_iteration or 1200)
    expert = model.predict(features.loc[predict_mask], num_iteration=best,
                           num_threads=args.threads)
    if not np.isfinite(expert).all():
        raise ValueError("Movement-only expert produced nonfinite predictions")
    output = rows.loc[predict_mask, ["MVT_ID_mvt", "target", "airport", "month", "time"]].copy()
    output["fold"] = args.fold
    output["expert"] = np.maximum(expert, 0).astype("float32")
    output.to_parquet(args.output_dir / f"{args.fold}_oof.parquet", index=False)
    model.save_model(str(args.output_dir / f"{args.fold}.txt"))
    report = {"fold": args.fold, "heldout_months": list(months),
              "ordinary_complement_rows": int(ordinary.sum()),
              "fit_rows": int(fit_mask.sum()), "internal_early_rows": int(internal.sum()),
              "heldout_gate_rows": int(predict_mask.sum()),
              "best_round": best, "feature_count": len(features.columns),
              "source": "2025 local train labels, no competition leaderboard feedback"}
    (args.output_dir / f"{args.fold}_fit.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def rmse(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(y - pred))))


def day_bootstrap(frame: pd.DataFrame, base: np.ndarray, candidate: np.ndarray,
                  mask: np.ndarray, seed: int) -> dict:
    dates = pd.to_datetime(frame.loc[mask, "MVT_TIME_UTC_mvt"], utc=True).dt.floor("D")
    groups, unique = pd.factorize(dates, sort=True)
    if len(unique) < 20:
        raise ValueError("Too few UTC days for fold stability assessment")
    yy = frame.target.to_numpy(dtype=float)[mask]
    count = np.bincount(groups, minlength=len(unique)).astype(float)
    base_sse = np.bincount(groups, weights=np.square(yy - base[mask]),
                           minlength=len(unique))
    new_sse = np.bincount(groups, weights=np.square(yy - candidate[mask]),
                          minlength=len(unique))
    rng = np.random.default_rng(seed)
    drawn = rng.integers(0, len(unique), size=(1000, len(unique)))
    denominator = count[drawn].sum(axis=1)
    gain = (np.sqrt(base_sse[drawn].sum(axis=1) / denominator)
            - np.sqrt(new_sse[drawn].sum(axis=1) / denominator))
    return {"days": len(unique), "repeats": 1000,
            "observed_gain_sec": rmse(yy, base[mask]) - rmse(yy, candidate[mask]),
            "gain_ci95_sec": [float(value) for value in np.quantile(gain, [.025, .975])],
            "fraction_positive": float(np.mean(gain > 0))}


def evaluate(args: argparse.Namespace) -> dict:
    ensure_protocol(args.output_dir)
    verify_reference(args.v5_oof)
    manifest = json.loads((args.output_dir / "features_manifest.json").read_text(encoding="utf-8"))
    if sha256(args.cache_dir / "features.parquet") != manifest["baseline_features_sha256"]:
        raise ValueError("Baseline gate metadata changed after feature preparation")
    base = pd.read_parquet(args.v5_oof,
                           columns=["MVT_ID_mvt", "target", "fold", "airport",
                                    "month", "MVT_TIME_UTC_mvt", "a_valid", "selected"])
    if base.MVT_ID_mvt.isna().any() or base.MVT_ID_mvt.duplicated().any():
        raise ValueError("Frozen v5 IDs are missing or duplicated")
    if set(base.fold) != set(FOLDS) or not np.isfinite(base.target).all():
        raise ValueError("Frozen v5 OOF coverage or labels changed")
    rows = read_baseline_rows(args.cache_dir,
                              ["MVT_ID_mvt", "target", "proxy", "airport", "month"])
    positions = pd.Index(rows.MVT_ID_mvt).get_indexer(base.MVT_ID_mvt)
    if np.any(positions < 0):
        raise ValueError("Frozen v5 contains IDs absent from baseline training rows")
    if (not np.array_equal(base.target.to_numpy(), rows.target.to_numpy()[positions])
            or not np.array_equal(base.airport.astype("string").to_numpy(),
                                  rows.airport.astype("string").to_numpy()[positions])
            or not np.array_equal(base.month.to_numpy(), rows.month.to_numpy()[positions])):
        raise ValueError("Frozen v5 labels, airport, or month differ from baseline")
    gate = read_gate(args.cache_dir, rows)[positions]
    gate &= ~base.a_valid.to_numpy(dtype=bool)
    sources = []
    for fold in FOLDS:
        part = pd.read_parquet(args.output_dir / f"{fold}_oof.parquet",
                               columns=["MVT_ID_mvt", "fold", "expert"])
        if not part.fold.eq(fold).all():
            raise ValueError(f"Expert fold marker differs for {fold}")
        sources.append(part)
    source = pd.concat(sources, ignore_index=True)
    if source.MVT_ID_mvt.duplicated().any():
        raise ValueError("Movement-only OOF IDs repeat across folds")
    aligned = source.set_index("MVT_ID_mvt").reindex(base.MVT_ID_mvt)
    expert = aligned.expert.to_numpy(dtype=float)
    if not np.array_equal(np.isfinite(expert), gate):
        raise ValueError("Movement-only expert coverage differs from predeclared gate")
    if not np.array_equal(aligned.fold.to_numpy()[gate], base.fold.to_numpy()[gate]):
        raise ValueError("Movement-only expert fold differs from frozen v5")
    y = base.target.to_numpy(dtype=float)
    frozen = base.selected.to_numpy(dtype=float)
    if not np.isfinite(frozen).all() or np.any(frozen < 0):
        raise ValueError("Frozen v5 policy is not finite and clipped")
    predictions = {}
    for weight in WEIGHTS:
        pred = frozen.copy()
        pred[gate] = np.maximum((1 - weight) * frozen[gate]
                                + weight * expert[gate], 0)
        predictions[str(weight)] = pred
    scores = {}
    for fold in FOLDS:
        mask = base.fold.eq(fold).to_numpy()
        scores[fold] = {"n": int(mask.sum()), "gate_n": int((mask & gate).sum()),
                        "all_finite_rmse_sec": {name: rmse(y[mask], pred[mask])
                                                for name, pred in predictions.items()},
                        "gate_rmse_sec": {name: rmse(y[mask & gate], pred[mask & gate])
                                          for name, pred in predictions.items()}}
    selected = min(WEIGHTS,
                   key=lambda weight: (scores["seasonal_jan_jul"]["all_finite_rmse_sec"][str(weight)],
                                       weight))
    stability = {}
    for offset, fold in enumerate(FOLDS):
        mask = base.fold.eq(fold).to_numpy()
        stability[fold] = day_bootstrap(base, frozen, predictions[str(selected)],
                                        mask, SEED + offset)
    passed = (selected > 0 and all(
        scores[fold]["all_finite_rmse_sec"][str(selected)]
        < scores[fold]["all_finite_rmse_sec"]["0.0"]
        and stability[fold]["gain_ci95_sec"][0] > 0 for fold in FOLDS))
    report = {"reference_sha256": REFERENCE_SHA256,
              "selection": "Jan/Jul all-finite RMSE only; Nov/Dec unchanged weight",
              "weights": list(WEIGHTS), "scores": scores,
              "selected_weight": selected, "day_stability": stability,
              "both_existing_folds_passed": passed,
              "promoted": False,
              "promotion_pending": "Independent April/October paired architecture audit" if passed else
                                   "Local two-fold gate failed"}
    (args.output_dir / "validation.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame({"MVT_ID_mvt": base.MVT_ID_mvt, "fold": base.fold,
                  "target": y, "airport": base.airport, "month": base.month,
                  "MVT_TIME_UTC_mvt": base.MVT_TIME_UTC_mvt,
                  "gate": gate, "v5": frozen,
                  "expert": expert,
                  "selected_candidate": predictions[str(selected)]}).to_parquet(
                      args.output_dir / "validation_predictions.parquet", index=False)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("protocol", "prepare", "fit-fold", "evaluate"),
                        required=True)
    parser.add_argument("--fold", choices=tuple(FOLDS))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/baseline"))
    parser.add_argument("--arrival-cache", type=Path,
                        default=Path("artifacts/v5-arrival-clean/training_arrival_features.parquet"))
    parser.add_argument("--weather-file", type=Path,
                        default=Path("data/external/weather.parquet"))
    parser.add_argument("--v5-oof", type=Path,
                        default=Path("artifacts/v5-ensemble/validation_predictions.parquet"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("artifacts/v6-movement-only"))
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--min-free-gib", type=float, default=10.0)
    args = parser.parse_args()
    if args.threads < 1 or args.threads > 3:
        raise ValueError("Movement-only fit is limited to 3 CPU threads")
    if args.mode == "fit-fold" and not args.fold:
        parser.error("--fit-fold requires --fold")
    if args.mode == "protocol":
        result = ensure_protocol(args.output_dir)
    elif args.mode == "prepare":
        result = prepare(args)
    elif args.mode == "fit-fold":
        result = fit_fold(args)
    else:
        result = evaluate(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
