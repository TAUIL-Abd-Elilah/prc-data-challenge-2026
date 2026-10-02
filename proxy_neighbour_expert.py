"""Ranking-safe neighbouring departure AOBT proxy congestion experiment.

Raw feature reads are restricted to released DEP MVT/AOBT, runway and airport
covariates. Departure BLOCK and TAXITIME columns are never read as predictors.
The query row is excluded because past windows end strictly before its takeoff
and future windows begin strictly after its takeoff.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl

from catboost_expert import add_flight_weather, load_cache
from solution import _training_files


RAW_DEP_COLS = ["MVT_ID_mvt", "ADEP_mvt", "RUNWAY_mvt", "MVT_TIME_UTC_mvt",
                "AOBT_3_flt"]
WINDOWS = (900, 3600)
WEIGHTS = (0.0, 0.1, 0.25, 0.5, 1.0)
FOLDS = {"seasonal_jan_jul": (1, 7), "forward_nov_dec": (11, 12)}
SEED = 20261002


def feature_names() -> list[str]:
    return [f"nbr_{scope}_{direction}{window // 60}m_{stat}"
            for scope in ("airport", "runway")
            for direction in ("past", "future")
            for window in WINDOWS
            for stat in ("count", "mean", "std")]


def protocol() -> dict:
    return {
        "purpose": "2025 local model comparison; no ranking labels or public-score tuning",
        "raw_predictor_columns": RAW_DEP_COLS,
        "raw_phase_filter": "PHASE_mvt == DEP",
        "forbidden_predictor_columns": ["BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt"],
        "source_proxy": "MVT_TIME_UTC_mvt - AOBT_3_flt in [0,7200] seconds",
        "query_exclusion": "Past [t-window,t) and future (t,t+window] omit the query and all same-second peers",
        "groupings": ["ADEP_mvt", "ADEP_mvt + RUNWAY_mvt"],
        "windows_seconds": list(WINDOWS),
        "feature_names": feature_names(),
        "model": {"objective": "direct residual target-minus-own-AOBT-proxy",
                  "algorithm": "LightGBM regression", "max_rounds": 800,
                  "num_leaves": 31, "num_threads": 2,
                  "learning_rate": .045, "min_data_in_leaf": 150,
                  "internal_early_stop_rounds": 80,
                  "eligible_training_target": "finite, 0..86400 seconds; own proxy 0..7200",
                  "heldout_scoring_target": "all finite labels, including negative and extreme"},
        "validation": {"selection_months": [1, 7], "forward_months": [11, 12],
                       "comparison": "frozen v5 selected, clipped >=0",
                       "blend_weights": list(WEIGHTS),
                       "choose": "lowest all-finite seasonal RMSE; ties select lower weight",
                       "forward": "apply unchanged selected weight",
                       "bootstrap": "1000 paired UTC-day resamples within each held-out month; seed 20261002",
                       "promotion_gate": "selected weight >0, RMSE gain and 95% day-block lower bound >0 in both folds"},
        "limitations": "Repeatedly examined 2025 folds and inherited v5 base-model limitations make this a comparison estimate, not untouched 2026 performance."
    }


def _seconds(values: pd.Series) -> np.ndarray:
    dt = pd.to_datetime(values, utc=True, errors="coerce")
    return dt.dt.as_unit("ns").astype("int64").to_numpy() // 1_000_000_000


def _group_features(raw: pd.DataFrame, keys: np.ndarray, prefix: str,
                    time_sec: np.ndarray, proxy: np.ndarray,
                    valid: np.ndarray, out: pd.DataFrame) -> None:
    query_groups = pd.Series(keys).groupby(keys, sort=False).indices
    events = pd.Series(keys[valid]).groupby(keys[valid], sort=False).indices
    event_global = np.flatnonzero(valid)
    for group_key, query_idx in query_groups.items():
        if group_key not in events:
            continue
        event_idx = event_global[np.asarray(events[group_key], dtype=np.int64)]
        event_idx = event_idx[np.argsort(time_sec[event_idx], kind="stable")]
        times = time_sec[event_idx]
        values = proxy[event_idx].astype(float)
        cumulative = np.r_[0., np.cumsum(values)]
        cumulative_sq = np.r_[0., np.cumsum(values ** 2)]
        query_idx = np.asarray(query_idx, dtype=np.int64)
        query_time = time_sec[query_idx]
        for direction in ("past", "future"):
            for window in WINDOWS:
                if direction == "past":
                    start = np.searchsorted(times, query_time - window, side="left")
                    end = np.searchsorted(times, query_time, side="left")
                else:
                    start = np.searchsorted(times, query_time, side="right")
                    end = np.searchsorted(times, query_time + window, side="right")
                count = end - start
                total = cumulative[end] - cumulative[start]
                total_sq = cumulative_sq[end] - cumulative_sq[start]
                mean = np.divide(total, count, out=np.full(len(count), np.nan),
                                 where=count > 0)
                second = np.divide(total_sq, count, out=np.full(len(count), np.nan),
                                   where=count > 0)
                std = np.sqrt(np.maximum(second - mean ** 2, 0))
                stem = f"nbr_{prefix}_{direction}{window // 60}m_"
                out.iloc[query_idx, out.columns.get_loc(stem + "count")] = count.astype(np.float32)
                out.iloc[query_idx, out.columns.get_loc(stem + "mean")] = mean.astype(np.float32)
                out.iloc[query_idx, out.columns.get_loc(stem + "std")] = std.astype(np.float32)


def neighbour_features(raw: pd.DataFrame) -> pd.DataFrame:
    n = len(raw)
    names = feature_names()
    out = pd.DataFrame(index=raw.index)
    for name in names:
        out[name] = (np.zeros(n, dtype=np.float32) if name.endswith("_count")
                     else np.full(n, np.nan, dtype=np.float32))
    time_sec = _seconds(raw.MVT_TIME_UTC_mvt)
    aobt_sec = _seconds(raw.AOBT_3_flt)
    proxy = (time_sec - aobt_sec).astype(float)
    valid = (raw.MVT_TIME_UTC_mvt.notna().to_numpy()
             & raw.AOBT_3_flt.notna().to_numpy()
             & np.isfinite(proxy) & (proxy >= 0) & (proxy <= 7200))
    airport = raw.ADEP_mvt.astype("string").fillna("?").to_numpy(dtype=str)
    runway = raw.RUNWAY_mvt.astype("string").fillna("?").to_numpy(dtype=str)
    _group_features(raw, airport, "airport", time_sec, proxy, valid, out)
    _group_features(raw, np.char.add(np.char.add(airport, "|"), runway),
                    "runway", time_sec, proxy, valid, out)
    # Queries with no movement timestamp cannot define traffic windows.
    bad_query = raw.MVT_TIME_UTC_mvt.isna().to_numpy()
    if bad_query.any():
        out.loc[bad_query, :] = np.nan
    return out


def _build_one(paths: list[Path], rows_path: Path, output: Path) -> dict:
    start = time.monotonic()
    raw = (pl.scan_parquet([str(p) for p in paths])
           .filter(pl.col("PHASE_mvt") == "DEP")
           .select(RAW_DEP_COLS).collect().to_pandas())
    ids = pd.read_parquet(rows_path, columns=["MVT_ID_mvt"])
    if (raw.MVT_ID_mvt.isna().any() or raw.MVT_ID_mvt.duplicated().any()
            or ids.MVT_ID_mvt.isna().any() or ids.MVT_ID_mvt.duplicated().any()):
        raise ValueError("Raw/cached departure IDs must be unique and non-null")
    ordered = ids.merge(raw, on="MVT_ID_mvt", how="left", sort=False,
                        validate="one_to_one")
    if len(ordered) != len(ids) or ordered.MVT_TIME_UTC_mvt.isna().any():
        raise ValueError("Raw covariates do not align with baseline cached rows")
    feat = neighbour_features(ordered)
    feat.insert(0, "MVT_ID_mvt", ids.MVT_ID_mvt.to_numpy())
    output.parent.mkdir(parents=True, exist_ok=True)
    feat.to_parquet(output, index=False)
    raw_proxy = (_seconds(raw.MVT_TIME_UTC_mvt) - _seconds(raw.AOBT_3_flt)).astype(float)
    plausible = (raw.AOBT_3_flt.notna().to_numpy() & np.isfinite(raw_proxy)
                 & (raw_proxy >= 0) & (raw_proxy <= 7200))
    return {"rows": len(ids), "valid_proxy_events": int(plausible.sum()),
            "feature_count": len(names := feature_names()), "feature_names": names,
            "seconds": round(time.monotonic() - start, 3), "path": str(output)}


def build_features(args: argparse.Namespace) -> dict:
    report = {"protocol": protocol()}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    protocol_path = args.output_dir / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text(encoding="utf-8")) != report["protocol"]:
            raise ValueError("Existing neighbour protocol differs; refusing to overwrite it")
    else:
        protocol_path.write_text(json.dumps(report["protocol"], indent=2),
                                 encoding="utf-8")
    report["training"] = _build_one(_training_files(args.data_dir),
                                     args.cache_dir / "training_rows.parquet",
                                     args.output_dir / "training_neighbour_features.parquet")
    gc.collect()
    report["ranking"] = _build_one([args.data_dir / "ranking.parquet"],
                                    args.cache_dir / "ranking_rows.parquet",
                                    args.output_dir / "ranking_neighbour_features.parquet")
    (args.output_dir / "feature_build.json").write_text(json.dumps(report, indent=2),
                                                         encoding="utf-8")
    return {"training": report["training"], "ranking": report["ranking"],
            "protocol_path": str(args.output_dir / "protocol.json")}


def _model_params() -> dict:
    return dict(objective="regression", metric="rmse", learning_rate=.045,
                num_leaves=31, min_data_in_leaf=150, feature_fraction=.85,
                bagging_fraction=.85, bagging_freq=1, lambda_l2=12,
                max_cat_threshold=64, cat_smooth=20, verbosity=-1,
                num_threads=2, seed=SEED, feature_fraction_seed=SEED,
                bagging_seed=SEED, deterministic=True, force_col_wise=True)


def _load_all(args: argparse.Namespace, ranking: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, features = load_cache(args.cache_dir, ranking)
    features = add_flight_weather(rows, features, args.data_dir,
                                  args.weather_file, ranking)
    suffix = "ranking" if ranking else "training"
    neighbour = pd.read_parquet(args.output_dir / f"{suffix}_neighbour_features.parquet")
    if (len(rows) != len(neighbour) or
            not np.array_equal(rows.MVT_ID_mvt.to_numpy(),
                               neighbour.MVT_ID_mvt.to_numpy())):
        raise ValueError("Neighbour feature cache does not align with baseline rows")
    features = pd.concat([features.reset_index(drop=True),
                          neighbour.drop(columns="MVT_ID_mvt").reset_index(drop=True)], axis=1)
    if features.columns.duplicated().any() or not all(c in features for c in feature_names()):
        raise ValueError("Neighbour feature schema is incomplete or duplicated")
    return rows, features


def _rmse(y: np.ndarray, p: np.ndarray) -> float:
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("Nonfinite score input")
    return float(np.sqrt(np.mean((y - p) ** 2)))


def _masks(rows: pd.DataFrame, months: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    y = rows.target.to_numpy(dtype=float)
    proxy = rows.proxy.to_numpy(dtype=float)
    proxy_valid = np.isfinite(proxy) & (proxy >= 0) & (proxy <= 7200)
    heldout = rows.month.isin(months).to_numpy(dtype=bool)
    train_core = np.isfinite(y) & (y >= 0) & (y <= 86400) & proxy_valid
    test = heldout & np.isfinite(y) & proxy_valid
    return train_core & ~heldout, test


def _fold_fit(name: str, months: tuple[int, int], rows: pd.DataFrame,
              features: pd.DataFrame, reference: pd.DataFrame,
              args: argparse.Namespace) -> tuple[pd.DataFrame, dict]:
    train_mask, test_mask = _masks(rows, months)
    train_idx = np.flatnonzero(train_mask)
    test_idx = np.flatnonzero(test_mask)
    rng = np.random.default_rng(SEED + (0 if name == "seasonal_jan_jul" else 17))
    shuffled = rng.permutation(train_idx)
    early_n = max(30000, int(.06 * len(shuffled)))
    early_idx, fit_idx = shuffled[:early_n], shuffled[early_n:]
    y = rows.target.to_numpy(dtype=float)
    proxy = rows.proxy.to_numpy(dtype=float)
    residual = y - proxy
    cats = [c for c in features if isinstance(features[c].dtype, pd.CategoricalDtype)]
    train_set = lgb.Dataset(features.iloc[fit_idx], label=residual[fit_idx],
                            categorical_feature=cats, free_raw_data=True)
    early_set = lgb.Dataset(features.iloc[early_idx], label=residual[early_idx],
                            categorical_feature=cats, reference=train_set,
                            free_raw_data=True)
    start = time.monotonic()
    progress_path = args.output_dir / f"{name}_progress.json"

    def record_progress(env: lgb.callback.CallbackEnv) -> None:
        if (env.iteration + 1) % 100 != 0:
            return
        metric = env.evaluation_result_list[0][2] if env.evaluation_result_list else None
        progress_path.write_text(json.dumps({
            "fold": name, "iteration": env.iteration + 1,
            "early_rmse": metric,
            "elapsed_sec": round(time.monotonic() - start, 1)}), encoding="utf-8")

    record_progress.order = 11
    record_progress.before_iteration = False
    model = lgb.train(_model_params(), train_set, num_boost_round=800,
                      valid_sets=[early_set],
                      callbacks=[lgb.early_stopping(80, verbose=False), record_progress])
    fit_sec = time.monotonic() - start
    expert = proxy[test_idx] + model.predict(features.iloc[test_idx], num_threads=2)
    if not np.isfinite(expert).all():
        raise ValueError("Nonfinite neighbour held-out expert prediction")
    held = rows.iloc[test_idx][["MVT_ID_mvt", "target"]].copy()
    held["neighbour_expert"] = expert
    held = held.merge(reference[["MVT_ID_mvt", "target", "selected", "fold",
                                 "a_valid", "MVT_TIME_UTC_mvt"]],
                      on="MVT_ID_mvt", how="left", validate="one_to_one",
                      suffixes=("", "_reference"))
    if (len(held) != len(test_idx) or held.selected.isna().any()
            or not held.fold.eq(name).all() or not held.a_valid.all()
            or not np.allclose(held.target, held.target_reference)):
        raise ValueError("Neighbour heldout and frozen v5 reference misaligned")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(str(args.output_dir / f"{name}.txt"))
    held.to_parquet(args.output_dir / f"{name}_oof.parquet", index=False)
    fold = reference.loc[reference.fold.eq(name),
                          ["MVT_ID_mvt", "target", "selected", "a_valid",
                           "MVT_TIME_UTC_mvt"]]
    full = fold.merge(held[["MVT_ID_mvt", "neighbour_expert"]], on="MVT_ID_mvt",
                      how="left", validate="one_to_one")
    truth = full.target.to_numpy(dtype=float)
    base = full.selected.to_numpy(dtype=float)
    candidate = full.neighbour_expert.to_numpy(dtype=float)
    has = np.isfinite(candidate)
    if not np.array_equal(has, full.a_valid.to_numpy(dtype=bool)):
        raise ValueError("Held-out expert coverage does not equal frozen v5 valid-AOBT mask")
    scores = {}
    for weight in WEIGHTS:
        prediction = base.copy()
        prediction[has] = np.maximum(base[has] + weight * (candidate[has] - base[has]), 0)
        scores[str(weight)] = _rmse(truth, prediction)
    report = {"fold": name, "heldout_months": months,
              "train_fit_n": len(fit_idx), "internal_early_n": len(early_idx),
              "heldout_eligible_n": len(test_idx), "heldout_all_finite_n": len(full),
              "best_iteration": model.best_iteration,
              "fit_sec": round(fit_sec, 1), "scores_all_finite_clipped": scores}
    (args.output_dir / f"{name}_validation.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    del train_set, early_set, model, held
    gc.collect()
    return full, report


def _day_bootstrap_gain(full: pd.DataFrame, weight: float,
                        repetitions: int = 1000) -> dict:
    base = full.selected.to_numpy(dtype=float)
    candidate = full.neighbour_expert.to_numpy(dtype=float)
    has = np.isfinite(candidate)
    pred = base.copy()
    pred[has] = np.maximum(base[has] + weight * (candidate[has] - base[has]), 0)
    y = full.target.to_numpy(dtype=float)
    month = pd.to_datetime(full.MVT_TIME_UTC_mvt, utc=True).dt.month
    day = pd.to_datetime(full.MVT_TIME_UTC_mvt, utc=True).dt.floor("D")
    daily = pd.DataFrame({"month": month, "day": day,
                          "n": np.ones(len(full), dtype=np.int64),
                          "base_sse": (y - base) ** 2,
                          "candidate_sse": (y - pred) ** 2})
    daily = daily.groupby(["month", "day"], sort=True, observed=True)[
        ["n", "base_sse", "candidate_sse"]].sum().reset_index()
    groups = [part[["n", "base_sse", "candidate_sse"]].to_numpy(dtype=float)
              for _, part in daily.groupby("month", sort=True)]
    rng = np.random.default_rng(SEED)
    gains = np.empty(repetitions, dtype=float)
    for i in range(repetitions):
        totals = np.zeros(3, dtype=float)
        for data in groups:
            sampled = data[rng.integers(0, len(data), size=len(data))]
            totals += sampled.sum(axis=0)
        gains[i] = np.sqrt(totals[1] / totals[0]) - np.sqrt(totals[2] / totals[0])
    return {"point_gain_sec": _rmse(y, base) - _rmse(y, pred),
            "ci95_lower_sec": float(np.quantile(gains, .025)),
            "ci95_upper_sec": float(np.quantile(gains, .975)),
            "repetitions": repetitions, "utc_days": int(len(daily))}


def validate(args: argparse.Namespace) -> dict:
    saved_protocol = json.loads((args.output_dir / "protocol.json").read_text(encoding="utf-8"))
    if saved_protocol != protocol():
        raise ValueError("Protocol changed after feature build")
    rows, features = _load_all(args, False)
    reference = pd.read_parquet(args.v5_oof,
                                columns=["MVT_ID_mvt", "target", "selected", "fold",
                                         "a_valid", "MVT_TIME_UTC_mvt"])
    if reference.MVT_ID_mvt.isna().any() or reference.MVT_ID_mvt.duplicated().any():
        raise ValueError("v5 OOF IDs must be unique and non-null")
    seasonal_full, seasonal = _fold_fit("seasonal_jan_jul", FOLDS["seasonal_jan_jul"],
                                        rows, features, reference, args)
    chosen = min(WEIGHTS, key=lambda w: (seasonal["scores_all_finite_clipped"][str(w)], w))
    seasonal["day_bootstrap"] = _day_bootstrap_gain(seasonal_full, chosen)
    print(json.dumps({"seasonal": seasonal, "chosen_weight": chosen}, indent=2), flush=True)
    del seasonal_full
    gc.collect()
    forward_full, forward = _fold_fit("forward_nov_dec", FOLDS["forward_nov_dec"],
                                      rows, features, reference, args)
    forward["day_bootstrap"] = _day_bootstrap_gain(forward_full, chosen)
    print(json.dumps({"forward": forward, "fixed_weight": chosen}, indent=2), flush=True)
    del forward_full
    gc.collect()
    passes = (chosen > 0
              and seasonal["day_bootstrap"]["ci95_lower_sec"] > 0
              and forward["day_bootstrap"]["ci95_lower_sec"] > 0
              and seasonal["day_bootstrap"]["point_gain_sec"] > 0
              and forward["day_bootstrap"]["point_gain_sec"] > 0)
    report = {"chosen_weight": chosen, "promote": passes,
              "seasonal_jan_jul": seasonal, "forward_nov_dec": forward,
              "protocol": saved_protocol}
    (args.output_dir / "validation.json").write_text(json.dumps(report, indent=2),
                                                     encoding="utf-8")
    return report


def ablation_seasonal(args: argparse.Namespace) -> dict:
    """One matched seasonal fit without neighbour columns; no blend selection."""
    output_dir = args.output_dir / "ablation"
    output_dir.mkdir(parents=True, exist_ok=True)
    ablation_protocol = {
        "question": "Do neighbour AOBT covariates improve this fixed 31-leaf direct residual model?",
        "comparison": "No-neighbour versus saved with-neighbour seasonal experts on identical valid-AOBT OOF rows",
        "train_months_excluded": [1, 7], "max_rounds": 800,
        "early_stop_rounds": 80, "seed": SEED,
        "training_split": "Identical np.random.default_rng(seed) permutation and first max(30000,6%) internal early set",
        "score": "Both full expert taxi-time predictions clipped >=0, on all finite valid-proxy seasonal labels",
        "bootstrap": "1000 paired UTC-day resamples within Jan and Jul; lower 95% gain >0 is positive evidence",
        "ranking_fit": False,
    }
    (output_dir / "protocol.json").write_text(json.dumps(ablation_protocol, indent=2),
                                              encoding="utf-8")
    rows, features = load_cache(args.cache_dir, False)
    features = add_flight_weather(rows, features, args.data_dir,
                                  args.weather_file, False)
    train_mask, test_mask = _masks(rows, FOLDS["seasonal_jan_jul"])
    train_idx = np.flatnonzero(train_mask)
    test_idx = np.flatnonzero(test_mask)
    rng = np.random.default_rng(SEED)
    shuffled = rng.permutation(train_idx)
    early_n = max(30000, int(.06 * len(shuffled)))
    early_idx, fit_idx = shuffled[:early_n], shuffled[early_n:]
    existing = pd.read_parquet(args.output_dir / "seasonal_jan_jul_oof.parquet",
                               columns=["MVT_ID_mvt", "target", "neighbour_expert",
                                        "MVT_TIME_UTC_mvt"])
    if (len(existing) != len(test_idx) or
            not np.array_equal(existing.MVT_ID_mvt.to_numpy(),
                               rows.MVT_ID_mvt.to_numpy()[test_idx])):
        raise ValueError("Saved with-neighbour OOF rows are not the matched test set")
    y = rows.target.to_numpy(dtype=float)
    proxy = rows.proxy.to_numpy(dtype=float)
    residual = y - proxy
    cats = [c for c in features if isinstance(features[c].dtype, pd.CategoricalDtype)]
    train_set = lgb.Dataset(features.iloc[fit_idx], label=residual[fit_idx],
                            categorical_feature=cats, free_raw_data=True)
    early_set = lgb.Dataset(features.iloc[early_idx], label=residual[early_idx],
                            categorical_feature=cats, reference=train_set,
                            free_raw_data=True)
    start = time.monotonic()
    progress_path = output_dir / "baseline_progress.json"

    def record_progress(env: lgb.callback.CallbackEnv) -> None:
        if (env.iteration + 1) % 100 == 0:
            progress_path.write_text(json.dumps({
                "iteration": env.iteration + 1,
                "early_rmse": env.evaluation_result_list[0][2],
                "elapsed_sec": round(time.monotonic() - start, 1)}),
                encoding="utf-8")

    record_progress.order = 11
    record_progress.before_iteration = False
    model = lgb.train(_model_params(), train_set, num_boost_round=800,
                      valid_sets=[early_set],
                      callbacks=[lgb.early_stopping(80, verbose=False), record_progress])
    fit_sec = time.monotonic() - start
    baseline = proxy[test_idx] + model.predict(features.iloc[test_idx], num_threads=2)
    model.save_model(str(output_dir / "baseline_seasonal.txt"))
    y_test = existing.target.to_numpy(dtype=float)
    if not np.allclose(y_test, y[test_idx]):
        raise ValueError("Saved neighbour target and baseline target differ")
    existing["baseline_expert"] = baseline
    existing.to_parquet(output_dir / "matched_seasonal_oof.parquet", index=False)
    old_pred = np.maximum(baseline, 0)
    new_pred = np.maximum(existing.neighbour_expert.to_numpy(dtype=float), 0)
    paired = existing[["target", "MVT_TIME_UTC_mvt", "neighbour_expert"]].copy()
    paired["selected"] = old_pred
    interval = _day_bootstrap_gain(paired, 1.0)
    with_report = json.loads((args.output_dir / "seasonal_jan_jul_validation.json")
                             .read_text(encoding="utf-8"))
    report = {"n": len(existing), "baseline_rmse_sec": _rmse(y_test, old_pred),
              "neighbour_rmse_sec": _rmse(y_test, new_pred),
              "paired_day_gain": interval,
              "baseline_best_iteration": model.best_iteration,
              "neighbour_best_iteration": with_report["best_iteration"],
              "baseline_fit_sec": round(fit_sec, 1),
              "fit_rows": len(fit_idx), "internal_early_rows": len(early_idx),
              "protocol": ablation_protocol}
    (output_dir / "validation.json").write_text(json.dumps(report, indent=2),
                                                encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("build-features", "validate",
                                          "ablation-seasonal"), required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/baseline"))
    parser.add_argument("--weather-file", type=Path, default=Path("data/external/weather.parquet"))
    parser.add_argument("--v5-oof", type=Path,
                        default=Path("artifacts/v5-ensemble/validation_predictions.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/v6-neighbour"))
    args = parser.parse_args()
    action = {"build-features": build_features, "validate": validate,
              "ablation-seasonal": ablation_seasonal}[args.mode]
    print(json.dumps(action(args), indent=2), flush=True)


if __name__ == "__main__":
    main()
