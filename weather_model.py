"""Independent weather-augmented taxi-out residual model experiment.

Uses the feature/row caches written by solution.py, joins the CC0 NOAA GHCNh
hourly table, and predicts (taxi-out seconds - supplied off-block proxy seconds)
only where the supplied proxy is finite. No target or block-time values are
used as predictors.

Examples::

    python weather_model.py train --threads 2
    python weather_model.py complete-oof --threads 2
    python weather_model.py predict-only --threads 2
    python weather_model.py compare
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


DEFAULT_BASELINE = Path("artifacts/baseline")
DEFAULT_WEATHER = Path("data/external/weather.parquet")
DEFAULT_OUTPUT = Path("artifacts/weather")
FORBIDDEN_FEATURES = ("BLOCK_TIME", "TAXITIME", "target", "MVT_ID")
FOLDS = {"seasonal_jan_jul": (1, 7), "forward_nov_dec": (11, 12)}


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    valid = np.isfinite(actual) & np.isfinite(predicted)
    if not valid.any():
        return math.nan
    return float(np.sqrt(np.mean((actual[valid] - predicted[valid]) ** 2)))


def params(threads: int) -> dict:
    return {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.035,
        "num_leaves": 127,
        "min_data_in_leaf": 40,
        "lambda_l2": 15.0,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "max_cat_threshold": 64,
        "cat_smooth": 20,
        "verbosity": -1,
        "num_threads": threads,
        "seed": 20261002,
        "feature_fraction_seed": 20261002,
        "bagging_seed": 20261002,
        "deterministic": True,
        "force_col_wise": True,
    }


def load_cached_features(baseline_dir: Path, ranking: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    stem = "ranking" if ranking else "training"
    feature_file = baseline_dir / ("ranking_features.parquet" if ranking else "features.parquet")
    row_file = baseline_dir / f"{stem}_rows.parquet"
    if not feature_file.exists() or not row_file.exists():
        raise FileNotFoundError(f"Required baseline caches missing: {feature_file} and/or {row_file}")
    features = pd.read_parquet(feature_file)
    rows = pd.read_parquet(row_file)
    if len(features) != len(rows):
        raise ValueError("Baseline feature and row caches have different row counts")
    if rows["MVT_ID_mvt"].isna().any() or rows["MVT_ID_mvt"].duplicated().any():
        raise ValueError("Baseline row IDs must be unique and non-null")
    if not features["ADEP_mvt"].astype("string").fillna("__MISSING__").eq(rows["airport"].astype("string")).all():
        raise ValueError("Baseline features and rows are not aligned by airport")
    return features, rows


def add_weather(features: pd.DataFrame, rows: pd.DataFrame, weather_file: Path) -> pd.DataFrame:
    weather = pd.read_parquet(weather_file)
    if weather.duplicated(["airport", "weather_hour_utc"]).any():
        raise ValueError("Weather table has duplicate airport-hour keys")
    weather["weather_hour_utc"] = pd.to_datetime(weather["weather_hour_utc"], utc=True)
    lookup = weather.set_index(["airport", "weather_hour_utc"])
    hours = pd.to_datetime(rows["time"], utc=True).dt.floor("h")
    keys = pd.MultiIndex.from_arrays([rows["airport"].astype("string"), hours], names=lookup.index.names)
    joined = lookup.reindex(keys).reset_index(drop=True)
    joined = joined.drop(columns=["wx_station_id"])
    for column in joined.columns:
        if pd.api.types.is_bool_dtype(joined[column]):
            joined[column] = joined[column].fillna(False).astype("int8")
        else:
            joined[column] = pd.to_numeric(joined[column], errors="coerce").astype("float32")

    # NOAA wind direction describes where wind blows *from*. Runway number
    # approximates the takeoff heading in tens of degrees.
    runway = features["RUNWAY_mvt"].astype("string").str.extract(r"^(\d{2})", expand=False)
    heading_number = pd.to_numeric(runway, errors="coerce")
    heading = (heading_number.where(heading_number.between(1, 36)) * 10).astype("float32")
    angle = np.deg2rad(joined["wx_wind_direction_deg"].to_numpy() - heading.to_numpy())
    speed = joined["wx_wind_speed_mps"].to_numpy()
    joined["wx_headwind_mps"] = (speed * np.cos(angle)).astype("float32")
    joined["wx_crosswind_abs_mps"] = (speed * np.abs(np.sin(angle))).astype("float32")

    x = pd.concat([features.reset_index(drop=True), joined], axis=1)
    forbidden = [column for column in x if any(token.lower() in column.lower() for token in FORBIDDEN_FEATURES)]
    if forbidden:
        raise ValueError(f"Forbidden predictor columns: {forbidden}")
    if len(x) != len(rows):
        raise ValueError("Weather join changed the row count")
    return x


def fit(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_valid: pd.DataFrame | None,
    y_valid: np.ndarray | None,
    threads: int,
    rounds: int,
) -> lgb.Booster:
    categorical = [column for column in x_train if isinstance(x_train[column].dtype, pd.CategoricalDtype)]
    train_set = lgb.Dataset(x_train, label=y_train, categorical_feature=categorical, free_raw_data=True)
    if x_valid is None:
        return lgb.train(params(threads), train_set, num_boost_round=rounds,
                         callbacks=[lgb.log_evaluation(period=200)])
    valid_set = lgb.Dataset(x_valid, label=y_valid, reference=train_set, free_raw_data=True)
    return lgb.train(params(threads), train_set, num_boost_round=rounds,
                     valid_sets=[valid_set],
                     callbacks=[lgb.early_stopping(100, verbose=True), lgb.log_evaluation(period=100)])


def attach_baseline(oof: pd.DataFrame, baseline_dir: Path, fold: str) -> pd.DataFrame:
    baseline_path = baseline_dir / f"{fold}_oof.parquet"
    if not baseline_path.exists():
        oof["baseline_hybrid"] = np.nan
        return oof
    baseline = pd.read_parquet(baseline_path, columns=["MVT_ID_mvt", "row_index", "hybrid"])
    baseline = baseline.rename(columns={"hybrid": "baseline_hybrid"})
    matched = oof.merge(baseline, on=["MVT_ID_mvt", "row_index"], how="left", sort=False, validate="one_to_one")
    if len(matched) != len(oof) or matched["baseline_hybrid"].isna().any():
        raise ValueError(f"Baseline {fold} OOF does not align by row index and movement ID")
    return matched


def fold_report(oof: pd.DataFrame) -> dict:
    y = oof["target"].to_numpy(dtype=np.float64)
    weather = oof["weather_prediction"].to_numpy(dtype=np.float64)
    proxy = oof["proxy"].to_numpy(dtype=np.float64)
    result = {
        "n": int(len(oof)),
        "raw_proxy_rmse": rmse(y, proxy),
        "weather_rmse": rmse(y, weather),
    }
    if "baseline_hybrid" in oof and oof["baseline_hybrid"].notna().all():
        baseline = oof["baseline_hybrid"].to_numpy(dtype=np.float64)
        result["baseline_hybrid_rmse"] = rmse(y, baseline)
        # Diagnostic blend only; do not select a competition submission from
        # a blend tuned on the same validation rows without further checks.
        result["equal_blend_rmse"] = rmse(y, 0.5 * weather + 0.5 * baseline)
    return result


def train(baseline_dir: Path, weather_file: Path, output_dir: Path, threads: int, rounds: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    features, rows = load_cached_features(baseline_dir, ranking=False)
    x = add_weather(features, rows, weather_file)
    del features
    gc.collect()
    y = pd.to_numeric(rows["target"], errors="coerce").to_numpy(dtype=np.float32)
    proxy = pd.to_numeric(rows["proxy"], errors="coerce").to_numpy(dtype=np.float32)
    month = pd.to_numeric(rows["month"], errors="coerce").to_numpy()
    eligible = np.isfinite(y) & (y >= 0) & (y <= 86_400) & np.isfinite(proxy)
    residual_target = y - proxy
    if eligible.sum() < 100_000:
        raise ValueError(f"Unexpectedly few finite target/proxy rows: {eligible.sum()}")
    print(f"Weather features: {len(x.columns)}; valid proxy train rows: {eligible.sum():,}", flush=True)

    reports: dict[str, dict] = {}
    best_rounds: list[int] = []
    for fold, months in FOLDS.items():
        holdout = np.isin(month, months)
        train_mask = eligible & ~holdout
        valid_mask = eligible & holdout
        if train_mask.sum() < 1000 or valid_mask.sum() < 1000:
            raise ValueError(f"Insufficient rows for {fold}")
        print(f"{fold}: {train_mask.sum():,} train, {valid_mask.sum():,} valid", flush=True)
        model = fit(x.loc[train_mask], residual_target[train_mask],
                    x.loc[valid_mask], residual_target[valid_mask], threads, rounds)
        best = int(model.best_iteration or rounds)
        best_rounds.append(best)
        model.save_model(str(output_dir / f"{fold}_residual.txt"))
        residual_prediction = model.predict(x.loc[valid_mask], num_threads=threads, num_iteration=best)
        row_idx = np.flatnonzero(valid_mask)
        oof = rows.loc[valid_mask, ["MVT_ID_mvt", "target", "proxy", "month", "airport", "time"]].copy()
        oof["row_index"] = row_idx
        oof["weather_residual"] = residual_prediction.astype(np.float32)
        oof["weather_prediction"] = (proxy[valid_mask] + residual_prediction).astype(np.float32)
        oof = attach_baseline(oof, baseline_dir, fold)
        oof.to_parquet(output_dir / f"{fold}_oof.parquet", index=False)
        reports[fold] = {**fold_report(oof), "best_round": best,
                         "training_rows": int(train_mask.sum())}
        print(json.dumps({"fold": fold, **reports[fold]}, indent=2), flush=True)
        del model, oof
        gc.collect()

    final_rounds = int(np.median(best_rounds))
    print(f"Fitting final weather residual model for {final_rounds} rounds", flush=True)
    final_model = fit(x.loc[eligible], residual_target[eligible], None, None, threads, final_rounds)
    final_model.save_model(str(output_dir / "residual.txt"))
    metadata = {
        "features": list(x.columns),
        "weather_source": "NOAA NCEI GHCNh CC0-1.0",
        "weather_table": str(weather_file),
        "folds": reports,
        "final_rounds": final_rounds,
        "training_rows": int(eligible.sum()),
    }
    (output_dir / "model.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Saved final model to {output_dir / 'residual.txt'}", flush=True)


def compare(baseline_dir: Path, output_dir: Path) -> None:
    report_path = output_dir / "model.json"
    metadata = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    reports = {}
    for fold in FOLDS:
        oof_path = output_dir / f"{fold}_oof.parquet"
        if not oof_path.exists():
            continue
        oof = pd.read_parquet(oof_path)
        oof = oof.drop(columns="baseline_hybrid", errors="ignore")
        oof = attach_baseline(oof, baseline_dir, fold)
        oof.to_parquet(oof_path, index=False)
        reports[fold] = fold_report(oof)
        print(json.dumps({"fold": fold, **reports[fold]}, indent=2))
    if metadata:
        for fold, result in reports.items():
            metadata["folds"].setdefault(fold, {}).update(result)
        metadata["oof_coverage"] = "all finite target and proxy rows in each holdout month"
        report_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def complete_oof(baseline_dir: Path, weather_file: Path, output_dir: Path, threads: int) -> None:
    """Predict all finite-label holdout rows, including excluded training tails."""
    features, rows = load_cached_features(baseline_dir, ranking=False)
    x = add_weather(features, rows, weather_file)
    del features
    gc.collect()
    y = pd.to_numeric(rows["target"], errors="coerce").to_numpy(dtype=np.float64)
    proxy = pd.to_numeric(rows["proxy"], errors="coerce").to_numpy(dtype=np.float64)
    month = pd.to_numeric(rows["month"], errors="coerce").to_numpy()
    metadata_path = output_dir / "model.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    if metadata and list(x.columns) != metadata["features"]:
        raise ValueError("Weather feature schema differs from trained model")
    reports = {}
    for fold, months in FOLDS.items():
        mask = np.isfinite(y) & np.isfinite(proxy) & np.isin(month, months)
        model_file = output_dir / f"{fold}_residual.txt"
        model = lgb.Booster(model_file=str(model_file))
        residual_prediction = model.predict(x.loc[mask], num_threads=threads)
        oof = rows.loc[mask, ["MVT_ID_mvt", "target", "proxy", "month", "airport", "time"]].copy()
        oof["row_index"] = np.flatnonzero(mask)
        oof["weather_residual"] = residual_prediction.astype(np.float32)
        oof["weather_prediction"] = (proxy[mask] + residual_prediction).astype(np.float32)
        oof = attach_baseline(oof, baseline_dir, fold)
        if oof["MVT_ID_mvt"].duplicated().any():
            raise ValueError(f"Duplicate IDs in {fold} weather OOF")
        oof.to_parquet(output_dir / f"{fold}_oof.parquet", index=False)
        reports[fold] = fold_report(oof)
        print(json.dumps({"fold": fold, **reports[fold]}, indent=2), flush=True)
    if metadata:
        for fold, result in reports.items():
            metadata["folds"].setdefault(fold, {}).update(result)
        metadata["oof_coverage"] = "all finite target and proxy rows in each holdout month"
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def predict_only(baseline_dir: Path, weather_file: Path, output_dir: Path, threads: int) -> None:
    metadata = json.loads((output_dir / "model.json").read_text(encoding="utf-8"))
    features, rows = load_cached_features(baseline_dir, ranking=True)
    x = add_weather(features, rows, weather_file)
    if list(x.columns) != metadata["features"]:
        raise ValueError("Ranking weather features differ from model training schema")
    proxy = pd.to_numeric(rows["proxy"], errors="coerce").to_numpy(dtype=np.float64)
    valid = np.isfinite(proxy)
    model = lgb.Booster(model_file=str(output_dir / "residual.txt"))
    residual = np.full(len(rows), np.nan, dtype=np.float64)
    residual[valid] = model.predict(x.loc[valid], num_threads=threads)
    pred = proxy + residual
    ranking = rows[["MVT_ID_mvt", "proxy", "airport", "month", "time"]].copy()
    ranking["weather_residual"] = residual
    ranking["weather_prediction"] = np.maximum(pred, 0)
    if ranking["MVT_ID_mvt"].duplicated().any() or not np.isfinite(pred[valid]).all():
        raise ValueError("Ranking IDs duplicate or valid-proxy predictions are non-finite")
    output_dir.mkdir(parents=True, exist_ok=True)
    ranking.to_parquet(output_dir / "ranking_predictions.parquet", index=False)
    print(json.dumps({"rows": len(rows), "valid_proxy_rows": int(valid.sum()),
                      "prediction_file": str(output_dir / "ranking_predictions.parquet")}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "complete-oof", "compare", "predict-only"))
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--weather-file", type=Path, default=DEFAULT_WEATHER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=1500)
    args = parser.parse_args()
    if args.threads < 1 or args.rounds < 1:
        parser.error("--threads and --rounds must be positive")
    if args.command == "train":
        train(args.baseline_dir, args.weather_file, args.output_dir, args.threads, args.rounds)
    elif args.command == "complete-oof":
        complete_oof(args.baseline_dir, args.weather_file, args.output_dir, args.threads)
    elif args.command == "compare":
        compare(args.baseline_dir, args.output_dir)
    else:
        predict_only(args.baseline_dir, args.weather_file, args.output_dir, args.threads)


if __name__ == "__main__":
    main()
