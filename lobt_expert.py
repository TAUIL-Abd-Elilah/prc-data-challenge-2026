"""Independent LOBT residual expert for released PRC flight covariates.

The model predicts taxi-out minus (takeoff minus NM LOBT). Labels are read only
from the public training months; ranking predictions use public covariates.
Month-held-out predictions are saved for every finite, plausible LOBT proxy.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl

from solution import _training_files
from weather_model import add_weather, load_cached_features


FOLDS = {"seasonal_jan_jul": (1, 7), "forward_nov_dec": (11, 12)}
MAX_LOBT_PROXY = 172_800.0


def rmse(y: np.ndarray, p: np.ndarray) -> float:
    valid = np.isfinite(y) & np.isfinite(p)
    return float(np.sqrt(np.mean((y[valid] - p[valid]) ** 2))) if valid.any() else math.nan


def source_columns(paths: list[Path]) -> pd.DataFrame:
    """Read only published predictors, retaining baseline departure row order."""
    raw = (pl.scan_parquet([str(p) for p in paths])
           .filter(pl.col("PHASE_mvt") == "DEP")
           .select("MVT_ID_mvt", "FLIGHT_ID_mvt", "FLIGHT_mvt", "ADEP_mvt",
                   "MVT_TIME_UTC_mvt", "LOBT_flt")
           .collect().to_pandas())
    return raw


def load_features(data_dir: Path, cache_dir: Path, weather_file: Path,
                  ranking: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    features, rows = load_cached_features(cache_dir, ranking)
    paths = [data_dir / "ranking.parquet"] if ranking else _training_files(data_dir)
    if not ranking and len(paths) != 12:
        raise ValueError(f"Expected 12 canonical monthly training files, found {len(paths)}")
    raw = source_columns(paths)
    if len(raw) != len(rows) or not np.array_equal(raw["MVT_ID_mvt"].to_numpy(),
                                                   rows["MVT_ID_mvt"].to_numpy()):
        raise ValueError("Cached features and raw LOBT rows do not align by MVT_ID_mvt")
    takeoff = pd.to_datetime(raw["MVT_TIME_UTC_mvt"], utc=True, errors="coerce")
    lobt = pd.to_datetime(raw["LOBT_flt"], utc=True, errors="coerce")
    rows["lobt_proxy_sec"] = (takeoff - lobt).dt.total_seconds().to_numpy(dtype=np.float32)
    flight = raw["FLIGHT_mvt"].astype("string").fillna("?")
    features["flight_name"] = flight.astype("category")
    features["airport_flight"] = (raw["ADEP_mvt"].astype("string").fillna("?") + "_" + flight).astype("category")
    features["flight_id_missing"] = raw["FLIGHT_ID_mvt"].isna().to_numpy(dtype=np.int8)
    features["lobt_proxy_unclipped"] = rows["lobt_proxy_sec"].to_numpy(dtype=np.float32)
    del raw, flight
    gc.collect()
    x = add_weather(features, rows, weather_file)
    del features
    gc.collect()
    forbidden = [c for c in x if "TAXITIME" in c.upper() or "BLOCK_TIME" in c.upper()]
    if forbidden:
        raise ValueError(f"Target-derived predictor columns: {forbidden}")
    return rows, x


def params(threads: int) -> dict:
    return dict(objective="regression", metric="rmse", learning_rate=0.045,
                num_leaves=63, min_data_in_leaf=40, lambda_l2=15.0,
                feature_fraction=0.85, bagging_fraction=0.85, bagging_freq=1,
                max_cat_threshold=64, cat_smooth=20, verbosity=-1,
                num_threads=threads, seed=20261002, feature_fraction_seed=20261002,
                bagging_seed=20261002, deterministic=True, force_col_wise=True)


def fit(x_train: pd.DataFrame, y_train: np.ndarray,
        x_valid: pd.DataFrame | None, y_valid: np.ndarray | None,
        threads: int, rounds: int) -> lgb.Booster:
    categories = [c for c in x_train if isinstance(x_train[c].dtype, pd.CategoricalDtype)]
    train = lgb.Dataset(x_train, y_train, categorical_feature=categories,
                        free_raw_data=True)
    if x_valid is None:
        return lgb.train(params(threads), train, num_boost_round=rounds,
                         callbacks=[lgb.log_evaluation(period=200)])
    valid = lgb.Dataset(x_valid, y_valid, reference=train,
                        categorical_feature=categories, free_raw_data=True)
    return lgb.train(params(threads), train, num_boost_round=rounds,
                     valid_sets=[valid], callbacks=[
                         lgb.early_stopping(100, verbose=True),
                         lgb.log_evaluation(period=100)])


def reference_predictions(base_dir: Path, ensemble_dir: Path,
                          fold: str) -> pd.DataFrame:
    baseline = pd.read_parquet(base_dir / f"{fold}_oof.parquet",
                               columns=["MVT_ID_mvt", "hybrid", "direct"])
    if baseline["MVT_ID_mvt"].duplicated().any():
        raise ValueError("Duplicate IDs in baseline OOF")
    ensemble_path = ensemble_dir / "validation_predictions.parquet"
    if ensemble_path.exists():
        ens = pd.read_parquet(ensemble_path,
                              columns=["MVT_ID_mvt", "fold", "nested_ensemble"])
        ens = ens.loc[ens["fold"].eq(fold), ["MVT_ID_mvt", "nested_ensemble"]]
        if ens["MVT_ID_mvt"].duplicated().any():
            raise ValueError("Duplicate IDs in ensemble OOF")
        baseline = baseline.merge(ens, on="MVT_ID_mvt", how="left",
                                  sort=False, validate="one_to_one")
    return baseline.rename(columns={"hybrid": "baseline_prediction",
                                    "direct": "direct_prediction"})


def group_metrics(oof: pd.DataFrame) -> dict:
    y = oof["target"].to_numpy(dtype=np.float64)
    lobt = oof["lobt_prediction"].to_numpy(dtype=np.float64)
    aobt_valid = oof["aobt_valid"].to_numpy(dtype=bool)
    gap = oof["aobt_lobt_abs_gap"].to_numpy(dtype=np.float64)
    categories = {
        "all": np.ones(len(oof), dtype=bool),
        "aobt_valid": aobt_valid,
        "aobt_invalid": ~aobt_valid,
        "aobt_valid_gap_gt_3600": aobt_valid & (gap > 3600),
        "aobt_valid_gap_gt_1800": aobt_valid & (gap > 1800),
        "aobt_invalid_target_gt_7200": ~aobt_valid & (y > 7200),
        "aobt_valid_target_gt_7200": aobt_valid & (y > 7200),
    }
    report = {}
    for name, mask in categories.items():
        if not mask.any():
            report[name] = {"n": 0}
            continue
        section = {"n": int(mask.sum()), "lobt_prediction": rmse(y[mask], lobt[mask]),
                   "raw_lobt": rmse(y[mask], oof.loc[mask, "lobt_proxy_sec"].to_numpy(dtype=np.float64))}
        for col in ("baseline_prediction", "direct_prediction", "nested_ensemble"):
            if col in oof:
                section[col] = rmse(y[mask], oof.loc[mask, col].to_numpy(dtype=np.float64))
        if "nested_ensemble" in oof and np.isfinite(oof.loc[mask, "nested_ensemble"]).all():
            existing = oof.loc[mask, "nested_ensemble"].to_numpy(dtype=np.float64)
            section["blend_25pct_lobt"] = rmse(y[mask], 0.75 * existing + 0.25 * lobt[mask])
            section["blend_50pct_lobt"] = rmse(y[mask], 0.5 * existing + 0.5 * lobt[mask])
            difference = lobt[mask] - existing
            alpha = np.clip(np.dot(y[mask] - existing, difference) /
                            np.dot(difference, difference), 0, 1) if np.dot(difference, difference) > 0 else 0
            section["in_sample_optimal_blend_weight"] = float(alpha)
        report[name] = section
    return report


def policy_metrics(oof: pd.DataFrame) -> dict:
    """Evaluate a fixed, covariate-only switch against the existing ensemble."""
    y = oof["target"].to_numpy(dtype=np.float64)
    prior = oof["nested_ensemble"].to_numpy(dtype=np.float64)
    lobt = oof["lobt_prediction"].to_numpy(dtype=np.float64)
    valid_aobt = oof["aobt_valid"].to_numpy(dtype=bool)
    gap = oof["aobt_lobt_abs_gap"].to_numpy(dtype=np.float64)
    replace = ~valid_aobt
    blend = valid_aobt & (gap > 3600)
    candidate = np.where(replace, lobt,
                         np.where(blend, 0.5 * prior + 0.5 * lobt, prior))
    return {"n": int(len(oof)), "aobt_invalid_replacements": int(replace.sum()),
            "aobt_valid_large_gap_blends": int(blend.sum()),
            "prior_rmse": rmse(y, prior), "conditional_rmse": rmse(y, candidate),
            "rule": "Use LOBT prediction if AOBT invalid; otherwise blend 50% where absolute AOBT-LOBT gap exceeds 3600 sec."}


def report_saved_predictions(output_dir: Path) -> None:
    report_path = output_dir / "validation.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    parts = []
    for fold in FOLDS:
        oof = pd.read_parquet(output_dir / f"{fold}_oof.parquet")
        report["folds"][fold]["conditional_policy"] = policy_metrics(oof)
        parts.append(oof)
    report["pooled"]["conditional_policy"] = policy_metrics(pd.concat(parts, ignore_index=True))
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({fold: report["folds"][fold]["conditional_policy"]
                      for fold in FOLDS}, indent=2))


def train(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, x = load_features(args.data_dir, args.cache_dir, args.weather_file, False)
    y = pd.to_numeric(rows["target"], errors="coerce").to_numpy(dtype=np.float64)
    aobt = pd.to_numeric(rows["proxy"], errors="coerce").to_numpy(dtype=np.float64)
    lobt = rows["lobt_proxy_sec"].to_numpy(dtype=np.float64)
    month = rows["month"].to_numpy()
    eligible = np.isfinite(y) & np.isfinite(lobt) & (lobt >= 0) & (lobt <= MAX_LOBT_PROXY)
    if eligible.sum() < 100_000:
        raise ValueError(f"Unexpectedly few usable LOBT rows: {eligible.sum()}")
    print(f"LOBT training rows: {eligible.sum():,}; features: {len(x.columns)}", flush=True)
    residual = y - lobt
    reports: dict = {"training_rows": int(eligible.sum()), "folds": {},
                     "notes": "Labels only from 2025 training data; ranking uses released predictors."}
    best_rounds: list[int] = []
    pooled: list[pd.DataFrame] = []
    for fold, months in FOLDS.items():
        holdout = np.isin(month, months)
        training = eligible & ~holdout
        testing = eligible & holdout
        print(f"{fold}: train {training.sum():,}; holdout {testing.sum():,}", flush=True)
        model = fit(x.loc[training], residual[training], x.loc[testing], residual[testing],
                    args.threads, args.rounds)
        best = int(model.best_iteration or args.rounds)
        best_rounds.append(best)
        model.save_model(str(args.output_dir / f"{fold}_residual.txt"))
        pred = lobt[testing] + model.predict(x.loc[testing], num_threads=args.threads,
                                             num_iteration=best)
        oof = rows.loc[testing, ["MVT_ID_mvt", "target", "month", "airport", "time", "proxy", "lobt_proxy_sec"]].copy()
        oof["row_index"] = np.flatnonzero(testing)
        oof["fold"] = fold
        oof["aobt_valid"] = np.isfinite(aobt[testing])
        oof["aobt_lobt_abs_gap"] = np.where(np.isfinite(aobt[testing]),
                                            np.abs(lobt[testing] - aobt[testing]), np.nan)
        oof["lobt_prediction"] = pred.astype(np.float32)
        reference = reference_predictions(args.cache_dir, args.ensemble_dir, fold)
        oof = oof.merge(reference, on="MVT_ID_mvt", how="left", sort=False,
                        validate="one_to_one")
        if oof["baseline_prediction"].isna().any() or (
            "nested_ensemble" in oof and oof["nested_ensemble"].isna().any()
        ):
            raise ValueError("OOF reference predictions did not align")
        oof.to_parquet(args.output_dir / f"{fold}_oof.parquet", index=False)
        pooled.append(oof)
        reports["folds"][fold] = {"best_round": best, **group_metrics(oof)}
        print(json.dumps({"fold": fold, **reports["folds"][fold]}, indent=2), flush=True)
        del model, oof
        gc.collect()

    both = pd.concat(pooled, ignore_index=True)
    reports["pooled"] = group_metrics(both)
    # Gate the final fit on held-out improvement. The fixed 25% blend check is
    # inspectable and avoids selecting a weight on the same validation rows.
    segments = ("aobt_valid_gap_gt_3600", "aobt_valid_gap_gt_1800",
                "aobt_invalid_target_gt_7200")
    reports["final_fit_supported"] = bool(any(all(
        reports["folds"][fold][segment].get("n", 0) >= 5 and
        reports["folds"][fold][segment].get("blend_25pct_lobt", math.inf) <
        reports["folds"][fold][segment].get("nested_ensemble", math.inf)
        for fold in FOLDS
    ) for segment in segments))
    (args.output_dir / "validation.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    if not reports["final_fit_supported"]:
        print("No held-out segment supported a fixed blend; skipping ranking fit.", flush=True)
        return

    final_rounds = int(np.median(best_rounds))
    print(f"Fitting final LOBT residual model for {final_rounds} rounds", flush=True)
    model = fit(x.loc[eligible], residual[eligible], None, None,
                args.threads, final_rounds)
    model.save_model(str(args.output_dir / "residual.txt"))
    del model, rows, x, pooled, both
    gc.collect()
    ranking_rows, ranking_x = load_features(args.data_dir, args.cache_dir,
                                            args.weather_file, True)
    rank_lobt = ranking_rows["lobt_proxy_sec"].to_numpy(dtype=np.float64)
    rank_aobt = ranking_rows["proxy"].to_numpy(dtype=np.float64)
    valid = np.isfinite(rank_lobt) & (rank_lobt >= 0) & (rank_lobt <= MAX_LOBT_PROXY)
    model = lgb.Booster(model_file=str(args.output_dir / "residual.txt"))
    prediction = rank_lobt[valid] + model.predict(ranking_x.loc[valid],
                                                  num_threads=args.threads)
    out = ranking_rows.loc[valid, ["MVT_ID_mvt", "airport", "month", "proxy", "lobt_proxy_sec"]].copy()
    out["aobt_valid"] = np.isfinite(rank_aobt[valid])
    out["aobt_lobt_abs_gap"] = np.where(np.isfinite(rank_aobt[valid]),
                                        np.abs(rank_lobt[valid] - rank_aobt[valid]), np.nan)
    out["lobt_prediction"] = prediction.astype(np.float32)
    if out["MVT_ID_mvt"].duplicated().any() or not np.isfinite(prediction).all():
        raise ValueError("Ranking LOBT predictions have missing or duplicate IDs")
    out.to_parquet(args.output_dir / "ranking_predictions.parquet", index=False)
    reports["final_rounds"] = final_rounds
    reports["ranking_rows"] = int(len(out))
    (args.output_dir / "validation.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(json.dumps({"final_rounds": final_rounds, "ranking_rows": len(out)}, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/baseline"))
    parser.add_argument("--weather-file", type=Path, default=Path("data/external/weather.parquet"))
    parser.add_argument("--ensemble-dir", type=Path, default=Path("artifacts/ensemble"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/lobt"))
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=900)
    parser.add_argument("--report-only", action="store_true",
                        help="Summarize already saved OOF predictions without retraining")
    args = parser.parse_args()
    if args.threads < 1 or args.rounds < 1:
        parser.error("--threads and --rounds must be positive")
    if args.report_only:
        report_saved_predictions(args.output_dir)
    else:
        train(args)


if __name__ == "__main__":
    main()
