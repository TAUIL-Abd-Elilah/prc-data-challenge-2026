"""Independent categorical taxi-out residual expert with held-out 2025 months.

The model uses only released prediction-time features cached by solution.py.
No ranking labels, BLOCK time, or TAXITIME value is used as a predictor.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool


FOLDS = {"seasonal_jan_jul": (1, 7), "forward_nov_dec": (11, 12)}
BLEND_WEIGHTS = (0.0, .25, .5, 1.0)


def rmse(y: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - prediction) ** 2)))


def load_cache(cache_dir: Path, ranking: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    prefix = "ranking" if ranking else "training"
    rows = pd.read_parquet(cache_dir / f"{prefix}_rows.parquet")
    features = pd.read_parquet(cache_dir / ("ranking_features.parquet" if ranking else "features.parquet"))
    if len(rows) != len(features):
        raise ValueError("Cached rows and features differ in length")
    if any("TAXITIME" in col.upper() or "BLOCK_TIME" in col.upper() or
           col.lower() == "target" for col in features.columns):
        raise ValueError("A label-derived feature was found in the prediction cache")
    return rows, features


def add_flight_weather(rows: pd.DataFrame, features: pd.DataFrame,
                       data_dir: Path, weather_file: Path,
                       ranking: bool) -> pd.DataFrame:
    """Reuse the existing, ranking-safe weather and recurring-flight joins."""
    from airport_models import add_flight
    from solution import _training_files
    from weather_model import add_weather

    files = [data_dir / "ranking.parquet"] if ranking else _training_files(data_dir)
    features = add_weather(features, rows, weather_file)
    features = add_flight(features, rows, files)
    return features


def training_sample(rows: pd.DataFrame, eligible: np.ndarray, max_rows: int,
                    seed: int) -> np.ndarray:
    """Retain rare long valid-proxy cases, then sample the ordinary rows."""
    indices = np.flatnonzero(eligible)
    if len(indices) <= max_rows:
        return indices
    proxy = rows.proxy.to_numpy(dtype=float)
    y = rows.target.to_numpy(dtype=float)
    rare = indices[(proxy[indices] > 2400) | (y[indices] > 7200)]
    if len(rare) >= max_rows:
        rng = np.random.default_rng(seed)
        return np.sort(rng.choice(rare, size=max_rows, replace=False))
    ordinary = np.setdiff1d(indices, rare, assume_unique=True)
    rng = np.random.default_rng(seed)
    chosen = rng.choice(ordinary, size=max_rows - len(rare), replace=False)
    return np.sort(np.concatenate([rare, chosen]))


def model_params(iterations: int, threads: int, depth: int, seed: int) -> dict:
    return dict(loss_function="RMSE", eval_metric="RMSE", iterations=iterations,
                learning_rate=.055, depth=depth, l2_leaf_reg=10,
                random_strength=.5, bagging_temperature=.5,
                max_ctr_complexity=1, one_hot_max_size=20,
                border_count=128, thread_count=threads,
                used_ram_limit="8gb", random_seed=seed,
                allow_writing_files=False, verbose=100)


def fit_model(rows: pd.DataFrame, features: pd.DataFrame, train_mask: np.ndarray,
              max_rows: int, iterations: int, threads: int, depth: int,
              seed: int, early_stop: bool) -> tuple[CatBoostRegressor, dict]:
    selected = training_sample(rows, train_mask, max_rows, seed)
    rng = np.random.default_rng(seed + 10)
    order = rng.permutation(len(selected))
    if early_stop:
        n_early = max(5000, int(.06 * len(selected)))
        early_idx = selected[order[:n_early]]
        train_idx = selected[order[n_early:]]
    else:
        early_idx = np.array([], dtype=np.int64)
        train_idx = selected
    categories = features.select_dtypes(include="category").columns.tolist()
    target = (rows.target.to_numpy(dtype=float) - rows.proxy.to_numpy(dtype=float))
    train_pool = Pool(features.iloc[train_idx], label=target[train_idx],
                      cat_features=categories)
    kwargs = {}
    if len(early_idx):
        kwargs["eval_set"] = Pool(features.iloc[early_idx], label=target[early_idx],
                                  cat_features=categories)
        kwargs["early_stopping_rounds"] = 80
        kwargs["use_best_model"] = True
    model = CatBoostRegressor(**model_params(iterations, threads, depth, seed))
    start = time.monotonic()
    model.fit(train_pool, **kwargs)
    fit_seconds = time.monotonic() - start
    report = {"eligible_training_rows": int(train_mask.sum()),
              "sampled_rows": int(len(selected)),
              "fitted_rows": int(len(train_idx)),
              "internal_early_stop_rows": int(len(early_idx)),
              "best_iteration": int(model.get_best_iteration()) if early_stop else iterations,
              "tree_count": int(model.tree_count_),
              "fit_seconds": fit_seconds,
              "cat_features": categories,
              "features": features.columns.tolist()}
    del train_pool
    gc.collect()
    return model, report


def fit_fold(name: str, months: tuple[int, int], rows: pd.DataFrame,
             features: pd.DataFrame, v3: pd.DataFrame,
             args: argparse.Namespace) -> dict:
    y = rows.target.to_numpy(dtype=float)
    proxy = rows.proxy.to_numpy(dtype=float)
    valid_proxy = np.isfinite(proxy) & (proxy >= 0) & (proxy <= 7200)
    core = np.isfinite(y) & (y >= 0) & (y <= 86400) & valid_proxy
    heldout = rows.month.isin(months).to_numpy()
    train_mask = core & ~heldout
    test_mask = heldout & np.isfinite(y) & valid_proxy
    model, fit_report = fit_model(rows, features, train_mask, args.sample_max,
                                   args.iterations, args.threads, args.depth,
                                   args.seed, True)
    query_idx = np.flatnonzero(test_mask)
    start = time.monotonic()
    residual = model.predict(features.iloc[query_idx], thread_count=args.threads)
    predict_seconds = time.monotonic() - start
    prediction = proxy[query_idx] + residual
    output = rows.iloc[query_idx][["MVT_ID_mvt", "target", "proxy", "month", "airport"]].copy()
    output["catboost_prediction"] = prediction
    output = output.merge(v3[["MVT_ID_mvt", "selected"]].rename(
        columns={"selected": "v3_prediction"}),
        on="MVT_ID_mvt", how="left", validate="one_to_one")
    if output.v3_prediction.isna().any() or len(output) != len(query_idx):
        raise ValueError("Frozen v3 comparison does not cover CatBoost heldout rows")
    yy = output.target.to_numpy(dtype=float)
    v3p = output.v3_prediction.to_numpy(dtype=float)
    cp = output.catboost_prediction.to_numpy(dtype=float)
    scores = {str(weight): rmse(yy, (1 - weight) * v3p + weight * cp)
              for weight in BLEND_WEIGHTS}
    by_airport = {}
    for airport, group in output.groupby("airport"):
        by_airport[str(airport)] = {
            "n": int(len(group)),
            "v3_rmse_sec": rmse(group.target.to_numpy(dtype=float),
                                group.v3_prediction.to_numpy(dtype=float)),
            "catboost_rmse_sec": rmse(group.target.to_numpy(dtype=float),
                                      group.catboost_prediction.to_numpy(dtype=float)),
            "quarter_blend_rmse_sec": rmse(group.target.to_numpy(dtype=float),
                                           (.75 * group.v3_prediction +
                                            .25 * group.catboost_prediction).to_numpy(dtype=float)),
        }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(str(args.output_dir / f"{name}.cbm"))
    output.to_parquet(args.output_dir / f"{name}_oof.parquet", index=False)
    report = {"fold": name, "heldout_months": months,
              "heldout_n": int(len(output)), "fit": fit_report,
              "predict_seconds": predict_seconds,
              "scores": scores, "by_airport": by_airport}
    (args.output_dir / f"{name}_validation.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/baseline"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--weather-file", type=Path,
                        default=Path("data/external/weather.parquet"))
    parser.add_argument("--with-flight-weather", action="store_true")
    parser.add_argument("--v3-dir", type=Path, default=Path("artifacts/lobt_ensemble"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/catboost"))
    parser.add_argument("--fold", choices=(*FOLDS, "both"), default="seasonal_jan_jul")
    parser.add_argument("--sample-max", type=int, default=400000)
    parser.add_argument("--iterations", type=int, default=450)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--depth", type=int, default=7)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.threads > 6:
        raise ValueError("Use at most six CatBoost threads to share the workspace host")
    rows, features = load_cache(args.cache_dir, False)
    if args.with_flight_weather:
        features = add_flight_weather(rows, features, args.data_dir,
                                      args.weather_file, False)
        print(f"Added recurring flight and weather features: {len(features.columns)} total",
              flush=True)
    v3 = pd.read_parquet(args.v3_dir / "validation_predictions.parquet",
                         columns=["MVT_ID_mvt", "selected"])
    requested = FOLDS if args.fold == "both" else {args.fold: FOLDS[args.fold]}
    reports = {}
    for name, months in requested.items():
        print(f"Training {name} CatBoost expert", flush=True)
        reports[name] = fit_fold(name, months, rows, features, v3, args)
        print(json.dumps({"fold": name, "scores": reports[name]["scores"],
                          "fit_seconds": reports[name]["fit"]["fit_seconds"]}, indent=2),
              flush=True)
    (args.output_dir / "validation.json").write_text(
        json.dumps(reports, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
