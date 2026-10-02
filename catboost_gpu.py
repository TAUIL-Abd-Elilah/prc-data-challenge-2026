"""Independent GPU CatBoost residual experiment on local 2025 OOF folds.

The frozen v3 prediction changes only on valid-AOBT rows. Hyperparameter and
blend decisions use January/July only; November/December is a forward check.
No ranking labels or movement BLOCK timestamps are read.
"""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

from catboost_expert import add_flight_weather, load_cache, rmse


FOLDS = {"seasonal_jan_jul": (1, 7), "forward_nov_dec": (11, 12)}
BLEND_WEIGHTS = (0.0, .1, .25, .5, 1.0)
FINAL_WEIGHT = .25


def gpu_memory_mib() -> int | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=10)
        return int(result.stdout.strip().splitlines()[0])
    except Exception:
        return None


def model_params(iterations: int, threads: int, depth: int, seed: int) -> dict:
    return dict(
        task_type="GPU", devices="0", gpu_ram_part=.5,
        loss_function="RMSE", eval_metric="RMSE",
        iterations=iterations, learning_rate=.055, depth=depth,
        l2_leaf_reg=10, random_strength=.5, bagging_temperature=.5,
        max_ctr_complexity=1, one_hot_max_size=20, border_count=128,
        thread_count=threads, used_ram_limit="8gb", random_seed=seed,
        allow_writing_files=False, verbose=100,
    )


def load_inputs(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    start = time.monotonic()
    rows, features = load_cache(args.cache_dir, False)
    features = add_flight_weather(rows, features, args.data_dir,
                                  args.weather_file, False)
    elapsed = time.monotonic() - start
    v3 = pd.read_parquet(args.v3_dir / "validation_predictions.parquet",
                         columns=["MVT_ID_mvt", "target", "fold", "selected", "a_valid"])
    if v3.MVT_ID_mvt.duplicated().any() or len(rows) != len(features):
        raise ValueError("ID uniqueness or cache alignment failed")
    cats = features.select_dtypes(include="category").columns.tolist()
    info = {"load_seconds": round(elapsed, 3), "n_rows": len(rows),
            "n_features": len(features.columns), "feature_names": features.columns.tolist(),
            "categories": cats, "gpu_memory_after_load_mib": gpu_memory_mib()}
    return rows, features, v3, info


def eligible_masks(rows: pd.DataFrame, months: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    target = rows.target.to_numpy(dtype=float)
    proxy = rows.proxy.to_numpy(dtype=float)
    valid_proxy = np.isfinite(proxy) & (proxy >= 0) & (proxy <= 7200)
    heldout = rows.month.isin(months).to_numpy(dtype=bool)
    core = np.isfinite(target) & (target >= 0) & (target <= 86400) & valid_proxy
    test = heldout & np.isfinite(target) & valid_proxy
    return core & ~heldout, test


def benchmark(rows: pd.DataFrame, features: pd.DataFrame,
              args: argparse.Namespace, load_info: dict) -> dict:
    train_mask, _ = eligible_masks(rows, FOLDS["seasonal_jan_jul"])
    indices = np.flatnonzero(train_mask)
    rng = np.random.default_rng(args.seed)
    chosen = rng.choice(indices, size=min(args.benchmark_rows, len(indices)), replace=False)
    target = rows.target.to_numpy(dtype=float) - rows.proxy.to_numpy(dtype=float)
    cats = load_info["categories"]
    pool_start = time.monotonic()
    pool = Pool(features.iloc[chosen], label=target[chosen], cat_features=cats)
    pool_seconds = time.monotonic() - pool_start
    model = CatBoostRegressor(**model_params(args.benchmark_trees, args.threads,
                                            args.depth, args.seed))
    start = time.monotonic()
    model.fit(pool)
    fit_seconds = time.monotonic() - start
    report = {
        "mode": "benchmark", "rows": len(chosen), "trees": args.benchmark_trees,
        "depth": args.depth, "pool_seconds": round(pool_seconds, 3),
        "fit_seconds": round(fit_seconds, 3), "load": load_info,
        "gpu_memory_after_fit_mib": gpu_memory_mib(),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "benchmark.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def fit_fold(name: str, months: tuple[int, int], rows: pd.DataFrame,
             features: pd.DataFrame, v3: pd.DataFrame,
             args: argparse.Namespace, load_info: dict) -> tuple[pd.DataFrame, dict]:
    train_mask, test_mask = eligible_masks(rows, months)
    train_idx = np.flatnonzero(train_mask)
    query_idx = np.flatnonzero(test_mask)
    rng = np.random.default_rng(args.seed + (0 if name == "seasonal_jan_jul" else 7))
    order = rng.permutation(train_idx)
    n_early = max(20000, int(.06 * len(order)))
    early_idx, fit_idx = order[:n_early], order[n_early:]
    target = rows.target.to_numpy(dtype=float) - rows.proxy.to_numpy(dtype=float)
    cats = load_info["categories"]
    pool_start = time.monotonic()
    train_pool = Pool(features.iloc[fit_idx], label=target[fit_idx], cat_features=cats)
    early_pool = Pool(features.iloc[early_idx], label=target[early_idx], cat_features=cats)
    pool_seconds = time.monotonic() - pool_start
    print(f"{name}: GPU Pool ready, rows={len(fit_idx):,}, early={len(early_idx):,}, "
          f"pool_sec={pool_seconds:.1f}, gpu_mem_mib={gpu_memory_mib()}", flush=True)
    model = CatBoostRegressor(**model_params(args.iterations, args.threads,
                                            args.depth, args.seed))
    fit_start = time.monotonic()
    model.fit(train_pool, eval_set=early_pool, early_stopping_rounds=120,
              use_best_model=True)
    fit_seconds = time.monotonic() - fit_start
    gpu_after = gpu_memory_mib()
    del train_pool, early_pool
    gc.collect()
    pred_start = time.monotonic()
    residual = model.predict(features.iloc[query_idx], thread_count=args.threads)
    predict_seconds = time.monotonic() - pred_start
    expert_prediction = rows.proxy.to_numpy(dtype=float)[query_idx] + residual
    output = rows.iloc[query_idx][["MVT_ID_mvt", "target", "month", "airport", "proxy"]].copy()
    output["gpu_prediction"] = expert_prediction
    output = output.merge(v3[["MVT_ID_mvt", "target", "selected", "fold", "a_valid"]],
                          on="MVT_ID_mvt", how="left", validate="one_to_one",
                          suffixes=("", "_v3"))
    if (len(output) != len(query_idx) or output.selected.isna().any()
            or not output.fold.eq(name).all() or not output.a_valid.all()
            or not np.allclose(output.target, output.target_v3)):
        raise ValueError("Frozen v3 validation alignment failed")
    y = output.target.to_numpy(dtype=float)
    base = output.selected.to_numpy(dtype=float)
    expert = output.gpu_prediction.to_numpy(dtype=float)
    scores = {str(w): rmse(y, base + w * (expert - base)) for w in BLEND_WEIGHTS}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(str(args.output_dir / f"{name}.cbm"))
    output.to_parquet(args.output_dir / f"{name}_oof.parquet", index=False)
    report = {
        "fold": name, "heldout_months": months,
        "training_eligible_rows": int(train_mask.sum()),
        "fitted_rows": len(fit_idx), "internal_early_stop_rows": len(early_idx),
        "heldout_eligible_rows": len(query_idx),
        "pool_seconds": round(pool_seconds, 3),
        "fit_seconds": round(fit_seconds, 3),
        "predict_seconds": round(predict_seconds, 3),
        "best_iteration": int(model.get_best_iteration()),
        "tree_count": int(model.tree_count_),
        "gpu_memory_after_fit_mib": gpu_after,
        "scores_eligible": scores,
    }
    (args.output_dir / f"{name}_validation.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    return output, report


def all_row_score(v3: pd.DataFrame, expert: pd.DataFrame, weight: float,
                  fold: str) -> dict:
    base = v3.loc[v3.fold.eq(fold), ["MVT_ID_mvt", "target", "selected"]]
    merged = base.merge(expert[["MVT_ID_mvt", "gpu_prediction"]],
                        on="MVT_ID_mvt", how="left", validate="one_to_one")
    pred = merged.selected.to_numpy(dtype=float).copy()
    eligible = merged.gpu_prediction.notna().to_numpy()
    pred[eligible] += weight * (
        merged.gpu_prediction.to_numpy(dtype=float)[eligible] - pred[eligible])
    return {"n": len(merged), "eligible_n": int(eligible.sum()),
            "v3_rmse_sec": rmse(merged.target.to_numpy(dtype=float),
                                merged.selected.to_numpy(dtype=float)),
            "blend_rmse_sec": rmse(merged.target.to_numpy(dtype=float), pred)}


def final_predict(args: argparse.Namespace) -> dict:
    """Fit all eligible 2025 labels and prepare an unsubmitted ranking artifact."""
    if args.iterations != 1500 or args.depth != 8:
        raise ValueError("Final model uses the frozen 1500-tree, depth-8 configuration")
    load_start = time.monotonic()
    rows, features = load_cache(args.cache_dir, False)
    features = add_flight_weather(rows, features, args.data_dir,
                                  args.weather_file, False)
    train_load_seconds = time.monotonic() - load_start
    target = rows.target.to_numpy(dtype=float)
    proxy = rows.proxy.to_numpy(dtype=float)
    eligible = (np.isfinite(target) & (target >= 0) & (target <= 86400)
                & np.isfinite(proxy) & (proxy >= 0) & (proxy <= 7200))
    idx = np.flatnonzero(eligible)
    if len(idx) < 1_000_000:
        raise ValueError("Unexpectedly few eligible 2025 training rows")
    cats = features.select_dtypes(include="category").columns.tolist()
    model = CatBoostRegressor(**model_params(args.iterations, args.threads,
                                            args.depth, args.seed))
    pool_start = time.monotonic()
    pool = Pool(features.iloc[idx], label=(target - proxy)[idx], cat_features=cats)
    pool_seconds = time.monotonic() - pool_start
    print(f"Final GPU Pool ready: {len(idx):,} rows, pool_sec={pool_seconds:.1f}, "
          f"gpu_mem_mib={gpu_memory_mib()}", flush=True)
    fit_start = time.monotonic()
    model.fit(pool)
    fit_seconds = time.monotonic() - fit_start
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(str(args.output_dir / "full_2025.cbm"))
    del pool
    gc.collect()

    ranking_load_start = time.monotonic()
    rank_rows, rank_features = load_cache(args.cache_dir, True)
    rank_features = add_flight_weather(rank_rows, rank_features, args.data_dir,
                                       args.weather_file, True)
    ranking_load_seconds = time.monotonic() - ranking_load_start
    if list(rank_features.columns) != list(features.columns):
        raise ValueError("Training and ranking feature columns differ")
    if rank_rows.MVT_ID_mvt.isna().any() or rank_rows.MVT_ID_mvt.duplicated().any():
        raise ValueError("Ranking movement IDs are not unique and non-null")
    rank_proxy = rank_rows.proxy.to_numpy(dtype=float)
    rank_valid = np.isfinite(rank_proxy) & (rank_proxy >= 0) & (rank_proxy <= 7200)
    rank_idx = np.flatnonzero(rank_valid)
    predict_start = time.monotonic()
    residual = model.predict(rank_features.iloc[rank_idx], thread_count=args.threads)
    predict_seconds = time.monotonic() - predict_start
    rank_expert = np.full(len(rank_rows), np.nan, dtype=float)
    rank_expert[rank_idx] = rank_proxy[rank_idx] + residual
    expert = pd.DataFrame({"MVT_ID_mvt": rank_rows.MVT_ID_mvt,
                           "gpu_prediction": rank_expert})
    expert.to_parquet(args.output_dir / "ranking_expert.parquet", index=False)

    base = pd.read_parquet(args.v3_dir / "predictions.parquet",
                           columns=["MVT_ID_mvt", "TAXITIME_SEC_mvt"])
    if base.MVT_ID_mvt.isna().any() or base.MVT_ID_mvt.duplicated().any():
        raise ValueError("Frozen v3 ranking IDs are not unique and non-null")
    merged = base.merge(expert, on="MVT_ID_mvt", how="left", validate="one_to_one",
                        sort=False)
    if len(merged) != len(base):
        raise ValueError("Ranking merge changed row count")
    frozen = merged.TAXITIME_SEC_mvt.to_numpy(dtype=float)
    gpu = merged.gpu_prediction.to_numpy(dtype=float)
    has_gpu = np.isfinite(gpu)
    prediction = frozen.copy()
    prediction[has_gpu] += FINAL_WEIGHT * (gpu[has_gpu] - frozen[has_gpu])
    prediction = np.maximum(prediction, 0)
    if (not np.isfinite(prediction).all() or
            not np.array_equal(prediction[~has_gpu], frozen[~has_gpu])):
        raise ValueError("Final prediction invalid or changed an ineligible v3 row")
    output = pd.DataFrame({"MVT_ID_mvt": merged.MVT_ID_mvt,
                           "TAXITIME_SEC_mvt": prediction})
    template = pd.read_parquet(args.data_dir / "submitting.parquet",
                               columns=["MVT_ID_mvt"])
    if len(template) != len(output) or not np.array_equal(
            template.MVT_ID_mvt.to_numpy(), output.MVT_ID_mvt.to_numpy()):
        raise ValueError("Final artifact does not preserve submission template order")
    output.to_parquet(args.output_dir / "predictions.parquet", index=False)
    report = {
        "scope": "2025 released training labels; ranking covariates only; no ranking labels",
        "training_eligible_rows": int(eligible.sum()),
        "training_total_departures": len(rows),
        "ranking_total_departures": len(rank_rows),
        "ranking_gpu_eligible_rows": int(rank_valid.sum()),
        "ranking_v3_unchanged_rows": int((~has_gpu).sum()),
        "blend_weight_frozen_from_seasonal": FINAL_WEIGHT,
        "depth": args.depth, "iterations": args.iterations,
        "fit_seconds": round(fit_seconds, 3),
        "pool_seconds": round(pool_seconds, 3),
        "train_load_seconds": round(train_load_seconds, 3),
        "ranking_load_seconds": round(ranking_load_seconds, 3),
        "predict_seconds": round(predict_seconds, 3),
        "feature_count": len(features.columns),
        "category_count": len(cats),
        "model_path": str((args.output_dir / "full_2025.cbm").resolve()),
        "expert_path": str((args.output_dir / "ranking_expert.parquet").resolve()),
        "predictions_path": str((args.output_dir / "predictions.parquet").resolve()),
        "output_min": float(prediction.min()),
        "output_max": float(prediction.max()),
    }
    (args.output_dir / "final_report.json").write_text(json.dumps(report, indent=2),
                                                        encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/baseline"))
    parser.add_argument("--weather-file", type=Path, default=Path("data/external/weather.parquet"))
    parser.add_argument("--v3-dir", type=Path, default=Path("artifacts/lobt_ensemble"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/catboost/gpu"))
    parser.add_argument("--mode", choices=("benchmark", "fit", "final-predict"),
                        default="benchmark")
    parser.add_argument("--benchmark-rows", type=int, default=50000)
    parser.add_argument("--benchmark-trees", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=1500)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.iterations > 2000 or args.threads > 4:
        raise ValueError("Cap each model at 2000 trees and four host threads")
    if args.mode == "final-predict":
        print(json.dumps(final_predict(args), indent=2), flush=True)
        return
    rows, features, v3, load_info = load_inputs(args)
    if args.mode == "benchmark":
        print(json.dumps(benchmark(rows, features, args, load_info), indent=2), flush=True)
        return
    reports = {"load": load_info, "selection_fold": "seasonal_jan_jul",
               "blend_weights": BLEND_WEIGHTS}
    seasonal, seasonal_report = fit_fold("seasonal_jan_jul", FOLDS["seasonal_jan_jul"],
                                         rows, features, v3, args, load_info)
    selected_weight = min(BLEND_WEIGHTS,
                          key=lambda w: seasonal_report["scores_eligible"][str(w)])
    reports["seasonal_jan_jul"] = seasonal_report
    reports["selected_weight"] = selected_weight
    reports["seasonal_all_rows"] = all_row_score(v3, seasonal, selected_weight,
                                                  "seasonal_jan_jul")
    print(json.dumps({"seasonal": reports["seasonal_all_rows"],
                      "selected_weight": selected_weight,
                      "fit_seconds": seasonal_report["fit_seconds"]}, indent=2), flush=True)
    if selected_weight > 0:
        forward, forward_report = fit_fold("forward_nov_dec", FOLDS["forward_nov_dec"],
                                           rows, features, v3, args, load_info)
        reports["forward_nov_dec"] = forward_report
        reports["forward_all_rows"] = all_row_score(v3, forward, selected_weight,
                                                    "forward_nov_dec")
        print(json.dumps({"forward": reports["forward_all_rows"],
                          "fit_seconds": forward_report["fit_seconds"]}, indent=2), flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "validation.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
