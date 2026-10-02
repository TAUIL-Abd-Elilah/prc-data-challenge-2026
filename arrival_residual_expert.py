"""Leakage-free seasonal/forward ARR-ground direct residual expert.

Only ARR movement BLOCK/TAXITIME fields are read as traffic covariates.
Departure targets enter training labels from the existing 2025 cache only.
No held-out departure labels enter a fold's fitted model or early stop set.
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

from arrival_features import build_arrival_traffic
from catboost_expert import add_flight_weather, load_cache
from solution import _training_files


FOLDS = {"seasonal_jan_jul": (1, 7), "forward_nov_dec": (11, 12)}
WEIGHTS = (0.0, 0.1, 0.25, 0.5, 1.0)
DEP_COLS = ["MVT_ID_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "STAND_mvt"]
ARR_COLS = ["MVT_ID_mvt", "ADES_mvt", "MVT_TIME_UTC_mvt",
            "BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt", "STAND_mvt"]


def _rmse(y: np.ndarray, pred: np.ndarray) -> float:
    if not np.isfinite(y).all() or not np.isfinite(pred).all():
        raise ValueError("Nonfinite score input")
    return float(np.sqrt(np.mean((y - pred) ** 2)))


def _params(threads: int) -> dict:
    return dict(objective="regression", metric="rmse", learning_rate=.045,
                num_leaves=63, min_data_in_leaf=100, feature_fraction=.85,
                bagging_fraction=.85, bagging_freq=1, lambda_l2=12,
                max_cat_threshold=64, cat_smooth=20, verbosity=-1,
                num_threads=threads, seed=2026, feature_fraction_seed=2026,
                bagging_seed=2026, deterministic=True, force_col_wise=True)


def _build_one(paths: list[Path], row_file: Path, output: Path) -> dict:
    started = time.monotonic()
    source = [str(x) for x in paths]
    deps = (pl.scan_parquet(source).filter(pl.col("PHASE_mvt") == "DEP")
            .select(DEP_COLS).collect().to_pandas())
    arr = (pl.scan_parquet(source).filter(pl.col("PHASE_mvt") == "ARR")
           .select(ARR_COLS).collect().to_pandas())
    rows = pd.read_parquet(row_file, columns=["MVT_ID_mvt"])
    if (deps.MVT_ID_mvt.isna().any() or deps.MVT_ID_mvt.duplicated().any()
            or arr.MVT_ID_mvt.isna().any() or arr.MVT_ID_mvt.duplicated().any()
            or rows.MVT_ID_mvt.isna().any() or rows.MVT_ID_mvt.duplicated().any()):
        raise ValueError("Input movement IDs must be unique and non-null")
    ordered = rows.merge(deps, on="MVT_ID_mvt", how="left", sort=False,
                         validate="one_to_one")
    if len(ordered) != len(rows) or ordered.MVT_TIME_UTC_mvt.isna().any():
        raise ValueError("Raw departure covariates do not align with cached rows")
    features = build_arrival_traffic(ordered, arr)
    features.insert(0, "MVT_ID_mvt", rows.MVT_ID_mvt.to_numpy())
    output.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output, index=False)
    return {"rows": len(rows), "arrivals": len(arr), "columns": features.columns.tolist(),
            "seconds": round(time.monotonic() - started, 3), "path": str(output)}


def build_features(args: argparse.Namespace) -> dict:
    if args.threads > 4:
        raise ValueError("Limit this CPU experiment to four threads")
    train = _build_one(_training_files(args.data_dir),
                       args.cache_dir / "training_rows.parquet",
                       args.output_dir / "training_arrival_features.parquet")
    gc.collect()
    rank = _build_one([args.data_dir / "ranking.parquet"],
                      args.cache_dir / "ranking_rows.parquet",
                      args.output_dir / "ranking_arrival_features.parquet")
    report = {"training": train, "ranking": rank,
              "scope": "ARR fields only; DEP raw columns exclude BLOCK/TAXITIME"}
    (args.output_dir / "feature_build.json").write_text(json.dumps(report, indent=2),
                                                         encoding="utf-8")
    return report


def _load_all(args: argparse.Namespace, ranking: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, features = load_cache(args.cache_dir, ranking)
    features = add_flight_weather(rows, features, args.data_dir,
                                  args.weather_file, ranking)
    path = args.output_dir / ("ranking_arrival_features.parquet" if ranking
                              else "training_arrival_features.parquet")
    arr = pd.read_parquet(path)
    if (len(rows) != len(arr) or not np.array_equal(rows.MVT_ID_mvt.to_numpy(),
                                                   arr.MVT_ID_mvt.to_numpy())):
        raise ValueError("ARR feature cache is not aligned to baseline rows")
    features = pd.concat([features.reset_index(drop=True),
                          arr.drop(columns="MVT_ID_mvt").reset_index(drop=True)], axis=1)
    if features.columns.duplicated().any():
        raise ValueError("Duplicate feature names after ARR augmentation")
    return rows, features


def _masks(rows: pd.DataFrame, months: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    y = rows.target.to_numpy(dtype=float)
    proxy = rows.proxy.to_numpy(dtype=float)
    valid_proxy = np.isfinite(proxy) & (proxy >= 0) & (proxy <= 7200)
    heldout = rows.month.isin(months).to_numpy(dtype=bool)
    core = np.isfinite(y) & (y >= 0) & (y <= 86400) & valid_proxy
    query = heldout & np.isfinite(y) & valid_proxy
    return core & ~heldout, query


def _fit_fold(name: str, months: tuple[int, int], rows: pd.DataFrame,
              features: pd.DataFrame, reference: pd.DataFrame,
              args: argparse.Namespace) -> tuple[pd.DataFrame, dict]:
    train_mask, test_mask = _masks(rows, months)
    train_idx = np.flatnonzero(train_mask)
    test_idx = np.flatnonzero(test_mask)
    rng = np.random.default_rng(2026 if name == "seasonal_jan_jul" else 2033)
    indices = rng.permutation(train_idx)
    early_n = max(30000, int(.06 * len(indices)))
    early, fit = indices[:early_n], indices[early_n:]
    y = rows.target.to_numpy(dtype=float)
    proxy = rows.proxy.to_numpy(dtype=float)
    residual = y - proxy
    cats = [c for c in features if isinstance(features[c].dtype, pd.CategoricalDtype)]
    train_set = lgb.Dataset(features.iloc[fit], label=residual[fit],
                            categorical_feature=cats, free_raw_data=True)
    early_set = lgb.Dataset(features.iloc[early], label=residual[early],
                            categorical_feature=cats, reference=train_set,
                            free_raw_data=True)
    start = time.monotonic()
    model = lgb.train(_params(args.threads), train_set,
                      num_boost_round=args.max_rounds, valid_sets=[early_set],
                      callbacks=[lgb.early_stopping(100, verbose=False),
                                 lgb.log_evaluation(100)])
    fit_seconds = time.monotonic() - start
    expert = proxy[test_idx] + model.predict(features.iloc[test_idx],
                                              num_threads=args.threads)
    held = rows.iloc[test_idx][["MVT_ID_mvt", "target"]].copy()
    held["arrival_direct_expert"] = expert
    held = held.merge(reference[["MVT_ID_mvt", "target", "selected", "fold", "a_valid"]],
                      on="MVT_ID_mvt", how="left", validate="one_to_one",
                      suffixes=("", "_reference"))
    if (len(held) != len(test_idx) or held.selected.isna().any()
            or not held.fold.eq(name).all() or not held.a_valid.all()
            or not np.allclose(held.target, held.target_reference)):
        raise ValueError("Fold validation alignment to frozen v4 failed")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(str(args.output_dir / f"{name}.txt"))
    held.to_parquet(args.output_dir / f"{name}_oof.parquet", index=False)
    base_fold = reference.loc[reference.fold.eq(name),
                              ["MVT_ID_mvt", "target", "selected"]]
    full = base_fold.merge(held[["MVT_ID_mvt", "arrival_direct_expert"]],
                           on="MVT_ID_mvt", how="left", validate="one_to_one")
    truth = full.target.to_numpy(dtype=float)
    base = full.selected.to_numpy(dtype=float)
    expert_full = full.arrival_direct_expert.to_numpy(dtype=float)
    use = np.isfinite(expert_full)
    scores = {}
    for weight in WEIGHTS:
        pred = base.copy()
        pred[use] = np.maximum(base[use] + weight * (expert_full[use] - base[use]), 0)
        scores[str(weight)] = _rmse(truth, pred)
    info = {"fold": name, "months": months, "fitted_n": len(fit),
            "internal_early_n": len(early), "heldout_eligible_n": len(held),
            "heldout_all_finite_n": len(full), "best_iteration": model.best_iteration,
            "fit_seconds": round(fit_seconds, 3), "scores_all_finite_clipped": scores}
    (args.output_dir / f"{name}_validation.json").write_text(
        json.dumps(info, indent=2), encoding="utf-8")
    del train_set, early_set, model, full, held
    gc.collect()
    return base_fold, info


def validate(args: argparse.Namespace) -> dict:
    if args.threads > 4 or args.max_rounds > 2000:
        raise ValueError("Cap experiment at four CPU threads and 2000 rounds")
    rows, features = _load_all(args, False)
    reference = pd.read_parquet(args.v4_dir / "validation_predictions.parquet",
                                columns=["MVT_ID_mvt", "target", "selected", "fold", "a_valid"])
    if reference.MVT_ID_mvt.duplicated().any():
        raise ValueError("Frozen v4 OOF IDs duplicated")
    results: dict = {"features": features.columns.tolist(), "blend_weights": WEIGHTS,
                     "selection_fold": "seasonal_jan_jul", "folds": {}}
    _, seasonal = _fit_fold("seasonal_jan_jul", FOLDS["seasonal_jan_jul"],
                            rows, features, reference, args)
    selected = min(WEIGHTS, key=lambda w: (seasonal["scores_all_finite_clipped"][str(w)], w))
    results["folds"]["seasonal_jan_jul"] = seasonal
    results["selected_weight"] = selected
    print(json.dumps({"seasonal": seasonal["scores_all_finite_clipped"],
                      "selected_weight": selected}, indent=2), flush=True)
    _, forward = _fit_fold("forward_nov_dec", FOLDS["forward_nov_dec"],
                           rows, features, reference, args)
    results["folds"]["forward_nov_dec"] = forward
    print(json.dumps({"forward": forward["scores_all_finite_clipped"],
                      "fixed_weight": selected}, indent=2), flush=True)
    results["gain_both_folds"] = (
        selected > 0
        and seasonal["scores_all_finite_clipped"][str(selected)]
            < seasonal["scores_all_finite_clipped"]["0.0"]
        and forward["scores_all_finite_clipped"][str(selected)]
            < forward["scores_all_finite_clipped"]["0.0"])
    results["clean_split"] = "Each fold model excludes all corresponding held-out DEP labels."
    (args.output_dir / "validation.json").write_text(json.dumps(results, indent=2),
                                                     encoding="utf-8")
    return results


def final_predict(args: argparse.Namespace) -> dict:
    report = json.loads((args.output_dir / "validation.json").read_text(encoding="utf-8"))
    if not report["gain_both_folds"]:
        raise ValueError("Clean ARR model did not improve both validation folds")
    weight = float(report["selected_weight"])
    rounds = int(np.median([report["folds"][f]["best_iteration"] for f in FOLDS]))
    rows, features = _load_all(args, False)
    y = rows.target.to_numpy(dtype=float)
    proxy = rows.proxy.to_numpy(dtype=float)
    core = (np.isfinite(y) & (y >= 0) & (y <= 86400)
            & np.isfinite(proxy) & (proxy >= 0) & (proxy <= 7200))
    cats = [c for c in features if isinstance(features[c].dtype, pd.CategoricalDtype)]
    train_set = lgb.Dataset(features.loc[core], label=(y - proxy)[core],
                            categorical_feature=cats, free_raw_data=True)
    start = time.monotonic()
    model = lgb.train(_params(args.threads), train_set, num_boost_round=rounds,
                      callbacks=[lgb.log_evaluation(100)])
    fit_seconds = time.monotonic() - start
    model.save_model(str(args.output_dir / "final.txt"))
    del rows, features, train_set
    gc.collect()

    rank_rows, rank_features = _load_all(args, True)
    if list(rank_features.columns) != report["features"]:
        raise ValueError("Training/ranking feature schemas differ")
    rank_proxy = rank_rows.proxy.to_numpy(dtype=float)
    valid = np.isfinite(rank_proxy) & (rank_proxy >= 0) & (rank_proxy <= 7200)
    expert = np.full(len(rank_rows), np.nan, dtype=float)
    expert[valid] = rank_proxy[valid] + model.predict(
        rank_features.iloc[np.flatnonzero(valid)], num_threads=args.threads)
    raw = pd.DataFrame({"MVT_ID_mvt": rank_rows.MVT_ID_mvt,
                        "arrival_direct_expert": expert})
    raw.to_parquet(args.output_dir / "ranking_raw_expert.parquet", index=False)
    frozen = pd.read_parquet(args.v4_ranking,
                             columns=["MVT_ID_mvt", "TAXITIME_SEC_mvt"])
    merged = frozen.merge(raw, on="MVT_ID_mvt", how="left", sort=False,
                          validate="one_to_one")
    if len(merged) != len(frozen):
        raise ValueError("Ranking candidate merge changed row count")
    pred = merged.TAXITIME_SEC_mvt.to_numpy(dtype=float).copy()
    exp = merged.arrival_direct_expert.to_numpy(dtype=float)
    has = np.isfinite(exp)
    pred[has] = np.maximum(pred[has] + weight * (exp[has] - pred[has]), 0)
    if not np.isfinite(pred).all():
        raise ValueError("Ranking ARR prediction nonfinite")
    template = pd.read_parquet(args.data_dir / "submitting.parquet", columns=["MVT_ID_mvt"])
    if not np.array_equal(template.MVT_ID_mvt.to_numpy(), merged.MVT_ID_mvt.to_numpy()):
        raise ValueError("Ranking output does not match submission template order")
    pd.DataFrame({"MVT_ID_mvt": merged.MVT_ID_mvt,
                  "TAXITIME_SEC_mvt": pred}).to_parquet(
                      args.output_dir / "ranking_proposal.parquet", index=False)
    final_report = {"weight": weight, "rounds": rounds, "core_training_rows": int(core.sum()),
                    "ranking_rows": len(merged), "valid_proxy_ranking_rows": int(has.sum()),
                    "fit_seconds": round(fit_seconds, 3),
                    "frozen_v4_ranking": str(args.v4_ranking)}
    (args.output_dir / "final_report.json").write_text(json.dumps(final_report, indent=2),
                                                         encoding="utf-8")
    return final_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("build-features", "validate", "final-predict"),
                        required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/baseline"))
    parser.add_argument("--weather-file", type=Path, default=Path("data/external/weather.parquet"))
    parser.add_argument("--v4-dir", type=Path, default=Path("artifacts/v4"))
    parser.add_argument("--v4-ranking", type=Path,
                        default=Path("artifacts/catboost/source/sequential_predictions.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/v5-arrival-clean"))
    parser.add_argument("--max-rounds", type=int, default=1200)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    action = {"build-features": build_features, "validate": validate,
              "final-predict": final_predict}[args.mode]
    print(json.dumps(action(args), indent=2), flush=True)


if __name__ == "__main__":
    main()
