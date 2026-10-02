"""Small specialist for Rome departures without a plausible NM off-block time.

An out-of-fold classifier estimates whether taxi-out matches takeoff minus the
published schedule. A second model predicts the nonmatching outcome, and a
ratio model supplies a linearly extrapolating alternative for large delays.
The ranking file contributes only public covariates, never hidden labels.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl

from missing_expert import augment


TRAIN_PATTERN = re.compile(r"training_2025-\d\d-01_202[56]-\d\d-01\.parquet$")
FOLDS = {"seasonal_jan_jul": (1, 7), "forward_nov_dec": (11, 12)}


def rmse(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - pred) ** 2)))


def _cache_rows(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def _indexed_features(path: Path, row_indices: np.ndarray) -> pd.DataFrame:
    result = (pl.scan_parquet(str(path)).with_row_index("__row")
              .filter(pl.col("__row").is_in(row_indices.astype(int).tolist()))
              .collect().sort("__row").drop("__row").to_pandas())
    if len(result) != len(row_indices):
        raise ValueError("Feature cache selection lost rows")
    return result


def load_subset(data_dir: Path, cache_dir: Path, training: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    if training:
        paths = sorted(p for p in data_dir.glob("training_2025-*.parquet")
                       if TRAIN_PATTERN.fullmatch(p.name))
        if len(paths) != 12:
            raise ValueError(f"Expected 12 canonical training files, got {len(paths)}")
        rows_path = cache_dir / "training_rows.parquet"
        features_path = cache_dir / "features.parquet"
    else:
        paths = [data_dir / "ranking.parquet"]
        rows_path = cache_dir / "ranking_rows.parquet"
        features_path = cache_dir / "ranking_features.parquet"
    rows = _cache_rows(rows_path)
    mask = rows["airport"].eq("LIRF").to_numpy() & ~np.isfinite(rows["proxy"].to_numpy())
    indices = np.flatnonzero(mask)
    subset = rows.iloc[indices].reset_index(drop=True)
    x = _indexed_features(features_path, indices)

    columns = ["MVT_ID_mvt", "FLIGHT_ID_mvt", "FLIGHT_mvt", "ADEP_mvt",
               "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"]
    raw = (pl.scan_parquet([str(p) for p in paths])
           .filter((pl.col("PHASE_mvt") == "DEP") & (pl.col("ADEP_mvt") == "LIRF"))
           .select(columns).collect().to_pandas())
    raw = raw.set_index("MVT_ID_mvt").loc[subset["MVT_ID_mvt"].to_numpy()].reset_index()
    x = augment(x, raw)
    if not np.array_equal(raw["MVT_ID_mvt"].to_numpy(), subset["MVT_ID_mvt"].to_numpy()):
        raise ValueError("Raw rows and cached features are misaligned")
    subset["schedule_proxy_sec"] = x["schedule_proxy_unclipped"].astype(np.float64)
    subset["flight_prefix"] = (raw["FLIGHT_mvt"].astype("string")
                               .str.extract(r"^([A-Za-z]{1,3})", expand=False)
                               .fillna("__MISSING__"))
    subset["destination"] = x["ADES_mvt"].astype("string").fillna("__MISSING__")
    return subset, x


def _fit(x: pd.DataFrame, y: np.ndarray, kind: str, threads: int,
         rounds: int, weights: np.ndarray | None = None) -> lgb.Booster:
    categorical = [col for col in x if isinstance(x[col].dtype, pd.CategoricalDtype)]
    params = dict(objective="binary" if kind == "classifier" else "regression",
                  metric="binary_logloss" if kind == "classifier" else "rmse",
                  learning_rate=0.035, num_leaves=9, min_data_in_leaf=22,
                  lambda_l2=25.0, feature_fraction=0.85,
                  bagging_fraction=0.9, bagging_freq=1, cat_smooth=20,
                  max_cat_threshold=64, verbosity=-1, num_threads=threads,
                  seed=2026, deterministic=True, force_col_wise=True)
    data = lgb.Dataset(x, label=y, weight=weights, categorical_feature=categorical)
    return lgb.train(params, data, num_boost_round=rounds)


def _route_evidence(reference: pd.DataFrame, query: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    ref = reference.copy()
    ref["exact"] = (ref["target"].to_numpy() - ref["schedule_proxy_sec"].to_numpy()) ** 2 <= 60 ** 2
    route = (ref.loc[ref["schedule_proxy_sec"] > 7200]
             .groupby(["flight_prefix", "destination"], dropna=False)
             .agg(route_count=("exact", "size"), route_exact_rate=("exact", "mean"))
             .reset_index())
    merged = query[["flight_prefix", "destination"]].merge(
        route, on=["flight_prefix", "destination"], how="left", sort=False)
    return (merged["route_count"].fillna(0).to_numpy(dtype=np.int32),
            merged["route_exact_rate"].to_numpy(dtype=np.float64))


def fit_predict(reference: pd.DataFrame, x_reference: pd.DataFrame,
                query: pd.DataFrame, x_query: pd.DataFrame,
                threads: int, rounds: int) -> pd.DataFrame:
    y = reference["target"].to_numpy(dtype=np.float64)
    schedule = reference["schedule_proxy_sec"].to_numpy(dtype=np.float64)
    exact = (np.abs(y - schedule) <= 60).astype(np.int8)

    classifier = _fit(x_reference, exact, "classifier", threads, rounds)
    p_exact = classifier.predict(x_query, num_threads=threads)

    nonmatch = exact == 0
    fallback = _fit(x_reference.loc[nonmatch], y[nonmatch], "regression", threads, rounds)
    fallback_pred = np.maximum(0, fallback.predict(x_query, num_threads=threads))

    ratio_mask = np.isfinite(schedule) & (schedule > 3600) & np.isfinite(y)
    ratio_target = np.clip(y[ratio_mask] / schedule[ratio_mask], -0.5, 5.0)
    weights = np.clip(schedule[ratio_mask] / 3600, 0.5, 10.0) ** 2
    ratio = _fit(x_reference.loc[ratio_mask], ratio_target, "regression",
                 threads, rounds, weights)
    q_schedule = query["schedule_proxy_sec"].to_numpy(dtype=np.float64)
    ratio_pred = np.where(q_schedule > 3600,
                          q_schedule * np.clip(ratio.predict(x_query, num_threads=threads), 0, 4),
                          fallback_pred)

    route_count, route_exact_rate = _route_evidence(reference, query)
    reliable = ((route_count >= 3) & (route_exact_rate >= .8) & (q_schedule > 12000))
    route_p = np.where(reliable, np.maximum(p_exact, route_exact_rate), p_exact)
    mixture = p_exact * q_schedule + (1 - p_exact) * fallback_pred
    mixture_route = route_p * q_schedule + (1 - route_p) * fallback_pred
    return pd.DataFrame({
        "MVT_ID_mvt": query["MVT_ID_mvt"].to_numpy(),
        "schedule_proxy_sec": q_schedule,
        "p_exact": p_exact,
        "fallback_nonmatch": fallback_pred,
        "mixture": np.maximum(mixture, 0),
        "mixture_route": np.maximum(mixture_route, 0),
        "ratio_model": np.maximum(ratio_pred, 0),
        "ratio_mixture_half": np.maximum(.5 * ratio_pred + .5 * mixture_route, 0),
        "route_count": route_count,
        "route_exact_rate": route_exact_rate,
        "reliable_route": reliable,
        "month": query["month"].to_numpy(),
    })


def evaluate(output: pd.DataFrame, baseline: pd.DataFrame | None = None) -> dict:
    y = output["target"].to_numpy(dtype=np.float64)
    if baseline is not None:
        output = output.merge(baseline[["MVT_ID_mvt", "direct", "schedule_residual"]],
                              on="MVT_ID_mvt", how="left", validate="one_to_one")
    result = {}
    for col in ("mixture", "mixture_route", "ratio_model", "ratio_mixture_half",
                "direct", "schedule_residual"):
        if col in output:
            pred = output[col].to_numpy(dtype=np.float64)
            result[col] = rmse(y, pred) if np.isfinite(pred).all() else None
    for threshold in (12000, 20000, 30000, 86400):
        selected = output["schedule_proxy_sec"].to_numpy() > threshold
        result[f"tail_over_{threshold}"] = {
            "rows": int(selected.sum()),
            **{col: rmse(y[selected], output.loc[selected, col].to_numpy(dtype=np.float64))
               for col in ("mixture", "mixture_route", "ratio_model", "ratio_mixture_half")
               if selected.any()},
        }
    exact = np.abs(y - output["schedule_proxy_sec"].to_numpy()) <= 60
    result["exact_rate"] = float(exact.mean())
    result["brier"] = float(np.mean((exact.astype(float) - output["p_exact"].to_numpy()) ** 2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/baseline"))
    parser.add_argument("--comparison-dir", type=Path, default=Path("artifacts/missing"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/lirf"))
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=220)
    args = parser.parse_args()
    train_rows, train_x = load_subset(args.data_dir, args.cache_dir, True)
    rank_rows, rank_x = load_subset(args.data_dir, args.cache_dir, False)
    if train_rows["target"].isna().any():
        raise ValueError("Missing labels in Rome training subset")

    reports: dict = {"training_rows": len(train_rows), "ranking_rows": len(rank_rows),
                     "folds": {}, "notes": "All Rome departures with missing or implausible AOBT proxy; ranking labels not read."}
    oof_parts = []
    for fold, months in FOLDS.items():
        valid = train_rows["month"].isin(months).to_numpy()
        result = fit_predict(train_rows.loc[~valid].reset_index(drop=True),
                             train_x.loc[~valid].reset_index(drop=True),
                             train_rows.loc[valid].reset_index(drop=True),
                             train_x.loc[valid].reset_index(drop=True),
                             args.threads, args.rounds)
        result.insert(1, "fold", fold)
        result.insert(2, "target", train_rows.loc[valid, "target"].to_numpy())
        comparison = args.comparison_dir / f"{fold}_oof.parquet"
        baseline = pd.read_parquet(comparison) if comparison.exists() else None
        reports["folds"][fold] = evaluate(result, baseline)
        reports["folds"][fold]["rows"] = int(valid.sum())
        oof_parts.append(result)
    oof = pd.concat(oof_parts, ignore_index=True)
    rank = fit_predict(train_rows, train_x, rank_rows, rank_x, args.threads, args.rounds)

    # Publish the specialist for combination only if it improves at least one
    # full held-out fold against the existing missing-proxy specialists.
    improved = any(
        min(reports["folds"][fold][col] for col in
            ("mixture", "mixture_route", "ratio_model", "ratio_mixture_half")) <
        min(reports["folds"][fold][col] for col in ("direct", "schedule_residual"))
        for fold in FOLDS
    )
    reports["improves_existing_on_at_least_one_fold"] = bool(improved)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "validation.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    if improved:
        oof.to_parquet(args.output_dir / "oof.parquet", index=False)
        rank.to_parquet(args.output_dir / "ranking.parquet", index=False)
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
