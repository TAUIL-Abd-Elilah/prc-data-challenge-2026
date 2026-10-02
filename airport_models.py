"""Fit airport-specific residual experts with weather and recurring flight categories."""
from __future__ import annotations
import argparse
import gc
import json
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
from solution import _training_files
from weather_model import add_weather, load_cached_features, rmse


def add_flight(x, rows, files):
    raw = pl.scan_parquet([str(p) for p in files]).filter(pl.col("PHASE_mvt") == "DEP").select(
        ["MVT_ID_mvt", "FLIGHT_mvt"]).collect().to_pandas()
    assert np.array_equal(raw.MVT_ID_mvt.to_numpy(), rows.MVT_ID_mvt.to_numpy())
    x["flight_name"] = raw.FLIGHT_mvt.astype("string").fillna("?").astype("category")
    return x


def fit(x, y, xv, yv, threads, rounds):
    cats = [c for c in x if isinstance(x[c].dtype, pd.CategoricalDtype)]
    ds = lgb.Dataset(x, y, categorical_feature=cats)
    params = dict(objective="regression", metric="rmse", learning_rate=.045,
                  num_leaves=63, min_data_in_leaf=35, lambda_l2=12,
                  feature_fraction=.9, bagging_fraction=.9, bagging_freq=1,
                  cat_smooth=25, max_cat_threshold=128, verbosity=-1,
                  num_threads=threads, seed=2026, deterministic=True, force_col_wise=True)
    if xv is None:
        return lgb.train(params, ds, num_boost_round=rounds)
    vs = lgb.Dataset(xv, yv, reference=ds, categorical_feature=cats)
    return lgb.train(params, ds, num_boost_round=rounds, valid_sets=[vs],
                     callbacks=[lgb.early_stopping(100, verbose=False)])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--baseline-dir", type=Path, default=Path("artifacts/baseline"))
    p.add_argument("--output-dir", type=Path, default=Path("artifacts/airports"))
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--rounds", type=int, default=1200)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    x, rows = load_cached_features(args.baseline_dir, ranking=False)
    x = add_weather(x, rows, args.data_dir / "external/weather.parquet")
    x = add_flight(x, rows, _training_files(args.data_dir))
    y, proxy = rows.target.to_numpy(dtype=float), rows.proxy.to_numpy(dtype=float)
    month, airport = rows.month.to_numpy(), rows.airport.to_numpy()
    eligible = np.isfinite(y) & np.isfinite(proxy)
    target = y - proxy
    reports, best_rounds = {}, {}
    for fold, months in {"seasonal_jan_jul": [1, 7], "forward_nov_dec": [11, 12]}.items():
        predictions = np.full(len(rows), np.nan)
        reports[fold] = {}
        for a in sorted(pd.unique(airport)):
            train = eligible & (airport == a) & ~np.isin(month, months)
            valid = eligible & (airport == a) & np.isin(month, months)
            model = fit(x.loc[train], target[train], x.loc[valid], target[valid], args.threads, args.rounds)
            pred = proxy[valid] + model.predict(x.loc[valid], num_threads=args.threads)
            predictions[valid] = pred
            best_rounds.setdefault(a, []).append(model.best_iteration)
            model.save_model(str(args.output_dir / f"{fold}_{a}.txt"))
            reports[fold][a] = {"rmse": rmse(y[valid], pred), "best_round": model.best_iteration}
            print(json.dumps({"fold": fold, "airport": a, **reports[fold][a]}), flush=True)
            del model
            gc.collect()
        valid = eligible & np.isin(month, months)
        oof = rows.loc[valid].copy()
        oof["airport_prediction"] = predictions[valid]
        oof.to_parquet(args.output_dir / f"{fold}_oof.parquet", index=False)
        reports[fold]["overall"] = rmse(y[valid], predictions[valid])
        print(json.dumps({"fold": fold, "overall_rmse": reports[fold]["overall"]}), flush=True)
    (args.output_dir / "validation.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    rounds = {a: int(np.median(v)) for a, v in best_rounds.items()}
    for a, n in rounds.items():
        train = eligible & (airport == a)
        model = fit(x.loc[train], target[train], None, None, args.threads, n)
        model.save_model(str(args.output_dir / f"{a}.txt"))
        print(f"Fitted full airport {a} with {n} rounds", flush=True)
    (args.output_dir / "model.json").write_text(json.dumps({"rounds": rounds, "features": list(x)}, indent=2), encoding="utf-8")
    del x
    gc.collect()
    xr, rank = load_cached_features(args.baseline_dir, ranking=True)
    xr = add_weather(xr, rank, args.data_dir / "external/weather.parquet")
    xr = add_flight(xr, rank, [args.data_dir / "ranking.parquet"])
    pred = np.full(len(rank), np.nan)
    for a in rounds:
        valid = np.isfinite(rank.proxy.to_numpy()) & rank.airport.eq(a).to_numpy()
        model = lgb.Booster(model_file=str(args.output_dir / f"{a}.txt"))
        pred[valid] = rank.loc[valid, "proxy"].to_numpy() + model.predict(xr.loc[valid], num_threads=args.threads)
    output = rank.copy()
    output["airport_prediction"] = pred
    output.to_parquet(args.output_dir / "ranking_predictions.parquet", index=False)


if __name__ == "__main__":
    main()
