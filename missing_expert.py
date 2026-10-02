"""Validate small specialists for departures lacking a usable NM off-block time."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl

from solution import _training_files, _rmse, build_features, load_data


def augment(x: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    x = x.copy()
    flight = raw["FLIGHT_mvt"].astype("string").fillna("?")
    x["flight_name"] = flight.astype("category")
    x["airport_flight"] = (raw["ADEP_mvt"].astype("string") + "_" + flight).astype("category")
    x["flight_id_missing"] = raw["FLIGHT_ID_mvt"].isna().astype("int8")
    takeoff = pd.to_datetime(raw["MVT_TIME_UTC_mvt"], utc=True)
    sched = pd.to_datetime(raw["SCHED_TIME_UTC_mvt"], utc=True)
    x["schedule_proxy_unclipped"] = (takeoff - sched).dt.total_seconds().astype("float32")
    x["schedule_hour"] = (sched.dt.hour + sched.dt.minute / 60).astype("float32")
    x["schedule_minute"] = sched.dt.minute.astype("float32")
    x["takeoff_minute"] = takeoff.dt.minute.astype("float32")
    x["takeoff_second"] = takeoff.dt.second.astype("float32")
    return x


def fit(x, y, xv, yv, rounds, threads):
    categories = [c for c in x if isinstance(x[c].dtype, pd.CategoricalDtype)]
    dataset = lgb.Dataset(x, y, categorical_feature=categories)
    params = dict(objective="regression", metric="rmse", learning_rate=.035,
                  num_leaves=31, min_data_in_leaf=15, lambda_l2=25,
                  feature_fraction=.9, bagging_fraction=.9, bagging_freq=1,
                  cat_smooth=15, max_cat_threshold=128, verbosity=-1,
                  num_threads=threads, seed=2026, deterministic=True, force_col_wise=True)
    if xv is None:
        return lgb.train(params, dataset, num_boost_round=rounds)
    valid = lgb.Dataset(xv, yv, reference=dataset, categorical_feature=categories)
    return lgb.train(params, dataset, num_boost_round=rounds, valid_sets=[valid],
                     callbacks=[lgb.early_stopping(100, verbose=False)])


def predict(args):
    metadata = json.loads((args.output_dir / "model.json").read_text(encoding="utf-8"))
    deps, traffic = load_data([args.data_dir / "ranking.parquet"])
    ranking_x, proxy = build_features(deps, traffic)
    valid = ~np.isfinite(proxy)
    ranking_x = augment(ranking_x.loc[valid].reset_index(drop=True), deps.loc[valid].reset_index(drop=True))
    rank_sched = ranking_x.schedule_proxy_unclipped.to_numpy(dtype=np.float64)
    rank_base = np.where(np.isfinite(rank_sched) & (rank_sched >= 0) & (rank_sched < 172800), rank_sched, 900.)
    prediction = pd.DataFrame({"MVT_ID_mvt": deps.loc[valid, "MVT_ID_mvt"].to_numpy()})
    for name in metadata["rounds"]:
        model = lgb.Booster(model_file=str(args.output_dir / f"{name}.txt"))
        prediction[name] = model.predict(ranking_x, num_threads=args.threads) + (rank_base if name == "schedule_residual" else 0)
    prediction.to_parquet(args.output_dir / "predictions.parquet", index=False)
    print(f"Saved missing-proxy models and predictions for {len(prediction)} ranking rows", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--cache-dir", type=Path, default=Path("artifacts/baseline"))
    p.add_argument("--output-dir", type=Path, default=Path("artifacts/missing"))
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--rounds", type=int, default=1400)
    p.add_argument("--predict-only", action="store_true")
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.predict_only:
        predict(args)
        return
    rows = pd.read_parquet(args.cache_dir / "training_rows.parquet")
    missing = ~np.isfinite(rows.proxy)
    raw_cols = ["MVT_ID_mvt", "FLIGHT_ID_mvt", "FLIGHT_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"]
    raw = pl.scan_parquet([str(f) for f in _training_files(args.data_dir)]).filter(
        pl.col("PHASE_mvt") == "DEP").select(raw_cols).collect().to_pandas()
    assert np.array_equal(raw.MVT_ID_mvt.to_numpy(), rows.MVT_ID_mvt.to_numpy())
    raw = raw.loc[missing].reset_index(drop=True)
    x = pd.read_parquet(args.cache_dir / "features.parquet").loc[missing].reset_index(drop=True)
    x = augment(x, raw)
    rows = rows.loc[missing].reset_index(drop=True)
    y = rows.target.to_numpy(dtype=np.float64)
    month = rows.month.to_numpy()
    schedule = x.schedule_proxy_unclipped.to_numpy(dtype=np.float64)
    base = np.where(np.isfinite(schedule) & (schedule >= 0) & (schedule < 172800), schedule, 900.)
    folds = {"seasonal_jan_jul": np.isin(month, [1, 7]), "forward_nov_dec": np.isin(month, [11, 12])}
    report, rounds_by_model = {}, {"direct": [], "schedule_residual": []}
    for name, valid in folds.items():
        train = ~valid & np.isfinite(y)
        valid &= np.isfinite(y)
        output = rows.loc[valid].copy()
        for model_name in rounds_by_model:
            offset = base if model_name == "schedule_residual" else np.zeros(len(y))
            model = fit(x.loc[train], y[train] - offset[train], x.loc[valid],
                        y[valid] - offset[valid], args.rounds, args.threads)
            pred = model.predict(x.loc[valid], num_threads=args.threads) + offset[valid]
            output[model_name] = pred
            rounds_by_model[model_name].append(model.best_iteration)
            model.save_model(str(args.output_dir / f"{name}_{model_name}.txt"))
        output.to_parquet(args.output_dir / f"{name}_oof.parquet", index=False)
        report[name] = {c: {"rmse": _rmse(output.target.to_numpy(), output[c].to_numpy()),
                            "by_airport": {str(a): _rmse(f.target.to_numpy(), f[c].to_numpy())
                                           for a, f in output.groupby("airport")}}
                        for c in rounds_by_model}
        report[name]["n"] = len(output)
        print(json.dumps({name: report[name]}, indent=2), flush=True)
    final_rounds = {n: int(np.median(v)) for n, v in rounds_by_model.items()}
    for name, rounds in final_rounds.items():
        offset = base if name == "schedule_residual" else np.zeros(len(y))
        model = fit(x, y - offset, None, None, rounds, args.threads)
        model.save_model(str(args.output_dir / f"{name}.txt"))
    (args.output_dir / "validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.output_dir / "model.json").write_text(json.dumps({"rounds": final_rounds, "features": list(x)}, indent=2), encoding="utf-8")
    predict(args)


if __name__ == "__main__":
    main()
