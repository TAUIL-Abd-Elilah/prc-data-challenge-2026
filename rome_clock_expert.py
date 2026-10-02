"""Local Rome long-schedule clock experiment, with frozen v4 validation.

The classifier estimates whether takeoff-minus-schedule is the label source;
the regressor estimates the label/schedule ratio otherwise. Only released
prediction-time fields enter either model. January/July selects a coarse
mixture blend, then November/December checks that frozen choice.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from catboost import CatBoostClassifier, CatBoostRegressor, Pool

from missing_catboost import FOLDS, columns, day_bootstrap, load_inputs, masks, params, rmse, train_model
from solution import _training_files


CLOCK_COLUMNS = ("sched_hour_utc", "sched_minute_utc", "sched_second_utc",
                 "sched_weekday_utc", "sched_quarter_hour",
                 "sched_calendar_day_offset")
SCALES = (0.0, 0.25, 0.5, 1.0)


def add_schedule_clock(args: argparse.Namespace, rows: pd.DataFrame,
                       features: pd.DataFrame, ranking: bool = False) -> None:
    files = [args.data_dir / "ranking.parquet"] if ranking else _training_files(args.data_dir)
    raw = (pl.scan_parquet([str(path) for path in files])
           .filter(pl.col("PHASE_mvt") == "DEP")
           .select(["MVT_ID_mvt", "SCHED_TIME_UTC_mvt"])
           .collect().to_pandas())
    if not np.array_equal(raw.MVT_ID_mvt.to_numpy(), rows.MVT_ID_mvt.to_numpy()):
        raise ValueError("Raw schedule clock and cached IDs do not align")
    sched = pd.to_datetime(raw.SCHED_TIME_UTC_mvt, utc=True, errors="coerce")
    movement = pd.to_datetime(rows.time, utc=True, errors="coerce")
    features["sched_hour_utc"] = sched.dt.hour.to_numpy(dtype="float32", na_value=np.nan)
    features["sched_minute_utc"] = sched.dt.minute.to_numpy(dtype="float32", na_value=np.nan)
    features["sched_second_utc"] = sched.dt.second.to_numpy(dtype="float32", na_value=np.nan)
    features["sched_weekday_utc"] = sched.dt.weekday.to_numpy(dtype="float32", na_value=np.nan)
    quarter = ((sched.dt.minute % 15 == 0) & (sched.dt.second == 0))
    features["sched_quarter_hour"] = quarter.astype("float32").to_numpy()
    day_delta = (movement.dt.floor("D") - sched.dt.floor("D")).dt.days
    features["sched_calendar_day_offset"] = day_delta.to_numpy(dtype="float32", na_value=np.nan)
    if any("BLOCK_TIME" in col.upper() or "TAXITIME" in col.upper()
           for col in features.columns):
        raise ValueError("Forbidden target-derived feature")


def long_columns(features: pd.DataFrame) -> tuple[list[str], list[str]]:
    names, cats = columns(features, long=True)
    if any(col not in features for col in CLOCK_COLUMNS):
        raise ValueError("Missing scheduled clock fields")
    return names + list(CLOCK_COLUMNS), cats


def fit_fold(args: argparse.Namespace, fold: str, months: tuple[int, int],
             rows: pd.DataFrame, features: pd.DataFrame) -> dict:
    mask = masks(rows, features, months)
    y = rows.target.to_numpy(dtype=float)
    schedule = features.schedule_proxy_unclipped.to_numpy(dtype=float)
    names, cats = long_columns(features)
    train_idx = np.flatnonzero(mask["lirf_train"])
    test_idx = np.flatnonzero(mask["lirf_test"])
    if len(train_idx) < 100 or len(test_idx) < 10:
        raise ValueError("Insufficient Rome long-schedule rows")
    ratio_y = np.divide(y, schedule, out=np.zeros_like(y),
                        where=np.isfinite(schedule) & (schedule != 0))
    ratio_y = np.clip(ratio_y, 0, 4)
    ratio_weight = np.clip(schedule / 3600, .5, 10) ** 2
    ratio, ratio_report = train_model(args, "rome_clock_ratio", "regression",
                                      features, train_idx, ratio_y, names, cats,
                                      args.iterations, 4, ratio_weight)
    exact_y = (np.abs(y - schedule) <= 60).astype(np.int8)
    classifier, exact_report = train_model(args, "rome_clock_exact", "classifier",
                                            features, train_idx, exact_y, names, cats,
                                            args.iterations, 4)
    xx = features.iloc[test_idx][names]
    ratio_pred = np.clip(ratio.predict(xx, thread_count=args.threads), 0, 4)
    ratio_pred *= schedule[test_idx]
    exact_p = classifier.predict_proba(xx, thread_count=args.threads)[:, 1]
    mixture = exact_p * schedule[test_idx] + (1 - exact_p) * ratio_pred
    if not np.isfinite(mixture).all():
        raise ValueError("Non-finite Rome mixture prediction")
    output = rows.iloc[test_idx][["MVT_ID_mvt", "target", "airport", "month", "time"]].copy()
    output["fold"] = fold
    output["schedule_proxy_unclipped"] = schedule[test_idx]
    output["ratio_candidate"] = ratio_pred
    output["p_schedule_exact"] = exact_p
    output["mixture_candidate"] = mixture
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ratio.save_model(str(args.output_dir / f"{fold}_ratio.cbm"))
    classifier.save_model(str(args.output_dir / f"{fold}_exact.cbm"))
    output.to_parquet(args.output_dir / f"{fold}_oof.parquet", index=False)
    report = {"fold": fold, "heldout_months": list(months),
              "train_rows": int(len(train_idx)), "test_gate_rows": int(len(test_idx)),
              "exact_train_rate": float(exact_y[train_idx].mean()),
              "exact_test_rate_diagnostic": float(exact_y[test_idx].mean()),
              "ratio_model": ratio_report, "exact_model": exact_report,
              "feature_names": names,
              "label_note": "Exact-source class is |target - takeoff-minus-schedule proxy| <= 60 seconds; target used only for training and heldout diagnostics"}
    (args.output_dir / f"{fold}_training.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    return report


def evaluate(args: argparse.Namespace) -> dict:
    parts = [pd.read_parquet(args.output_dir / f"{fold}_oof.parquet") for fold in FOLDS]
    expert = pd.concat(parts, ignore_index=True)
    if expert.MVT_ID_mvt.duplicated().any():
        raise ValueError("Rome OOF IDs duplicated across folds")
    v4 = pd.read_parquet(args.v4_dir / "validation_predictions.parquet",
                         columns=["MVT_ID_mvt", "target", "fold", "selected",
                                  "airport", "month", "MVT_TIME_UTC_mvt"])
    frame = v4.merge(expert[["MVT_ID_mvt", "target", "fold", "mixture_candidate"]],
                     on="MVT_ID_mvt", how="left", validate="one_to_one",
                     suffixes=("", "_expert"))
    gate = frame.mixture_candidate.notna().to_numpy(dtype=bool)
    if not np.allclose(frame.loc[gate, "target"], frame.loc[gate, "target_expert"]):
        raise ValueError("Rome labels do not match frozen v4 IDs")
    if not frame.loc[gate, "fold"].eq(frame.loc[gate, "fold_expert"]).all():
        raise ValueError("Rome folds do not match frozen v4 IDs")
    y = frame.target.to_numpy(dtype=float)
    base = frame.selected.to_numpy(dtype=float)
    if not np.isfinite(y).all() or not np.isfinite(base).all() or (base < 0).any():
        raise ValueError("Frozen v4 OOF is not all-finite and clipped")
    candidate = frame.mixture_candidate.to_numpy(dtype=float)
    predictions = {}
    report = {"reference": "Frozen clipped v4 all-finite OOF",
              "policy": "Mixture only, blended on no-NM invalid-AOBT LIRF schedule>7200 gate; coarse scale chosen on January/July, frozen for November/December",
              "gate_rows": {fold: int((gate & frame.fold.eq(fold).to_numpy()).sum())
                            for fold in FOLDS}, "scales": {}}
    for scale in SCALES:
        pred = base.copy()
        pred[gate] = np.maximum(base[gate] + scale * (candidate[gate] - base[gate]), 0)
        predictions[scale] = pred
        fold_report = {}
        for fold in FOLDS:
            fold_mask = frame.fold.eq(fold).to_numpy(dtype=bool)
            gate_fold = fold_mask & gate
            fold_report[fold] = {
                "all_finite_n": int(fold_mask.sum()),
                "all_finite_rmse_sec": rmse(y[fold_mask], pred[fold_mask]),
                "gate_rmse_sec": rmse(y[gate_fold], pred[gate_fold]),
                "gate_n": int(gate_fold.sum())}
        report["scales"][str(scale)] = fold_report
    choice = min(SCALES, key=lambda scale:
                 report["scales"][str(scale)]["seasonal_jan_jul"]["all_finite_rmse_sec"])
    report["selected_scale"] = choice
    report["selected_day_bootstrap"] = {}
    for i, fold in enumerate(FOLDS):
        fold_mask = frame.fold.eq(fold).to_numpy(dtype=bool)
        report["selected_day_bootstrap"][fold] = day_bootstrap(
            frame, base, predictions[choice], fold_mask, args.seed + i * 7)
    report["passed_local_gate"] = bool(choice > 0 and all(
        report["selected_day_bootstrap"][fold]["observed_rmse_gain_sec"] > 0
        and report["selected_day_bootstrap"][fold]["bootstrap_gain_95pct_interval_sec"][0] > 0
        for fold in FOLDS))
    report["gate_rule"] = "Positive all-finite gain and positive day-block 95% lower bound in both folds; otherwise reject"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame({"MVT_ID_mvt": frame.MVT_ID_mvt, "target": y, "fold": frame.fold,
                  "airport": frame.airport, "v4": base, "gate": gate,
                  "mixture_candidate": candidate,
                  "selected_prediction": predictions[choice]}).to_parquet(
                      args.output_dir / "validation_predictions.parquet", index=False)
    return report


def final_fit_predict(args: argparse.Namespace) -> dict:
    validation = json.loads((args.output_dir / "validation.json").read_text(encoding="utf-8"))
    if not validation["passed_local_gate"]:
        raise ValueError("Rome clock experiment did not pass the predeclared local gate")
    reports = [json.loads((args.output_dir / f"{fold}_training.json").read_text(
        encoding="utf-8")) for fold in FOLDS]
    ratio_iterations = int(np.median([r["ratio_model"]["best_iteration"] + 1 for r in reports]))
    exact_iterations = int(np.median([r["exact_model"]["best_iteration"] + 1 for r in reports]))
    rows, features = load_inputs(args)
    add_schedule_clock(args, rows, features)
    names, cats = long_columns(features)
    y = rows.target.to_numpy(dtype=float)
    schedule = features.schedule_proxy_unclipped.to_numpy(dtype=float)
    no_nm = (features.AOBT_3_flt_missing.to_numpy(dtype=bool)
             & features.LOBT_flt_missing.to_numpy(dtype=bool))
    train = (no_nm & ~np.isfinite(rows.proxy.to_numpy(dtype=float))
             & rows.airport.eq("LIRF").to_numpy(dtype=bool)
             & np.isfinite(schedule) & (schedule > 3600)
             & np.isfinite(y) & (y >= 0))
    idx = np.flatnonzero(train)
    ratio_y = np.clip(np.divide(y, schedule, out=np.zeros_like(y),
                                where=np.isfinite(schedule) & (schedule != 0)), 0, 4)
    ratio_weight = np.clip(schedule / 3600, .5, 10) ** 2
    ratio = CatBoostRegressor(**params(args, "regression", ratio_iterations, 4))
    classifier = CatBoostClassifier(**params(args, "classifier", exact_iterations, 4))
    start = time.monotonic()
    ratio.fit(Pool(features.iloc[idx][names], label=ratio_y[idx],
                   weight=ratio_weight[idx], cat_features=cats))
    exact_y = (np.abs(y - schedule) <= 60).astype(np.int8)
    classifier.fit(Pool(features.iloc[idx][names], label=exact_y[idx], cat_features=cats))
    fit_seconds = time.monotonic() - start
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ratio.save_model(str(args.output_dir / "final_ratio.cbm"))
    classifier.save_model(str(args.output_dir / "final_exact.cbm"))
    train_n = len(idx)
    del rows, features, y, schedule, no_nm, idx, ratio_y, ratio_weight, exact_y
    gc.collect()

    rank_rows, rank_features = load_inputs(args, ranking=True)
    add_schedule_clock(args, rank_rows, rank_features, ranking=True)
    if long_columns(rank_features) != (names, cats):
        raise ValueError("Ranking Rome feature schema differs from training")
    rank_schedule = rank_features.schedule_proxy_unclipped.to_numpy(dtype=float)
    gate = (rank_features.AOBT_3_flt_missing.to_numpy(dtype=bool)
            & rank_features.LOBT_flt_missing.to_numpy(dtype=bool)
            & ~np.isfinite(rank_rows.proxy.to_numpy(dtype=float))
            & rank_rows.airport.eq("LIRF").to_numpy(dtype=bool)
            & np.isfinite(rank_schedule) & (rank_schedule > 7200))
    rank_idx = np.flatnonzero(gate)
    raw = np.full(len(rank_rows), np.nan, dtype=float)
    p_exact = np.full(len(rank_rows), np.nan, dtype=float)
    ratio_pred = np.full(len(rank_rows), np.nan, dtype=float)
    xx = rank_features.iloc[rank_idx][names]
    ratio_pred[rank_idx] = np.clip(ratio.predict(xx, thread_count=args.threads), 0, 4) * rank_schedule[rank_idx]
    p_exact[rank_idx] = classifier.predict_proba(xx, thread_count=args.threads)[:, 1]
    raw[rank_idx] = p_exact[rank_idx] * rank_schedule[rank_idx] + (1 - p_exact[rank_idx]) * ratio_pred[rank_idx]
    if not np.isfinite(raw[gate]).all():
        raise ValueError("Non-finite Rome ranking expert on gate")
    expert = pd.DataFrame({"MVT_ID_mvt": rank_rows.MVT_ID_mvt, "rome_clock_gate": gate,
                           "schedule_proxy_unclipped": rank_schedule,
                           "rome_clock_ratio": ratio_pred,
                           "rome_clock_p_schedule_exact": p_exact,
                           "rome_clock_raw_expert": raw})
    expert.to_parquet(args.output_dir / "ranking_expert.parquet", index=False)
    manifest = {"purpose": "Locally validated raw Rome long-schedule ranking expert; no submission made",
                "training_rows": int(train_n), "ranking_gate_rows": int(gate.sum()),
                "ratio_iterations": ratio_iterations, "exact_iterations": exact_iterations,
                "fit_seconds": fit_seconds, "selected_scale": validation["selected_scale"],
                "output_file": str((args.output_dir / "ranking_expert.parquet").resolve())}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/baseline"))
    parser.add_argument("--v4-dir", type=Path, default=Path("artifacts/v4"))
    parser.add_argument("--weather-file", type=Path, default=Path("data/external/weather.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/v5-rome-clock"))
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--mode", choices=("fit", "evaluate", "final-predict"), default="fit")
    args = parser.parse_args()
    if not 1 <= args.threads <= 4:
        raise ValueError("Rome specialist is limited to four CPU threads")
    if args.mode == "evaluate":
        report = evaluate(args)
        print(json.dumps({"selected_scale": report["selected_scale"],
                          "passed_local_gate": report["passed_local_gate"],
                          "scales": report["scales"],
                          "day_bootstrap": report["selected_day_bootstrap"]}, indent=2))
        return
    if args.mode == "final-predict":
        print(json.dumps(final_fit_predict(args), indent=2))
        return
    rows, features = load_inputs(args)
    add_schedule_clock(args, rows, features)
    for fold, months in FOLDS.items():
        print(f"Fitting Rome clock specialist for {fold}", flush=True)
        result = fit_fold(args, fold, months, rows, features)
        print(json.dumps({"fold": fold, "train_rows": result["train_rows"],
                          "test_gate_rows": result["test_gate_rows"],
                          "fit_seconds": [result["ratio_model"]["fit_seconds"],
                                          result["exact_model"]["fit_seconds"]]}, indent=2), flush=True)
    del rows, features
    gc.collect()
    report = evaluate(args)
    print(json.dumps({"selected_scale": report["selected_scale"],
                      "passed_local_gate": report["passed_local_gate"],
                      "scales": report["scales"],
                      "day_bootstrap": report["selected_day_bootstrap"]}, indent=2))
    if report["passed_local_gate"]:
        print(json.dumps(final_fit_predict(args), indent=2))


if __name__ == "__main__":
    main()
