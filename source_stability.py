"""Measure fixed expert gains with paired calendar-day bootstrap.

Uses frozen 2025 out-of-fold predictions and cached movement times. It does not
fit a model, choose a policy, read ranking labels, or use leaderboard scores.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


FOLDS = ("seasonal_jan_jul", "forward_nov_dec")
GPU_WEIGHT = 0.25


def load_candidate(args: argparse.Namespace) -> pd.DataFrame:
    if args.candidate == "source":
        pred = pd.read_parquet(args.source_predictions, columns=[
            "MVT_ID_mvt", "target", "fold", "v3", "source_candidate", "corrected"])
        return pred.rename(columns={"source_candidate": "candidate"})

    base = pd.read_parquet(args.v3_predictions, columns=[
        "MVT_ID_mvt", "target", "fold", "selected"])
    if base.MVT_ID_mvt.isna().any() or base.MVT_ID_mvt.duplicated().any():
        raise ValueError("Frozen v3 IDs must be unique and non-null")
    pieces = []
    for fold in FOLDS:
        part = pd.read_parquet(args.gpu_dir / f"{fold}_oof.parquet", columns=[
            "MVT_ID_mvt", "target", "fold", "gpu_prediction"])
        if not part.fold.eq(fold).all():
            raise ValueError(f"GPU OOF has wrong fold in {fold}")
        pieces.append(part)
    gpu = pd.concat(pieces, ignore_index=True)
    if gpu.MVT_ID_mvt.isna().any() or gpu.MVT_ID_mvt.duplicated().any():
        raise ValueError("GPU OOF IDs must be unique and non-null")
    pred = base.merge(gpu, on="MVT_ID_mvt", how="left", validate="one_to_one",
                      suffixes=("", "_gpu"))
    covered = pred.gpu_prediction.notna()
    if not pred.loc[covered, "fold"].eq(pred.loc[covered, "fold_gpu"]).all():
        raise ValueError("GPU and v3 OOF folds disagree")
    if not np.allclose(pred.loc[covered, "target"], pred.loc[covered, "target_gpu"]):
        raise ValueError("GPU and v3 OOF targets disagree")
    pred["candidate"] = covered
    pred["v3"] = pred.selected
    pred["corrected"] = pred.v3
    pred.loc[covered, "corrected"] += GPU_WEIGHT * (
        pred.loc[covered, "gpu_prediction"] - pred.loc[covered, "v3"])
    if args.candidate == "gpu_then_source":
        source_parts = []
        for fold in FOLDS:
            part = pd.read_parquet(args.source_dir / f"{fold}_oof.parquet", columns=[
                "MVT_ID_mvt", "target", "v3_prediction", "schedule_proxy_sec",
                "p_schedule_exact"])
            part["fold_source"] = fold
            source_parts.append(part)
        source = pd.concat(source_parts, ignore_index=True)
        if source.MVT_ID_mvt.isna().any() or source.MVT_ID_mvt.duplicated().any():
            raise ValueError("Source-classifier OOF IDs must be unique and non-null")
        pred = pred.merge(source, on="MVT_ID_mvt", how="left", validate="one_to_one",
                          suffixes=("", "_source"))
        selected = pred.p_schedule_exact.notna()
        if not pred.loc[selected, "fold"].eq(pred.loc[selected, "fold_source"]).all():
            raise ValueError("Source-classifier OOF folds disagree")
        if not np.allclose(pred.loc[selected, "target"], pred.loc[selected, "target_source"]):
            raise ValueError("Source-classifier OOF targets disagree")
        if not np.allclose(pred.loc[selected, "v3"], pred.loc[selected, "v3_prediction"]):
            raise ValueError("Source-classifier and frozen v3 predictions disagree")
        pred["v3"] = pred.corrected.copy()  # Fixed GPU blend is the comparator.
        pred["corrected"] = pred.v3
        pred.loc[selected, "corrected"] += 0.5 * pred.loc[selected, "p_schedule_exact"] * (
            pred.loc[selected, "schedule_proxy_sec"] - pred.loc[selected, "v3"])
        pred["candidate"] = selected
    return pred[["MVT_ID_mvt", "target", "fold", "v3", "candidate", "corrected"]]


def summarize(frame: pd.DataFrame) -> dict:
    n = len(frame)
    base_sse = float(frame.base_se.sum())
    corrected_sse = float(frame.corrected_se.sum())
    return {
        "n": n,
        "candidate_n": int(frame.candidate.sum()),
        "changed_n": int(frame.changed.sum()),
        "baseline_rmse_sec": float(np.sqrt(base_sse / n)),
        "corrected_rmse_sec": float(np.sqrt(corrected_sse / n)),
        "rmse_gain_sec": float(np.sqrt(base_sse / n) - np.sqrt(corrected_sse / n)),
        "paired_sse_gain_sec2": base_sse - corrected_sse,
    }


def bootstrap_days(frame: pd.DataFrame, repetitions: int, seed: int) -> dict:
    """Resample complete UTC days, stratified by calendar month."""
    daily = (frame.groupby(["month", "date"], observed=True, sort=True)
             .agg(n=("base_se", "size"), base_sse=("base_se", "sum"),
                  corrected_sse=("corrected_se", "sum"))
             .reset_index())
    rng = np.random.default_rng(seed)
    sampled_n = np.zeros(repetitions, dtype=np.float64)
    sampled_base = np.zeros(repetitions, dtype=np.float64)
    sampled_corrected = np.zeros(repetitions, dtype=np.float64)
    for _, month_days in daily.groupby("month", sort=True):
        values = month_days[["n", "base_sse", "corrected_sse"]].to_numpy(dtype=np.float64)
        draws = rng.integers(0, len(values), size=(repetitions, len(values)))
        totals = values[draws].sum(axis=1)
        sampled_n += totals[:, 0]
        sampled_base += totals[:, 1]
        sampled_corrected += totals[:, 2]
    gain = np.sqrt(sampled_base / sampled_n) - np.sqrt(sampled_corrected / sampled_n)
    return {
        "method": "Paired UTC-day resampling within each calendar month",
        "repetitions": repetitions,
        "seed": seed,
        "calendar_days": int(len(daily)),
        "rmse_gain_ci95_sec": [float(v) for v in np.quantile(gain, [0.025, 0.975])],
        "rmse_gain_median_sec": float(np.median(gain)),
        "resamples_with_positive_gain_fraction": float(np.mean(gain > 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=("source", "gpu", "gpu_then_source"),
                        default="source")
    parser.add_argument("--source-predictions", type=Path,
                        default=Path("artifacts/catboost/source/validation_predictions.parquet"))
    parser.add_argument("--v3-predictions", type=Path,
                        default=Path("artifacts/lobt_ensemble/validation_predictions.parquet"))
    parser.add_argument("--gpu-dir", type=Path, default=Path("artifacts/catboost/gpu"))
    parser.add_argument("--source-dir", type=Path, default=Path("artifacts/catboost/source"))
    parser.add_argument("--rows", type=Path,
                        default=Path("artifacts/baseline/training_rows.parquet"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repetitions", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.repetitions < 100:
        parser.error("Use at least 100 bootstrap repetitions")

    pred = load_candidate(args)
    rows = pd.read_parquet(args.rows, columns=["MVT_ID_mvt", "time", "airport"])
    if pred.MVT_ID_mvt.isna().any() or pred.MVT_ID_mvt.duplicated().any():
        raise ValueError("Prediction IDs must be unique and non-null")
    if rows.MVT_ID_mvt.isna().any() or rows.MVT_ID_mvt.duplicated().any():
        raise ValueError("Cached row IDs must be unique and non-null")
    frame = pred.merge(rows, on="MVT_ID_mvt", how="left", validate="one_to_one")
    if len(frame) != len(pred) or frame.time.isna().any() or frame.airport.isna().any():
        raise ValueError("Cached movement times or airports do not cover predictions")
    if not frame.fold.isin(FOLDS).all():
        raise ValueError("Unknown validation fold")
    values = frame[["target", "v3", "corrected"]].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("OOF target or predictions contain non-finite values")
    frame["candidate"] = frame.candidate.astype(bool)
    frame["changed"] = np.abs(frame.corrected - frame.v3) > 1e-9
    if (frame.changed & ~frame.candidate).any():
        raise ValueError("Correction changed a row outside the fixed candidate segment")
    utc = pd.to_datetime(frame.time, utc=True)
    frame["month"] = utc.dt.month
    frame["date"] = utc.dt.strftime("%Y-%m-%d")
    frame["base_se"] = (frame.target - frame.v3) ** 2
    frame["corrected_se"] = (frame.target - frame.corrected) ** 2
    frame["paired_sse_gain"] = frame.base_se - frame.corrected_se

    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": f"2025 out-of-fold labels only; fixed {args.candidate} correction; no policy optimization",
        "candidate": args.candidate,
        "baseline_name": "fixed_gpu_blend" if args.candidate == "gpu_then_source" else "frozen_v3",
        "fixed_weight": ({"gpu": GPU_WEIGHT, "source": 0.5}
                         if args.candidate == "gpu_then_source" else
                         GPU_WEIGHT if args.candidate == "gpu" else 0.5),
        "bootstrap_note": "Describes day-to-day variation in repeatedly examined 2025 folds; not an untouched 2026 performance interval.",
        "folds": {},
        "combined": {
            **summarize(frame),
            "bootstrap": bootstrap_days(frame, args.repetitions, args.seed + 2),
        },
        "by_airport": [],
        "by_date": [],
        "by_airport_date": [],
    }
    for i, fold in enumerate(FOLDS):
        subset = frame.loc[frame.fold.eq(fold)]
        report["folds"][fold] = {
            **summarize(subset),
            "bootstrap": bootstrap_days(subset, args.repetitions, args.seed + i),
        }
    for keys, name in ((["fold", "airport"], "by_airport"),
                       (["fold", "date"], "by_date"),
                       (["fold", "airport", "date"], "by_airport_date")):
        for group, subset in frame.groupby(keys, observed=True, sort=True):
            if not isinstance(group, tuple):
                group = (group,)
            report[name].append({**dict(zip(keys, group)), **summarize(subset)})

    output = args.output or Path(f"artifacts/catboost/{args.candidate}/stability.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"folds": report["folds"], "combined": report["combined"],
                      "by_airport": report["by_airport"]}, indent=2))


if __name__ == "__main__":
    main()
