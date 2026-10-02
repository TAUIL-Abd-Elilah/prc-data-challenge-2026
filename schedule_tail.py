"""Cross-validated Rome missing-NM taxi-out tail expert.

Only 2025 training labels calibrate the expert. The ranking file contributes
published covariates, never hidden taxi-out or off-block values. The outputs
cover every LIRF departure lacking AOBT, including extreme and negative labels.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")

import numpy as np
import pandas as pd
import polars as pl


TRAIN_PATTERN = re.compile(r"training_2025-\d\d-01_202[56]-\d\d-01\.parquet$")
THRESHOLDS = (12000, 20000, 30000, 40000, 60000, 86400)
FOLDS = {"seasonal_jan_jul": (1, 7), "forward_nov_dec": (11, 12)}
EXPERT_THRESHOLD = 12000
ROUTE_MIN_COUNT = 3
ROUTE_EXACT_RATE = 0.8


def rounded(value: float) -> float | None:
    return round(float(value), 3) if np.isfinite(value) else None


def rmse(y: np.ndarray, pred: np.ndarray) -> float | None:
    return rounded(np.sqrt(np.mean((y - pred) ** 2))) if len(y) else None


def load(paths: list[Path], training: bool) -> pd.DataFrame:
    columns = ["MVT_ID_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
               "FLIGHT_mvt", "ADES_mvt", "RUNWAY_mvt", "STAND_mvt"]
    if training:
        columns.append("TAXITIME_SEC_mvt")
    frame = (pl.scan_parquet([str(p) for p in paths])
             .filter((pl.col("PHASE_mvt") == "DEP") &
                     (pl.col("ADEP_mvt") == "LIRF") &
                     pl.col("AOBT_3_flt").is_null())
             .select(columns)
             .with_columns((pl.col("MVT_TIME_UTC_mvt") -
                            pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds()
                           .cast(pl.Float64).alias("schedule_proxy_sec"))
             .collect().to_pandas())
    frame["month"] = frame["MVT_TIME_UTC_mvt"].dt.month
    frame["flight_prefix"] = (frame["FLIGHT_mvt"].astype("string")
                              .str.extract(r"^([A-Za-z]{1,3})", expand=False)
                              .fillna("__MISSING__"))
    frame["ADES_mvt"] = frame["ADES_mvt"].astype("string").fillna("__MISSING__")
    if training:
        frame["target_sec"] = pd.to_numeric(frame["TAXITIME_SEC_mvt"], errors="coerce")
        frame["residual_sec"] = frame["target_sec"] - frame["schedule_proxy_sec"]
        frame["schedule_exact_60"] = frame["residual_sec"].abs() <= 60
    return frame


def analog_stats(reference: pd.DataFrame) -> pd.DataFrame:
    source = reference[reference["schedule_proxy_sec"] > 7200]
    return (source.groupby(["flight_prefix", "ADES_mvt"], dropna=False)
            .agg(analog_count=("schedule_exact_60", "size"),
                 analog_exact_rate=("schedule_exact_60", "mean"),
                 analog_median_residual_sec=("residual_sec", "median"))
            .reset_index())


def predict(reference: pd.DataFrame, query: pd.DataFrame, fold: str | None) -> pd.DataFrame:
    result = query.merge(analog_stats(reference), on=["flight_prefix", "ADES_mvt"], how="left")
    fallback = float(reference.loc[reference["schedule_proxy_sec"] < 7200, "target_sec"].mean())
    if not np.isfinite(fallback):
        fallback = float(reference["target_sec"].mean())
    gated = reference.loc[reference["schedule_proxy_sec"] > EXPERT_THRESHOLD, "residual_sec"]
    offset = float(gated.mean()) if len(gated) else 0.0
    schedule = result["schedule_proxy_sec"].to_numpy(dtype=np.float64)
    reliable = ((result["analog_count"] >= ROUTE_MIN_COUNT) &
                (result["analog_exact_rate"] >= ROUTE_EXACT_RATE)).fillna(False).to_numpy()
    group_offset = result["analog_median_residual_sec"].fillna(0).to_numpy(dtype=np.float64)
    route_prediction = schedule + np.where(reliable, group_offset, offset)
    expert_gate = np.isfinite(schedule) & (schedule > EXPERT_THRESHOLD)
    selected = np.where(expert_gate, route_prediction, fallback)

    output = pd.DataFrame({
        "MVT_ID_mvt": result["MVT_ID_mvt"].to_numpy(),
        "schedule_proxy_sec": schedule,
        "schedule_raw_sec": schedule,
        "schedule_mean_offset_sec": schedule + offset,
        "route_calibrated_sec": route_prediction,
        "selected_candidate_sec": selected,
        "fallback_mean_sec": np.full(len(result), fallback),
        "gate_12000": schedule > 12000,
        "gate_20000": schedule > 20000,
        "gate_30000": schedule > 30000,
        "analog_count": result["analog_count"].fillna(0).astype("int32").to_numpy(),
        "analog_exact_rate": result["analog_exact_rate"].to_numpy(dtype=np.float64),
        "analog_median_residual_sec": result["analog_median_residual_sec"].to_numpy(dtype=np.float64),
        "reliable_route_analog": reliable,
        "flight_prefix": result["flight_prefix"].to_numpy(),
        "ADES_mvt": result["ADES_mvt"].to_numpy(),
        "month": result["month"].to_numpy(dtype=np.int8),
    })
    if fold is not None:
        output.insert(1, "fold", fold)
        output.insert(2, "target_sec", result["target_sec"].to_numpy(dtype=np.float64))
    return output


def fold_thresholds(reference: pd.DataFrame, query: pd.DataFrame) -> dict:
    output = {}
    y = query["target_sec"].to_numpy(dtype=np.float64)
    schedule = query["schedule_proxy_sec"].to_numpy(dtype=np.float64)
    reference_schedule = reference["schedule_proxy_sec"].to_numpy(dtype=np.float64)
    reference_y = reference["target_sec"].to_numpy(dtype=np.float64)
    fallback = reference_y[reference_schedule < 7200].mean()
    if not np.isfinite(fallback):
        fallback = reference_y.mean()
    for threshold in THRESHOLDS:
        gate = np.isfinite(schedule) & (schedule > threshold)
        ref_gate = np.isfinite(reference_schedule) & (reference_schedule > threshold)
        val_y = y[gate]
        val_schedule = schedule[gate]
        residual = val_y - val_schedule
        train_offset = (reference_y[ref_gate] - reference_schedule[ref_gate]).mean() if ref_gate.any() else 0.0
        raw_all = np.where(gate, schedule, fallback)
        offset_all = np.where(gate, schedule + train_offset, fallback)
        output[str(threshold)] = {
            "training_gate_rows": int(ref_gate.sum()),
            "validation_gate_rows": int(gate.sum()),
            "nonlong_validation_rows": int((val_y <= 7200).sum()),
            "signed_residual_mean_sec": rounded(residual.mean()) if residual.size else None,
            "signed_residual_median_sec": rounded(np.median(residual)) if residual.size else None,
            "signed_residual_std_sec": rounded(residual.std()) if residual.size else None,
            "tail_rmse_raw_schedule_sec": rmse(val_y, val_schedule),
            "tail_rmse_training_mean_target_sec": rmse(
                val_y, np.full_like(val_y, reference_y[ref_gate].mean())) if ref_gate.any() else None,
            "tail_rmse_training_mean_offset_sec": rmse(val_y, val_schedule + train_offset),
            "all_missing_rmse_raw_gate_sec": rmse(y, raw_all),
            "all_missing_rmse_mean_offset_gate_sec": rmse(y, offset_all),
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/tail"))
    args = parser.parse_args()
    files = sorted(p for p in args.data_dir.glob("training_2025-*.parquet")
                   if TRAIN_PATTERN.fullmatch(p.name))
    if len(files) != 12:
        raise ValueError(f"Expected 12 canonical training files, found {len(files)}")
    rank_path = args.data_dir / "ranking.parquet"
    if not rank_path.exists():
        raise FileNotFoundError(rank_path)
    training = load(files, True)
    ranking = load([rank_path], False)
    if training["MVT_ID_mvt"].duplicated().any() or ranking["MVT_ID_mvt"].duplicated().any():
        raise ValueError("Movement IDs are not unique")

    out_of_fold = []
    report: dict = {"rules": {"expert_threshold_sec": EXPERT_THRESHOLD,
                             "route_min_count": ROUTE_MIN_COUNT,
                             "route_exact_rate": ROUTE_EXACT_RATE,
                             "fallback": "Training LIRF missing-NM mean target for schedule proxy <7200s"},
                    "folds": {}, "threshold_choice": {}}
    for name, months in FOLDS.items():
        valid = training[training["month"].isin(months)]
        reference = training[~training["month"].isin(months)]
        predictions = predict(reference, valid, name)
        out_of_fold.append(predictions)
        report["folds"][name] = {
            "train_rows": len(reference), "valid_rows": len(valid),
            "thresholds": fold_thresholds(reference, valid),
            "candidate_rmse_all_missing_sec": rmse(
                predictions["target_sec"].to_numpy(),
                predictions["selected_candidate_sec"].to_numpy()),
            "route_analog_valid_rows": int(predictions["reliable_route_analog"].sum()),
        }
    oof = pd.concat(out_of_fold, ignore_index=True)
    ranking_predictions = predict(training, ranking, None)
    # Choose a gate using the same held-out rows for every candidate. This is
    # only a standalone fallback comparison; an ensemble can reweight the
    # expert against its own out-of-fold baseline predictions.
    for threshold in THRESHOLDS:
        parts = [report["folds"][fold]["thresholds"][str(threshold)] for fold in FOLDS]
        total = sum(report["folds"][fold]["valid_rows"] for fold in FOLDS)
        weighted_mse = sum((part["all_missing_rmse_mean_offset_gate_sec"] ** 2) *
                           report["folds"][fold]["valid_rows"] for fold, part in zip(FOLDS, parts)) / total
        report["threshold_choice"][str(threshold)] = {
            "pooled_all_missing_rmse_mean_offset_gate_sec": rounded(np.sqrt(weighted_mse)),
            "heldout_gate_rows": sum(part["validation_gate_rows"] for part in parts),
            "heldout_nonlong_gate_rows": sum(part["nonlong_validation_rows"] for part in parts),
        }
    best = min(THRESHOLDS, key=lambda t: report["threshold_choice"][str(t)][
        "pooled_all_missing_rmse_mean_offset_gate_sec"])
    report["threshold_choice"]["selected_sec"] = best
    report["ranking_rows"] = len(ranking_predictions)
    report["ranking_over_86400"] = (ranking_predictions.loc[
        ranking_predictions["schedule_proxy_sec"] > 86400,
        ["schedule_proxy_sec", "flight_prefix", "ADES_mvt", "analog_count",
         "analog_exact_rate", "schedule_raw_sec", "selected_candidate_sec"]]
        .to_dict(orient="records"))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    oof.to_parquet(args.output_dir / "oof.parquet", index=False)
    ranking_predictions.to_parquet(args.output_dir / "ranking.parquet", index=False)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"oof_rows": len(oof), "ranking_rows": len(ranking_predictions),
                      "selected_threshold_sec": best,
                      "output_dir": str(args.output_dir.resolve())}, indent=2))


if __name__ == "__main__":
    main()
