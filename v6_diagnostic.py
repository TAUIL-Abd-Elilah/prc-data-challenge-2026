"""Aggregate 2025 v5 OOF error budget and same-flight ARR clock audit.

Uses released movement/NM timestamps as covariates and local 2025 labels only
for aggregate diagnostics. Writes no movement-level or private-row report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from solution import _training_files


DEP_COLS = ["MVT_ID_mvt", "FLIGHT_ID_mvt", "ADEP_mvt", "ADES_mvt",
            "MVT_TIME_UTC_mvt", "AOBT_3_flt", "ARVT_3_flt"]
ARR_COLS = ["FLIGHT_ID_mvt", "ADEP_mvt", "ADES_mvt", "MVT_TIME_UTC_mvt",
            "AOBT_3_flt", "ARVT_3_flt"]


def _rmse(residual: pd.Series | np.ndarray) -> float:
    values = np.asarray(residual, dtype=float)
    return float(np.sqrt(np.mean(values ** 2))) if len(values) else float("nan")


def _group(frame: pd.DataFrame, keys: list[str], total_sse: float) -> list[dict]:
    group = frame.groupby(keys, observed=True, dropna=False)
    result = group.agg(n=("se", "size"), sse=("se", "sum"),
                       rmse=("error", _rmse)).reset_index()
    result["sse_share"] = result.sse / total_sse
    result = result.sort_values("sse", ascending=False)
    return json.loads(result.to_json(orient="records"))


def error_budget(oof: pd.DataFrame) -> dict:
    frame = oof.copy()
    frame["error"] = frame.target - frame.selected
    frame["se"] = frame.error ** 2
    total = float(frame.se.sum())
    frame["target_bin"] = pd.cut(frame.target,
                                  [-np.inf, 0, 600, 1200, 1800, 3600,
                                   7200, 14400, 86400, np.inf])
    return {"rows": len(frame), "rmse_sec": _rmse(frame.error),
            "total_sse": total,
            "top_error_sse_share": {str(k): float(frame.se.nlargest(k).sum() / total)
                                    for k in (1, 2, 10, 100, 1000)},
            "by_valid": _group(frame, ["a_valid"], total),
            "by_airport": _group(frame, ["airport"], total),
            "by_target_bin": _group(frame, ["target_bin"], total),
            "largest_airport_valid_target_cells":
                _group(frame, ["airport", "a_valid", "target_bin"], total)[:25]}


def _summary(frame: pd.DataFrame, arrival_col: str, residual_col: str) -> dict:
    if frame.empty:
        return {"n": 0}
    arr = frame[arrival_col].to_numpy(dtype=float)
    dep = frame[residual_col].to_numpy(dtype=float)
    finite = np.isfinite(arr) & np.isfinite(dep)
    arr, dep = arr[finite], dep[finite]
    if len(arr) == 0:
        return {"n": 0}
    return {"n": len(arr), "arrival_median_sec": float(np.median(arr)),
            "arrival_p95_abs_sec": float(np.quantile(np.abs(arr), .95)),
            "departure_residual_median_sec": float(np.median(dep)),
            "departure_residual_rmse_sec": _rmse(dep),
            "pearson": float(np.corrcoef(arr, dep)[0, 1]) if len(arr) > 2 else None,
            "mean_product": float(np.mean(arr * dep))}


def same_flight_arrival_clock(files: list[Path], oof: pd.DataFrame,
                              rows_path: Path) -> dict:
    paths = [str(p) for p in files]
    dep_raw = (pl.scan_parquet(paths).filter(pl.col("PHASE_mvt") == "DEP")
               .select(DEP_COLS).collect().to_pandas())
    arr_raw = (pl.scan_parquet(paths).filter(pl.col("PHASE_mvt") == "ARR")
               .select(ARR_COLS).collect().to_pandas())
    arr_raw = arr_raw[arr_raw.FLIGHT_ID_mvt.notna()]
    arr_raw = arr_raw[~arr_raw.FLIGHT_ID_mvt.duplicated(keep=False)]
    rows = pd.read_parquet(rows_path, columns=["MVT_ID_mvt", "proxy"])
    dep = oof.drop(columns="MVT_TIME_UTC_mvt").merge(
        rows, on="MVT_ID_mvt", how="left", validate="one_to_one")
    dep = dep.merge(dep_raw, on="MVT_ID_mvt", how="left", validate="one_to_one")
    if dep.proxy.isna().all() or dep.MVT_TIME_UTC_mvt.isna().any():
        raise ValueError("OOF/raw/cache covariates do not align")
    pair = dep.merge(arr_raw, on="FLIGHT_ID_mvt", how="left", suffixes=("_dep", "_arr"),
                     validate="many_to_one")
    dep_time = pd.to_datetime(pair.MVT_TIME_UTC_mvt_dep, utc=True)
    arr_time = pd.to_datetime(pair.MVT_TIME_UTC_mvt_arr, utc=True)
    dt = (arr_time - dep_time).dt.total_seconds()
    same_route = (pair.ADEP_mvt_dep.eq(pair.ADEP_mvt_arr)
                  & pair.ADES_mvt_dep.eq(pair.ADES_mvt_arr))
    valid_pair = same_route & dt.between(900, 64800)
    pair = pair.loc[valid_pair].copy()
    pair["arrival_delta"] = (pd.to_datetime(pair.MVT_TIME_UTC_mvt_arr, utc=True)
                              - pd.to_datetime(pair.ARVT_3_flt_arr, utc=True)).dt.total_seconds()
    pair["aobt_delta"] = (pd.to_datetime(pair.AOBT_3_flt_arr, utc=True)
                           - pd.to_datetime(pair.AOBT_3_flt_dep, utc=True)).dt.total_seconds()
    pair["dep_residual"] = pair.target - pair.proxy
    pair["v5_error"] = pair.target - pair.selected
    arr_clock = pair.arrival_delta.to_numpy(dtype=float)
    plausible = np.isfinite(arr_clock) & (np.abs(arr_clock) <= 7200)
    subset = pair.loc[plausible & pair.a_valid].copy()
    ranges = [-7200, -1800, -600, -180, -60, 60, 180, 600, 1800, 7200]
    subset["arrival_delta_bin"] = pd.cut(subset.arrival_delta, ranges)
    binned = []
    for key, group in subset.groupby("arrival_delta_bin", observed=True):
        binned.append({"range": str(key), "n": len(group),
                       "arrival_median_sec": float(group.arrival_delta.median()),
                       "dep_residual_mean_sec": float(group.dep_residual.mean()),
                       "dep_residual_median_sec": float(group.dep_residual.median()),
                       "dep_residual_rmse_sec": _rmse(group.dep_residual),
                       "v5_rmse_sec": _rmse(group.v5_error)})
    folds = {name: _summary(group, "arrival_delta", "dep_residual")
             for name, group in subset.groupby("fold")}
    ai = pair.aobt_delta.to_numpy(dtype=float)
    report = {"matched_pairs": len(pair), "oof_rows": len(oof),
              "valid_aobt_matched": int(pair.a_valid.sum()),
              "arrival_nm_arvt_available": int(pair.arrival_delta.notna().sum()),
              "plausible_delta_valid_aobt": len(subset),
              "matched_v5_sse_share": float(np.square(pair.v5_error).sum()
                                           / np.square(oof.target - oof.selected).sum()),
              "overall_valid": _summary(subset, "arrival_delta", "dep_residual"),
              "folds": folds, "binned": binned,
              "aobt_crossrecord_n": int(np.isfinite(ai).sum()),
              "aobt_crossrecord_abs_gt60": int((np.abs(ai[np.isfinite(ai)]) > 60).sum())}
    return report


def rome_stand_audit(oof: pd.DataFrame, rows_path: Path,
                     baseline_features: Path, arrival_features: Path) -> dict:
    """Assess whether released same-stand ARR history resolves Rome ambiguity."""
    ids = pd.read_parquet(rows_path, columns=["MVT_ID_mvt"])
    schedule = pd.read_parquet(baseline_features,
                               columns=["takeoff_minus_SCHED_TIME_UTC_mvt"])
    if len(ids) != len(schedule):
        raise ValueError("Baseline feature cache is misaligned")
    schedule.insert(0, "MVT_ID_mvt", ids.MVT_ID_mvt.to_numpy())
    stand = pd.read_parquet(arrival_features,
                            columns=["MVT_ID_mvt", "arr_same_stand_last_block_gap"])
    frame = oof.loc[oof.airport.eq("LIRF") & ~oof.a_valid,
                     ["MVT_ID_mvt", "target", "selected", "fold"]]
    frame = frame.merge(schedule, on="MVT_ID_mvt", validate="one_to_one")
    frame = frame.merge(stand, on="MVT_ID_mvt", validate="one_to_one")
    frame["se"] = (frame.target - frame.selected) ** 2
    total_sse = float(np.square(oof.target - oof.selected).sum())
    # The 1-3h published schedule gap is the largest Rome invalid-AOBT cell.
    subset = frame.loc[frame.takeoff_minus_SCHED_TIME_UTC_mvt.between(3600, 12000)].copy()
    subset["stand_gap"] = pd.cut(subset.arr_same_stand_last_block_gap,
                                  [-1, 3600, 7200, 14400, 28800, 86400])
    cells = []
    for (fold, bin_value), group in subset.groupby(["fold", "stand_gap"], observed=True):
        if len(group) < 10:
            continue
        cells.append({"fold": fold, "stand_gap": str(bin_value), "n": len(group),
                      "target_mean_sec": float(group.target.mean()),
                      "prediction_mean_sec": float(group.selected.mean()),
                      "mean_error_sec": float((group.target - group.selected).mean()),
                      "rmse_sec": _rmse(group.target - group.selected),
                      "target_gt7200_rate": float((group.target > 7200).mean())})
    return {"rome_invalid_n": len(frame), "rome_invalid_sse_share":
            float(frame.se.sum() / total_sse),
            "schedule_3600_to_12000_n": len(subset), "cells_n_ge10": cells}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--v5-oof", type=Path,
                        default=Path("artifacts/v5-ensemble/validation_predictions.parquet"))
    parser.add_argument("--cache-rows", type=Path,
                        default=Path("artifacts/baseline/training_rows.parquet"))
    parser.add_argument("--baseline-features", type=Path,
                        default=Path("artifacts/baseline/features.parquet"))
    parser.add_argument("--arrival-features", type=Path,
                        default=Path("artifacts/v5-arrival-clean/training_arrival_features.parquet"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("artifacts/v6-diagnostic"))
    args = parser.parse_args()
    oof = pd.read_parquet(args.v5_oof,
                          columns=["MVT_ID_mvt", "target", "selected", "fold",
                                   "airport", "month", "a_valid", "MVT_TIME_UTC_mvt"])
    if oof.MVT_ID_mvt.isna().any() or oof.MVT_ID_mvt.duplicated().any():
        raise ValueError("v5 OOF movement IDs must be unique and non-null")
    report = {"scope": "Aggregate 2025 OOF labels only; no ranking labels or opaque IDs",
              "error_budget": error_budget(oof),
              "same_flight_arrival_clock": same_flight_arrival_clock(
                  _training_files(args.data_dir), oof, args.cache_rows),
              "rome_stand_audit": rome_stand_audit(
                  oof, args.cache_rows, args.baseline_features, args.arrival_features)}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2),
                                                   encoding="utf-8")
    print(json.dumps(report["same_flight_arrival_clock"], indent=2))


if __name__ == "__main__":
    main()
