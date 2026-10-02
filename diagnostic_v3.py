"""Read-only local-label diagnostic for the v3 out-of-fold predictions.

Uses January, July, November, and December training covariates only. It never
reads ranking labels or movement block timestamps.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OOF = ROOT / "artifacts/lobt_ensemble/validation_predictions.parquet"
OUT = ROOT / "artifacts/diagnostic-v3"
MONTHS = ("01", "07", "11", "12")
RAW_COLS = [
    "MVT_ID_mvt", "FLIGHT_ID_mvt", "FLIGHT_mvt", "ADEP_mvt", "ADES_mvt",
    "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "AOBT_3_flt", "LOBT_flt",
    "IOBT_flt", "EOBT_1_flt", "RUNWAY_mvt", "STAND_mvt",
]


def stats(frame: pd.DataFrame) -> dict:
    n = len(frame)
    if n == 0:
        return {"n": 0, "rmse": None, "sse": 0.0}
    return {
        "n": n,
        "rmse": round(float(np.sqrt(frame.sq_error.mean())), 3),
        "sse": round(float(frame.sq_error.sum()), 3),
        "mean_error": round(float(frame.error.mean()), 3),
        "median_abs_error": round(float(frame.error.abs().median()), 3),
        "target_over_7200": int((frame.target > 7200).sum()),
        "target_negative": int((frame.target < 0).sum()),
    }


def grouped(frame: pd.DataFrame, cols: list[str]) -> dict:
    result = {}
    for key, group in frame.groupby(cols, observed=True, dropna=False, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        result["|".join(str(x) for x in key)] = stats(group)
    return result


def read_local_rows(ids: pd.Index) -> pd.DataFrame:
    parts = []
    for month in MONTHS:
        start = f"2025-{month}-01"
        end_month = int(month) + 1
        end = f"2025-{end_month:02d}-01" if end_month <= 12 else "2026-01-01"
        path = ROOT / "data" / f"training_{start}_{end}.parquet"
        raw = pd.read_parquet(path, columns=RAW_COLS)
        raw = raw.loc[raw.MVT_ID_mvt.isin(ids)].copy()
        raw["source_month"] = int(month)
        parts.append(raw)
    rows = pd.concat(parts, ignore_index=True)
    if rows.MVT_ID_mvt.duplicated().any() or rows.MVT_ID_mvt.isna().any():
        raise ValueError("Duplicate/null local training movement ID")
    return rows


def main() -> None:
    pred = pd.read_parquet(OOF)
    if pred.MVT_ID_mvt.duplicated().any():
        raise ValueError("Duplicate OOF movement ID")
    raw = read_local_rows(pd.Index(pred.MVT_ID_mvt))
    frame = pred.merge(raw, on="MVT_ID_mvt", how="left", validate="one_to_one")
    if frame.MVT_TIME_UTC_mvt.isna().any():
        raise ValueError("OOF movement ID missing local training covariates")
    frame["error"] = frame.target - frame.selected
    frame["sq_error"] = frame.error ** 2
    frame["schedule_proxy"] = (frame.MVT_TIME_UTC_mvt - frame.SCHED_TIME_UTC_mvt).dt.total_seconds()
    frame["aobt_proxy"] = (frame.MVT_TIME_UTC_mvt - frame.AOBT_3_flt).dt.total_seconds()
    frame["lobt_proxy"] = (frame.MVT_TIME_UTC_mvt - frame.LOBT_flt).dt.total_seconds()
    frame["aobt_lobt_abs_gap"] = (frame.AOBT_3_flt - frame.LOBT_flt).dt.total_seconds().abs()
    frame["flight_record"] = frame.FLIGHT_ID_mvt.notna()

    gap = frame.schedule_proxy.to_numpy(dtype=float)
    frame["schedule_bin"] = pd.cut(gap,
        bins=[-np.inf, 0, 1800, 3600, 7200, 12000, 20000, 30000, 60000, np.inf],
        labels=["<=0", "0-1800", "1800-3600", "3600-7200", "7200-12000",
                "12000-20000", "20000-30000", "30000-60000", ">60000"])
    frame["target_bin"] = pd.cut(frame.target,
        bins=[-np.inf, 0, 1800, 3600, 7200, 12000, 30000, 60000, np.inf],
        labels=["<0", "0-1800", "1800-3600", "3600-7200", "7200-12000",
                "12000-30000", "30000-60000", ">60000"], right=False)
    frame["source_class"] = np.select(
        [
            frame.a_valid & frame.aobt_lobt_abs_gap.gt(3600).fillna(False),
            frame.a_valid,
            ~frame.a_valid & frame.LOBT_flt.notna(),
            ~frame.a_valid & frame.flight_record,
        ],
        ["valid_AOBT_LOBT_disagree_gt1h", "valid_AOBT_other",
         "invalid_AOBT_LOBT_available", "invalid_AOBT_flight_record"],
        default="invalid_AOBT_no_flight_record",
    )

    total_sse = float(frame.sq_error.sum())
    rank = frame.sort_values("sq_error", ascending=False)
    top_columns = ["MVT_ID_mvt", "fold", "ADEP_mvt", "FLIGHT_mvt", "target",
                   "selected", "base", "lobt_prediction", "error", "sq_error",
                   "source_class", "schedule_proxy", "aobt_proxy", "lobt_proxy",
                   "aobt_lobt_abs_gap", "flight_record", "source_month", "RUNWAY_mvt", "STAND_mvt"]
    report = {
        "scope": "local labeled 2025 OOF rows only; no ranking labels or BLOCK timestamp read",
        "overall": stats(frame),
        "top_error_share": {str(k): round(float(rank.sq_error.head(k).sum() / total_sse), 6)
                            for k in (1, 2, 5, 10, 20, 50, 100, 500, 1000)},
        "by_fold": grouped(frame, ["fold"]),
        "by_airport": grouped(frame, ["ADEP_mvt"]),
        "by_source_class": grouped(frame, ["source_class"]),
        "by_airport_source": grouped(frame, ["ADEP_mvt", "source_class"]),
        "by_target_bin": grouped(frame, ["target_bin"]),
        "by_schedule_bin": grouped(frame, ["schedule_bin"]),
        "LIRF_invalid_by_schedule_bin": grouped(frame[(frame.ADEP_mvt == "LIRF") & ~frame.a_valid], ["schedule_bin"]),
        "LIRF_invalid_schedule_target_crosstab": pd.crosstab(
            frame.loc[(frame.ADEP_mvt == "LIRF") & ~frame.a_valid, "schedule_bin"],
            frame.loc[(frame.ADEP_mvt == "LIRF") & ~frame.a_valid, "target_bin"],
            dropna=False).astype(int).to_dict(),
        "top_30": rank[top_columns].head(30).replace({np.nan: None}).to_dict(orient="records"),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    frame[top_columns + ["schedule_bin", "target_bin"]].to_parquet(OUT / "row_diagnostics.parquet", index=False)
    print(json.dumps({"overall": report["overall"],
                      "top_error_share": report["top_error_share"],
                      "output": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
