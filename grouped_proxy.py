"""Small, reproducible grouped baseline for PRC 2026 taxi-out prediction.

The organizer supplies takeoff and NM actual off-block timestamps in ranking.
This model estimates the airport movement off-block mismatch from 2025 labels,
using hierarchical group means, and falls back to grouped target means when the
NM timestamp is absent or implausible. It never reads ranking target values.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl


GAP_EDGES = (300, 600, 900, 1200, 1500, 1800, 2400, 3600)
DELTA_EDGES = (-1800, -900, -600, -300, -120, 0, 120, 300, 600, 900, 1800)
SCHEDULE_EDGES = (-3600, -1800, -900, -300, 0, 300, 900, 1800, 3600, 7200, 14400)
MAX_PROXY_SEC = 7200


def _bin(expr: pl.Expr, edges: tuple[int, ...]) -> pl.Expr:
    count = pl.sum_horizontal([(expr > edge).fill_null(False).cast(pl.Int8) for edge in edges])
    return pl.when(expr.is_null()).then(-1).otherwise(count).cast(pl.Int16)


def _features(paths: list[Path], *, labeled: bool) -> pl.LazyFrame:
    if not paths:
        raise FileNotFoundError("No Parquet files supplied")
    scan = pl.scan_parquet([str(p) for p in paths]).filter(pl.col("PHASE_mvt") == "DEP")
    cols = ["MVT_ID_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "AOBT_3_flt",
            "LOBT_flt", "SCHED_TIME_UTC_mvt", "STAND_mvt"]
    if labeled:
        cols.append("TAXITIME_SEC_mvt")
    scan = scan.select(cols)
    a_gap = (pl.col("MVT_TIME_UTC_mvt") - pl.col("AOBT_3_flt")).dt.total_seconds()
    delta = (pl.col("AOBT_3_flt") - pl.col("LOBT_flt")).dt.total_seconds()
    sched_gap = (pl.col("MVT_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds()
    scan = scan.with_columns(
        pl.col("ADEP_mvt").fill_null("__MISSING__").alias("airport"),
        pl.col("MVT_TIME_UTC_mvt").dt.month().alias("month"),
        pl.col("STAND_mvt").fill_null("__MISSING__").str.extract(r"^([A-Za-z]{1,3})", 1)
          .fill_null("__MISSING__").alias("stand_zone"),
        a_gap.alias("a_gap"), delta.alias("delta"), sched_gap.alias("sched_gap"),
        _bin(a_gap, GAP_EDGES).alias("gap_bin"),
        _bin(delta, DELTA_EDGES).alias("delta_bin"),
        _bin(sched_gap, SCHEDULE_EDGES).alias("sched_bin"),
        a_gap.is_between(0, MAX_PROXY_SEC).fill_null(False).alias("a_valid"),
    )
    if labeled:
        scan = scan.with_columns(
            (pl.col("TAXITIME_SEC_mvt").cast(pl.Float64) - pl.col("a_gap")).alias("error")
        )
    return scan


def _group_stats(df: pl.LazyFrame, keys: list[str], value: str) -> pd.DataFrame:
    result = df.group_by(keys).agg(
        pl.len().alias("n"), pl.col(value).sum().alias("total")
    ).collect(engine="streaming").to_pandas()
    return result


def _coarsen(fine: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    return fine.groupby(keys, observed=True, dropna=False, as_index=False)[["n", "total"]].sum()


def _table_records(frame: pd.DataFrame) -> list[dict]:
    # Group columns are strings or integer bins, so native JSON is sufficient.
    return frame.to_dict(orient="records")


def fit_stats(features: pl.LazyFrame, *, exclude_months: tuple[int, ...] = ()) -> dict:
    train = features.filter(pl.col("TAXITIME_SEC_mvt").is_not_null())
    if exclude_months:
        train = train.filter(~pl.col("month").is_in(exclude_months))
    proxy = train.filter(pl.col("a_valid"))
    missing = train.filter(~pl.col("a_valid"))
    a_fine = _group_stats(proxy, ["airport", "gap_bin", "delta_bin"], "error")
    f_fine = _group_stats(missing, ["airport", "sched_bin", "stand_zone"], "TAXITIME_SEC_mvt")
    if a_fine.empty or f_fine.empty:
        raise ValueError("Training lacks valid proxy or fallback examples")
    a_airport = _coarsen(a_fine, ["airport"])
    a_gap = _coarsen(a_fine, ["airport", "gap_bin"])
    f_airport = _coarsen(f_fine, ["airport"])
    f_schedule = _coarsen(f_fine, ["airport", "sched_bin"])
    return {
        "proxy_global": float(a_fine.total.sum() / a_fine.n.sum()),
        "fallback_global": float(f_fine.total.sum() / f_fine.n.sum()),
        "proxy_rows": int(a_fine.n.sum()),
        "fallback_rows": int(f_fine.n.sum()),
        "a_airport": _table_records(a_airport),
        "a_gap": _table_records(a_gap),
        "a_fine": _table_records(a_fine),
        "f_airport": _table_records(f_airport),
        "f_schedule": _table_records(f_schedule),
        "f_fine": _table_records(f_fine),
    }


def _smooth(base: np.ndarray, rows: pd.DataFrame, stats: list[dict],
            keys: list[str], strength: float) -> np.ndarray:
    table = pd.DataFrame.from_records(stats)
    left = rows[keys].reset_index(drop=True)
    joined = left.merge(table, on=keys, how="left", sort=False, validate="many_to_one")
    n = joined["n"].fillna(0).to_numpy(dtype=np.float64)
    total = joined["total"].fillna(0).to_numpy(dtype=np.float64)
    return (total + strength * base) / (n + strength)


def predict_grouped(rows: pd.DataFrame, stats: dict) -> np.ndarray:
    n = len(rows)
    pred = np.empty(n, dtype=np.float64)
    valid = rows["a_valid"].fillna(False).to_numpy(dtype=bool)
    if valid.any():
        a = rows.loc[valid].reset_index(drop=True)
        parent = np.full(len(a), stats["proxy_global"], dtype=np.float64)
        parent = _smooth(parent, a, stats["a_airport"], ["airport"], 100)
        parent = _smooth(parent, a, stats["a_gap"], ["airport", "gap_bin"], 20)
        parent = _smooth(parent, a, stats["a_fine"],
                         ["airport", "gap_bin", "delta_bin"], 30)
        pred[valid] = a["a_gap"].to_numpy(dtype=np.float64) + parent
    if (~valid).any():
        f = rows.loc[~valid].reset_index(drop=True)
        parent = np.full(len(f), stats["fallback_global"], dtype=np.float64)
        parent = _smooth(parent, f, stats["f_airport"], ["airport"], 200)
        parent = _smooth(parent, f, stats["f_schedule"], ["airport", "sched_bin"], 100)
        parent = _smooth(parent, f, stats["f_fine"],
                         ["airport", "sched_bin", "stand_zone"], 50)
        pred[~valid] = parent
    return np.maximum(pred, 0)


def _metric(y: np.ndarray, pred: np.ndarray) -> dict:
    if not len(y):
        return {"n": 0, "rmse_sec": None, "mae_sec": None}
    e = pred - y
    return {"n": int(len(y)), "rmse_sec": float(np.sqrt(np.mean(e * e))),
            "mae_sec": float(np.mean(np.abs(e)))}


def validate(features: pl.LazyFrame, stats: dict,
             holdout_months: tuple[int, ...] = (1, 7)) -> tuple[dict, pd.DataFrame]:
    cols = ["MVT_ID_mvt", "airport", "month", "a_gap", "a_valid", "gap_bin",
            "delta_bin", "sched_bin", "stand_zone", "TAXITIME_SEC_mvt"]
    rows = features.filter(pl.col("month").is_in(holdout_months)).select(cols).collect(engine="streaming").to_pandas()
    rows = rows.loc[rows.TAXITIME_SEC_mvt.notna()].reset_index(drop=True)
    pred = predict_grouped(rows, stats)
    y = rows.TAXITIME_SEC_mvt.to_numpy(dtype=np.float64)
    valid = rows.a_valid.to_numpy(dtype=bool)
    report = {"overall": _metric(y, pred),
              "proxy_valid": _metric(y[valid], pred[valid]),
              "proxy_invalid": _metric(y[~valid], pred[~valid]),
              "by_month": {}, "by_airport": {}}
    for month in holdout_months:
        mask = rows.month.eq(month).to_numpy()
        report["by_month"][str(month)] = _metric(y[mask], pred[mask])
    for airport in sorted(rows.airport.unique()):
        mask = rows.airport.eq(airport).to_numpy()
        report["by_airport"][str(airport)] = _metric(y[mask], pred[mask])
    predictions = rows[["MVT_ID_mvt", "month", "airport", "a_valid", "TAXITIME_SEC_mvt"]].copy()
    predictions["prediction_sec"] = pred
    return report, predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/grouped"))
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="Allow training before all 12 monthly files arrive")
    args = parser.parse_args()
    boundaries = pd.date_range("2025-01-01", periods=13, freq="MS")
    expected = [args.data_dir / f"training_{boundaries[i].date()}_{boundaries[i + 1].date()}.parquet"
                for i in range(12)]
    files = [p for p in expected if p.exists()]
    if not files or (len(files) != 12 and not args.allow_incomplete):
        parser.error(f"Expected 12 training files, found {len(files)}; use --allow-incomplete for an experiment")
    features = _features(files, labeled=True)
    validation_stats = fit_stats(features, exclude_months=(1, 7))
    report, val_predictions = validate(features, validation_stats)
    forward_stats = fit_stats(features, exclude_months=(11, 12))
    forward_report, forward_predictions = validate(features, forward_stats,
                                                    holdout_months=(11, 12))
    final_stats = fit_stats(features)
    ranking = _features([args.data_dir / "ranking.parquet"], labeled=False)
    cols = ["MVT_ID_mvt", "airport", "a_gap", "a_valid", "gap_bin",
            "delta_bin", "sched_bin", "stand_zone"]
    rank_rows = ranking.select(cols).collect(engine="streaming").to_pandas()
    rank_pred = predict_grouped(rank_rows, final_stats)
    prediction = pd.Series(rank_pred, index=rank_rows.MVT_ID_mvt.to_numpy())
    template = pd.read_parquet(args.data_dir / "submitting.parquet", columns=["MVT_ID_mvt"])
    ids = template.MVT_ID_mvt
    if ids.isna().any() or ids.duplicated().any() or set(ids) != set(prediction.index):
        raise ValueError("Submission template IDs do not match ranking departures")
    output = template.copy()
    output["TAXITIME_SEC_mvt"] = ids.map(prediction).to_numpy(dtype=np.float64)
    if not np.isfinite(output.TAXITIME_SEC_mvt).all():
        raise ValueError("Grouped predictions contain non-finite values")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {"training_files": [p.name for p in files], "complete_year": len(files) == 12,
                "holdout_months": [1, 7], "max_proxy_sec": MAX_PROXY_SEC,
                "gap_edges": GAP_EDGES, "delta_edges": DELTA_EDGES,
                "schedule_edges": SCHEDULE_EDGES}
    report["metadata"] = metadata
    forward_report["metadata"] = {**metadata, "holdout_months": [11, 12]}
    (args.output_dir / "validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.output_dir / "forward_validation.json").write_text(json.dumps(forward_report, indent=2), encoding="utf-8")
    (args.output_dir / "model.json").write_text(json.dumps({"metadata": metadata,
                                                            "stats": final_stats}), encoding="utf-8")
    val_predictions.to_parquet(args.output_dir / "validation_predictions.parquet", index=False)
    forward_predictions.to_parquet(args.output_dir / "forward_validation_predictions.parquet", index=False)
    output.to_parquet(args.output_dir / "predictions.parquet", index=False)
    print(json.dumps({"training_files": len(files), "holdout": report["overall"],
                      "proxy_valid": report["proxy_valid"],
                      "proxy_invalid": report["proxy_invalid"],
                      "ranking_rows": len(output), "output_dir": str(args.output_dir.resolve())}, indent=2))


if __name__ == "__main__":
    main()
