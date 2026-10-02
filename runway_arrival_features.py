"""Audit runway arrival/departure sequencing with released movement fields.

Feature construction is label-free: raw reads contain phase, airport, runway,
movement time and departure movement ID only. Missing or placeholder runways
never form a shared group. Counts use strict time bounds, so a departure cannot
see itself or any same-second peer in the past/future departure headways.
The audit reads only released 2025 v5 OOF labels after features are saved.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from solution import _training_files


RAW_DEP = ("MVT_ID_mvt", "ADEP_mvt", "RUNWAY_mvt", "MVT_TIME_UTC_mvt")
RAW_ARR = ("ADES_mvt", "RUNWAY_mvt", "MVT_TIME_UTC_mvt")
PLACEHOLDERS = {"", "?", "UNKNOWN", "UNKN", "UNK", "N/A", "NA",
                "NONE", "NULL", "NIL", "__MISSING__"}
WINDOWS = (900, 3600)
FEATURES = (
    "runway_arr_past15m_count", "runway_arr_future15m_count",
    "runway_arr_past60m_count", "runway_arr_future60m_count",
    "runway_arr_prev_gap_sec", "runway_arr_next_gap_sec",
    "runway_dep_prev_gap_sec", "runway_dep_next_gap_sec",
)
SEED = 20261002


def protocol() -> dict:
    return {"raw_phase_filter": "DEP for queries, ARR for arrival events",
            "raw_departure_columns": list(RAW_DEP),
            "raw_arrival_columns": list(RAW_ARR),
            "forbidden_departure_predictors": ["BLOCK_TIME_UTC_mvt",
                                                "TAXITIME_SEC_mvt"],
            "grouping": "exact airport + published runway code",
            "excluded_runway_labels": sorted(PLACEHOLDERS),
            "time_windows_seconds": list(WINDOWS),
            "intervals": "past [t-window,t), future (t,t+window]; same-second peers excluded",
            "features": list(FEATURES),
            "validation_scope": "2025 v5 out-of-fold residuals for January, July, November and December",
            "audit_controls": "airport, runway, month, UTC hour, airport-relative departure-density tertile",
            "audit_only": True, "ranking_labels_read": False}


def freeze_protocol(path: Path) -> None:
    value = protocol()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise ValueError("Existing runway-arrival protocol differs from source")
    else:
        path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def seconds(values: pd.Series) -> np.ndarray:
    dt = pd.to_datetime(values, utc=True, errors="coerce")
    return dt.dt.as_unit("ns").astype("int64").to_numpy() // 1_000_000_000


def normalized(values: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    text = values.astype("string").fillna("").str.strip().str.upper()
    value = text.to_numpy(dtype=str)
    good = ~np.isin(value, list(PLACEHOLDERS))
    return value, good


def time_groups(key: np.ndarray, valid: np.ndarray,
                time_sec: np.ndarray) -> dict[str, np.ndarray]:
    global_index = np.flatnonzero(valid)
    group = pd.Series(key[global_index]).groupby(key[global_index],
                                                  sort=False).indices
    return {str(k): np.sort(time_sec[global_index[np.asarray(index,
                                                              dtype=np.int64)]])
            for k, index in group.items()}


def build_from_raw(dep: pd.DataFrame, arr: pd.DataFrame) -> pd.DataFrame:
    n = len(dep)
    airport, good_airport = normalized(dep.ADEP_mvt)
    runway, good_runway = normalized(dep.RUNWAY_mvt)
    arr_airport, arr_good_airport = normalized(arr.ADES_mvt)
    arr_runway, arr_good_runway = normalized(arr.RUNWAY_mvt)
    q_time = seconds(dep.MVT_TIME_UTC_mvt)
    arr_time = seconds(arr.MVT_TIME_UTC_mvt)
    query_good = good_airport & good_runway & dep.MVT_TIME_UTC_mvt.notna().to_numpy()
    arr_good = (arr_good_airport & arr_good_runway
                & arr.MVT_TIME_UTC_mvt.notna().to_numpy())
    dep_key = np.char.add(np.char.add(airport, "|"), runway)
    arr_key = np.char.add(np.char.add(arr_airport, "|"), arr_runway)
    arr_groups = time_groups(arr_key, arr_good, arr_time)
    dep_groups = time_groups(dep_key, query_good, q_time)

    feature = {name: np.full(n, np.nan, dtype=np.float32) for name in FEATURES}
    query_index = np.flatnonzero(query_good)
    query_groups = pd.Series(dep_key[query_index]).groupby(
        dep_key[query_index], sort=False).indices
    for group_key, local_index in query_groups.items():
        idx = query_index[np.asarray(local_index, dtype=np.int64)]
        q = q_time[idx]
        arrivals = arr_groups.get(str(group_key), np.array([], dtype=np.int64))
        departures = dep_groups[str(group_key)]
        arr_left = np.searchsorted(arrivals, q, side="left")
        arr_right = np.searchsorted(arrivals, q, side="right")
        dep_left = np.searchsorted(departures, q, side="left")
        dep_right = np.searchsorted(departures, q, side="right")
        for window in WINDOWS:
            minute = window // 60
            feature[f"runway_arr_past{minute}m_count"][idx] = (
                arr_left - np.searchsorted(arrivals, q - window, side="left"))
            feature[f"runway_arr_future{minute}m_count"][idx] = (
                np.searchsorted(arrivals, q + window, side="right") - arr_right)
        for scope, event, left, right in (
            ("arr", arrivals, arr_left, arr_right),
            ("dep", departures, dep_left, dep_right)):
            previous = np.full(len(idx), np.nan, dtype=np.float32)
            following = np.full(len(idx), np.nan, dtype=np.float32)
            has_previous = left > 0
            has_following = right < len(event)
            previous[has_previous] = (q[has_previous]
                                      - event[left[has_previous] - 1]).astype(np.float32)
            following[has_following] = (event[right[has_following]]
                                        - q[has_following]).astype(np.float32)
            feature[f"runway_{scope}_prev_gap_sec"][idx] = previous
            feature[f"runway_{scope}_next_gap_sec"][idx] = following

    result = pd.DataFrame({"MVT_ID_mvt": dep.MVT_ID_mvt,
                           "airport": dep.ADEP_mvt,
                           "runway": dep.RUNWAY_mvt,
                           "MVT_TIME_UTC_mvt": dep.MVT_TIME_UTC_mvt,
                           "valid_runway": query_good,
                           **feature})
    if result.MVT_ID_mvt.isna().any() or result.MVT_ID_mvt.duplicated().any():
        raise ValueError("Departure movement IDs must be unique and non-null")
    if np.any(feature["runway_dep_prev_gap_sec"][query_good]
              [np.isfinite(feature["runway_dep_prev_gap_sec"][query_good])] <= 0):
        raise ValueError("A departure query saw itself or a same-second peer")
    return result


def build_one(paths: list[Path], rows_path: Path, output: Path) -> dict:
    start = time.monotonic()
    scan = pl.scan_parquet([str(path) for path in paths])
    dep = (scan.filter(pl.col("PHASE_mvt") == "DEP")
           .select(list(RAW_DEP)).collect().to_pandas())
    arr = (scan.filter(pl.col("PHASE_mvt") == "ARR")
           .select(list(RAW_ARR)).collect().to_pandas())
    ids = pd.read_parquet(rows_path, columns=["MVT_ID_mvt"])
    if (ids.MVT_ID_mvt.isna().any() or ids.MVT_ID_mvt.duplicated().any()
            or dep.MVT_ID_mvt.isna().any() or dep.MVT_ID_mvt.duplicated().any()):
        raise ValueError("Raw and cached departure IDs must be unique")
    dep = ids.merge(dep, on="MVT_ID_mvt", how="left", sort=False,
                    validate="one_to_one")
    if len(dep) != len(ids) or dep.MVT_TIME_UTC_mvt.isna().any():
        raise ValueError("Raw departure covariates do not cover the cached rows")
    frame = build_from_raw(dep, arr)
    if not np.array_equal(frame.MVT_ID_mvt.to_numpy(), ids.MVT_ID_mvt.to_numpy()):
        raise ValueError("Feature cache moved departure IDs out of order")
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False)
    return {"rows": len(frame), "arrival_events": len(arr),
            "valid_runway_queries": int(frame.valid_runway.sum()),
            "features": FEATURES,
            "path": str(output), "seconds": round(time.monotonic() - start, 2)}


def build(args: argparse.Namespace) -> dict:
    freeze_protocol(args.output_dir / "protocol.json")
    report = {"training": build_one(
        _training_files(args.data_dir), args.cache_dir / "training_rows.parquet",
        args.output_dir / "training_runway_arrival_features.parquet")}
    report["ranking"] = build_one(
        [args.data_dir / "ranking.parquet"],
        args.cache_dir / "ranking_rows.parquet",
        args.output_dir / "ranking_runway_arrival_features.parquet")
    (args.output_dir / "feature_build.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    return report


def adjusted_slope(frame: pd.DataFrame, xcol: str,
                   strata: list[str] | None = None) -> dict:
    """Within-runway/hour/month/departure-density association, not a model fit."""
    if strata is None:
        strata = ["airport", "runway", "month", "hour", "dep_bin"]
    eligible = frame[xcol].notna() & frame.residual.notna()
    data = frame.loc[eligible].copy()
    grouping = data.groupby(strata, sort=False, observed=True)
    n = grouping[xcol].transform("size")
    dx = data[xcol] - grouping[xcol].transform("mean")
    dy = data.residual - grouping.residual.transform("mean")
    use = n.ge(20) & dx.notna() & dy.notna()
    numerator = float(np.dot(dx[use], dy[use]))
    denominator = float(np.dot(dx[use], dx[use]))
    return {"n": int(use.sum()), "strata": int(data.loc[use, strata].drop_duplicates().shape[0]),
            "slope_sec_per_unit": (numerator / denominator if denominator > 0 else None),
            "numerator": numerator, "denominator": denominator}


def day_bootstrap_slope(frame: pd.DataFrame, xcol: str,
                        repeats: int = 1000,
                        strata: list[str] | None = None) -> dict:
    """Resample UTC days after within-stratum centering for association uncertainty."""
    if strata is None:
        strata = ["airport", "runway", "month", "hour", "dep_bin"]
    grouping = frame.groupby(strata, sort=False, observed=True)
    count = grouping[xcol].transform("size")
    dx = frame[xcol] - grouping[xcol].transform("mean")
    dy = frame.residual - grouping.residual.transform("mean")
    use = count.ge(20) & dx.notna() & dy.notna()
    contribution = pd.DataFrame({
        "day": frame.loc[use, "utc_date"].to_numpy(),
        "numerator": (dx[use] * dy[use]).to_numpy(),
        "denominator": (dx[use] * dx[use]).to_numpy(),
    }).groupby("day", sort=True).sum()
    if len(contribution) < 5:
        return {"days": len(contribution), "ci95_sec_per_unit": None}
    values = contribution.to_numpy(dtype=float)
    rng = np.random.default_rng(SEED)
    indices = rng.integers(0, len(values), size=(repeats, len(values)))
    samples = values[indices].sum(axis=1)
    slope = samples[:, 0] / samples[:, 1]
    return {"days": len(contribution),
            "ci95_sec_per_unit": np.quantile(slope, [.025, .975]).tolist()}


def audit(args: argparse.Namespace) -> dict:
    freeze_protocol(args.output_dir / "protocol.json")
    v5 = pd.read_parquet(args.v5_oof,
                         columns=["MVT_ID_mvt", "target", "selected", "a_valid",
                                  "airport", "month", "MVT_TIME_UTC_mvt"])
    feat = pd.read_parquet(args.output_dir / "training_runway_arrival_features.parquet")
    if v5.MVT_ID_mvt.duplicated().any() or feat.MVT_ID_mvt.duplicated().any():
        raise ValueError("OOF and feature movement IDs must be unique")
    frame = v5.merge(feat.drop(columns=["airport", "MVT_TIME_UTC_mvt"]),
                     on="MVT_ID_mvt", how="left", validate="one_to_one")
    if len(frame) != len(v5) or frame.valid_runway.isna().any():
        raise ValueError("Runway features do not cover every v5 OOF movement ID")
    rows = pd.read_parquet(args.cache_dir / "training_rows.parquet",
                           columns=["MVT_ID_mvt"])
    dep = pd.read_parquet(args.cache_dir / "features.parquet",
                          columns=["dep_prev15", "arr_prev15"])
    if len(rows) != len(dep):
        raise ValueError("Departure-density cache alignment failed")
    rows["dep_prev15"] = dep.dep_prev15.to_numpy(dtype=float)
    rows["arr_prev15"] = dep.arr_prev15.to_numpy(dtype=float)
    frame = frame.merge(rows, on="MVT_ID_mvt", how="left", validate="one_to_one")
    if frame[["dep_prev15", "arr_prev15"]].isna().any().any():
        raise ValueError("Airport traffic-density values are missing")
    # Only the four months with frozen v5 out-of-fold predictions are audited.
    core = (frame.a_valid.to_numpy(dtype=bool)
            & frame.valid_runway.to_numpy(dtype=bool)
            & np.isfinite(frame.target.to_numpy(dtype=float))
            & np.isfinite(frame.selected.to_numpy(dtype=float)))
    data = frame.loc[core].copy()
    data["residual"] = data.target - data.selected
    utc = pd.to_datetime(data.MVT_TIME_UTC_mvt, utc=True)
    data["hour"] = utc.dt.hour
    data["month"] = utc.dt.month
    data["utc_date"] = utc.dt.date
    data["runway"] = data.runway.astype("string").str.strip().str.upper()
    data["dep_bin"] = data.groupby("airport", observed=True).dep_prev15.transform(
        lambda values: pd.qcut(values.rank(method="first"), 3,
                               labels=False)).astype("int8")
    data["arr_bin"] = data.groupby("airport", observed=True).arr_prev15.transform(
        lambda values: np.minimum(
            np.floor(values.rank(method="average", pct=True) * 3), 2)
    ).astype("int8")
    for direction in ("prev", "next"):
        gap = data[f"runway_arr_{direction}_gap_sec"].to_numpy(dtype=float)
        data[f"runway_arr_{direction}_within180s"] = (
            np.isfinite(gap) & (gap <= 180)).astype("float32")
    exposures = ("runway_arr_past15m_count", "runway_arr_future15m_count",
                 "runway_arr_prev_within180s", "runway_arr_next_within180s")
    records = {}
    for exposure in exposures:
        records[exposure] = {"overall": adjusted_slope(data, exposure),
                             "by_month": {str(month): adjusted_slope(part, exposure)
                                          for month, part in data.groupby("month", sort=True)},
                             "by_airport_month": [
                                 {"airport": str(airport), "month": int(month),
                                  **adjusted_slope(part, exposure)}
                                 for (airport, month), part in data.groupby(
                                     ["airport", "month"], sort=True)]}
    robust = data.copy()
    robust["residual"] = robust.residual.clip(-600, 600)
    past_count = "runway_arr_past15m_count"
    robust_association = {
        "definition": "valid-AOBT residual clipped to [-600,600] seconds; audit only",
        "overall": adjusted_slope(robust, past_count),
        "by_month": {str(month): {**adjusted_slope(part, past_count),
                                   **day_bootstrap_slope(part, past_count)}
                     for month, part in robust.groupby("month", sort=True)},
        "ordinary_target_0_to_7200": adjusted_slope(
            data.loc[data.target.between(0, 7200)], past_count),
    }
    # A second control holds general airport ARR traffic approximately fixed,
    # testing whether same-runway ARR flow adds information beyond total ARR.
    traffic_strata = ["airport", "runway", "month", "hour", "dep_bin", "arr_bin"]
    robust_association["with_airport_arrival_density_control"] = {
        "overall": adjusted_slope(robust, past_count, traffic_strata),
        "by_month": {str(month): {
            **adjusted_slope(part, past_count, traffic_strata),
            **day_bootstrap_slope(part, past_count, strata=traffic_strata)}
            for month, part in robust.groupby("month", sort=True)},
    }
    # Signed residuals by a fixed exposure bin, after subtracting each stratum's
    # mean. This is descriptive evidence; it is not used to fit a predictor.
    strata = ["airport", "runway", "month", "hour", "dep_bin"]
    data["adjusted_residual"] = data.residual - data.groupby(
        strata, observed=True).residual.transform("mean")
    data["arr_past15_bin"] = pd.cut(
        data.runway_arr_past15m_count,
        bins=[-.1, .5, 2.5, 5.5, np.inf],
        labels=["0", "1-2", "3-5", "6+"])
    bins = data.groupby(["month", "airport", "arr_past15_bin"],
                        observed=True).agg(
        n=("adjusted_residual", "size"),
        adjusted_bias_sec=("adjusted_residual", "mean"),
        raw_bias_sec=("residual", "mean")).reset_index()
    report = {"scope": "2025 Jan/Jul/Nov/Dec valid-AOBT v5 OOF only; no model fit",
              "all_finite_oof_n": len(v5),
              "valid_runway_valid_aobt_n": len(data),
              "controls": strata, "exposures": records,
              "past15_robust_sensitivity": robust_association,
              "arrival_count_bins": bins.to_dict("records"),
              "limits": "Associations are descriptive; unmeasured runway operations and schedule mix can confound them. Repeatedly inspected months are not an untouched final validation."}
    (args.output_dir / "association_audit.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    return {"valid_runway_valid_aobt_n": len(data),
            "month_slopes_past15": records["runway_arr_past15m_count"]["by_month"],
            "month_slopes_future15": records["runway_arr_future15m_count"]["by_month"],
            "past15_robust_sensitivity": robust_association,
            "report": str(args.output_dir / "association_audit.json")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("build", "audit"), required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--cache-dir", type=Path,
                        default=Path("artifacts/baseline"))
    parser.add_argument("--v5-oof", type=Path,
                        default=Path("artifacts/v5-ensemble/validation_predictions.parquet"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("artifacts/v6-runway-arrival"))
    args = parser.parse_args()
    result = build(args) if args.mode == "build" else audit(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
