"""Independent, label-free spot audit for the frozen v14 training event bank.

`plan` and `synthetic` access no competition rows. Run `audit` only after the
published training cache build completes. This oracle deliberately does not use
the builder's event selection or encoding functions.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pyarrow.parquet as pq

from solution import _training_files


ROOT = Path(__file__).resolve().parent
FEATURE_DIR = ROOT / "artifacts/v14-event-sequence/features"
BASELINE = ROOT / "artifacts/baseline/training_rows.parquet"
DATA_DIR = ROOT / "data"
RESULT = ROOT / "artifacts/v14-event-sequence/oracle_training_audit.json"
ROWS = 2_085_047
WINDOW_NS = 3_600_000_000_000
MAX_PROXY_NS = 7_200_000_000_000
SAMPLE_SIZE = 24
PEAK_OBSERVED_RSS_BYTES = 0
PLACEHOLDERS = {"", "?", "-", "NA", "N/A", "NAN", "NONE", "NULL", "UNKNOWN", "\\N"}
DEP = ("MVT_ID_mvt", "FLIGHT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "RUNWAY_mvt",
       "MVT_TIME_UTC_mvt", "AOBT_3_flt")
ARR = ("MVT_ID_mvt", "FLIGHT_ID_mvt", "PHASE_mvt", "ADES_mvt", "RUNWAY_mvt",
       "MVT_TIME_UTC_mvt")
BASELINE_COLUMNS = ("MVT_ID_mvt", "airport", "time", "month")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rss_bytes() -> int:
    import psutil
    return psutil.Process().memory_info().rss


def check_rss() -> None:
    global PEAK_OBSERVED_RSS_BYTES
    PEAK_OBSERVED_RSS_BYTES = max(PEAK_OBSERVED_RSS_BYTES, rss_bytes())
    if PEAK_OBSERVED_RSS_BYTES >= 2 * 1024**3:
        raise MemoryError("Independent oracle exceeded the 2 GiB RSS budget")


def norm(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    token = str(value).strip().upper()
    return "" if token in PLACEHOLDERS else token


def flight_key(value: Any) -> tuple[str, Any] | None:
    if value is None or pd.isna(value) or isinstance(value, (bool, np.bool_)):
        return None
    if isinstance(value, (int, np.integer)):
        return ("numeric", int(value))
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number) or not number.is_integer() or abs(number) >= 2**53:
            return None
        return ("numeric", int(number))
    token = str(value).strip()
    return None if not norm(token) else ("string", token)


def ns(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None
    try:
        stamp = pd.Timestamp(value)
        if stamp.tzinfo is None:
            stamp = stamp.tz_localize("UTC")
        else:
            stamp = stamp.tz_convert("UTC")
        number = int(stamp.value)
    except (ValueError, OverflowError, TypeError):
        return None
    limit = max(WINDOW_NS, MAX_PROXY_NS)
    return number if np.iinfo(np.int64).min + limit <= number <= np.iinfo(np.int64).max - limit else None


def month_key(time_ns: int | None) -> tuple[int, int] | None:
    if time_ns is None:
        return None
    stamp = pd.Timestamp(time_ns, unit="ns", tz="UTC")
    return stamp.year, stamp.month


def proxy(aobt: Any, takeoff_ns: int | None) -> tuple[bool, int | None, float]:
    start = ns(aobt)
    if start is None or takeoff_ns is None:
        return False, None, 0.0
    difference = takeoff_ns - start
    return ((True, start, float(np.float32(difference / MAX_PROXY_NS)))
            if 0 < difference <= MAX_PROXY_NS else (False, None, 0.0))


def event_rows(query: dict, dep: pd.DataFrame, arr: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict]:
    """Brute-force oracle: filter every peer, sort by distance, then encode."""
    values = np.zeros((32, 6), dtype=np.float16)
    mask = np.zeros(32, dtype=np.uint8)
    qtime = ns(query["MVT_TIME_UTC_mvt"])
    airport = norm(query["ADEP_mvt"])
    if qtime is None or not airport:
        return values, mask, {"past": 0, "future": 0, "tied_times": 0,
                              "same_flight_excluded": 0, "month_excluded": 0}
    month = month_key(qtime)
    qrunway = norm(query["RUNWAY_mvt"])
    qflight = flight_key(query["FLIGHT_ID_mvt"])
    qid = query["MVT_ID_mvt"]
    before, after = [], []
    tied_times = same_flight = month_excluded = 0
    for frame, phase, airport_column in ((dep, "DEP", "ADEP_mvt"),
                                          (arr, "ARR", "ADES_mvt")):
        for peer in frame.to_dict("records"):
            if norm(peer[airport_column]) != airport:
                continue
            ptime = ns(peer["MVT_TIME_UTC_mvt"])
            if ptime is None:
                continue
            if month_key(ptime) != month:
                month_excluded += 1
                continue
            delta = ptime - qtime
            if delta == 0:
                tied_times += 1
                continue
            if not (-WINDOW_NS <= delta < 0 or 0 < delta <= WINDOW_NS):
                continue
            if peer["MVT_ID_mvt"] == qid:
                continue
            pflight = flight_key(peer["FLIGHT_ID_mvt"])
            if qflight is not None and pflight == qflight:
                same_flight += 1
                continue
            runway = norm(peer["RUNWAY_mvt"])
            valid, aobt, interval = (proxy(peer["AOBT_3_flt"], ptime)
                                     if phase == "DEP" else (False, None, 0.0))
            tie = (0 if phase == "ARR" else 1, runway,
                   aobt if valid else np.iinfo(np.int64).max)
            record = (ptime, tie, phase, runway, valid, aobt, interval)
            (before if delta < 0 else after).append(record)
    before = sorted(sorted(before, key=lambda x: (-x[0], x[1]))[:16],
                    key=lambda x: (x[0], x[1]))
    after = sorted(after, key=lambda x: (x[0], x[1]))[:16]
    positions = list(range(16 - len(before), 16)) + list(range(16, 16 + len(after)))
    for slot, (ptime, _, phase, runway, valid, aobt, interval) in zip(
            positions, before + after):
        relation = -1 if not qrunway or not runway else (1 if runway == qrunway else 0)
        channels = ((ptime - qtime) / WINDOW_NS,
                    -1 if phase == "ARR" else 1,
                    relation,
                    float(np.clip((aobt - qtime) / WINDOW_NS, -1, 1)) if valid else 0.0,
                    1.0 if valid else 0.0,
                    interval if valid else 0.0)
        values[slot] = np.asarray(channels, dtype=np.float16)
        mask[slot] = 1
    return values, mask, {"past": len(before), "future": len(after),
                          "tied_times": tied_times, "same_flight_excluded": same_flight,
                          "month_excluded": month_excluded}


def candidates(cache_presence: np.memmap, cache_ids: np.ndarray) -> list[dict]:
    """Read only four label-free baseline columns in bounded Parquet batches."""
    parquet = pq.ParquetFile(BASELINE)
    if parquet.metadata.num_rows != ROWS:
        raise ValueError("Baseline training row count changed")
    chosen: dict[tuple[int, str, str], list[dict]] = defaultdict(list)
    offset = 0
    for batch in parquet.iter_batches(batch_size=16384, columns=list(BASELINE_COLUMNS)):
        part = batch.to_pandas()
        stop = offset + len(part)
        if not np.array_equal(part.MVT_ID_mvt.to_numpy(), cache_ids[offset:stop]):
            raise ValueError("Saved event IDs differ from baseline order")
        dates = pd.to_datetime(part.time, utc=True, errors="coerce")
        if dates.isna().any():
            raise ValueError("Baseline query time contains NaT")
        start = dates.dt.to_period("M").dt.start_time.dt.tz_localize("UTC")
        end = start + pd.offsets.MonthBegin(1)
        edge = ((dates - start).dt.total_seconds().to_numpy() < 3600) | (
            (end - dates).dt.total_seconds().to_numpy() <= 3600)
        empty = np.asarray(cache_presence[offset:stop]).sum(axis=1) == 0
        month = part.month.to_numpy(dtype=int)
        airport = part.airport.astype("string").fillna("").str.strip().str.upper().to_numpy()
        for local in range(len(part)):
            key_base = (int(month[local]), str(airport[local]))
            item = {"row": offset + local, "id": part.MVT_ID_mvt.iloc[local],
                    "airport": str(airport[local]), "time": dates.iloc[local],
                    "month": int(month[local])}
            for category, include in (("general", True), ("boundary", bool(edge[local])),
                                      ("empty", bool(empty[local]))):
                if include:
                    slot = chosen[key_base + (category,)]
                    if len(slot) < 2:
                        slot.append(item)
        offset = stop
        check_rss()
    if offset != ROWS:
        raise ValueError("Baseline iterator ended before the saved event bank")
    pool = [(category, item) for (_, _, category), items in chosen.items() for item in items]
    selected: list[dict] = []
    used = set()
    for category, quota in (("empty", 4), ("boundary", 4)):
        options = sorted((item for kind, item in pool if kind == category),
                         key=lambda x: (x["month"], x["airport"], x["row"]))
        for item in options:
            if len([x for x in selected if x["category"] == category]) >= quota:
                break
            if item["row"] not in used:
                selected.append({**item, "category": category})
                used.add(item["row"])
    options = sorted((item for _, item in pool), key=lambda x: x["row"])
    while len(selected) < SAMPLE_SIZE:
        remaining = [item for item in options if item["row"] not in used]
        if not remaining:
            break
        months = {item["month"] for item in selected}
        airports = {item["airport"] for item in selected}
        item = max(remaining, key=lambda x: (int(x["month"] not in months),
                                             int(x["airport"] not in airports),
                                             -x["row"]))
        selected.append({**item, "category": "general"})
        used.add(item["row"])
    if len(selected) != SAMPLE_SIZE:
        raise ValueError("Could not choose 24 distinct sealed training queries")
    return selected


def month_airport_peers(year: int, month: int, airport: str,
                        raw_paths: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = pl.scan_parquet([str(path) for path in raw_paths])
    month_predicate = ((pl.col("MVT_TIME_UTC_mvt").dt.year() == year)
                       & (pl.col("MVT_TIME_UTC_mvt").dt.month() == month))
    dep = (source.filter(pl.col("PHASE_mvt") == "DEP")
           .filter(month_predicate)
           .filter(pl.col("ADEP_mvt").cast(pl.String).str.strip_chars().str.to_uppercase()
                   == airport)
           .select(list(DEP)).collect().to_pandas())
    arr = (source.filter(pl.col("PHASE_mvt") == "ARR")
           .filter(month_predicate)
           .filter(pl.col("ADES_mvt").cast(pl.String).str.strip_chars().str.to_uppercase()
                   == airport)
           .select(list(ARR)).collect().to_pandas())
    check_rss()
    return dep, arr


def synthetic() -> dict:
    q = {"MVT_ID_mvt": 1, "FLIGHT_ID_mvt": 7, "ADEP_mvt": "AAAA",
         "RUNWAY_mvt": " 10 ", "MVT_TIME_UTC_mvt": pd.Timestamp("2025-01-01T00:00:00Z")}
    dep = pd.DataFrame([
        {**q, "MVT_ID_mvt": 2, "FLIGHT_ID_mvt": 8,
         "MVT_TIME_UTC_mvt": pd.Timestamp("2024-12-31T23:00:00Z"),
         "AOBT_3_flt": pd.Timestamp("2024-12-31T22:59:00Z")},
        {**q, "MVT_ID_mvt": 3, "FLIGHT_ID_mvt": 7,
         "MVT_TIME_UTC_mvt": pd.Timestamp("2025-01-01T00:00:01Z"),
         "AOBT_3_flt": pd.Timestamp("2025-01-01T00:00:00Z")},
        {**q, "MVT_ID_mvt": 4, "FLIGHT_ID_mvt": 8,
         "MVT_TIME_UTC_mvt": pd.Timestamp("2025-01-01T00:00:02Z"),
         "AOBT_3_flt": pd.Timestamp("2025-01-01T00:00:00Z")},
        {**q, "MVT_ID_mvt": 5, "FLIGHT_ID_mvt": 8,
         "MVT_TIME_UTC_mvt": pd.Timestamp("2025-01-01T00:00:00Z"),
         "AOBT_3_flt": pd.Timestamp("2024-12-31T23:59:00Z")},
    ])
    arr = pd.DataFrame(columns=list(ARR))
    matrix, mask, info = event_rows(q, dep, arr)
    if (info["future"] != 1 or info["past"] != 0 or info["same_flight_excluded"] != 1
            or info["tied_times"] != 1 or info["month_excluded"] != 1
            or mask.sum() != 1 or mask[16] != 1 or matrix[16, 0] <= 0
            or matrix[16, 4] != 1 or np.any(matrix[mask == 0] != 0)):
        raise AssertionError("Independent synthetic boundary/identity/tie oracle failed")
    anchor = pd.Timestamp("2025-01-15T12:00:00Z")
    mid = {**q, "MVT_ID_mvt": 100, "FLIGHT_ID_mvt": 100,
           "MVT_TIME_UTC_mvt": anchor}
    dense = []
    for minute in range(1, 18):
        dense.append({**mid, "MVT_ID_mvt": 100 + minute,
                      "FLIGHT_ID_mvt": 200 + minute,
                      "MVT_TIME_UTC_mvt": anchor - pd.Timedelta(minutes=minute),
                      "AOBT_3_flt": anchor - pd.Timedelta(minutes=minute + 2)})
    dense.extend([
        {**mid, "MVT_ID_mvt": 200, "FLIGHT_ID_mvt": 300,
         "MVT_TIME_UTC_mvt": anchor + pd.Timedelta(hours=1),
         "AOBT_3_flt": anchor + pd.Timedelta(minutes=58)},
        {**mid, "MVT_ID_mvt": 201, "FLIGHT_ID_mvt": 301,
         "MVT_TIME_UTC_mvt": anchor + pd.Timedelta(hours=1, nanoseconds=1),
         "AOBT_3_flt": anchor + pd.Timedelta(minutes=58)},
        {**mid, "MVT_ID_mvt": 202, "FLIGHT_ID_mvt": 302,
         "MVT_TIME_UTC_mvt": anchor + pd.Timedelta(seconds=1),
         "AOBT_3_flt": anchor - pd.Timedelta(minutes=1)},
    ])
    tied_arrival = {"MVT_ID_mvt": 203, "FLIGHT_ID_mvt": 303,
                    "PHASE_mvt": "ARR", "ADES_mvt": "AAAA", "RUNWAY_mvt": "10",
                    "MVT_TIME_UTC_mvt": anchor + pd.Timedelta(seconds=1)}
    dense_matrix, dense_mask, dense_info = event_rows(mid, pd.DataFrame(dense),
                                                        pd.DataFrame([tied_arrival]))
    if (dense_info["past"] != 16 or dense_info["future"] != 3
            or dense_mask.sum() != 19
            or dense_matrix[0, 0] != np.float16(-16 / 60)
            or dense_matrix[16, 1] != -1  # ARR before DEP at a tied peer time.
            or dense_matrix[17, 1] != 1
            or dense_matrix[18, 0] != 1  # Exact +3600-second endpoint included.
            or np.any(dense_matrix[dense_mask == 0] != 0)):
        raise AssertionError("Independent closest-sixteen/tie/window oracle failed")
    return {"passed": True, "real_rows_read": False,
            "month_boundary": True, "same_flight": True, "tie_exclusion": True,
            "closest_sixteen": True, "arrival_before_departure_tie": True,
            "strict_future_window": True,
            "float16_mask_and_padding": True}


def audit() -> dict:
    import event_sequence_features_v14 as builder
    if RESULT.exists():
        raise FileExistsError("Oracle result is immutable")
    own_before = sha256(Path(__file__))
    receipt = builder.verify(scope="training", directory=FEATURE_DIR)
    ids = np.load(FEATURE_DIR / "training_ids.npy", mmap_mode="r")
    presence = np.memmap(FEATURE_DIR / "training_presence.u8.memmap", dtype=np.uint8,
                         mode="r", shape=(ROWS, 32))
    events = np.memmap(FEATURE_DIR / "training_events.f16.memmap", dtype=np.float16,
                       mode="r", shape=(ROWS, 32, 6))
    sample = candidates(presence, ids)
    raw_paths = _training_files(DATA_DIR)
    groups: dict[tuple[int, int, str], list[dict]] = defaultdict(list)
    for item in sample:
        stamp = item["time"]
        groups[(stamp.year, stamp.month, item["airport"])].append(item)
    mismatch = 0
    channel_mismatch = np.zeros(6, dtype=np.int64)
    totals = defaultdict(int)
    for (year, month, airport), queries in sorted(groups.items()):
        dep, arr = month_airport_peers(year, month, airport, raw_paths)
        if dep.MVT_ID_mvt.isna().any() or dep.MVT_ID_mvt.duplicated().any():
            raise ValueError("Raw DEP query IDs are not unique in an audited group")
        by_id = dep.set_index("MVT_ID_mvt", drop=False)
        for item in queries:
            if item["id"] not in by_id.index:
                raise ValueError("Sampled baseline query absent from raw DEP projection")
            query = by_id.loc[item["id"]].to_dict()
            if ns(query["MVT_TIME_UTC_mvt"]) != ns(item["time"]):
                raise ValueError("Sampled query UTC time differs from baseline")
            expected, expected_mask, detail = event_rows(query, dep, arr)
            row = item["row"]
            saved, saved_mask = np.asarray(events[row]), np.asarray(presence[row])
            different = expected.view(np.uint16) != saved.view(np.uint16)
            bad = bool(np.any(different) or not np.array_equal(expected_mask, saved_mask))
            mismatch += int(bad)
            channel_mismatch += different.sum(axis=0)
            totals["past_events"] += detail["past"]
            totals["future_events"] += detail["future"]
            totals["same_flight_excluded"] += detail["same_flight_excluded"]
            totals["same_time_excluded"] += detail["tied_times"]
            totals["month_excluded"] += detail["month_excluded"]
        check_rss()
    if builder.verify(scope="training", directory=FEATURE_DIR) != receipt:
        raise ValueError("Training event cache receipt changed during independent audit")
    if sha256(Path(__file__)) != own_before:
        raise ValueError("Oracle source changed during audit")
    report = {"schema_version": 1, "passed": mismatch == 0,
              "sample_rows": len(sample), "months": sorted({x["month"] for x in sample}),
              "airport_count": len({x["airport"] for x in sample}),
              "empty_sample_rows": sum(x["category"] == "empty" for x in sample),
              "boundary_sample_rows": sum(x["category"] == "boundary" for x in sample),
              "mismatched_rows": mismatch,
              "mismatched_float16_values_by_channel": channel_mismatch.tolist(),
              "aggregate_selection_evidence": dict(totals),
              "oracle_source_sha256": own_before,
              "builder_receipt_sha256": sha256(FEATURE_DIR / "training_build.json"),
              "event_bank_sha256": sha256(FEATURE_DIR / "training_events.f16.memmap"),
              "presence_sha256": sha256(FEATURE_DIR / "training_presence.u8.memmap"),
              "ids_sha256": sha256(FEATURE_DIR / "training_ids.npy"),
              "departure_labels_read": False, "ranking_values_read": False,
              "peak_observed_rss_bytes": max(PEAK_OBSERVED_RSS_BYTES, rss_bytes())}
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    with RESULT.open("x", encoding="utf-8") as output:
        json.dump(report, output, indent=2)
        output.write("\n")
    if mismatch:
        raise AssertionError("Independent raw event oracle found saved cache mismatches")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("plan", "synthetic", "audit"))
    args = parser.parse_args()
    if args.mode == "plan":
        result = {"status": "code_only", "sample_rows": SAMPLE_SIZE,
                  "source_columns": {"DEP": list(DEP), "ARR": list(ARR),
                                     "baseline": list(BASELINE_COLUMNS)},
                  "oracle_uses_builder_selection": False,
                  "audit_requires_completed_cache": True,
                  "real_rows_read": False}
    elif args.mode == "synthetic":
        result = synthetic()
    else:
        result = audit()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
