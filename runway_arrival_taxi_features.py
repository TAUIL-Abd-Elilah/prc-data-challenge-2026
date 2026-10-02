"""Released-ARR same-runway taxi and active-ground context for departures.

This builder reads only ID, departure airport, runway and movement time from
departure rows. Arrival taxi and block values are selected after filtering to
PHASE=ARR. The complete retrospective arrival batch is released for ranking;
these features do not claim real-time availability at each departure.

Completed ARR windows are anchored to BLOCK_TIME_UTC_mvt: [t-window, t).
Active ARR means MVT_TIME_UTC_mvt < t < BLOCK_TIME_UTC_mvt. Both use only
taxi intervals with finite taxi time in [0,7200] seconds and a BLOCK-MVT
duration matching that taxi time within one second.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from runway_arrival_features import normalized
from solution import _training_files


DEP_COLUMNS = ("MVT_ID_mvt", "ADEP_mvt", "RUNWAY_mvt",
               "MVT_TIME_UTC_mvt")
ARR_COLUMNS = ("ADES_mvt", "RUNWAY_mvt", "MVT_TIME_UTC_mvt",
               "BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt")
WINDOWS_SECONDS = (900, 3600, 10800)
NS_PER_SECOND = 1_000_000_000
FEATURES = tuple(
    f"runway_arr_taxi_completed_{seconds // 60}m_{stat}"
    for seconds in WINDOWS_SECONDS for stat in ("count", "mean", "std")
) + ("runway_arr_taxiing_now_count",)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def available_memory_gib() -> float:
    if os.name == "nt":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(MemoryStatus)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise OSError("GlobalMemoryStatusEx failed")
        return status.ullAvailPhys / 2**30
    info = Path("/proc/meminfo")
    if info.exists():
        for line in info.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024 / 2**30
    raise RuntimeError("Cannot establish free physical memory")


def require_memory(min_free_gib: float) -> None:
    free = available_memory_gib()
    if free < min_free_gib:
        raise MemoryError(f"Need {min_free_gib:g} GiB free for ARR feature build; "
                          f"currently {free:.2f} GiB")


def source_paths(data_dir: Path) -> tuple[list[Path], Path]:
    training = _training_files(data_dir)
    if len(training) != 12 or len({path.name for path in training}) != 12:
        raise ValueError("Expected exactly twelve canonical 2025 training files")
    ranking = data_dir / "ranking.parquet"
    if not ranking.exists():
        raise FileNotFoundError(ranking)
    return training, ranking


def protocol_spec() -> dict:
    return {
        "purpose": "Fixed label-free released-ARR same-runway taxi feature cache",
        "raw_query": {"phase": "DEP", "columns": list(DEP_COLUMNS),
                      "forbidden": ["BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt"]},
        "raw_events": {"phase": "ARR", "columns": list(ARR_COLUMNS)},
        "group": "normalized departure ADEP + RUNWAY equals arrival ADES + RUNWAY; missing/placeholder runways excluded",
        "valid_arrival": "finite TAXITIME_SEC_mvt in [0,7200], nonnegative BLOCK-MVT, and absolute duration discrepancy <=1 second",
        "windows_seconds": list(WINDOWS_SECONDS),
        "completed_interval": "ARR BLOCK_TIME in [DEP MVT_TIME-window, DEP MVT_TIME); count/mean/population std of ARR taxi time",
        "active_interval": "ARR MVT_TIME < DEP MVT_TIME < ARR BLOCK_TIME; count of valid intervals",
        "query_missing_policy": "invalid airport/runway or movement time gives NaN for every field; valid group without events gives zero counts and NaN mean/std",
        "features": list(FEATURES),
        "ranking_availability": "retrospective batch ARR TAXITIME/BLOCK fields are released in ranking; not real-time forecasting",
        "departure_labels_used": False,
        "released_arrival_taxi_used": True,
        "validation_policy": "No model fit or label-based feature selection in this builder; February/August 2025 reserved for a separately frozen paired test",
    }


def frozen_protocol(args: argparse.Namespace) -> dict:
    """Hash every raw/cache input before allowing a feature output to be written."""
    training, ranking = source_paths(args.data_dir)
    cache_files = (args.cache_dir / "training_rows.parquet",
                   args.cache_dir / "ranking_rows.parquet")
    value = {
        "spec": protocol_spec(),
        "source_sha256": {
            "script": sha256(Path(__file__).resolve()),
            "normalization_script": sha256(
                Path(__file__).resolve().parent / "runway_arrival_features.py"),
            "raw_training": {path.name: sha256(path) for path in training},
            "raw_ranking": sha256(ranking),
            "baseline_training_rows": sha256(cache_files[0]),
            "baseline_ranking_rows": sha256(cache_files[1]),
        },
    }
    path = args.output_dir / "protocol.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise ValueError("Frozen runway ARR taxi protocol or source hash changed")
    else:
        write_json(path, value)
    return value


def phase_views(scan: pl.LazyFrame) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    """The ARR phase filter precedes taxi/block selection by construction."""
    dep = scan.filter(pl.col("PHASE_mvt") == "DEP").select(list(DEP_COLUMNS))
    arr = scan.filter(pl.col("PHASE_mvt") == "ARR").select(list(ARR_COLUMNS))
    return dep, arr


def _nanoseconds(series: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    stamp = pd.to_datetime(series, utc=True, errors="coerce")
    return (stamp.dt.as_unit("ns").astype("int64").to_numpy(),
            stamp.notna().to_numpy())


def build_from_frames(dep: pd.DataFrame, arr: pd.DataFrame) -> pd.DataFrame:
    if list(dep) != list(DEP_COLUMNS) or list(arr) != list(ARR_COLUMNS):
        raise ValueError("Raw phase frames do not match strict column allowlists")
    count = len(dep)
    feature = {name: np.full(count, np.nan, dtype=np.float32)
               for name in FEATURES}
    dep_airport, good_dep_airport = normalized(dep.ADEP_mvt)
    dep_runway, good_dep_runway = normalized(dep.RUNWAY_mvt)
    arr_airport, good_arr_airport = normalized(arr.ADES_mvt)
    arr_runway, good_arr_runway = normalized(arr.RUNWAY_mvt)
    query_time, good_query_time = _nanoseconds(dep.MVT_TIME_UTC_mvt)
    arrival_time, good_arrival_time = _nanoseconds(arr.MVT_TIME_UTC_mvt)
    block_time, good_block_time = _nanoseconds(arr.BLOCK_TIME_UTC_mvt)
    taxi = pd.to_numeric(arr.TAXITIME_SEC_mvt,
                         errors="coerce").to_numpy(dtype=float)
    valid_time = good_arrival_time & good_block_time
    duration = np.full(len(arr), np.nan, dtype=np.float64)
    duration[valid_time] = (block_time[valid_time] - arrival_time[valid_time]) / NS_PER_SECOND
    consistent = (valid_time & np.isfinite(taxi) & (taxi >= 0)
                  & (taxi <= 7200) & (duration >= 0)
                  & (np.abs(duration - taxi) <= 1))
    query_valid = good_dep_airport & good_dep_runway & good_query_time
    event_valid = good_arr_airport & good_arr_runway & consistent
    for name in FEATURES:
        if name.endswith("_count"):
            feature[name][query_valid] = 0
    query_key = np.char.add(np.char.add(dep_airport, "|"), dep_runway)
    event_key = np.char.add(np.char.add(arr_airport, "|"), arr_runway)
    query_global = np.flatnonzero(query_valid)
    event_global = np.flatnonzero(event_valid)
    query_groups = pd.Series(query_key[query_valid]).groupby(
        query_key[query_valid], sort=False).indices
    event_groups = pd.Series(event_key[event_valid]).groupby(
        event_key[event_valid], sort=False).indices
    for key, local_query in query_groups.items():
        local_event = event_groups.get(key)
        if local_event is None:
            continue
        qi = query_global[np.asarray(local_query, dtype=np.int64)]
        ei = event_global[np.asarray(local_event, dtype=np.int64)]
        block_order = np.argsort(block_time[ei], kind="stable")
        ordered = ei[block_order]
        blocks = block_time[ordered]
        values = taxi[ordered]
        cumulative = np.r_[0., np.cumsum(values, dtype=np.float64)]
        cumulative_square = np.r_[0., np.cumsum(values * values,
                                                 dtype=np.float64)]
        t = query_time[qi]
        right = np.searchsorted(blocks, t, side="left")
        for window in WINDOWS_SECONDS:
            left = np.searchsorted(blocks, t - window * NS_PER_SECOND,
                                   side="left")
            n = right - left
            total = cumulative[right] - cumulative[left]
            total_square = cumulative_square[right] - cumulative_square[left]
            mean = np.divide(total, n, out=np.full(len(qi), np.nan), where=n > 0)
            second = np.divide(total_square, n,
                               out=np.full(len(qi), np.nan), where=n > 0)
            std = np.sqrt(np.maximum(second - mean * mean, 0))
            prefix = f"runway_arr_taxi_completed_{window // 60}m_"
            feature[prefix + "count"][qi] = n.astype(np.float32)
            feature[prefix + "mean"][qi] = mean.astype(np.float32)
            feature[prefix + "std"][qi] = std.astype(np.float32)
        positive_duration = ei[arrival_time[ei] < block_time[ei]]
        landing_sorted = np.sort(arrival_time[positive_duration])
        active_blocks = np.sort(block_time[positive_duration])
        active = (np.searchsorted(landing_sorted, t, side="left")
                  - np.searchsorted(active_blocks, t, side="right"))
        if (active < 0).any():
            raise ValueError("Consistent ARR taxi intervals produced negative occupancy")
        feature["runway_arr_taxiing_now_count"][qi] = active.astype(np.float32)
    out = pd.DataFrame(feature)
    if list(out) != list(FEATURES) or np.isinf(out.to_numpy()).any():
        raise ValueError("ARR runway taxi feature schema or values invalid")
    return out


def _read_phase_frames(paths: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    scan = pl.scan_parquet([str(path) for path in paths])
    dep_scan, arr_scan = phase_views(scan)
    return dep_scan.collect().to_pandas(), arr_scan.collect().to_pandas()


def _build_one(args: argparse.Namespace, paths: list[Path],
               baseline: Path, output: Path) -> dict:
    require_memory(args.min_free_gib)
    dep, arr = _read_phase_frames(paths)
    ids = pd.read_parquet(baseline, columns=["MVT_ID_mvt"])
    if (ids.MVT_ID_mvt.isna().any() or ids.MVT_ID_mvt.duplicated().any()
            or dep.MVT_ID_mvt.isna().any() or dep.MVT_ID_mvt.duplicated().any()):
        raise ValueError("Raw and baseline departure IDs must be unique/non-null")
    if len(dep) != len(ids):
        raise ValueError("Raw departure count differs from baseline rows")
    aligned = ids.merge(dep, on="MVT_ID_mvt", how="left", sort=False,
                        validate="one_to_one")
    if (len(aligned) != len(ids)
            or not np.array_equal(aligned.MVT_ID_mvt.to_numpy(),
                                  ids.MVT_ID_mvt.to_numpy())
            or aligned.MVT_TIME_UTC_mvt.isna().any()):
        raise ValueError("Raw departure covariates do not match baseline ID/order")
    values = build_from_frames(aligned[list(DEP_COLUMNS)], arr)
    values.insert(0, "MVT_ID_mvt", ids.MVT_ID_mvt.to_numpy(copy=True))
    if (len(values) != len(ids) or not np.array_equal(values.MVT_ID_mvt,
                                                       ids.MVT_ID_mvt)):
        raise ValueError("Feature output differs from baseline ID/order")
    temp = output.with_suffix(output.suffix + ".tmp")
    values.to_parquet(temp, index=False)
    temp.replace(output)
    return {"rows": len(values), "taxi_range_arrival_events": int(
                ((pd.to_numeric(arr.TAXITIME_SEC_mvt, errors="coerce") >= 0)
                 & (pd.to_numeric(arr.TAXITIME_SEC_mvt, errors="coerce") <= 7200)).sum()),
            "feature_names": list(FEATURES),
            "nonnull_fraction": {name: float(values[name].notna().mean())
                                 for name in FEATURES},
            "output_sha256": sha256(output)}


def build(args: argparse.Namespace) -> None:
    frozen_protocol(args)
    require_memory(args.min_free_gib)
    training, ranking = source_paths(args.data_dir)
    output = args.output_dir
    if any((output / name).exists() for name in (
            "training_runway_arrival_taxi_features.parquet",
            "ranking_runway_arrival_taxi_features.parquet", "feature_build.json")):
        raise FileExistsError("ARR taxi outputs already exist; inspect their manifest before rebuilding")
    train_path = output / "training_runway_arrival_taxi_features.parquet"
    rank_path = output / "ranking_runway_arrival_taxi_features.parquet"
    train_report = _build_one(args, training,
                              args.cache_dir / "training_rows.parquet", train_path)
    rank_report = _build_one(args, [ranking],
                             args.cache_dir / "ranking_rows.parquet", rank_path)
    write_json(output / "feature_build.json", {
        "protocol_sha256": sha256(output / "protocol.json"),
        "training": train_report, "ranking": rank_report,
        "departure_labels_used": False,
        "released_arrival_taxi_used": True,
    })
    print(json.dumps({"training_rows": train_report["rows"],
                      "ranking_rows": rank_report["rows"],
                      "feature_count": len(FEATURES)}, indent=2))


def synthetic_check() -> None:
    """Small deterministic boundary/provenance check, with no file or label IO."""
    base = pd.Timestamp("2025-01-15T12:00:00Z")
    row = {"PHASE_mvt": "DEP", "MVT_ID_mvt": 1, "ADEP_mvt": "KAAA",
           "ADES_mvt": "KBBB", "RUNWAY_mvt": "09",
           "MVT_TIME_UTC_mvt": base,
           "BLOCK_TIME_UTC_mvt": base + pd.Timedelta(days=4),
           "TAXITIME_SEC_mvt": 999999.}
    raw = [row]
    raw.append({**row, "MVT_ID_mvt": 2, "RUNWAY_mvt": "?"})
    raw.append({**row, "MVT_ID_mvt": 3, "RUNWAY_mvt": "36"})

    def arrival(block_offset: int, taxi: float, *, runway="09",
                movement_offset: int | None = None) -> None:
        block = base + pd.Timedelta(seconds=block_offset)
        mvt = base + pd.Timedelta(seconds=(block_offset - taxi
                                            if movement_offset is None
                                            else movement_offset))
        raw.append({"PHASE_mvt": "ARR", "MVT_ID_mvt": 100 + len(raw),
                    "ADEP_mvt": "KCCC", "ADES_mvt": "KAAA",
                    "RUNWAY_mvt": runway, "MVT_TIME_UTC_mvt": mvt,
                    "BLOCK_TIME_UTC_mvt": block,
                    "TAXITIME_SEC_mvt": taxi})

    arrival(-900, 300)        # 15-minute lower bound included
    arrival(-300, 1200)
    arrival(-901, 200)        # 15-minute lower bound minus 1 sec excluded
    arrival(-10800, 60)      # 180-minute lower bound included
    arrival(-1000, 0)        # zero taxi belongs in completed windows
    arrival(0, 600)          # completion exactly at query excluded
    arrival(0, 0)            # zero-duration arrival at query is not active
    arrival(600, 720)        # active at query
    arrival(100, 100)        # landing exactly at query: inactive
    arrival(-200, 500, movement_offset=-300)  # bad duration: excluded
    arrival(-100, 400, runway="27")            # different runway
    arrival(-100, 7201)      # taxi above valid bound: excluded
    mixed = pl.from_pandas(pd.DataFrame(raw)).lazy()
    dep_scan, arr_scan = phase_views(mixed)
    dep, arr = dep_scan.collect().to_pandas(), arr_scan.collect().to_pandas()
    if (list(dep) != list(DEP_COLUMNS) or list(arr) != list(ARR_COLUMNS)
            or "TAXITIME_SEC_mvt" in dep or "BLOCK_TIME_UTC_mvt" in dep
            or len(dep) != 3 or len(arr) != 12):
        raise AssertionError("Phase-specific raw column provenance failed")
    out = build_from_frames(dep, arr)
    observed = out.iloc[0]
    for minutes, expected in ((15, [300., 1200.]),
                              (60, [300., 1200., 200., 0.]),
                              (180, [300., 1200., 200., 60., 0.])):
        prefix = f"runway_arr_taxi_completed_{minutes}m_"
        if not (observed[prefix + "count"] == len(expected)
                and np.isclose(observed[prefix + "mean"], np.mean(expected))
                and np.isclose(observed[prefix + "std"], np.std(expected))):
            raise AssertionError(f"Block-anchored {minutes}m boundary failed")
    if (observed["runway_arr_taxiing_now_count"] != 1
            or not out.iloc[1].isna().all()
            or out.iloc[2][[name for name in FEATURES
                              if name.endswith("_count")]].ne(0).any()
            or out.iloc[2][[name for name in FEATURES
                              if not name.endswith("_count")]].notna().any()):
        raise AssertionError("Active occupancy or missing-runway policy failed")
    print(json.dumps({"synthetic_check": "passed", "features": len(FEATURES),
                      "phase_split": {"DEP": len(dep), "ARR": len(arr)}}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("synthetic", "prepare", "build"),
                        default="synthetic")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--cache-dir", type=Path,
                        default=Path("artifacts/baseline"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("artifacts/v10-runway-arrival-taxi"))
    parser.add_argument("--min-free-gib", type=float, default=4.0)
    args = parser.parse_args()
    if args.mode == "synthetic":
        synthetic_check()
    elif args.mode == "prepare":
        value = frozen_protocol(args)
        print(json.dumps({"protocol": str(args.output_dir / "protocol.json"),
                          "source_count": len(value["source_sha256"]["raw_training"]) + 1}))
    else:
        build(args)


if __name__ == "__main__":
    main()
