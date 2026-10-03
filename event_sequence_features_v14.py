"""Prospective, label-free v14 event sequence cache.

Real preparation/builds require the published source SHA. The default build is
training only. Ranking preparation and build require the separately verified
v14 trainer terminal and a frozen ranking input seal. No model is fitted here.
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from solution import _training_files


ROOT = Path(__file__).resolve().parent
SPEC = ROOT / "reports/event_sequence_missing_spec_v14.json"
DATA_DIR = ROOT / "data"
CACHE_DIR = ROOT / "artifacts/baseline"
MOVEMENT_DIR = ROOT / "artifacts/v6-movement-only"
DEFAULT_DIR = ROOT / "artifacts/v14-event-sequence/features"
EVENT_SHAPE = (32, 6)
PAST_SLOTS = 16
FUTURE_SLOTS = 16
WINDOW_NS = 3_600_000_000_000
PROXY_MAX_NS = 7_200_000_000_000
EXPECTED_ROWS = {"training": 2_085_047, "ranking": 344_841}
SPEC_SHA256 = "26782bcd1c46662bb705525fa5a72c4d9eb1d57616bbca1ee42edcff1a50aabb"
MOVEMENT_SOURCE_SHA256 = "bacd7a8c9446ea740969f3eeb1f59d4839eb83a8b6a348c7d3f07a8bf7f5ea26"
MOVEMENT_MANIFEST_SHA256 = "1066c5e1c47402a4858381055522f330d09dc9274da298277f40f4f8d6084825"
MOVEMENT_FEATURES_SHA256 = "454c9c8c9220564bd9b208c666143056725541683e05da774a479f55e6083a5f"
PLACEHOLDERS = frozenset(("", "?", "-", "NA", "N/A", "NAN", "NONE", "NULL", "UNKNOWN", "\\N"))

DEP_COLUMNS = ("MVT_ID_mvt", "FLIGHT_ID_mvt", "PHASE_mvt", "ADEP_mvt",
               "RUNWAY_mvt", "MVT_TIME_UTC_mvt", "AOBT_3_flt")
QUERY_COLUMNS = DEP_COLUMNS[:-1]
ARR_COLUMNS = ("MVT_ID_mvt", "FLIGHT_ID_mvt", "PHASE_mvt", "ADES_mvt",
               "RUNWAY_mvt", "MVT_TIME_UTC_mvt")
FORBIDDEN = frozenset(("BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt", "target"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def available_memory_gib() -> float:
    if os.name == "nt":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
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
    if Path("/proc/meminfo").exists():
        for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024 / 2**30
    raise RuntimeError("Available physical memory cannot be established")


def require_memory() -> None:
    available = available_memory_gib()
    if available < 10.0:
        raise MemoryError(f"v14 real build requires >=10 GiB free; found {available:.2f} GiB")


def source_sha_check(published_source_sha256: str | None) -> str:
    actual = sha256(Path(__file__).resolve())
    if published_source_sha256 is None or actual != published_source_sha256.lower():
        raise ValueError("Published builder SHA256 is required and must match exactly")
    if sha256(SPEC) != SPEC_SHA256:
        raise ValueError("Published v14 specification SHA256 changed")
    return actual


def canonical_path(path: Path) -> str:
    return str(Path(path).resolve())


def _paths(scope: str, data_dir: Path, cache_dir: Path, movement_dir: Path,
           trainer_terminal: Path | None = None) -> dict[str, Path]:
    if scope not in EXPECTED_ROWS:
        raise ValueError("scope must be training or ranking")
    paths: dict[str, Path] = {
        "builder_source": Path(__file__).resolve(),
        "training_file_selector_source": ROOT / "solution.py",
        "movement_feature_source": ROOT / "movement_only_expert.py",
        "published_spec": SPEC,
        "baseline_rows": cache_dir / ("training_rows.parquet" if scope == "training"
                                      else "ranking_rows.parquet"),
        "baseline_features": cache_dir / ("features.parquet" if scope == "training"
                                          else "ranking_features.parquet"),
        "movement_feature_manifest": movement_dir / "features_manifest.json",
        "movement_prepared_integrity": movement_dir / "prepared_integrity.json",
        "movement_training_features": movement_dir / "features.parquet",
        "movement_training_row_ids": movement_dir / "row_ids.parquet",
    }
    if scope == "training":
        for index, raw in enumerate(_training_files(data_dir), start=1):
            paths[f"raw_training_{index:02d}"] = Path(raw)
    else:
        paths["raw_ranking"] = data_dir / "ranking.parquet"
        if trainer_terminal is None:
            raise ValueError("Ranking inventory requires a verified trainer terminal path")
        paths["trainer_terminal"] = trainer_terminal
    return paths


def source_inventory(scope: str = "training", data_dir: Path = DATA_DIR,
                     cache_dir: Path = CACHE_DIR, movement_dir: Path = MOVEMENT_DIR,
                     directory: Path = DEFAULT_DIR,
                     trainer_terminal: Path | None = None) -> dict[str, Any]:
    """Hash-only inventory; it never reads Parquet row values."""
    del directory
    paths = _paths(scope, Path(data_dir), Path(cache_dir), Path(movement_dir),
                   trainer_terminal)
    inventory = {name: {"path": canonical_path(path), "sha256": sha256(path)}
                 for name, path in paths.items()}
    if inventory["movement_feature_source"]["sha256"] != MOVEMENT_SOURCE_SHA256:
        raise ValueError("Published movement feature source changed")
    if inventory["movement_feature_manifest"]["sha256"] != MOVEMENT_MANIFEST_SHA256:
        raise ValueError("Published 79-field movement manifest changed")
    if inventory["movement_training_features"]["sha256"] != MOVEMENT_FEATURES_SHA256:
        raise ValueError("Exact prepared movement features changed")
    manifest = json.loads(paths["movement_feature_manifest"].read_text(encoding="utf-8"))
    integrity = json.loads(paths["movement_prepared_integrity"].read_text(encoding="utf-8"))
    if (manifest.get("rows") != EXPECTED_ROWS["training"]
            or len(manifest.get("features", [])) != 79
            or len(manifest.get("categorical", [])) != 13
            or integrity.get("method") != "independent_rebuild_exact"
            or integrity.get("rows") != EXPECTED_ROWS["training"]
            or integrity.get("original_manifest_sha256") != MOVEMENT_MANIFEST_SHA256
            or integrity.get("prepared_features_sha256") != MOVEMENT_FEATURES_SHA256
            or integrity.get("row_ids_sha256")
            != inventory["movement_training_row_ids"]["sha256"]
            or integrity.get("feature_names") != manifest.get("features")
            or integrity.get("categorical_names") != manifest.get("categorical")
            or not all(integrity.get("comparison", {}).get(key) is True for key in (
                "schema_equal", "categorical_levels_equal", "feature_values_equal",
                "row_id_order_equal", "manifest_equal"))):
        raise ValueError("Movement prepared-cache provenance is inconsistent")
    if scope == "training" and (
            manifest.get("baseline_rows_sha256") != inventory["baseline_rows"]["sha256"]
            or manifest.get("baseline_features_sha256") != inventory["baseline_features"]["sha256"]):
        raise ValueError("Movement source baseline no longer matches the label-free cache inputs")
    if (sha256(paths["movement_feature_manifest"]) != MOVEMENT_MANIFEST_SHA256
            or sha256(paths["movement_prepared_integrity"])
            != inventory["movement_prepared_integrity"]["sha256"]):
        raise ValueError("Movement provenance changed during inspection")
    return inventory


def _rank_authorization(directory: Path, terminal_path: Path | None) -> tuple[dict, Path]:
    """Lazy import prevents a feature-builder/trainer import cycle."""
    if terminal_path is None:
        raise ValueError("Ranking requires --trainer-terminal")
    expected = ROOT / "artifacts/v14-event-sequence/terminal.json"
    if Path(terminal_path).resolve() != expected.resolve():
        raise ValueError("Ranking terminal must be the canonical v14 terminal")
    try:
        import v14_event_sequence_expert as trainer
    except ImportError as exc:
        raise RuntimeError("Published v14 trainer verifier is unavailable") from exc
    terminal = trainer.require_terminal(
        output_dir=expected.parent, feature_dir=Path(directory),
        movement_dir=MOVEMENT_DIR, cache_dir=CACHE_DIR)
    if not isinstance(terminal, dict) or terminal.get("passed") is not True:
        raise ValueError("The v14 trainer terminal has not passed every local gate")
    return terminal, expected


def protocol(scope: str = "training", directory: Path = DEFAULT_DIR,
             trainer_terminal: Path | None = None) -> dict[str, Any]:
    if sha256(SPEC) != SPEC_SHA256:
        raise ValueError("Published v14 specification changed")
    inventory = source_inventory(scope, directory=directory,
                                 trainer_terminal=trainer_terminal)
    value: dict[str, Any] = {
        "schema_version": 1,
        "scope": scope,
        "purpose": "label-free OTHER-movement event sequence; no departure target or block input",
        "spec_sha256": SPEC_SHA256,
        "inputs": inventory,
        "rows": EXPECTED_ROWS[scope],
        "event_shape_per_row": list(EVENT_SHAPE),
        "event_dtype": "float16",
        "presence_dtype": "uint8",
        "event_columns": [x["name"] for x in json.loads(SPEC.read_text(encoding="utf-8"))
                          ["event_builder"]["six_float16_channels_in_order"]],
        "query_columns": list(QUERY_COLUMNS),
        "peer_dep_columns": list(DEP_COLUMNS),
        "peer_arr_columns": list(ARR_COLUMNS),
        "raw_phase_filter_before_projection": True,
        "forbidden_raw_columns": sorted(FORBIDDEN),
        "month_partition": "UTC calendar year/month and normalized airport",
        "window_ns": WINDOW_NS,
        "past_slots": PAST_SLOTS,
        "future_slots": FUTURE_SLOTS,
        "ids_never_predictors": True,
    }
    if scope == "ranking":
        terminal, path = _rank_authorization(directory, trainer_terminal)
        value["verified_trainer_terminal_sha256"] = sha256(path)
        value["verified_trainer_terminal_status"] = bool(terminal["passed"])
        seal_path = Path(directory) / "ranking_inputs.json"
        if not seal_path.exists():
            raise FileNotFoundError("Ranking input seal must precede ranking protocol")
        expected_seal = {"schema_version": 1, "scope": "ranking",
                         "inputs": inventory,
                         "verified_trainer_terminal_sha256": sha256(path)}
        if json.loads(seal_path.read_text(encoding="utf-8")) != expected_seal:
            raise ValueError("Ranking input seal/source bytes changed")
        value["ranking_input_seal_sha256"] = sha256(seal_path)
    return value


def _protocol_path(scope: str, directory: Path) -> Path:
    return Path(directory) / f"{scope}_protocol.json"


def _output_paths(scope: str, directory: Path) -> dict[str, Path]:
    base = Path(directory)
    return {
        "events": base / f"{scope}_events.f16.memmap",
        "presence": base / f"{scope}_presence.u8.memmap",
        "ids": base / f"{scope}_ids.npy",
        "report": base / f"{scope}_build.json",
    }


def _stage_path(directory: Path, suffix: str) -> Path:
    Path(directory).mkdir(parents=True, exist_ok=True)
    fd, value = tempfile.mkstemp(prefix=".v14-stage-", suffix=suffix, dir=directory)
    os.close(fd)
    return Path(value)


def _exclusive_publish(staged: Path, final: Path) -> None:
    try:
        os.link(staged, final)
    finally:
        staged.unlink(missing_ok=True)


def _write_json_exclusive(path: Path, value: dict) -> None:
    staged = _stage_path(path.parent, ".json")
    try:
        with staged.open("w", encoding="utf-8", newline="\n") as target:
            json.dump(value, target, indent=2, sort_keys=True)
            target.write("\n")
            target.flush()
            os.fsync(target.fileno())
        _exclusive_publish(staged, path)
    finally:
        staged.unlink(missing_ok=True)


def prepare(scope: str = "training", directory: Path = DEFAULT_DIR,
            trainer_terminal: Path | None = None) -> dict[str, Any]:
    """Freeze hashes before any raw/cache value reads."""
    require_memory()
    directory = Path(directory)
    if scope == "ranking":
        _, terminal_path = _rank_authorization(directory, trainer_terminal)
        inventory = source_inventory(scope, directory=directory,
                                     trainer_terminal=terminal_path)
        seal = {"schema_version": 1, "scope": "ranking", "inputs": inventory,
                "verified_trainer_terminal_sha256": sha256(terminal_path)}
        seal_path = directory / "ranking_inputs.json"
        if seal_path.exists():
            if json.loads(seal_path.read_text(encoding="utf-8")) != seal:
                raise ValueError("Existing ranking input seal changed")
        else:
            _write_json_exclusive(seal_path, seal)
    planned = protocol(scope, directory, trainer_terminal)
    path = _protocol_path(scope, directory)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != planned:
            raise ValueError("Immutable v14 input protocol/source hashes changed")
        return existing
    _write_json_exclusive(path, planned)
    return planned


def _verify_protocol(scope: str, directory: Path,
                     trainer_terminal: Path | None) -> tuple[dict, str]:
    path = _protocol_path(scope, directory)
    if not path.exists():
        raise FileNotFoundError(f"Missing published input protocol: {path}")
    saved = json.loads(path.read_text(encoding="utf-8"))
    fresh = protocol(scope, directory, trainer_terminal)
    if saved != fresh:
        raise ValueError("Raw/cache/source or ranking terminal differs from frozen protocol")
    return saved, sha256(path)


def project_raw(scan: pl.LazyFrame) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    """Filter phase before selecting a narrowly allowed column list."""
    names = set(scan.collect_schema().names())
    missing = (set(DEP_COLUMNS) | set(ARR_COLUMNS)) - names
    if missing:
        raise ValueError(f"Raw source lacks required allowed fields: {sorted(missing)}")
    if FORBIDDEN.intersection(DEP_COLUMNS) or FORBIDDEN.intersection(ARR_COLUMNS):
        raise AssertionError("Forbidden label in phase-first raw projection")
    dep = scan.filter(pl.col("PHASE_mvt") == "DEP").select(list(DEP_COLUMNS))
    arr = scan.filter(pl.col("PHASE_mvt") == "ARR").select(list(ARR_COLUMNS))
    return dep, arr


def _normalize(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    normalized = str(value).strip().upper()
    return "" if normalized in PLACEHOLDERS else normalized


def flight_identity(value: Any) -> tuple[str, Any] | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return None
    if isinstance(value, (int, np.integer)):
        return ("numeric", int(value))
    if isinstance(value, (float, np.floating)):
        number = float(value)
        # Above 2**53 a floating source cannot preserve adjacent integer IDs.
        # Treat it as unusable rather than excluding an unrelated movement.
        if (not math.isfinite(number) or not number.is_integer()
                or abs(number) >= 2**53):
            return None
        return ("numeric", int(number))
    normalized = str(value).strip()
    if _normalize(normalized) == "":
        return None
    return ("string", normalized)


def timestamp_ns(values: pd.Series) -> tuple[np.ndarray, np.ndarray, pd.Series]:
    """Preserve nanosecond ties while rejecting out-of-int64-ns timestamps."""
    stamps = pd.to_datetime(values, utc=True, errors="coerce")
    micros = stamps.dt.as_unit("us").astype("int64").to_numpy()
    margin = max(WINDOW_NS, PROXY_MAX_NS)
    low_ns = int(np.iinfo(np.int64).min) + margin
    high_ns = int(np.iinfo(np.int64).max) - margin
    low_us = -(-low_ns // 1_000)
    high_us = high_ns // 1_000
    good = stamps.notna().to_numpy() & (micros >= low_us) & (micros <= high_us)
    safe = stamps.mask(~good)
    ns = safe.dt.as_unit("ns").astype("int64").to_numpy()
    return ns, good, safe


def _align_queries(dep: pd.DataFrame, baseline_ids: np.ndarray) -> pd.DataFrame:
    raw = pd.Index(dep["MVT_ID_mvt"])
    baseline = pd.Index(baseline_ids)
    if (raw.has_duplicates or baseline.has_duplicates or raw.isna().any()
            or baseline.isna().any() or len(raw) != len(baseline)):
        raise ValueError("Raw/baseline departure IDs are not exact unique sets")
    order = raw.get_indexer(baseline)
    if (order < 0).any() or not raw.isin(baseline).all():
        raise ValueError("Raw/baseline departure ID coverage differs")
    aligned = dep.iloc[order].reset_index(drop=True)
    if not np.array_equal(aligned.MVT_ID_mvt.to_numpy(), baseline_ids):
        raise ValueError("Baseline departure ID order was not preserved")
    return aligned


def _prepare_peer_and_query(dep: pd.DataFrame, arr: pd.DataFrame,
                            baseline_ids: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    query = _align_queries(dep.loc[:, list(QUERY_COLUMNS)], baseline_ids)
    dep_peer = dep.rename(columns={"ADEP_mvt": "airport", "AOBT_3_flt": "peer_aobt"})
    arr_peer = arr.rename(columns={"ADES_mvt": "airport"}).copy()
    arr_peer["peer_aobt"] = pd.NaT
    peer_cols = ("MVT_ID_mvt", "FLIGHT_ID_mvt", "PHASE_mvt", "airport",
                 "RUNWAY_mvt", "MVT_TIME_UTC_mvt", "peer_aobt")
    peers = pd.concat([dep_peer.loc[:, list(peer_cols)],
                       arr_peer.loc[:, list(peer_cols)]], ignore_index=True)
    return query, peers


def _enrich_times(query: pd.DataFrame, peers: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    for frame, airport_col in ((query, "ADEP_mvt"), (peers, "airport")):
        frame["airport_norm"] = [_normalize(x) for x in frame[airport_col]]
        frame["runway_norm"] = [_normalize(x) for x in frame["RUNWAY_mvt"]]
        frame["flight_key"] = [flight_identity(x) for x in frame["FLIGHT_ID_mvt"]]
        nanos, good, safe = timestamp_ns(frame["MVT_TIME_UTC_mvt"])
        frame["time_ns"] = nanos
        frame["good_time"] = good
        frame["year_month"] = (safe.dt.year.fillna(-1).astype("int32") * 100
                               + safe.dt.month.fillna(-1).astype("int32"))
    aobt_ns, aobt_good, _ = timestamp_ns(peers["peer_aobt"])
    peers["peer_aobt_ns"] = aobt_ns
    valid = np.zeros(len(peers), dtype=np.uint8)
    interval = np.zeros(len(peers), dtype=np.float32)
    mvt = peers["time_ns"].to_numpy(dtype=np.int64)
    phase = peers["PHASE_mvt"].to_numpy()
    for i in np.flatnonzero(peers["good_time"].to_numpy(dtype=bool) & aobt_good
                            & (phase == "DEP")):
        difference = int(mvt[i]) - int(aobt_ns[i])
        if 0 < difference <= PROXY_MAX_NS:
            valid[i] = 1
            interval[i] = difference / PROXY_MAX_NS
    peers["proxy_valid"] = valid
    peers["proxy_interval"] = interval
    peers["phase_order"] = np.where(phase == "ARR", 0, 1).astype(np.int8)
    peers["aobt_sort"] = np.where(valid != 0, aobt_ns,
                                  np.iinfo(np.int64).max).astype(np.int64)
    return query, peers


@dataclass(frozen=True)
class EventGroup:
    times: np.ndarray
    ids: np.ndarray
    flight: np.ndarray
    phases: np.ndarray
    runways: np.ndarray
    proxy_valid: np.ndarray
    aobt_ns: np.ndarray
    proxy_interval: np.ndarray


def _group_events(peers: pd.DataFrame) -> dict[tuple[int, str], EventGroup]:
    usable = peers.loc[peers.good_time & peers.airport_norm.ne("")
                       & peers.PHASE_mvt.isin(("ARR", "DEP"))]
    groups: dict[tuple[int, str], EventGroup] = {}
    for key, part in usable.groupby(["year_month", "airport_norm"], sort=False):
        ordered = part.sort_values(["time_ns", "phase_order", "runway_norm",
                                    "aobt_sort"], kind="mergesort").reset_index(drop=True)
        groups[(int(key[0]), str(key[1]))] = EventGroup(
            times=ordered.time_ns.to_numpy(dtype=np.int64),
            ids=ordered.MVT_ID_mvt.to_numpy(),
            flight=ordered.flight_key.to_numpy(dtype=object),
            phases=ordered.PHASE_mvt.to_numpy(),
            runways=ordered.runway_norm.to_numpy(),
            proxy_valid=ordered.proxy_valid.to_numpy(dtype=np.uint8),
            aobt_ns=ordered.peer_aobt_ns.to_numpy(dtype=np.int64),
            proxy_interval=ordered.proxy_interval.to_numpy(dtype=np.float32),
        )
    return groups


def _select_side(times: np.ndarray, peer_ids: np.ndarray,
                 peer_flight: np.ndarray, query_time: int, query_id: Any,
                 query_flight: tuple[str, Any] | None, *, past: bool) -> list[int]:
    selected: list[int] = []
    if past:
        position = int(np.searchsorted(times, query_time, side="left")) - 1
        while position >= 0 and int(times[position]) >= query_time - WINDOW_NS:
            same_time = int(times[position])
            first = int(np.searchsorted(times, same_time, side="left"))
            for index in range(first, position + 1):
                if peer_ids[index] == query_id:
                    continue
                if query_flight is not None and peer_flight[index] == query_flight:
                    continue
                selected.append(index)
                if len(selected) == PAST_SLOTS:
                    return sorted(selected)
            position = first - 1
    else:
        position = int(np.searchsorted(times, query_time, side="right"))
        while position < len(times) and int(times[position]) <= query_time + WINDOW_NS:
            same_time = int(times[position])
            last = int(np.searchsorted(times, same_time, side="right"))
            for index in range(position, last):
                if peer_ids[index] == query_id:
                    continue
                if query_flight is not None and peer_flight[index] == query_flight:
                    continue
                selected.append(index)
                if len(selected) == FUTURE_SLOTS:
                    return selected
            position = last
    return sorted(selected) if past else selected


def _fill_one(events: np.ndarray, presence: np.ndarray, row: int,
              query_time: int, query_good: bool, query_airport: str,
              query_runway: str, query_id: Any,
              query_flight: tuple[str, Any] | None,
              group: EventGroup | None) -> tuple[int, int]:
    if group is None or not query_good or not query_airport:
        return 0, 0
    qtime = int(query_time)
    times = group.times
    past = _select_side(times, group.ids, group.flight, qtime, query_id,
                        query_flight, past=True)
    future = _select_side(times, group.ids, group.flight, qtime, query_id,
                          query_flight, past=False)
    positions = (list(range(PAST_SLOTS - len(past), PAST_SLOTS))
                 + list(range(PAST_SLOTS, PAST_SLOTS + len(future))))
    selected = past + future
    qrunway = query_runway
    for slot, index in zip(positions, selected):
        delta = (int(times[index]) - qtime) / WINDOW_NS
        runway_relation = (-1 if not qrunway or not group.runways[index]
                           else 1 if qrunway == group.runways[index] else 0)
        valid = int(group.proxy_valid[index])
        event = (float(np.clip(delta, -1, 1)),
                 -1.0 if group.phases[index] == "ARR" else 1.0,
                 float(runway_relation),
                 float(np.clip((int(group.aobt_ns[index]) - qtime) / WINDOW_NS,
                               -1, 1)) if valid else 0.0,
                 float(valid),
                 float(group.proxy_interval[index]) if valid else 0.0)
        events[row, slot, :] = np.asarray(event, dtype=np.float16)
        presence[row, slot] = 1
    return len(past), len(future)


def build_arrays(query_raw: pd.DataFrame, dep_peer_raw: pd.DataFrame,
                 arr_peer_raw: pd.DataFrame, baseline_ids: np.ndarray,
                 events: np.ndarray, presence: np.ndarray) -> dict[str, Any]:
    """Pure event transform; labels are unavailable by construction."""
    if events.shape != (len(baseline_ids), *EVENT_SHAPE) or presence.shape != (len(baseline_ids), 32):
        raise ValueError("Output buffers lack exact baseline shape")
    if events.dtype != np.float16 or presence.dtype != np.uint8:
        raise ValueError("Output buffers require float16/uint8")
    if (FORBIDDEN.intersection(query_raw) or FORBIDDEN.intersection(dep_peer_raw)
            or FORBIDDEN.intersection(arr_peer_raw)):
        raise ValueError("Forbidden label field reached the event builder")
    if (list(query_raw) != list(QUERY_COLUMNS)
            or list(dep_peer_raw) != list(DEP_COLUMNS)
            or list(arr_peer_raw) != list(ARR_COLUMNS)):
        raise ValueError("Raw phase-first allowlist/order changed")
    if not np.array_equal(query_raw.MVT_ID_mvt.to_numpy(), dep_peer_raw.MVT_ID_mvt.to_numpy()):
        raise ValueError("Query and DEP peer projections differ before alignment")
    events[:] = 0
    presence[:] = 0
    # Reattach peer AOBT only to the peer table; the query table retains six
    # movement-only fields throughout feature computation.
    query, peers = _prepare_peer_and_query(dep_peer_raw, arr_peer_raw,
                                           baseline_ids)
    if not query.loc[:, list(QUERY_COLUMNS)].equals(
            _align_queries(query_raw, baseline_ids).loc[:, list(QUERY_COLUMNS)]):
        raise ValueError("Query movement-only projection differs from DEP peer rows")
    query, peers = _enrich_times(query, peers)
    groups = _group_events(peers)
    del peers
    gc.collect()
    past_total = future_total = zero_rows = 0
    q_month = query.year_month.to_numpy(dtype=np.int32)
    q_airport = query.airport_norm.to_numpy()
    q_runway = query.runway_norm.to_numpy()
    q_time = query.time_ns.to_numpy(dtype=np.int64)
    q_good = query.good_time.to_numpy(dtype=bool)
    q_id = query.MVT_ID_mvt.to_numpy()
    q_flight = query.flight_key.to_numpy(dtype=object)
    for row in range(len(query)):
        group = groups.get((int(q_month[row]), q_airport[row]))
        past, future = _fill_one(events, presence, row,
                                 int(q_time[row]), bool(q_good[row]),
                                 q_airport[row], q_runway[row], q_id[row],
                                 q_flight[row], group)
        past_total += past
        future_total += future
        zero_rows += (past + future == 0)
    for start in range(0, len(baseline_ids), 32_768):
        end = min(start + 32_768, len(baseline_ids))
        e, m = events[start:end], presence[start:end]
        if (not np.isfinite(e).all() or not np.isin(m, (0, 1)).all()
                or np.any(e[m == 0] != 0)):
            raise ValueError("Built event values, presence or padding are invalid")
    return {"rows": len(baseline_ids), "past_events": int(past_total),
            "future_events": int(future_total), "all_padding_rows": int(zero_rows),
            "source_labels_read": False}


def _raw_frames(scope: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_paths = (_training_files(DATA_DIR) if scope == "training"
                 else [DATA_DIR / "ranking.parquet"])
    scan = pl.scan_parquet([str(path) for path in raw_paths])
    dep_lazy, arr_lazy = project_raw(scan)
    dep = dep_lazy.collect().to_pandas()
    arr = arr_lazy.collect().to_pandas()
    # Derive the query view solely from the allowed DEP projection; do not
    # pass a query's AOBT column into build_arrays.
    return dep.loc[:, list(QUERY_COLUMNS)], dep, arr


def _baseline_ids(scope: str) -> np.ndarray:
    path = CACHE_DIR / ("training_rows.parquet" if scope == "training"
                        else "ranking_rows.parquet")
    frame = pd.read_parquet(path, columns=["MVT_ID_mvt"])
    ids = frame.MVT_ID_mvt.to_numpy()
    index = pd.Index(ids)
    if (len(ids) != EXPECTED_ROWS[scope] or index.has_duplicates
            or index.isna().any()):
        raise ValueError("Baseline IDs/count are not exact and unique")
    return ids


def build(scope: str = "training", directory: Path = DEFAULT_DIR,
          trainer_terminal: Path | None = None) -> dict[str, Any]:
    """Build the immutable cache after a previously published input protocol."""
    require_memory()
    directory = Path(directory)
    if scope == "ranking":
        _rank_authorization(directory, trainer_terminal)
    frozen, protocol_sha = _verify_protocol(scope, directory, trainer_terminal)
    output = _output_paths(scope, directory)
    existing = [str(path) for path in output.values() if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite v14 cache files: {existing}")
    ids = _baseline_ids(scope)
    query, dep, arr = _raw_frames(scope)
    event_path = _stage_path(directory, ".f16.memmap")
    mask_path = _stage_path(directory, ".u8.memmap")
    ids_path = _stage_path(directory, ".npy")
    staged_paths = (event_path, mask_path, ids_path)
    try:
        event_shape = (len(ids), *EVENT_SHAPE)
        mask_shape = (len(ids), EVENT_SHAPE[0])
        events = np.memmap(event_path, dtype=np.float16, mode="w+", shape=event_shape)
        presence = np.memmap(mask_path, dtype=np.uint8, mode="w+", shape=mask_shape)
        summary = build_arrays(query, dep, arr, ids, events, presence)
        events.flush()
        presence.flush()
        del events, presence, query, dep, arr
        with ids_path.open("wb") as target:
            np.save(target, ids, allow_pickle=False)
            target.flush()
            os.fsync(target.fileno())
        # Rehash every frozen source and input before publishing any cache file.
        if protocol(scope, directory, trainer_terminal) != frozen:
            raise ValueError("Source/input hashes changed during v14 feature construction")
        report = {
            "schema_version": 1,
            "scope": scope,
            "protocol_sha256": protocol_sha,
            "rows": len(ids),
            "event_shape": list(event_shape),
            "presence_shape": list(mask_shape),
            "event_dtype": "float16",
            "presence_dtype": "uint8",
            "id_dtype": str(ids.dtype),
            "cache_sha256": {"events": sha256(event_path),
                             "presence": sha256(mask_path), "ids": sha256(ids_path)},
            "input_sha256": {name: item["sha256"]
                             for name, item in frozen["inputs"].items()},
            "summary": summary,
            "label_values_read": False,
            "query_nm_used_as_feature": False,
        }
        # Hard-link creation is exclusive and atomic per file on one volume;
        # report publishes last, so a partial transaction never verifies.
        for name, staged in (("events", event_path), ("presence", mask_path),
                             ("ids", ids_path)):
            _exclusive_publish(staged, output[name])
        _write_json_exclusive(output["report"], report)
    finally:
        for path in staged_paths:
            path.unlink(missing_ok=True)
    return verify(scope, directory, trainer_terminal=trainer_terminal)


def verify(scope: str = "training", directory: Path = DEFAULT_DIR,
           *, trainer_terminal: Path | None = None) -> dict[str, Any]:
    """Read-only full source, cache, baseline-order and value-integrity replay."""
    directory = Path(directory)
    if scope == "ranking" and trainer_terminal is None:
        trainer_terminal = ROOT / "artifacts/v14-event-sequence/terminal.json"
    frozen, protocol_sha = _verify_protocol(scope, directory, trainer_terminal)
    output = _output_paths(scope, directory)
    if any(not path.exists() for path in output.values()):
        raise FileNotFoundError("v14 cache is incomplete; immutable outputs cannot be reused")
    report = json.loads(output["report"].read_text(encoding="utf-8"))
    shape = (EXPECTED_ROWS[scope], *EVENT_SHAPE)
    mask_shape = (EXPECTED_ROWS[scope], EVENT_SHAPE[0])
    if (report.get("scope") != scope or report.get("rows") != EXPECTED_ROWS[scope]
            or report.get("protocol_sha256") != protocol_sha
            or report.get("event_shape") != list(shape)
            or report.get("presence_shape") != list(mask_shape)
            or report.get("event_dtype") != "float16"
            or report.get("presence_dtype") != "uint8"
            or report.get("input_sha256") != {
                name: item["sha256"] for name, item in frozen["inputs"].items()}
            or report.get("label_values_read") is not False
            or report.get("query_nm_used_as_feature") is not False):
        raise ValueError("v14 cache report/source/protocol mismatch")
    expected_bytes = {"events": int(np.prod(shape)) * np.dtype(np.float16).itemsize,
                      "presence": int(np.prod(mask_shape)) * np.dtype(np.uint8).itemsize}
    for name in ("events", "presence", "ids"):
        if sha256(output[name]) != report["cache_sha256"][name]:
            raise ValueError(f"v14 {name} bytes changed")
        if name in expected_bytes and output[name].stat().st_size != expected_bytes[name]:
            raise ValueError(f"v14 {name} byte length changed")
    ids = np.load(output["ids"], mmap_mode="r", allow_pickle=False)
    baseline = _baseline_ids(scope)
    if (ids.dtype.name != report["id_dtype"] or len(ids) != len(baseline)
            or not np.array_equal(ids, baseline)):
        raise ValueError("v14 cache IDs differ from baseline order")
    events = np.memmap(output["events"], dtype=np.float16, mode="r", shape=shape)
    presence = np.memmap(output["presence"], dtype=np.uint8, mode="r", shape=mask_shape)
    for start in range(0, len(ids), 32_768):
        end = min(start + 32_768, len(ids))
        e = events[start:end]
        m = presence[start:end]
        if (not np.isfinite(e).all() or not np.isin(m, (0, 1)).all()
                or np.any(e[m == 0] != 0)):
            raise ValueError("v14 event values, presence or padding changed")
    if protocol(scope, directory, trainer_terminal) != frozen:
        raise ValueError("v14 sources changed during verification")
    return {
        "scope": scope,
        "passed": True,
        "departure_labels_used": False,
        "rows": len(ids),
        "event_shape": list(shape),
        "presence_shape": list(mask_shape),
        "protocol_sha256": protocol_sha,
        "build_report_sha256": sha256(output["report"]),
        "input_sha256": report["input_sha256"],
        "cache_sha256": report["cache_sha256"],
        "ids_order_verified": True,
        "values_verified": True,
    }


def open_arrays(scope: str, directory: Path, verified_receipt: dict,
                mode: str = "r") -> tuple[np.ndarray, np.memmap, np.memmap, dict]:
    """Return read-only disk-backed arrays after replaying the given receipt."""
    if mode != "r":
        raise ValueError("v14 model consumers may only open read-only memmaps")
    actual = verify(scope, directory)
    if actual != verified_receipt:
        raise ValueError("v14 verified receipt changed before memmap opening")
    output = _output_paths(scope, directory)
    ids = np.load(output["ids"], mmap_mode="r", allow_pickle=False)
    events = np.memmap(output["events"], dtype=np.float16, mode="r",
                       shape=(len(ids), *EVENT_SHAPE))
    presence = np.memmap(output["presence"], dtype=np.uint8, mode="r",
                         shape=(len(ids), EVENT_SHAPE[0]))
    report = json.loads(output["report"].read_text(encoding="utf-8"))
    return ids, events, presence, report


def _synthetic_frames() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    base = pd.Timestamp("2025-01-15T12:00:00Z")
    dep_rows: list[dict] = []
    for i in range(1, 91):
        offset = i - 45
        if i == 1:
            offset = 0
        stamp = base + pd.Timedelta(seconds=offset)
        dep_rows.append({
            "MVT_ID_mvt": float(i), "FLIGHT_ID_mvt": float(100 if i in (1, 2) else i + 100),
            "PHASE_mvt": "DEP", "ADEP_mvt": "EDDF", "RUNWAY_mvt": "25C" if i % 3 else "18",
            "MVT_TIME_UTC_mvt": stamp,
            "AOBT_3_flt": stamp - pd.Timedelta(seconds=300),
        })
    dep = pd.DataFrame(dep_rows, columns=list(DEP_COLUMNS))
    arr_rows = [{"MVT_ID_mvt": float(1000 + i), "FLIGHT_ID_mvt": float(2000 + i),
                 "PHASE_mvt": "ARR", "ADES_mvt": "EDDF",
                 "RUNWAY_mvt": "25C" if i % 2 else "18",
                 "MVT_TIME_UTC_mvt": base + pd.Timedelta(seconds=i - 10)}
                for i in range(20)]
    arr = pd.DataFrame(arr_rows, columns=list(ARR_COLUMNS))
    return dep, arr, dep.MVT_ID_mvt.to_numpy()


def _synthetic_oracle(group: EventGroup, query_time: int,
                      query_id: Any, flight: tuple[str, Any] | None,
                      past: bool) -> list[int]:
    """Independent brute-force selection rule, including canonical tie keys."""
    choices: list[tuple[tuple[Any, ...], int]] = []
    for index, peer_time in enumerate(group.times):
        dt = int(peer_time) - query_time
        if ((past and not (-WINDOW_NS <= dt < 0))
                or (not past and not (0 < dt <= WINDOW_NS))):
            continue
        if group.ids[index] == query_id or (flight is not None and group.flight[index] == flight):
            continue
        phase_order = 0 if group.phases[index] == "ARR" else 1
        aobt_sort = (int(group.aobt_ns[index]) if group.proxy_valid[index]
                     else int(np.iinfo(np.int64).max))
        choices.append(((abs(dt), phase_order, group.runways[index], aobt_sort), index))
    chosen = [index for _, index in sorted(choices, key=lambda item: item[0])[:16]]
    return sorted(chosen)


def self_test() -> dict[str, Any]:
    """No real raw, cache, labels, ranking or model reads/writes."""
    assert flight_identity(np.int64(2**53)) != flight_identity(np.int64(2**53 + 1))
    assert flight_identity(float(2**53)) is None
    assert flight_identity("  abc-123 ") == ("string", "abc-123")
    dep, arr, ids = _synthetic_frames()
    query = dep.loc[:, list(QUERY_COLUMNS)].copy()
    event = np.empty((len(ids), *EVENT_SHAPE), dtype=np.float16)
    presence = np.empty((len(ids), EVENT_SHAPE[0]), dtype=np.uint8)
    result = build_arrays(query, dep, arr, ids, event, presence)
    assert result["rows"] == 90 and int(presence[0].sum()) == 32
    assert presence[0, :16].all() and presence[0, 16:].all()
    assert np.isfinite(event).all() and not np.any(event[presence == 0] != 0)

    # Query's own NM clock and forbidden label poison cannot reach its channels.
    poisoned = dep.copy()
    poisoned.loc[0, "AOBT_3_flt"] = dep.loc[0, "MVT_TIME_UTC_mvt"] - pd.Timedelta(seconds=3_000)
    changed = np.empty_like(event)
    changed_mask = np.empty_like(presence)
    build_arrays(query, poisoned, arr, ids, changed, changed_mask)
    assert np.array_equal(event[0], changed[0]) and np.array_equal(presence[0], changed_mask[0])
    poison_raw = pd.concat([dep.assign(ADES_mvt="EDDF"),
                            arr.assign(ADEP_mvt="EDDF", AOBT_3_flt=pd.NaT)],
                           ignore_index=True)
    poison_raw["BLOCK_TIME_UTC_mvt"] = "forbidden-label-poison-A"
    poison_raw["TAXITIME_SEC_mvt"] = 987_654
    projected_dep, projected_arr = project_raw(pl.from_pandas(poison_raw).lazy())
    first = (projected_dep.collect().to_pandas(), projected_arr.collect().to_pandas())
    poison_raw["BLOCK_TIME_UTC_mvt"] = "forbidden-label-poison-B"
    poison_raw["TAXITIME_SEC_mvt"] = -987_654
    next_dep, next_arr = project_raw(pl.from_pandas(poison_raw).lazy())
    assert first[0].equals(next_dep.collect().to_pandas())
    assert first[1].equals(next_arr.collect().to_pandas())
    assert not (FORBIDDEN & set(first[0])) and not (FORBIDDEN & set(first[1]))

    # Renaming every opaque movement ID changes neither tie choice nor values.
    renamed_dep = dep.copy()
    renamed_arr = arr.copy()
    renamed_dep["MVT_ID_mvt"] = dep.MVT_ID_mvt.to_numpy() + 10_000
    renamed_arr["MVT_ID_mvt"] = arr.MVT_ID_mvt.to_numpy() + 10_000
    ids_new = renamed_dep.MVT_ID_mvt.to_numpy()
    renamed_events = np.empty_like(event)
    renamed_mask = np.empty_like(presence)
    build_arrays(renamed_dep.loc[:, list(QUERY_COLUMNS)], renamed_dep,
                 renamed_arr, ids_new, renamed_events, renamed_mask)
    assert np.array_equal(event, renamed_events)
    assert np.array_equal(presence, renamed_mask)

    prepared_q, prepared_p = _prepare_peer_and_query(dep, arr, ids)
    prepared_q, prepared_p = _enrich_times(prepared_q, prepared_p)
    groups = _group_events(prepared_p)
    group = groups[(202501, "EDDF")]
    qtime = int(prepared_q.time_ns.iloc[0])
    qflight = prepared_q.flight_key.iloc[0]
    for pos in range(len(prepared_q)):
        qt = int(prepared_q.time_ns.iloc[pos])
        qid = prepared_q.MVT_ID_mvt.iloc[pos]
        fid = prepared_q.flight_key.iloc[pos]
        for past in (True, False):
            actual = _select_side(group.times, group.ids, group.flight,
                                  qt, qid, fid, past=past)
            assert actual == _synthetic_oracle(group, qt, qid, fid, past)
    # Another record of the same query flight and all exact-time peers vanish.
    assert not any(group.ids[index] == 2.0 for index in
                   _select_side(group.times, group.ids, group.flight,
                                qtime, 1.0, qflight, past=True)
                   + _select_side(group.times, group.ids, group.flight,
                                  qtime, 1.0, qflight, past=False))

    # Exact strict query time and inclusive +/- one-hour outer boundaries.
    boundary = dep.iloc[[0]].copy()
    boundary.loc[:, "MVT_ID_mvt"] = 501.0
    boundary.loc[:, "FLIGHT_ID_mvt"] = 501.0
    extra = []
    for index, ns in enumerate((-WINDOW_NS - 1, -WINDOW_NS, -100, 0,
                                100, WINDOW_NS, WINDOW_NS + 1), start=502):
        row = boundary.iloc[0].copy()
        row["MVT_ID_mvt"] = float(index)
        row["FLIGHT_ID_mvt"] = float(index)
        row["MVT_TIME_UTC_mvt"] = pd.Timestamp(base_time := "2025-01-15T12:00:00Z") + pd.Timedelta(ns, unit="ns")
        row["AOBT_3_flt"] = row["MVT_TIME_UTC_mvt"] - pd.Timedelta(seconds=300)
        extra.append(row)
    boundary_dep = pd.concat([boundary, pd.DataFrame(extra)], ignore_index=True)
    empty_arr = pd.DataFrame(columns=list(ARR_COLUMNS))
    boundary_ids = boundary_dep.MVT_ID_mvt.to_numpy()
    boundary_events = np.empty((len(boundary_ids), *EVENT_SHAPE), dtype=np.float16)
    boundary_mask = np.empty((len(boundary_ids), 32), dtype=np.uint8)
    build_arrays(boundary_dep.loc[:, list(QUERY_COLUMNS)], boundary_dep,
                 empty_arr, boundary_ids, boundary_events, boundary_mask)
    assert int(boundary_mask[0].sum()) == 4  # -3600, -100ns, +100ns, +3600
    assert boundary_mask[0, :16].sum() == 2 and boundary_mask[0, 16:].sum() == 2

    # Cross-month neighbors, NaT queries and year 3000 never leak or overflow.
    other = boundary_dep.iloc[[0, 1, 2]].copy()
    other.loc[0, "MVT_TIME_UTC_mvt"] = pd.Timestamp("2025-01-31T23:59:59Z")
    other.loc[1, "MVT_TIME_UTC_mvt"] = pd.Timestamp("2025-02-01T00:00:00Z")
    other.loc[2, "MVT_TIME_UTC_mvt"] = pd.NaT
    other.loc[0, "AOBT_3_flt"] = pd.NaT
    other.loc[1, "AOBT_3_flt"] = pd.NaT
    other.loc[2, "AOBT_3_flt"] = pd.NaT
    oids = other.MVT_ID_mvt.to_numpy()
    oe = np.empty((len(oids), *EVENT_SHAPE), dtype=np.float16)
    om = np.empty((len(oids), 32), dtype=np.uint8)
    build_arrays(other.loc[:, list(QUERY_COLUMNS)], other, empty_arr, oids, oe, om)
    assert int(om[0].sum()) == 0 and int(om[2].sum()) == 0
    extreme = pd.Series([pd.Timestamp("2025-01-01T00:00:00.000000100Z"),
                         pd.Timestamp("2025-01-01T00:00:00.000000200Z"),
                         pd.Timestamp("3000-01-01T00:00:00Z"), pd.NaT])
    time_ns, good, _ = timestamp_ns(extreme)
    assert good.tolist() == [True, True, False, False]
    assert int(time_ns[1]) - int(time_ns[0]) == 100
    return {"synthetic": "passed", "rows": 90, "event_channels": 6,
            "past_slots": 16, "future_slots": 16,
            "raw_label_values_read": False, "real_cache_or_model_read": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("self-test", "prepare", "build", "verify"),
                        required=True)
    parser.add_argument("--scope", choices=("training", "ranking"), default="training")
    parser.add_argument("--directory", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--trainer-terminal", type=Path)
    parser.add_argument("--published-source-sha256")
    args = parser.parse_args()
    if args.mode == "self-test":
        result = self_test()
    else:
        source_sha_check(args.published_source_sha256)
        if args.mode == "prepare":
            result = prepare(args.scope, args.directory, args.trainer_terminal)
        elif args.mode == "build":
            result = build(args.scope, args.directory, args.trainer_terminal)
        else:
            result = verify(args.scope, args.directory,
                            trainer_terminal=args.trainer_terminal)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
