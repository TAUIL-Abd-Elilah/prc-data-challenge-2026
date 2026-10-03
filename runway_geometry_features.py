"""Prospective, label-free runway geometry and other-runway traffic features.

The only competition movement values read are phase, movement ID, airport,
runway, and movement timestamp. Departure BLOCK/TAXITIME and flight labels are
never selected. The complete released batch makes past events retrospective;
these features do not claim availability for live operational forecasting.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import sys
import tempfile

import numpy as np
import pandas as pd
import polars as pl
import pyarrow


ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = ROOT / "artifacts/prospective-runway-geometry"
DEFAULT_OUTPUT = DEFAULT_SOURCE / "features"
DEFAULT_DATA = ROOT / "data"
DEFAULT_BASELINE = ROOT / "artifacts/baseline"
DEFAULT_WEATHER = ROOT / "data/external/weather.parquet"
SPEC_PATH = DEFAULT_SOURCE / "feature_spec.json"
MIN_FREE_GIB = 4.0
EARTH_RADIUS_M = 6_371_000.0
PARALLEL_MAX_BEARING_DEG = 15.0
PARALLEL_MAX_DISTANCE_M = 2_000.0
WINDOWS_SECONDS = (900, 3_600)
PINNED_COMMIT = "59574226b6df9417f5a7a8d03eeb6c7158a76168"
PINNED_SOURCE_SHA256 = {
    "LICENSE": "6b0382b16279f26ff69014300541967a356a666eb0b91b422f6862f6b7dad17e",
    "airports.csv": "a5f3b2ba662528c92fcda43fbbb11f00261eec580b339d4ac98537f882f9fd9d",
    "runways.csv": "7be01aa01204b13edde729d603505cf4c1661d274128b70a137c2285be728197",
}
AIRPORTS = ("EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LTFM", "LSZH")
PLACEHOLDERS = frozenset({"", "?", "UNKNOWN", "UNKN", "UNK", "N/A", "NA", "NONE", "NULL", "NIL", "__MISSING__"})
RAW_DEP = ("MVT_ID_mvt", "ADEP_mvt", "RUNWAY_mvt", "MVT_TIME_UTC_mvt")
RAW_ARR = ("MVT_ID_mvt", "ADES_mvt", "RUNWAY_mvt", "MVT_TIME_UTC_mvt")
WEATHER_COLS = ("airport", "weather_hour_utc", "wx_wind_direction_deg", "wx_wind_speed_mps")
FEATURES = (
    "rgeom_heading_sin", "rgeom_heading_cos", "rgeom_strip_length_m",
    "rgeom_headwind_mps", "rgeom_tailwind_positive_mps", "rgeom_crosswind_abs_mps",
    "rgeom_intersecting_other_strip_count", "rgeom_near_parallel_other_strip_count",
    "rgeom_intersecting_arr_past15_count", "rgeom_intersecting_arr_past60_count",
    "rgeom_intersecting_dep_past15_count", "rgeom_intersecting_dep_past60_count",
    "rgeom_near_parallel_arr_past15_count", "rgeom_near_parallel_arr_past60_count",
    "rgeom_near_parallel_dep_past15_count", "rgeom_near_parallel_dep_past60_count",
)
COUNT_FEATURES = FEATURES[8:]
STATIC_COUNT_FEATURES = FEATURES[6:8]


@dataclass(frozen=True)
class Strip:
    airport: str
    sid: int  # Internal 2024 CSV row index; never a model predictor.
    le_code: str
    he_code: str
    le_lat: float
    le_lon: float
    he_lat: float
    he_lon: float
    length_m: float
    le_heading_deg: float
    he_heading_deg: float


@dataclass
class Geometry:
    strips: dict[tuple[str, int], Strip]
    code_map: dict[tuple[str, str], tuple[int, float] | None]
    neighbors: dict[tuple[str, int], dict[str, tuple[int, ...]]]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_exclusive(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as out:
        json.dump(value, out, indent=2, allow_nan=False)
        out.write("\n")


def normalized(values: pd.Series) -> np.ndarray:
    return (values.astype("string").fillna("").str.strip().str.upper()
            .to_numpy(dtype=str))


def normalized_one(value: object) -> str:
    return "" if pd.isna(value) else str(value).strip().upper()


def available_memory_gib() -> float:
    if os.name == "nt":
        import ctypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(MemoryStatus)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise OSError("GlobalMemoryStatusEx failed")
        return status.ullAvailPhys / 2**30
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text(encoding="ascii").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024 / 2**30
    raise RuntimeError("Cannot establish free physical memory")


def require_memory(min_free_gib: float) -> None:
    if not np.isfinite(min_free_gib) or min_free_gib < MIN_FREE_GIB:
        raise ValueError("Runway geometry memory floor cannot be below 4 GiB")
    available = available_memory_gib()
    if available < min_free_gib:
        raise MemoryError(f"Need {min_free_gib:g} GiB free, have {available:.2f} GiB")


def initial_bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    a, b = math.radians(lat1), math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    y = math.sin(dlon) * math.cos(b)
    x = math.cos(a) * math.sin(b) - math.sin(a) * math.cos(b) * math.cos(dlon)
    return math.degrees(math.atan2(y, x)) % 360.0


def great_circle_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(max(a, 0.0))))


def project_local(lat: float, lon: float, origin_lat: float, origin_lon: float) -> np.ndarray:
    return np.array((EARTH_RADIUS_M * math.radians(lon - origin_lon) * math.cos(math.radians(origin_lat)),
                     EARTH_RADIUS_M * math.radians(lat - origin_lat)), dtype=np.float64)


def cross(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def segment_intersects(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
    """Closed-segment intersection, including shared endpoint and overlap."""
    r, s = b - a, d - c
    denom = cross(r, s)
    ac = c - a
    tolerance = 1e-9
    if abs(denom) <= tolerance:
        if abs(cross(ac, r)) > tolerance:
            return False
        rr = float(np.dot(r, r))
        if rr <= tolerance:
            return point_segment_distance(a, c, d) <= tolerance
        t0, t1 = float(np.dot(ac, r) / rr), float(np.dot(d - a, r) / rr)
        return max(min(t0, t1), 0.0) <= min(max(t0, t1), 1.0) + tolerance
    t = cross(ac, s) / denom
    u = cross(ac, r) / denom
    return -tolerance <= t <= 1 + tolerance and -tolerance <= u <= 1 + tolerance


def point_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    length_sq = float(np.dot(ab, ab))
    if length_sq == 0:
        return float(np.linalg.norm(p - a))
    t = float(np.clip(np.dot(p - a, ab) / length_sq, 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def segment_distance(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> float:
    if segment_intersects(a, b, c, d):
        return 0.0
    return min(point_segment_distance(a, c, d), point_segment_distance(b, c, d),
               point_segment_distance(c, a, b), point_segment_distance(d, a, b))


def unoriented_bearing_diff_deg(left: float, right: float) -> float:
    return abs(((left - right + 90.0) % 180.0) - 90.0)


def strip_relation(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray,
                   own_bearing: float, other_bearing: float) -> str | None:
    if segment_intersects(a, b, c, d):
        return "intersecting"
    if (unoriented_bearing_diff_deg(own_bearing, other_bearing)
            <= PARALLEL_MAX_BEARING_DEG + 1e-9
            and segment_distance(a, b, c, d) <= PARALLEL_MAX_DISTANCE_M + 1e-9):
        return "near_parallel"
    return None


def build_geometry_from_frames(airports: pd.DataFrame, runways: pd.DataFrame) -> Geometry:
    """Exact ICAO/ident and exact trimmed runway-end codes only."""
    strips: dict[tuple[str, int], Strip] = {}
    code_candidates: dict[tuple[str, str], list[tuple[int, float]]] = defaultdict(list)
    by_airport: dict[str, list[Strip]] = defaultdict(list)
    ident = normalized(airports["ident"])
    gps = normalized(airports["gps_code"])
    airport_lookup: dict[str, str] = {}
    for airport in AIRPORTS:
        matches = sorted(set(ident[(ident == airport) | (gps == airport)]))
        if len(matches) != 1:
            raise ValueError(f"Pinned OurAirports airport mapping is not unique: {airport}")
        airport_lookup[airport] = matches[0]
    runway_airport = normalized(runways["airport_ident"])
    for airport in AIRPORTS:
        for sid in np.flatnonzero(runway_airport == airport_lookup[airport]):
            row = runways.iloc[int(sid)]
            values = [pd.to_numeric(row[c], errors="coerce") for c in (
                "le_latitude_deg", "le_longitude_deg", "he_latitude_deg", "he_longitude_deg")]
            if (not np.isfinite(values).all() or not all(-90 <= value <= 90 for value in (values[0], values[2]))
                    or not all(-180 <= value <= 180 for value in (values[1], values[3]))):
                continue
            le_lat, le_lon, he_lat, he_lon = map(float, values)
            length = great_circle_m(le_lat, le_lon, he_lat, he_lon)
            if not np.isfinite(length) or length < 1:
                continue
            le_code, he_code = normalized_one(row["le_ident"]), normalized_one(row["he_ident"])
            strip = Strip(airport, int(sid), le_code, he_code, le_lat, le_lon, he_lat, he_lon,
                          length, initial_bearing_deg(le_lat, le_lon, he_lat, he_lon),
                          initial_bearing_deg(he_lat, he_lon, le_lat, le_lon))
            strips[(airport, int(sid))] = strip
            by_airport[airport].append(strip)
            if le_code not in PLACEHOLDERS:
                code_candidates[(airport, le_code)].append((int(sid), strip.le_heading_deg))
            if he_code not in PLACEHOLDERS:
                code_candidates[(airport, he_code)].append((int(sid), strip.he_heading_deg))
    code_map = {key: values[0] if len({sid for sid, _ in values}) == 1 else None
                for key, values in code_candidates.items()}
    neighbors: dict[tuple[str, int], dict[str, tuple[int, ...]]] = {}
    for airport, airport_strips in by_airport.items():
        endpoint_lat = [value for strip in airport_strips for value in (strip.le_lat, strip.he_lat)]
        endpoint_lon = [value for strip in airport_strips for value in (strip.le_lon, strip.he_lon)]
        origin_lat, origin_lon = float(np.mean(endpoint_lat)), float(np.mean(endpoint_lon))
        segments = {strip.sid: (project_local(strip.le_lat, strip.le_lon, origin_lat, origin_lon),
                                 project_local(strip.he_lat, strip.he_lon, origin_lat, origin_lon))
                    for strip in airport_strips}
        for own in airport_strips:
            intersecting, nearby_parallel = [], []
            a, b = segments[own.sid]
            for other in airport_strips:
                if other.sid == own.sid:
                    continue  # Reciprocal runway end is the same physical strip.
                c, d = segments[other.sid]
                relation = strip_relation(a, b, c, d, own.le_heading_deg, other.le_heading_deg)
                if relation == "intersecting":
                    intersecting.append(other.sid)
                elif relation == "near_parallel":
                    nearby_parallel.append(other.sid)
            neighbors[(airport, own.sid)] = {
                "intersecting": tuple(sorted(intersecting)),
                "near_parallel": tuple(sorted(nearby_parallel)),
            }
    return Geometry(strips, code_map, neighbors)


def load_geometry(source_dir: Path) -> Geometry:
    airport_file = source_dir / "airports.csv"
    runway_file = source_dir / "runways.csv"
    airports = pd.read_csv(airport_file, usecols=["ident", "gps_code"], dtype="string")
    runways = pd.read_csv(runway_file, usecols=["airport_ident", "le_ident", "he_ident",
                                              "le_latitude_deg", "le_longitude_deg",
                                              "he_latitude_deg", "he_longitude_deg"], dtype="string")
    return build_geometry_from_frames(airports, runways)


def valid_id_order(raw_dep: pd.DataFrame, baseline_ids: pd.DataFrame) -> pd.DataFrame:
    raw = pd.Index(raw_dep["MVT_ID_mvt"])
    baseline = pd.Index(baseline_ids["MVT_ID_mvt"])
    if (len(raw) != len(baseline) or raw.has_duplicates or baseline.has_duplicates
            or raw.isna().any() or baseline.isna().any()
            or not raw.isin(baseline).all() or not baseline.isin(raw).all()):
        raise ValueError("Raw and baseline departure IDs lack exact unique coverage")
    if np.array_equal(raw.to_numpy(), baseline.to_numpy()):
        return raw_dep.reset_index(drop=True)
    positions = pd.Series(np.arange(len(raw), dtype=np.int64), index=raw)
    order = positions.reindex(baseline).to_numpy()
    if pd.isna(order).any():
        raise ValueError("Raw/baseline departure ID alignment failed")
    aligned = raw_dep.iloc[order.astype(np.int64)].reset_index(drop=True)
    if not np.array_equal(aligned.MVT_ID_mvt.to_numpy(), baseline.to_numpy()):
        raise ValueError("Baseline departure order was not preserved")
    return aligned


def timestamp_ns(values: pd.Series) -> tuple[np.ndarray, np.ndarray, pd.Series]:
    stamps = pd.to_datetime(values, utc=True, errors="coerce")
    # Arrow/Pandas microsecond timestamps can represent year 3000, while int64
    # nanoseconds cannot. Keep one maximum-window margin for q-window math.
    micros = stamps.dt.as_unit("us").astype("int64").to_numpy()
    margin_ns = max(WINDOWS_SECONDS) * 1_000_000_000
    low_ns = int(np.iinfo(np.int64).min) + margin_ns
    high_ns = int(np.iinfo(np.int64).max) - margin_ns
    low_us = -(-low_ns // 1_000)  # Exact integer ceil, including negative dates.
    high_us = high_ns // 1_000
    good = stamps.notna().to_numpy() & (micros >= low_us) & (micros <= high_us)
    safe_stamps = stamps.mask(~good)
    # Convert the original safe values, rather than the microsecond bounds
    # array, so strict ordering at submicrosecond resolution is preserved.
    nanos = safe_stamps.dt.as_unit("ns").astype("int64").to_numpy()
    return nanos, good, safe_stamps


def event_groups(dep: pd.DataFrame, arr: pd.DataFrame, geometry: Geometry) -> dict[tuple[str, int, str], np.ndarray]:
    grouped: dict[tuple[str, int, str], list[int]] = defaultdict(list)
    for phase, frame, airport_column in (("DEP", dep, "ADEP_mvt"), ("ARR", arr, "ADES_mvt")):
        airport = normalized(frame[airport_column])
        runway = normalized(frame["RUNWAY_mvt"])
        time_ns, good_time, _ = timestamp_ns(frame["MVT_TIME_UTC_mvt"])
        for a, r, t, good in zip(airport, runway, time_ns, good_time):
            if not good:
                continue
            match = geometry.code_map.get((a, r))
            if match is not None:
                grouped[(a, match[0], phase)].append(int(t))
    return {key: np.sort(np.asarray(values, dtype=np.int64)) for key, values in grouped.items()}


def wind_components(heading_deg: np.ndarray, wind_from_deg: np.ndarray,
                    wind_speed_mps: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    head = np.full(len(heading_deg), np.nan, dtype=np.float64)
    tail = head.copy()
    crosswind = head.copy()
    valid = (np.isfinite(heading_deg) & np.isfinite(wind_from_deg)
             & np.isfinite(wind_speed_mps) & (wind_speed_mps >= 0)
             & (wind_from_deg >= 0) & (wind_from_deg <= 360))
    angle = np.deg2rad(wind_from_deg[valid] - heading_deg[valid])
    head[valid] = wind_speed_mps[valid] * np.cos(angle)
    tail[valid] = np.maximum(-head[valid], 0.0)
    crosswind[valid] = wind_speed_mps[valid] * np.abs(np.sin(angle))
    return head, tail, crosswind


def compute_features(dep: pd.DataFrame, arr: pd.DataFrame, baseline_ids: pd.DataFrame,
                     weather: pd.DataFrame, geometry: Geometry) -> pd.DataFrame:
    """Build the fixed 16 fields; no labels or opaque ID values become features."""
    query = valid_id_order(dep, baseline_ids)
    n = len(query)
    output = {name: np.full(n, np.nan, dtype=np.float32) for name in FEATURES}
    airport, runway = normalized(query["ADEP_mvt"]), normalized(query["RUNWAY_mvt"])
    qtime, good_time, stamps = timestamp_ns(query["MVT_TIME_UTC_mvt"])
    heading = np.full(n, np.nan, dtype=np.float64)
    query_groups: dict[tuple[str, int], list[int]] = defaultdict(list)
    for i, (a, r) in enumerate(zip(airport, runway)):
        match = geometry.code_map.get((a, r))
        if match is None:
            continue
        sid, direction = match
        strip = geometry.strips[(a, sid)]
        related = geometry.neighbors[(a, sid)]
        heading[i] = direction
        angle = math.radians(direction)
        output[FEATURES[0]][i] = math.sin(angle)
        output[FEATURES[1]][i] = math.cos(angle)
        output[FEATURES[2]][i] = strip.length_m
        output[FEATURES[6]][i] = len(related["intersecting"])
        output[FEATURES[7]][i] = len(related["near_parallel"])
        if good_time[i]:
            query_groups[(a, sid)].append(i)

    # Exact airport + UTC-hour lookup. No nearby-hour filling or extra weather columns.
    if list(weather.columns) != list(WEATHER_COLS):
        raise ValueError("Weather input must contain only the four allowed columns in order")
    wx = weather.copy()
    wx["airport"] = wx["airport"].astype("string")
    wx["weather_hour_utc"] = pd.to_datetime(wx["weather_hour_utc"], utc=True, errors="coerce")
    if (wx.airport.isna().any() or wx.weather_hour_utc.isna().any()
            or not wx.weather_hour_utc.eq(wx.weather_hour_utc.dt.floor("h")).all()
            or wx.duplicated(["airport", "weather_hour_utc"]).any()):
        raise ValueError("Weather airport-hour keys must be unique, valid, and exact hours")
    lookup = wx.set_index(["airport", "weather_hour_utc"])
    hours = stamps.dt.floor("h")
    keys = pd.MultiIndex.from_arrays([pd.Series(airport, dtype="string"), hours],
                                     names=lookup.index.names)
    matched = lookup.reindex(keys)
    direction = pd.to_numeric(matched["wx_wind_direction_deg"], errors="coerce").to_numpy(dtype=float)
    speed = pd.to_numeric(matched["wx_wind_speed_mps"], errors="coerce").to_numpy(dtype=float)
    head, tail, crosswind = wind_components(heading, direction, speed)
    for name, values in zip(FEATURES[3:6], (head, tail, crosswind)):
        output[name][:] = values.astype(np.float32)

    groups = event_groups(dep, arr, geometry)
    for (a, sid), index in query_groups.items():
        idx = np.asarray(index, dtype=np.int64)
        q = qtime[idx]
        for relation in ("intersecting", "near_parallel"):
            neighbor_ids = geometry.neighbors[(a, sid)][relation]
            for phase in ("ARR", "DEP"):
                pieces = [groups[(a, other, phase)] for other in neighbor_ids
                          if (a, other, phase) in groups]
                events = (np.sort(np.concatenate(pieces)) if pieces
                          else np.empty(0, dtype=np.int64))
                upper = np.searchsorted(events, q, side="left")
                for window in WINDOWS_SECONDS:
                    lower = np.searchsorted(events, q - window * 1_000_000_000, side="left")
                    name = f"rgeom_{relation}_{phase.lower()}_past{window // 60}_count"
                    output[name][idx] = (upper - lower).astype(np.float32)

    result = pd.DataFrame({"MVT_ID_mvt": baseline_ids.MVT_ID_mvt.to_numpy(), **output})
    verify_output(result, baseline_ids)
    return result


def verify_output(frame: pd.DataFrame, baseline_ids: pd.DataFrame) -> None:
    if (list(frame.columns) != ["MVT_ID_mvt", *FEATURES] or len(frame) != len(baseline_ids)
            or not np.array_equal(frame.MVT_ID_mvt.to_numpy(), baseline_ids.MVT_ID_mvt.to_numpy())
            or frame.MVT_ID_mvt.isna().any() or frame.MVT_ID_mvt.duplicated().any()):
        raise ValueError("Runway geometry output ID/order/schema does not equal baseline")
    for name in FEATURES:
        values = frame[name].to_numpy()
        if values.dtype != np.float32 or np.isinf(values).any():
            raise ValueError(f"Invalid float32/nonfinite output: {name}")
        if name.endswith("_count"):
            present = values[np.isfinite(values)]
            if (present < 0).any() or not np.equal(present, np.floor(present)).all():
                raise ValueError(f"Invalid negative or fractional count: {name}")
    # A geometry mismatch means every feature is missing, rather than zero.
    geometry_missing = frame[FEATURES[0]].isna().to_numpy()
    if frame.loc[geometry_missing, list(FEATURES)].notna().any().any():
        raise ValueError("Unmatched runway retained a model feature")


def training_paths(data_dir: Path) -> list[Path]:
    paths = sorted(path for path in data_dir.glob("training_2025-*.parquet")
                   if re.fullmatch(r"training_2025-\d{2}-01_20\d{2}-\d{2}-01\.parquet", path.name))
    if len(paths) != 12 or len({path.name for path in paths}) != 12:
        raise FileNotFoundError("Expected exactly twelve canonical 2025 training files")
    return paths


def phase_views(paths: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    scan = pl.scan_parquet([str(path) for path in paths])
    schema = set(scan.collect_schema().names())
    required = {"PHASE_mvt", *RAW_DEP, *RAW_ARR}
    if not required.issubset(schema):
        raise ValueError("Released movement schema is missing required covariates")
    dep = (scan.filter(pl.col("PHASE_mvt") == "DEP").select(list(RAW_DEP))
           .collect(engine="streaming").to_pandas())
    arr = (scan.filter(pl.col("PHASE_mvt") == "ARR").select(list(RAW_ARR))
           .collect(engine="streaming").to_pandas())
    return dep, arr


def spec() -> dict:
    return {
        "purpose": "Prospective 16-column original runway geometry, true wind, and other-strip traffic family",
        "source": {
            "provider": "OurAirports", "commit": PINNED_COMMIT,
            "data_page": "https://ourairports.com/data/",
            "data_dictionary": "https://ourairports.com/help/data-dictionary.html",
            "repository": "https://github.com/davidmegginson/ourairports-data",
            "license": "Public Domain on provider data page; pinned repository LICENSE is Unlicense",
            "pinned_sha256": PINNED_SOURCE_SHA256,
            "temporal_note": "2024-12-30 snapshot predates both periods; later runway changes may be unrepresented. Current closed flags are excluded.",
        },
        "scientific_motivation": "https://web.mit.edu/hamsa/www/pubs/SimaiakisBalakrishnanTRR.pdf",
        "originality": "Geometry and traffic algorithms in this source are original; no research code or datasets beyond the licensed OurAirports snapshot are reused.",
        "movement_phase_first_allowlist": {"DEP": ["PHASE_mvt", *RAW_DEP], "ARR": ["PHASE_mvt", *RAW_ARR]},
        "weather_allowlist": list(WEATHER_COLS),
        "baseline_id_allowlist": ["MVT_ID_mvt"],
        "forbidden_values": ["BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt", "departure labels", "ranking labels"],
        "features": list(FEATURES), "feature_dtype": "float32",
        "geometry": {
            "airport_runway_matching": "Exact ICAO airport and trimmed-uppercase runway-end identifier only; ambiguous/invalid mapping missing",
            "length_heading": "Endpoint-to-reciprocal haversine length and true initial bearing; reverse end gets reverse true bearing",
            "relation_projection": "Per-airport local equirectangular metric projection from mean runway endpoints",
            "intersecting": "Closed physical line segments intersect; own reciprocal physical strip excluded",
            "near_parallel": "Nonintersecting strips with unoriented bearing difference <=15 degrees and segment distance <=2000 m",
            "wind": "NOAA wind-from direction, exact airport/UTC hour; signed headwind, max(-headwind,0) tailwind, absolute crosswind",
        },
        "counts": {
            "groups": ["other intersecting strips", "other nearby parallel strips"],
            "phases": ["ARR", "DEP"], "windows_seconds": list(WINDOWS_SECONDS),
            "bounds": "Past [query MVT minus window, query MVT); own row and tied-time events excluded",
            "only_event_values": ["airport", "runway", "MVT_TIME_UTC_mvt"],
            "ranking_availability": "Full retrospective ranking batch is released; not a live forecast",
        },
        "missing": {
            "unmapped_runway": "All 16 NaN",
            "missing_query_MVT": "Heading sin/cos, strip length and two strip-neighbor counts remain; three wind and eight traffic counts NaN",
            "missing_invalid_NOAA_wind": "Only the three wind components NaN",
            "valid_group_without_events": "Eight traffic counts are zero",
        },
        "output": "MVT_ID_mvt for exact alignment only, followed by exactly 16 float32 features; no ID as model predictor",
        "input_seal": "Prepare pins helper/spec/source/raw/weather/baseline ID cache/runtime hashes before feature reads; build verifies before and after",
        "output_seal": "Exclusive cache/report names, Parquet readback, exact baseline ID/order and float32/finite-or-NaN checks",
        "memory_floor_gib": MIN_FREE_GIB,
        "validation_status": "No feature cache, model fitting, label comparison, ranking selection, or submission has run; root freezes any later test policy separately.",
    }


def ensure_spec(path: Path = SPEC_PATH) -> Path:
    current = spec()
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved != current:
            raise ValueError("Prospective runway geometry spec differs from source")
    else:
        write_json_exclusive(path, current)
    return path


def require_spec(path: Path = SPEC_PATH) -> Path:
    if not path.is_file():
        raise FileNotFoundError("Run --mode spec and review the frozen prospective specification first")
    if json.loads(path.read_text(encoding="utf-8")) != spec():
        raise ValueError("Prospective runway geometry spec differs from source")
    return path


def source_paths(args: argparse.Namespace) -> list[Path]:
    return [args.source_dir / "source_receipt.json", *(args.source_dir / name for name in PINNED_SOURCE_SHA256)]


def verify_pinned_source(args: argparse.Namespace) -> None:
    receipt = json.loads((args.source_dir / "source_receipt.json").read_text(encoding="utf-8"))
    if receipt.get("commit_sha") != PINNED_COMMIT or receipt.get("commit_committer_utc") >= "2025-01-01T00:00:00Z":
        raise ValueError("OurAirports receipt is not the pinned pre-2025 snapshot")
    for filename, expected in PINNED_SOURCE_SHA256.items():
        if (sha256(args.source_dir / filename) != expected
                or receipt.get("files", {}).get(filename, {}).get("sha256") != expected):
            raise ValueError(f"Pinned OurAirports source changed: {filename}")
    license_text = (args.source_dir / "LICENSE").read_text(encoding="utf-8").lower()
    if "public domain" not in license_text or "unlicense.org" not in license_text:
        raise ValueError("Pinned OurAirports Unlicense terms changed")


def input_paths(args: argparse.Namespace) -> list[Path]:
    ranking = args.data_dir / "ranking.parquet"
    others = [*training_paths(args.data_dir), ranking,
              args.baseline_dir / "training_rows.parquet", args.baseline_dir / "ranking_rows.parquet",
              args.weather_file, require_spec(args.source_dir / "feature_spec.json"),
              Path(__file__).resolve(), *source_paths(args)]
    for path in others:
        if not path.is_file():
            raise FileNotFoundError(path)
    return others


def runtime_snapshot() -> dict:
    return {"python": sys.version.split()[0], "platform": platform.platform(),
            "numpy": np.__version__, "pandas": pd.__version__, "polars": pl.__version__,
            "pyarrow": pyarrow.__version__}


def input_snapshot(args: argparse.Namespace) -> dict:
    verify_pinned_source(args)
    return {"spec_sha256": sha256(require_spec(args.source_dir / "feature_spec.json")),
            "source_script_sha256": sha256(Path(__file__).resolve()),
            "runtime": runtime_snapshot(),
            "input_sha256": {str(path.resolve().relative_to(ROOT) if path.resolve().is_relative_to(ROOT)
                                  else path.resolve()): sha256(path) for path in input_paths(args)}}


def prepare(args: argparse.Namespace) -> dict:
    require_memory(args.min_free_gib)
    value = {"spec": spec(), "snapshot": input_snapshot(args)}
    path = args.output_dir / "protocol.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise ValueError("Existing runway geometry protocol/input/runtime seal differs")
    else:
        write_json_exclusive(path, value)
    return {"protocol": str(path), "protocol_sha256": sha256(path)}


def require_prepared(args: argparse.Namespace) -> tuple[dict, str]:
    path = args.output_dir / "protocol.json"
    if not path.is_file():
        raise FileNotFoundError("Run --mode prepare before any full feature build")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value != {"spec": spec(), "snapshot": input_snapshot(args)}:
        raise ValueError("Runway geometry input/source/runtime seal changed")
    return value, sha256(path)


def build(args: argparse.Namespace) -> dict:
    require_memory(args.min_free_gib)  # First full-build memory gate, before any feature reads.
    protocol, protocol_sha = require_prepared(args)
    outputs = (args.output_dir / "training_runway_geometry_features.parquet",
               args.output_dir / "ranking_runway_geometry_features.parquet",
               args.output_dir / "build_receipt.json")
    if any(path.exists() for path in outputs):
        raise FileExistsError("Runway geometry output is already present; refusing overwrite")
    geometry = load_geometry(args.source_dir)
    weather = pd.read_parquet(args.weather_file, columns=list(WEATHER_COLS))
    with tempfile.TemporaryDirectory(prefix=".runway_geometry_", dir=args.output_dir) as work:
        temporary = Path(work)
        written = []
        for name, paths, baseline_file in (
            ("training", training_paths(args.data_dir), args.baseline_dir / "training_rows.parquet"),
            ("ranking", [args.data_dir / "ranking.parquet"], args.baseline_dir / "ranking_rows.parquet"),
        ):
            require_memory(args.min_free_gib)  # Second and later gates before each all-movement read.
            dep, arr = phase_views(paths)
            ids = pd.read_parquet(baseline_file, columns=["MVT_ID_mvt"])
            result = compute_features(dep, arr, ids, weather, geometry)
            target = temporary / f"{name}_runway_geometry_features.parquet"
            result.to_parquet(target, index=False)
            readback = pd.read_parquet(target)
            verify_output(readback, ids)
            if not readback.equals(result):
                raise ValueError("Runway geometry Parquet readback values changed")
            written.append({"name": name, "rows": len(result), "sha256": sha256(target),
                            "bytes": target.stat().st_size,
                            "geometry_mapped_rows": int(result[FEATURES[0]].notna().sum()),
                            "wind_available_rows": int(result[FEATURES[3]].notna().sum())})
            del dep, arr, ids, result, readback
        if protocol["snapshot"] != input_snapshot(args) or sha256(args.output_dir / "protocol.json") != protocol_sha:
            raise ValueError("Runway geometry input/protocol seal changed during build")
        for item, destination in zip(written, outputs[:2]):
            if destination.exists():
                raise FileExistsError(destination)
            (temporary / f"{item['name']}_runway_geometry_features.parquet").rename(destination)
            if sha256(destination) != item["sha256"]:
                raise ValueError("Published runway geometry cache changed")
    if protocol["snapshot"] != input_snapshot(args) or sha256(args.output_dir / "protocol.json") != protocol_sha:
        raise ValueError("Runway geometry input/protocol seal changed before receipt")
    for item, destination in zip(written, outputs[:2]):
        if not destination.is_file() or sha256(destination) != item["sha256"]:
            raise ValueError("Published runway geometry cache changed before receipt")
    report = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "protocol_sha256": protocol_sha, "features": list(FEATURES),
              "source_script_sha256": sha256(Path(__file__).resolve()),
              "outputs": written, "departure_labels_used": False, "ranking_labels_used": False}
    write_json_exclusive(outputs[2], report)
    return report


def self_test() -> dict:
    """All tests are synthetic: geometry, limits, missingness, and brute-force traffic."""
    a = np.array((0., 0.)); b = np.array((1_000., 0.))
    assert segment_intersects(a, b, np.array((500., -300.)), np.array((500., 300.)))
    assert segment_intersects(a, b, np.array((1_000., 0.)), np.array((1_000., 400.)))
    assert not segment_intersects(a, b, np.array((0., 2_000.)), np.array((1_000., 2_000.)))
    assert segment_distance(a, b, np.array((0., 2_000.)), np.array((1_000., 2_000.))) == 2_000.
    assert segment_distance(a, b, np.array((0., 2_000.01)), np.array((1_000., 2_000.01))) > 2_000.
    assert unoriented_bearing_diff_deg(0., 15.) == 15.
    assert unoriented_bearing_diff_deg(0., 165.) == 15.
    assert unoriented_bearing_diff_deg(0., 15.01) > 15.
    assert strip_relation(a, b, np.array((500., -300.)), np.array((500., 300.)), 0., 90.) == "intersecting"
    assert strip_relation(a, b, np.array((0., 2_000.)), np.array((1_000., 2_000.)), 0., 15.) == "near_parallel"
    assert strip_relation(a, b, np.array((0., 2_000.01)), np.array((1_000., 2_000.01)), 0., 15.) is None
    assert strip_relation(a, b, np.array((0., 2_000.)), np.array((1_000., 2_000.)), 0., 15.01) is None
    airports = pd.DataFrame({"ident": AIRPORTS, "gps_code": AIRPORTS})
    runways = pd.DataFrame([
        ("EDDF", "09", "27", 0., -0.01, 0., 0.01),
        ("EDDF", "18", "36", -0.01, 0., 0.01, 0.),
        ("EDDF", "09L", "27R", 0.01, -0.01, 0.01, 0.01),
        ("EDDF", "09F", "27F", 0.03, -0.01, 0.03, 0.01),
    ], columns=["airport_ident", "le_ident", "he_ident", "le_latitude_deg",
                "le_longitude_deg", "he_latitude_deg", "he_longitude_deg"])
    geom = build_geometry_from_frames(airports, runways)
    own = geom.code_map[("EDDF", "09")]
    reverse = geom.code_map[("EDDF", "27")]
    assert own is not None and reverse is not None and own[0] == reverse[0]
    assert math.isclose(geom.strips[("EDDF", own[0])].length_m,
                        great_circle_m(0., -0.01, 0., 0.01), abs_tol=1e-7)
    assert math.isclose(own[1], 90., abs_tol=1e-7) and math.isclose(reverse[1], 270., abs_tol=1e-7)
    assert geom.neighbors[("EDDF", own[0])]["intersecting"] == (1,)
    assert geom.neighbors[("EDDF", own[0])]["near_parallel"] == (2,)
    head, tail, crosswind = wind_components(
        np.array([90., 90., 90., 90.]), np.array([90., 270., 180., np.nan]),
        np.array([10., 10., 10., 10.]))
    assert np.allclose(head[:3], [10., -10., 0.], atol=1e-8)
    assert np.allclose(tail[:3], [0., 10., 0.], atol=1e-8)
    assert np.allclose(crosswind[:3], [0., 0., 10.], atol=1e-8)
    assert np.isnan(head[3]) and np.isnan(tail[3]) and np.isnan(crosswind[3])
    base = pd.Timestamp("2025-01-01T12:00:00Z")
    dep = pd.DataFrame([
        (1., "EDDF", "09", base),
        (2., "EDDF", "27", base + pd.Timedelta(minutes=1)),
        (3., "EDDF", "18", base - pd.Timedelta(minutes=15)),
        (4., "EDDF", "09L", base - pd.Timedelta(minutes=60)),
        (5., "EDDF", "09F", base - pd.Timedelta(minutes=2)),
        (6., "EDDF", "09", pd.NaT),
        (7., "EDDF", "?", base),
    ], columns=RAW_DEP)
    arr = pd.DataFrame([
        (10., "EDDF", "18", base - pd.Timedelta(minutes=60)),
        (11., "EDDF", "18", base - pd.Timedelta(minutes=15)),
        (12., "EDDF", "18", base),  # Tied time must not count.
        (13., "EDDF", "09L", base - pd.Timedelta(minutes=14)),
    ], columns=RAW_ARR)
    ids = pd.DataFrame({"MVT_ID_mvt": dep.MVT_ID_mvt.iloc[::-1].to_numpy()})
    wx = pd.DataFrame({"airport": ["EDDF"], "weather_hour_utc": [base.floor("h")],
                       "wx_wind_direction_deg": [90.], "wx_wind_speed_mps": [10.]})
    features = compute_features(dep, arr, ids, wx, geom)
    row = features.set_index("MVT_ID_mvt")
    assert row.loc[1., "rgeom_intersecting_arr_past15_count"] == 1
    assert row.loc[1., "rgeom_intersecting_arr_past60_count"] == 2
    assert row.loc[1., "rgeom_near_parallel_arr_past15_count"] == 1
    assert row.loc[1., "rgeom_near_parallel_dep_past60_count"] == 1
    assert row.loc[1., "rgeom_intersecting_dep_past15_count"] == 1
    assert row.loc[1., "rgeom_headwind_mps"] == 10
    assert row.loc[2., "rgeom_headwind_mps"] == -10
    assert np.isnan(row.loc[6., "rgeom_intersecting_arr_past15_count"])
    assert np.isnan(row.loc[6., "rgeom_headwind_mps"])
    assert np.isfinite(row.loc[6., "rgeom_strip_length_m"])
    assert row.loc[7., list(FEATURES)].isna().all()
    far_future = pd.Series([pd.Timestamp("3000-01-01T00:00:00Z"),
                            pd.Timestamp.max.tz_localize("UTC"),
                            pd.Timestamp.min.tz_localize("UTC")])
    _, future_valid, safe_stamps = timestamp_ns(far_future)
    assert not future_valid.any() and safe_stamps.isna().all()
    fine_stamps = pd.Series([base - pd.Timedelta(nanoseconds=100), base, base])
    fine_ns, fine_valid, _ = timestamp_ns(fine_stamps)
    assert fine_valid.all() and fine_ns[1] - fine_ns[0] == 100 and fine_ns[1] == fine_ns[2]
    fine_arr = pd.DataFrame([
        (20., "EDDF", "18", base - pd.Timedelta(nanoseconds=100)),
        (21., "EDDF", "18", base),
    ], columns=RAW_ARR)
    fine_dep = dep.loc[dep.MVT_ID_mvt.eq(1.)].copy()
    fine_feature = compute_features(fine_dep, fine_arr, fine_dep[["MVT_ID_mvt"]], wx, geom)
    assert fine_feature.loc[0, "rgeom_intersecting_arr_past15_count"] == 1
    bad_wx = wx.copy()
    bad_wx.loc[0, "wx_wind_direction_deg"] = np.nan
    without_direction = compute_features(dep, arr, ids, bad_wx, geom)
    assert without_direction.loc[:, list(FEATURES[3:6])].isna().all().all()
    pd.testing.assert_frame_equal(without_direction.drop(columns=list(FEATURES[3:6])),
                                  features.drop(columns=list(FEATURES[3:6])))
    bad_wx.loc[0, "wx_wind_direction_deg"] = 90.
    bad_wx.loc[0, "wx_wind_speed_mps"] = -1.
    assert compute_features(dep, arr, ids, bad_wx, geom).loc[:, list(FEATURES[3:6])].isna().all().all()
    try:
        require_memory(0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("Memory floor can be disabled")
    # Randomized brute-force parity across both physical-strip classes and both phases.
    rng = np.random.default_rng(20261003)
    codes = np.array(["09", "18", "09L", "09F"])
    rows_dep, rows_arr = [], []
    for i in range(90):
        code = str(rng.choice(codes))
        stamp = base + pd.Timedelta(seconds=int(rng.integers(-5_000, 5_001)))
        rows_dep.append((float(i + 100), "EDDF", code, stamp))
        rows_arr.append((float(i + 1_000), "EDDF", code, stamp))
    many_dep = pd.DataFrame(rows_dep, columns=RAW_DEP)
    many_arr = pd.DataFrame(rows_arr, columns=RAW_ARR)
    many_ids = many_dep[["MVT_ID_mvt"]].copy()
    calculated = compute_features(many_dep, many_arr, many_ids, wx, geom)
    for i, query in many_dep.iterrows():
        match = geom.code_map[("EDDF", query.RUNWAY_mvt)]
        assert match is not None
        related = geom.neighbors[("EDDF", match[0])]
        for relation in ("intersecting", "near_parallel"):
            for phase, events in (("ARR", many_arr), ("DEP", many_dep)):
                for window in WINDOWS_SECONDS:
                    start = query.MVT_TIME_UTC_mvt - pd.Timedelta(seconds=window)
                    expected = sum(
                        1 for _, event in events.iterrows()
                        if (geom.code_map[("EDDF", event.RUNWAY_mvt)][0] in related[relation]
                            and start <= event.MVT_TIME_UTC_mvt < query.MVT_TIME_UTC_mvt))
                    name = f"rgeom_{relation}_{phase.lower()}_past{window // 60}_count"
                    if calculated.loc[i, name] != expected:
                        raise AssertionError(f"Synthetic brute-force parity failed: {name} row {i}")
    return {"synthetic_rows": len(many_dep), "feature_count": len(FEATURES),
            "checks": ["directional great-circle endpoints", "intersection/parallel limits",
                       "reciprocal exclusion", "strict time windows/ties", "missingness",
                       "wind sign", "random brute-force traffic parity"], "passed": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("spec", "prepare", "build", "self-test"))
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--weather-file", type=Path, default=DEFAULT_WEATHER)
    parser.add_argument("--min-free-gib", type=float, default=MIN_FREE_GIB)
    args = parser.parse_args()
    if args.mode == "spec":
        path = ensure_spec(args.source_dir / "feature_spec.json")
        result = {"spec": str(path), "sha256": sha256(path), "features": len(FEATURES)}
    elif args.mode == "self-test":
        result = self_test()
    elif args.mode == "prepare":
        result = prepare(args)
    else:
        result = build(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
