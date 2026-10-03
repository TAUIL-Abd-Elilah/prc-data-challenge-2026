"""Released departure interval-flow context, without departure labels.

Every event and query comes from the released DEP movement fields.  AOBT_3_flt
is an NM off-block proxy, not an observed taxi start.  The complete batch is
used retrospectively, so these features do not claim real-time availability.
No model, label, leaderboard result, or opaque movement-ID ordering is used.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from runway_arrival_features import normalized


RAW_DEP = ("MVT_ID_mvt", "ADEP_mvt", "RUNWAY_mvt",
           "MVT_TIME_UTC_mvt", "AOBT_3_flt")
COUNT_KINDS = ("takeoffs_between", "proxy_starts_between",
               "active_at_takeoff", "overtakers", "left_behind")
FEATURES = tuple(f"taxi_flow_{scope}_{kind}_count"
                 for scope in ("airport", "runway") for kind in COUNT_KINDS)
NS_PER_SECOND = 1_000_000_000
MAX_PROXY_NS = 7_200 * NS_PER_SECOND
DEFAULT_OUTPUT = Path("artifacts/v11-taxi-flow")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
    if not np.isfinite(min_free_gib) or min_free_gib < 0:
        raise ValueError("Minimum free-memory threshold must be finite and nonnegative")
    free = available_memory_gib()
    if free < min_free_gib:
        raise MemoryError(f"Need {min_free_gib:g} GiB free for full feature build; "
                          f"currently {free:.2f} GiB")


def source_paths(data_dir: Path) -> tuple[list[Path], Path]:
    training = sorted(path for path in data_dir.glob("training_2025-*.parquet")
                      if re.fullmatch(
                          r"training_2025-\d{2}-01_20\d{2}-\d{2}-01\.parquet",
                          path.name))
    if len(training) != 12 or len({path.name for path in training}) != 12:
        raise ValueError("Expected exactly twelve canonical 2025 training files")
    ranking = data_dir / "ranking.parquet"
    if not ranking.is_file():
        raise FileNotFoundError(ranking)
    return training, ranking


def protocol_spec() -> dict:
    return {
        "purpose": "Fixed label-free retrospective taxi interval-flow features",
        "scientific_source": {
            "inspiration": "MIT aircraft taxi queue study",
            "url": "https://dspace.mit.edu/entities/publication/f0891972-627a-409e-9fbc-2e12fe4ba28b",
            "qualification": "Original released-NM proxy features; not a true measured taxi queue. No paper code or data reused.",
        },
        "raw_phase_filter": "PHASE_mvt == DEP before selecting any feature columns",
        "raw_dep_allowlist": list(RAW_DEP),
        "raw_dep_forbidden": ["BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt"],
        "id_use": "Exact alignment, uniqueness, and self-exclusion only; never a predictor",
        "normalization": "Airport and runway strip/uppercase using runway_arrival_features.normalized; missing/placeholders invalid",
        "quality_valid_interval": "finite AOBT_3_flt and MVT timestamps, 0 <= MVT-AOBT <= 7200 seconds",
        "query": "Own quality-valid interval [s=AOBT_3_flt,t=MVT]; otherwise every count is NaN",
        "scopes": ["normalized airport", "normalized airport plus runway"],
        "counts": {
            "takeoffs_between": "other MVT strictly s < MVT < t; other AOBT may be missing",
            "proxy_starts_between": "quality-valid other interval, strictly s < AOBT < t",
            "active_at_takeoff": "quality-valid other interval, AOBT <= t < MVT",
            "overtakers": "quality-valid other interval, AOBT > s and MVT < t",
            "left_behind": "quality-valid other interval, AOBT < s and MVT > t",
        },
        "boundaries": "All ten counts exclude the own row, including tied times; zero own duration gives zero takeoffs/starts/overtakers but may have active/left-behind",
        "missing_group": "Invalid airport gives NaN in both scopes; invalid runway gives NaN in runway scope only",
        "features": list(FEATURES),
        "dtype": "float32",
        "ranking_availability": "Complete released DEP batch available retrospectively; not a real-time forecast",
        "departure_labels_used": False,
        "validation_policy": "No model fit or labels in this builder; May/September labels remain reserved",
    }


def _source_snapshot(args: argparse.Namespace) -> dict:
    training, ranking = source_paths(args.data_dir)
    baseline_training = args.cache_dir / "training_rows.parquet"
    baseline_ranking = args.cache_dir / "ranking_rows.parquet"
    for path in (baseline_training, baseline_ranking):
        if not path.is_file():
            raise FileNotFoundError(path)
    return {
        "builder": sha256(Path(__file__).resolve()),
        "normalization": sha256(Path(__file__).resolve().parent /
                                "runway_arrival_features.py"),
        "raw_training": {path.name: sha256(path) for path in training},
        "raw_ranking": sha256(ranking),
        "baseline_training_rows": sha256(baseline_training),
        "baseline_ranking_rows": sha256(baseline_ranking),
    }


def _write_json_exclusive(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as out:
        json.dump(value, out, indent=2)
        out.write("\n")


def prepare(args: argparse.Namespace) -> dict:
    """Freeze all sources before any Parquet feature values are read."""
    value = {"spec": protocol_spec(), "source_sha256": _source_snapshot(args)}
    path = args.output_dir / "protocol.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise ValueError("Existing frozen protocol differs from current sources")
    else:
        try:
            _write_json_exclusive(path, value)
        except FileExistsError:
            if json.loads(path.read_text(encoding="utf-8")) != value:
                raise ValueError("Concurrent frozen protocol differs from current sources")
    return value


def _assert_sources(args: argparse.Namespace, frozen: dict) -> None:
    if _source_snapshot(args) != frozen["source_sha256"]:
        raise ValueError("A builder, normalization, raw, or baseline source changed")
    if json.loads((args.output_dir / "protocol.json").read_text(
            encoding="utf-8")) != frozen:
        raise ValueError("Frozen protocol changed")


def phase_view(scan: pl.LazyFrame) -> pl.LazyFrame:
    """The DEP filter precedes the strict covariate projection."""
    return scan.filter(pl.col("PHASE_mvt") == "DEP").select(list(RAW_DEP))


def _nanoseconds(values: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    stamp = pd.to_datetime(values, utc=True, errors="coerce")
    # Pandas 3 may retain microsecond-resolution years outside the int64-ns
    # range. Mark those as missing before conversion rather than raising.
    within_ns = stamp.between(pd.Timestamp.min.tz_localize("UTC"),
                              pd.Timestamp.max.tz_localize("UTC"))
    bounded = stamp.where(within_ns, pd.NaT)
    return (bounded.dt.as_unit("ns").astype("int64").to_numpy(),
            within_ns.to_numpy(dtype=bool))


class Fenwick:
    """Integer prefix counts over precompressed time coordinates."""

    def __init__(self, size: int):
        self.tree = np.zeros(size + 1, dtype=np.int64)

    def add(self, coordinate: int) -> None:
        pos = coordinate + 1
        while pos < len(self.tree):
            self.tree[pos] += 1
            pos += pos & -pos

    def prefix(self, count: int) -> int:
        """Number of inserted events in compressed coordinates [0,count)."""
        total = 0
        pos = count
        while pos:
            total += int(self.tree[pos])
            pos -= pos & -pos
        return total


def _overtakers(starts: np.ndarray, ends: np.ndarray,
                query_s: np.ndarray, query_t: np.ndarray) -> np.ndarray:
    """Count event starts > query s and event ends < query t in O(n log n)."""
    result = np.zeros(len(query_s), dtype=np.int64)
    if not len(starts):
        return result
    coordinate = np.unique(starts)
    event_order = np.argsort(ends, kind="stable")
    query_order = np.argsort(query_t, kind="stable")
    event_positions = np.searchsorted(coordinate, starts)
    query_prefix = np.searchsorted(coordinate, query_s, side="right")
    tree = Fenwick(len(coordinate))
    cursor = 0
    for qi in query_order:
        while cursor < len(event_order) and ends[event_order[cursor]] < query_t[qi]:
            event = event_order[cursor]
            tree.add(int(event_positions[event]))
            cursor += 1
        result[qi] = cursor - tree.prefix(int(query_prefix[qi]))
    return result


def _left_behind(starts: np.ndarray, ends: np.ndarray,
                 query_s: np.ndarray, query_t: np.ndarray) -> np.ndarray:
    """Count event starts < query s and event ends > query t in O(n log n)."""
    result = np.zeros(len(query_s), dtype=np.int64)
    if not len(starts):
        return result
    coordinate = np.unique(ends)
    event_order = np.argsort(starts, kind="stable")
    query_order = np.argsort(query_s, kind="stable")
    event_positions = np.searchsorted(coordinate, ends)
    query_prefix = np.searchsorted(coordinate, query_t, side="right")
    tree = Fenwick(len(coordinate))
    cursor = 0
    for qi in query_order:
        while cursor < len(event_order) and starts[event_order[cursor]] < query_s[qi]:
            event = event_order[cursor]
            tree.add(int(event_positions[event]))
            cursor += 1
        result[qi] = cursor - tree.prefix(int(query_prefix[qi]))
    return result


def _scope_counts(group_code: np.ndarray, group_valid: np.ndarray,
                  valid_mvt: np.ndarray, valid_interval: np.ndarray,
                  starts: np.ndarray, ends: np.ndarray,
                  out: dict[str, np.ndarray], scope: str) -> None:
    indices = np.flatnonzero(group_valid)
    groups = pd.Series(np.arange(len(indices))).groupby(
        group_code[indices], sort=False).indices
    names = [f"taxi_flow_{scope}_{kind}_count" for kind in COUNT_KINDS]
    for local in groups.values():
        group = indices[np.asarray(local, dtype=np.int64)]
        query = group[valid_interval[group]]
        if not len(query):
            continue
        event_mvt = group[valid_mvt[group]]
        event_interval = group[valid_interval[group]]
        s, t = starts[query], ends[query]
        all_takeoffs = np.sort(ends[event_mvt])
        valid_starts = np.sort(starts[event_interval])
        valid_ends = np.sort(ends[event_interval])
        counts = (
            np.maximum(0, np.searchsorted(all_takeoffs, t, side="left")
                       - np.searchsorted(all_takeoffs, s, side="right")),
            np.maximum(0, np.searchsorted(valid_starts, t, side="left")
                       - np.searchsorted(valid_starts, s, side="right")),
            np.searchsorted(valid_starts, t, side="right")
            - np.searchsorted(valid_ends, t, side="right"),
            _overtakers(starts[event_interval], ends[event_interval], s, t),
            _left_behind(starts[event_interval], ends[event_interval], s, t),
        )
        # The own end equals t and own start equals s. Strict bounds above
        # therefore exclude the own event in every count, even with ties.
        if any((value < 0).any() for value in counts):
            raise ValueError("Negative flow count from valid intervals")
        for name, value in zip(names, counts):
            out[name][query] = value.astype(np.float32)


def build_from_dep(dep: pd.DataFrame) -> pd.DataFrame:
    if list(dep) != list(RAW_DEP):
        raise ValueError("Raw DEP frame differs from strict covariate allowlist")
    if dep.MVT_ID_mvt.isna().any() or dep.MVT_ID_mvt.duplicated().any():
        raise ValueError("DEP movement IDs must be unique and non-null")
    n = len(dep)
    out = {name: np.full(n, np.nan, dtype=np.float32) for name in FEATURES}
    airport, good_airport = normalized(dep.ADEP_mvt)
    runway, good_runway = normalized(dep.RUNWAY_mvt)
    ends, valid_mvt = _nanoseconds(dep.MVT_TIME_UTC_mvt)
    starts, valid_start = _nanoseconds(dep.AOBT_3_flt)
    both_time = valid_mvt & valid_start
    # For ordered signed int64 timestamps, unsigned subtraction represents
    # the exact mathematical difference across the entire ns epoch range.
    # Signed subtraction would wrap for Timestamp.min/max pairs.
    ordered = both_time & (ends >= starts)
    duration = np.zeros(n, dtype=np.uint64)
    duration[ordered] = (ends[ordered].view(np.uint64)
                         - starts[ordered].view(np.uint64))
    valid_interval = ordered & (duration <= np.uint64(MAX_PROXY_NS))
    airport_code, _ = pd.factorize(airport, sort=False)
    runway_code, _ = pd.factorize(pd.MultiIndex.from_arrays(
        [airport, runway]), sort=False)
    _scope_counts(airport_code, good_airport, valid_mvt, valid_interval,
                  starts, ends, out, "airport")
    _scope_counts(runway_code, good_airport & good_runway, valid_mvt,
                  valid_interval, starts, ends, out, "runway")
    result = pd.DataFrame(out)
    if (list(result) != list(FEATURES) or
            any(result[name].dtype != np.dtype("float32") for name in FEATURES) or
            np.isinf(result.to_numpy()).any()):
        raise ValueError("Feature schema, dtype, or values invalid")
    for scope, valid_group in (("airport", good_airport),
                               ("runway", good_airport & good_runway)):
        expected_finite = valid_interval & valid_group
        columns = [f"taxi_flow_{scope}_{kind}_count" for kind in COUNT_KINDS]
        if not np.array_equal(result[columns].notna().all(axis=1).to_numpy(),
                              expected_finite):
            raise ValueError(f"{scope} feature validity does not match protocol")
    return result


def _read_aligned(paths: list[Path], baseline: Path) -> pd.DataFrame:
    dep = phase_view(pl.scan_parquet([str(path) for path in paths])).collect(
    ).to_pandas()
    ids = pd.read_parquet(baseline, columns=["MVT_ID_mvt"])
    if (ids.MVT_ID_mvt.isna().any() or ids.MVT_ID_mvt.duplicated().any() or
            dep.MVT_ID_mvt.isna().any() or dep.MVT_ID_mvt.duplicated().any() or
            len(ids) != len(dep)):
        raise ValueError("Raw/baseline departure IDs must be unique, complete, and non-null")
    aligned = ids.merge(dep, on="MVT_ID_mvt", how="left", sort=False,
                        validate="one_to_one", indicator="_source_match")
    if (len(aligned) != len(ids) or
            not aligned._source_match.eq("both").all() or
            not np.array_equal(aligned.MVT_ID_mvt.to_numpy(),
                               ids.MVT_ID_mvt.to_numpy())):
        raise ValueError("Raw departure IDs do not match baseline IDs/order")
    return aligned[list(RAW_DEP)]


def _build_one(paths: list[Path], baseline: Path, stage: Path,
               min_free_gib: float) -> dict:
    require_memory(min_free_gib)
    dep = _read_aligned(paths, baseline)
    values = build_from_dep(dep)
    values.insert(0, "MVT_ID_mvt", dep.MVT_ID_mvt.to_numpy(copy=True))
    if (values.MVT_ID_mvt.isna().any() or
            not np.array_equal(values.MVT_ID_mvt.to_numpy(),
                               dep.MVT_ID_mvt.to_numpy())):
        raise ValueError("Built feature rows differ from exact baseline ID/order")
    values.to_parquet(stage, index=False)
    return {"rows": len(values), "feature_names": list(FEATURES),
            "feature_dtypes": {name: str(values[name].dtype) for name in FEATURES},
            "nonnull_fraction": {name: float(values[name].notna().mean())
                                 for name in FEATURES},
            "sha256": sha256(stage)}


def build(args: argparse.Namespace) -> None:
    if not np.isfinite(args.min_free_gib) or args.min_free_gib < 4.0:
        raise ValueError("Full build memory floor cannot be below 4 GiB")
    require_memory(args.min_free_gib)
    frozen = prepare(args)
    _assert_sources(args, frozen)
    training, ranking = source_paths(args.data_dir)
    final_training = args.output_dir / "training_taxi_interval_flow_features.parquet"
    final_ranking = args.output_dir / "ranking_taxi_interval_flow_features.parquet"
    manifest = args.output_dir / "feature_build.json"
    if any(path.exists() for path in (final_training, final_ranking, manifest)):
        raise FileExistsError("Flow feature output already exists; never overwrite")
    with tempfile.TemporaryDirectory(prefix=".taxi_flow_", dir=args.output_dir) as work:
        temp = Path(work)
        train_stage = temp / "training.parquet"
        rank_stage = temp / "ranking.parquet"
        train_report = _build_one(training,
                                  args.cache_dir / "training_rows.parquet",
                                  train_stage, args.min_free_gib)
        rank_report = _build_one([ranking],
                                 args.cache_dir / "ranking_rows.parquet",
                                 rank_stage, args.min_free_gib)
        _assert_sources(args, frozen)
        # Hard links publish bytes exclusively on the same filesystem; this
        # refuses overwrite even if a competing process creates a path now.
        os.link(train_stage, final_training)
        os.link(rank_stage, final_ranking)
    if (sha256(final_training) != train_report["sha256"] or
            sha256(final_ranking) != rank_report["sha256"]):
        raise ValueError("Published output bytes differ from staged output")
    _assert_sources(args, frozen)
    _write_json_exclusive(manifest, {
        "protocol_sha256": sha256(args.output_dir / "protocol.json"),
        "training": train_report, "ranking": rank_report,
        "departure_labels_used": False,
        "source_sha256": frozen["source_sha256"],
    })
    print(json.dumps({"training_rows": train_report["rows"],
                      "ranking_rows": rank_report["rows"],
                      "feature_count": len(FEATURES)}, indent=2))


def _brute_force(dep: pd.DataFrame) -> pd.DataFrame:
    """Only used on a tiny manufactured dataset, independent of sweeps."""
    airport, valid_airport = normalized(dep.ADEP_mvt)
    runway, valid_runway = normalized(dep.RUNWAY_mvt)
    end, valid_end = _nanoseconds(dep.MVT_TIME_UTC_mvt)
    start, valid_start = _nanoseconds(dep.AOBT_3_flt)
    good = valid_start & valid_end
    for index in np.flatnonzero(good):
        exact_duration = int(end[index]) - int(start[index])
        if not 0 <= exact_duration <= MAX_PROXY_NS:
            good[index] = False
    out = {name: np.full(len(dep), np.nan, dtype=np.float32)
           for name in FEATURES}
    for i in range(len(dep)):
        if not good[i] or not valid_airport[i]:
            continue
        for scope in ("airport", "runway"):
            if scope == "runway" and not valid_runway[i]:
                continue
            count = np.zeros(5, dtype=np.int64)
            for j in range(len(dep)):
                if i == j or not valid_airport[j] or airport[i] != airport[j]:
                    continue
                if scope == "runway" and (
                        not valid_runway[j] or runway[i] != runway[j]):
                    continue
                if valid_end[j] and start[i] < end[j] < end[i]:
                    count[0] += 1
                if not good[j]:
                    continue
                if start[i] < start[j] < end[i]:
                    count[1] += 1
                if start[j] <= end[i] < end[j]:
                    count[2] += 1
                if start[j] > start[i] and end[j] < end[i]:
                    count[3] += 1
                if start[j] < start[i] and end[j] > end[i]:
                    count[4] += 1
            for kind, value in zip(COUNT_KINDS, count):
                out[f"taxi_flow_{scope}_{kind}_count"][i] = value
    return pd.DataFrame(out)


def synthetic_check() -> None:
    """Deterministic manufactured rows; no competition files or labels read."""
    base = pd.Timestamp("2025-01-01T00:00:00Z")

    def row(identifier: int, start: int | None, end: int | None,
            airport: str = " KAAA ", runway: str = " 09 ",
            phase: str = "DEP") -> dict:
        return {"PHASE_mvt": phase, "MVT_ID_mvt": identifier,
                "ADEP_mvt": airport, "RUNWAY_mvt": runway,
                "MVT_TIME_UTC_mvt": (base + pd.Timedelta(seconds=end)
                                     if end is not None else pd.NaT),
                "AOBT_3_flt": (base + pd.Timedelta(seconds=start)
                               if start is not None else pd.NaT),
                "BLOCK_TIME_UTC_mvt": base + pd.Timedelta(days=10),
                "TAXITIME_SEC_mvt": 999_999.0}

    raw = pd.DataFrame([
        row(1, 0, 10),       # ordinary query
        row(2, 1, 3),        # overtaker
        row(3, -1, 20),      # left behind and active
        row(4, 3, 8),        # overtaker with tied earlier end
        row(5, None, 5),     # missing AOBT: takeoffs only
        row(6, 0, 0),        # zero own interval, tied with query start
        row(7, 10, 10),      # equal query end, not between
        row(8, 2, 12, runway="27"),
        row(9, 2, 12, airport="KBBB"),
        row(10, 2, 12, runway="?"),
        row(11, 2, 12, airport="?"),
        row(12, 5, 5),       # zero taxi at strict interior time
        row(13, 0, 10),      # identical interval, self/tie test
        row(1, 0, 10, phase="ARR"),  # duplicate DEP ID must disappear with phase filter
        row(15, 0, 8000),    # invalid other interval, takeoff not between
        row(16, None, 6, runway="27"),
        row(17, 4, None),
        row(18, -10, 4),
        row(19, 5, 5, runway="?"),
    ])
    projected = phase_view(pl.from_pandas(raw).lazy()).collect().to_pandas()
    if (list(projected) != list(RAW_DEP) or len(projected) != len(raw) - 1 or
            any(name in projected for name in
                ("BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt"))):
        raise AssertionError("DEP phase/projection provenance failed")
    tampered = raw.copy()
    tampered["BLOCK_TIME_UTC_mvt"] = base - pd.Timedelta(days=10)
    tampered["TAXITIME_SEC_mvt"] = -999_999.0
    tampered.loc[tampered.PHASE_mvt == "ARR", "AOBT_3_flt"] = (
        base + pd.Timedelta(days=10))
    pd.testing.assert_frame_equal(
        projected, phase_view(pl.from_pandas(tampered).lazy()).collect().to_pandas())
    actual = build_from_dep(projected)
    expected = _brute_force(projected)
    if not actual.equals(expected):
        difference = [name for name in FEATURES
                      if not actual[name].equals(expected[name])]
        raise AssertionError(f"Fenwick/sweep differs from brute force: {difference}")
    by_id = dict(zip(projected.MVT_ID_mvt, range(len(projected))))
    def count(identifier: int, scope: str, kind: str) -> float:
        return float(actual.loc[by_id[identifier],
                                f"taxi_flow_{scope}_{kind}_count"])
    if (count(1, "airport", "takeoffs_between") != 7 or
            count(1, "runway", "takeoffs_between") != 5 or
            count(1, "runway", "overtakers") != 3 or
            count(1, "runway", "left_behind") != 1 or
            count(6, "airport", "takeoffs_between") != 0 or
            count(6, "airport", "proxy_starts_between") != 0 or
            count(6, "airport", "overtakers") != 0 or
            not np.isnan(count(10, "runway", "takeoffs_between")) or
            not np.isfinite(count(10, "airport", "takeoffs_between")) or
            not np.isnan(count(11, "airport", "takeoffs_between")) or
            not np.isnan(count(5, "airport", "takeoffs_between"))):
        raise AssertionError("Explicit boundary/missing-value expectations failed")
    try:
        duplicate = pd.concat([projected, projected.iloc[[0]]], ignore_index=True)
        build_from_dep(duplicate)
    except ValueError as error:
        if "unique" not in str(error):
            raise
    else:
        raise AssertionError("Duplicate IDs were accepted")
    rng = np.random.default_rng(20261003)
    random_rows = []
    for k in range(160):
        start = (None if k % 13 == 0 else
                 float(rng.integers(-6, 18)) + (0.5 if k % 7 == 0 else 0.0))
        if k % 17 == 0:
            end = None
        elif start is None:
            end = float(rng.integers(-6, 18))
        else:
            end = start + float(rng.choice([-2, 0, 0, 1, 3, 7_200, 7_201]))
        random_rows.append(row(
            1000 + k, start, end,
            airport=str(rng.choice(["KAAA", "kaaa ", "KBBB", "?"])),
            runway=str(rng.choice(["09", " 09 ", "27", "?"]))))
    random_dep = phase_view(pl.from_pandas(pd.DataFrame(random_rows)).lazy(
    )).collect().to_pandas()
    random_actual = build_from_dep(random_dep)
    random_expected = _brute_force(random_dep)
    if not random_actual.equals(random_expected):
        raise AssertionError("Randomized manufactured boundary/tie proof failed")
    minimum = pd.Timestamp.min.tz_localize("UTC")
    maximum = pd.Timestamp.max.tz_localize("UTC")
    far_future = pd.Timestamp("3000-01-01T00:00:00Z")
    far_past = pd.Timestamp("1600-01-01T00:00:00Z")
    extreme = pd.DataFrame([
        (2001, "KAAA", "09", minimum, maximum),  # negative mathematical span
        (2002, "KAAA", "09", maximum, minimum),  # positive > uint64/2 span
        (2003, "KAAA", "09", minimum + pd.Timedelta(seconds=1), minimum),
        (2004, "KAAA", "09", maximum, maximum - pd.Timedelta(seconds=1)),
        (2005, "KAAA", "09", far_future, minimum),
        (2006, "KAAA", "09", maximum, far_past),
    ], columns=list(RAW_DEP))
    extreme_actual = build_from_dep(extreme)
    extreme_expected = _brute_force(extreme)
    if not extreme_actual.equals(extreme_expected):
        raise AssertionError("Extreme timestamp counts differ from exact arithmetic")
    if (not extreme_actual.iloc[[0, 1, 4, 5]].isna().all().all()
            or not extreme_actual.iloc[[2, 3]].notna().all().all()):
        raise AssertionError("Out-of-range or wrapped timestamps were accepted")
    for invalid_minimum in (float("nan"), float("inf"), float("-inf")):
        try:
            require_memory(invalid_minimum)
        except ValueError:
            pass
        else:
            raise AssertionError("Nonfinite memory gate was accepted")
    print(json.dumps({"synthetic": "pass", "rows": len(projected),
                      "random_rows": len(random_dep), "extreme_rows": len(extreme),
                      "features": len(FEATURES),
                      "provenance": "DEP allowlist, no BLOCK/TAXI",
                      "comparison": "exact brute-force counts including NaN mask"}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("show-spec", "synthetic", "prepare", "build"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/baseline"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-free-gib", type=float, default=4.0)
    args = parser.parse_args()
    if args.mode == "show-spec":
        print(json.dumps(protocol_spec(), indent=2))
    elif args.mode == "synthetic":
        synthetic_check()
    elif args.mode == "prepare":
        value = prepare(args)
        print(json.dumps({"protocol": str(args.output_dir / "protocol.json"),
                          "source_sha256": value["source_sha256"]}, indent=2))
    else:
        build(args)


if __name__ == "__main__":
    main()
