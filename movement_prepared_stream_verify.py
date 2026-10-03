"""Bounded independent rebuild and exact audit of the movement-only feature cache.

``contract`` and ``preview`` do not read training values. ``verify`` is a
separate, opt-in exact rebuild: it never changes the frozen movement source or
protocol and creates the accepted seal only after every comparison passes.
The public aligned iterator is also used by saved-model inference.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
from itertools import zip_longest
import json
from pathlib import Path
import tempfile
from typing import Iterator

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

import movement_only_expert as movement
from solution import AIRPORT_TZ, _training_files
from weather_model import add_weather


RAW_COLUMNS = ("MVT_ID_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
               "FLIGHT_mvt")
ROW_COLUMNS = ("MVT_ID_mvt", "airport", "time")
INFERENCE_ROW_COLUMNS = ("MVT_ID_mvt", "target", "proxy", "month", "airport", "time")
BATCH_SIZE = 32768


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_global_category_dtypes(path: Path, names: list[str] | tuple[str, ...],
                                abort_free_gib: float | None = None
                                ) -> dict[str, pd.CategoricalDtype]:
    """Read one full category column at a time to preserve global levels/order."""
    result = {}
    for name in names:
        if (abort_free_gib is not None and
                movement.available_memory_gib() < abort_free_gib):
            raise MemoryError("Streaming category scan aborted before low-memory pressure")
        values = pd.read_parquet(path, columns=[name])[name]
        if not isinstance(values.dtype, pd.CategoricalDtype):
            raise ValueError(f"{path}: {name} is no longer categorical")
        result[name] = pd.CategoricalDtype(
            categories=values.cat.categories.copy(), ordered=values.cat.ordered)
        del values
    return result


class FrameCursor:
    """Take exactly n rows across arbitrary bounded input batch boundaries."""

    def __init__(self, source: Iterator[pd.DataFrame], columns: list[str],
                 categorical_dtypes: dict[str, pd.CategoricalDtype] | None = None):
        self.source = iter(source)
        self.columns = columns
        self.categorical_dtypes = categorical_dtypes or {}
        self.buffer: pd.DataFrame | None = None

    def _fill(self) -> bool:
        if self.buffer is not None and len(self.buffer):
            return True
        for frame in self.source:
            if list(frame) != self.columns:
                raise ValueError("Streaming source column schema/order differs")
            if not len(frame):
                continue
            frame = frame.reset_index(drop=True)
            for name, dtype in self.categorical_dtypes.items():
                if name not in frame:
                    raise ValueError(f"Streaming category missing: {name}")
                before = frame[name]
                frame[name] = before.astype(dtype)
                if not before.isna().equals(frame[name].isna()):
                    raise ValueError(f"Streaming category lost values: {name}")
            self.buffer = frame
            return True
        self.buffer = None
        return False

    def take(self, n: int) -> pd.DataFrame:
        if n < 1:
            raise ValueError("take(n) requires positive n")
        pieces = []
        remaining = n
        while remaining:
            if not self._fill():
                raise ValueError(f"Streaming source ended {remaining} rows early")
            assert self.buffer is not None
            count = min(remaining, len(self.buffer))
            pieces.append(self.buffer.iloc[:count].copy())
            self.buffer = (self.buffer.iloc[count:].copy()
                           if count < len(self.buffer) else None)
            remaining -= count
        result = (pieces[0] if len(pieces) == 1 else
                  pd.concat(pieces, ignore_index=True))
        return result.reset_index(drop=True)

    def finish(self) -> None:
        try:
            if self._fill():
                raise ValueError("Streaming source contains extra rows")
        finally:
            self.close()

    def close(self) -> None:
        self.buffer = None
        if hasattr(self.source, "close"):
            self.source.close()


class ParquetFrameCursor(FrameCursor):
    """Public bounded cursor over a Parquet file with optional global categories."""

    def __init__(self, path: Path, columns: list[str] | tuple[str, ...],
                 batch_size: int = BATCH_SIZE,
                 categorical_dtypes: dict[str, pd.CategoricalDtype] | None = None):
        if batch_size < 1 or batch_size > 65536:
            raise ValueError("Parquet batch size must be in 1..65536")
        parquet = pq.ParquetFile(path)
        names = list(columns)
        if any(name not in parquet.schema_arrow.names for name in names):
            raise ValueError(f"{path}: requested columns absent from Parquet schema")
        batches = parquet.iter_batches(batch_size=batch_size, columns=names)
        super().__init__((batch.to_pandas() for batch in batches), names,
                         categorical_dtypes)
        self.path = path
        self.parquet = parquet

    def close(self) -> None:
        super().close()
        self.parquet.close()


def raw_departure_frames(paths: list[Path], batch_size: int
                         ) -> Iterator[pd.DataFrame]:
    selected = ["PHASE_mvt", *RAW_COLUMNS]
    for path in paths:
        parquet = pq.ParquetFile(path)
        try:
            if any(name not in parquet.schema_arrow.names for name in selected):
                raise ValueError(f"{path}: required released departure column missing")
            for batch in parquet.iter_batches(batch_size=batch_size, columns=selected):
                phase = batch.column(batch.schema.get_field_index("PHASE_mvt"))
                mask = pc.equal(phase, "DEP")
                filtered = batch.filter(mask)
                if filtered.num_rows:
                    yield pa.Table.from_batches([filtered]).select(
                        list(RAW_COLUMNS)).to_pandas()
        finally:
            parquet.close()


def raw_departure_cursor(paths: list[Path], batch_size: int = BATCH_SIZE
                         ) -> FrameCursor:
    return FrameCursor(raw_departure_frames(paths, batch_size),
                       list(RAW_COLUMNS))


def iter_prepared_training_batches(output_dir: Path, cache_dir: Path,
                                   batch_size: int = BATCH_SIZE
                                   ) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
    """Yield exact-order `(rows, features)` with globally restored categories.

    Call ``movement.verify_prepared_integrity`` before use when a current
    physical-cache/source seal is required. This iterator itself checks the
    full manifest feature schema, every row ID, airport, and exhaustion.
    """
    manifest = json.loads((output_dir / "features_manifest.json").read_text(
        encoding="utf-8"))
    feature_path = output_dir / "features.parquet"
    feature_file = pq.ParquetFile(feature_path)
    names = manifest["features"]
    if (feature_file.metadata.num_rows != manifest["rows"]
            or feature_file.schema_arrow.names != names):
        raise ValueError("Prepared feature schema/count differs from manifest")
    categorical = load_global_category_dtypes(feature_path,
                                               manifest["categorical"])
    features = ParquetFrameCursor(feature_path, names, batch_size, categorical)
    rows = ParquetFrameCursor(cache_dir / "training_rows.parquet",
                              INFERENCE_ROW_COLUMNS, batch_size)
    ids = ParquetFrameCursor(output_dir / "row_ids.parquet",
                             ["MVT_ID_mvt"], batch_size)
    seen_ids = np.empty(manifest["rows"], dtype=np.float64)
    try:
        for offset in range(0, manifest["rows"], batch_size):
            n = min(batch_size, manifest["rows"] - offset)
            feature_part, row_part, id_part = (features.take(n), rows.take(n),
                                               ids.take(n))
            movement.assert_safe_matrix(feature_part, names)
            if (not np.array_equal(row_part.MVT_ID_mvt.to_numpy(),
                                   id_part.MVT_ID_mvt.to_numpy())
                    or row_part.MVT_ID_mvt.isna().any()
                    or not feature_part.ADEP_mvt.astype("string").eq(
                        row_part.airport.astype("string")).all()):
                raise ValueError(f"Prepared training row alignment differs at {offset}")
            seen_ids[offset:offset+n] = row_part.MVT_ID_mvt.to_numpy(
                dtype=np.float64)
            yield row_part, feature_part
        if len(np.unique(seen_ids)) != manifest["rows"]:
            raise ValueError("Prepared training movement IDs repeat")
        features.finish()
        rows.finish()
        ids.finish()
    finally:
        features.close()
        rows.close()
        ids.close()


def flight_category_dtype(paths: list[Path], batch_size: int,
                          abort_free_gib: float | None = None
                          ) -> pd.CategoricalDtype:
    """Independently derive the full FLIGHT_mvt vocabulary from raw departures."""
    names: set[str] = set()
    for frame in raw_departure_frames(paths, batch_size):
        if (abort_free_gib is not None and
                movement.available_memory_gib() < abort_free_gib):
            raise MemoryError("Raw flight vocabulary scan aborted before low-memory pressure")
        flight = frame.FLIGHT_mvt.astype("string").fillna("__MISSING__")
        names.update(flight.unique().tolist())
    categories = pd.Series(list(names), dtype="string").astype(
        "category").cat.categories
    return pd.CategoricalDtype(categories=categories, ordered=False)


def independent_batch(cached: pd.DataFrame, rows: pd.DataFrame,
                      raw: pd.DataFrame, arrivals: pd.DataFrame,
                      flight_dtype: pd.CategoricalDtype,
                      weather_file: Path) -> tuple[pd.DataFrame, list[str]]:
    """Apply the unchanged movement.prepare formulas to one aligned batch."""
    if (not np.array_equal(raw.MVT_ID_mvt.to_numpy(),
                           rows.MVT_ID_mvt.to_numpy())
            or not np.array_equal(arrivals.MVT_ID_mvt.to_numpy(),
                                  rows.MVT_ID_mvt.to_numpy())
            or not cached.ADEP_mvt.astype("string").eq(
                rows.airport.astype("string")).all()):
        raise ValueError("Raw, ARR, cached features and baseline IDs/airports differ")
    mvt = pd.to_datetime(raw.MVT_TIME_UTC_mvt, utc=True, errors="coerce")
    sched = pd.to_datetime(raw.SCHED_TIME_UTC_mvt, utc=True, errors="coerce")
    if not mvt.eq(pd.to_datetime(rows.time, utc=True)).all():
        raise ValueError("Raw movement time differs from baseline row time")

    features = cached.copy()
    flight = raw.FLIGHT_mvt.astype("string").fillna("__MISSING__")
    features["flight_name_mvt"] = flight.astype(flight_dtype)
    if features.flight_name_mvt.isna().any():
        raise ValueError("Raw flight name fell outside the independently derived vocabulary")
    features["mvt_utc_minute"] = mvt.dt.minute.astype("float32")
    features["mvt_utc_second"] = mvt.dt.second.astype("float32")
    features["schedule_utc_hour"] = sched.dt.hour.astype("float32")
    features["schedule_utc_minute"] = sched.dt.minute.astype("float32")
    features["schedule_utc_second"] = sched.dt.second.astype("float32")
    features["schedule_utc_weekday"] = sched.dt.dayofweek.astype("float32")
    schedule_local_hour = np.full(len(rows), np.nan, dtype=np.float32)
    schedule_local_weekday = np.full(len(rows), np.nan, dtype=np.float32)
    for airport, timezone_name in AIRPORT_TZ.items():
        idx = np.flatnonzero(rows.airport.eq(airport).to_numpy())
        if len(idx):
            local = sched.iloc[idx].dt.tz_convert(timezone_name)
            schedule_local_hour[idx] = local.dt.hour.to_numpy(dtype=np.float32)
            schedule_local_weekday[idx] = local.dt.dayofweek.to_numpy(dtype=np.float32)
    features["schedule_local_hour"] = schedule_local_hour
    features["schedule_local_weekday"] = schedule_local_weekday
    gap = (mvt - sched).dt.total_seconds()
    features["mvt_schedule_gap_seconds"] = gap.clip(-604800, 604800).astype("float32")
    features["mvt_schedule_day_offset"] = np.floor(gap / 86400).clip(
        -30, 30).astype("float32")
    features["schedule_missing"] = sched.isna().astype("int8")
    features = add_weather(features, rows[["airport", "time"]], weather_file)
    for name in movement.ARRIVAL_COLUMNS:
        features[name] = pd.to_numeric(arrivals[name], errors="coerce").astype(
            "float32")
    cached_names = list(movement.SAFE_CACHED_NUMERIC +
                        movement.SAFE_CACHED_CATEGORICAL)
    weather_names = [name for name in features if name.startswith("wx_")]
    names = (cached_names + list(movement.DERIVED_COLUMNS) + weather_names +
             list(movement.ARRIVAL_COLUMNS))
    movement.assert_safe_matrix(features, names)
    return features, names


def source_paths(args: argparse.Namespace) -> dict[str, Path]:
    paths = {
        "baseline_training_rows": args.cache_dir / "training_rows.parquet",
        "baseline_features": args.cache_dir / "features.parquet",
        "released_arrival_cache": args.arrival_cache,
        "noaa_weather": args.weather_file,
        "frozen_v5_oof": args.v5_oof,
        "frozen_movement_protocol": args.output_dir / "protocol.json",
        "original_prepared_features": args.output_dir / "features.parquet",
        "original_row_ids": args.output_dir / "row_ids.parquet",
        "original_features_manifest": args.output_dir / "features_manifest.json",
        "stream_verifier_source": Path(__file__).resolve(),
        "movement_formula_source": Path(movement.__file__).resolve(),
        "weather_formula_source": Path(add_weather.__code__.co_filename).resolve(),
        "training_file_selector_source": Path(_training_files.__code__.co_filename).resolve(),
    }
    paths.update({f"training_{path.name}": path for path in
                  _training_files(args.data_dir)})
    return paths


def source_snapshot(paths: dict[str, Path]) -> dict[str, str]:
    return {name: sha256(path) for name, path in paths.items()}


def exact_file_comparison(args: argparse.Namespace, rebuilt_dir: Path,
                          manifest: dict, batch_size: int) -> dict:
    """Match the original verifier's schema, category, value, ID and manifest gates."""
    old_path = args.output_dir / "features.parquet"
    new_path = rebuilt_dir / "features.parquet"
    original_file = pq.ParquetFile(old_path)
    rebuilt_file = pq.ParquetFile(new_path)
    if (original_file.metadata.num_rows != rebuilt_file.metadata.num_rows
            or not original_file.schema_arrow.equals(rebuilt_file.schema_arrow,
                                                     check_metadata=True)):
        raise ValueError("Independently streamed feature schema or metadata differs")
    old_manifest = json.loads((args.output_dir / "features_manifest.json").read_text(
        encoding="utf-8"))
    new_manifest = json.loads((rebuilt_dir / "features_manifest.json").read_text(
        encoding="utf-8"))
    if old_manifest != new_manifest or old_manifest != manifest:
        raise ValueError("Independently streamed feature manifest differs")
    old_categories = load_global_category_dtypes(
        old_path, manifest["categorical"], args.abort_free_gib)
    new_categories = load_global_category_dtypes(
        new_path, manifest["categorical"], args.abort_free_gib)
    for name in manifest["categorical"]:
        old, new = old_categories[name], new_categories[name]
        if (not old.categories.equals(new.categories)
                or old.ordered != new.ordered):
            raise ValueError(f"Independently streamed categories differ: {name}")

    old_ids = ParquetFrameCursor(args.output_dir / "row_ids.parquet",
                                  ["MVT_ID_mvt"], batch_size)
    new_ids = ParquetFrameCursor(rebuilt_dir / "row_ids.parquet",
                                  ["MVT_ID_mvt"], batch_size)
    old_values = ParquetFrameCursor(old_path, manifest["features"], batch_size,
                                     old_categories)
    new_values = ParquetFrameCursor(new_path, manifest["features"], batch_size,
                                     new_categories)
    try:
        for offset in range(0, manifest["rows"], batch_size):
            if movement.available_memory_gib() < args.abort_free_gib:
                raise MemoryError("Exact comparison aborted before low-memory pressure")
            n = min(batch_size, manifest["rows"] - offset)
            if not old_ids.take(n).equals(new_ids.take(n)):
                raise ValueError(f"Independently streamed row ID order differs at {offset}")
            if not old_values.take(n).equals(new_values.take(n)):
                raise ValueError(f"Independently streamed feature values differ at {offset}")
        for cursor in (old_ids, new_ids, old_values, new_values):
            cursor.finish()
    finally:
        for cursor in (old_ids, new_ids, old_values, new_values):
            cursor.close()
    return {"schema_equal": True, "categorical_levels_equal": True,
            "feature_values_equal": True, "row_id_order_equal": True,
            "manifest_equal": True,
            "rebuild_features_sha256": sha256(new_path),
            "rebuild_row_ids_sha256": sha256(rebuilt_dir / "row_ids.parquet")}


def rebuild_in_batches(args: argparse.Namespace, rebuilt_dir: Path,
                       source_hashes: dict[str, str]) -> tuple[dict, dict]:
    original_manifest = json.loads((args.output_dir / "features_manifest.json")
                                   .read_text(encoding="utf-8"))
    total = int(original_manifest["rows"])
    cached_names = list(movement.SAFE_CACHED_NUMERIC +
                        movement.SAFE_CACHED_CATEGORICAL)
    raw_files = _training_files(args.data_dir)
    baseline_path = args.cache_dir / "training_rows.parquet"
    cached_path = args.cache_dir / "features.parquet"
    original_path = args.output_dir / "features.parquet"
    ids_path = args.output_dir / "row_ids.parquet"
    arrival_columns = ["MVT_ID_mvt", *movement.ARRIVAL_COLUMNS]
    for path in (baseline_path, cached_path, original_path,
                 ids_path, args.arrival_cache):
        if pq.ParquetFile(path).metadata.num_rows != total:
            raise ValueError(f"{path}: departure row count differs from prepared manifest")
    if pq.ParquetFile(original_path).schema_arrow.names != original_manifest["features"]:
        raise ValueError("Original prepared feature schema/order differs from manifest")
    if pq.ParquetFile(ids_path).schema_arrow.names != ["MVT_ID_mvt"]:
        raise ValueError("Original row-ID schema differs")

    baseline_categories = load_global_category_dtypes(
        cached_path, movement.SAFE_CACHED_CATEGORICAL, args.abort_free_gib)
    independent_flight_dtype = flight_category_dtype(
        raw_files, args.batch_size, args.abort_free_gib)
    expected_categorical = [*movement.SAFE_CACHED_CATEGORICAL,
                            "flight_name_mvt"]
    if original_manifest["categorical"] != expected_categorical:
        raise ValueError("Prepared categorical name/order differs from independent sources")
    original_categories = load_global_category_dtypes(
        original_path, original_manifest["categorical"], args.abort_free_gib)
    baseline_rows = ParquetFrameCursor(baseline_path, ROW_COLUMNS,
                                       args.batch_size)
    baseline_features = ParquetFrameCursor(cached_path, cached_names,
                                           args.batch_size,
                                           baseline_categories)
    arrivals = ParquetFrameCursor(args.arrival_cache, arrival_columns,
                                  args.batch_size)
    raw = raw_departure_cursor(raw_files, args.batch_size)
    old_ids = ParquetFrameCursor(ids_path, ["MVT_ID_mvt"], args.batch_size)
    old_features = ParquetFrameCursor(original_path,
                                      original_manifest["features"],
                                      args.batch_size, original_categories)
    seen_ids = np.empty(total, dtype=np.float64)
    original_schema = pq.ParquetFile(original_path).schema_arrow
    original_id_schema = pq.ParquetFile(ids_path).schema_arrow
    feature_writer: pq.ParquetWriter | None = None
    id_writer: pq.ParquetWriter | None = None
    minimum_free = movement.available_memory_gib()
    names: list[str] | None = None
    try:
        for offset in range(0, total, args.batch_size):
            available = movement.available_memory_gib()
            minimum_free = min(minimum_free, available)
            if available < args.abort_free_gib:
                raise MemoryError("Streaming verifier aborted before low-memory pressure")
            n = min(args.batch_size, total - offset)
            row_part = baseline_rows.take(n)
            cached_part = baseline_features.take(n)
            arrival_part = arrivals.take(n)
            raw_part = raw.take(n)
            old_id_part = old_ids.take(n)
            old_feature_part = old_features.take(n)
            if (row_part.MVT_ID_mvt.isna().any()
                    or not np.array_equal(row_part.MVT_ID_mvt.to_numpy(),
                                          old_id_part.MVT_ID_mvt.to_numpy())):
                raise ValueError(f"Prepared ID order differs from baseline at {offset}")
            seen_ids[offset:offset+n] = row_part.MVT_ID_mvt.to_numpy(
                dtype=np.float64)
            rebuilt_part, current_names = independent_batch(
                cached_part, row_part, raw_part, arrival_part,
                independent_flight_dtype, args.weather_file)
            if names is None:
                names = current_names
                if names != original_manifest["features"]:
                    raise ValueError("Independent feature name/order differs from manifest")
            elif current_names != names:
                raise ValueError("Independent feature schema changed across batches")
            if not old_feature_part.equals(rebuilt_part):
                raise ValueError(f"Independently rebuilt logical features differ at {offset}")
            feature_table = pa.Table.from_pandas(rebuilt_part,
                                                 preserve_index=False)
            id_table = pa.Table.from_pandas(
                row_part[["MVT_ID_mvt"]], preserve_index=False)
            if (not feature_table.schema.equals(original_schema,
                                                check_metadata=True)
                    or not id_table.schema.equals(original_id_schema,
                                                  check_metadata=True)):
                raise ValueError(f"Independent Arrow schema/metadata differs at {offset}")
            if feature_writer is None:
                feature_writer = pq.ParquetWriter(
                    rebuilt_dir / "features.parquet", feature_table.schema)
                id_writer = pq.ParquetWriter(
                    rebuilt_dir / "row_ids.parquet", id_table.schema)
            assert id_writer is not None
            feature_writer.write_table(feature_table)
            id_writer.write_table(id_table)
            if offset % (args.batch_size * 16) == 0:
                print(json.dumps({"phase": "stream_rebuild", "rows_checked": offset+n,
                                  "total_rows": total,
                                  "free_gib": round(available, 2)}), flush=True)
            del row_part, cached_part, arrival_part, raw_part
            del old_id_part, old_feature_part, rebuilt_part
            del feature_table, id_table
            gc.collect()
        for cursor in (baseline_rows, baseline_features, arrivals, raw,
                       old_ids, old_features):
            cursor.finish()
        if len(np.unique(seen_ids)) != total:
            raise ValueError("Baseline/prepared movement IDs repeat")
    finally:
        if feature_writer is not None:
            feature_writer.close()
        if id_writer is not None:
            id_writer.close()
        for cursor in (baseline_rows, baseline_features, arrivals, raw,
                       old_ids, old_features):
            cursor.close()
    if names is None:
        raise ValueError("Independent rebuild contained zero departures")
    expected_manifest = {
        "rows": total, "features": names,
        "categorical": expected_categorical,
        "baseline_rows_sha256": source_hashes["baseline_training_rows"],
        "baseline_features_sha256": source_hashes["baseline_features"],
        "arrival_cache_sha256": source_hashes["released_arrival_cache"],
        "weather_file_sha256": source_hashes["noaa_weather"],
        "reference_sha256": movement.REFERENCE_SHA256,
        "weather_source": "NOAA NCEI GHCNh CC0-1.0",
        "arrival_source": "Released ARR movement cache",
    }
    if expected_manifest != original_manifest:
        raise ValueError("Independently regenerated manifest differs")
    (rebuilt_dir / "features_manifest.json").write_text(
        json.dumps(expected_manifest, indent=2) + "\n", encoding="utf-8")
    return expected_manifest, {"minimum_available_gib": minimum_free,
                               "batch_size": args.batch_size,
                               "rows_checked": total}


def verify(args: argparse.Namespace) -> dict:
    if args.batch_size < 1 or args.batch_size > 65536:
        raise ValueError("Streaming batch size must be in 1..65536")
    if args.min_free_gib < args.abort_free_gib or args.abort_free_gib < 0.5:
        raise ValueError("Invalid streaming-only memory guard")
    available = movement.available_memory_gib()
    if available < args.min_free_gib:
        raise MemoryError(f"Streaming verifier needs {args.min_free_gib:g} GiB free; "
                          f"currently {available:.2f} GiB")
    protocol_path = args.output_dir / "protocol.json"
    if json.loads(protocol_path.read_text(encoding="utf-8")) != movement.protocol():
        raise ValueError("Frozen movement protocol differs")
    seal_path = args.output_dir / "prepared_integrity.json"
    if seal_path.exists():
        raise FileExistsError("Prepared integrity seal already exists; no overwrite")
    if sha256(args.v5_oof) != movement.REFERENCE_SHA256:
        raise ValueError("Frozen v5 OOF reference changed")
    paths = source_paths(args)
    before = source_snapshot(paths)
    rebuilt_dir = Path(tempfile.mkdtemp(prefix="verification-stream-",
                                        dir=args.output_dir))
    manifest, memory = rebuild_in_batches(args, rebuilt_dir, before)
    comparison = exact_file_comparison(args, rebuilt_dir, manifest,
                                       args.batch_size)
    after = source_snapshot(paths)
    if before != after:
        raise ValueError("Source/cache/verifier bytes changed during streaming rebuild")
    if seal_path.exists():
        raise FileExistsError("A prepared integrity seal appeared during verification")
    comparison.update({"stream_verifier_source_sha256":
                       before["stream_verifier_source"],
                       "batch_size": args.batch_size,
                       "minimum_available_gib": memory["minimum_available_gib"],
                       "initial_available_gib": available,
                       "independent_rebuild_dir": str(rebuilt_dir)})
    seal = movement.write_prepared_integrity(
        args, method="independent_rebuild_exact", comparison=comparison)
    movement.verify_prepared_integrity(args)
    return {"method": seal["method"], "rows": manifest["rows"],
            "features": len(manifest["features"]),
            "prepared_features_sha256": seal["prepared_features_sha256"],
            "minimum_available_gib": memory["minimum_available_gib"],
            "rebuild_dir": str(rebuilt_dir)}


def contract() -> dict:
    return {
        "modes": ["contract", "preview", "verify"],
        "verify": "Independent bounded rebuild from baseline, canonical raw DEP, ARR and weather sources; exact physical-cache schema/metadata/categories/values/IDs/manifest comparison; only then write original independent_rebuild_exact seal",
        "helper": "iter_prepared_training_batches(output_dir, cache_dir, batch_size=32768) yields (rows, features) with exact order and global categories",
        "limits": "New verifier only: batches <=65536, memory checked before and during; original model-fit 10 GiB gate unchanged",
        "source": "Only released training inputs; ranking data and leaderboard are never read",
    }


def preview(args: argparse.Namespace) -> dict:
    names = {
        "baseline_rows": args.cache_dir / "training_rows.parquet",
        "baseline_features": args.cache_dir / "features.parquet",
        "arrival_training": args.arrival_cache,
        "prepared_features": args.output_dir / "features.parquet",
        "prepared_row_ids": args.output_dir / "row_ids.parquet",
    }
    metadata = {name: {"rows": pq.ParquetFile(path).metadata.num_rows,
                       "columns": pq.ParquetFile(path).metadata.num_columns,
                       "row_groups": pq.ParquetFile(path).metadata.num_row_groups,
                       "file_mib": round(path.stat().st_size / 2**20, 2)}
                for name, path in names.items()}
    return {"metadata_only": True, "inputs": metadata,
            "canonical_training_files": len(_training_files(args.data_dir)),
            "seal_exists": (args.output_dir / "prepared_integrity.json").exists(),
            "batch_size": args.batch_size,
            "new_stream_min_free_gib": args.min_free_gib,
            "original_fit_min_free_gib_unchanged": 10.0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("contract", "preview", "verify"),
                        default="contract")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--cache-dir", type=Path,
                        default=Path("artifacts/baseline"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("artifacts/v6-movement-only"))
    parser.add_argument("--arrival-cache", type=Path,
                        default=Path("artifacts/v5-arrival-clean/training_arrival_features.parquet"))
    parser.add_argument("--weather-file", type=Path,
                        default=Path("data/external/weather.parquet"))
    parser.add_argument("--v5-oof", type=Path,
                        default=Path("artifacts/v5-ensemble/validation_predictions.parquet"))
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--min-free-gib", type=float, default=3.5)
    parser.add_argument("--abort-free-gib", type=float, default=1.5)
    args = parser.parse_args()
    if args.mode == "contract":
        result = contract()
    elif args.mode == "preview":
        result = preview(args)
    else:
        result = verify(args)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
