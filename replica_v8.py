"""Isolated, prospective reproduction of the frozen v8 missing-clock route.

The original v8 entry points authenticate the original run's v5/v7 bytes.
This adapter accepts new, hash-bound parent receipts in a disjoint run root and
reuses the published pure feature/model functions. It never changes those
modules or authenticates a rebuilt parent with an original-run digest.

Only ``plan`` and ``self-test`` are authorized before source/spec publication.
Real modes require this file's published SHA-256 and an independent parent
manifest. They must be peer reviewed before use.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

import movement_only_expert as movement
from solution import AIRPORT_TZ, _training_files
from weather_model import add_weather


SOURCE_ROOT = Path(__file__).resolve().parent
SPEC_PATH = SOURCE_ROOT / "reports/clean_replication_v8_spec.json"
SPEC_SHA256 = "b3f0524acd90fe9ce4ceb4d11df334a71757aec094b2bd4a89fa6c94b3d47060"
FOLDS = {"seasonal_jan_jul": (1, 7), "forward_nov_dec": (11, 12)}
WEIGHTS = (0.0, 0.25, 0.5, 1.0)
FIXED_WEIGHT = 1.0
EXPECTED_OOF_ROWS = 672_428
EXPECTED_RANKING_ROWS = 344_841
MIN_FREE_GIB = 10.0
HEX64 = re.compile(r"^[0-9a-f]{64}$")
HEX40 = re.compile(r"^[0-9a-f]{40}$")
PARENT_ROLES = (
    *(f"raw_training_{month:02d}" for month in range(1, 13)),
    "raw_ranking", "submission_template", "baseline_training_rows",
    "baseline_training_features", "baseline_ranking_rows",
    "baseline_ranking_features", "arrival_training_features",
    "arrival_ranking_features", "noaa_weather", "v5_validation_oof",
    "v5_protocol", "v5_validation_report", "v5_producer_receipt",
    "v7_validation_oof", "v7_protocol", "v7_validation_report",
    "v7_ranking_predictions", "v7_ranking_manifest", "v7_producer_receipt",
)
PARENT_MODEL_PREFIXES = {"v5": "v5_model_", "v7": "v7_model_"}
ORIGINAL_SOURCE_ROLES = (
    "movement_only_expert.py", "missing_catboost.py", "compose_current_candidate.py",
    "solution.py", "weather_model.py", "reports/reserved_guard_protocol.json",
)
COMMON_PARENT_INPUT_ROLES = (
    *(f"raw_training_{month:02d}" for month in range(1, 13)),
    "raw_ranking", "submission_template", "baseline_training_rows",
    "baseline_training_features", "baseline_ranking_rows",
    "baseline_ranking_features", "arrival_training_features",
    "arrival_ranking_features", "noaa_weather",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False) + "\n").encode("utf-8")


def value_sha256(value: Any) -> str:
    return hashlib.sha256(json_bytes(value)).hexdigest()


def ordered_sha(values: pd.Series | np.ndarray) -> str:
    """Stable ID hash; never hashes object-array pointer bytes."""
    digest = hashlib.sha256()
    for value in values:
        if pd.isna(value):
            raise ValueError("Ordered identity contains null")
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def numeric_sha(values: np.ndarray, dtype: str) -> str:
    return hashlib.sha256(np.asarray(values, dtype=dtype).tobytes()).hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def output_root(run_root: Path) -> Path:
    return Path(run_root).resolve(strict=True) / "v8-movement-replica"


def strict_run_root(run_root: Path) -> Path:
    root = Path(run_root).resolve(strict=True)
    source = SOURCE_ROOT.resolve(strict=True)
    if root == source or root in source.parents or source in root.parents:
        raise ValueError("Replica run root must be disjoint from source repository")
    return root


def under_run_root(run_root: Path, relative: str, *, require_file: bool = True) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("Parent manifest paths must be nonempty run-root-relative paths")
    if re.match(r"^[A-Za-z]:", relative) or relative.startswith(("/", "\\")):
        raise ValueError("Absolute parent path is prohibited")
    if ".." in Path(relative).parts:
        raise ValueError("Parent path traversal is prohibited")
    path = run_root / relative
    resolved = path.resolve(strict=require_file)
    if resolved == run_root or run_root not in resolved.parents:
        raise ValueError("Parent path escapes the isolated run root")
    if require_file and not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def check_source_spec() -> dict:
    if sha256(SPEC_PATH) != SPEC_SHA256:
        raise ValueError("Published replica specification bytes changed")
    spec = read_json(SPEC_PATH)
    for relative in ORIGINAL_SOURCE_ROLES:
        expected = spec["frozen_original_sources_sha256"][relative]
        if sha256(SOURCE_ROOT / relative) != expected:
            raise ValueError(f"Frozen original source changed: {relative}")
    return spec


def check_published_source(expected_sha256: str | None) -> str:
    actual = sha256(Path(__file__).resolve())
    if expected_sha256 is None or actual != expected_sha256.lower():
        raise ValueError("Published replica source SHA-256 is required for real modes")
    check_source_spec()
    return actual


def source_snapshot() -> dict[str, str]:
    spec = check_source_spec()
    paths = {"replica_v8.py": Path(__file__).resolve(),
             "reports/clean_replication_v8_spec.json": SPEC_PATH}
    paths.update({name: SOURCE_ROOT / name for name in
                  spec["frozen_original_sources_sha256"]})
    return {name: sha256(path) for name, path in sorted(paths.items())}


def _require_sha(value: str, name: str, pattern=HEX64) -> None:
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise ValueError(f"{name} requires lowercase canonical hex SHA")


def validate_parent_metadata(manifest: dict) -> None:
    """Metadata-only validation, including a synthetic fixture self-test path."""
    if manifest.get("schema_version") != 1 or manifest.get("status") != "complete":
        raise ValueError("Parent manifest must be a completed version-1 run")
    files = manifest.get("files")
    if not isinstance(files, dict) or not set(PARENT_ROLES).issubset(files):
        raise ValueError("Parent manifest lacks mandatory raw/cache/v5/v7 roles")
    for parent, prefix in PARENT_MODEL_PREFIXES.items():
        if not any(role.startswith(prefix) for role in files):
            raise ValueError(f"Independent {parent} model bytes are absent")
    if len(files) != len(set(files)):
        raise ValueError("Repeated parent role")
    for name, item in files.items():
        if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
            raise ValueError(f"Invalid exact parent file entry: {name}")
        if not isinstance(item["path"], str) or not item["path"]:
            raise ValueError(f"Missing parent relative path: {name}")
        _require_sha(item["sha256"], f"parent {name}")
    if len({item["path"].casefold() for item in files.values()}) != len(files):
        raise ValueError("Multiple parent roles alias one path")
    parents = manifest.get("parents")
    if not isinstance(parents, dict) or set(parents) != {"v5", "v7"}:
        raise ValueError("Named independent v5 and v7 parent receipts required")
    for name in ("v5", "v7"):
        parent = parents[name]
        if (not isinstance(parent, dict) or parent.get("status") != "complete"
                or parent.get("name") != name):
            raise ValueError(f"Invalid {name} parent identity/status")
        _require_sha(parent.get("source_commit"), f"{name} source commit", HEX40)
        _require_sha(parent.get("producer_receipt_sha256"), f"{name} receipt")
        _require_sha(parent.get("ordered_ids_sha256"), f"{name} ordered IDs")
        _require_sha(parent.get("feature_schema_sha256"), f"{name} feature schema")
        source_map = parent.get("source_sha256")
        if not isinstance(source_map, dict) or not source_map:
            raise ValueError(f"{name} source SHA inventory missing")
        for role, digest in source_map.items():
            _require_sha(digest, f"{name} source {role}")
        if parent.get("heldout_folds") != {"seasonal_jan_jul": [1, 7],
                                           "forward_nov_dec": [11, 12]}:
            raise ValueError(f"{name} heldout splits differ from published folds")
        if parent.get("fit_and_early_exclude_heldout") is not True:
            raise ValueError(f"{name} fitted or early-stopped on heldout months")
        if not isinstance(parent.get("published_params_and_weights"), dict):
            raise ValueError(f"{name} published parameters/weights missing")
        published = parent["published_params_and_weights"]
        fixed = ({"selected_stage": "deep_missing_clean_arrival",
                  "valid_deep_weight": 0.5, "missing_direct_weight": 0.5,
                  "clean_arrival_weight": 0.25} if name == "v5" else
                 {"selected_weight": 0.5, "depth": 10,
                  "max_iterations": 10000, "seed": 2026})
        if any(published.get(key) != value for key, value in fixed.items()):
            raise ValueError(f"{name} published parent parameters/weights differ")
        outputs = parent.get("output_sha256")
        if not isinstance(outputs, dict) or not outputs:
            raise ValueError(f"{name} output lineage missing")
        required_outputs = ({"v5_validation_oof", "v5_protocol", "v5_validation_report"}
                            if name == "v5" else
                            {"v7_validation_oof", "v7_protocol", "v7_validation_report",
                             "v7_ranking_predictions", "v7_ranking_manifest"})
        required_outputs.update(role for role in files if role.startswith(
            PARENT_MODEL_PREFIXES[name]))
        if not required_outputs.issubset(outputs):
            raise ValueError(f"{name} producer receipt omits required output/model bytes")
        for role, digest in outputs.items():
            if role not in files or files[role]["sha256"] != digest:
                raise ValueError(f"{name} output {role} does not match named file")
        inputs = parent.get("input_sha256")
        if (not isinstance(inputs, dict)
                or not set(COMMON_PARENT_INPUT_ROLES).issubset(inputs)):
            raise ValueError(f"{name} producer receipt omits raw/cache/weather inputs")
        for role, digest in inputs.items():
            if role not in files or files[role]["sha256"] != digest:
                raise ValueError(f"{name} input {role} differs from named clean-run file")
    if parents["v5"]["producer_receipt_sha256"] != files["v5_producer_receipt"]["sha256"]:
        raise ValueError("v5 receipt SHA differs from named parent bytes")
    if parents["v7"]["producer_receipt_sha256"] != files["v7_producer_receipt"]["sha256"]:
        raise ValueError("v7 receipt SHA differs from named parent bytes")
    lineage = manifest.get("lineage")
    if (not isinstance(lineage, dict)
            or lineage.get("v7_parent_v5_receipt_sha256")
            != parents["v5"]["producer_receipt_sha256"]
            or lineage.get("v7_parent_v5_oof_sha256")
            != files["v5_validation_oof"]["sha256"]
            or lineage.get("v8_parent_v7_oof_sha256")
            != files["v7_validation_oof"]["sha256"]
            or lineage.get("v8_parent_v7_ranking_sha256")
            != files["v7_ranking_predictions"]["sha256"]):
        raise ValueError("Recursive v5/v7/v8 SHA lineage differs")


def parent_files(run_root: Path, manifest: dict) -> dict[str, Path]:
    validate_parent_metadata(manifest)
    paths = {role: under_run_root(run_root, item["path"])
             for role, item in manifest["files"].items()}
    if len(set(paths.values())) != len(paths):
        raise ValueError("Parent roles resolve to an aliased file")
    for role, path in paths.items():
        if sha256(path) != manifest["files"][role]["sha256"]:
            raise ValueError(f"Parent bytes changed: {role}")
    for name in ("v5", "v7"):
        receipt = read_json(paths[f"{name}_producer_receipt"])
        advertised = manifest["parents"][name]
        for key in ("name", "status", "source_commit", "source_sha256",
                    "ordered_ids_sha256", "feature_schema_sha256", "heldout_folds",
                    "fit_and_early_exclude_heldout", "published_params_and_weights",
                    "output_sha256", "input_sha256"):
            if receipt.get(key) != advertised.get(key):
                raise ValueError(f"{name} producer receipt differs: {key}")
        if name == "v7" and (receipt.get("parent_v5_receipt_sha256")
                             != manifest["lineage"]["v7_parent_v5_receipt_sha256"]
                             or receipt.get("parent_v5_oof_sha256")
                             != manifest["lineage"]["v7_parent_v5_oof_sha256"]):
            raise ValueError("v7 producer receipt does not bind independent v5")
    v5_report = read_json(paths["v5_validation_report"])
    if (v5_report.get("selection", {}).get("selected_stage")
            != "deep_missing_clean_arrival"
            or v5_report.get("selection", {}).get("clean_arrival_promoted") is not True
            or v5_report.get("selection", {}).get("clean_arrival_seasonal_weight") != 0.25
            or v5_report.get("policy") !=
            "clipped v4 -> .5 deep blend on valid AOBT -> .5 ordinary missing direct blend on its disjoint gate -> clean ARR at its fixed seasonal weight if it improves both folds; legacy ARR remains diagnostic only"):
        raise ValueError("Independent v5 parent did not reproduce the published route")
    v7_report = read_json(paths["v7_validation_report"])
    architecture = v7_report.get("protocol", {}).get("architecture", {})
    if (v7_report.get("promoted") is not True
            or v7_report.get("selected_weight") != 0.5
            or v7_report.get("validation_predictions_sha256")
            != manifest["files"]["v7_validation_oof"]["sha256"]
            or architecture.get("depth") != 10
            or architecture.get("max_iterations") != 10000
            or architecture.get("random_seed") != 2026):
        raise ValueError("Independent v7 parent failed its original published gate")
    rank_report = read_json(paths["v7_ranking_manifest"])
    recorded_rank_sha = rank_report.get("predictions_sha256",
                                        rank_report.get("prediction_sha256"))
    if (rank_report.get("rows") != EXPECTED_RANKING_ROWS
            or recorded_rank_sha != manifest["files"]["v7_ranking_predictions"]["sha256"]
            or rank_report.get("template_order_verified") is not True):
        raise ValueError("Independent v7 ranking parent lacks exact sealed template output")
    return paths


def parent_snapshot(run_root: Path, manifest_path: Path) -> tuple[dict, dict[str, Path], dict]:
    run_root = strict_run_root(run_root)
    manifest_path = Path(manifest_path).resolve(strict=True)
    if manifest_path != run_root / "parents/v8_parents.json":
        raise ValueError("Named parent manifest must be at parents/v8_parents.json")
    manifest = read_json(manifest_path)
    paths = parent_files(run_root, manifest)
    return manifest, paths, {"manifest_sha256": sha256(manifest_path),
                             "file_sha256": {role: sha256(path) for role, path in paths.items()},
                             "source_sha256": source_snapshot()}


def assert_snapshot(run_root: Path, manifest_path: Path, expected: dict) -> None:
    _, _, fresh = parent_snapshot(run_root, manifest_path)
    if fresh != expected:
        raise ValueError("Isolated parent or published source changed during operation")


def write_new_json(path: Path, value: dict, run_root: Path) -> None:
    path = path.resolve(strict=False)
    if run_root not in path.parents or path.exists():
        raise FileExistsError("Replica output is outside isolated root or already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, staging = tempfile.mkstemp(prefix=".replica-stage-", suffix=".json", dir=path.parent)
    os.close(fd)
    staged = Path(staging)
    try:
        with staged.open("wb") as stream:
            stream.write(json_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.link(staged, path)
    finally:
        staged.unlink(missing_ok=True)


def write_new_parquet(path: Path, frame: pd.DataFrame, run_root: Path) -> None:
    path = path.resolve(strict=False)
    if run_root not in path.parents or path.exists():
        raise FileExistsError("Replica output is outside isolated root or already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, staging = tempfile.mkstemp(prefix=".replica-stage-", suffix=".parquet", dir=path.parent)
    os.close(fd)
    staged = Path(staging)
    try:
        frame.to_parquet(staged, index=False)
        os.link(staged, path)
    finally:
        staged.unlink(missing_ok=True)


def save_new_model(path: Path, model: Any, run_root: Path) -> None:
    path = path.resolve(strict=False)
    if run_root not in path.parents or path.exists():
        raise FileExistsError("Replica model is outside isolated root or already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, staging = tempfile.mkstemp(prefix=".replica-stage-", suffix=path.suffix, dir=path.parent)
    os.close(fd)
    staged = Path(staging)
    try:
        model.save_model(str(staged))
        os.link(staged, path)
    finally:
        staged.unlink(missing_ok=True)


def require_memory() -> None:
    if movement.available_memory_gib() < MIN_FREE_GIB:
        raise MemoryError("Replica real build/fit requires at least 10 GiB free RAM")


def real_context(args: argparse.Namespace, *, heavy: bool = False) -> tuple[Path, Path, dict, dict, dict]:
    check_published_source(args.published_source_sha256)
    root = strict_run_root(args.run_root)
    manifest_path = root / "parents/v8_parents.json"
    manifest, files, snapshot = parent_snapshot(root, manifest_path)
    if heavy:
        require_memory()
    out = output_root(root).resolve(strict=False)
    if root not in out.parents:
        raise ValueError("Replica output directory resolves outside isolated run root")
    for path in (out, manifest_path):
        if SOURCE_ROOT == path or SOURCE_ROOT in path.parents:
            raise ValueError("Replica private artifact collides with repository")
    return root, out, manifest, files, snapshot


def plan() -> dict:
    """No parent file or competition data access."""
    spec = check_source_spec()
    return {"status": "prospective_only", "parent_manifest": "<run_root>/parents/v8_parents.json",
            "required_parent_file_roles": list(PARENT_ROLES) +
            ["v5_model_*", "v7_model_*"],
            "routes": ["v5_clean_parent", "v7_clean_parent", "v8_missing_clock_replica"],
            "real_modes_need_publication_and_parent_receipts": True,
            "current_scope": spec["reproduction_limit"],
            "real_values_read": False}


def parent_args(files: dict[str, Path], out: Path) -> argparse.Namespace:
    cache = files["baseline_training_rows"].parent
    expected = {"baseline_training_rows": cache / "training_rows.parquet",
                "baseline_training_features": cache / "features.parquet",
                "baseline_ranking_rows": cache / "ranking_rows.parquet",
                "baseline_ranking_features": cache / "ranking_features.parquet"}
    if any(files[role] != path for role, path in expected.items()):
        raise ValueError("The four baseline caches must share canonical names and directory")
    data_dir = files["raw_ranking"].parent
    if (files["raw_ranking"] != data_dir / "ranking.parquet"
            or files["submission_template"] != data_dir / "submitting.parquet"):
        raise ValueError("Raw ranking/template must use canonical data names")
    raw = [files[f"raw_training_{month:02d}"] for month in range(1, 13)]
    if list(map(Path.resolve, _training_files(data_dir))) != raw:
        raise ValueError("Monthly raw training file order differs from sealed parent roles")
    if files["v5_validation_oof"].parent != files["v5_validation_report"].parent:
        raise ValueError("v5 OOF/report must share an independent parent directory")
    if files["v7_validation_oof"].parent != files["v7_validation_report"].parent:
        raise ValueError("v7 OOF/report must share an independent parent directory")
    return argparse.Namespace(cache_dir=cache, data_dir=data_dir,
                              arrival_cache=files["arrival_training_features"],
                              ranking_arrival_cache=files["arrival_ranking_features"],
                              weather_file=files["noaa_weather"],
                              v5_oof=files["v5_validation_oof"],
                              v7_oof=files["v7_validation_oof"],
                              v7_ranking=files["v7_ranking_predictions"],
                              output_dir=out, threads=3, seed=2026)


def ensure_protocol(root: Path, out: Path, manifest: dict, snapshot: dict) -> dict:
    planned = {"schema_version": 1, "replica_of": "published_local_v8_missing_clock_route",
               "source_commit_at_spec": read_json(SPEC_PATH)["source_commit_at_spec"],
               "scientific_spec_sha256": SPEC_SHA256,
               "parent_manifest_sha256": snapshot["manifest_sha256"],
               "parent_file_sha256": snapshot["file_sha256"],
               "source_sha256": snapshot["source_sha256"],
               "parent_lineage": manifest["lineage"],
               "folds": {name: list(months) for name, months in FOLDS.items()},
               "missing_weight": FIXED_WEIGHT,
               "feature_count": 79, "categorical_count": 13,
               "training_label_interval_sec": [0, 7200],
               "ordinary_day_split": "absolute_utc_epoch_day_modulo_11_equals_zero_is_early",
               "no_leaderboard_feedback": True}
    path = out / "protocol.json"
    if path.exists():
        if read_json(path) != planned:
            raise ValueError("Immutable replica protocol or parent/source SHA changed")
    else:
        write_new_json(path, planned, root)
    return planned


def prepared_paths(out: Path) -> tuple[Path, Path, Path]:
    return out / "features.parquet", out / "row_ids.parquet", out / "features_manifest.json"


def prepare(args: argparse.Namespace) -> dict:
    root, out, parent, files, snapshot = real_context(args, heavy=True)
    p = parent_args(files, out)
    ensure_protocol(root, out, parent, snapshot)
    features_path, ids_path, report_path = prepared_paths(out)
    if any(path.exists() for path in (features_path, ids_path, report_path)):
        raise FileExistsError("Replica prepared outputs already exist")
    names = list(movement.SAFE_CACHED_NUMERIC + movement.SAFE_CACHED_CATEGORICAL)
    features = pd.read_parquet(files["baseline_training_features"], columns=names)
    rows = movement.read_baseline_rows(p.cache_dir,
                                       ["MVT_ID_mvt", "target", "proxy", "month",
                                        "airport", "time"])
    if len(features) != len(rows):
        raise ValueError("Baseline cached rows/features differ")
    if not features.ADEP_mvt.astype("string").eq(rows.airport.astype("string")).all():
        raise ValueError("Baseline feature airports differ from row airports")
    raw = (pl.scan_parquet([str(files[f"raw_training_{m:02d}"]) for m in range(1, 13)])
           .filter(pl.col("PHASE_mvt") == "DEP")
           .select(["MVT_ID_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
                    "FLIGHT_mvt"]).collect().to_pandas())
    if not np.array_equal(raw.MVT_ID_mvt.to_numpy(), rows.MVT_ID_mvt.to_numpy()):
        raise ValueError("Raw departure IDs differ from cached order")
    mvt = pd.to_datetime(raw.MVT_TIME_UTC_mvt, utc=True, errors="coerce")
    sched = pd.to_datetime(raw.SCHED_TIME_UTC_mvt, utc=True, errors="coerce")
    if not mvt.eq(pd.to_datetime(rows.time, utc=True)).all():
        raise ValueError("Raw movement times differ from cached rows")
    features["flight_name_mvt"] = raw.FLIGHT_mvt.astype("string").fillna(
        "__MISSING__").astype("category")
    features["mvt_utc_minute"] = mvt.dt.minute.astype("float32")
    features["mvt_utc_second"] = mvt.dt.second.astype("float32")
    features["schedule_utc_hour"] = sched.dt.hour.astype("float32")
    features["schedule_utc_minute"] = sched.dt.minute.astype("float32")
    features["schedule_utc_second"] = sched.dt.second.astype("float32")
    features["schedule_utc_weekday"] = sched.dt.dayofweek.astype("float32")
    schedule_local_hour = np.full(len(rows), np.nan, dtype=np.float32)
    schedule_local_weekday = np.full(len(rows), np.nan, dtype=np.float32)
    for airport, timezone_name in AIRPORT_TZ.items():
        ix = np.flatnonzero(rows.airport.eq(airport).to_numpy())
        if len(ix):
            local = sched.iloc[ix].dt.tz_convert(timezone_name)
            schedule_local_hour[ix] = local.dt.hour.to_numpy(dtype=np.float32)
            schedule_local_weekday[ix] = local.dt.dayofweek.to_numpy(dtype=np.float32)
    features["schedule_local_hour"] = schedule_local_hour
    features["schedule_local_weekday"] = schedule_local_weekday
    gap = (mvt - sched).dt.total_seconds()
    features["mvt_schedule_gap_seconds"] = gap.clip(-604800, 604800).astype("float32")
    features["mvt_schedule_day_offset"] = np.floor(gap / 86400).clip(-30, 30).astype("float32")
    features["schedule_missing"] = sched.isna().astype("int8")
    del raw, mvt, sched, gap
    gc.collect()
    features = add_weather(features, rows[["airport", "time"]], p.weather_file)
    arrivals = pd.read_parquet(p.arrival_cache,
                               columns=["MVT_ID_mvt", *movement.ARRIVAL_COLUMNS])
    if not np.array_equal(arrivals.MVT_ID_mvt.to_numpy(), rows.MVT_ID_mvt.to_numpy()):
        raise ValueError("Released ARR cache differs in movement ID order")
    for name in movement.ARRIVAL_COLUMNS:
        features[name] = pd.to_numeric(arrivals[name], errors="coerce").astype("float32")
    del arrivals
    gc.collect()
    weather_names = [name for name in features if name.startswith("wx_")]
    feature_names = names + list(movement.DERIVED_COLUMNS) + weather_names + list(
        movement.ARRIVAL_COLUMNS)
    movement.assert_safe_matrix(features, feature_names)
    cats = [name for name in feature_names
            if isinstance(features[name].dtype, pd.CategoricalDtype)]
    if len(feature_names) != 79 or len(cats) != 13:
        raise ValueError("Replica must preserve the original 79/13 movement schema")
    assert_snapshot(root, root / "parents/v8_parents.json", snapshot)
    write_new_parquet(features_path, features, root)
    write_new_parquet(ids_path, pd.DataFrame({"MVT_ID_mvt": rows.MVT_ID_mvt}), root)
    report = {"rows": len(rows), "features": feature_names, "categorical": cats,
              "feature_schema_sha256": value_sha256(
                  [{"name": name, "dtype": str(features[name].dtype)} for name in feature_names]),
              "ordered_ids_sha256": ordered_sha(rows.MVT_ID_mvt),
              "prepared_features_sha256": sha256(features_path),
              "row_ids_sha256": sha256(ids_path),
              "baseline_rows_sha256": snapshot["file_sha256"]["baseline_training_rows"],
              "baseline_features_sha256": snapshot["file_sha256"]["baseline_training_features"],
              "weather_file_sha256": snapshot["file_sha256"]["noaa_weather"],
              "arrival_cache_sha256": snapshot["file_sha256"]["arrival_training_features"],
              "parent_manifest_sha256": snapshot["manifest_sha256"],
              "protocol_sha256": sha256(out / "protocol.json"),
              "source_snapshot_sha256": value_sha256(snapshot["source_sha256"]),
              "ids_never_predictors": True, "departure_block_or_taxi_as_feature": False}
    assert_snapshot(root, root / "parents/v8_parents.json", snapshot)
    write_new_json(report_path, report, root)
    return {"rows": len(rows), "feature_count": len(feature_names),
            "categorical_count": len(cats), "prepared_manifest_sha256": sha256(report_path)}


def load_prepared(root: Path, out: Path, files: dict[str, Path],
                  snapshot: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    features_path, ids_path, report_path = prepared_paths(out)
    report = read_json(report_path)
    if (report.get("prepared_features_sha256") != sha256(features_path)
            or report.get("row_ids_sha256") != sha256(ids_path)
            or report.get("parent_manifest_sha256") != snapshot["manifest_sha256"]
            or report.get("protocol_sha256") != sha256(out / "protocol.json")
            or report.get("source_snapshot_sha256") != value_sha256(snapshot["source_sha256"])
            or len(report.get("features", [])) != 79
            or len(report.get("categorical", [])) != 13):
        raise ValueError("Replica prepared feature receipt or parent lineage changed")
    p = parent_args(files, out)
    features = pd.read_parquet(features_path)
    rows = movement.read_baseline_rows(p.cache_dir,
                                       ["MVT_ID_mvt", "target", "proxy", "month",
                                        "airport", "time"])
    ids = pd.read_parquet(ids_path, columns=["MVT_ID_mvt"])
    if (len(features) != len(rows)
            or not np.array_equal(ids.MVT_ID_mvt.to_numpy(), rows.MVT_ID_mvt.to_numpy())
            or ordered_sha(rows.MVT_ID_mvt) != report["ordered_ids_sha256"]):
        raise ValueError("Prepared feature/row IDs differ")
    movement.assert_safe_matrix(features, report["features"])
    if [name for name in features if isinstance(features[name].dtype, pd.CategoricalDtype)] != report["categorical"]:
        raise ValueError("Prepared categorical schema differs")
    schema_sha = value_sha256([{"name": name, "dtype": str(features[name].dtype)}
                               for name in report["features"]])
    if schema_sha != report["feature_schema_sha256"]:
        raise ValueError("Prepared dtype schema differs")
    return features, rows, report


def ordinary_masks(rows: pd.DataFrame, heldout_months: tuple[int, ...]) -> dict[str, np.ndarray]:
    y = rows.target.to_numpy(dtype=np.float32)
    heldout = np.isin(rows.month.to_numpy(dtype=np.int16), heldout_months)
    ordinary = np.isfinite(y) & (y >= 0) & (y <= 7200) & ~heldout
    day = pd.to_datetime(rows.time, utc=True, errors="coerce").dt.floor("D")
    if day.isna().any():
        raise ValueError("All departures need a UTC movement day")
    epoch_day = day.dt.as_unit("ns").astype("int64").to_numpy() // 86_400_000_000_000
    early = ordinary & (epoch_day % 11 == 0)
    fit = ordinary & ~early
    if np.any(heldout & (fit | early)):
        raise ValueError("Heldout month entered ordinary fit or early stop")
    return {"ordinary": ordinary, "fit": fit, "early": early, "heldout": heldout}


def split_receipt(rows: pd.DataFrame, masks: dict[str, np.ndarray],
                  scored: np.ndarray) -> dict:
    ids = rows.MVT_ID_mvt
    y = rows.target.to_numpy(dtype=np.float32)
    return {"ordered_fit_ids_sha256": ordered_sha(ids[masks["fit"]]),
            "ordered_early_ids_sha256": ordered_sha(ids[masks["early"]]),
            "ordered_heldout_ids_sha256": ordered_sha(ids[masks["heldout"]]),
            "ordered_scored_ids_sha256": ordered_sha(ids[scored]),
            "fit_labels_f32_sha256": numeric_sha(y[masks["fit"]], "<f4"),
            "early_labels_f32_sha256": numeric_sha(y[masks["early"]], "<f4"),
            "scored_labels_f32_sha256": numeric_sha(y[scored], "<f4")}


def fit_fold(args: argparse.Namespace) -> dict:
    if args.fold not in FOLDS:
        raise ValueError("Only original January/July and November/December folds exist")
    root, out, parent, files, snapshot = real_context(args, heavy=True)
    p = parent_args(files, out)
    ensure_protocol(root, out, parent, snapshot)
    model_path = out / f"{args.fold}.txt"
    oof_path = out / f"{args.fold}_oof.parquet"
    report_path = out / f"{args.fold}_fit.json"
    if any(path.exists() for path in (model_path, oof_path, report_path)):
        raise FileExistsError("Replica fold outputs already exist")
    features, rows, manifest = load_prepared(root, out, files, snapshot)
    months = FOLDS[args.fold]
    masks = ordinary_masks(rows, months)
    gate = movement.read_gate(p.cache_dir, rows)
    y = rows.target.to_numpy(dtype=np.float32)
    score = masks["heldout"] & gate & np.isfinite(y)
    if score.sum() < 100:
        raise ValueError("Insufficient heldout no-NM non-LIRF rows")
    model, fit_report = movement.fit_movement_model(
        p, features, rows, manifest["categorical"], months)
    if (fit_report["fit_rows"] != int(masks["fit"].sum())
            or fit_report["internal_early_rows"] != int(masks["early"].sum())
            or fit_report["ordinary_complement_rows"] != int(masks["ordinary"].sum())):
        raise ValueError("Original pure LightGBM fit used unexpected month/day masks")
    best = int(fit_report["best_round"])
    raw = model.predict(features.loc[score], num_iteration=best, num_threads=3)
    if len(raw) != int(score.sum()) or not np.isfinite(raw).all():
        raise ValueError("Replica fold expert lacks finite gate coverage")
    output = rows.loc[score, ["MVT_ID_mvt", "target", "airport", "month", "time"]].copy()
    output["fold"] = args.fold
    output["expert"] = np.maximum(raw, 0).astype("float32")
    assert_snapshot(root, root / "parents/v8_parents.json", snapshot)
    save_new_model(model_path, model, root)
    write_new_parquet(oof_path, output, root)
    report = {"fold": args.fold, "heldout_months": list(months), **fit_report,
              "split": split_receipt(rows, masks, score),
              "heldout_gate_rows": int(score.sum()),
              "feature_count": len(features.columns),
              "categorical_count": len(manifest["categorical"]),
              "features_manifest_sha256": sha256(out / "features_manifest.json"),
              "model_sha256": sha256(model_path), "oof_sha256": sha256(oof_path),
              "parent_manifest_sha256": snapshot["manifest_sha256"],
              "protocol_sha256": sha256(out / "protocol.json"),
              "params": movement.model_params(3),
              "max_rounds": 1200, "early_stop_rounds": 100,
              "source": "independent 2025 parent lineage; no leaderboard feedback"}
    assert_snapshot(root, root / "parents/v8_parents.json", snapshot)
    write_new_json(report_path, report, root)
    return {"fold": args.fold, "best_round": best,
            "heldout_gate_rows": int(score.sum()), "passed_provenance": True}


def verify_fold(root: Path, out: Path, files: dict[str, Path], fold: str,
                manifest: dict, snapshot: dict) -> dict:
    report = read_json(out / f"{fold}_fit.json")
    if (report.get("fold") != fold
            or report.get("heldout_months") != list(FOLDS[fold])
            or report.get("features_manifest_sha256")
            != sha256(out / "features_manifest.json")
            or report.get("model_sha256") != sha256(out / f"{fold}.txt")
            or report.get("oof_sha256") != sha256(out / f"{fold}_oof.parquet")
            or report.get("parent_manifest_sha256") != snapshot["manifest_sha256"]
            or report.get("protocol_sha256") != sha256(out / "protocol.json")
            or report.get("params") != movement.model_params(3)
            or report.get("max_rounds") != 1200
            or report.get("early_stop_rounds") != 100
            or report.get("feature_count") != 79
            or report.get("categorical_count") != 13
            or not 1 <= int(report.get("best_round", 0)) <= 1200):
        raise ValueError(f"Replica {fold} fit/model/OOF receipt differs")
    features, rows, prepared = load_prepared(root, out, files, snapshot)
    if prepared["features"] != manifest["features"]:
        raise ValueError(f"Replica {fold} prepared feature order changed")
    p = parent_args(files, out)
    masks = ordinary_masks(rows, FOLDS[fold])
    score = (masks["heldout"] & movement.read_gate(p.cache_dir, rows)
             & np.isfinite(rows.target.to_numpy(dtype=float)))
    if (report.get("split") != split_receipt(rows, masks, score)
            or report.get("heldout_gate_rows") != int(score.sum())
            or report.get("fit_rows") != int(masks["fit"].sum())
            or report.get("internal_early_rows") != int(masks["early"].sum())
            or report.get("ordinary_complement_rows") != int(masks["ordinary"].sum())):
        raise ValueError(f"Replica {fold} fit/early/heldout/scored IDs differ")
    oof = pd.read_parquet(out / f"{fold}_oof.parquet")
    expected_cols = ["MVT_ID_mvt", "target", "airport", "month", "time",
                     "fold", "expert"]
    selected = rows.loc[score]
    if (list(oof) != expected_cols or len(oof) != len(selected)
            or not np.array_equal(oof.MVT_ID_mvt.to_numpy(), selected.MVT_ID_mvt.to_numpy())
            or not np.array_equal(oof.target.to_numpy(dtype=float),
                                  selected.target.to_numpy(dtype=float))
            or not np.array_equal(oof.airport.astype("string").to_numpy(),
                                  selected.airport.astype("string").to_numpy())
            or not np.array_equal(oof.month.to_numpy(dtype=int),
                                  selected.month.to_numpy(dtype=int))
            or not np.array_equal(pd.to_datetime(oof.time, utc=True).to_numpy(),
                                  pd.to_datetime(selected.time, utc=True).to_numpy())
            or not oof.fold.eq(fold).all()):
        raise ValueError(f"Replica {fold} saved OOF metadata differs from exact heldout gate")
    import lightgbm as lgb
    model = lgb.Booster(model_file=str(out / f"{fold}.txt"))
    if (model.current_iteration() != int(report["best_round"])
            or list(model.feature_name()) != manifest["features"]):
        raise ValueError(f"Replica {fold} saved model trees/feature order differ")
    replay = np.maximum(model.predict(features.loc[score],
                                      num_iteration=int(report["best_round"]),
                                      num_threads=3), 0).astype("float32")
    if not np.array_equal(replay, oof.expert.to_numpy(dtype=np.float32)):
        raise ValueError(f"Replica {fold} saved OOF differs from saved-model predictions")
    return report


def read_parent_oof(files: dict[str, Path], role: str,
                    rows: pd.DataFrame) -> pd.DataFrame:
    columns = ["MVT_ID_mvt", "target", "fold", "airport", "month",
               "MVT_TIME_UTC_mvt", "a_valid",
               "selected" if role == "v5" else "candidate"]
    frame = pd.read_parquet(files[f"{role}_validation_oof"], columns=columns)
    if (len(frame) != EXPECTED_OOF_ROWS or frame.MVT_ID_mvt.isna().any()
            or frame.MVT_ID_mvt.duplicated().any()
            or set(frame.fold.unique()) != set(FOLDS)):
        raise ValueError(f"Independent {role} parent OOF universe differs")
    positions = pd.Index(rows.MVT_ID_mvt).get_indexer(frame.MVT_ID_mvt)
    if np.any(positions < 0):
        raise ValueError(f"Independent {role} parent OOF IDs absent from baseline")
    values = frame["selected" if role == "v5" else "candidate"].to_numpy(dtype=float)
    if (not np.isfinite(values).all() or np.any(values < 0)
            or not np.isfinite(frame.target.to_numpy(dtype=float)).all()
            or not np.array_equal(frame.target.to_numpy(dtype=float),
                                  rows.target.to_numpy(dtype=float)[positions])
            or not np.array_equal(frame.month.to_numpy(dtype=int),
                                  rows.month.to_numpy(dtype=int)[positions])
            or not np.array_equal(frame.airport.astype("string").to_numpy(),
                                  rows.airport.astype("string").to_numpy()[positions])
            or not np.array_equal(pd.to_datetime(frame.MVT_TIME_UTC_mvt, utc=True).to_numpy(),
                                  pd.to_datetime(rows.time.iloc[positions], utc=True).to_numpy())):
        raise ValueError(f"Independent {role} parent OOF metadata/labels differ")
    proxy = rows.proxy.to_numpy(dtype=float)[positions]
    valid = np.isfinite(proxy) & (proxy >= 0) & (proxy <= 7200)
    if not np.array_equal(frame.a_valid.to_numpy(dtype=bool), valid):
        raise ValueError(f"Independent {role} valid-AOBT mask differs")
    for name, months in FOLDS.items():
        if not frame.loc[frame.fold.eq(name), "month"].isin(months).all():
            raise ValueError(f"Independent {role} fold/month mapping differs")
    return frame


def evaluate(args: argparse.Namespace) -> dict:
    root, out, parent, files, snapshot = real_context(args, heavy=True)
    p = parent_args(files, out)
    ensure_protocol(root, out, parent, snapshot)
    report_path = out / "validation.json"
    prediction_path = out / "validation_predictions.parquet"
    if report_path.exists() or prediction_path.exists():
        raise FileExistsError("Replica validation outputs already exist")
    manifest = read_json(out / "features_manifest.json")
    for fold in FOLDS:
        verify_fold(root, out, files, fold, manifest, snapshot)
    rows = movement.read_baseline_rows(p.cache_dir,
                                       ["MVT_ID_mvt", "target", "proxy", "airport",
                                        "month", "time"])
    v5 = read_parent_oof(files, "v5", rows)
    positions = pd.Index(rows.MVT_ID_mvt).get_indexer(v5.MVT_ID_mvt)
    gate = movement.read_gate(p.cache_dir, rows)[positions]
    gate &= ~v5.a_valid.to_numpy(dtype=bool)
    parts = []
    for fold in FOLDS:
        part = pd.read_parquet(out / f"{fold}_oof.parquet",
                               columns=["MVT_ID_mvt", "fold", "expert"])
        if not part.fold.eq(fold).all() or part.MVT_ID_mvt.duplicated().any():
            raise ValueError(f"Replica {fold} OOF identity differs")
        parts.append(part)
    source = pd.concat(parts, ignore_index=True)
    if source.MVT_ID_mvt.duplicated().any():
        raise ValueError("Replica fold OOF IDs overlap")
    aligned = source.set_index("MVT_ID_mvt").reindex(v5.MVT_ID_mvt)
    expert = aligned.expert.to_numpy(dtype=float)
    if (not np.array_equal(np.isfinite(expert), gate)
            or not np.array_equal(aligned.fold.to_numpy()[gate], v5.fold.to_numpy()[gate])):
        raise ValueError("Replica expert does not cover exactly the missing-clock gate")
    y = v5.target.to_numpy(dtype=float)
    frozen = v5.selected.to_numpy(dtype=float)
    candidates: dict[float, np.ndarray] = {}
    for weight in WEIGHTS:
        pred = frozen.copy()
        pred[gate] = np.maximum((1 - weight) * frozen[gate] + weight * expert[gate], 0)
        candidates[weight] = pred
    scores = {}
    for fold in FOLDS:
        mask = v5.fold.eq(fold).to_numpy(dtype=bool)
        scores[fold] = {"n": int(mask.sum()), "gate_n": int((mask & gate).sum()),
                        "all_finite_rmse_sec": {str(weight): movement.rmse(y[mask], pred[mask])
                                                for weight, pred in candidates.items()},
                        "gate_rmse_sec": {str(weight): movement.rmse(y[mask & gate],
                                                                    pred[mask & gate])
                                          for weight, pred in candidates.items()}}
    selected = min(WEIGHTS, key=lambda w:
                   (scores["seasonal_jan_jul"]["all_finite_rmse_sec"][str(w)], w))
    stability = {}
    for offset, fold in enumerate(FOLDS):
        mask = v5.fold.eq(fold).to_numpy(dtype=bool)
        stability[fold] = movement.day_bootstrap(
            v5, frozen, candidates[FIXED_WEIGHT], mask, movement.SEED + offset)
    passed = bool(selected == FIXED_WEIGHT and all(
        scores[fold]["all_finite_rmse_sec"][str(FIXED_WEIGHT)]
        < scores[fold]["all_finite_rmse_sec"]["0.0"]
        and stability[fold]["gain_ci95_sec"][0] > 0 for fold in FOLDS))
    output = pd.DataFrame({"MVT_ID_mvt": v5.MVT_ID_mvt, "fold": v5.fold,
                           "target": y, "airport": v5.airport, "month": v5.month,
                           "MVT_TIME_UTC_mvt": v5.MVT_TIME_UTC_mvt,
                           "gate": gate, "v5": frozen, "expert": expert,
                           "selected_candidate": candidates[FIXED_WEIGHT]})
    assert_snapshot(root, root / "parents/v8_parents.json", snapshot)
    write_new_parquet(prediction_path, output, root)
    report = {"selected_weight": FIXED_WEIGHT, "seasonal_selected_weight": selected,
              "both_existing_folds_passed": passed, "scores": scores,
              "day_stability": stability, "weights": list(WEIGHTS),
              "selection": "Jan/Jul all-finite RMSE, smaller-weight ties",
              "v5_parent_sha256": snapshot["file_sha256"]["v5_validation_oof"],
              "features_manifest_sha256": sha256(out / "features_manifest.json"),
              "prediction_sha256": sha256(prediction_path),
              "fold_report_sha256": {name: sha256(out / f"{name}_fit.json") for name in FOLDS},
              "parent_manifest_sha256": snapshot["manifest_sha256"],
              "protocol_sha256": sha256(out / "protocol.json"),
              "promotion_pending": "paired April/October and February/August audits"}
    assert_snapshot(root, root / "parents/v8_parents.json", snapshot)
    write_new_json(report_path, report, root)
    return {"selected_weight": selected, "passed": passed,
            "gate_rows": int(gate.sum()), "all_finite_rows": len(v5)}


def require_validation(root: Path, out: Path, files: dict[str, Path],
                       snapshot: dict) -> dict:
    require_memory()
    report = read_json(out / "validation.json")
    for fold in FOLDS:
        verify_fold(root, out, files, fold,
                    read_json(out / "features_manifest.json"), snapshot)
    if (report.get("selected_weight") != FIXED_WEIGHT
            or report.get("seasonal_selected_weight") != FIXED_WEIGHT
            or not report.get("both_existing_folds_passed")
            or report.get("parent_manifest_sha256") != snapshot["manifest_sha256"]
            or report.get("prediction_sha256")
            != sha256(out / "validation_predictions.parquet")
            or report.get("features_manifest_sha256")
            != sha256(out / "features_manifest.json")
            or report.get("protocol_sha256") != sha256(out / "protocol.json")):
        raise ValueError("Replica original two-fold gate did not pass or changed")
    for fold in FOLDS:
        if report.get("fold_report_sha256", {}).get(fold) != sha256(out / f"{fold}_fit.json"):
            raise ValueError(f"Replica {fold} model/OOF report changed after validation")
        if (report["scores"][fold]["all_finite_rmse_sec"]["1.0"]
                >= report["scores"][fold]["all_finite_rmse_sec"]["0.0"]
                or report["day_stability"][fold]["gain_ci95_sec"][0] <= 0):
            raise ValueError(f"Replica original {fold} RMSE/day gate failed")
    p = parent_args(files, out)
    rows = movement.read_baseline_rows(p.cache_dir,
                                       ["MVT_ID_mvt", "target", "proxy", "airport",
                                        "month", "time"])
    v5 = read_parent_oof(files, "v5", rows)
    saved = pd.read_parquet(out / "validation_predictions.parquet")
    fields = ["MVT_ID_mvt", "fold", "target", "airport", "month",
              "MVT_TIME_UTC_mvt", "gate", "v5", "expert", "selected_candidate"]
    if (list(saved) != fields or len(saved) != EXPECTED_OOF_ROWS
            or not np.array_equal(saved.MVT_ID_mvt.to_numpy(), v5.MVT_ID_mvt.to_numpy())
            or not np.array_equal(saved.target.to_numpy(dtype=float),
                                  v5.target.to_numpy(dtype=float))
            or not np.array_equal(saved.fold.to_numpy(), v5.fold.to_numpy())
            or not np.array_equal(saved.month.to_numpy(dtype=int),
                                  v5.month.to_numpy(dtype=int))
            or not np.array_equal(saved.airport.astype("string").to_numpy(),
                                  v5.airport.astype("string").to_numpy())
            or not np.array_equal(pd.to_datetime(saved.MVT_TIME_UTC_mvt,
                                                 utc=True).to_numpy(),
                                  pd.to_datetime(v5.MVT_TIME_UTC_mvt,
                                                 utc=True).to_numpy())
            or not np.array_equal(saved.v5.to_numpy(dtype=float),
                                  v5.selected.to_numpy(dtype=float))):
        raise ValueError("Replica saved all-finite validation metadata differs")
    positions = pd.Index(rows.MVT_ID_mvt).get_indexer(v5.MVT_ID_mvt)
    gate = (movement.read_gate(p.cache_dir, rows)[positions]
            & ~v5.a_valid.to_numpy(dtype=bool))
    if not np.array_equal(saved.gate.to_numpy(dtype=bool), gate):
        raise ValueError("Replica validation missing-clock mask differs")
    parts = [pd.read_parquet(out / f"{fold}_oof.parquet",
                             columns=["MVT_ID_mvt", "expert"]) for fold in FOLDS]
    source = pd.concat(parts, ignore_index=True)
    if source.MVT_ID_mvt.duplicated().any():
        raise ValueError("Replica fold OOF ID overlap during validation replay")
    expert = source.set_index("MVT_ID_mvt").reindex(v5.MVT_ID_mvt).expert.to_numpy(
        dtype=float)
    if (not np.array_equal(np.isfinite(expert), gate)
            or not np.array_equal(saved.expert.to_numpy(dtype=float)[gate], expert[gate])):
        raise ValueError("Replica validated expert differs from saved model OOF")
    frozen = v5.selected.to_numpy(dtype=float)
    y = v5.target.to_numpy(dtype=float)
    candidates = {}
    for weight in WEIGHTS:
        prediction = frozen.copy()
        prediction[gate] = np.maximum((1 - weight) * frozen[gate]
                                      + weight * expert[gate], 0)
        candidates[weight] = prediction
    if not np.array_equal(saved.selected_candidate.to_numpy(dtype=float),
                          candidates[FIXED_WEIGHT]):
        raise ValueError("Replica saved fixed policy formula differs")
    for offset, fold in enumerate(FOLDS):
        mask = v5.fold.eq(fold).to_numpy(dtype=bool)
        item = report["scores"][fold]
        for weight, prediction in candidates.items():
            if (not np.isclose(item["all_finite_rmse_sec"][str(weight)],
                               movement.rmse(y[mask], prediction[mask]),
                               rtol=0, atol=1e-10)
                    or not np.isclose(item["gate_rmse_sec"][str(weight)],
                                      movement.rmse(y[mask & gate],
                                                    prediction[mask & gate]),
                                      rtol=0, atol=1e-10)):
                raise ValueError(f"Replica {fold} saved fixed-weight scores differ")
        replay = movement.day_bootstrap(v5, frozen, candidates[FIXED_WEIGHT],
                                        mask, movement.SEED + offset)
        stored = report["day_stability"][fold]
        if (replay["days"] != stored["days"]
                or replay["repeats"] != stored["repeats"]
                or not np.allclose(replay["gain_ci95_sec"], stored["gain_ci95_sec"],
                                   rtol=0, atol=1e-10)):
            raise ValueError(f"Replica {fold} saved day CI differs")
    selected = min(WEIGHTS, key=lambda weight:
                   (report["scores"]["seasonal_jan_jul"]["all_finite_rmse_sec"][str(weight)],
                    weight))
    if selected != FIXED_WEIGHT:
        raise ValueError("Replica seasonal selection no longer chooses fixed weight one")
    return report


def catboost_complement_split(rows: pd.DataFrame, train_mask: np.ndarray,
                              heldout: np.ndarray) -> dict:
    """Mirror the published ordinary direct CatBoost fit/early split exactly."""
    if np.any(train_mask & heldout):
        raise ValueError("CatBoost comparator included a heldout month")
    indexes = np.flatnonzero(train_mask)
    rng = np.random.default_rng(2026 + len("ordinary_direct"))
    shuffled = rng.permutation(indexes)
    n_early = max(50 if len(indexes) < 1000 else 500, int(.1 * len(indexes)))
    n_early = min(n_early, max(1, len(indexes) // 3))
    early, fit = shuffled[:n_early], shuffled[n_early:]
    return {"ordered_fit_ids_sha256": ordered_sha(rows.MVT_ID_mvt.iloc[fit]),
            "ordered_early_ids_sha256": ordered_sha(rows.MVT_ID_mvt.iloc[early]),
            "train_rows": int(len(indexes)), "fit_rows": int(len(fit)),
            "early_rows": int(len(early))}


def audit(args: argparse.Namespace, stem: str) -> dict:
    if stem not in ("april_october", "february_august"):
        raise ValueError("Only predeclared matched month audits are available")
    root, out, parent, files, snapshot = real_context(args, heavy=True)
    p = parent_args(files, out)
    ensure_protocol(root, out, parent, snapshot)
    validation = require_validation(root, out, files, snapshot)
    if stem == "february_august":
        require_audit(root, out, files, snapshot, "april_october")
    months = (4, 10) if stem == "april_october" else (2, 8)
    seed = movement.SEED + 19 if stem == "april_october" else movement.SEED
    movement_path = out / f"{stem}_movement.txt"
    comparator_path = out / f"{stem}_reference.cbm"
    prediction_path = out / f"{stem}_predictions.parquet"
    report_path = out / ("fresh_audit.json" if stem == "april_october" else "reserved_audit.json")
    if any(path.exists() for path in (movement_path, comparator_path,
                                      prediction_path, report_path)):
        raise FileExistsError("Replica paired audit outputs already exist")
    features, rows, manifest = load_prepared(root, out, files, snapshot)
    y = rows.target.to_numpy(dtype=float)
    ordinary = ordinary_masks(rows, months)
    scored = ordinary["heldout"] & movement.read_gate(p.cache_dir, rows) & np.isfinite(y)
    if scored.sum() < 100:
        raise ValueError("Insufficient finite no-NM matched audit gate rows")
    indexes = np.flatnonzero(scored)
    frame = rows.iloc[indexes][["MVT_ID_mvt", "target", "airport", "month", "time"]].copy()
    frame = frame.rename(columns={"time": "MVT_TIME_UTC_mvt"})
    expert, movement_fit = movement.fit_movement_model(
        p, features, rows, manifest["categorical"], months)
    if (movement_fit["fit_rows"] != int(ordinary["fit"].sum())
            or movement_fit["internal_early_rows"] != int(ordinary["early"].sum())):
        raise ValueError("Movement audit fit/early IDs differ from original policy")
    movement_prediction = np.maximum(expert.predict(
        features.iloc[indexes], num_iteration=movement_fit["best_round"],
        num_threads=3), 0)
    if not np.isfinite(movement_prediction).all():
        raise ValueError("Movement paired audit prediction is nonfinite")
    movement_split = split_receipt(rows, ordinary, scored)
    del features
    gc.collect()

    from missing_catboost import columns as catboost_columns
    from missing_catboost import load_inputs as catboost_load_inputs
    from missing_catboost import train_model as catboost_train_model
    cb_rows, cb_features = catboost_load_inputs(p)
    if not np.array_equal(cb_rows.MVT_ID_mvt.to_numpy(), rows.MVT_ID_mvt.to_numpy()):
        raise ValueError("Comparator rows differ from prepared movement IDs")
    cb_y = cb_rows.target.to_numpy(dtype=float)
    cb_proxy = cb_rows.proxy.to_numpy(dtype=float)
    cb_no_nm = (cb_features.AOBT_3_flt_missing.to_numpy(dtype=bool)
                & cb_features.LOBT_flt_missing.to_numpy(dtype=bool))
    cb_train = (cb_no_nm & ~np.isfinite(cb_proxy) & np.isfinite(cb_y)
                & (cb_y >= 0) & (cb_y <= 7200) & ~ordinary["heldout"])
    cb_names, cb_cats = catboost_columns(cb_features, long=False)
    cb_split = catboost_complement_split(cb_rows, cb_train, ordinary["heldout"])
    comparator, comparator_fit = catboost_train_model(
        p, "ordinary_direct", "regression", cb_features,
        np.flatnonzero(cb_train), cb_y, cb_names, cb_cats, 700, 6)
    if (comparator_fit["train_rows"] != cb_split["train_rows"]
            or comparator_fit["fit_rows"] != cb_split["fit_rows"]
            or comparator_fit["internal_early_rows"] != cb_split["early_rows"]):
        raise ValueError("Pure CatBoost comparator used unexpected fit/early split")
    reference_prediction = np.maximum(comparator.predict(
        cb_features.iloc[indexes][cb_names], thread_count=3), 0)
    if not np.isfinite(reference_prediction).all():
        raise ValueError("CatBoost paired comparator prediction is nonfinite")
    del cb_features, cb_rows
    gc.collect()
    fixed = np.maximum((1 - FIXED_WEIGHT) * reference_prediction
                       + FIXED_WEIGHT * movement_prediction, 0)
    yy = frame.target.to_numpy(dtype=float)
    month = frame.month.to_numpy(dtype=int)
    scores = {str(m): {"n": int((month == m).sum()),
                       "reference_rmse_sec": movement.rmse(yy[month == m],
                                                           reference_prediction[month == m]),
                       "candidate_rmse_sec": movement.rmse(yy[month == m], fixed[month == m])}
              for m in months}
    pooled = movement.day_bootstrap(frame, reference_prediction, fixed,
                                    np.ones(len(frame), dtype=bool), seed)
    passed = bool(all(scores[str(m)]["candidate_rmse_sec"]
                      < scores[str(m)]["reference_rmse_sec"] for m in months)
                  and pooled["gain_ci95_sec"][0] > 0)
    frame["reference_direct"] = reference_prediction.astype("float32")
    frame["movement_direct"] = movement_prediction.astype("float32")
    frame["fixed_blend"] = fixed.astype("float32")
    assert_snapshot(root, root / "parents/v8_parents.json", snapshot)
    save_new_model(movement_path, expert, root)
    save_new_model(comparator_path, comparator, root)
    write_new_parquet(prediction_path, frame, root)
    report = {"heldout_months": list(months), "fixed_weight": FIXED_WEIGHT,
              "original_validation_sha256": sha256(out / "validation.json"),
              "features_manifest_sha256": sha256(out / "features_manifest.json"),
              "prepared_features_sha256": sha256(out / "features.parquet"),
              "row_ids_sha256": sha256(out / "row_ids.parquet"),
              "prediction_sha256": sha256(prediction_path),
              "gate_rows": len(frame), "coverage_verified": True,
              "comparator_architecture": "ordinary direct CatBoost 700-cap depth6, 70-patience, seed2026",
              "comparator_feature_names": cb_names, "comparator_categorical": cb_cats,
              "comparator_split": cb_split, "comparator_training": comparator_fit,
              "movement_split": movement_split, "movement_training": movement_fit,
              "model_sha256": {"catboost": sha256(comparator_path),
                               "movement": sha256(movement_path)},
              "month_scores": scores, "pooled_day_stability": pooled,
              "bootstrap_seed": seed, "passed": passed,
              "no_in_sample_v5_predictions": True,
              "parent_manifest_sha256": snapshot["manifest_sha256"],
              "protocol_sha256": sha256(out / "protocol.json"),
              "prior_original_weight_sha256": sha256(out / "validation.json")}
    if stem == "february_august":
        report["reserved_guard_protocol_sha256"] = sha256(
            SOURCE_ROOT / "reports/reserved_guard_protocol.json")
        report["fresh_audit_sha256"] = sha256(out / "fresh_audit.json")
    assert_snapshot(root, root / "parents/v8_parents.json", snapshot)
    write_new_json(report_path, report, root)
    return {"audit": stem, "passed": passed, "gate_rows": len(frame),
            "month_scores": scores, "pooled_gain_ci95_sec": pooled["gain_ci95_sec"]}


def require_audit(root: Path, out: Path, files: dict[str, Path],
                  snapshot: dict, stem: str) -> dict:
    require_memory()
    months = (4, 10) if stem == "april_october" else (2, 8)
    report_path = out / ("fresh_audit.json" if stem == "april_october" else "reserved_audit.json")
    report = read_json(report_path)
    prediction_path = out / f"{stem}_predictions.parquet"
    if (report.get("heldout_months") != list(months)
            or report.get("fixed_weight") != FIXED_WEIGHT
            or not report.get("passed")
            or not report.get("coverage_verified")
            or not report.get("no_in_sample_v5_predictions")
            or report.get("prediction_sha256") != sha256(prediction_path)
            or report.get("model_sha256", {}).get("movement")
            != sha256(out / f"{stem}_movement.txt")
            or report.get("model_sha256", {}).get("catboost")
            != sha256(out / f"{stem}_reference.cbm")
            or report.get("original_validation_sha256") != sha256(out / "validation.json")
            or report.get("features_manifest_sha256")
            != sha256(out / "features_manifest.json")
            or report.get("prepared_features_sha256") != sha256(out / "features.parquet")
            or report.get("row_ids_sha256") != sha256(out / "row_ids.parquet")
            or report.get("parent_manifest_sha256") != snapshot["manifest_sha256"]
            or report.get("protocol_sha256") != sha256(out / "protocol.json")):
        raise ValueError(f"Replica {stem} audit receipt failed")
    if stem == "february_august" and (
            report.get("fresh_audit_sha256") != sha256(out / "fresh_audit.json")
            or report.get("reserved_guard_protocol_sha256")
            != sha256(SOURCE_ROOT / "reports/reserved_guard_protocol.json")):
        raise ValueError("Replica February/August guard source changed")
    for month in months:
        item = report["month_scores"][str(month)]
        if item["candidate_rmse_sec"] >= item["reference_rmse_sec"]:
            raise ValueError(f"Replica {stem} month {month} failed")
    if report["pooled_day_stability"]["gain_ci95_sec"][0] <= 0:
        raise ValueError(f"Replica {stem} pooled day CI failed")
    paired = pd.read_parquet(prediction_path)
    expected_columns = ["MVT_ID_mvt", "target", "airport", "month",
                        "MVT_TIME_UTC_mvt", "reference_direct",
                        "movement_direct", "fixed_blend"]
    if (list(paired) != expected_columns or len(paired) != report["gate_rows"]
            or paired.MVT_ID_mvt.isna().any()
            or paired.MVT_ID_mvt.duplicated().any()):
        raise ValueError(f"Replica {stem} paired schema/ID coverage differs")
    p = parent_args(files, out)
    rows = movement.read_baseline_rows(p.cache_dir,
                                       ["MVT_ID_mvt", "target", "proxy", "airport",
                                        "month", "time"])
    expected = (rows.month.isin(months).to_numpy(dtype=bool)
                & movement.read_gate(p.cache_dir, rows)
                & np.isfinite(rows.target.to_numpy(dtype=float)))
    selected = rows.loc[expected]
    if (not np.array_equal(paired.MVT_ID_mvt.to_numpy(), selected.MVT_ID_mvt.to_numpy())
            or not np.array_equal(paired.target.to_numpy(dtype=float),
                                  selected.target.to_numpy(dtype=float))
            or not np.array_equal(paired.month.to_numpy(dtype=int),
                                  selected.month.to_numpy(dtype=int))
            or not np.array_equal(paired.airport.astype("string").to_numpy(),
                                  selected.airport.astype("string").to_numpy())
            or not np.array_equal(pd.to_datetime(paired.MVT_TIME_UTC_mvt, utc=True).to_numpy(),
                                  pd.to_datetime(selected.time, utc=True).to_numpy())):
        raise ValueError(f"Replica {stem} paired IDs/labels/months/time differ")
    ordinary = ordinary_masks(rows, months)
    if report.get("movement_split") != split_receipt(rows, ordinary, expected):
        raise ValueError(f"Replica {stem} movement fit/early/scored IDs differ")
    flags = pd.read_parquet(p.cache_dir / "features.parquet",
                            columns=["AOBT_3_flt_missing", "LOBT_flt_missing"])
    cb_y = rows.target.to_numpy(dtype=float)
    cb_train = (flags.AOBT_3_flt_missing.to_numpy(dtype=bool)
                & flags.LOBT_flt_missing.to_numpy(dtype=bool)
                & ~np.isfinite(rows.proxy.to_numpy(dtype=float))
                & np.isfinite(cb_y) & (cb_y >= 0) & (cb_y <= 7200)
                & ~ordinary["heldout"])
    if report.get("comparator_split") != catboost_complement_split(
            rows, cb_train, ordinary["heldout"]):
        raise ValueError(f"Replica {stem} comparator fit/early IDs differ")
    movement_features, movement_rows, prepared = load_prepared(root, out, files, snapshot)
    if not np.array_equal(movement_rows.MVT_ID_mvt.to_numpy(), rows.MVT_ID_mvt.to_numpy()):
        raise ValueError(f"Replica {stem} prepared movement rows changed")
    import lightgbm as lgb
    movement_model = lgb.Booster(model_file=str(out / f"{stem}_movement.txt"))
    best_round = int(report["movement_training"]["best_round"])
    if (movement_model.current_iteration() != best_round
            or not 1 <= best_round <= 1200
            or list(movement_model.feature_name()) != prepared["features"]):
        raise ValueError(f"Replica {stem} saved movement model architecture differs")
    replay_movement = np.maximum(movement_model.predict(
        movement_features.loc[expected], num_iteration=best_round,
        num_threads=3), 0).astype("float32")
    if not np.array_equal(replay_movement,
                          paired.movement_direct.to_numpy(dtype=np.float32)):
        raise ValueError(f"Replica {stem} movement OOF differs from saved model")
    del movement_features, movement_rows, movement_model
    gc.collect()
    from catboost import CatBoostRegressor
    from missing_catboost import load_inputs as catboost_load_inputs
    cb_rows, cb_features = catboost_load_inputs(p)
    if (not np.array_equal(cb_rows.MVT_ID_mvt.to_numpy(), rows.MVT_ID_mvt.to_numpy())
            or not np.array_equal(cb_rows.target.to_numpy(dtype=float), cb_y)):
        raise ValueError(f"Replica {stem} CatBoost comparator data changed")
    comparator_model = CatBoostRegressor()
    comparator_model.load_model(str(out / f"{stem}_reference.cbm"))
    cb_names = report["comparator_feature_names"]
    cb_params = comparator_model.get_all_params()
    if (list(comparator_model.feature_names_) != cb_names
            or comparator_model.tree_count_ != int(report["comparator_training"]["best_iteration"]) + 1
            or not 1 <= comparator_model.tree_count_ <= 700
            or int(cb_params.get("depth", -1)) != 6
            or int(cb_params.get("random_seed", -1)) != 2026
            or cb_params.get("loss_function") != "RMSE"):
        raise ValueError(f"Replica {stem} saved comparator model architecture differs")
    replay_reference = np.maximum(comparator_model.predict(
        cb_features.loc[expected, cb_names], thread_count=3), 0).astype("float32")
    if not np.array_equal(replay_reference,
                          paired.reference_direct.to_numpy(dtype=np.float32)):
        raise ValueError(f"Replica {stem} comparator OOF differs from saved model")
    del cb_rows, cb_features, comparator_model
    gc.collect()
    yy = paired.target.to_numpy(dtype=float)
    reference = paired.reference_direct.to_numpy(dtype=float)
    expert = paired.movement_direct.to_numpy(dtype=float)
    candidate = paired.fixed_blend.to_numpy(dtype=float)
    if (not np.isfinite(reference).all() or not np.isfinite(expert).all()
            or not np.isfinite(candidate).all() or np.any(candidate < 0)
            or not np.allclose(candidate, np.maximum(
                (1 - FIXED_WEIGHT) * reference + FIXED_WEIGHT * expert, 0),
                               rtol=0, atol=1e-5)):
        raise ValueError(f"Replica {stem} fixed formula differs")
    for month in months:
        mask = paired.month.eq(month).to_numpy(dtype=bool)
        item = report["month_scores"][str(month)]
        if (not np.isclose(movement.rmse(yy[mask], reference[mask]),
                           item["reference_rmse_sec"], rtol=0, atol=1e-4)
                or not np.isclose(movement.rmse(yy[mask], candidate[mask]),
                                  item["candidate_rmse_sec"], rtol=0, atol=1e-4)):
            raise ValueError(f"Replica {stem} saved month scores differ")
    replay = movement.day_bootstrap(
        paired, reference, candidate, np.ones(len(paired), dtype=bool),
        movement.SEED + 19 if stem == "april_october" else movement.SEED)
    saved = report["pooled_day_stability"]
    if (replay["days"] != saved["days"] or replay["repeats"] != saved["repeats"]
            or not np.allclose(replay["gain_ci95_sec"], saved["gain_ci95_sec"],
                               rtol=0, atol=1e-4)
            or not np.isclose(replay["observed_gain_sec"],
                              saved["observed_gain_sec"], rtol=0, atol=1e-4)):
        raise ValueError(f"Replica {stem} saved day stability differs")
    return report


def compose_oof(args: argparse.Namespace) -> dict:
    root, out, parent, files, snapshot = real_context(args)
    p = parent_args(files, out)
    ensure_protocol(root, out, parent, snapshot)
    require_validation(root, out, files, snapshot)
    fresh = require_audit(root, out, files, snapshot, "april_october")
    reserved = require_audit(root, out, files, snapshot, "february_august")
    report_path = out / "composition_report.json"
    prediction_path = out / "composition_predictions.parquet"
    if report_path.exists() or prediction_path.exists():
        raise FileExistsError("Replica composition outputs already exist")
    rows = movement.read_baseline_rows(p.cache_dir,
                                       ["MVT_ID_mvt", "target", "proxy", "airport",
                                        "month", "time"])
    v7 = read_parent_oof(files, "v7", rows)
    v5 = read_parent_oof(files, "v5", rows).set_index("MVT_ID_mvt").reindex(
        v7.MVT_ID_mvt).reset_index()
    movement_oof = pd.read_parquet(out / "validation_predictions.parquet")
    if (movement_oof.MVT_ID_mvt.isna().any() or movement_oof.MVT_ID_mvt.duplicated().any()
            or len(movement_oof) != len(v7)):
        raise ValueError("Replica missing-clock OOF lacks common finite universe")
    movement_oof = movement_oof.set_index("MVT_ID_mvt").reindex(
        v7.MVT_ID_mvt).reset_index()
    if (movement_oof.expert.notna().sum() != 7_841
            or not np.array_equal(v7.MVT_ID_mvt.to_numpy(), v5.MVT_ID_mvt.to_numpy())
            or not np.array_equal(v7.MVT_ID_mvt.to_numpy(),
                                  movement_oof.MVT_ID_mvt.to_numpy())
            or not np.array_equal(v7.target.to_numpy(dtype=float),
                                  v5.target.to_numpy(dtype=float))
            or not np.array_equal(v7.target.to_numpy(dtype=float),
                                  movement_oof.target.to_numpy(dtype=float))
            or not np.array_equal(v7.a_valid.to_numpy(dtype=bool),
                                  v5.a_valid.to_numpy(dtype=bool))
            or not np.array_equal(v7.fold.to_numpy(), movement_oof.fold.to_numpy())
            or not np.array_equal(v7.month.to_numpy(dtype=int),
                                  movement_oof.month.to_numpy(dtype=int))
            or not np.array_equal(v7.airport.astype("string").to_numpy(),
                                  movement_oof.airport.astype("string").to_numpy())
            or not np.array_equal(pd.to_datetime(v7.MVT_TIME_UTC_mvt, utc=True).to_numpy(),
                                  pd.to_datetime(movement_oof.MVT_TIME_UTC_mvt,
                                                 utc=True).to_numpy())):
        raise ValueError("Replica v5/v7/movement OOF ID/label/gate metadata differ")
    positions = pd.Index(rows.MVT_ID_mvt).get_indexer(v7.MVT_ID_mvt)
    proxy = rows.proxy.to_numpy(dtype=float)[positions]
    valid = np.isfinite(proxy) & (proxy >= 0) & (proxy <= 7200)
    missing = movement.read_gate(p.cache_dir, rows)[positions]
    if (np.any(valid & missing) or not np.array_equal(valid, v7.a_valid.to_numpy(dtype=bool))
            or not np.array_equal(missing, movement_oof.gate.to_numpy(dtype=bool))
            or not np.array_equal(v5.selected.to_numpy(dtype=float)[missing],
                                  v7.candidate.to_numpy(dtype=float)[missing])
            or not np.array_equal(v5.selected.to_numpy(dtype=float),
                                  movement_oof.v5.to_numpy(dtype=float))):
        raise ValueError("Replica v5/v7 missing route is not the published disjoint policy")
    expert = movement_oof.expert.to_numpy(dtype=float)
    old = v7.candidate.to_numpy(dtype=float)
    component = movement_oof.selected_candidate.to_numpy(dtype=float)
    if (not np.array_equal(np.isfinite(expert), missing)
            or not np.isfinite(component).all()
            or not np.array_equal(component[~missing],
                                  v5.selected.to_numpy(dtype=float)[~missing])
            or not np.allclose(component[missing], np.maximum(expert[missing], 0),
                               rtol=0, atol=1e-6)):
        raise ValueError("Replica fixed-weight missing component differs")
    combined = old.copy()
    combined[missing] = component[missing]
    if (not np.array_equal(combined[~missing], old[~missing])
            or not np.isfinite(combined).all() or np.any(combined < 0)):
        raise ValueError("Replica OOF changed non-missing route or is invalid")
    scores = {}
    for fold in FOLDS:
        mask = v7.fold.eq(fold).to_numpy(dtype=bool)
        bootstrap = movement.day_bootstrap(v7, old, combined, mask, 20261003)
        yy = v7.target.to_numpy(dtype=float)[mask]
        before, after = movement.rmse(yy, old[mask]), movement.rmse(yy, combined[mask])
        scores[fold] = {"rows": int(mask.sum()), "missing_gate_rows": int((mask & missing).sum()),
                        "v7_rmse": before, "combined_rmse": after, "bootstrap": bootstrap,
                        "passed": bool(after < before and bootstrap["gain_ci95_sec"][0] > 0)}
    if (scores["seasonal_jan_jul"]["missing_gate_rows"] != 4976
            or scores["forward_nov_dec"]["missing_gate_rows"] != 2865):
        raise ValueError("Replica published missing-clock OOF gate coverage differs")
    passed = all(item["passed"] for item in scores.values())
    output = pd.DataFrame({"MVT_ID_mvt": v7.MVT_ID_mvt, "target": v7.target,
                           "fold": v7.fold, "month": v7.month,
                           "MVT_TIME_UTC_mvt": v7.MVT_TIME_UTC_mvt,
                           "a_valid": valid, "missing_gate": missing,
                           "v7": old, "missing_component": component,
                           "combined": combined})
    assert_snapshot(root, root / "parents/v8_parents.json", snapshot)
    write_new_parquet(prediction_path, output, root)
    report = {"passed": passed, "scores": scores,
              "rows_all_finite": len(v7), "n_valid": int(valid.sum()),
              "n_missing_clock": int(missing.sum()),
              "valid_route": "v7_unchanged", "missing_route": "fixed_movement_weight_1",
              "disjoint_masks_verified": True, "outside_masks_unchanged": True,
              "v5_v7_missing_equal": True,
              "parent_v5_oof_sha256": snapshot["file_sha256"]["v5_validation_oof"],
              "parent_v7_oof_sha256": snapshot["file_sha256"]["v7_validation_oof"],
              "movement_validation_sha256": sha256(out / "validation.json"),
              "fresh_audit_sha256": sha256(out / "fresh_audit.json"),
              "reserved_audit_sha256": sha256(out / "reserved_audit.json"),
              "fresh_passed": bool(fresh["passed"]),
              "reserved_passed": bool(reserved["passed"]),
              "validation_predictions_sha256": sha256(prediction_path),
              "parent_manifest_sha256": snapshot["manifest_sha256"],
              "protocol_sha256": sha256(out / "protocol.json"),
              "decision": "accepted_locally" if passed else "rejected_without_retuning"}
    assert_snapshot(root, root / "parents/v8_parents.json", snapshot)
    write_new_json(report_path, report, root)
    return {"passed": passed, "scores": scores,
            "missing_gate_rows": int(missing.sum()), "rows": len(v7)}


def require_composition(root: Path, out: Path, files: dict[str, Path],
                        snapshot: dict) -> dict:
    require_validation(root, out, files, snapshot)
    require_audit(root, out, files, snapshot, "april_october")
    require_audit(root, out, files, snapshot, "february_august")
    report = read_json(out / "composition_report.json")
    if (not report.get("passed") or report.get("decision") != "accepted_locally"
            or report.get("rows_all_finite") != EXPECTED_OOF_ROWS
            or report.get("n_missing_clock") != 7_841
            or report.get("valid_route") != "v7_unchanged"
            or report.get("missing_route") != "fixed_movement_weight_1"
            or report.get("parent_v5_oof_sha256")
            != snapshot["file_sha256"]["v5_validation_oof"]
            or report.get("parent_v7_oof_sha256")
            != snapshot["file_sha256"]["v7_validation_oof"]
            or report.get("movement_validation_sha256") != sha256(out / "validation.json")
            or report.get("fresh_audit_sha256") != sha256(out / "fresh_audit.json")
            or report.get("reserved_audit_sha256") != sha256(out / "reserved_audit.json")
            or report.get("validation_predictions_sha256")
            != sha256(out / "composition_predictions.parquet")
            or report.get("parent_manifest_sha256") != snapshot["manifest_sha256"]
            or report.get("protocol_sha256") != sha256(out / "protocol.json")):
        raise ValueError("Replica full-policy composition gate is absent or changed")
    for fold in FOLDS:
        item = report["scores"][fold]
        if (not item["passed"] or item["combined_rmse"] >= item["v7_rmse"]
                or item["bootstrap"]["gain_ci95_sec"][0] <= 0):
            raise ValueError(f"Replica full-policy {fold} gate failed")
    p = parent_args(files, out)
    rows = movement.read_baseline_rows(p.cache_dir,
                                       ["MVT_ID_mvt", "target", "proxy", "airport",
                                        "month", "time"])
    v7 = read_parent_oof(files, "v7", rows)
    paired = pd.read_parquet(out / "composition_predictions.parquet")
    expected_cols = ["MVT_ID_mvt", "target", "fold", "month", "MVT_TIME_UTC_mvt",
                     "a_valid", "missing_gate", "v7", "missing_component", "combined"]
    if (list(paired) != expected_cols or len(paired) != EXPECTED_OOF_ROWS
            or not np.array_equal(paired.MVT_ID_mvt.to_numpy(), v7.MVT_ID_mvt.to_numpy())
            or not np.array_equal(paired.target.to_numpy(dtype=float),
                                  v7.target.to_numpy(dtype=float))
            or not np.array_equal(paired.fold.to_numpy(), v7.fold.to_numpy())
            or not np.array_equal(paired.month.to_numpy(dtype=int),
                                  v7.month.to_numpy(dtype=int))
            or not np.array_equal(paired.v7.to_numpy(dtype=float),
                                  v7.candidate.to_numpy(dtype=float))):
        raise ValueError("Replica full-policy paired OOF metadata differs")
    positions = pd.Index(rows.MVT_ID_mvt).get_indexer(v7.MVT_ID_mvt)
    gate = movement.read_gate(p.cache_dir, rows)[positions]
    if (not np.array_equal(paired.missing_gate.to_numpy(dtype=bool), gate)
            or not np.array_equal(paired.a_valid.to_numpy(dtype=bool),
                                  v7.a_valid.to_numpy(dtype=bool))
            or np.any(gate & v7.a_valid.to_numpy(dtype=bool))):
        raise ValueError("Replica full-policy OOF gate changed")
    old = paired.v7.to_numpy(dtype=float)
    candidate = paired.combined.to_numpy(dtype=float)
    component = paired.missing_component.to_numpy(dtype=float)
    if (not np.array_equal(candidate[~gate], old[~gate])
            or not np.allclose(candidate[gate], np.maximum(component[gate], 0),
                               rtol=0, atol=1e-6)
            or not np.isfinite(candidate).all() or np.any(candidate < 0)):
        raise ValueError("Replica full-policy OOF changed outside missing route")
    for fold in FOLDS:
        mask = paired.fold.eq(fold).to_numpy(dtype=bool)
        item = report["scores"][fold]
        before = movement.rmse(paired.target.to_numpy(dtype=float)[mask], old[mask])
        after = movement.rmse(paired.target.to_numpy(dtype=float)[mask], candidate[mask])
        replay = movement.day_bootstrap(paired, old, candidate, mask, 20261003)
        if (not np.isclose(before, item["v7_rmse"], rtol=0, atol=1e-5)
                or not np.isclose(after, item["combined_rmse"], rtol=0, atol=1e-5)
                or not np.allclose(replay["gain_ci95_sec"],
                                   item["bootstrap"]["gain_ci95_sec"],
                                   rtol=0, atol=1e-5)):
            raise ValueError(f"Replica full-policy {fold} metric replay differs")
    return report


def fit_final(args: argparse.Namespace) -> dict:
    root, out, parent, files, snapshot = real_context(args, heavy=True)
    p = parent_args(files, out)
    ensure_protocol(root, out, parent, snapshot)
    require_composition(root, out, files, snapshot)
    model_path = out / "final_movement_only.txt"
    report_path = out / "final_model.json"
    if model_path.exists() or report_path.exists():
        raise FileExistsError("Replica final model/report already exist")
    features, rows, manifest = load_prepared(root, out, files, snapshot)
    fold_reports = {name: verify_fold(root, out, files, name, manifest, snapshot)
                    for name in FOLDS}
    rounds = int(np.median([fold_reports[name]["best_round"] for name in FOLDS]))
    if not 1 <= rounds <= 1200:
        raise ValueError("Original two-fold median is outside fixed LightGBM cap")
    y = rows.target.to_numpy(dtype=np.float32)
    ordinary = np.isfinite(y) & (y >= 0) & (y <= 7200)
    if ordinary.sum() != 2_084_094:
        raise ValueError("Replica ordinary full-data label coverage differs")
    import lightgbm as lgb
    train_set = lgb.Dataset(features.loc[ordinary], label=y[ordinary],
                            categorical_feature=manifest["categorical"],
                            free_raw_data=True)
    model = lgb.train(movement.model_params(3), train_set, num_boost_round=rounds,
                      callbacks=[lgb.log_evaluation(period=200)])
    if model.current_iteration() != rounds or list(model.feature_name()) != manifest["features"]:
        raise ValueError("Replica final model tree count or feature order differs")
    assert_snapshot(root, root / "parents/v8_parents.json", snapshot)
    save_new_model(model_path, model, root)
    report = {"final_rounds": rounds,
              "fold_best_rounds": {name: fold_reports[name]["best_round"] for name in FOLDS},
              "training_rows": int(ordinary.sum()),
              "ordered_training_ids_sha256": ordered_sha(rows.MVT_ID_mvt[ordinary]),
              "training_labels_f32_sha256": numeric_sha(y[ordinary], "<f4"),
              "feature_count": len(manifest["features"]),
              "categorical_count": len(manifest["categorical"]),
              "feature_schema_sha256": manifest["feature_schema_sha256"],
              "selected_weight": FIXED_WEIGHT,
              "model_sha256": sha256(model_path),
              "features_manifest_sha256": sha256(out / "features_manifest.json"),
              "validation_sha256": sha256(out / "validation.json"),
              "fresh_audit_sha256": sha256(out / "fresh_audit.json"),
              "reserved_audit_sha256": sha256(out / "reserved_audit.json"),
              "composition_sha256": sha256(out / "composition_report.json"),
              "parent_manifest_sha256": snapshot["manifest_sha256"],
              "protocol_sha256": sha256(out / "protocol.json"),
              "ranking_prediction_created": False}
    assert_snapshot(root, root / "parents/v8_parents.json", snapshot)
    write_new_json(report_path, report, root)
    return {"final_rounds": rounds, "training_rows": int(ordinary.sum()),
            "model_sha256": report["model_sha256"]}


def require_final(root: Path, out: Path, files: dict[str, Path],
                  snapshot: dict) -> dict:
    require_composition(root, out, files, snapshot)
    report = read_json(out / "final_model.json")
    fold_reports = {name: read_json(out / f"{name}_fit.json") for name in FOLDS}
    expected_rounds = int(np.median([fold_reports[name]["best_round"] for name in FOLDS]))
    if (report.get("model_sha256") != sha256(out / "final_movement_only.txt")
            or report.get("final_rounds") != expected_rounds
            or report.get("fold_best_rounds")
            != {name: fold_reports[name]["best_round"] for name in FOLDS}
            or report.get("training_rows") != 2_084_094
            or report.get("feature_count") != 79
            or report.get("categorical_count") != 13
            or report.get("selected_weight") != FIXED_WEIGHT
            or report.get("features_manifest_sha256")
            != sha256(out / "features_manifest.json")
            or report.get("validation_sha256") != sha256(out / "validation.json")
            or report.get("fresh_audit_sha256") != sha256(out / "fresh_audit.json")
            or report.get("reserved_audit_sha256") != sha256(out / "reserved_audit.json")
            or report.get("composition_sha256") != sha256(out / "composition_report.json")
            or report.get("parent_manifest_sha256") != snapshot["manifest_sha256"]
            or report.get("protocol_sha256") != sha256(out / "protocol.json")):
        raise ValueError("Replica final full-data model or gate receipt differs")
    return report


def ranking_snapshot(out: Path, files: dict[str, Path], parent_snapshot_value: dict) -> dict:
    names = ("raw_ranking", "submission_template", "baseline_ranking_rows",
             "baseline_ranking_features", "arrival_ranking_features",
             "noaa_weather", "v7_ranking_predictions")
    inventory = {name: {"path": str(files[name]), "sha256": sha256(files[name])}
                 for name in names}
    local = {"final_model": out / "final_movement_only.txt",
             "final_model_report": out / "final_model.json",
             "composition_report": out / "composition_report.json",
             "composition_predictions": out / "composition_predictions.parquet",
             "prepared_manifest": out / "features_manifest.json"}
    inventory.update({name: {"path": str(path), "sha256": sha256(path)}
                      for name, path in local.items()})
    return {"schema_version": 1, "purpose": "seal ranking bytes before any value read",
            "parent_manifest_sha256": parent_snapshot_value["manifest_sha256"],
            "source_sha256": parent_snapshot_value["source_sha256"],
            "inputs": inventory, "fixed_weight": FIXED_WEIGHT,
            "reference_route": "independent_v7_ranking",
            "uploaded": False}


def seal_ranking(args: argparse.Namespace) -> dict:
    root, out, parent, files, snapshot = real_context(args)
    parent_args(files, out)
    ensure_protocol(root, out, parent, snapshot)
    require_final(root, out, files, snapshot)
    seal = ranking_snapshot(out, files, snapshot)
    path = out / "ranking_inputs.json"
    if path.exists():
        if read_json(path) != seal:
            raise ValueError("Frozen ranking inputs changed")
    else:
        write_new_json(path, seal, root)
    assert_snapshot(root, root / "parents/v8_parents.json", snapshot)
    return {"ranking_inputs_sha256": sha256(path),
            "template_sha256": snapshot["file_sha256"]["submission_template"],
            "reference_sha256": snapshot["file_sha256"]["v7_ranking_predictions"]}


def require_ranking_seal(out: Path, files: dict[str, Path], snapshot: dict) -> dict:
    path = out / "ranking_inputs.json"
    saved = read_json(path)
    fresh = ranking_snapshot(out, files, snapshot)
    if saved != fresh:
        raise ValueError("Ranking source/model/reference bytes differ from pre-read seal")
    return saved


def predict(args: argparse.Namespace) -> dict:
    root, out, parent, files, snapshot = real_context(args, heavy=True)
    p = parent_args(files, out)
    ensure_protocol(root, out, parent, snapshot)
    final = require_final(root, out, files, snapshot)
    ranking_seal = require_ranking_seal(out, files, snapshot)
    output_path = out / "predictions.parquet"
    expert_path = out / "ranking_expert.parquet"
    report_path = out / "ranking_manifest.json"
    if any(path.exists() for path in (output_path, expert_path, report_path)):
        raise FileExistsError("Replica ranking outputs already exist")
    manifest = read_json(out / "features_manifest.json")
    if (manifest.get("weather_file_sha256") != sha256(p.weather_file)
            or manifest.get("features") is None):
        raise ValueError("Replica ranking feature source differs from training")
    features, rows = movement.build_ranking_features(p, manifest)
    template = pd.read_parquet(files["submission_template"])
    reference = pd.read_parquet(files["v7_ranking_predictions"])
    exact_columns = ["MVT_ID_mvt", "TAXITIME_SEC_mvt"]
    ids = template.MVT_ID_mvt.to_numpy()
    if (list(template) != exact_columns or list(reference) != exact_columns
            or len(ids) != EXPECTED_RANKING_ROWS or len(rows) != len(ids)
            or template.MVT_ID_mvt.isna().any() or template.MVT_ID_mvt.duplicated().any()
            or not np.array_equal(rows.MVT_ID_mvt.to_numpy(), ids)
            or not np.array_equal(reference.MVT_ID_mvt.to_numpy(), ids)):
        raise ValueError("Replica ranking/template/reference ID order or schema differs")
    old = reference.TAXITIME_SEC_mvt.to_numpy(dtype=np.float64, copy=True)
    if not np.isfinite(old).all() or np.any(old < 0):
        raise ValueError("Independent v7 ranking parent is not finite/nonnegative")
    proxy = rows.proxy.to_numpy(dtype=float)
    valid = np.isfinite(proxy) & (proxy >= 0) & (proxy <= 7200)
    gate = movement.read_gate(p.cache_dir, rows, ranking=True)
    if (valid.sum() != 339_377 or gate.sum() != 4_907
            or np.any(valid & gate)):
        raise ValueError("Replica published valid/missing ranking gate coverage differs")
    import lightgbm as lgb
    model = lgb.Booster(model_file=str(out / "final_movement_only.txt"))
    if (model.current_iteration() != final["final_rounds"]
            or list(model.feature_name()) != manifest["features"]):
        raise ValueError("Readback final model rounds or 79 feature names differ")
    raw = model.predict(features.loc[gate], num_iteration=final["final_rounds"],
                        num_threads=3)
    if len(raw) != int(gate.sum()) or not np.isfinite(raw).all():
        raise ValueError("Replica ranking expert lacks finite gate coverage")
    expert = np.maximum(raw, 0)
    updated = old.copy()
    updated[gate] = np.maximum((1 - FIXED_WEIGHT) * old[gate]
                               + FIXED_WEIGHT * expert, 0)
    if (not np.array_equal(updated[~gate], old[~gate])
            or not np.array_equal(updated[valid], old[valid])
            or not np.isfinite(updated).all() or np.any(updated < 0)):
        raise ValueError("Replica ranking altered non-gate predictions")
    full_expert = np.full(len(rows), np.nan, dtype=np.float32)
    full_expert[gate] = expert.astype(np.float32)
    assert_snapshot(root, root / "parents/v8_parents.json", snapshot)
    require_ranking_seal(out, files, snapshot)
    write_new_parquet(expert_path, pd.DataFrame({"MVT_ID_mvt": ids, "gate": gate,
                                                 "movement_expert": full_expert}), root)
    write_new_parquet(output_path, pd.DataFrame({"MVT_ID_mvt": ids,
                                                 "TAXITIME_SEC_mvt": updated}), root)
    readback = pd.read_parquet(output_path)
    if (not np.array_equal(readback.MVT_ID_mvt.to_numpy(), ids)
            or not np.array_equal(readback.TAXITIME_SEC_mvt.to_numpy(dtype=float),
                                  updated)):
        raise ValueError("Replica ranking readback differs from template")
    assert_snapshot(root, root / "parents/v8_parents.json", snapshot)
    require_ranking_seal(out, files, snapshot)
    report = {"ranking_rows": len(rows), "valid_gate_rows": int(valid.sum()),
              "missing_gate_rows": int(gate.sum()),
              "changed_rows": int(np.count_nonzero(updated != old)),
              "selected_weight": FIXED_WEIGHT,
              "independent_v7_reference_sha256": snapshot["file_sha256"]["v7_ranking_predictions"],
              "ranking_inputs_sha256": sha256(out / "ranking_inputs.json"),
              "model_sha256": final["model_sha256"],
              "expert_sha256": sha256(expert_path),
              "prediction_sha256": sha256(output_path),
              "prediction_bytes": output_path.stat().st_size,
              "template_order_verified": True,
              "non_gate_unchanged": True, "finite_nonnegative": True,
              "parent_manifest_sha256": snapshot["manifest_sha256"],
              "source_snapshot_sha256": value_sha256(snapshot["source_sha256"]),
              "uploaded": False, "byte_identical_to_original_run_claimed": False}
    write_new_json(report_path, report, root)
    return {"rows": len(rows), "missing_gate_rows": int(gate.sum()),
            "changed_rows": report["changed_rows"], "sha256": report["prediction_sha256"],
            "uploaded": False}


def synthetic_self_test() -> dict:
    """No filesystem parent, raw value, cache, model, or leaderboard access."""
    fake_sha = lambda label: hashlib.sha256(label.encode("utf-8")).hexdigest()
    files = {role: {"path": f"parents/synthetic/{role}.fixture",
                    "sha256": fake_sha(role)} for role in PARENT_ROLES}
    files["v5_model_synthetic"] = {"path": "parents/synthetic/v5_model.fixture",
                                    "sha256": fake_sha("v5_model_synthetic")}
    files["v7_model_synthetic"] = {"path": "parents/synthetic/v7_model.fixture",
                                    "sha256": fake_sha("v7_model_synthetic")}
    fold_map = {name: list(months) for name, months in FOLDS.items()}
    v5_outputs = {role: files[role]["sha256"] for role in files
                  if role in ("v5_validation_oof", "v5_protocol", "v5_validation_report")
                  or role.startswith("v5_model_")}
    v7_outputs = {role: files[role]["sha256"] for role in files
                  if role in ("v7_validation_oof", "v7_protocol", "v7_validation_report",
                              "v7_ranking_predictions", "v7_ranking_manifest")
                  or role.startswith("v7_model_")}
    input_hashes = {role: files[role]["sha256"] for role in COMMON_PARENT_INPUT_ROLES}
    parents = {
        "v5": {"name": "v5", "status": "complete", "source_commit": "a" * 40,
               "producer_receipt_sha256": files["v5_producer_receipt"]["sha256"],
               "ordered_ids_sha256": fake_sha("v5_ids"),
               "feature_schema_sha256": fake_sha("v5_schema"),
               "source_sha256": {"producer": fake_sha("v5_source")},
               "heldout_folds": fold_map, "fit_and_early_exclude_heldout": True,
               "published_params_and_weights": {
                   "selected_stage": "deep_missing_clean_arrival",
                   "valid_deep_weight": 0.5, "missing_direct_weight": 0.5,
                   "clean_arrival_weight": 0.25},
               "output_sha256": v5_outputs, "input_sha256": input_hashes},
        "v7": {"name": "v7", "status": "complete", "source_commit": "b" * 40,
               "producer_receipt_sha256": files["v7_producer_receipt"]["sha256"],
               "ordered_ids_sha256": fake_sha("v7_ids"),
               "feature_schema_sha256": fake_sha("v7_schema"),
               "source_sha256": {"producer": fake_sha("v7_source")},
               "heldout_folds": fold_map, "fit_and_early_exclude_heldout": True,
               "published_params_and_weights": {
                   "selected_weight": 0.5, "depth": 10,
                   "max_iterations": 10000, "seed": 2026},
               "output_sha256": v7_outputs, "input_sha256": input_hashes}}
    fixture = {"schema_version": 1, "status": "complete", "files": files,
               "parents": parents,
               "lineage": {"v7_parent_v5_receipt_sha256": files["v5_producer_receipt"]["sha256"],
                           "v7_parent_v5_oof_sha256": files["v5_validation_oof"]["sha256"],
                           "v8_parent_v7_oof_sha256": files["v7_validation_oof"]["sha256"],
                           "v8_parent_v7_ranking_sha256": files["v7_ranking_predictions"]["sha256"]}}
    validate_parent_metadata(fixture)
    poisoned = json.loads(json.dumps(fixture))
    poisoned["lineage"]["v7_parent_v5_oof_sha256"] = fake_sha("wrong")
    try:
        validate_parent_metadata(poisoned)
    except ValueError:
        pass
    else:
        raise AssertionError("Tampered parent lineage passed metadata validation")
    poisoned = json.loads(json.dumps(fixture))
    poisoned["parents"]["v7"]["published_params_and_weights"]["selected_weight"] = 1.0
    try:
        validate_parent_metadata(poisoned)
    except ValueError:
        pass
    else:
        raise AssertionError("Changed parent policy weight passed")
    sample = pd.DataFrame({"MVT_ID_mvt": ["one", "two", "three", "four"],
                           "target": [100., 200., 300., 400.],
                           "month": [1, 7, 2, 11],
                           "time": pd.to_datetime(["2025-01-01T12:00:00Z",
                                                   "2025-07-02T12:00:00Z",
                                                   "2025-02-03T12:00:00Z",
                                                   "2025-11-04T12:00:00Z"], utc=True)})
    split = ordinary_masks(sample, (1, 7))
    if (not np.array_equal(split["heldout"], [True, True, False, False])
            or np.any(split["fit"] & split["heldout"])
            or np.any(split["early"] & split["heldout"])):
        raise AssertionError("Complement-month fit/early split leaked heldout rows")
    old = np.array([100., 200., 300.])
    expert = np.array([90., 230., -2.])
    gate = np.array([False, True, True])
    selected = old.copy()
    selected[gate] = np.maximum((1 - FIXED_WEIGHT) * old[gate]
                                + FIXED_WEIGHT * np.maximum(expert[gate], 0), 0)
    if not np.array_equal(selected, [100., 230., 0.]):
        raise AssertionError("Fixed weight-one, gate-only clipped policy differs")
    if ordered_sha(["a", "b"]) == ordered_sha(["b", "a"]):
        raise AssertionError("Ordered ID seal did not detect order change")
    return {"synthetic_parent_manifest": "passed", "tampered_lineage": "refused",
            "tampered_parent_weight": "refused", "heldout_split": "passed",
            "fixed_gate_formula": "passed", "ordered_id_hash": "passed",
            "real_raw_cache_or_model_values_read": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("plan", "self-test", "verify-parents",
                                           "prepare", "fit-fold", "evaluate",
                                           "fresh-audit", "reserved-audit", "compose-oof",
                                           "fit-final", "ranking-seal", "predict"),
                        default="plan")
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--published-source-sha256")
    parser.add_argument("--fold", choices=tuple(FOLDS))
    args = parser.parse_args()
    if args.mode == "plan":
        result = plan()
    elif args.mode == "self-test":
        result = synthetic_self_test()
    else:
        if args.run_root is None:
            parser.error("real modes require an isolated --run-root")
        if args.mode == "verify-parents":
            root, _, manifest, files, snapshot = real_context(args)
            result = {"status": "verified_parent_bytes_only",
                      "run_root": str(root), "parent_manifest_sha256": snapshot["manifest_sha256"],
                      "parent_roles": len(files),
                      "v5_source_commit": manifest["parents"]["v5"]["source_commit"],
                      "v7_source_commit": manifest["parents"]["v7"]["source_commit"],
                      "parquet_values_read": False}
        elif args.mode == "prepare":
            result = prepare(args)
        elif args.mode == "fit-fold":
            if args.fold is None:
                parser.error("fit-fold requires --fold")
            result = fit_fold(args)
        elif args.mode == "evaluate":
            result = evaluate(args)
        elif args.mode == "fresh-audit":
            result = audit(args, "april_october")
        elif args.mode == "reserved-audit":
            result = audit(args, "february_august")
        elif args.mode == "compose-oof":
            result = compose_oof(args)
        elif args.mode == "fit-final":
            result = fit_final(args)
        elif args.mode == "ranking-seal":
            result = seal_ranking(args)
        elif args.mode == "predict":
            result = predict(args)
        else:
            raise AssertionError("Unreachable mode")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
