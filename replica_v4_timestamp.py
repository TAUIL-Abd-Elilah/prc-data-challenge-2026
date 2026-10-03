"""Isolated, receipt-bound replica of the v4 reference and v5 timestamp expert.

Only plan and self-test are permitted until this new source and its spec have
been reviewed and published. Real modes require independent clean v3/GPU/source
parents under a disjoint run root and an operator publication attestation.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SOURCE_ROOT = Path(__file__).resolve().parent
SPEC = SOURCE_ROOT / "reports/clean_replication_timestamp_spec.json"
SOURCE_SHA256 = {
    "deep_timestamp_expert.py": "8edf1eb0b00b065c2d6b5fbb1d7b85b013ba861b12c55a574d53dc8ead14ad82",
    "v4_reference.py": "5bf163d64863cc695f0a3fdfab379e2d76b6c056332622f9ffa9d9ee5eeddaa0",
    "catboost_expert.py": "1e1ecfec5acfe41524f028339984ae56fdb1b8c6e360fde704acdd64258345f2",
    "solution.py": "13848cd8483737c1e5db0ac4e1e90c0b64eead8ce289cc17870944f58715aa11",
    "airport_models.py": "68f1e7b6f45ef0cfec11049760f5d87e7609ac9c9cd4fad3bd5c28b5e12563f0",
    "weather_model.py": "6dbfe4f41550b97da604e698c00c3fbbafc0ef6489ea6b47e4099e5658b7c1ec",
}
FOLDS = {"seasonal_jan_jul": (1, 7), "forward_nov_dec": (11, 12)}
WEIGHTS = (0.0, 0.1, 0.25, 0.5, 1.0)
FIXED_WEIGHT = 0.5
MIN_FREE_GIB = 10.0
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
HEX40 = re.compile(r"[0-9a-f]{40}\Z")


def month_path(month: int) -> str:
    year_end, month_end = (2026, 1) if month == 12 else (2025, month + 1)
    return (f"data/training_2025-{month:02d}-01_"
            f"{year_end}-{month_end:02d}-01.parquet")


CANONICAL = {
    **{f"raw_training_{m:02d}": month_path(m) for m in range(1, 13)},
    "raw_ranking": "data/ranking.parquet",
    "submission_template": "data/submitting.parquet",
    "baseline_training_rows": "artifacts/baseline/training_rows.parquet",
    "baseline_training_features": "artifacts/baseline/features.parquet",
    "baseline_ranking_rows": "artifacts/baseline/ranking_rows.parquet",
    "baseline_ranking_features": "artifacts/baseline/ranking_features.parquet",
    "noaa_weather": "data/external/weather.parquet",
    "v3_validation_oof": "artifacts/lobt_ensemble/validation_predictions.parquet",
    "v3_ranking_predictions": "artifacts/lobt_ensemble/predictions.parquet",
    "v3_validation_report": "artifacts/lobt_ensemble/validation.json",
    "v3_producer_receipt": "parents/v3_producer_receipt.json",
    "gpu_seasonal_oof": "artifacts/catboost/gpu/seasonal_jan_jul_oof.parquet",
    "gpu_forward_oof": "artifacts/catboost/gpu/forward_nov_dec_oof.parquet",
    "gpu_validation_report": "artifacts/catboost/gpu/validation.json",
    "gpu_ranking_expert": "artifacts/catboost/gpu/ranking_expert.parquet",
    "gpu_ranking_predictions": "artifacts/catboost/gpu/predictions.parquet",
    "gpu_producer_receipt": "parents/gpu_producer_receipt.json",
    "source_seasonal_oof": "artifacts/catboost/source/seasonal_jan_jul_oof.parquet",
    "source_forward_oof": "artifacts/catboost/source/forward_nov_dec_oof.parquet",
    "source_validation_report": "artifacts/catboost/source/validation_sequential_gpu.json",
    "source_ranking_probabilities": "artifacts/catboost/source/ranking_source_probabilities.parquet",
    "source_sequential_predictions": "artifacts/catboost/source/sequential_predictions.parquet",
    "source_producer_receipt": "parents/source_producer_receipt.json",
}
DATA_ROLES = tuple(k for k in CANONICAL if k.startswith("raw_training_")) + (
    "raw_ranking", "submission_template", "baseline_training_rows",
    "baseline_training_features", "baseline_ranking_rows",
    "baseline_ranking_features", "noaa_weather")
PRODUCERS = {
    "v3": {"receipt": "v3_producer_receipt", "prefix": "v3_model_",
           "outputs": ("v3_validation_oof", "v3_ranking_predictions", "v3_validation_report")},
    "gpu": {"receipt": "gpu_producer_receipt", "prefix": "gpu_model_",
            "outputs": ("gpu_seasonal_oof", "gpu_forward_oof", "gpu_validation_report",
                        "gpu_ranking_expert", "gpu_ranking_predictions")},
    "source": {"receipt": "source_producer_receipt", "prefix": "source_model_",
               "outputs": ("source_seasonal_oof", "source_forward_oof",
                           "source_validation_report", "source_ranking_probabilities",
                           "source_sequential_predictions")},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def hash_value(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def float_hash(values: Any) -> str:
    array = np.asarray(values, dtype="<f8")
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError("Probe predictions must be finite 1D float64")
    return hashlib.sha256(array.tobytes()).hexdigest()


def id_hash(values: Any) -> str:
    array = np.asarray(values, dtype=np.int64)
    if array.ndim != 1:
        raise ValueError("IDs must be one-dimensional")
    return hashlib.sha256(array.astype("<i8", copy=False).tobytes()).hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def require_sha(value: Any, role: str, *, commit: bool = False) -> None:
    regex = HEX40 if commit else HEX64
    if not isinstance(value, str) or not regex.fullmatch(value):
        raise ValueError(f"Invalid SHA for {role}")


def strict_root(run_root: Path) -> Path:
    root = Path(run_root).resolve(strict=True)
    source = SOURCE_ROOT.resolve(strict=True)
    if root == source or root in source.parents or source in root.parents:
        raise ValueError("Replica run root must be disjoint from source repository")
    return root


def strict_child(root: Path, relative: str, *, existing: bool = True) -> Path:
    candidate = Path(relative)
    if not relative or candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError("Only run-root-relative child paths are allowed")
    path = (root / candidate).resolve(strict=existing)
    if root not in path.parents:
        raise ValueError("Private path escapes isolated run root")
    if existing and not path.is_file():
        raise FileNotFoundError(path)
    return path


def source_snapshot() -> dict[str, str]:
    actual = {name: sha256(SOURCE_ROOT / name) for name in SOURCE_SHA256}
    if actual != SOURCE_SHA256:
        raise ValueError("A published original scientific helper changed")
    actual["reports/clean_replication_timestamp_spec.json"] = sha256(SPEC)
    actual["replica_v4_timestamp.py"] = sha256(Path(__file__).resolve())
    return actual


def publication_check(expected: str | None) -> dict[str, str]:
    require_sha(expected, "operator-published adapter SHA")
    snapshot = source_snapshot()
    if snapshot["replica_v4_timestamp.py"] != expected:
        raise ValueError("Adapter source differs from operator-published SHA")
    spec = read_json(SPEC)
    if (spec.get("schema_version") != 1
            or spec.get("status") != "prospective_source_only_no_real_run_authorized"):
        raise ValueError("Replica spec identity changed")
    return snapshot


def validate_manifest_metadata(manifest: dict) -> None:
    if manifest.get("schema_version") != 1 or manifest.get("status") != "complete":
        raise ValueError("Independent parent manifest not complete")
    files = manifest.get("files")
    if not isinstance(files, dict) or not set(CANONICAL).issubset(files):
        raise ValueError("Incomplete clean parent file inventory")
    for role, path in CANONICAL.items():
        item = files[role]
        if not isinstance(item, dict) or item.get("path") != path:
            raise ValueError(f"Parent role {role} must use canonical isolated path")
        require_sha(item.get("sha256"), role)
    for role, item in files.items():
        if role in CANONICAL:
            continue
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise ValueError("Invalid additional parent model file")
        if not any(role.startswith(p["prefix"]) for p in PRODUCERS.values()):
            raise ValueError("Unrecognized additional parent file role")
        require_sha(item.get("sha256"), role)
    if len({item["path"] for item in files.values()}) != len(files):
        raise ValueError("Two parent roles alias one file")
    producers = manifest.get("producers")
    if not isinstance(producers, dict) or set(producers) != set(PRODUCERS):
        raise ValueError("Named v3, GPU and source producer receipts required")
    if manifest.get("heldout_folds") != {k: list(v) for k, v in FOLDS.items()}:
        raise ValueError("Parent heldout months differ from published folds")
    for name, policy in PRODUCERS.items():
        producer = producers[name]
        if not isinstance(producer, dict) or producer.get("name") != name or producer.get("status") != "complete":
            raise ValueError(f"Invalid {name} producer declaration")
        require_sha(producer.get("source_commit"), f"{name} source commit", commit=True)
        require_sha(producer.get("receipt_sha256"), f"{name} receipt")
        if producer["receipt_sha256"] != files[policy["receipt"]]["sha256"]:
            raise ValueError(f"Detached {name} producer receipt")
        sources = producer.get("source_sha256")
        if not isinstance(sources, dict) or not sources:
            raise ValueError(f"Missing {name} producer source inventory")
        for path, digest in sources.items():
            if not isinstance(path, str) or not path.endswith(".py"):
                raise ValueError("Invalid producer source role")
            require_sha(digest, path)
        models = [role for role in files if role.startswith(policy["prefix"])]
        if not models:
            raise ValueError(f"No independently rebuilt {name} model bytes")
        outputs = producer.get("output_sha256")
        required_outputs = set(policy["outputs"]) | set(models)
        if not isinstance(outputs, dict) or not required_outputs.issubset(outputs):
            raise ValueError(f"Incomplete {name} model/report/OOF outputs")
        if any(outputs[role] != files[role]["sha256"] for role in required_outputs):
            raise ValueError(f"Detached {name} output from source receipt")
        required_inputs = set(DATA_ROLES)
        if name != "v3":
            required_inputs |= set(PRODUCERS["v3"]["outputs"]) | {"v3_producer_receipt"}
        if name == "source":
            required_inputs |= set(PRODUCERS["gpu"]["outputs"]) | {"gpu_producer_receipt"}
        inputs = producer.get("input_sha256")
        if not isinstance(inputs, dict) or not required_inputs.issubset(inputs):
            raise ValueError(f"Incomplete {name} raw/cache/upstream input lineage")
        if any(inputs[role] != files[role]["sha256"] for role in required_inputs):
            raise ValueError(f"Detached {name} producer input")
        if producer.get("heldout_folds") != {k: list(v) for k, v in FOLDS.items()}:
            raise ValueError(f"{name} producer heldout months differ")
        if producer.get("fit_and_early_exclude_heldout") is not True:
            raise ValueError(f"{name} producer did not prove heldout separation")
        choices = producer.get("published_choices")
        if not isinstance(choices, dict):
            raise ValueError(f"{name} scientific choices missing")
        fixed = ({"route": "lobt_ensemble"} if name == "v3" else
                 {"gpu_weight": 0.25} if name == "gpu" else
                 {"source_scale": 0.5, "gpu_weight": 0.25})
        if any(choices.get(key) != value for key, value in fixed.items()):
            raise ValueError(f"{name} producer deviates from published parent policy")
        require_sha(producer.get("ordered_validation_ids_sha256"), f"{name} OOF IDs")
        require_sha(producer.get("ordered_ranking_ids_sha256"), f"{name} ranking IDs")
        require_sha(producer.get("feature_schema_sha256"), f"{name} feature schema")
        if producer.get("independent_model_and_policy_replay_passed") is not True:
            raise ValueError(f"{name} parent model and fixed policy replay not certified")


def verify_parent_receipts(manifest: dict, paths: dict[str, Path]) -> None:
    files = manifest["files"]
    for name, policy in PRODUCERS.items():
        receipt = read_json(paths[policy["receipt"]])
        declared = manifest["producers"][name]
        # The manifest records the receipt's SHA externally. The receipt
        # cannot contain its own SHA without an impossible hash fixed point.
        if receipt != {key: value for key, value in declared.items()
                       if key != "receipt_sha256"}:
            raise ValueError(f"{name} producer receipt and manifest declaration differ")
        # The receipt is itself hash-bound in the manifest; these direct checks
        # ensure every named source/cache/OOF/model input is from this run.
        for role, digest in receipt["input_sha256"].items():
            if role not in files or digest != files[role]["sha256"]:
                raise ValueError(f"{name} receipt input {role} is detached")
        for role, digest in receipt["output_sha256"].items():
            if role not in files or digest != files[role]["sha256"]:
                raise ValueError(f"{name} receipt output {role} is detached")


def parent_snapshot(root: Path, sources: dict[str, str]) -> tuple[dict, dict[str, Path], dict]:
    manifest_path = strict_child(root, "parents/v4_timestamp_parents.json")
    manifest = read_json(manifest_path)
    validate_manifest_metadata(manifest)
    paths = {role: strict_child(root, item["path"])
             for role, item in manifest["files"].items()}
    discovered = set((root / "data").glob("training_2025-*.parquet"))
    expected_raw = {paths[f"raw_training_{month:02d}"] for month in range(1, 13)}
    if discovered != expected_raw:
        raise ValueError("Original feature loader would read unsealed or missing training months")
    digests = {role: sha256(path) for role, path in paths.items()}
    if any(digests[role] != item["sha256"] for role, item in manifest["files"].items()):
        raise ValueError("Clean parent bytes differ from sealed manifest")
    verify_parent_receipts(manifest, paths)
    snapshot = {"manifest_sha256": sha256(manifest_path),
                "parent_file_sha256": digests, "source_sha256": sources}
    return manifest, paths, snapshot


def assert_snapshot(root: Path, expected: dict) -> None:
    sources = source_snapshot()
    _, _, current = parent_snapshot(root, sources)
    if current != expected:
        raise ValueError("Source, spec or isolated parent changed during operation")


def require_memory() -> None:
    import psutil
    free_gib = psutil.virtual_memory().available / (1024 ** 3)
    if free_gib < MIN_FREE_GIB:
        raise MemoryError(f"At least {MIN_FREE_GIB:g} GiB available RAM required; {free_gib:.2f} GiB")


def real_context(args: argparse.Namespace) -> tuple[Path, Path, dict, dict[str, Path], dict]:
    sources = publication_check(args.published_source_sha256)
    root = strict_root(args.run_root)
    require_memory()
    manifest, paths, snapshot = parent_snapshot(root, sources)
    output = (root / "artifacts/replica-v4-timestamp").resolve(strict=False)
    if root not in output.parents:
        raise ValueError("Output directory escapes isolated run root")
    return root, output, manifest, paths, snapshot


def exclusive_bytes(path: Path, data: bytes, root: Path) -> None:
    path = path.resolve(strict=False)
    if root not in path.parents or path.exists():
        raise FileExistsError("Replica output is outside run root or already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".replica-", dir=path.parent)
    temp = Path(name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temp, path)  # exclusive on Windows and POSIX
    finally:
        temp.unlink(missing_ok=True)


def exclusive_json(path: Path, value: dict, root: Path) -> None:
    exclusive_bytes(path, json.dumps(value, indent=2, allow_nan=False).encode("utf-8"), root)


def exclusive_parquet(path: Path, value: pd.DataFrame, root: Path) -> None:
    path = path.resolve(strict=False)
    if root not in path.parents or path.exists():
        raise FileExistsError("Replica output is outside run root or already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".replica-", suffix=".parquet", dir=path.parent)
    os.close(fd)
    temp = Path(name)
    try:
        value.to_parquet(temp, index=False)
        os.link(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def exclusive_model(path: Path, model: Any, root: Path) -> None:
    path = path.resolve(strict=False)
    if root not in path.parents or path.exists():
        raise FileExistsError("Replica model is outside run root or already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".replica-", suffix=".cbm", dir=path.parent)
    os.close(fd)
    temp = Path(name)
    try:
        model.save_model(str(temp))
        os.link(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def protocol(root: Path, out: Path, snapshot: dict) -> dict:
    value = {"schema_version": 1, "replica_of": "v4_reference_plus_depth9_timestamp",
             "source_sha256": snapshot["source_sha256"],
             "parent_manifest_sha256": snapshot["manifest_sha256"],
             "parent_file_sha256": snapshot["parent_file_sha256"],
             "folds": {k: list(v) for k, v in FOLDS.items()},
             "fixed_weight": FIXED_WEIGHT, "requested_iterations": 4500,
             "depth": 9, "threads": 2, "minimum_free_gib": MIN_FREE_GIB,
             "historical_v4_rmse_byte_equality_required": False}
    path = out / "protocol.json"
    if path.exists():
        if read_json(path) != value:
            raise ValueError("Immutable replica protocol changed")
    else:
        exclusive_json(path, value, root)
    return value


def adapter_args(paths: dict[str, Path], output: Path) -> argparse.Namespace:
    return argparse.Namespace(data_dir=paths["raw_ranking"].parent,
                              cache_dir=paths["baseline_training_rows"].parent,
                              weather_file=paths["noaa_weather"],
                              output_dir=output, iterations=4500, depth=9,
                              threads=2)


def schema_of(features: pd.DataFrame) -> dict:
    columns = [{"name": str(name), "dtype": str(features[name].dtype)}
               for name in features.columns]
    categories = {}
    for name in features.select_dtypes(include="category"):
        cat = features[name].cat
        categories[name] = hash_value({"values": [str(v) for v in cat.categories],
                                       "ordered": bool(cat.ordered)})
    return {"columns": columns, "category_vocabulary_sha256": categories,
            "categorical_columns": list(categories)}


def row_metadata(rows: pd.DataFrame) -> None:
    required = {"MVT_ID_mvt", "target", "proxy", "month", "airport"}
    if not required.issubset(rows) or rows.MVT_ID_mvt.isna().any() or rows.MVT_ID_mvt.duplicated().any():
        raise ValueError("Invalid independent baseline row metadata")
    if len(rows) != 2085047:
        raise ValueError("2025 departure cache row coverage differs")
    if not rows.month.between(1, 12).all():
        raise ValueError("Training month metadata invalid")


def validate_reference(frame: pd.DataFrame, rows: pd.DataFrame) -> dict:
    row_metadata(rows)
    required = {"MVT_ID_mvt", "target", "fold", "a_valid", "selected",
                "selected_unclipped", "month", "MVT_TIME_UTC_mvt"}
    if not required.issubset(frame) or len(frame) != 672428:
        raise ValueError("v4 all-finite reference coverage/schema differs")
    if frame.MVT_ID_mvt.isna().any() or frame.MVT_ID_mvt.duplicated().any():
        raise ValueError("v4 OOF ID duplication/null")
    held = rows.month.isin((1, 7, 11, 12)) & np.isfinite(rows.target.to_numpy(dtype=float))
    wanted = rows.loc[held, ["MVT_ID_mvt", "target", "proxy", "month"]].copy()
    time_col = "MVT_TIME_UTC_mvt" if "MVT_TIME_UTC_mvt" in rows else "time"
    wanted["MVT_TIME_UTC_mvt"] = rows.loc[held, time_col].to_numpy()
    if len(wanted) != len(frame) or set(wanted.MVT_ID_mvt) != set(frame.MVT_ID_mvt):
        raise ValueError("v4 OOF does not cover exact all-finite heldout IDs")
    aligned = frame.merge(wanted, on="MVT_ID_mvt", how="left", validate="one_to_one",
                          suffixes=("", "_base"))
    if not np.array_equal(aligned.target.to_numpy(dtype=float), aligned.target_base.to_numpy(dtype=float)):
        raise ValueError("v4 OOF labels differ from independently rebuilt cache")
    if not np.array_equal(aligned.month.to_numpy(), aligned.month_base.to_numpy()):
        raise ValueError("v4 OOF month metadata differs")
    t1 = pd.to_datetime(aligned.MVT_TIME_UTC_mvt, utc=True, errors="coerce")
    t2 = pd.to_datetime(aligned.MVT_TIME_UTC_mvt_base, utc=True, errors="coerce")
    if t1.isna().any() or not t1.equals(t2):
        raise ValueError("v4 OOF UTC movement times differ")
    expected_fold = np.where(aligned.month.isin(FOLDS["seasonal_jan_jul"]),
                             "seasonal_jan_jul", "forward_nov_dec")
    if not np.array_equal(aligned.fold.to_numpy(), expected_fold):
        raise ValueError("v4 OOF fold metadata differs")
    proxy = aligned.proxy.to_numpy(dtype=float)
    valid = np.isfinite(proxy) & (proxy >= 0) & (proxy <= 7200)
    if int(valid.sum()) != 663664:
        raise ValueError("v4 valid-AOBT all-finite gate coverage differs")
    if not np.array_equal(aligned.a_valid.to_numpy(dtype=bool), valid):
        raise ValueError("v4 own-AOBT gate differs")
    pred = aligned.selected.to_numpy(dtype=float)
    raw = aligned.selected_unclipped.to_numpy(dtype=float)
    if (not np.isfinite(pred).all() or not np.isfinite(raw).all()
            or not np.array_equal(pred, np.maximum(raw, 0))):
        raise ValueError("v4 clipped output policy differs")
    return {"rows": len(frame), "ordered_ids_sha256": id_hash(frame.MVT_ID_mvt),
            "all_finite_rmse": float(np.sqrt(np.mean((aligned.target - pred) ** 2))),
            "valid_proxy_rows": int(valid.sum()), "invalid_proxy_rows": int((~valid).sum())}


def verify_parent_validation_ids(paths: dict[str, Path], manifest: dict) -> None:
    groups = {
        "v3": [paths["v3_validation_oof"]],
        "gpu": [paths["gpu_seasonal_oof"], paths["gpu_forward_oof"]],
        "source": [paths["source_seasonal_oof"], paths["source_forward_oof"]],
    }
    for name, files in groups.items():
        ids = pd.concat([pd.read_parquet(path, columns=["MVT_ID_mvt"])["MVT_ID_mvt"]
                         for path in files], ignore_index=True)
        if ids.isna().any() or ids.duplicated().any():
            raise ValueError(f"{name} parent OOF has null or duplicate IDs")
        if id_hash(ids) != manifest["producers"][name]["ordered_validation_ids_sha256"]:
            raise ValueError(f"{name} parent OOF ordered ID receipt differs")


def write_reference(args: argparse.Namespace) -> dict:
    root, out, manifest, paths, snapshot = real_context(args)
    protocol(root, out, snapshot)
    frame_path = out / "v4/validation_predictions.parquet"
    report_path = out / "v4/reference.json"
    if frame_path.exists() or report_path.exists():
        raise FileExistsError("Reference already exists; no overwrite or resume")
    from v4_reference import load_v4
    verify_parent_validation_ids(paths, manifest)
    rows = pd.read_parquet(paths["baseline_training_rows"])
    frame = load_v4(root=root, verify_snapshot=False)
    metrics = validate_reference(frame, rows)
    assert_snapshot(root, snapshot)
    exclusive_parquet(frame_path, frame, root)
    readback = pd.read_parquet(frame_path)
    if not readback.equals(frame):
        raise ValueError("v4 Parquet readback differs")
    report = {"status": "complete", "source_sha256": snapshot["source_sha256"],
              "parent_manifest_sha256": snapshot["manifest_sha256"],
              "parent_file_sha256": snapshot["parent_file_sha256"],
              "reference_sha256": sha256(frame_path), **metrics}
    assert_snapshot(root, snapshot)
    exclusive_json(report_path, report, root)
    return report


def assert_reference_replay(saved: pd.DataFrame, rebuilt: pd.DataFrame) -> None:
    if list(saved.columns) != list(rebuilt.columns) or not saved.equals(rebuilt):
        raise ValueError("Saved v4 reference does not replay from sealed v3/GPU/source OOF parents")


def require_reference(root: Path, out: Path, paths: dict[str, Path],
                      snapshot: dict) -> tuple[pd.DataFrame, dict]:
    frame_path = out / "v4/validation_predictions.parquet"
    report = read_json(out / "v4/reference.json")
    if (report.get("status") != "complete" or report.get("source_sha256") != snapshot["source_sha256"]
            or report.get("parent_manifest_sha256") != snapshot["manifest_sha256"]
            or report.get("parent_file_sha256") != snapshot["parent_file_sha256"]
            or report.get("reference_sha256") != sha256(frame_path)):
        raise ValueError("Replica v4 reference seal differs")
    frame = pd.read_parquet(frame_path)
    from v4_reference import load_v4
    rebuilt = load_v4(root=root, verify_snapshot=False)
    assert_reference_replay(frame, rebuilt)
    rows = pd.read_parquet(paths["baseline_training_rows"])
    derived = validate_reference(frame, rows)
    if any(report.get(key) != value for key, value in derived.items()):
        raise ValueError("Replica v4 reference receipt values differ")
    return frame, report


def split_indices(rows: pd.DataFrame, fold: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = rows.target.to_numpy(dtype=float)
    proxy = rows.proxy.to_numpy(dtype=float)
    valid = np.isfinite(proxy) & (proxy >= 0) & (proxy <= 7200)
    held = rows.month.isin(FOLDS[fold]).to_numpy()
    train = np.flatnonzero(~held & valid & np.isfinite(y) & (y >= 0) & (y <= 86400))
    test = np.flatnonzero(held & valid & np.isfinite(y))
    order = np.random.default_rng(2026).permutation(train)
    n_early = max(20000, int(.06 * len(order)))
    return order[n_early:], order[:n_early], test


def score_fold(reference: pd.DataFrame, fold: str, expert: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    full = reference.loc[reference.fold.eq(fold)].merge(expert, on="MVT_ID_mvt",
                                                        how="left", validate="one_to_one")
    base = full.selected.to_numpy(dtype=float)
    alternate = full.expert.fillna(full.selected).to_numpy(dtype=float)
    y = full.target.to_numpy(dtype=float)
    scores = {str(weight): float(np.sqrt(np.mean((y - np.maximum(
        base + weight * (alternate - base), 0)) ** 2))) for weight in WEIGHTS}
    full["timestamp_selected"] = np.maximum(base + FIXED_WEIGHT * (alternate - base), 0)
    if not np.array_equal(full.loc[full.expert.isna(), "timestamp_selected"].to_numpy(),
                          full.loc[full.expert.isna(), "selected"].to_numpy()):
        raise ValueError("Invalid AOBT rows changed")
    return scores, full


def verify_model(model_path: Path, features: pd.DataFrame,
                 *, requested_iterations: int, expected_trees: int) -> Any:
    from catboost import CatBoostRegressor
    model = CatBoostRegressor()
    model.load_model(str(model_path))
    if model.feature_names_ != list(features) or int(model.tree_count_) != expected_trees:
        raise ValueError("Saved CatBoost feature order or tree count differs")
    cats = [features.columns.get_loc(col) for col in features.select_dtypes(include="category")]
    if list(model.get_cat_feature_indices()) != cats:
        raise ValueError("Saved CatBoost categorical column positions differ")
    params = model.get_all_params()
    expected = {"task_type": "GPU", "depth": 9, "random_seed": 2026,
                "loss_function": "RMSE", "learning_rate": 0.04,
                "l2_leaf_reg": 12, "border_count": 128,
                "max_ctr_complexity": 1, "one_hot_max_size": 20}
    for key, value in expected.items():
        actual = params.get(key)
        if isinstance(value, float):
            if actual is None or not np.isclose(float(actual), value, atol=1e-8, rtol=1e-8):
                raise ValueError(f"Saved CatBoost {key} differs")
        elif actual != value:
            raise ValueError(f"Saved CatBoost {key} differs")
    if "iterations" in params and int(params["iterations"]) != requested_iterations:
        raise ValueError("Saved CatBoost requested iteration cap differs")
    return model


def fold_paths(out: Path, fold: str) -> dict[str, Path]:
    folder = out / "deep"
    return {"model": folder / f"{fold}.cbm", "oof": folder / f"{fold}_oof.parquet",
            "native_report": folder / f"{fold}_validation.json",
            "receipt": folder / f"{fold}_receipt.json"}


def verify_fold(out: Path, fold: str, rows: pd.DataFrame,
                features: pd.DataFrame, reference: pd.DataFrame,
                snapshot: dict) -> tuple[dict, pd.DataFrame]:
    paths = fold_paths(out, fold)
    receipt = read_json(paths["receipt"])
    native = read_json(paths["native_report"])
    if (receipt.get("fold") != fold or receipt.get("status") != "complete"
            or receipt.get("parent_manifest_sha256") != snapshot["manifest_sha256"]
            or receipt.get("source_sha256") != snapshot["source_sha256"]
            or receipt.get("v4_reference_sha256") != sha256(out / "v4/validation_predictions.parquet")
            or receipt.get("schema") != schema_of(features)
            or receipt.get("native_report_sha256") != sha256(paths["native_report"])
            or receipt.get("model_sha256") != sha256(paths["model"])
            or receipt.get("oof_sha256") != sha256(paths["oof"])):
        raise ValueError(f"{fold} model/OOF/feature receipt differs")
    fit, early, test = split_indices(rows, fold)
    expected_coverage = {"seasonal_jan_jul": (1722655, 338802),
                         "forward_nov_dec": (1736577, 324862)}[fold]
    if (len(fit) + len(early), len(test)) != expected_coverage:
        raise ValueError("Original timestamp train/heldout coverage differs")
    ids = rows.MVT_ID_mvt
    for key, index in (("fit_ids_sha256", fit), ("early_ids_sha256", early),
                       ("heldout_ids_sha256", test)):
        if receipt.get(key) != id_hash(ids.iloc[index]):
            raise ValueError(f"{fold} split {key} differs")
    if receipt.get("training_eligible") != len(fit) + len(early):
        raise ValueError("Complement training coverage differs")
    if native.get("fold") != fold or native.get("features") != list(features):
        raise ValueError("Original fold report feature order differs")
    trees = int(native.get("trees", 0))
    if not 1 <= trees <= 4500 or receipt.get("trees") != trees or native.get("training_eligible") != len(fit) + len(early):
        raise ValueError("Saved fold tree count or training count differs")
    model = verify_model(paths["model"], features, requested_iterations=4500,
                         expected_trees=trees)
    oof = pd.read_parquet(paths["oof"])
    if list(oof) != ["MVT_ID_mvt", "expert"] or not np.array_equal(
            oof.MVT_ID_mvt.to_numpy(), ids.iloc[test].to_numpy()):
        raise ValueError("Saved timestamp OOF IDs/order differ")
    proxy = rows.proxy.to_numpy(dtype=float)
    replay = proxy[test] + model.predict(features.iloc[test], thread_count=2)
    if not np.allclose(oof.expert.to_numpy(dtype=float), replay,
                       rtol=1e-11, atol=1e-8, equal_nan=False):
        raise ValueError("Saved timestamp OOF does not replay from saved model")
    if not np.isfinite(replay).all():
        raise ValueError("Timestamp expert produced nonfinite OOF values")
    scores, full = score_fold(reference, fold, oof)
    if (int(native.get("n_all_finite", -1)) != len(full)
            or int(native.get("n_eligible", -1)) != len(oof)
            or any(not np.isclose(native["scores_all_finite"][key], value,
                                      rtol=1e-11, atol=1e-8)
                   for key, value in scores.items())):
        raise ValueError("Original all-finite fold scores do not replay")
    if receipt.get("scores_all_finite") != scores:
        raise ValueError("Receipt fold scores differ from replay")
    return receipt, oof


def fit_fold(args: argparse.Namespace) -> dict:
    root, out, _, paths, snapshot = real_context(args)
    if args.fold not in FOLDS:
        raise ValueError("Unknown original fold")
    protocol(root, out, snapshot)
    reference, ref_report = require_reference(root, out, paths, snapshot)
    fold = args.fold
    expected = fold_paths(out, fold)
    if any(path.exists() for path in expected.values()):
        raise FileExistsError("Fold artifact exists; no refit, overwrite or resume")
    from deep_timestamp_expert import fit_fold as original_fit_fold, load_features
    p = adapter_args(paths, out / "deep" / f"_stage_{fold}")
    if p.output_dir.exists():
        raise FileExistsError("Original fold staging directory exists")
    rows, features = load_features(p, ranking=False)
    row_metadata(rows)
    if len(rows) != len(features):
        raise ValueError("Timestamp features and baseline rows differ")
    schema = schema_of(features)
    fit, early, test = split_indices(rows, fold)
    result = original_fit_fold(fold, FOLDS[fold], rows, features, reference, p)
    stage = {key: p.output_dir / path.name for key, path in expected.items() if key != "receipt"}
    native = read_json(stage["native_report"])
    if result != native:
        raise ValueError("Original trainer report differs from saved report")
    # Read the saved stage model itself before exclusive promotion.
    candidate = pd.read_parquet(stage["oof"])
    if not np.array_equal(candidate.MVT_ID_mvt.to_numpy(), rows.MVT_ID_mvt.iloc[test].to_numpy()):
        raise ValueError("Original trainer OOF heldout IDs differ")
    saved = verify_model(stage["model"], features, requested_iterations=4500,
                         expected_trees=int(native["trees"]))
    replay = rows.proxy.to_numpy(dtype=float)[test] + saved.predict(features.iloc[test], thread_count=2)
    if not np.allclose(candidate.expert.to_numpy(dtype=float), replay,
                       rtol=1e-11, atol=1e-8):
        raise ValueError("Original trainer saved model/OOF replay failed")
    for key in ("model", "oof", "native_report"):
        assert_snapshot(root, snapshot)
        os.link(stage[key], expected[key])  # exclusive same-volume promotion
    for key in ("model", "oof", "native_report"):
        stage[key].unlink()
    p.output_dir.rmdir()
    scores, _ = score_fold(reference, fold, candidate)
    receipt = {"status": "complete", "fold": fold, "heldout_months": list(FOLDS[fold]),
               "source_sha256": snapshot["source_sha256"],
               "parent_manifest_sha256": snapshot["manifest_sha256"],
               "v4_reference_sha256": ref_report["reference_sha256"],
               "schema": schema, "fit_ids_sha256": id_hash(rows.MVT_ID_mvt.iloc[fit]),
               "early_ids_sha256": id_hash(rows.MVT_ID_mvt.iloc[early]),
               "heldout_ids_sha256": id_hash(rows.MVT_ID_mvt.iloc[test]),
               "training_eligible": len(fit) + len(early), "trees": int(native["trees"]),
               "requested_iterations": 4500, "depth": 9,
               "native_report_sha256": sha256(expected["native_report"]),
               "model_sha256": sha256(expected["model"]),
               "oof_sha256": sha256(expected["oof"]),
               "scores_all_finite": scores}
    assert_snapshot(root, snapshot)
    exclusive_json(expected["receipt"], receipt, root)
    verify_fold(out, fold, rows, features, reference, snapshot)
    return receipt


def validation_value(reference: pd.DataFrame, fold_oof: dict[str, pd.DataFrame],
                     receipts: dict[str, dict]) -> tuple[dict, pd.DataFrame]:
    frames = []
    scores = {}
    for fold in FOLDS:
        fold_scores, full = score_fold(reference, fold, fold_oof[fold])
        if receipts[fold]["scores_all_finite"] != fold_scores:
            raise ValueError("Fold receipt scores changed")
        scores[fold] = fold_scores
        frames.append(full[["MVT_ID_mvt", "target", "fold", "month", "MVT_TIME_UTC_mvt",
                            "a_valid", "selected", "expert", "timestamp_selected"]])
    selected = min(WEIGHTS, key=lambda w: scores["seasonal_jan_jul"][str(w)])
    if selected != FIXED_WEIGHT:
        raise ValueError("Original January/July weight is no longer 0.5; no retuning")
    if not scores["forward_nov_dec"][str(FIXED_WEIGHT)] < scores["forward_nov_dec"]["0.0"]:
        raise ValueError("Original forward fold no longer improves at fixed 0.5")
    all_oof = pd.concat(frames, ignore_index=True)
    if len(all_oof) != len(reference) or set(all_oof.MVT_ID_mvt) != set(reference.MVT_ID_mvt):
        raise ValueError("Timestamp all-finite OOF coverage differs")
    return {"status": "passed", "selected_weight": FIXED_WEIGHT,
            "folds": scores, "all_finite_rows": len(all_oof),
            "ordered_ids_sha256": id_hash(all_oof.MVT_ID_mvt)}, all_oof


def evaluate(args: argparse.Namespace) -> dict:
    root, out, _, paths, snapshot = real_context(args)
    protocol(root, out, snapshot)
    reference, ref_receipt = require_reference(root, out, paths, snapshot)
    report_path = out / "deep/validation.json"
    pred_path = out / "deep/validation_predictions.parquet"
    if report_path.exists() or pred_path.exists():
        raise FileExistsError("Timestamp evaluation already exists")
    from deep_timestamp_expert import load_features
    rows, features = load_features(adapter_args(paths, out / "deep"), ranking=False)
    row_metadata(rows)
    receipts, oof = {}, {}
    for fold in FOLDS:
        receipts[fold], oof[fold] = verify_fold(out, fold, rows, features, reference, snapshot)
    value, full = validation_value(reference, oof, receipts)
    assert_snapshot(root, snapshot)
    exclusive_parquet(pred_path, full, root)
    if not pd.read_parquet(pred_path).equals(full):
        raise ValueError("Timestamp validation Parquet readback differs")
    report = {**value, "source_sha256": snapshot["source_sha256"],
              "parent_manifest_sha256": snapshot["manifest_sha256"],
              "v4_reference_sha256": ref_receipt["reference_sha256"],
              "fold_receipt_sha256": {fold: sha256(fold_paths(out, fold)["receipt"]) for fold in FOLDS},
              "validation_predictions_sha256": sha256(pred_path)}
    assert_snapshot(root, snapshot)
    exclusive_json(report_path, report, root)
    return report


def require_validation(root: Path, out: Path, paths: dict[str, Path], snapshot: dict,
                       *, features: pd.DataFrame | None = None,
                       rows: pd.DataFrame | None = None) -> tuple[dict, pd.DataFrame]:
    reference, ref_receipt = require_reference(root, out, paths, snapshot)
    from deep_timestamp_expert import load_features
    if rows is None or features is None:
        rows, features = load_features(adapter_args(paths, out / "deep"), ranking=False)
    row_metadata(rows)
    receipts, oof = {}, {}
    for fold in FOLDS:
        receipts[fold], oof[fold] = verify_fold(out, fold, rows, features, reference, snapshot)
    expected, frame = validation_value(reference, oof, receipts)
    report = read_json(out / "deep/validation.json")
    saved = pd.read_parquet(out / "deep/validation_predictions.parquet")
    if (report.get("status") != "passed" or report.get("source_sha256") != snapshot["source_sha256"]
            or report.get("parent_manifest_sha256") != snapshot["manifest_sha256"]
            or report.get("v4_reference_sha256") != ref_receipt["reference_sha256"]
            or report.get("fold_receipt_sha256") != {
                fold: sha256(fold_paths(out, fold)["receipt"]) for fold in FOLDS}
            or report.get("validation_predictions_sha256") != sha256(out / "deep/validation_predictions.parquet")
            or any(report.get(key) != value for key, value in expected.items())
            or not saved.equals(frame)):
        raise ValueError("Timestamp selected policy/OOF score replay differs")
    return report, frame


def fit_final(args: argparse.Namespace) -> dict:
    root, out, _, paths, snapshot = real_context(args)
    protocol(root, out, snapshot)
    model_path = out / "deep/full_2025.cbm"
    report_path = out / "deep/full_2025_receipt.json"
    if model_path.exists() or report_path.exists():
        raise FileExistsError("Final timestamp model already exists")
    from deep_timestamp_expert import load_features, params
    from catboost import CatBoostRegressor, Pool
    import gc
    p = adapter_args(paths, out / "deep")
    rows, features = load_features(p, ranking=False)
    report, _ = require_validation(root, out, paths, snapshot, features=features, rows=rows)
    fold_receipts = {fold: read_json(fold_paths(out, fold)["receipt"]) for fold in FOLDS}
    iterations = int(np.median([fold_receipts[fold]["trees"] for fold in FOLDS]))
    y = rows.target.to_numpy(dtype=float)
    proxy = rows.proxy.to_numpy(dtype=float)
    eligible = (np.isfinite(y) & (y >= 0) & (y <= 86400)
                & np.isfinite(proxy) & (proxy >= 0) & (proxy <= 7200))
    train = np.flatnonzero(eligible)
    if len(train) != 2061428:
        raise ValueError("Original full timestamp training coverage differs")
    cats = features.select_dtypes(include="category").columns.tolist()
    parameters = params(p)
    parameters["iterations"] = iterations
    model = CatBoostRegressor(**parameters)
    pool = Pool(features.iloc[train], label=(y - proxy)[train], cat_features=cats)
    model.fit(pool)
    del pool
    gc.collect()
    assert_snapshot(root, snapshot)
    exclusive_model(model_path, model, root)
    saved = verify_model(model_path, features, requested_iterations=iterations,
                         expected_trees=int(model.tree_count_))
    if int(model.tree_count_) != iterations:
        raise ValueError("Final saved tree count differs from original-fold median")
    probe = train[:min(256, len(train))]
    in_memory = model.predict(features.iloc[probe], thread_count=2)
    readback = saved.predict(features.iloc[probe], thread_count=2)
    if not np.allclose(in_memory, readback, rtol=1e-12, atol=1e-9):
        raise ValueError("Final CatBoost prediction changed after save/readback")
    value = {"status": "complete", "source_sha256": snapshot["source_sha256"],
             "parent_manifest_sha256": snapshot["manifest_sha256"],
             "validation_sha256": sha256(out / "deep/validation.json"),
             "selected_weight": report["selected_weight"],
             "original_fold_trees": {fold: fold_receipts[fold]["trees"] for fold in FOLDS},
             "iterations": iterations, "actual_trees": int(model.tree_count_),
             "training_eligible": len(train),
             "training_ids_sha256": id_hash(rows.MVT_ID_mvt.iloc[train]),
             "probe_ids_sha256": id_hash(rows.MVT_ID_mvt.iloc[probe]),
             "probe_readback_prediction_sha256": float_hash(readback),
             "schema": schema_of(features), "model_sha256": sha256(model_path)}
    assert_snapshot(root, snapshot)
    exclusive_json(report_path, value, root)
    return value


def require_final(root: Path, out: Path, paths: dict[str, Path], snapshot: dict,
                  *, rows: pd.DataFrame | None = None,
                  features: pd.DataFrame | None = None) -> dict:
    from deep_timestamp_expert import load_features
    if rows is None or features is None:
        rows, features = load_features(adapter_args(paths, out / "deep"), ranking=False)
    validation, _ = require_validation(root, out, paths, snapshot, rows=rows, features=features)
    receipt = read_json(out / "deep/full_2025_receipt.json")
    folds = {fold: read_json(fold_paths(out, fold)["receipt"])["trees"] for fold in FOLDS}
    iterations = int(np.median(list(folds.values())))
    y = rows.target.to_numpy(dtype=float)
    proxy = rows.proxy.to_numpy(dtype=float)
    eligible = (np.isfinite(y) & (y >= 0) & (y <= 86400)
                & np.isfinite(proxy) & (proxy >= 0) & (proxy <= 7200))
    train = np.flatnonzero(eligible)
    if len(train) != 2061428:
        raise ValueError("Original full timestamp training coverage differs")
    model_path = out / "deep/full_2025.cbm"
    probe = train[:min(256, len(train))]
    if (receipt.get("status") != "complete" or receipt.get("source_sha256") != snapshot["source_sha256"]
            or receipt.get("parent_manifest_sha256") != snapshot["manifest_sha256"]
            or receipt.get("validation_sha256") != sha256(out / "deep/validation.json")
            or receipt.get("selected_weight") != validation["selected_weight"]
            or receipt.get("original_fold_trees") != folds
            or receipt.get("iterations") != iterations or receipt.get("actual_trees") != iterations
            or receipt.get("training_eligible") != len(train)
            or receipt.get("training_ids_sha256") != id_hash(rows.MVT_ID_mvt.iloc[train])
            or receipt.get("probe_ids_sha256") != id_hash(rows.MVT_ID_mvt.iloc[probe])
            or receipt.get("schema") != schema_of(features)
            or receipt.get("model_sha256") != sha256(model_path)):
        raise ValueError("Final timestamp model provenance differs")
    saved = verify_model(model_path, features, requested_iterations=iterations,
                         expected_trees=iterations)
    replay = saved.predict(features.iloc[probe], thread_count=2)
    if receipt.get("probe_readback_prediction_sha256") != float_hash(replay):
        raise ValueError("Final saved model prediction probe differs")
    return receipt


def verify_v4_rank(paths: dict[str, Path], manifest: dict) -> tuple[pd.DataFrame, np.ndarray]:
    template = pd.read_parquet(paths["submission_template"], columns=["MVT_ID_mvt"])
    if template.MVT_ID_mvt.isna().any() or template.MVT_ID_mvt.duplicated().any():
        raise ValueError("Submission template IDs are null or duplicated")
    ordered = id_hash(template.MVT_ID_mvt)
    for name in PRODUCERS:
        if manifest["producers"][name]["ordered_ranking_ids_sha256"] != ordered:
            raise ValueError(f"{name} parent ranking ordered ID receipt differs")
    v3 = pd.read_parquet(paths["v3_ranking_predictions"],
                         columns=["MVT_ID_mvt", "TAXITIME_SEC_mvt"])
    gpu_expert = pd.read_parquet(paths["gpu_ranking_expert"],
                                 columns=["MVT_ID_mvt", "gpu_prediction"])
    gpu = pd.read_parquet(paths["gpu_ranking_predictions"],
                           columns=["MVT_ID_mvt", "TAXITIME_SEC_mvt"])
    source = pd.read_parquet(paths["source_ranking_probabilities"],
                             columns=["MVT_ID_mvt", "source_candidate",
                                      "schedule_proxy_sec", "p_schedule_exact"])
    v4 = pd.read_parquet(paths["source_sequential_predictions"],
                          columns=["MVT_ID_mvt", "TAXITIME_SEC_mvt"])
    rank_rows = pd.read_parquet(paths["baseline_ranking_rows"],
                                columns=["MVT_ID_mvt", "proxy"])
    for name, frame in (("v3", v3), ("gpu_expert", gpu_expert), ("gpu", gpu),
                        ("source", source), ("v4", v4), ("rank_rows", rank_rows)):
        if len(frame) != len(template) or frame.MVT_ID_mvt.isna().any() or frame.MVT_ID_mvt.duplicated().any():
            raise ValueError(f"{name} ranking parent ID coverage differs")
        if not frame.MVT_ID_mvt.equals(template.MVT_ID_mvt):
            raise ValueError(f"{name} ranking parent order differs from template")
    if len(template) != 344841:
        raise ValueError("2026 template row count differs")
    proxy = rank_rows.proxy.to_numpy(dtype=float)
    valid = np.isfinite(proxy) & (proxy >= 0) & (proxy <= 7200)
    if int(valid.sum()) != 339377:
        raise ValueError("Original ranking own-AOBT gate coverage differs")
    base = v3.TAXITIME_SEC_mvt.to_numpy(dtype=float)
    expert = gpu_expert.gpu_prediction.to_numpy(dtype=float)
    expected_gpu = base.copy()
    available = np.isfinite(expert)
    if not valid[available].all():
        raise ValueError("GPU expert uses invalid AOBT proxy")
    expected_gpu[available] += .25 * (expert[available] - expected_gpu[available])
    expected_gpu = np.maximum(expected_gpu, 0)
    if not np.allclose(gpu.TAXITIME_SEC_mvt.to_numpy(dtype=float), expected_gpu,
                       rtol=1e-11, atol=1e-8):
        raise ValueError("GPU ranking parent fixed 0.25 policy does not replay")
    source_gate = source.source_candidate.to_numpy(dtype=bool)
    if not valid[source_gate].all():
        raise ValueError("Source ranking gate includes invalid AOBT")
    schedule = source.schedule_proxy_sec.to_numpy(dtype=float)
    probability = source.p_schedule_exact.to_numpy(dtype=float)
    if not (np.isfinite(schedule[source_gate]).all() and np.isfinite(probability[source_gate]).all()):
        raise ValueError("Source ranking parent has nonfinite correction inputs")
    expected_v4 = expected_gpu.copy()
    expected_v4[source_gate] += .5 * probability[source_gate] * (
        schedule[source_gate] - expected_v4[source_gate])
    expected_v4 = np.maximum(expected_v4, 0)
    saved = v4.TAXITIME_SEC_mvt.to_numpy(dtype=float)
    if not np.allclose(saved, expected_v4, rtol=1e-11, atol=1e-8) or not np.isfinite(saved).all():
        raise ValueError("v4 sequential ranking parent fixed policy does not replay")
    if not np.array_equal(saved[~valid], base[~valid]):
        raise ValueError("v4 parent changed invalid-AOBT ranking rows")
    return template, saved


def ranking_seal(args: argparse.Namespace) -> dict:
    root, out, _, paths, snapshot = real_context(args)
    protocol(root, out, snapshot)
    path = out / "deep/ranking_inputs.json"
    if path.exists():
        raise FileExistsError("Ranking input seal already exists")
    # All file hashes are computed before reading ranking feature values.
    # Final verification necessarily loads training features; those are sealed.
    require_final(root, out, paths, snapshot)
    value = {"status": "sealed_before_ranking_values", "source_sha256": snapshot["source_sha256"],
             "parent_manifest_sha256": snapshot["manifest_sha256"],
             "parent_file_sha256": snapshot["parent_file_sha256"],
             "model_sha256": sha256(out / "deep/full_2025.cbm"),
             "final_receipt_sha256": sha256(out / "deep/full_2025_receipt.json"),
             "validation_sha256": sha256(out / "deep/validation.json")}
    assert_snapshot(root, snapshot)
    exclusive_json(path, value, root)
    return value


def require_ranking_seal(out: Path, snapshot: dict) -> dict:
    value = read_json(out / "deep/ranking_inputs.json")
    if value != {"status": "sealed_before_ranking_values",
                 "source_sha256": snapshot["source_sha256"],
                 "parent_manifest_sha256": snapshot["manifest_sha256"],
                 "parent_file_sha256": snapshot["parent_file_sha256"],
                 "model_sha256": sha256(out / "deep/full_2025.cbm"),
                 "final_receipt_sha256": sha256(out / "deep/full_2025_receipt.json"),
                 "validation_sha256": sha256(out / "deep/validation.json")}:
        raise ValueError("Ranking input seal differs")
    return value


def predict(args: argparse.Namespace) -> dict:
    root, out, manifest, paths, snapshot = real_context(args)
    protocol(root, out, snapshot)
    output_path = out / "deep/predictions.parquet"
    expert_path = out / "deep/ranking_expert.parquet"
    report_path = out / "deep/ranking_manifest.json"
    if any(path.exists() for path in (output_path, expert_path, report_path)):
        raise FileExistsError("Ranking output already exists")
    seal = require_ranking_seal(out, snapshot)
    from deep_timestamp_expert import load_features
    p = adapter_args(paths, out / "deep")
    train_rows, train_features = load_features(p, ranking=False)
    final = require_final(root, out, paths, snapshot,
                          rows=train_rows, features=train_features)
    del train_rows, train_features
    template, base = verify_v4_rank(paths, manifest)
    rank_rows, rank_x = load_features(p, ranking=True)
    if not rank_rows.MVT_ID_mvt.equals(template.MVT_ID_mvt):
        raise ValueError("Timestamp ranking cache differs from template order")
    rank_schema = schema_of(rank_x)
    # Ranking categories may contain new 2026 levels; CatBoost accepts those.
    # Feature names, order, dtypes and categorical positions must still match.
    if (rank_schema["columns"] != final["schema"]["columns"]
            or rank_schema["categorical_columns"] != final["schema"]["categorical_columns"]):
        raise ValueError("Timestamp ranking feature schema/order differs from full fit")
    proxy = rank_rows.proxy.to_numpy(dtype=float)
    valid = np.isfinite(proxy) & (proxy >= 0) & (proxy <= 7200)
    if int(valid.sum()) != 339377:
        raise ValueError("Original timestamp ranking gate coverage differs")
    model = verify_model(out / "deep/full_2025.cbm", rank_x,
                         requested_iterations=final["iterations"],
                         expected_trees=final["actual_trees"])
    expert = np.full(len(rank_rows), np.nan, dtype=float)
    expert[valid] = proxy[valid] + model.predict(rank_x.loc[valid], thread_count=2)
    updated = base.copy()
    updated[valid] = np.maximum(base[valid] + FIXED_WEIGHT * (expert[valid] - base[valid]), 0)
    if (not np.isfinite(updated).all() or np.any(updated < 0)
            or not np.array_equal(updated[~valid], base[~valid])):
        raise ValueError("Timestamp ranking finite/invalid-row policy differs")
    assert_snapshot(root, snapshot)
    require_ranking_seal(out, snapshot)
    exclusive_parquet(expert_path, pd.DataFrame({"MVT_ID_mvt": template.MVT_ID_mvt,
                                                 "expert": expert}), root)
    exclusive_parquet(output_path, pd.DataFrame({"MVT_ID_mvt": template.MVT_ID_mvt,
                                                 "TAXITIME_SEC_mvt": updated}), root)
    expert_check = pd.read_parquet(expert_path)
    if (not expert_check.MVT_ID_mvt.equals(template.MVT_ID_mvt)
            or not np.array_equal(expert_check.expert.to_numpy(dtype=float), expert, equal_nan=True)):
        raise ValueError("Ranking expert Parquet round-trip differs")
    check = pd.read_parquet(output_path)
    if (not check.MVT_ID_mvt.equals(template.MVT_ID_mvt)
            or not np.array_equal(check.TAXITIME_SEC_mvt.to_numpy(dtype=float), updated)):
        raise ValueError("Ranking Parquet round-trip differs")
    value = {"status": "complete", "source_sha256": snapshot["source_sha256"],
             "parent_manifest_sha256": snapshot["manifest_sha256"],
             "ranking_seal_sha256": sha256(out / "deep/ranking_inputs.json"),
             "final_model_sha256": final["model_sha256"],
             "template_rows": len(template), "ordered_ids_sha256": id_hash(template.MVT_ID_mvt),
             "valid_proxy_rows": int(valid.sum()), "invalid_proxy_rows_unchanged": int((~valid).sum()),
             "ranking_feature_schema": rank_schema,
             "fixed_weight": FIXED_WEIGHT, "expert_sha256": sha256(expert_path),
             "predictions_sha256": sha256(output_path),
             "finite_nonnegative_and_template_order_verified": True}
    assert_snapshot(root, snapshot)
    require_ranking_seal(out, snapshot)
    exclusive_json(report_path, value, root)
    return value


def plan() -> dict:
    """No private data, parent metadata, model, cache or leaderboard access."""
    return {"status": "prospective_only", "spec": str(SPEC),
            "parent_manifest": "<run_root>/parents/v4_timestamp_parents.json",
            "required_parent_roles": list(CANONICAL),
            "additional_model_role_prefixes": [p["prefix"] for p in PRODUCERS.values()],
            "modes_after_publication": ["v4-reference", "fit-fold", "evaluate",
                                        "fit-final", "ranking-seal", "predict"],
            "upstream_clean_producers": list(PRODUCERS),
            "later_missing_clock_arrival_and_v6_plus_adapters_in_scope": False}


def synthetic_self_test() -> dict:
    """Only in-memory fabricated metadata, no competition data or model calls."""
    fake = lambda role: hashlib.sha256(role.encode()).hexdigest()
    files = {role: {"path": path, "sha256": fake(role)} for role, path in CANONICAL.items()}
    for name, policy in PRODUCERS.items():
        role = policy["prefix"] + "synthetic"
        files[role] = {"path": f"parents/synthetic/{role}.cbm", "sha256": fake(role)}
    producers = {}
    for name, policy in PRODUCERS.items():
        outputs = set(policy["outputs"]) | {policy["prefix"] + "synthetic"}
        inputs = set(DATA_ROLES)
        if name != "v3":
            inputs |= set(PRODUCERS["v3"]["outputs"]) | {"v3_producer_receipt"}
        if name == "source":
            inputs |= set(PRODUCERS["gpu"]["outputs"]) | {"gpu_producer_receipt"}
        producers[name] = {
            "name": name, "status": "complete", "source_commit": "a" * 40,
            "receipt_sha256": files[policy["receipt"]]["sha256"],
            "source_sha256": {f"{name}.py": fake(f"{name}.py")},
            "output_sha256": {role: files[role]["sha256"] for role in outputs},
            "input_sha256": {role: files[role]["sha256"] for role in inputs},
            "heldout_folds": {k: list(v) for k, v in FOLDS.items()},
            "fit_and_early_exclude_heldout": True,
            "published_choices": ({"route": "lobt_ensemble"} if name == "v3" else
                                  {"gpu_weight": .25} if name == "gpu" else
                                  {"gpu_weight": .25, "source_scale": .5}),
            "ordered_validation_ids_sha256": fake(name + "-ids"),
            "ordered_ranking_ids_sha256": fake(name + "-ranking-ids"),
            "independent_model_and_policy_replay_passed": True,
            "feature_schema_sha256": fake(name + "-schema")}
    manifest = {"schema_version": 1, "status": "complete", "files": files,
                "producers": producers,
                "heldout_folds": {k: list(v) for k, v in FOLDS.items()}}
    validate_manifest_metadata(manifest)
    poisoned = copy.deepcopy(manifest)
    poisoned["producers"]["source"]["input_sha256"]["raw_training_01"] = fake("other")
    try:
        validate_manifest_metadata(poisoned)
    except ValueError:
        pass
    else:
        raise AssertionError("Tampered raw parent input was accepted")
    poisoned = copy.deepcopy(manifest)
    poisoned["producers"]["gpu"]["published_choices"]["gpu_weight"] = .5
    try:
        validate_manifest_metadata(poisoned)
    except ValueError:
        pass
    else:
        raise AssertionError("Changed published GPU parent blend was accepted")
    rows = pd.DataFrame({"MVT_ID_mvt": [1, 2, 3, 4, 5],
                         "month": [1, 7, 11, 12, 4],
                         "target": [30., -3., 45., 75., 60.],
                         "proxy": [20., 30., np.nan, 7201., 50.]})
    fit, early, test = split_indices(rows, "seasonal_jan_jul")
    assert np.array_equal(test, [0, 1])
    assert not np.isin(rows.month.iloc[np.r_[fit, early]], [1, 7]).any()
    base = np.array([10., 20., 30.])
    expert = np.array([-50., 40., np.nan])
    valid = np.array([True, True, False])
    out = base.copy()
    out[valid] = np.maximum(base[valid] + .5 * (expert[valid] - base[valid]), 0)
    assert np.array_equal(out, [0., 30., 30.])
    synthetic_ref = pd.DataFrame({"MVT_ID_mvt": [1, 2], "selected": [20., 30.]})
    assert_reference_replay(synthetic_ref, synthetic_ref.copy())
    bad_ref = synthetic_ref.copy()
    bad_ref.loc[0, "selected"] = 21.
    try:
        assert_reference_replay(bad_ref, synthetic_ref)
    except ValueError:
        pass
    else:
        raise AssertionError("Reauthored v4 reference passed exact parent replay")
    with tempfile.TemporaryDirectory() as name:
        isolated = Path(name).resolve()
        if SOURCE_ROOT not in isolated.parents and isolated not in SOURCE_ROOT.parents:
            try:
                strict_child(isolated, "../escape", existing=False)
            except ValueError:
                pass
            else:
                raise AssertionError("Path escape was accepted")
    return {"synthetic_manifest": "passed", "tampered_parent_input": "refused",
            "changed_parent_weight": "refused", "heldout_and_early_masks": "passed",
            "fixed_valid_only_formula": "passed", "path_escape": "refused",
            "reauthored_v4_reference": "refused",
            "real_values_or_models_read": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("plan", "self-test", "v4-reference",
                                            "fit-fold", "evaluate", "fit-final",
                                            "ranking-seal", "predict"), default="plan")
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--fold", choices=tuple(FOLDS))
    parser.add_argument("--published-source-sha256")
    args = parser.parse_args()
    if args.mode == "plan":
        result = plan()
    elif args.mode == "self-test":
        result = synthetic_self_test()
    else:
        if args.run_root is None:
            parser.error("Real modes require --run-root")
        if args.mode == "v4-reference":
            result = write_reference(args)
        elif args.mode == "fit-fold":
            if args.fold is None:
                parser.error("--mode fit-fold requires --fold")
            result = fit_fold(args)
        elif args.mode == "evaluate":
            result = evaluate(args)
        elif args.mode == "fit-final":
            result = fit_final(args)
        elif args.mode == "ranking-seal":
            result = ranking_seal(args)
        else:
            result = predict(args)
    print(json.dumps(result, indent=2, allow_nan=False, default=str))


if __name__ == "__main__":
    main()
