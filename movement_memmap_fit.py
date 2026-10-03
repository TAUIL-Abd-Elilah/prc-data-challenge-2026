"""Bounded, provisional memmap fit of the frozen movement-only LightGBM model.

This is an operational route for the same complementary-month architecture in
``movement_only_expert.fit_movement_model``. It does not score, select, predict
OOF/ranking rows, or replace any accepted model. A fit must be launched as the
single direct child of ``run_bounded_worker.py``; the child's output remains
provisional until ``verify-run`` validates the external watchdog receipt.

The prepared feature cache must first have its exact independent integrity
seal. The existing full-pandas 10 GiB gate is unchanged. This route requires
3 GiB free initially and a separate watchdog with a 1.5 GiB free-memory floor
and a 2.0 GiB worker-RSS ceiling. No nested process is created here.
"""

from __future__ import annotations

import argparse
from contextlib import closing
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import psutil

import lightgbm_stream_feasibility as synthetic
import movement_only_expert as movement
import movement_prepared_stream_verify as streaming


ROOT = Path(__file__).resolve().parent
FOLDS = {
    "seasonal_jan_jul": (1, 7),
    "forward_nov_dec": (11, 12),
    "april_october": (4, 10),
    "february_august": (2, 8),
}
SUPPORTED_FIT_FOLD = "seasonal_jan_jul"
ORIGINAL_RECEIPT = Path("reports/movement_fold_model_provenance_v2.json")
ORIGINAL_RECEIPT_SHA256 = "cf6a42e5b11b6bded69032ccf6d28ea915f8a58def39499bcfa3c1c429e0d1ae"
BATCH_SIZE = 32768
MIN_INITIAL_FREE_GIB = 3.0
MAX_WORKER_RSS_GIB = 2.0
MIN_WATCHDOG_FREE_GIB = 1.5
NATIVE_THREADS = 3
ROUNDS = 1200
EARLY_STOP = 100


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_new_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as destination:
        destination.write(json.dumps(value, indent=2, default=str) + "\n")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def memory_sample() -> dict[str, float]:
    return {
        "available_gib": psutil.virtual_memory().available / 2**30,
        "worker_rss_gib": psutil.Process().memory_info().rss / 2**30,
    }


def movement_args(args: argparse.Namespace) -> argparse.Namespace:
    copied = argparse.Namespace(**vars(args))
    copied.output_dir = args.movement_dir
    return copied


def source_paths(args: argparse.Namespace) -> dict[str, Path]:
    paths = streaming.source_paths(movement_args(args))
    paths.update({
        "prepared_integrity": args.movement_dir / "prepared_integrity.json",
        "memmap_worker_source": Path(__file__).resolve(),
        "external_watchdog_source": ROOT / "run_bounded_worker.py",
        "synthetic_parity_source": Path(synthetic.__file__).resolve(),
    })
    return paths


def snapshot(paths: dict[str, Path]) -> dict[str, str]:
    return {role: sha256(path) for role, path in paths.items()}


def check_workspace_and_run(args: argparse.Namespace) -> None:
    if Path.cwd().resolve() != ROOT:
        raise ValueError("Run from the repository root")
    if args.run_dir.resolve() == args.movement_dir.resolve():
        raise ValueError("The bounded fit must not write into the frozen movement directory")
    artifacts = (ROOT / "artifacts").resolve()
    if not args.run_dir.resolve().is_relative_to(artifacts):
        raise ValueError("Run directory must be inside the ignored artifacts directory")
    if args.run_dir.exists():
        raise FileExistsError("Run directory already exists; no overwrite or resume")
    if not args.watchdog_report.resolve().is_relative_to(artifacts):
        raise ValueError("Watchdog receipt must be inside the ignored artifacts directory")
    if args.watchdog_report.exists():
        raise FileExistsError("Watchdog receipt path already exists")
    if args.batch_size < 1 or args.batch_size > 65536:
        raise ValueError("Batch size must be in 1..65536")
    if psutil.virtual_memory().available / 2**30 < MIN_INITIAL_FREE_GIB:
        raise MemoryError("Bounded worker requires at least 3 GiB available before launch")


def masks_from_rows(rows: pd.DataFrame, months: tuple[int, int]
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Exact dtype, time conversion, and row order of the frozen fit function."""
    y = rows.target.to_numpy(dtype=np.float32)
    heldout = np.isin(rows.month.to_numpy(dtype=np.int16), months)
    ordinary = np.isfinite(y) & (y >= 0) & (y <= 7200) & ~heldout
    day = pd.to_datetime(rows.time, utc=True, errors="coerce").dt.floor("D")
    if day.isna().any():
        raise ValueError("Every training departure must have a movement date")
    day_number = (day.dt.as_unit("ns").astype("int64").to_numpy()
                  // 86_400_000_000_000)
    internal = ordinary & (day_number % 11 == 0)
    fit = ordinary & ~internal
    return y, ordinary, fit, internal


def build_masks(args: argparse.Namespace, n: int, months: tuple[int, int]
                ) -> tuple[dict, Path, Path]:
    fit_path = args.run_dir / "fit_mask.uint8"
    early_path = args.run_dir / "early_mask.uint8"
    fit_mask = np.memmap(fit_path, mode="w+", dtype=np.uint8, shape=(n,))
    early_mask = np.memmap(early_path, mode="w+", dtype=np.uint8, shape=(n,))
    counts = {"ordinary_complement_rows": 0, "fit_rows": 0,
              "internal_early_rows": 0}
    cursor = streaming.ParquetFrameCursor(
        args.cache_dir / "training_rows.parquet",
        streaming.INFERENCE_ROW_COLUMNS, args.batch_size)
    try:
        for offset in range(0, n, args.batch_size):
            part = cursor.take(min(args.batch_size, n - offset))
            _, ordinary, fit, early = masks_from_rows(part, months)
            length = len(part)
            fit_mask[offset:offset + length] = fit.astype(np.uint8)
            early_mask[offset:offset + length] = early.astype(np.uint8)
            counts["ordinary_complement_rows"] += int(ordinary.sum())
            counts["fit_rows"] += int(fit.sum())
            counts["internal_early_rows"] += int(early.sum())
        cursor.finish()
    finally:
        cursor.close()
        fit_mask.flush()
        early_mask.flush()
        del fit_mask, early_mask
    if counts["fit_rows"] < 100_000 or counts["internal_early_rows"] < 10_000:
        raise ValueError("Insufficient ordinary fit or internal early-stop rows")
    return counts, fit_path, early_path


def category_metadata(frame: pd.DataFrame, manifest: dict
                      ) -> list[list]:
    if list(frame) != manifest["features"]:
        raise ValueError("Prepared features differ in name or order")
    actual = [name for name in frame
              if isinstance(frame[name].dtype, pd.CategoricalDtype)]
    if actual != manifest["categorical"]:
        raise ValueError("Prepared categorical order differs from manifest")
    return [list(frame[name].cat.categories) for name in actual]


def fill_matrices(args: argparse.Namespace, manifest: dict, counts: dict,
                  fit_mask_path: Path, early_mask_path: Path) -> tuple[dict, dict]:
    """Write exactly ordered LightGBM-encoded matrices and float32 labels."""
    n, columns = int(manifest["rows"]), manifest["features"]
    cat_names = manifest["categorical"]
    mask_fit = np.memmap(fit_mask_path, mode="r", dtype=np.uint8, shape=(n,))
    mask_early = np.memmap(early_mask_path, mode="r", dtype=np.uint8, shape=(n,))
    paths = {
        "fit_matrix": args.run_dir / "fit_matrix.dat",
        "early_matrix": args.run_dir / "early_matrix.dat",
        "fit_labels": args.run_dir / "fit_labels.float32",
        "early_labels": args.run_dir / "early_labels.float32",
    }
    train_values = early_values = fit_labels = early_labels = None
    categories = None
    matrix_dtype = None
    fit_offset = early_offset = 0
    iterator = streaming.iter_prepared_training_batches(
        args.movement_dir, args.cache_dir, args.batch_size)
    with closing(iterator):
        for source_offset, (rows, features) in enumerate(iterator):
            offset = source_offset * args.batch_size
            size = len(rows)
            if offset + size > n:
                raise ValueError("Stream exceeds prepared manifest row count")
            y, _, fit, early = masks_from_rows(rows, FOLDS[args.fold])
            if (not np.array_equal(mask_fit[offset:offset + size], fit)
                    or not np.array_equal(mask_early[offset:offset + size], early)):
                raise ValueError(f"Streamed masks differ at source row {offset}")
            current = category_metadata(features, manifest)
            if categories is None:
                categories = current
                probe, names, categorical, observed = lgb.basic._data_from_pandas(
                    features.iloc[:1], columns, cat_names, categories)
                if names != columns or categorical != cat_names or observed != categories:
                    raise ValueError("LightGBM pandas encoder schema differs")
                matrix_dtype = probe.dtype
                shape_fit = (counts["fit_rows"], len(columns))
                shape_early = (counts["internal_early_rows"], len(columns))
                train_values = np.memmap(paths["fit_matrix"], mode="w+",
                                         dtype=matrix_dtype, shape=shape_fit, order="C")
                early_values = np.memmap(paths["early_matrix"], mode="w+",
                                         dtype=matrix_dtype, shape=shape_early, order="C")
                fit_labels = np.memmap(paths["fit_labels"], mode="w+",
                                       dtype=np.float32, shape=(shape_fit[0],))
                early_labels = np.memmap(paths["early_labels"], mode="w+",
                                         dtype=np.float32, shape=(shape_early[0],))
            elif current != categories:
                raise ValueError("Global categorical levels changed between batches")
            for selected, matrix, label, written in (
                (fit, train_values, fit_labels, fit_offset),
                (early, early_values, early_labels, early_offset),
            ):
                amount = int(selected.sum())
                if not amount:
                    continue
                part = features.loc[selected]
                encoded, names, categorical, observed = lgb.basic._data_from_pandas(
                    part, columns, cat_names, categories)
                if (encoded.dtype != matrix_dtype or names != columns
                        or categorical != cat_names or observed != categories
                        or encoded.shape != (amount, len(columns))):
                    raise ValueError("LightGBM batch encoding differs from frozen pandas schema")
                matrix[written:written + amount] = encoded
                label[written:written + amount] = y[selected]
                if selected is fit:
                    fit_offset += amount
                else:
                    early_offset += amount
    if (fit_offset != counts["fit_rows"]
            or early_offset != counts["internal_early_rows"]):
        raise ValueError("Fit or internal early row count changed during streaming")
    for array in (train_values, early_values, fit_labels, early_labels):
        assert array is not None
        array.flush()
    del train_values, early_values, fit_labels, early_labels, mask_fit, mask_early
    gc.collect()
    if matrix_dtype is None or categories is None:
        raise ValueError("Prepared feature stream was empty")
    metadata = {
        "shape_fit": [counts["fit_rows"], len(columns)],
        "shape_early": [counts["internal_early_rows"], len(columns)],
        "matrix_dtype": str(matrix_dtype), "label_dtype": "float32",
        "row_order": "Original prepared/baseline order filtered by frozen fit/early masks",
        "features": columns, "categorical": cat_names,
        "pandas_categorical": categories,
        "matrix_and_mask_sha256": {role: sha256(path) for role, path in {
            **paths, "fit_mask": fit_mask_path, "early_mask": early_mask_path}.items()},
    }
    return metadata, paths


def train_from_memmaps(args: argparse.Namespace, metadata: dict,
                       paths: dict[str, Path]) -> tuple[lgb.Booster, dict]:
    """Same native fit call and parameters, in this one watched process."""
    for role, path in paths.items():
        if sha256(path) != metadata["matrix_and_mask_sha256"][role]:
            raise ValueError(f"Encoded {role} changed before native fit")
    fit_shape = tuple(metadata["shape_fit"])
    early_shape = tuple(metadata["shape_early"])
    dtype = np.dtype(metadata["matrix_dtype"])
    if dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise ValueError("LightGBM matrix dtype must be float32 or float64")
    matrix_fit = np.memmap(paths["fit_matrix"], mode="r", dtype=dtype,
                           shape=fit_shape, order="C")
    matrix_early = np.memmap(paths["early_matrix"], mode="r", dtype=dtype,
                             shape=early_shape, order="C")
    label_fit = np.memmap(paths["fit_labels"], mode="r", dtype=np.float32,
                          shape=(fit_shape[0],))
    label_early = np.memmap(paths["early_labels"], mode="r", dtype=np.float32,
                            shape=(early_shape[0],))
    train_set = lgb.Dataset(matrix_fit, label=label_fit,
                            feature_name=metadata["features"],
                            categorical_feature=metadata["categorical"],
                            free_raw_data=True)
    train_set.pandas_categorical = metadata["pandas_categorical"]
    early_set = lgb.Dataset(matrix_early, label=label_early,
                            reference=train_set,
                            feature_name=metadata["features"],
                            categorical_feature=metadata["categorical"],
                            free_raw_data=True)
    params = movement.model_params(NATIVE_THREADS)
    started = time.monotonic()
    model = lgb.train(params, train_set, num_boost_round=ROUNDS,
                      valid_sets=[early_set], callbacks=[
                          lgb.early_stopping(EARLY_STOP, verbose=True),
                          lgb.log_evaluation(period=100)])
    report = {"best_round": int(model.best_iteration or ROUNDS),
              "num_trees": int(model.num_trees()),
              "fit_seconds": time.monotonic() - started,
              "params": params, "max_rounds": ROUNDS,
              "early_stop_rounds": EARLY_STOP,
              "lightgbm_version": lgb.__version__,
              "native_threads": NATIVE_THREADS}
    return model, report


def fit_fold(args: argparse.Namespace) -> dict:
    if args.fold != SUPPORTED_FIT_FOLD:
        raise ValueError("Only seasonal Jan/Jul equivalence fitting is enabled")
    check_workspace_and_run(args)
    paths = source_paths(args)
    before = snapshot(paths)
    seal = movement.verify_prepared_integrity(movement_args(args))
    if seal.get("method") != "independent_rebuild_exact":
        raise ValueError("Memmap fit requires the independently rebuilt exact prepared cache seal")
    movement.verify_reference(args.v5_oof)
    manifest = read_json(args.movement_dir / "features_manifest.json")
    if (seal["rows"] != manifest["rows"] or
            seal["feature_names"] != manifest["features"] or
            seal["categorical_names"] != manifest["categorical"]):
        raise ValueError("Prepared seal and manifest feature schema differ")
    args.run_dir.mkdir(parents=True, exist_ok=False)
    months = FOLDS[args.fold]
    started = time.monotonic()
    memory = {"before_masks": memory_sample()}
    counts, fit_mask, early_mask = build_masks(args, int(manifest["rows"]), months)
    memory["after_masks"] = memory_sample()
    metadata, matrix_paths = fill_matrices(args, manifest, counts,
                                           fit_mask, early_mask)
    memory["after_matrices"] = memory_sample()
    write_new_json(args.run_dir / "matrix_manifest.json", {
        "status": "encoded_pending_fit", "fold": args.fold,
        "heldout_months": list(months), **counts, **metadata,
        "source_sha256_before": before,
        "prepared_seal_sha256": before["prepared_integrity"],
    })
    if snapshot(paths) != before:
        raise ValueError("Training sources changed during memmap construction")
    memory["before_native_fit"] = memory_sample()
    model, fit_report = train_from_memmaps(args, metadata, matrix_paths)
    memory["after_native_fit"] = memory_sample()
    model_path = args.run_dir / "model.txt"
    if model_path.exists():
        raise FileExistsError("Provisional model already exists")
    model.save_model(str(model_path))
    after = snapshot(paths)
    if after != before:
        raise ValueError("Training sources changed during LightGBM fitting")
    worker_report = {
        "status": "worker_complete_pending_watchdog",
        "fold": args.fold, "heldout_months": list(months),
        **counts, **fit_report,
        "model_sha256": sha256(model_path),
        "matrix_manifest_sha256": sha256(args.run_dir / "matrix_manifest.json"),
        "matrix_and_mask_sha256": metadata["matrix_and_mask_sha256"],
        "prepared_feature_schema": manifest["features"],
        "prepared_categorical_schema": manifest["categorical"],
        "prepared_manifest_sha256": before["original_features_manifest"],
        "prepared_seal_sha256": before["prepared_integrity"],
        "source_sha256_before": before,
        "source_sha256_after": after,
        "worker_pid": os.getpid(),
        "worker_cli_args": sys.argv[1:],
        "run_dir": str(args.run_dir.resolve()),
        "watchdog_report_expected": str(args.watchdog_report.resolve()),
        "worker_memory_samples": memory,
        "elapsed_seconds": time.monotonic() - started,
        "no_oof_prediction_or_selection": True,
        "no_ranking_output": True,
    }
    write_new_json(args.run_dir / "worker_report.json", worker_report)
    return {"status": worker_report["status"], "fold": args.fold,
            "best_round": fit_report["best_round"],
            "model_sha256": worker_report["model_sha256"],
            "requires_watchdog_receipt": str(args.watchdog_report)}


def verify_run(args: argparse.Namespace, *, write_receipt: bool = True) -> dict:
    """Check external PID watchdog and immutable artifacts; perform no fit."""
    if Path.cwd().resolve() != ROOT:
        raise ValueError("Run from repository root")
    if args.fold != SUPPORTED_FIT_FOLD:
        raise ValueError("Only seasonal Jan/Jul equivalence verification is enabled")
    report_path = args.run_dir / "worker_report.json"
    worker = read_json(report_path)
    watchdog = read_json(args.watchdog_report)
    if (worker.get("status") != "worker_complete_pending_watchdog"
            or worker.get("watchdog_report_expected") != str(args.watchdog_report.resolve())
            or worker.get("run_dir") != str(args.run_dir.resolve())
            or worker.get("fold") != args.fold
            or watchdog.get("worker") != Path(__file__).name
            or watchdog.get("success") is not True
            or watchdog.get("exit_code") != 0
            or watchdog.get("abort_reason") is not None
            or watchdog.get("child_pid") != worker.get("worker_pid")
            or watchdog.get("worker_source_unchanged") is not True
            or watchdog.get("worker_source_sha256") != sha256(Path(__file__))
            or watchdog.get("worker_args") != worker.get("worker_cli_args")
            or watchdog.get("monitor_source_sha256") !=
               worker.get("source_sha256_before", {}).get("external_watchdog_source")
            or watchdog.get("maximum_worker_rss_limit_gib", 99) > MAX_WORKER_RSS_GIB
            or watchdog.get("minimum_available_limit_gib", 0) < MIN_WATCHDOG_FREE_GIB
            or watchdog.get("peak_worker_rss_gib", 99) > MAX_WORKER_RSS_GIB
            or watchdog.get("minimum_available_gib", 0) < MIN_WATCHDOG_FREE_GIB):
        raise ValueError("Worker output lacks a successful bounded external watchdog receipt")
    if (worker["source_sha256_before"] != worker["source_sha256_after"]
            or snapshot(source_paths(args)) != worker["source_sha256_before"]
            or worker["model_sha256"] != sha256(args.run_dir / "model.txt")
            or worker["matrix_manifest_sha256"] !=
               sha256(args.run_dir / "matrix_manifest.json")):
        raise ValueError("Sources or provisional model changed after watched fit")
    if read_json(args.movement_dir / "prepared_integrity.json").get("method") != "independent_rebuild_exact":
        raise ValueError("Independent prepared cache seal is absent")
    matrix = read_json(args.run_dir / "matrix_manifest.json")
    for name, expected in worker["matrix_and_mask_sha256"].items():
        path = args.run_dir / ({
            "fit_matrix": "fit_matrix.dat", "early_matrix": "early_matrix.dat",
            "fit_labels": "fit_labels.float32", "early_labels": "early_labels.float32",
            "fit_mask": "fit_mask.uint8", "early_mask": "early_mask.uint8",
        }[name])
        if sha256(path) != expected:
            raise ValueError(f"Memmap artifact changed after watched fit: {name}")
    if (matrix["matrix_and_mask_sha256"] != worker["matrix_and_mask_sha256"]
            or matrix["source_sha256_before"] != worker["source_sha256_before"]
            or matrix["fold"] != worker["fold"]):
        raise ValueError("Provisional matrix and worker manifests disagree")
    saved = lgb.Booster(model_file=str(args.run_dir / "model.txt"))
    if (saved.feature_name() != matrix["features"]
            or saved.pandas_categorical != matrix["pandas_categorical"]
            or saved.num_trees() != int(worker["best_round"])
            or worker["heldout_months"] != list(FOLDS[args.fold])
            or worker["params"] != movement.model_params(NATIVE_THREADS)
            or worker["max_rounds"] != ROUNDS
            or worker["early_stop_rounds"] != EARLY_STOP):
        raise ValueError("Saved model or frozen fit policy differs")
    receipt = {
        "status": "operationally_verified_pending_scientific_integration",
        "fold": args.fold, "heldout_months": worker["heldout_months"],
        "worker_report_sha256": sha256(report_path),
        "watchdog_report_sha256": sha256(args.watchdog_report),
        "model_sha256": worker["model_sha256"],
        "matrix_manifest_sha256": worker["matrix_manifest_sha256"],
        "prepared_seal_sha256": worker["prepared_seal_sha256"],
        "source_sha256": worker["source_sha256_before"],
        "no_model_selection_or_prediction": True,
    }
    if write_receipt:
        write_new_json(args.run_dir / "watchdog_verified.json", receipt)
    return receipt


def verify_equivalence(args: argparse.Namespace) -> dict:
    """Require byte-identical original seasonal model, without scoring rows."""
    operational = verify_run(args, write_receipt=False)
    if read_json(args.run_dir / "watchdog_verified.json") != operational:
        raise ValueError("Independent watchdog verification receipt changed")
    if sha256(args.original_receipt) != args.original_receipt_sha256.lower():
        raise ValueError("Published original movement-model receipt changed")
    lineage = read_json(args.original_receipt)
    worker = read_json(args.run_dir / "worker_report.json")
    matrix = read_json(args.run_dir / "matrix_manifest.json")
    if (lineage.get("schema_version") != 1
            or lineage.get("features") != matrix["features"]
            or lineage.get("categorical") != matrix["categorical"]
            or lineage.get("provenance_input_sha256") !=
               movement.prepared_provenance(movement_args(args))):
        raise ValueError("Original source/cache lineage differs from the memmap fit")
    expected_sources = {
        "prepared_features": args.movement_dir / "features.parquet",
        "row_ids": args.movement_dir / "row_ids.parquet",
        "features_manifest": args.movement_dir / "features_manifest.json",
        "original_protocol": args.movement_dir / "protocol.json",
        "original_validation": args.movement_dir / "validation.json",
        "public_original_validation": Path("reports/movement_only_validation_v6.json"),
        "recorder_source": Path("record_movement_fold_provenance.py"),
    }
    if set(lineage.get("source_files", {})) != set(expected_sources):
        raise ValueError("Original model lineage source roles differ")
    for role, record in lineage["source_files"].items():
        path = Path(record["path"])
        if (path.resolve() != expected_sources[role].resolve()
                or record["sha256"] != sha256(path)):
            raise ValueError(f"Original model lineage source differs: {role}")
    entry = lineage["folds"][SUPPORTED_FIT_FOLD]
    original_model_path = args.movement_dir / f"{SUPPORTED_FIT_FOLD}.txt"
    original_fit_path = args.movement_dir / f"{SUPPORTED_FIT_FOLD}_fit.json"
    original_oof_path = args.movement_dir / f"{SUPPORTED_FIT_FOLD}_oof.parquet"
    original_fit = read_json(original_fit_path)
    if (Path(entry["model_path"]).resolve() != original_model_path.resolve()
            or Path(entry["fit_report_path"]).resolve() != original_fit_path.resolve()
            or Path(entry["original_oof_path"]).resolve() != original_oof_path.resolve()
            or entry["model_sha256"] != sha256(original_model_path)
            or entry["fit_report_sha256"] != sha256(original_fit_path)
            or entry["original_oof_sha256"] != sha256(original_oof_path)
            or entry["heldout_months"] != [1, 7]
            or original_fit["heldout_months"] != [1, 7]
            or original_fit["features_manifest_sha256"] !=
               sha256(args.movement_dir / "features_manifest.json")
            or int(entry["best_round"]) != int(original_fit["best_round"])
            or int(worker["best_round"]) != int(original_fit["best_round"])
            or any(int(worker[field]) != int(original_fit[field]) for field in
                   ("ordinary_complement_rows", "fit_rows", "internal_early_rows"))):
        raise ValueError("Original seasonal fit or exact fit/early rows differ")
    original_model = lgb.Booster(model_file=str(original_model_path))
    memmap_model = lgb.Booster(model_file=str(args.run_dir / "model.txt"))
    equal_text = (original_model_path.read_bytes() ==
                  (args.run_dir / "model.txt").read_bytes())
    if (not equal_text
            or original_model.num_trees() != int(entry["num_trees"])
            or memmap_model.num_trees() != int(entry["num_trees"])
            or original_model.feature_name() != matrix["features"]
            or memmap_model.feature_name() != matrix["features"]
            or original_model.pandas_categorical != matrix["pandas_categorical"]
            or memmap_model.pandas_categorical != matrix["pandas_categorical"]):
        raise ValueError("Complete original and memmap seasonal model text/schema differs")
    result = {
        "status": "seasonal_memmap_fit_byte_identical_to_frozen_original",
        "original_receipt_sha256": sha256(args.original_receipt),
        "watchdog_verified_sha256": sha256(args.run_dir / "watchdog_verified.json"),
        "original_model_sha256": sha256(original_model_path),
        "memmap_model_sha256": sha256(args.run_dir / "model.txt"),
        "original_fit_report_sha256": sha256(original_fit_path),
        "best_round": int(entry["best_round"]),
        "complete_model_text_equal": True,
        "no_prediction_or_score": True,
        "other_month_fit_routes_enabled": False,
    }
    write_new_json(args.run_dir / "seasonal_equivalence.json", result)
    return result


def contract() -> dict:
    return {
        "status": "code_only_not_fitted",
        "enabled_fit_fold": {SUPPORTED_FIT_FOLD: list(FOLDS[SUPPORTED_FIT_FOLD])},
        "other_folds": "Disabled pending externally watched seasonal model byte equivalence and a separate source/guard integration receipt",
        "frozen_model_params": movement.model_params(NATIVE_THREADS),
        "max_rounds": ROUNDS, "early_stop_rounds": EARLY_STOP,
        "ordinary_target_range_inclusive": [0, 7200],
        "internal_early_stop": "UTC day number modulo 11 equals zero",
        "initial_free_gib": MIN_INITIAL_FREE_GIB,
        "external_watchdog_min_available_gib": MIN_WATCHDOG_FREE_GIB,
        "external_watchdog_max_worker_rss_gib": MAX_WORKER_RSS_GIB,
        "direct_single_worker": True, "real_fit_performed": False,
    }


def synthetic_matrix_parity() -> dict:
    """Exercise this worker's real batch encoder on tiny noncompetition files."""
    frame, original_y, _, _ = synthetic.synthetic_frame()
    frame = frame.rename(columns={"airport": "ADEP_mvt"})
    airport_levels = list(frame.ADEP_mvt.cat.categories) + ["__MISSING__"]
    frame["ADEP_mvt"] = frame.ADEP_mvt.cat.set_categories(
        airport_levels).fillna("__MISSING__")
    n = len(frame)
    day_number = np.arange(n) // 55
    months = np.where(day_number % 12 == 0, 1,
                      np.where(day_number % 12 == 6, 7, 3))
    rows = pd.DataFrame({
        "MVT_ID_mvt": np.arange(1, n + 1, dtype=np.float64),
        "target": (original_y + 10).astype(np.float32),
        "proxy": np.full(n, np.nan),
        "month": months.astype(np.int16),
        "airport": frame.ADEP_mvt.astype("string"),
        "time": pd.Timestamp("2025-01-01", tz="UTC")
                + pd.to_timedelta(day_number, unit="D"),
    })
    y, ordinary, fit, early = masks_from_rows(rows, FOLDS["seasonal_jan_jul"])
    names = list(frame)
    cats = [name for name in names
            if isinstance(frame[name].dtype, pd.CategoricalDtype)]
    levels = [list(frame[name].cat.categories) for name in cats]
    expected_fit, _, _, _ = lgb.basic._data_from_pandas(
        frame.loc[fit], names, cats, levels)
    expected_early, _, _, _ = lgb.basic._data_from_pandas(
        frame.loc[early], names, cats, levels)
    with tempfile.TemporaryDirectory(prefix="movement-memmap-synthetic-") as temp:
        root = Path(temp)
        movement_dir, cache_dir, run_dir = (root / "movement", root / "cache",
                                             root / "run")
        for path in (movement_dir, cache_dir, run_dir):
            path.mkdir()
        frame.to_parquet(movement_dir / "features.parquet", index=False)
        rows[["MVT_ID_mvt"]].to_parquet(movement_dir / "row_ids.parquet",
                                          index=False)
        rows.to_parquet(cache_dir / "training_rows.parquet", index=False)
        manifest = {"rows": n, "features": names, "categorical": cats}
        write_new_json(movement_dir / "features_manifest.json", manifest)
        fit_path, early_path = (run_dir / "fit_mask.uint8",
                                run_dir / "early_mask.uint8")
        fit.astype(np.uint8).tofile(fit_path)
        early.astype(np.uint8).tofile(early_path)
        fake_args = argparse.Namespace(run_dir=run_dir, movement_dir=movement_dir,
                                       cache_dir=cache_dir,
                                       fold="seasonal_jan_jul", batch_size=37)
        counts = {"ordinary_complement_rows": int(ordinary.sum()),
                  "fit_rows": int(fit.sum()),
                  "internal_early_rows": int(early.sum())}
        metadata, paths = fill_matrices(fake_args, manifest, counts,
                                        fit_path, early_path)
        actual_fit = np.fromfile(paths["fit_matrix"],
                                 dtype=np.dtype(metadata["matrix_dtype"])).reshape(
                                     metadata["shape_fit"])
        actual_early = np.fromfile(paths["early_matrix"],
                                   dtype=np.dtype(metadata["matrix_dtype"])).reshape(
                                       metadata["shape_early"])
        fit_labels = np.fromfile(paths["fit_labels"], dtype=np.float32)
        early_labels = np.fromfile(paths["early_labels"], dtype=np.float32)
        exact = (np.array_equal(actual_fit, expected_fit, equal_nan=True)
                 and np.array_equal(actual_early, expected_early, equal_nan=True)
                 and np.array_equal(fit_labels, y[fit])
                 and np.array_equal(early_labels, y[early]))
        if not exact:
            raise ValueError("Worker's streamed matrices/labels differ from full pandas")
        return {"synthetic_rows": n, **counts,
                "uneven_batch_size": 37,
                "categorical_levels_and_order_preserved": True,
                "fit_early_matrix_and_label_values_exact": True,
                "matrix_dtype": metadata["matrix_dtype"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("contract", "synthetic-parity",
                                           "fit-fold", "verify-run",
                                           "verify-equivalence"),
                        default="contract")
    parser.add_argument("--fold", choices=tuple(FOLDS))
    parser.add_argument("--run-dir", type=Path,
                        default=Path("artifacts/movement-memmap/pending-run"))
    parser.add_argument("--watchdog-report", type=Path,
                        default=Path("artifacts/movement-memmap/pending-watchdog.json"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--cache-dir", type=Path,
                        default=Path("artifacts/baseline"))
    parser.add_argument("--movement-dir", type=Path,
                        default=Path("artifacts/v6-movement-only"))
    parser.add_argument("--arrival-cache", type=Path,
                        default=Path("artifacts/v5-arrival-clean/training_arrival_features.parquet"))
    parser.add_argument("--weather-file", type=Path,
                        default=Path("data/external/weather.parquet"))
    parser.add_argument("--v5-oof", type=Path,
                        default=Path("artifacts/v5-ensemble/validation_predictions.parquet"))
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--original-receipt", type=Path,
                        default=ORIGINAL_RECEIPT)
    parser.add_argument("--original-receipt-sha256",
                        default=ORIGINAL_RECEIPT_SHA256)
    args = parser.parse_args()
    if args.mode in ("fit-fold", "verify-run", "verify-equivalence") and args.fold is None:
        parser.error("The selected mode requires --fold")
    if args.mode == "contract":
        result = contract()
    elif args.mode == "synthetic-parity":
        baseline = synthetic.run()
        stress = synthetic.run(1000)
        if not all(row["encoded_values_exact"] and row["tree_dump_equal"]
                   and row["model_text_equal"] and row["prediction_bitwise_equal"]
                   for row in (baseline, stress)):
            raise ValueError("Synthetic pandas/memmap LightGBM parity failed")
        matrix = synthetic_matrix_parity()
        result = {"baseline": baseline, "sampled_binning_stress": stress,
                  "worker_batch_encoder": matrix,
                  "synthetic_only": True, "competition_fit_performed": False}
    elif args.mode == "fit-fold":
        result = fit_fold(args)
    elif args.mode == "verify-run":
        result = verify_run(args)
    else:
        result = verify_equivalence(args)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
