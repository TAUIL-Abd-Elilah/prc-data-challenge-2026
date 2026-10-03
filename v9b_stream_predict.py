"""Bounded saved-model inference for the frozen v9b valid-AOBT audit.

Run from the repository root. This is an operational alternative to
``v9b_movement_valid_audit.py``'s
full-matrix ``predict-fold`` mode. It reads the same sealed prepared features,
uses the same held-out mask and saved LightGBM model, and writes the exact OOF
schema and manifest accepted by ``v9b.verify_prediction``. It never fits a
competition model, selects a weight, scores labels, or reads ranking data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tempfile
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import movement_only_expert as movement
import movement_prepared_stream_verify as streaming
import v9b_movement_valid_audit as v9b
from solution import _training_files


BATCH_SIZE = 32768
MIN_FREE_GIB = 4.0
RECEIPT = Path("reports/movement_fold_model_provenance_v2.json")
RECEIPT_SHA256 = "cf6a42e5b11b6bded69032ccf6d28ea915f8a58def39499bcfa3c1c429e0d1ae"
OOF_COLUMNS = ["MVT_ID_mvt", "target", "MVT_TIME_UTC_mvt", "expert"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def movement_args(args: argparse.Namespace) -> argparse.Namespace:
    source = argparse.Namespace(**vars(args))
    source.output_dir = args.movement_dir
    return source


def source_paths(args: argparse.Namespace, name: str) -> dict[str, Path]:
    paths = {
        "receipt": args.receipt,
        "stream_source": Path(__file__).resolve(),
        "iterator_source": Path(streaming.__file__).resolve(),
        "movement_source": Path(movement.__file__).resolve(),
        "v9b_source": Path(v9b.__file__).resolve(),
        "prepared_integrity": args.movement_dir / "prepared_integrity.json",
        "prepared_features": args.movement_dir / "features.parquet",
        "row_ids": args.movement_dir / "row_ids.parquet",
        "features_manifest": args.movement_dir / "features_manifest.json",
        "movement_protocol": args.movement_dir / "protocol.json",
        "movement_validation": args.movement_dir / "validation.json",
        "public_movement_validation": Path("reports/movement_only_validation_v6.json"),
        "recorder_source": Path("record_movement_fold_provenance.py"),
        "v9b_protocol": args.output_dir / "protocol.json",
        "frozen_v9_protocol": args.v9_protocol,
        "v7_protocol": args.v7_dir / "protocol.json",
        "v7_oof": args.v7_dir / "validation_predictions.parquet",
        "training_rows": args.cache_dir / "training_rows.parquet",
        "baseline_features": args.cache_dir / "features.parquet",
        "arrival_cache": args.arrival_cache,
        "weather": args.weather_file,
        "v5_oof": args.v5_oof,
        "saved_model": args.movement_dir / f"{name}.txt",
        "fit_report": args.movement_dir / f"{name}_fit.json",
        "original_gate_oof": args.movement_dir / f"{name}_oof.parquet",
    }
    paths.update({f"raw_{path.name}": path
                  for path in _training_files(args.data_dir)})
    return paths


def snapshot(paths: dict[str, Path]) -> dict[str, str]:
    return {role: sha256(path) for role, path in paths.items()}


def check_receipt(args: argparse.Namespace, receipt: dict, name: str,
                  months: tuple[int, int], source_hashes: dict[str, str],
                  manifest: dict, model: lgb.Booster) -> int:
    """Require the separately published lineage for both original folds."""
    if (receipt.get("schema_version") != 1
            or receipt.get("features") != manifest["features"]
            or receipt.get("categorical") != manifest["categorical"]
            or set(receipt.get("folds", {})) != set(v9b.FOLDS)):
        raise ValueError("Pinned original movement fold receipt schema differs")
    expected_receipt_paths = {
        "prepared_features": args.movement_dir / "features.parquet",
        "row_ids": args.movement_dir / "row_ids.parquet",
        "features_manifest": args.movement_dir / "features_manifest.json",
        "original_protocol": args.movement_dir / "protocol.json",
        "original_validation": args.movement_dir / "validation.json",
        "public_original_validation": Path("reports/movement_only_validation_v6.json"),
        "recorder_source": Path("record_movement_fold_provenance.py"),
    }
    if set(receipt["source_files"]) != set(expected_receipt_paths):
        raise ValueError("Pinned receipt source roles differ")
    for role, record in receipt["source_files"].items():
        path = Path(record["path"])
        if (path.resolve() != expected_receipt_paths[role].resolve()
                or record["sha256"] != sha256(path)):
            raise ValueError(f"Pinned receipt source changed: {role}")
    if (receipt.get("provenance_input_sha256") !=
            movement.prepared_provenance(movement_args(args))):
        raise ValueError("Pinned original movement source provenance differs")
    validation = read_json(args.movement_dir / "validation.json")
    if validation.get("both_existing_folds_passed") is not True:
        raise ValueError("Original movement complementary folds did not pass")
    for fold, fold_months in v9b.FOLDS.items():
        entry = receipt["folds"][fold]
        model_path = args.movement_dir / f"{fold}.txt"
        fit_path = args.movement_dir / f"{fold}_fit.json"
        original_oof = args.movement_dir / f"{fold}_oof.parquet"
        fit = read_json(fit_path)
        if (Path(entry["model_path"]).resolve() != model_path.resolve()
                or Path(entry["fit_report_path"]).resolve() != fit_path.resolve()
                or Path(entry["original_oof_path"]).resolve() != original_oof.resolve()
                or entry["model_sha256"] != sha256(model_path)
                or entry["fit_report_sha256"] != sha256(fit_path)
                or entry["original_oof_sha256"] != sha256(original_oof)
                or entry["heldout_months"] != list(fold_months)
                or fit["heldout_months"] != list(fold_months)
                or int(entry["best_round"]) != int(fit["best_round"])
                or int(entry["feature_count"]) != len(manifest["features"])
                or int(fit["feature_count"]) != len(manifest["features"])
                or int(entry["original_heldout_gate_rows"]) !=
                   int(fit["heldout_gate_rows"])
                or pq.ParquetFile(original_oof).metadata.num_rows !=
                   int(entry["original_heldout_gate_rows"])
                or fit["features_manifest_sha256"] !=
                   source_hashes["features_manifest"]):
            raise ValueError(f"Pinned original movement fit differs: {fold}")
    chosen = receipt["folds"][name]
    rounds = int(chosen["best_round"])
    if (chosen["heldout_months"] != list(months)
            or int(chosen["num_trees"]) != model.num_trees()
            or not 1 <= rounds <= model.num_trees()
            or model.feature_name() != manifest["features"]
            or not isinstance(model.pandas_categorical, list)
            or len(model.pandas_categorical) != len(manifest["categorical"])):
        raise ValueError("Saved LightGBM model schema, categories, or rounds differ")
    return rounds


def check_batch_categories(frame: pd.DataFrame, manifest: dict,
                           model: lgb.Booster) -> None:
    if list(frame) != manifest["features"]:
        raise ValueError("Prepared streaming feature names/order differ")
    for name, levels in zip(manifest["categorical"],
                            model.pandas_categorical, strict=True):
        series = frame[name]
        if (not isinstance(series.dtype, pd.CategoricalDtype)
                or list(series.cat.categories) != list(levels)):
            raise ValueError(f"Prepared streaming category differs: {name}")


def predict_fold(args: argparse.Namespace) -> None:
    if args.fold is None:
        raise ValueError("predict-fold requires --fold")
    if Path.cwd().resolve() != Path(__file__).resolve().parent:
        raise ValueError("Run streaming inference from the repository root")
    movement.require_memory(MIN_FREE_GIB)
    name, months = args.fold, v9b.FOLDS[args.fold]
    output_path = args.output_dir / f"{name}_valid_oof.parquet"
    manifest_path = args.output_dir / f"{name}_prediction_manifest.json"
    audit_path = args.output_dir / f"{name}_stream_audit.json"
    temp_path = args.output_dir / f"{name}_valid_oof.tmp.parquet"
    if any(path.exists() for path in (output_path, manifest_path,
                                      audit_path, temp_path)):
        raise FileExistsError("Stream OOF or partial output exists; no overwrite")
    if not (args.output_dir / "protocol.json").exists():
        raise FileNotFoundError("Frozen v9b protocol is required before inference")
    if (not re.fullmatch(r"[0-9a-fA-F]{64}", args.receipt_sha256)
            or sha256(args.receipt) != args.receipt_sha256.lower()):
        raise ValueError("Published original-model provenance receipt changed")
    sources = source_paths(args, name)
    before = snapshot(sources)
    move_args = movement_args(args)
    seal = movement.verify_prepared_integrity(move_args)
    v9b.protocol(args)
    prepared_manifest = read_json(args.movement_dir / "features_manifest.json")
    if seal["rows"] != prepared_manifest["rows"]:
        raise ValueError("Prepared seal count differs from manifest")
    model_path = args.movement_dir / f"{name}.txt"
    model = lgb.Booster(model_file=str(model_path))
    if sha256(model_path) != before["saved_model"]:
        raise ValueError("Saved movement model changed during LightGBM load")
    receipt = read_json(args.receipt)
    rounds = check_receipt(args, receipt, name, months, before,
                           prepared_manifest, model)
    writer: pq.ParquetWriter | None = None
    n_seen = 0
    n_selected = 0
    try:
        for rows, features in streaming.iter_prepared_training_batches(
                args.movement_dir, args.cache_dir, BATCH_SIZE):
            movement.require_memory(1.0)
            check_batch_categories(features, prepared_manifest, model)
            target = rows.target.to_numpy(dtype=float)
            proxy = rows.proxy.to_numpy(dtype=float)
            valid = (rows.month.isin(months).to_numpy(dtype=bool)
                     & np.isfinite(target) & np.isfinite(proxy)
                     & (proxy >= 0) & (proxy <= 7200))
            n_seen += len(rows)
            if not valid.any():
                continue
            prediction = np.maximum(model.predict(
                features.loc[valid], num_iteration=rounds, num_threads=3), 0)
            if (len(prediction) != int(valid.sum())
                    or not np.isfinite(prediction).all()):
                raise ValueError("Saved movement model has incomplete finite coverage")
            part = pd.DataFrame({
                "MVT_ID_mvt": rows.loc[valid, "MVT_ID_mvt"].to_numpy(),
                "target": target[valid],
                "MVT_TIME_UTC_mvt": rows.loc[valid, "time"].to_numpy(),
                "expert": prediction,
            })
            if list(part) != OOF_COLUMNS:
                raise ValueError("Streaming OOF column order changed")
            table = pa.Table.from_pandas(part, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(temp_path, table.schema,
                                          compression="snappy")
            elif not table.schema.equals(writer.schema, check_metadata=True):
                raise ValueError("Streaming OOF batch schema differs; no coercion allowed")
            writer.write_table(table)
            n_selected += len(part)
        if writer is None or n_selected < 1 or n_seen != prepared_manifest["rows"]:
            raise ValueError("Streaming prepared rows or held-out gate are incomplete")
    finally:
        if writer is not None:
            writer.close()
    if (pq.ParquetFile(temp_path).metadata.num_rows != n_selected
            or list(pq.ParquetFile(temp_path).schema_arrow.names) != OOF_COLUMNS):
        raise ValueError("Streaming OOF Parquet count/schema differs")
    after = snapshot(sources)
    movement.verify_prepared_integrity(move_args)
    if before != after:
        raise ValueError("Streaming inference source changed while reading")
    temp_path.replace(output_path)
    producer_manifest = {
        "months": list(months), "n_valid_finite": n_selected,
        "model_source": "original_movement", "model_path": str(model_path),
        "model_sha256": before["saved_model"],
        "fit_report_sha256": before["fit_report"],
        "features_manifest_sha256": before["features_manifest"],
        "prepared_features_sha256": before["prepared_features"],
        "arrival_cache_sha256": before["arrival_cache"],
        "v5_oof_sha256": before["v5_oof"],
        "output_sha256": sha256(output_path),
    }
    with manifest_path.open("x", encoding="utf-8") as out:
        out.write(json.dumps(producer_manifest, indent=2) + "\n")
    expected = pd.read_parquet(args.v7_dir / "validation_predictions.parquet",
                               columns=["MVT_ID_mvt", "fold", "a_valid"])
    expected_ids = expected.loc[expected.fold.eq(name) & expected.a_valid,
                                "MVT_ID_mvt"]
    v9b.verify_prediction(args, name, expected_ids, months)
    audit = {"purpose": "Bounded saved-model inference; no fit or scoring",
             "fold": name, "batch_rows": BATCH_SIZE,
             "source_sha256_before": before,
             "source_sha256_after": after,
             "pinned_receipt_sha256": before["receipt"],
             "prepared_seal_sha256": before["prepared_integrity"],
             "model_rounds": rounds, "rows_scanned": n_seen,
             "rows_selected": n_selected,
             "producer_manifest_sha256": sha256(manifest_path),
             "output_sha256": sha256(output_path),
             "downstream_verify_prediction_passed": True}
    with audit_path.open("x", encoding="utf-8") as out:
        out.write(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({"fold": name, "rows": n_selected,
                      "output": str(output_path),
                      "manifest": str(manifest_path),
                      "verified": True}, indent=2), flush=True)


def parity_test() -> None:
    """Check categorical coding and exact batch prediction on synthetic data."""
    categories = ["C", "A", "D", "B"]
    dtype = pd.CategoricalDtype(categories=categories, ordered=False)
    values = (["A", "B", "C", "A", None, "D", "B", "C"] * 8)
    frame = pd.DataFrame({
        "category": pd.Series(values, dtype=dtype),
        "number": np.arange(len(values), dtype=np.float32) % 11,
    })
    y = np.array([{"A": 1.0, "B": 5.0, "C": 2.5,
                   "D": 8.0}.get(value, 4.0) for value in values])
    y += frame.number.to_numpy(dtype=float) * 0.3
    model = lgb.train(
        {"objective": "regression", "metric": "rmse", "num_leaves": 7,
         "min_data_in_leaf": 1, "learning_rate": 0.15,
         "deterministic": True, "force_col_wise": True,
         "num_threads": 1, "verbosity": -1, "seed": 2026},
        lgb.Dataset(frame, label=y, categorical_feature=["category"]),
        num_boost_round=12)
    full = model.predict(frame, num_iteration=12, num_threads=3)
    if len(np.unique(full)) < 2:
        raise ValueError("Synthetic parity model did not use its predictors")
    with tempfile.TemporaryDirectory(prefix="v9b-stream-parity-") as tmp:
        path = Path(tmp) / "synthetic.parquet"
        frame.to_parquet(path, index=False, row_group_size=5)
        dtypes = streaming.load_global_category_dtypes(path, ["category"])
        cursor = streaming.ParquetFrameCursor(
            path, ["category", "number"], batch_size=3,
            categorical_dtypes=dtypes)
        batches = []
        try:
            for n in (2, 5, 1, 7, 3, 11, 35):
                part = cursor.take(n)
                if list(part.category.cat.categories) != categories:
                    raise ValueError("Synthetic batch lost global category order")
                batches.append(model.predict(part, num_iteration=12,
                                             num_threads=3))
            cursor.finish()
        finally:
            cursor.parquet.close()
    batched = np.concatenate(batches)
    if not np.array_equal(full, batched):
        raise ValueError("Batched categorical LightGBM inference differs")
    print(json.dumps({"synthetic_rows": len(frame), "batch_sizes":
                      [2, 5, 1, 7, 3, 11, 35],
                      "categorical_levels": categories,
                      "exact_prediction_parity": True}, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("parity-test", "predict-fold"),
                        default="parity-test")
    parser.add_argument("--fold", choices=tuple(v9b.FOLDS))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--cache-dir", type=Path,
                        default=Path("artifacts/baseline"))
    parser.add_argument("--movement-dir", type=Path,
                        default=Path("artifacts/v6-movement-only"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("artifacts/v9b-movement-valid"))
    parser.add_argument("--v9-protocol", type=Path,
                        default=Path("artifacts/v9-movement-valid/protocol.json"))
    parser.add_argument("--v7-dir", type=Path,
                        default=Path("artifacts/v7-runway-traffic"))
    parser.add_argument("--arrival-cache", type=Path,
                        default=Path("artifacts/v5-arrival-clean/training_arrival_features.parquet"))
    parser.add_argument("--weather-file", type=Path,
                        default=Path("data/external/weather.parquet"))
    parser.add_argument("--v5-oof", type=Path,
                        default=Path("artifacts/v5-ensemble/validation_predictions.parquet"))
    parser.add_argument("--receipt", type=Path, default=RECEIPT)
    parser.add_argument("--receipt-sha256", default=RECEIPT_SHA256,
                        help="SHA-256 of the independently frozen own-fit receipt; "
                             "supply with --receipt for a relocated frozen receipt")
    args = parser.parse_args()
    if args.mode == "parity-test":
        parity_test()
    else:
        predict_fold(args)


if __name__ == "__main__":
    main()
