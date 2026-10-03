"""Bind existing movement fold models before new saved-model inference.

This records hashes and model metadata only. It never predicts, scores, trains,
or uses the leaderboard. The receipt is immutable and independently checkable.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import pyarrow.parquet as pq

import movement_only_expert as movement


def source_files(args: argparse.Namespace) -> dict:
    paths = {
        "prepared_features": args.output_dir / "features.parquet",
        "row_ids": args.output_dir / "row_ids.parquet",
        "features_manifest": args.output_dir / "features_manifest.json",
        "original_protocol": args.output_dir / "protocol.json",
        "original_validation": args.output_dir / "validation.json",
        "public_original_validation": args.public_validation,
        "recorder_source": Path(__file__).name,
    }
    return {role: {"path": str(path), "sha256": movement.sha256(Path(path))}
            for role, path in paths.items()}


def build_receipt(args: argparse.Namespace) -> dict:
    before = source_files(args)
    inputs = movement.prepared_provenance(args)
    manifest = json.loads((args.output_dir / "features_manifest.json").read_text())
    validation = json.loads((args.output_dir / "validation.json").read_text())
    if before["original_validation"]["sha256"] != before["public_original_validation"]["sha256"]:
        raise ValueError("Published original validation differs from local report")
    if (validation.get("both_existing_folds_passed") is not True
            or validation["features_manifest_sha256"] != before["features_manifest"]["sha256"]
            or validation["reference_sha256"] != inputs["frozen_v5_oof"]
            or manifest["baseline_rows_sha256"] != inputs["baseline_training_rows"]
            or manifest["baseline_features_sha256"] != inputs["baseline_features"]
            or manifest["arrival_cache_sha256"] != inputs["released_arrival_cache"]
            or manifest["weather_file_sha256"] != inputs["noaa_weather"]
            or manifest["reference_sha256"] != inputs["frozen_v5_oof"]):
        raise ValueError("Original fold validation or input lineage differs")
    if (pq.ParquetFile(args.output_dir / "features.parquet").metadata.num_rows != manifest["rows"]
            or pq.ParquetFile(args.output_dir / "row_ids.parquet").metadata.num_rows != manifest["rows"]):
        raise ValueError("Prepared cache row counts differ from manifest")
    folds = {}
    for name, months in movement.FOLDS.items():
        model_path = args.output_dir / f"{name}.txt"
        fit_path = args.output_dir / f"{name}_fit.json"
        oof_path = args.output_dir / f"{name}_oof.parquet"
        hashes = {"model_sha256": movement.sha256(model_path),
                  "fit_report_sha256": movement.sha256(fit_path),
                  "original_oof_sha256": movement.sha256(oof_path)}
        fit = json.loads(fit_path.read_text())
        model = lgb.Booster(model_file=str(model_path))
        rounds = int(fit["best_round"])
        if (fit["fold"] != name or fit["heldout_months"] != list(months)
                or fit["features_manifest_sha256"] != before["features_manifest"]["sha256"]
                or fit["feature_count"] != len(manifest["features"])
                or model.feature_name() != manifest["features"]
                or model.num_trees() != rounds or rounds < 1
                or pq.ParquetFile(oof_path).metadata.num_rows != fit["heldout_gate_rows"]):
            raise ValueError(f"{name}: original fit/model/OOF metadata differs")
        folds[name] = {"model_path": str(model_path), "fit_report_path": str(fit_path),
                       "original_oof_path": str(oof_path), **hashes,
                       "heldout_months": list(months), "best_round": rounds,
                       "feature_count": int(fit["feature_count"]),
                       "num_trees": model.num_trees(),
                       "original_heldout_gate_rows": int(fit["heldout_gate_rows"])}
        if (movement.sha256(model_path) != hashes["model_sha256"]
                or movement.sha256(fit_path) != hashes["fit_report_sha256"]
                or movement.sha256(oof_path) != hashes["original_oof_sha256"]):
            raise ValueError(f"{name}: inputs changed during metadata inspection")
        del model
    if source_files(args) != before or movement.prepared_provenance(args) != inputs:
        raise ValueError("Inputs changed during provenance recording")
    return {"schema_version": 1,
            "scope": "Existing fold metadata and hashes only, before new v9b inference",
            "source_files": before, "folds": folds,
            "features": manifest["features"], "categorical": manifest["categorical"],
            "provenance_input_sha256": inputs}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/v6-movement-only"))
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/baseline"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--arrival-cache", type=Path,
                        default=Path("artifacts/v5-arrival-clean/training_arrival_features.parquet"))
    parser.add_argument("--weather-file", type=Path, default=Path("data/external/weather.parquet"))
    parser.add_argument("--v5-oof", type=Path,
                        default=Path("artifacts/v5-ensemble/validation_predictions.parquet"))
    parser.add_argument("--public-validation", type=Path,
                        default=Path("reports/movement_only_validation_v6.json"))
    parser.add_argument("--receipt", type=Path,
                        default=Path("reports/movement_fold_model_provenance_v2.json"))
    args = parser.parse_args()
    report = build_receipt(args)
    if args.receipt.exists():
        if json.loads(args.receipt.read_text()) != report:
            raise ValueError("Existing immutable fold receipt differs")
        action = "verified"
    else:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        with args.receipt.open("x", encoding="utf-8") as destination:
            destination.write(json.dumps(report, indent=2) + "\n")
        action = "recorded"
    print(json.dumps({"action": action, "receipt": str(args.receipt),
                      "folds": list(report["folds"]), "prediction_performed": False}))


if __name__ == "__main__":
    main()
