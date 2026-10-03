"""Sealed inference adapter for the already validated v12 geometry model.

The original v12 ranking seal and its full nested scientific verification are
replayed exactly once by ``prepare``. A separately published adapter protocol
then lets ``predict`` verify every named source byte before and after loading
ranking values, without repeating the original bootstrap/gate computation.
This module changes no model, feature, route, weight or competition artifact.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from catboost import CatBoostRegressor

import v12_geometry_final as original
import v12_runway_geometry_expert as geometry_model


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "artifacts/v12-sealed-inference"
ORIGINAL_SOURCE_SHA256 = "73ce757e080e7d54fc3e5c1b2fc8991f37e073c6ec01dc71a514941a89c80ffe"
ORIGINAL_RANK_SEAL_SHA256 = "d5b19eeb7f139e98fdb301e29e946778ac9cde482c53c4c87a7d60998ffc8804"
ORIGINAL_MODEL_REPORT_SHA256 = "65c54807b4439a3edec51049fc8ef34d273c7037f66d6981fc70ea7b404564c9"
ORIGINAL_MODEL_SHA256 = "383a15c202d1061315673b2c40702108056ff2e7daec87f1d921281aabb53aec"
FIXED_WEIGHT = 0.25
EXPECTED_ROWS = 344_841
EXPECTED_VALID = 339_377
EXPECTED_OUTSIDE = 5_464
EXPECTED_MISSING = 4_907
EXPECTED_FEATURES = 200
EXPECTED_CATS = 24
EXPECTED_TREES = 9_998
EXPECTED_ELIGIBLE_TRAIN = 2_061_428
SCHEMA_VERSION = 1
COLUMNS = ["MVT_ID_mvt", "TAXITIME_SEC_mvt"]


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    before = sha(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if sha(path) != before or not isinstance(value, dict):
        raise ValueError(f"JSON source changed during read: {path}")
    return value


def publish_json(path: Path, value: dict) -> str:
    def writer(stage: Path) -> None:
        with stage.open("w", encoding="utf-8", newline="\n") as target:
            json.dump(value, target, indent=2, sort_keys=True, allow_nan=False)
            target.write("\n")
            target.flush()
            os.fsync(target.fileno())
    return original.publish_new(path, writer)


def require_published_source(expected: str) -> None:
    if (not isinstance(expected, str) or len(expected) != 64
            or expected.lower() != sha(Path(__file__).resolve())
            or sha(Path(original.__file__).resolve()) != ORIGINAL_SOURCE_SHA256):
        raise ValueError("Require published adapter and original v12 source SHAs")


def require_memory() -> None:
    original.require_memory()  # Hard 10 GiB floor; no caller override.


def protocol_path() -> Path:
    return OUT / "protocol.json"


def output_paths() -> dict[str, Path]:
    return {"expert": OUT / "ranking_expert.parquet",
            "predictions": OUT / "predictions.parquet",
            "manifest": OUT / "manifest.json"}


def named_paths() -> dict[str, Path]:
    """Original full-fit and ranking closure, including all transitive helpers."""
    paths = {f"final_{role}": Path(path)
             for role, path in original.final_fit_input_paths().items()}
    paths.update({f"ranking_{role}": Path(path)
                  for role, path in original.ranking_input_paths().items()})
    paths.update({
        "adapter_source": Path(__file__).resolve(),
        "original_source": Path(original.__file__).resolve(),
        "original_ranking_seal": original.ranking_seal_path(),
        "original_full_model": original.model_path(),
        "original_full_model_report": original.model_report_path(),
        "original_final_protocol": original.final_protocol_path(),
        "original_guard_terminal": original.GUARD_DIR / "terminal.json",
        "original_guard_protocol": original.guard_protocol_path(),
        "sealed_v9_reference": original.V9_SUBMISSION,
        "sealed_v8_current": ROOT / "artifacts/current-candidate/predictions.parquet",
        "sealed_v7_prior": ROOT / "artifacts/v7-runway-traffic/predictions.parquet",
        "sealed_v9_final_predictions":
            original.args_for_v12().v9_final_dir / "predictions.parquet",
        "ranking_template": original.TEMPLATE,
    })
    if len(paths) != len(set(paths)):
        raise ValueError("Duplicate source role in v12 adapter inventory")
    return paths


def snapshot(paths: dict[str, Path]) -> dict[str, dict[str, str]]:
    """Hash each byte stream once, retaining every named source role/path."""
    by_path: dict[Path, str] = {}
    result: dict[str, dict[str, str]] = {}
    for role, path in sorted(paths.items()):
        resolved = Path(path).resolve()
        if resolved not in by_path:
            by_path[resolved] = sha(resolved)
        result[role] = {"path": str(resolved), "sha256": by_path[resolved]}
    return result


def check_snapshot(expected: dict[str, dict[str, str]]) -> None:
    actual = snapshot(named_paths())
    if actual != expected:
        roles = [role for role in set(actual) | set(expected)
                 if actual.get(role) != expected.get(role)]
        raise ValueError(f"Sealed v12 adapter bytes/path changed: {roles[:8]}")


def model_metadata(report: dict) -> tuple[list[dict], list[int]]:
    schema = report.get("feature_schema")
    if (not isinstance(schema, list) or len(schema) != EXPECTED_FEATURES
            or len({field.get("name") for field in schema}) != EXPECTED_FEATURES
            or any(field.get("name") in original.FORBIDDEN for field in schema)):
        raise ValueError("Full v12 model feature schema differs")
    cats = [i for i, field in enumerate(schema) if field.get("dtype") == "category"]
    if (len(cats) != EXPECTED_CATS
            or report.get("categorical_indices") != cats
            or report.get("iterations") != EXPECTED_TREES
            or report.get("eligible_training_rows") != EXPECTED_ELIGIBLE_TRAIN
            or report.get("training_rows") != original.EXPECTED_TRAIN
            or report.get("selected_route") != "v12"
            or report.get("selected_weight") != FIXED_WEIGHT
            or report.get("fixed_catboost_params") != {
                **geometry_model.EXPECTED_CATBOOST_PARAMS,
                "iterations": EXPECTED_TREES}
            or report.get("model_sha256") != sha(original.model_path())
            or report.get("model_sha256") != ORIGINAL_MODEL_SHA256
            or report.get("final_protocol_sha256") != sha(original.final_protocol_path())
            or report.get("guard_terminal_sha256")
            != sha(original.GUARD_DIR / "terminal.json")
            or report.get("ranking_prediction_created") is not False):
        raise ValueError("Full v12 model report/parameters/source receipt changed")
    return schema, cats


def load_verified_model(report: dict) -> tuple[CatBoostRegressor, list[dict], list[int]]:
    schema, cats = model_metadata(report)
    model = CatBoostRegressor()
    model.load_model(str(original.model_path()))
    geometry_model.verify_saved_params(model.get_all_params())
    if (int(model.tree_count_) != EXPECTED_TREES
            or list(model.feature_names_) != [field["name"] for field in schema]
            or list(model.get_cat_feature_indices()) != cats):
        raise ValueError("Saved v12 model tree/feature/category metadata changed")
    return model, schema, cats


def prepare(expected_adapter_sha256: str,
            published_original_seal_sha256: str,
            published_original_model_report_sha256: str) -> dict:
    """Replay the original nested ranking gate once, then freeze its closure."""
    require_published_source(expected_adapter_sha256)
    require_memory()
    if protocol_path().exists() or any(path.exists() for path in output_paths().values()):
        raise FileExistsError("v12 adapter already prepared or has partial inference output")
    if (not isinstance(published_original_seal_sha256, str)
            or published_original_seal_sha256.lower() != ORIGINAL_RANK_SEAL_SHA256
            or sha(original.ranking_seal_path()) != ORIGINAL_RANK_SEAL_SHA256
            or not isinstance(published_original_model_report_sha256, str)
            or published_original_model_report_sha256.lower()
            != ORIGINAL_MODEL_REPORT_SHA256
            or sha(original.model_report_path()) != ORIGINAL_MODEL_REPORT_SHA256):
        raise ValueError("Original v12 ranking seal and model report must be published")
    before = snapshot(named_paths())
    verified = original.check_ranking_seal(ORIGINAL_SOURCE_SHA256)  # Exactly once.
    if (verified.get("schema_version") != 1
            or verified.get("selected_route") != "v12"
            or verified.get("selected_weight") != FIXED_WEIGHT
            or verified.get("status") != "sealed_before_any_2026_feature_value_read"
            or verified.get("source_sha256") != ORIGINAL_SOURCE_SHA256
            or verified.get("final_model_sha256") != sha(original.model_path())
            or verified.get("ranking_input_sha256") != {
                key: before[f"ranking_{key}"]["sha256"]
                for key in original.ranking_input_paths()}):
        raise ValueError("Original verified v12 ranking decision differs")
    report = read_json(original.model_report_path())
    _, schema, cats = load_verified_model(report)
    if (before != snapshot(named_paths())
            or before["original_ranking_seal"]["sha256"]
            != published_original_seal_sha256.lower()
            or before["original_full_model_report"]["sha256"]
            != published_original_model_report_sha256.lower()):
        raise ValueError("Original verified closure changed during adapter prepare")
    protocol = {
        "schema_version": SCHEMA_VERSION,
        "status": "original_v12_nested_ranking_gate_replayed_once_before_inference",
        "original_verified_passed": True,
        "original_verified_decision": "v12_fixed_geometry_weight_0.25",
        "original_verified_ranking_seal": verified,
        "original_ranking_seal_sha256": published_original_seal_sha256.lower(),
        "original_model_report_sha256": published_original_model_report_sha256.lower(),
        "original_source_sha256": ORIGINAL_SOURCE_SHA256,
        "adapter_source_sha256": expected_adapter_sha256.lower(),
        "fixed_weight": FIXED_WEIGHT,
        "model_sha256": before["original_full_model"]["sha256"],
        "model_trees": EXPECTED_TREES,
        "feature_schema": schema,
        "categorical_feature_indices": cats,
        "named_input_files": before,
        "ranking_values_read": False,
        "model_training_performed": False,
        "scientific_gates_replayed_on_prepare_only": True,
    }
    publish_json(protocol_path(), protocol)
    check_snapshot(before)
    return {"protocol_sha256": sha(protocol_path()),
            "original_nested_ranking_gate_replayed": True,
            "named_input_roles": len(before),
            "model_sha256": protocol["model_sha256"]}


def require_prepared(expected_adapter_sha256: str,
                     published_protocol_sha256: str) -> dict:
    require_published_source(expected_adapter_sha256)
    if (not isinstance(published_protocol_sha256, str)
            or len(published_protocol_sha256) != 64
            or sha(protocol_path()) != published_protocol_sha256.lower()):
        raise ValueError("Adapter protocol must be published before inference")
    frozen = read_json(protocol_path())
    if (frozen.get("schema_version") != SCHEMA_VERSION
            or frozen.get("status")
            != "original_v12_nested_ranking_gate_replayed_once_before_inference"
            or frozen.get("original_verified_passed") is not True
            or frozen.get("original_verified_decision") != "v12_fixed_geometry_weight_0.25"
            or frozen.get("original_verified_ranking_seal")
            != read_json(original.ranking_seal_path())
            or frozen.get("original_source_sha256") != ORIGINAL_SOURCE_SHA256
            or frozen.get("original_ranking_seal_sha256") != ORIGINAL_RANK_SEAL_SHA256
            or frozen.get("original_model_report_sha256") != ORIGINAL_MODEL_REPORT_SHA256
            or frozen.get("adapter_source_sha256") != expected_adapter_sha256.lower()
            or frozen.get("fixed_weight") != FIXED_WEIGHT
            or frozen.get("original_ranking_seal_sha256")
            != frozen["named_input_files"]["original_ranking_seal"]["sha256"]
            or frozen.get("original_model_report_sha256")
            != frozen["named_input_files"]["original_full_model_report"]["sha256"]
            or frozen.get("model_sha256")
            != frozen["named_input_files"]["original_full_model"]["sha256"]
            or frozen.get("model_sha256") != ORIGINAL_MODEL_SHA256
            or frozen.get("model_trees") != EXPECTED_TREES
            or len(frozen.get("feature_schema", [])) != EXPECTED_FEATURES
            or len(frozen.get("categorical_feature_indices", [])) != EXPECTED_CATS
            or frozen.get("ranking_values_read") is not False
            or frozen.get("model_training_performed") is not False
            or frozen.get("scientific_gates_replayed_on_prepare_only") is not True):
        raise ValueError("Published v12 adapter protocol changed")
    check_snapshot(frozen["named_input_files"])
    return frozen


def exact_reference_inputs(rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray,
                                                        np.ndarray, np.ndarray]:
    """Only ranking IDs/predictions and proxy; never read ranking target values."""
    if pq.ParquetFile(original.TEMPLATE).schema_arrow.names != COLUMNS:
        raise ValueError("Submission template column schema differs")
    template = pd.read_parquet(original.TEMPLATE, columns=["MVT_ID_mvt"])
    reference = pd.read_parquet(original.V9_SUBMISSION, columns=COLUMNS)
    v9_saved = pd.read_parquet(original.args_for_v12().v9_final_dir /
                               "predictions.parquet", columns=COLUMNS)
    v8 = pd.read_parquet(ROOT / "artifacts/current-candidate/predictions.parquet",
                         columns=COLUMNS)
    v7 = pd.read_parquet(ROOT / "artifacts/v7-runway-traffic/predictions.parquet",
                         columns=COLUMNS)
    frames = (template, reference, v9_saved, v8, v7, rows)
    if (len(template) != EXPECTED_ROWS
            or any(len(frame) != EXPECTED_ROWS or frame.MVT_ID_mvt.isna().any()
                   or frame.MVT_ID_mvt.duplicated().any()
                   or not np.array_equal(frame.MVT_ID_mvt.to_numpy(),
                                         template.MVT_ID_mvt.to_numpy())
                   for frame in frames)):
        raise ValueError("Exact 344841-row template/reference/feature ID order changed")
    if (any(pq.ParquetFile(path).schema_arrow.names != COLUMNS for path in (
            original.V9_SUBMISSION,
            original.args_for_v12().v9_final_dir / "predictions.parquet",
            ROOT / "artifacts/current-candidate/predictions.parquet",
            ROOT / "artifacts/v7-runway-traffic/predictions.parquet"))
            or sha(original.V9_SUBMISSION) != original.V9_SHA256):
        raise ValueError("Sealed current ranking reference schema or SHA changed")
    current = reference.TAXITIME_SEC_mvt.to_numpy(dtype=float, copy=True)
    v9_values = v9_saved.TAXITIME_SEC_mvt.to_numpy(dtype=float)
    v8_values = v8.TAXITIME_SEC_mvt.to_numpy(dtype=float)
    v7_values = v7.TAXITIME_SEC_mvt.to_numpy(dtype=float)
    proxy = rows.proxy.to_numpy(dtype=float)
    valid = np.isfinite(proxy) & (proxy >= 0) & (proxy <= 7200)
    if (int(valid.sum()) != EXPECTED_VALID
            or int((~valid).sum()) != EXPECTED_OUTSIDE
            or not np.isfinite(current).all() or (current < 0).any()
            or not np.array_equal(current, v9_values)
            or not np.array_equal(current[~valid], v8_values[~valid])
            or int(np.count_nonzero(v8_values[~valid] != v7_values[~valid]))
            != EXPECTED_MISSING):
        raise ValueError("Fixed valid gate or 4907-row v8 missing policy changed")
    ids = template.MVT_ID_mvt.to_numpy(copy=True)
    del template, reference, v9_saved, v8, v7, frames
    return ids, current, valid, proxy


def predict(expected_adapter_sha256: str,
            published_protocol_sha256: str) -> dict:
    require_published_source(expected_adapter_sha256)
    require_memory()
    outputs = output_paths()
    if any(path.exists() for path in outputs.values()):
        raise FileExistsError("v12 sealed-inference output already exists or is partial")
    frozen = require_prepared(expected_adapter_sha256,
                              published_protocol_sha256)
    protocol_sha = sha(protocol_path())
    before = snapshot(named_paths())  # Fresh hashes, before any ranking value read.
    if before != frozen["named_input_files"]:
        raise ValueError("v12 adapter input bytes changed before inference")
    report = read_json(original.model_report_path())
    model, feature_schema, cats = load_verified_model(report)
    if (feature_schema != frozen["feature_schema"]
            or cats != frozen["categorical_feature_indices"]
            or report["model_sha256"] != frozen["model_sha256"]):
        raise ValueError("Prepared v12 model schema differs from actual saved model")
    rows, features = original.load_ranking_features()  # Pure loader; no nested gates.
    if (original.schema(features) != feature_schema
            or rows.MVT_ID_mvt.isna().any() or rows.MVT_ID_mvt.duplicated().any()):
        raise ValueError("Actual ranking feature order/dtypes differ from original model")
    ids, current, valid, proxy = exact_reference_inputs(rows)
    raw = np.full(EXPECTED_ROWS, np.nan, dtype=float)
    raw[valid] = proxy[valid] + model.predict(features.loc[valid], thread_count=2)
    selected = original.fixed_ranking_blend(current, raw, valid, FIXED_WEIGHT)
    if (not np.array_equal(selected[~valid], current[~valid])
            or not np.isfinite(selected).all() or (selected < 0).any()):
        raise ValueError("Fixed v12 blend altered outside-valid rows or produced invalid output")
    expert = pd.DataFrame({"MVT_ID_mvt": ids.copy(), "a_valid": valid.copy(),
                           "raw_expert": raw.copy()})
    prediction = pd.DataFrame({"MVT_ID_mvt": ids.copy(),
                               "TAXITIME_SEC_mvt": selected.copy()})
    del model, rows, features
    gc.collect()
    if sha(protocol_path()) != protocol_sha:
        raise ValueError("Published adapter protocol changed during inference")
    check_snapshot(before)
    expert_sha = original.publish_new(
        outputs["expert"], lambda path: expert.to_parquet(path, index=False))
    pred_sha = original.publish_new(
        outputs["predictions"], lambda path: prediction.to_parquet(path, index=False))
    expert_back = pd.read_parquet(outputs["expert"])
    pred_back = pd.read_parquet(outputs["predictions"])
    if (list(expert_back) != ["MVT_ID_mvt", "a_valid", "raw_expert"]
            or list(pred_back) != COLUMNS
            or len(expert_back) != EXPECTED_ROWS or len(pred_back) != EXPECTED_ROWS
            or not np.array_equal(expert_back.MVT_ID_mvt.to_numpy(), ids)
            or not np.array_equal(pred_back.MVT_ID_mvt.to_numpy(), ids)
            or not np.array_equal(expert_back.a_valid.to_numpy(dtype=bool), valid)
            or not np.array_equal(expert_back.raw_expert.to_numpy(dtype=float), raw,
                                  equal_nan=True)
            or not np.array_equal(pred_back.TAXITIME_SEC_mvt.to_numpy(dtype=float), selected)
            or not np.array_equal(selected[~valid], current[~valid])
            or not np.isfinite(selected).all() or (selected < 0).any()):
        raise ValueError("Published v12 adapter output readback/route differs")
    del expert, prediction, expert_back, pred_back
    gc.collect()
    if sha(protocol_path()) != protocol_sha:
        raise ValueError("Published adapter protocol changed before manifest")
    check_snapshot(before)  # After feature/prediction frames are released.
    result = {
        "schema_version": SCHEMA_VERSION,
        "route": "sealed_v12_geometry_fixed_0.25",
        "fixed_weight": FIXED_WEIGHT,
        "rows": EXPECTED_ROWS,
        "valid_aobt_rows": EXPECTED_VALID,
        "outside_valid_rows": EXPECTED_OUTSIDE,
        "missing_clock_preserved_rows": EXPECTED_MISSING,
        "all_template_ids_in_order": True,
        "outside_valid_exactly_unchanged": True,
        "finite_nonnegative_predictions": True,
        "original_ranking_seal_sha256": frozen["original_ranking_seal_sha256"],
        "adapter_protocol_sha256": sha(protocol_path()),
        "adapter_source_sha256": expected_adapter_sha256.lower(),
        "model_sha256": frozen["model_sha256"],
        "model_report_sha256": frozen["original_model_report_sha256"],
        "feature_count": EXPECTED_FEATURES,
        "categorical_count": EXPECTED_CATS,
        "model_trees": EXPECTED_TREES,
        "named_input_files": before,
        "ranking_expert_sha256": expert_sha,
        "predictions_sha256": pred_sha,
        "predictions_bytes": outputs["predictions"].stat().st_size,
        "scientific_gates_replayed_during_predict": False,
        "model_training_performed": False,
        "uploaded": False,
    }
    publish_json(outputs["manifest"], result)
    if sha(protocol_path()) != protocol_sha:
        raise ValueError("Published adapter protocol changed after output")
    check_snapshot(before)
    return {"rows": EXPECTED_ROWS, "valid_aobt_rows": EXPECTED_VALID,
            "model_sha256": result["model_sha256"],
            "predictions_sha256": pred_sha,
            "manifest_sha256": sha(outputs["manifest"])}


def synthetic() -> dict:
    old = np.array([10., 20., 30., 40.])
    raw = np.array([8., np.nan, -10., np.nan])
    valid = np.array([True, False, True, False])
    selected = original.fixed_ranking_blend(old, raw, valid, FIXED_WEIGHT)
    if not np.array_equal(selected, np.array([9.5, 20., 20., 40.])):
        raise AssertionError("Fixed 0.25 v12 blend or outside-row identity differs")
    with tempfile.TemporaryDirectory(prefix="v12-sealed-small-") as folder:
        base = Path(folder)
        first = base / "one"
        first.write_bytes(b"a")
        frozen = snapshot({"one": first, "alias": first})
        if frozen["one"] != frozen["alias"]:
            raise AssertionError("Named source aliases lost exact byte identity")
        first.write_bytes(b"b")
        if snapshot({"one": first, "alias": first}) == frozen:
            raise AssertionError("Source mutation was not detected")
    return {"passed": True, "models_fitted": 0,
            "ranking_values_read": 0,
            "fixed_weight": FIXED_WEIGHT,
            "outside_gate_unchanged": True,
            "source_tamper_detected": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("synthetic", "prepare", "predict"))
    parser.add_argument("--published-source-sha256")
    parser.add_argument("--published-original-ranking-seal-sha256")
    parser.add_argument("--published-original-model-report-sha256")
    parser.add_argument("--published-protocol-sha256")
    args = parser.parse_args()
    if args.mode == "synthetic":
        result = synthetic()
    elif args.mode == "prepare":
        result = prepare(args.published_source_sha256,
                         args.published_original_ranking_seal_sha256,
                         args.published_original_model_report_sha256)
    else:
        result = predict(args.published_source_sha256,
                         args.published_protocol_sha256)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
