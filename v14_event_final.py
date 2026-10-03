"""Conditional full fit and ranking extension for the frozen v14 event expert.

This module can fit only after the independent v14 original, April/October,
and February/August gates have all passed and replayed. It never uploads or
versions a submission. Ranking inputs are sealed by the separate label-free
event builder before any ranking value is read here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

import event_sequence_features_v14 as builder
import movement_only_expert as movement
import v14_event_sequence_expert as trainer


ROOT = Path(__file__).resolve().parent
SPEC_SHA256 = "26782bcd1c46662bb705525fa5a72c4d9eb1d57616bbca1ee42edcff1a50aabb"
TRAINER_SHA256 = "c3d369614eb9272001e5a7e0277741248cbf206658d88ea942f6adff79154d26"
BUILDER_SHA256 = "3719fd5b0048e0590f77f094bf1bc3829982e9fe3db4115688dc196cec4405e2"
V8_RANK_SHA256 = "8fc6519610a573dd77a4b5ca18f49ab816a564754db13d26d91f26b53d04b9e5"
EXPECTED_TRAIN = 2_085_047
EXPECTED_RANK = 344_841
EXPECTED_GATE = 4_907
FINAL_DIR = ROOT / "artifacts/v14-event-sequence/final"
TRAINER_DIR = ROOT / "artifacts/v14-event-sequence"
FEATURE_DIR = TRAINER_DIR / "features"
MOVEMENT_DIR = ROOT / "artifacts/v6-movement-only"
CACHE_DIR = ROOT / "artifacts/baseline"
DATA_DIR = ROOT / "data"
V8_RANK = ROOT / "submissions/merry-mushroom_v8.parquet"
TEMPLATE = DATA_DIR / "submitting.parquet"
RANKING_ARRIVAL = ROOT / "artifacts/v5-arrival-clean/ranking_arrival_features.parquet"
WEATHER = DATA_DIR / "external/weather.parquet"
SCHEMA_VERSION = 1


def sha(path: Path) -> str:
    return trainer.sha256(Path(path))


def read_json(path: Path) -> dict:
    first = sha(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if sha(path) != first or not isinstance(value, dict):
        raise ValueError(f"JSON changed during read: {path}")
    return value


def write_json_exclusive(path: Path, value: dict) -> None:
    trainer.write_json(path, value)


def published_source(expected: str) -> None:
    if (not isinstance(expected, str) or len(expected) != 64
            or expected.lower() != sha(Path(__file__).resolve())
            or sha(Path(trainer.__file__).resolve()) != TRAINER_SHA256
            or sha(Path(builder.__file__).resolve()) != BUILDER_SHA256
            or sha(trainer.SPEC) != SPEC_SHA256):
        raise ValueError("Require exact published v14 final, trainer, builder, and spec bytes")


def require_memory() -> None:
    trainer.assert_min_memory()


def terminal_path() -> Path:
    return TRAINER_DIR / "terminal.json"


def require_terminal() -> dict:
    terminal = trainer.require_terminal(
        output_dir=TRAINER_DIR, feature_dir=FEATURE_DIR,
        movement_dir=MOVEMENT_DIR, cache_dir=CACHE_DIR)
    if (terminal.get("passed") is not True
            or terminal.get("route") != "exact_v8_no_nm_non_lirf_only"
            or terminal.get("replacement_weight") != 1.0
            or terminal.get("scientific_spec_sha256") != SPEC_SHA256
            or terminal.get("ranking_and_final_modes")
            != "not_implemented_pending_separate_extension"):
        raise ValueError("All original/fresh/reserved v14 terminal gates must pass")
    return terminal


def common_source_paths(reference: Path) -> dict[str, Path]:
    args = trainer.default_args()
    sources = {f"trainer_{key}": path for key, path in trainer.source_paths(args).items()}
    sources.update({
        "own_source": Path(__file__).resolve(),
        "trainer_source": Path(trainer.__file__).resolve(),
        "builder_source": Path(builder.__file__).resolve(),
        "terminal": terminal_path(),
        "original_validation": TRAINER_DIR / "original_validation.json",
        "original_predictions": TRAINER_DIR / "original_predictions.parquet",
        "fresh_audit": TRAINER_DIR / "fresh_april_october_audit.json",
        "fresh_predictions": TRAINER_DIR / "fresh_april_october_predictions.parquet",
        "reserved_audit": TRAINER_DIR / "reserved_february_august_audit.json",
        "reserved_predictions": TRAINER_DIR / "reserved_february_august_predictions.parquet",
        "ranking_reference": reference,
        "v8_ranking_reference": V8_RANK,
        "raw_ranking": DATA_DIR / "ranking.parquet",
        "submission_template": TEMPLATE,
        "ranking_baseline_rows": CACHE_DIR / "ranking_rows.parquet",
        "ranking_baseline_features": CACHE_DIR / "ranking_features.parquet",
        "ranking_arrival_cache": RANKING_ARRIVAL,
        "weather": WEATHER,
    })
    for fold in trainer.FOLDS:
        directory = TRAINER_DIR / fold
        for name in ("fit.json", "preprocessing.json", "best_model.pt",
                     "oof.parquet", "provenance.json"):
            sources[f"{fold}_{name}"] = directory / name
    return sources


def hashes(paths: dict[str, Path]) -> dict[str, str]:
    if len(paths) != len(set(paths)):
        raise ValueError("Duplicate source role")
    return {name: sha(path) for name, path in sorted(paths.items())}


def check_hashes(paths: dict[str, Path], expected: dict[str, str]) -> None:
    fresh = hashes(paths)
    if fresh != expected:
        changed = [key for key in expected if expected[key] != fresh.get(key)]
        raise ValueError(f"Frozen v14 final inputs changed: {changed[:8]}")


def canonical_reference(path: Path, expected_sha256: str) -> Path:
    actual = Path(path).resolve()
    if not actual.is_relative_to(ROOT.resolve()):
        raise ValueError("Ranking reference must be an explicit local artifact")
    if (not isinstance(expected_sha256, str) or len(expected_sha256) != 64
            or sha(actual) != expected_sha256.lower()):
        raise ValueError("Explicit ranking reference SHA256 differs")
    return actual


def protocol_path() -> Path:
    return FINAL_DIR / "protocol.json"


def model_path() -> Path:
    return FINAL_DIR / "full_2025.pt"


def model_report_path() -> Path:
    return FINAL_DIR / "full_2025.json"


def ranking_protocol_path() -> Path:
    return FINAL_DIR / "ranking_protocol.json"


def prepare_final(reference_path: Path, reference_sha256: str,
                  expected_source_sha256: str) -> dict:
    published_source(expected_source_sha256)
    require_memory()
    if protocol_path().exists():
        raise FileExistsError("Final protocol already frozen; never retarget ranking reference")
    reference = canonical_reference(reference_path, reference_sha256)
    before = hashes(common_source_paths(reference))
    terminal = require_terminal()
    if sha(terminal_path()) != before["terminal"]:
        raise ValueError("Canonical trainer terminal changed during preparation")
    epochs = terminal["final_epochs_if_separately_authorized"]
    original_epochs = terminal["original_best_epochs"]
    if (type(epochs) is not int or type(original_epochs) is not list
            or len(original_epochs) != 2 or not all(type(x) is int for x in original_epochs)
            or epochs != math.floor(np.median(original_epochs))
            or not 1 <= epochs <= trainer.MAX_EPOCHS):
        raise ValueError("Final epoch count must be floor median of original best epochs")
    value = {"schema_version": SCHEMA_VERSION, "status": "prepared_before_full_fit",
             "scientific_spec_sha256": SPEC_SHA256,
             "published_source_sha256": expected_source_sha256.lower(),
             "trainer_source_sha256": TRAINER_SHA256,
             "builder_source_sha256": BUILDER_SHA256,
             "trainer_terminal_sha256": before["terminal"],
             "training_event_build_sha256": before["trainer_event_build_report"],
             "training_movement_manifest_sha256": before["trainer_movement_manifest"],
             "ranking_reference_path": str(reference),
             "ranking_reference_sha256": reference_sha256.lower(),
             "v8_ranking_reference_sha256": V8_RANK_SHA256,
             "full_fit_epochs": epochs, "original_best_epochs": original_epochs,
             "fixed_replacement_weight": 1.0,
             "train_ordinary_target_range_sec": [0, 7200],
             "event_schema": [32, 6], "numeric_features": 66,
             "categorical_features": 13,
             "source_sha256": before,
             "repeated_2025_checks_are_not_untouched_generalization": True,
             "independent_gpu_reproduction_requires_own_input_and_model_seals": True}
    if before["v8_ranking_reference"] != V8_RANK_SHA256:
        raise ValueError("Sealed v8 local ranking source changed")
    check_hashes(common_source_paths(reference), before)
    write_json_exclusive(protocol_path(), value)
    check_hashes(common_source_paths(reference), before)
    return value


def check_prepared(expected_source_sha256: str) -> dict:
    published_source(expected_source_sha256)
    prepared = read_json(protocol_path())
    reference = Path(prepared["ranking_reference_path"])
    terminal = require_terminal()
    if (prepared.get("schema_version") != SCHEMA_VERSION
            or prepared.get("status") != "prepared_before_full_fit"
            or prepared.get("published_source_sha256") != expected_source_sha256.lower()
            or prepared.get("scientific_spec_sha256") != SPEC_SHA256
            or prepared.get("trainer_source_sha256") != TRAINER_SHA256
            or prepared.get("builder_source_sha256") != BUILDER_SHA256
            or prepared.get("trainer_terminal_sha256") != sha(terminal_path())
            or prepared.get("full_fit_epochs")
            != terminal["final_epochs_if_separately_authorized"]
            or prepared.get("original_best_epochs") != terminal["original_best_epochs"]
            or prepared.get("fixed_replacement_weight") != 1.0
            or prepared.get("ranking_reference_sha256") != sha(reference)
            or prepared.get("v8_ranking_reference_sha256") != V8_RANK_SHA256
            or prepared.get("source_sha256", {}).get("v8_ranking_reference")
            != V8_RANK_SHA256):
        raise ValueError("v14 final pre-fit protocol or terminal gate changed")
    canonical_reference(reference, prepared["ranking_reference_sha256"])
    check_hashes(common_source_paths(reference), prepared["source_sha256"])
    return prepared


def ordinary_indices(rows: pd.DataFrame) -> np.ndarray:
    y = rows.target.to_numpy(dtype=float)
    selected = np.flatnonzero(np.isfinite(y) & (y >= 0) & (y <= 7200))
    if len(rows) != EXPECTED_TRAIN or len(selected) < 1_500_000:
        raise ValueError("Full 2025 ordinary target universe differs")
    return selected


def verify_training_data(prepared: dict) -> tuple[pd.DataFrame, np.ndarray, np.memmap,
                                                  np.memmap, dict]:
    receipt = builder.verify(scope="training", directory=FEATURE_DIR)
    if (receipt.get("passed") is not True or receipt.get("departure_labels_used") is not False
            or receipt.get("rows") != EXPECTED_TRAIN
            or receipt.get("build_report_sha256")
            != prepared["training_event_build_sha256"]):
        raise ValueError("Sealed label-free event bank changed")
    args = trainer.default_args()
    trainer_sources = {key.removeprefix("trainer_"): value
                       for key, value in prepared["source_sha256"].items()
                       if key.startswith("trainer_")}
    trainer.verify_existing_sources(args, trainer_sources)
    rows = movement.read_baseline_rows(CACHE_DIR,
        ["MVT_ID_mvt", "target", "proxy", "month", "airport", "time"])
    ids, events, presence, _ = builder.open_arrays(
        scope="training", directory=FEATURE_DIR,
        verified_receipt=receipt, mode="r")
    prepared_ids = pd.read_parquet(MOVEMENT_DIR / "row_ids.parquet",
                                   columns=["MVT_ID_mvt"])
    if (len(rows) != EXPECTED_TRAIN or rows.MVT_ID_mvt.isna().any()
            or rows.MVT_ID_mvt.duplicated().any()
            or not np.array_equal(ids, rows.MVT_ID_mvt.to_numpy())
            or not np.array_equal(prepared_ids.MVT_ID_mvt.to_numpy(),
                                  rows.MVT_ID_mvt.to_numpy())
            or events.shape != (EXPECTED_TRAIN, 32, 6)
            or presence.shape != (EXPECTED_TRAIN, 32)):
        raise ValueError("Full event/movement/baseline ID order or schema differs")
    return rows, ids, events, presence, receipt


def final_model_package(model: trainer.EventSequenceCNN,
                        optimizer: torch.optim.AdamW, epochs: int,
                        history: list[dict], runtime: dict,
                        fit_ids_sha256: str, fit_target_sha256: str,
                        prep_sha256: str) -> dict:
    return {"model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epochs_completed": epochs, "history": history,
            "history_sha256": trainer.digest_json(history),
            "runtime": runtime,
            "vocab_sizes": [layer.num_embeddings for layer in model.embeddings],
            "ordered_fit_ids_sha256": fit_ids_sha256,
            "ordered_fit_target_sha256": fit_target_sha256,
            "preprocessing_sha256": prep_sha256,
            "requested": {"seed": trainer.SEED,
                          "batch_size": trainer.BATCH_SIZE,
                          "epochs": epochs,
                          "optimizer": "AdamW", "learning_rate": 0.001,
                          "weight_decay": 0.0001,
                          "loss": "mse_target_div_3600",
                          "event_shape": [32, 6], "numeric_features": 66,
                          "categorical_features": 13}}


def checkpoint_readback(path: Path, report: dict, prep: dict) -> trainer.EventSequenceCNN:
    if sha(path) != report["model_sha256"]:
        raise ValueError("v14 final model bytes changed")
    package = torch.load(path, map_location="cpu", weights_only=False)
    expected_sizes = [len(prep["vocabularies"][name]) + 2 for name in prep["categorical"]]
    requested = {"seed": trainer.SEED, "batch_size": trainer.BATCH_SIZE,
                 "epochs": report["epochs"], "optimizer": "AdamW",
                 "learning_rate": .001, "weight_decay": .0001,
                 "loss": "mse_target_div_3600", "event_shape": [32, 6],
                 "numeric_features": 66, "categorical_features": 13}
    groups = package["optimizer_state"]["param_groups"]
    states = package["optimizer_state"]["state"]
    expected_steps = (math.ceil(report["eligible_ordinary_rows"] / trainer.BATCH_SIZE)
                      * report["epochs"])
    actual_steps = []
    for state in states.values():
        step = state.get("step")
        actual_steps.append(int(torch.as_tensor(step).item()) if step is not None else -1)
    if (package["epochs_completed"] != report["epochs"]
            or package["requested"] != requested
            or package["vocab_sizes"] != expected_sizes
            or package["preprocessing_sha256"] != report["preprocessing_sha256"]
            or package["ordered_fit_ids_sha256"] != report["ordered_fit_ids_sha256"]
            or package["ordered_fit_target_sha256"] != report["ordered_fit_target_sha256"]
            or package["runtime"] != report["runtime"]
            or package["history_sha256"] != trainer.digest_json(package["history"])
            or package["history_sha256"] != report["history_sha256"]
            or len(package["history"]) != report["epochs"]
            or [item["epoch"] for item in package["history"]]
            != list(range(1, report["epochs"] + 1))
            or len(groups) != 1 or groups[0]["lr"] != .001
            or groups[0]["weight_decay"] != .0001
            or len(states) != len(groups[0]["params"])
            or not actual_steps or any(step != expected_steps for step in actual_steps)
            or report.get("optimizer_steps_per_parameter") != expected_steps
            or report.get("optimizer_parameter_states") != len(states)
            or report.get("actual_optimizer") != {
                "name": "AdamW", "learning_rate": .001, "weight_decay": .0001}):
        raise ValueError("Saved v14 final model/optimizer/epoch metadata differs")
    model = trainer.EventSequenceCNN(expected_sizes)
    model.load_state_dict(package["model_state"], strict=True)
    return model


def fit_final(expected_source_sha256: str) -> dict:
    published_source(expected_source_sha256)
    require_memory()
    if model_path().exists() or model_report_path().exists() or (FINAL_DIR / "preprocessing.json").exists():
        raise FileExistsError("Full v14 model has complete or partial prior output")
    prepared_header = read_json(protocol_path())
    reference = Path(prepared_header["ranking_reference_path"])
    before = hashes(common_source_paths(reference))
    prepared = check_prepared(expected_source_sha256)
    if before != prepared["source_sha256"]:
        raise ValueError("Final inputs changed before label read")
    rows, _, events, presence, event_receipt = verify_training_data(prepared)
    fit_idx = ordinary_indices(rows)
    fit_ids_sha = trainer.ids_digest(rows.MVT_ID_mvt.iloc[fit_idx])
    fit_y_sha = trainer.values_digest(rows.target.iloc[fit_idx].to_numpy(dtype=float))
    movement_manifest = read_json(MOVEMENT_DIR / "features_manifest.json")
    prep = trainer.fit_preprocessing_from_parquet(
        MOVEMENT_DIR / "features.parquet", fit_idx, movement_manifest, fit_ids_sha)
    if (len(prep["numeric"]) != 66 or len(prep["categorical"]) != 13
            or prep["feature_order"] != movement_manifest["features"]
            or prep["fit_ordered_id_sha256"] != fit_ids_sha):
        raise ValueError("Full fit-only movement preprocessing schema differs")
    check_hashes(common_source_paths(reference), before)
    prep_path = FINAL_DIR / "preprocessing.json"
    write_json_exclusive(prep_path, prep)
    view = trainer.RawMovementBatchView(MOVEMENT_DIR / "features.parquet", prep, len(rows))
    runtime = trainer.configure_determinism()
    device = torch.device("cuda:0")
    sizes = [len(prep["vocabularies"][name]) + 2 for name in prep["categorical"]]
    model = trainer.EventSequenceCNN(sizes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    epochs = prepared["full_fit_epochs"]
    y = rows.target.to_numpy(dtype=np.float32)
    history = []
    started = time.monotonic()
    for epoch in range(1, epochs + 1):
        model.train()
        permuted = np.random.default_rng(trainer.SEED + epoch).permutation(fit_idx)
        mse_sum = 0.0
        for offset in range(0, len(permuted), trainer.BATCH_SIZE):
            part = permuted[offset:offset + trainer.BATCH_SIZE]
            tensors = trainer.tensors_for(part, events, presence, view, device)
            label = torch.from_numpy(np.array(y[part] / 3600.0, dtype=np.float32,
                                               copy=True)).to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(*tensors)
            loss = F.mse_loss(prediction, label)
            loss.backward()
            optimizer.step()
            mse_sum += float(loss.detach().cpu()) * len(part)
        item = {"epoch": epoch, "fit_mse_target_div_3600": mse_sum / len(fit_idx)}
        history.append(item)
        print(json.dumps({"v14_final_progress": item}), flush=True)
    elapsed = time.monotonic() - started
    probe_idx = fit_idx[:min(256, len(fit_idx))]
    before_probe = trainer.predict_indices(model, probe_idx, events, presence,
                                           view, device)
    prep_sha = sha(prep_path)
    package = final_model_package(model, optimizer, epochs, history, runtime,
                                  fit_ids_sha, fit_y_sha, prep_sha)
    check_hashes(common_source_paths(reference), before)
    trainer.save_model_exclusive(package, model_path())
    report = {"schema_version": SCHEMA_VERSION, "selected_route": "v14_exact_no_nm_gate",
              "protocol_sha256": sha(protocol_path()),
              "trainer_terminal_sha256": sha(terminal_path()),
              "published_source_sha256": expected_source_sha256.lower(),
              "source_sha256": before, "event_training_receipt_sha256":
                  event_receipt["build_report_sha256"],
              "training_rows": int(len(rows)), "eligible_ordinary_rows": int(len(fit_idx)),
              "ordered_fit_ids_sha256": fit_ids_sha,
              "ordered_fit_target_sha256": fit_y_sha,
              "preprocessing_sha256": prep_sha,
              "vocabulary_sha256": prep["vocabulary_sha256"],
              "movement_raw_features_sha256": view.source_sha256,
              "movement_feature_order": prep["feature_order"],
              "movement_raw_schema": view.schema,
              "full_normalized_matrix_materialized": False,
              "actual_raw_dataframe_memory_bytes": int(view.frame.memory_usage(deep=True).sum()),
              "epochs": epochs, "original_best_epochs": prepared["original_best_epochs"],
              "history_sha256": package["history_sha256"],
              "actual_optimizer": {"name": type(optimizer).__name__,
                                   "learning_rate": optimizer.param_groups[0]["lr"],
                                   "weight_decay": optimizer.param_groups[0]["weight_decay"]},
              "optimizer_steps_per_parameter":
                  math.ceil(len(fit_idx) / trainer.BATCH_SIZE) * epochs,
              "optimizer_parameter_states": len(optimizer.state),
              "runtime": runtime, "fit_seconds": float(elapsed),
              "model_sha256": sha(model_path()),
              "probe_ids_sha256": trainer.ids_digest(rows.MVT_ID_mvt.iloc[probe_idx]),
              "probe_raw_prediction_sha256": trainer.values_digest(before_probe),
              "ranking_prediction_created": False}
    readback = checkpoint_readback(model_path(), report, prep).to(device)
    after_probe = trainer.predict_indices(readback, probe_idx, events, presence,
                                          view, device)
    if not np.array_equal(before_probe, after_probe):
        raise ValueError("Final checkpoint does not exactly replay fit probe")
    check_hashes(common_source_paths(reference), before)
    write_json_exclusive(model_report_path(), report)
    check_hashes(common_source_paths(reference), before)
    return {"eligible_ordinary_rows": len(fit_idx), "epochs": epochs,
            "model_sha256": report["model_sha256"],
            "checkpoint_probe_replayed": True}


def verify_final_model(expected_source_sha256: str, *, replay_probe: bool = True) -> dict:
    require_memory()
    header = read_json(protocol_path())
    reference = Path(header["ranking_reference_path"])
    before = hashes(common_source_paths(reference))
    prepared = check_prepared(expected_source_sha256)
    report = read_json(model_report_path())
    prep_path = FINAL_DIR / "preprocessing.json"
    prep = read_json(prep_path)
    if (report.get("schema_version") != SCHEMA_VERSION
            or report.get("selected_route") != "v14_exact_no_nm_gate"
            or report.get("protocol_sha256") != sha(protocol_path())
            or report.get("trainer_terminal_sha256") != sha(terminal_path())
            or report.get("published_source_sha256") != expected_source_sha256.lower()
            or report.get("source_sha256") != before
            or report.get("preprocessing_sha256") != sha(prep_path)
            or report.get("model_sha256") != sha(model_path())
            or report.get("event_training_receipt_sha256")
            != prepared["training_event_build_sha256"]
            or report.get("movement_raw_features_sha256")
            != sha(MOVEMENT_DIR / "features.parquet")
            or report.get("training_rows") != EXPECTED_TRAIN
            or type(report.get("eligible_ordinary_rows")) is not int
            or report["eligible_ordinary_rows"] < 1_500_000
            or report.get("epochs") != prepared["full_fit_epochs"]
            or report.get("original_best_epochs") != prepared["original_best_epochs"]
            or report.get("movement_feature_order") != prep["feature_order"]
            or report.get("vocabulary_sha256") != prep["vocabulary_sha256"]
            or prep["vocabulary_sha256"] != trainer.digest_json(prep["vocabularies"])
            or prep["fit_ordered_id_sha256"] != report["ordered_fit_ids_sha256"]
            or report.get("full_normalized_matrix_materialized") is not False
            or report.get("ranking_prediction_created") is not False):
        raise ValueError("Full event model/report/preprocessing/source receipt differs")
    model = checkpoint_readback(model_path(), report, prep)
    if replay_probe:
        rows, _, events, presence, _ = verify_training_data(prepared)
        fit_idx = ordinary_indices(rows)
        if (len(fit_idx) != report["eligible_ordinary_rows"]
                or trainer.ids_digest(rows.MVT_ID_mvt.iloc[fit_idx])
                != report["ordered_fit_ids_sha256"]
                or trainer.values_digest(rows.target.iloc[fit_idx].to_numpy(dtype=float))
                != report["ordered_fit_target_sha256"]):
            raise ValueError("Full 2025 fit eligibility/labels changed")
        view = trainer.RawMovementBatchView(MOVEMENT_DIR / "features.parquet",
                                            prep, len(rows))
        if view.schema != report["movement_raw_schema"]:
            raise ValueError("Full model raw feature schema changed")
        trainer.configure_determinism()
        probe = fit_idx[:min(256, len(fit_idx))]
        model = model.to("cuda:0")
        raw = trainer.predict_indices(model, probe, events, presence,
                                      view, torch.device("cuda:0"))
        if (trainer.ids_digest(rows.MVT_ID_mvt.iloc[probe])
                != report["probe_ids_sha256"]
                or trainer.values_digest(raw)
                != report["probe_raw_prediction_sha256"]):
            raise ValueError("Saved full model does not replay original probe")
    check_hashes(common_source_paths(reference), before)
    return report


class FrameBatchView:
    """Apply the frozen full-fit preprocessing only to requested rank rows."""

    def __init__(self, frame: pd.DataFrame, prep: dict):
        if list(frame) != prep["feature_order"]:
            raise ValueError("Ranking movement feature names/order differ from full fit")
        self.frame = frame
        self.prep = prep

    def take(self, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return trainer.transform_frame(self.frame.iloc[indices], self.prep)


def combine_reference(reference: np.ndarray, sealed_v8: np.ndarray,
                      gate: np.ndarray, expert: np.ndarray,
                      *, expected_gate: int) -> np.ndarray:
    current = np.asarray(reference, dtype=np.float64)
    v8 = np.asarray(sealed_v8, dtype=np.float64)
    mask = np.asarray(gate, dtype=bool)
    raw = np.asarray(expert, dtype=np.float64)
    if (len(current) != len(v8) or len(current) != len(mask)
            or int(mask.sum()) != expected_gate or len(raw) != expected_gate
            or not np.isfinite(current).all() or np.any(current < 0)
            or not np.isfinite(v8).all() or np.any(v8 < 0)
            or not np.isfinite(raw).all()
            or not np.array_equal(current[mask], v8[mask])):
        raise ValueError("Current ranking reference differs from sealed v8 missing gate")
    output = current.copy()
    output[mask] = np.maximum(raw, 0)  # Fixed weight one; no upper cap.
    if (not np.array_equal(output[~mask], current[~mask])
            or not np.isfinite(output).all() or np.any(output < 0)):
        raise ValueError("v14 replacement altered outside gate or produced invalid predictions")
    return output


def ranking_prebuild_paths(reference: Path) -> dict[str, Path]:
    paths = common_source_paths(reference)
    paths.update({
        "prepared_final_protocol": protocol_path(),
        "full_model": model_path(),
        "full_model_report": model_report_path(),
        "full_preprocessing": FINAL_DIR / "preprocessing.json",
    })
    inventory = builder.source_inventory(
        scope="ranking", data_dir=DATA_DIR, cache_dir=CACHE_DIR,
        movement_dir=MOVEMENT_DIR, directory=FEATURE_DIR,
        trainer_terminal=terminal_path())
    for role, item in inventory.items():
        paths[f"rank_source_{role}"] = Path(item["path"])
    return paths


def prepare_ranking(expected_source_sha256: str) -> dict:
    """Freeze model/reference/raw ranking bytes before the builder reads rows."""
    published_source(expected_source_sha256)
    require_memory()
    if ranking_protocol_path().exists():
        raise FileExistsError("Ranking prebuild seal is already frozen")
    if any((FEATURE_DIR / name).exists() for name in (
            "ranking_inputs.json", "ranking_protocol.json", "ranking_build.json",
            "ranking_events.f16.memmap", "ranking_presence.u8.memmap",
            "ranking_ids.npy")):
        raise FileExistsError("Ranking builder artifacts already exist before final input freeze")
    header = read_json(protocol_path())
    reference = Path(header["ranking_reference_path"])
    before = hashes(ranking_prebuild_paths(reference))
    prepared = check_prepared(expected_source_sha256)
    model_report = verify_final_model(expected_source_sha256, replay_probe=True)
    if before != hashes(ranking_prebuild_paths(reference)):
        raise ValueError("Ranking prebuild sources changed during verification")
    value = {"schema_version": SCHEMA_VERSION,
             "status": "frozen_before_ranking_builder_prepare_or_build",
             "published_source_sha256": expected_source_sha256.lower(),
             "scientific_spec_sha256": SPEC_SHA256,
             "trainer_terminal_sha256": sha(terminal_path()),
             "prepared_final_protocol_sha256": sha(protocol_path()),
             "full_model_sha256": model_report["model_sha256"],
             "full_model_report_sha256": sha(model_report_path()),
             "ranking_reference_path": str(reference),
             "ranking_reference_sha256": prepared["ranking_reference_sha256"],
             "sealed_v8_reference_sha256": V8_RANK_SHA256,
             "source_sha256": before,
             "ranking_values_read": False,
             "builder_cache_created": False}
    write_json_exclusive(ranking_protocol_path(), value)
    check_hashes(ranking_prebuild_paths(reference), before)
    return value


def check_ranking_prepared(expected_source_sha256: str) -> dict:
    published_source(expected_source_sha256)
    frozen = read_json(ranking_protocol_path())
    reference = Path(frozen["ranking_reference_path"])
    if (frozen.get("schema_version") != SCHEMA_VERSION
            or frozen.get("status") != "frozen_before_ranking_builder_prepare_or_build"
            or frozen.get("published_source_sha256") != expected_source_sha256.lower()
            or frozen.get("scientific_spec_sha256") != SPEC_SHA256
            or frozen.get("trainer_terminal_sha256") != sha(terminal_path())
            or frozen.get("prepared_final_protocol_sha256") != sha(protocol_path())
            or frozen.get("full_model_sha256") != sha(model_path())
            or frozen.get("full_model_report_sha256") != sha(model_report_path())
            or frozen.get("ranking_reference_sha256") != sha(reference)
            or frozen.get("sealed_v8_reference_sha256") != V8_RANK_SHA256
            or frozen.get("ranking_values_read") is not False
            or frozen.get("builder_cache_created") is not False):
        raise ValueError("Published ranking prebuild protocol changed")
    check_hashes(ranking_prebuild_paths(reference), frozen["source_sha256"])
    return frozen


def ranking_source_paths(reference: Path) -> dict[str, Path]:
    paths = ranking_prebuild_paths(reference)
    paths.update({
        "frozen_ranking_prebuild_protocol": ranking_protocol_path(),
        "rank_builder_input_seal": FEATURE_DIR / "ranking_inputs.json",
        "rank_builder_protocol": FEATURE_DIR / "ranking_protocol.json",
        "rank_builder_report": FEATURE_DIR / "ranking_build.json",
        "rank_events": FEATURE_DIR / "ranking_events.f16.memmap",
        "rank_presence": FEATURE_DIR / "ranking_presence.u8.memmap",
        "rank_ids": FEATURE_DIR / "ranking_ids.npy",
    })
    return paths


def read_rank_inputs(prepared: dict, receipt: dict) -> tuple[pd.DataFrame, pd.DataFrame,
                                                        np.ndarray, np.ndarray,
                                                        np.ndarray, np.memmap, np.memmap]:
    ids, events, presence, _ = builder.open_arrays(
        scope="ranking", directory=FEATURE_DIR, verified_receipt=receipt, mode="r")
    manifest = read_json(MOVEMENT_DIR / "features_manifest.json")
    rank_args = argparse.Namespace(cache_dir=CACHE_DIR, data_dir=DATA_DIR,
                                   weather_file=WEATHER,
                                   ranking_arrival_cache=RANKING_ARRIVAL)
    features, rows = movement.build_ranking_features(rank_args, manifest)
    template = pd.read_parquet(TEMPLATE, columns=["MVT_ID_mvt"])
    if (len(rows) != EXPECTED_RANK or len(features) != EXPECTED_RANK
            or len(template) != EXPECTED_RANK
            or template.MVT_ID_mvt.isna().any()
            or template.MVT_ID_mvt.duplicated().any()
            or not np.array_equal(rows.MVT_ID_mvt.to_numpy(), ids)
            or not np.array_equal(template.MVT_ID_mvt.to_numpy(), ids)
            or events.shape != (EXPECTED_RANK, 32, 6)
            or presence.shape != (EXPECTED_RANK, 32)):
        raise ValueError("Ranking raw/cache/template/event ID order differs")
    gate = movement.read_gate(CACHE_DIR, rows, ranking=True)
    if int(gate.sum()) != EXPECTED_GATE:
        raise ValueError("Exact sealed v8 no-NM non-LIRF ranking gate differs")
    reference_path = Path(prepared["ranking_reference_path"])
    source = pd.read_parquet(reference_path,
                             columns=["MVT_ID_mvt", "TAXITIME_SEC_mvt"])
    sealed = pd.read_parquet(V8_RANK, columns=["MVT_ID_mvt", "TAXITIME_SEC_mvt"])
    for label, frame in (("current", source), ("sealed_v8", sealed)):
        if (len(frame) != EXPECTED_RANK or frame.MVT_ID_mvt.isna().any()
                or frame.MVT_ID_mvt.duplicated().any()
                or not np.array_equal(frame.MVT_ID_mvt.to_numpy(), ids)
                or not np.isfinite(frame.TAXITIME_SEC_mvt.to_numpy(dtype=float)).all()):
            raise ValueError(f"{label} ranking reference lacks exact template ID coverage")
    current = source.TAXITIME_SEC_mvt.to_numpy(dtype=float)
    v8 = sealed.TAXITIME_SEC_mvt.to_numpy(dtype=float)
    if not np.array_equal(current[gate], v8[gate]):
        raise ValueError("Current valid-route reference changed the sealed v8 missing gate")
    return features, rows, gate, current, v8, events, presence


def predict_ranking(expected_source_sha256: str,
                    published_ranking_seal_sha256: str,
                    published_builder_input_sha256: str,
                    published_builder_protocol_sha256: str,
                    published_builder_build_sha256: str) -> dict:
    published_source(expected_source_sha256)
    require_memory()
    output_path = FINAL_DIR / "predictions.parquet"
    expert_path = FINAL_DIR / "ranking_expert.parquet"
    manifest_path = FINAL_DIR / "ranking_manifest.json"
    if any(path.exists() for path in (output_path, expert_path, manifest_path)):
        raise FileExistsError("Ranking prediction or partial prior output exists")
    header = read_json(protocol_path())
    reference_path = Path(header["ranking_reference_path"])
    required_publications = {
        "ranking_prebuild_protocol": (ranking_protocol_path(),
                                      published_ranking_seal_sha256),
        "builder_ranking_inputs": (FEATURE_DIR / "ranking_inputs.json",
                                   published_builder_input_sha256),
        "builder_ranking_protocol": (FEATURE_DIR / "ranking_protocol.json",
                                     published_builder_protocol_sha256),
        "builder_ranking_build": (FEATURE_DIR / "ranking_build.json",
                                  published_builder_build_sha256),
    }
    for role, (path, expected) in required_publications.items():
        if (not isinstance(expected, str) or len(expected) != 64
                or sha(path) != expected.lower()):
            raise ValueError(f"Require separately published {role} SHA256")
    frozen_rank = check_ranking_prepared(expected_source_sha256)
    before = hashes(ranking_source_paths(reference_path))
    prepared = check_prepared(expected_source_sha256)
    model_report = verify_final_model(expected_source_sha256, replay_probe=True)
    receipt = builder.verify(scope="ranking", directory=FEATURE_DIR,
                             trainer_terminal=terminal_path())
    if (receipt.get("passed") is not True or receipt.get("departure_labels_used") is not False
            or receipt.get("rows") != EXPECTED_RANK
            or receipt.get("ids_order_verified") is not True
            or receipt.get("values_verified") is not True):
        raise ValueError("Ranking event cache/terminal seal did not verify")
    if before != hashes(ranking_source_paths(reference_path)):
        raise ValueError("Ranking inputs changed before first value read")
    if before["frozen_ranking_prebuild_protocol"] != published_ranking_seal_sha256.lower():
        raise ValueError("Ranking prebuild publication changed")
    features, rows, gate, reference, v8, events, presence = read_rank_inputs(prepared, receipt)
    prep = read_json(FINAL_DIR / "preprocessing.json")
    view = FrameBatchView(features, prep)
    model = checkpoint_readback(model_path(), model_report, prep)
    trainer.configure_determinism()
    model = model.to("cuda:0")
    gate_idx = np.flatnonzero(gate)
    raw = trainer.predict_indices(model, gate_idx, events, presence,
                                  view, torch.device("cuda:0"))
    prediction = combine_reference(reference, v8, gate, raw,
                                   expected_gate=EXPECTED_GATE)
    if (len(prediction) != EXPECTED_RANK or not np.isfinite(prediction).all()
            or (prediction < 0).any()
            or not np.array_equal(prediction[~gate], reference[~gate])):
        raise ValueError("Ranking output changed current reference outside exact gate")
    check_hashes(ranking_source_paths(reference_path), before)
    output = pd.DataFrame({"MVT_ID_mvt": rows.MVT_ID_mvt,
                           "TAXITIME_SEC_mvt": prediction})
    expert = pd.DataFrame({"MVT_ID_mvt": rows.MVT_ID_mvt.iloc[gate_idx].to_numpy(),
                           "raw_expert": raw, "expert": np.maximum(raw, 0)})
    trainer.save_parquet_exclusive(expert, expert_path)
    trainer.save_parquet_exclusive(output, output_path)
    result = {"schema_version": SCHEMA_VERSION,
              "route": "v14_exact_no_nm_non_lirf_only",
              "fixed_weight": 1.0,
              "output_rows": EXPECTED_RANK,
              "gate_rows": EXPECTED_GATE,
              "gate_ids_sha256": trainer.ids_digest(rows.MVT_ID_mvt.iloc[gate_idx]),
              "template_order_verified": True,
              "current_v8_gate_equal_before_replacement": True,
              "outside_gate_equal_current_reference": True,
              "finite_nonnegative_predictions": True,
              "prepared_protocol_sha256": sha(protocol_path()),
              "ranking_prebuild_protocol_sha256": sha(ranking_protocol_path()),
              "ranking_prebuild_source_sha256": frozen_rank["source_sha256"],
              "published_builder_input_sha256": published_builder_input_sha256.lower(),
              "published_builder_protocol_sha256": published_builder_protocol_sha256.lower(),
              "published_builder_build_sha256": published_builder_build_sha256.lower(),
              "full_model_report_sha256": sha(model_report_path()),
              "full_model_sha256": sha(model_path()),
              "trainer_terminal_sha256": sha(terminal_path()),
              "rank_builder_receipt": receipt,
              "rank_input_sha256": before,
              "reference_path": str(reference_path),
              "reference_sha256": prepared["ranking_reference_sha256"],
              "sealed_v8_sha256": V8_RANK_SHA256,
              "raw_expert_sha256": trainer.values_digest(raw),
              "ranking_expert_sha256": sha(expert_path),
              "predictions_sha256": sha(output_path),
              "ranking_labels_read": False,
              "no_upload_or_version_created": True}
    check_hashes(ranking_source_paths(reference_path), before)
    write_json_exclusive(manifest_path, result)
    check_hashes(ranking_source_paths(reference_path), before)
    return {"output_rows": EXPECTED_RANK, "gate_rows": EXPECTED_GATE,
            "model_sha256": result["full_model_sha256"],
            "predictions_sha256": result["predictions_sha256"],
            "ranking_manifest_sha256": sha(manifest_path)}


def verify_ranking(expected_source_sha256: str) -> dict:
    published_source(expected_source_sha256)
    require_memory()
    header = read_json(protocol_path())
    reference_path = Path(header["ranking_reference_path"])
    frozen_rank = check_ranking_prepared(expected_source_sha256)
    before = hashes(ranking_source_paths(reference_path))
    prepared = check_prepared(expected_source_sha256)
    model_report = verify_final_model(expected_source_sha256, replay_probe=True)
    receipt = builder.verify(scope="ranking", directory=FEATURE_DIR,
                             trainer_terminal=terminal_path())
    manifest = read_json(FINAL_DIR / "ranking_manifest.json")
    output_path = FINAL_DIR / "predictions.parquet"
    expert_path = FINAL_DIR / "ranking_expert.parquet"
    if (manifest.get("predictions_sha256") != sha(output_path)
            or manifest.get("ranking_expert_sha256") != sha(expert_path)
            or manifest.get("rank_input_sha256") != before
            or manifest.get("rank_builder_receipt") != receipt
            or manifest.get("full_model_sha256") != sha(model_path())
            or manifest.get("full_model_report_sha256") != sha(model_report_path())
            or manifest.get("prepared_protocol_sha256") != sha(protocol_path())
            or manifest.get("ranking_prebuild_protocol_sha256") != sha(ranking_protocol_path())
            or manifest.get("ranking_prebuild_source_sha256") != frozen_rank["source_sha256"]
            or manifest.get("published_builder_input_sha256")
            != before["rank_builder_input_seal"]
            or manifest.get("published_builder_protocol_sha256")
            != before["rank_builder_protocol"]
            or manifest.get("published_builder_build_sha256")
            != before["rank_builder_report"]
            or manifest.get("reference_sha256") != prepared["ranking_reference_sha256"]
            or manifest.get("gate_rows") != EXPECTED_GATE
            or manifest.get("output_rows") != EXPECTED_RANK):
        raise ValueError("Saved ranking result or source seal differs")
    features, rows, gate, reference, v8, events, presence = read_rank_inputs(prepared, receipt)
    prep = read_json(FINAL_DIR / "preprocessing.json")
    model = checkpoint_readback(model_path(), model_report, prep)
    trainer.configure_determinism()
    raw = trainer.predict_indices(model.to("cuda:0"), np.flatnonzero(gate),
                                  events, presence, FrameBatchView(features, prep),
                                  torch.device("cuda:0"))
    expected = combine_reference(reference, v8, gate, raw, expected_gate=EXPECTED_GATE)
    saved = pd.read_parquet(output_path)
    saved_expert = pd.read_parquet(expert_path)
    if (list(saved) != ["MVT_ID_mvt", "TAXITIME_SEC_mvt"]
            or list(saved_expert) != ["MVT_ID_mvt", "raw_expert", "expert"]
            or not np.array_equal(saved.MVT_ID_mvt.to_numpy(), rows.MVT_ID_mvt.to_numpy())
            or not np.array_equal(saved.TAXITIME_SEC_mvt.to_numpy(dtype=float), expected)
            or not np.array_equal(saved_expert.MVT_ID_mvt.to_numpy(),
                                  rows.MVT_ID_mvt.iloc[np.flatnonzero(gate)].to_numpy())
            or not np.array_equal(saved_expert.raw_expert.to_numpy(dtype=float), raw)
            or not np.array_equal(saved_expert.expert.to_numpy(dtype=float), np.maximum(raw, 0))
            or manifest.get("raw_expert_sha256") != trainer.values_digest(raw)
            or manifest.get("gate_ids_sha256")
            != trainer.ids_digest(rows.MVT_ID_mvt.iloc[np.flatnonzero(gate)])):
        raise ValueError("Saved ranking predictions differ from fixed replay")
    check_hashes(ranking_source_paths(reference_path), before)
    return manifest


def self_test() -> dict:
    old = np.array([500.0, 800.0, 1_000.0])
    v8 = np.array([500.0, 800.0, 900.0])
    gate = np.array([False, True, False])
    output = combine_reference(old, v8, gate, np.array([-7.0]), expected_gate=1)
    if not np.array_equal(output, np.array([500.0, 0.0, 1_000.0])):
        raise AssertionError("Fixed replacement or clipping failed")
    failed = False
    try:
        combine_reference(old, np.array([500.0, 801.0, 900.0]), gate,
                          np.array([3.0]), expected_gate=1)
    except ValueError:
        failed = True
    if not failed:
        raise AssertionError("Changed sealed v8 gate was accepted")
    return {"passed": True, "models_fitted": 0, "ranking_rows_read": 0,
            "outside_gate_exact": True, "v8_gate_equality_required": True,
            "negative_direct_clipped": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True,
                        choices=("self-test", "show-contract", "prepare-final",
                                 "fit-final", "verify-final", "prepare-ranking",
                                 "predict-ranking",
                                 "verify-ranking"))
    parser.add_argument("--published-source-sha256")
    parser.add_argument("--ranking-reference", type=Path)
    parser.add_argument("--ranking-reference-sha256")
    parser.add_argument("--published-ranking-seal-sha256")
    parser.add_argument("--published-builder-input-sha256")
    parser.add_argument("--published-builder-protocol-sha256")
    parser.add_argument("--published-builder-build-sha256")
    args = parser.parse_args()
    if args.mode == "self-test":
        result = self_test()
    elif args.mode == "show-contract":
        result = {"spec_sha256": SPEC_SHA256,
                  "trainer_sha256": TRAINER_SHA256,
                  "builder_sha256": BUILDER_SHA256,
                  "terminal": str(terminal_path()),
                  "final_protocol": str(protocol_path()),
                  "ranking_prebuild_protocol": str(ranking_protocol_path()),
                  "builder_ranking_input_seal": str(FEATURE_DIR / "ranking_inputs.json"),
                  "builder_ranking_protocol": str(FEATURE_DIR / "ranking_protocol.json"),
                  "builder_ranking_build_receipt": str(FEATURE_DIR / "ranking_build.json"),
                  "ranking_reference_required_at_prepare": True,
                  "fixed_gate_rows": EXPECTED_GATE,
                  "models_fitted": 0}
    else:
        published_source(args.published_source_sha256)
        if args.mode == "prepare-final":
            if args.ranking_reference is None or args.ranking_reference_sha256 is None:
                parser.error("prepare-final needs --ranking-reference and --ranking-reference-sha256")
            result = prepare_final(args.ranking_reference,
                                   args.ranking_reference_sha256,
                                   args.published_source_sha256)
        elif args.mode == "fit-final":
            result = fit_final(args.published_source_sha256)
        elif args.mode == "verify-final":
            result = verify_final_model(args.published_source_sha256)
        elif args.mode == "prepare-ranking":
            result = prepare_ranking(args.published_source_sha256)
        elif args.mode == "predict-ranking":
            publications = (args.published_ranking_seal_sha256,
                            args.published_builder_input_sha256,
                            args.published_builder_protocol_sha256,
                            args.published_builder_build_sha256)
            if any(value is None for value in publications):
                parser.error("predict-ranking needs four separately published ranking seal SHA256s")
            result = predict_ranking(args.published_source_sha256, *publications)
        else:
            result = verify_ranking(args.published_source_sha256)
    print(json.dumps(result, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
