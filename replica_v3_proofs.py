"""Prospective clean genesis proofs for the first two original v3 stages.

This adapter never authenticates the remaining v3 stages or issues a full v3
producer receipt. Real modes require a separately published source attestation
and independently acquired, SHA-sealed inputs under a disjoint run root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np
import pandas as pd

import clean_parent_producers as parent


SOURCE_ROOT = Path(__file__).resolve().parent
SPEC = SOURCE_ROOT / "reports/clean_replication_v3_proof_spec.json"
SUPPORTED = ("baseline_train", "baseline_predict")
BLOCKED = tuple(stage for stage in parent.V3_STAGES if stage not in SUPPORTED)
FOLDS = {"seasonal_jan_jul": (1, 7), "forward_nov_dec": (11, 12)}
TRAIN_OUTPUTS = (
    "features.parquet", "training_rows.parquet", "seasonal_jan_jul_oof.parquet",
    "forward_nov_dec_oof.parquet", "validation.json", "model.json",
    "seasonal_jan_jul_direct.txt", "forward_nov_dec_direct.txt", "direct.txt",
)
PREDICT_OUTPUTS = (
    "ranking_features.parquet", "ranking_rows.parquet",
    "ranking_predictions.parquet", "proposal/merry-mushroom_v1.parquet",
)
STAGE_BASE = "artifacts/_clean_v3_proof_stage"
TRANSACTION_BASE = "parents/transactions"


def sha256(path: Path) -> str:
    return parent.sha256(path)


def bytes_sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def labels_sha(values: Any) -> str:
    array = np.asarray(values, dtype="<f4")
    if array.ndim != 1:
        raise ValueError("Training label vector must be one-dimensional")
    return bytes_sha(array.tobytes())


def require_source(published_sha: str | None = None, *, require_git: bool = False) -> dict:
    spec = parent.read_json(SPEC)
    if (spec.get("schema_version") != 1 or spec.get("supported_stages") != list(SUPPORTED)
            or spec.get("blocked_stages") != list(BLOCKED)):
        raise ValueError("Prospective baseline proof stage contract changed")
    pinned = {
        "clean_parent_producers.py": spec["producer_driver_sha256"],
        "reports/clean_parent_producers_spec.json": spec["producer_driver_spec_sha256"],
        "solution.py": spec["solution_sha256"],
    }
    for name, digest in pinned.items():
        parent.sha_required(digest, name)
        if sha256(SOURCE_ROOT / name) != digest:
            raise ValueError(f"Original clean producer source changed: {name}")
    parent.source_snapshot(spec["producer_driver_sha256"])
    source = {**pinned,
              "reports/clean_replication_v3_proof_spec.json": sha256(SPEC),
              "replica_v3_proofs.py": sha256(Path(__file__).resolve())}
    if published_sha is not None:
        parent.sha_required(published_sha, "published v3 proof adapter")
        if source["replica_v3_proofs.py"] != published_sha:
            raise ValueError("V3 proof adapter differs from publication attestation")
    if require_git:
        if published_sha is None:
            raise ValueError("Real stage requires a published source SHA")
        for name, digest in source.items():
            try:
                content = subprocess.run(
                    ["git", "-C", str(SOURCE_ROOT), "show", f"HEAD:{name}"],
                    check=True, capture_output=True).stdout
            except subprocess.CalledProcessError as exc:
                raise ValueError(f"Proof source {name} is absent from Git HEAD") from exc
            if bytes_sha(content) != digest:
                raise ValueError(f"Proof source {name} differs from Git HEAD")
    return source


def frozen_inputs(root: Path, published_sha: str) -> dict:
    source = require_source(published_sha, require_git=True)
    raw = parent.snapshot(root, parent.read_json(SPEC)["producer_driver_sha256"])
    commit = parent.published_git_commit(raw)
    return {"proof_source_sha256": source, "clean_raw_and_source": raw,
            "scientific_source_commit": commit}


def check_frozen(root: Path, expected: dict, published_sha: str) -> None:
    if frozen_inputs(root, published_sha) != expected:
        raise ValueError("A proof source or independently sealed raw input changed")


def equal_frame(saved: pd.DataFrame, expected: pd.DataFrame, name: str) -> None:
    try:
        pd.testing.assert_frame_equal(saved.reset_index(drop=True),
                                      expected.reset_index(drop=True),
                                      check_exact=True, check_dtype=True,
                                      check_categorical=True)
    except AssertionError as exc:
        raise ValueError(f"{name} differs from the original source reconstruction") from exc


def equal_metric(actual: Any, expected: Any, name: str) -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError(f"{name} keys differ")
        for key, value in expected.items():
            equal_metric(actual[key], value, f"{name}.{key}")
    elif isinstance(expected, bool):
        if actual is not expected:
            raise ValueError(f"{name} flag differs")
    elif expected is None:
        if actual is not None:
            raise ValueError(f"{name} null differs")
    elif isinstance(expected, (float, int)):
        if not isinstance(actual, (float, int)) or not math.isclose(
                float(actual), float(expected), rel_tol=1e-7, abs_tol=1e-6):
            if not (isinstance(actual, (float, int)) and math.isnan(float(actual))
                    and math.isnan(float(expected))):
                raise ValueError(f"{name} numerical value differs")
    elif actual != expected:
        raise ValueError(f"{name} value differs")


def checked_booster(path: Path, features: pd.DataFrame, rounds: int) -> Any:
    import lightgbm as lgb
    model = lgb.Booster(model_file=str(path))
    if (model.feature_name() != list(features.columns) or model.num_trees() != rounds
            or rounds < 1 or rounds > 900
            or not str(model.dump_model().get("objective", "")).startswith("regression")):
        raise ValueError(f"Saved LightGBM schema, objective or tree count differs: {path.name}")
    return model


def proof_baseline_train(root: Path, output_dir: Path) -> dict:
    """Reconstruct the original cache, fold predictions, report and final model."""
    import lightgbm as lgb  # noqa: F401 - require the published model runtime
    import solution

    parent.free_memory()
    raw_paths = [parent.child(root, parent.RAW[f"raw_training_{m:02d}"])
                 for m in range(1, 13)]
    deps, traffic = solution.load_data(raw_paths)
    x, proxy = solution.build_features(deps, traffic)
    if len(deps) != 2_085_047 or len(x) != len(deps):
        raise ValueError("Baseline full-2025 departure universe differs")
    forbidden = {"MVT_ID_mvt", "FLIGHT_ID_mvt", "TAXITIME_SEC_mvt"}
    if forbidden.intersection(x.columns) or any("BLOCK" in col.upper() for col in x.columns):
        raise ValueError("Baseline feature cache includes an opaque ID, target or BLOCK value")
    equal_frame(pd.read_parquet(output_dir / "features.parquet"), x, "training features")
    y = pd.to_numeric(deps["TAXITIME_SEC_mvt"], errors="coerce").to_numpy(dtype=np.float32)
    month = deps["MVT_TIME_UTC_mvt"].dt.month.to_numpy()
    airport = deps["ADEP_mvt"].astype("string").fillna("__MISSING__").to_numpy()
    rows = pd.DataFrame({"MVT_ID_mvt": deps["MVT_ID_mvt"], "target": y,
                         "proxy": proxy, "month": month, "airport": airport,
                         "time": deps["MVT_TIME_UTC_mvt"]})
    equal_frame(pd.read_parquet(output_dir / "training_rows.parquet"), rows,
                "training row metadata")
    report = parent.read_json(output_dir / "validation.json")
    metadata = parent.read_json(output_dir / "model.json")
    labeled = np.isfinite(y)
    core = labeled & (y >= 0) & (y <= 86_400)
    folds_expected: dict[str, dict] = {}
    pooled = []
    direct_rounds = []
    residual_rounds = []
    fold_proofs = {}
    model_names = []
    for name, held in FOLDS.items():
        valid_month = np.isin(month, held)
        train_idx = core & ~valid_month
        early_idx = core & valid_month
        valid_idx = labeled & valid_month
        if train_idx.sum() < 1000 or valid_idx.sum() < 100:
            raise ValueError(f"Original baseline {name} heldout fold is absent")
        saved = pd.read_parquet(output_dir / f"{name}_oof.parquet")
        direct_round = report["folds"][name]["direct_best_round"]
        direct = checked_booster(output_dir / f"{name}_direct.txt", x, direct_round)
        model_names.append(f"{name}_direct.txt")
        d = direct.predict(x.loc[valid_idx], num_threads=8).astype(np.float32)
        direct_rounds.append(direct_round)
        proxy_train = train_idx & np.isfinite(proxy)
        proxy_valid = valid_idx & np.isfinite(proxy)
        proxy_early = early_idx & np.isfinite(proxy)
        hybrid = d.copy()
        residual_round = report["folds"][name]["residual_best_round"]
        residual_path = output_dir / f"{name}_residual.txt"
        residual_expected = proxy_train.sum() >= 1000 and proxy_valid.sum() >= 100
        if residual_expected:
            if residual_round is None:
                raise ValueError(f"{name} expected a saved residual model")
            residual = checked_booster(residual_path, x, int(residual_round))
            model_names.append(residual_path.name)
            valid_positions = np.flatnonzero(np.isfinite(proxy[valid_idx]))
            hybrid[valid_positions] = (proxy[proxy_valid] + residual.predict(
                x.loc[proxy_valid], num_threads=8)).astype(np.float32)
            residual_rounds.append(int(residual_round))
        elif residual_round is not None or residual_path.exists():
            raise ValueError(f"{name} has an unneeded residual model")
        yi = y[valid_idx]
        pi = proxy[valid_idx]
        fallback = np.where(np.isfinite(pi), pi, d)
        expected_oof = rows.loc[valid_idx].copy()
        expected_oof["row_index"] = np.flatnonzero(valid_idx)
        expected_oof["direct"] = d
        expected_oof["raw_proxy_fallback"] = fallback
        expected_oof["hybrid"] = hybrid
        equal_frame(saved, expected_oof, f"{name} saved model OOF")
        folds_expected[name] = {
            "direct": solution._scores(yi, d, airport[valid_idx], np.isfinite(pi)),
            "proxy_with_direct_fallback": solution._scores(
                yi, fallback, airport[valid_idx], np.isfinite(pi)),
            "residual_with_direct_fallback": solution._scores(
                yi, hybrid, airport[valid_idx], np.isfinite(pi)),
            "direct_best_round": int(direct_round),
            "residual_best_round": int(residual_round) if residual_expected else None,
        }
        pooled.append((yi, d, fallback, hybrid, airport[valid_idx], np.isfinite(pi)))
        fold_proofs[name] = {
            "fit_ids_sha256": parent.ids_sha(rows.MVT_ID_mvt[train_idx]),
            "early_ids_sha256": parent.ids_sha(rows.MVT_ID_mvt[early_idx]),
            "held_all_ids_sha256": parent.ids_sha(rows.MVT_ID_mvt[valid_idx]),
            "fit_labels_sha256": labels_sha(y[train_idx]),
            "early_labels_sha256": labels_sha(y[early_idx]),
            "held_all_labels_sha256": labels_sha(yi),
            "direct_model_sha256": sha256(output_dir / f"{name}_direct.txt"),
            "residual_model_sha256": sha256(residual_path) if residual_expected else None,
            "oof_sha256": sha256(output_dir / f"{name}_oof.parquet"),
        }
    yy, dd, bb, hh, aa, pp = (np.concatenate([part[i] for part in pooled])
                               for i in range(6))
    best = (float("inf"), 0.0, 0.0)
    for share in (0.0, 0.25, 0.5, 0.75, 1.0):
        expert = (1 - share) * bb + share * hh
        diff = expert - dd
        denom = float(np.dot(diff.astype(np.float64), diff.astype(np.float64)))
        alpha = float(np.clip(np.dot((yy - dd).astype(np.float64), diff.astype(np.float64)) /
                              denom, 0, 1)) if denom > 0 else 0.0
        score = solution._rmse(yy, (1 - alpha) * dd + alpha * expert)
        if score < best[0]:
            best = (score, alpha, share)
    _, alpha, share = best
    blended = (1 - alpha) * dd + alpha * ((1 - share) * bb + share * hh)
    expected_report = {
        "folds": folds_expected, "blend_alpha": alpha, "residual_share": share,
        "pooled": solution._scores(yy, blended, aa, pp),
        "pooled_direct_rmse": solution._rmse(yy, dd),
        "pooled_proxy_rmse": solution._rmse(yy, bb),
        "pooled_hybrid_rmse": solution._rmse(yy, hh),
        "proxy_coverage": float(np.isfinite(proxy[labeled]).mean()),
    }
    equal_metric(report, expected_report, "baseline validation")
    final_direct_rounds = int(np.median(direct_rounds))
    checked_booster(output_dir / "direct.txt", x, final_direct_rounds)
    proxy_labeled = core & np.isfinite(proxy)
    final_residual_rounds = int(np.median(residual_rounds)) if residual_rounds else None
    final_residual_expected = (final_residual_rounds is not None and
                               proxy_labeled.sum() >= 1000 and share > 0)
    if final_residual_expected:
        checked_booster(output_dir / "residual.txt", x, final_residual_rounds)
        model_names.append("residual.txt")
    elif (output_dir / "residual.txt").exists():
        raise ValueError("Unselected full residual model appeared")
    expected_metadata = {
        "features": list(x.columns), "blend_alpha": alpha,
        "residual_share": share if final_residual_expected else 0.0,
        "direct_rounds": final_direct_rounds,
        "residual_rounds": final_residual_rounds,
        "training_rows": int(core.sum()),
        "proxy_training_rows": int(proxy_labeled.sum()),
        "core_label_range_sec": [0, 86400],
        "validation_includes_all_finite_labels": True,
        "max_proxy_sec": solution.MAX_PROXY_SEC,
    }
    equal_metric(metadata, expected_metadata, "full baseline model metadata")
    outputs = tuple(TRAIN_OUTPUTS) + tuple(name for name in model_names
                                            if name not in TRAIN_OUTPUTS)
    return {
        "stage": "baseline_train", "replay_passed": True,
        "original_source": "solution.py train --threads 8 --rounds 900",
        "training_rows": len(rows), "training_ids_sha256": parent.ids_sha(rows.MVT_ID_mvt),
        "training_labels_sha256": labels_sha(y),
        "feature_schema_sha256": parent.value_sha(parent.schema_of(x)),
        "folds": fold_proofs, "chosen_blend_alpha": alpha,
        "chosen_residual_share": share,
        "final_direct_rounds": final_direct_rounds,
        "final_residual_rounds": final_residual_rounds,
        "output_names": list(outputs),
    }


def proof_baseline_predict(root: Path, output_dir: Path) -> dict:
    """Rebuild ranking values from sealed raw and the proved baseline models."""
    import lightgbm as lgb
    import solution

    parent.free_memory()
    raw = parent.child(root, parent.RAW["raw_ranking"])
    template = pd.read_parquet(parent.child(root, parent.RAW["submission_template"]))
    deps, traffic = solution.load_data([raw])
    x, proxy = solution.build_features(deps, traffic)
    if len(deps) != 344_841 or len(template) != len(deps):
        raise ValueError("Ranking/template departure coverage differs")
    equal_frame(pd.read_parquet(output_dir / "ranking_features.parquet"), x,
                "ranking features")
    dep_ids = deps["MVT_ID_mvt"].to_numpy(copy=True)
    ranking_rows = pd.DataFrame({"MVT_ID_mvt": dep_ids, "proxy": proxy,
                                 "airport": deps["ADEP_mvt"].astype("string"),
                                 "month": deps["MVT_TIME_UTC_mvt"].dt.month,
                                 "time": deps["MVT_TIME_UTC_mvt"]})
    equal_frame(pd.read_parquet(output_dir / "ranking_rows.parquet"), ranking_rows,
                "ranking row metadata")
    metadata = parent.read_json(output_dir / "model.json")
    if set(metadata["features"]) != set(x.columns):
        raise ValueError("Saved baseline feature schema differs from ranking")
    x = x[metadata["features"]]
    direct = checked_booster(output_dir / "direct.txt", x, int(metadata["direct_rounds"]))
    pred_direct = direct.predict(x, num_threads=8)
    pred = pred_direct.copy()
    hybrid = pred_direct.copy()
    valid = np.isfinite(proxy)
    if metadata["blend_alpha"] > 0 and valid.any():
        expert = proxy[valid].astype(np.float64)
        if metadata["residual_share"] > 0:
            residual = checked_booster(output_dir / "residual.txt", x,
                                       int(metadata["residual_rounds"]))
            corrected = expert + residual.predict(x.loc[valid], num_threads=8)
            hybrid[valid] = corrected
            expert = ((1 - metadata["residual_share"]) * expert +
                      metadata["residual_share"] * corrected)
        pred[valid] = ((1 - metadata["blend_alpha"]) * pred_direct[valid] +
                       metadata["blend_alpha"] * expert)
    pred = np.maximum(pred, 0)
    expected = ranking_rows.copy()
    expected["direct"] = pred_direct
    expected["raw_proxy_fallback"] = np.where(valid, proxy, pred_direct)
    expected["hybrid"] = hybrid
    expected["selected"] = pred
    equal_frame(pd.read_parquet(output_dir / "ranking_predictions.parquet"), expected,
                "ranking saved-model policy")
    if list(template.columns) != ["MVT_ID_mvt", "TAXITIME_SEC_mvt"]:
        raise ValueError("Submitting template schema differs")
    if template.MVT_ID_mvt.isna().any() or template.MVT_ID_mvt.duplicated().any():
        raise ValueError("Submitting template IDs are not unique")
    if set(template.MVT_ID_mvt) != set(dep_ids) or not np.isfinite(pred).all():
        raise ValueError("Ranking IDs or finite predictions differ")
    expected_submission = template.copy()
    expected_submission["TAXITIME_SEC_mvt"] = template.MVT_ID_mvt.map(
        pd.Series(pred, index=dep_ids)).to_numpy(dtype=np.float64)
    saved_submission = pd.read_parquet(output_dir / "proposal/merry-mushroom_v1.parquet")
    equal_frame(saved_submission, expected_submission, "local proposal readback")
    if (saved_submission.TAXITIME_SEC_mvt.to_numpy(dtype=np.float64) < 0).any():
        raise ValueError("Ranking prediction is negative")
    return {"stage": "baseline_predict", "replay_passed": True,
            "ranking_rows": len(dep_ids),
            "ranking_ids_sha256": parent.ids_sha(dep_ids),
            "template_ids_sha256": parent.ids_sha(template.MVT_ID_mvt),
            "feature_schema_sha256": parent.value_sha(parent.schema_of(x)),
            "saved_direct_model_sha256": sha256(output_dir / "direct.txt"),
            "saved_residual_model_sha256": sha256(output_dir / "residual.txt")
            if metadata["residual_share"] > 0 else None,
            "output_names": list(PREDICT_OUTPUTS)}


def proof(stage: str, root: Path, directory: Path) -> dict:
    if stage == "baseline_train":
        return proof_baseline_train(root, directory)
    if stage == "baseline_predict":
        return proof_baseline_predict(root, directory)
    raise RuntimeError(f"Unproved v3 stage {stage} refuses before child launch")


def transaction_path(root: Path, stage: str) -> Path:
    return root / TRANSACTION_BASE / f"{stage}.json"


def prior_snapshot(root: Path, stage: str, published_sha: str,
                   frozen: dict) -> dict | None:
    if stage == "baseline_train":
        return None
    if stage != "baseline_predict":
        raise RuntimeError(f"Unproved v3 stage {stage} refuses")
    previous = verify_transaction(root, "baseline_train", published_sha, frozen)
    return {"baseline_train_transaction_sha256": sha256(transaction_path(root, "baseline_train")),
            "baseline_train_output_sha256": previous["output_sha256"]}


def check_prior(root: Path, stage: str, expected: dict | None) -> None:
    if stage == "baseline_train":
        if expected is not None:
            raise ValueError("Baseline training unexpectedly has a prior model")
        return
    if not isinstance(expected, dict):
        raise ValueError("Baseline prediction has no saved-model parent seal")
    if sha256(parent.child(root, f"{TRANSACTION_BASE}/baseline_train.json")) != expected[
            "baseline_train_transaction_sha256"]:
        raise ValueError("Baseline parent transaction changed")
    for name, digest in expected["baseline_train_output_sha256"].items():
        if sha256(parent.child(root, f"artifacts/baseline/{name}")) != digest:
            raise ValueError(f"Baseline parent model/cache changed: {name}")


def check_staged_model_inputs(stage_dir: Path, prior: dict) -> dict[str, str]:
    """Bind prediction-stage hardlinks to the actual baseline train transaction."""
    names = ["model.json", "direct.txt"]
    metadata = parent.read_json(stage_dir / "model.json")
    if metadata.get("blend_alpha", 0) > 0 and metadata.get("residual_share", 0) > 0:
        names.append("residual.txt")
    expected = prior["baseline_train_output_sha256"]
    observed = {}
    for name in names:
        if name not in expected:
            raise ValueError(f"Baseline model input {name} has no actual parent output")
        observed[name] = sha256(stage_dir / name)
        if observed[name] != expected[name]:
            raise ValueError(f"Staged baseline model input changed: {name}")
    return observed


def verify_transaction(root: Path, stage: str, published_sha: str,
                       frozen: dict) -> dict:
    if stage not in SUPPORTED:
        raise RuntimeError(f"Unproved v3 stage {stage} refuses")
    transaction = parent.read_json(parent.child(
        root, f"{TRANSACTION_BASE}/{stage}.json"))
    expected_command = parent.stage_commands(
        stage, root / STAGE_BASE / stage)
    if (transaction.get("stage") != stage or transaction.get("status") != "complete"
            or transaction.get("source_and_raw_snapshot") != frozen
            or transaction.get("commands") != expected_command
            or transaction.get("adapter_sha256") != published_sha
            or transaction.get("exclusive_promotion_verified") is not True):
        raise ValueError(f"{stage} actual genesis transaction differs")
    if stage == "baseline_train":
        if transaction.get("prior_snapshot") is not None:
            raise ValueError("Baseline train transaction has a detached prior")
    else:
        check_prior(root, stage, transaction.get("prior_snapshot"))
        observed_inputs = check_staged_model_inputs(
            root / STAGE_BASE / stage, transaction["prior_snapshot"])
        if transaction.get("staged_model_input_sha256") != observed_inputs:
            raise ValueError("Baseline prediction stage model aliases changed")
    log_path = parent.child(root, f"{STAGE_BASE}/{stage}/child.log")
    if sha256(log_path) != transaction.get("child_log_sha256"):
        raise ValueError(f"{stage} original child log changed")
    destination = root / parent.OUTPUT_DIR[stage]
    output_hashes = transaction.get("output_sha256")
    if not isinstance(output_hashes, dict) or not output_hashes:
        raise ValueError(f"{stage} output byte inventory is absent")
    for name, digest in output_hashes.items():
        if name not in transaction["proof"]["output_names"]:
            raise ValueError(f"{stage} unproved output role {name}")
        parent.sha_required(digest, f"{stage} output {name}")
        if sha256(parent.child(root, f"{parent.OUTPUT_DIR[stage]}/{name}")) != digest:
            raise ValueError(f"{stage} promoted output {name} changed")
    parent.free_memory()
    replay = proof(stage, root, destination)
    if replay != transaction["proof"] or set(output_hashes) != set(replay["output_names"]):
        raise ValueError(f"{stage} saved model/OOF proof differs on replay")
    if stage == "baseline_predict":
        check_prior(root, stage, transaction["prior_snapshot"])
    check_frozen(root, frozen, published_sha)
    return transaction


def run_stage(root_path: Path, stage: str, published_sha: str) -> dict:
    if stage not in SUPPORTED:
        raise RuntimeError(f"Unproved v3 stage {stage} refuses before child launch")
    root = parent.isolated_root(root_path)
    parent.free_memory()
    frozen = frozen_inputs(root, published_sha)
    prior = prior_snapshot(root, stage, published_sha, frozen)
    stage_dir = (root / STAGE_BASE / stage).resolve(strict=False)
    target = (root / parent.OUTPUT_DIR[stage]).resolve(strict=False)
    tx_path = transaction_path(root, stage).resolve(strict=False)
    if (root not in stage_dir.parents or root not in target.parents or
            root not in tx_path.parents or stage_dir.exists() or tx_path.exists()):
        raise FileExistsError("V3 proof stage/transaction exists or escapes isolated root")
    initial_outputs = TRAIN_OUTPUTS if stage == "baseline_train" else PREDICT_OUTPUTS
    if stage == "baseline_train":
        initial_outputs += ("seasonal_jan_jul_residual.txt",
                            "forward_nov_dec_residual.txt", "residual.txt")
    if any((target / name).exists() for name in initial_outputs):
        raise FileExistsError("Canonical baseline output exists; no overwrite")
    stage_dir.mkdir(parents=True, exist_ok=False)
    if stage == "baseline_predict":
        parent.prepare_stage_inputs(stage, root, stage_dir)
    staged_inputs = (check_staged_model_inputs(stage_dir, prior)
                     if stage == "baseline_predict" else None)
    check_frozen(root, frozen, published_sha)
    check_prior(root, stage, prior)
    commands = parent.stage_commands(stage, stage_dir)
    log_path = stage_dir / "child.log"
    with log_path.open("xb") as log:
        for command in commands:
            parent.free_memory()
            check_frozen(root, frozen, published_sha)
            check_prior(root, stage, prior)
            if stage == "baseline_predict":
                check_staged_model_inputs(stage_dir, prior)
            result = subprocess.run(command, cwd=root, stdout=log,
                                    stderr=subprocess.STDOUT, check=False, shell=False)
            log.flush()
            if result.returncode != 0:
                raise RuntimeError(f"Original {stage} child exited {result.returncode}; "
                                   "partial stage cannot be resumed")
            check_frozen(root, frozen, published_sha)
            check_prior(root, stage, prior)
            if stage == "baseline_predict":
                check_staged_model_inputs(stage_dir, prior)
    replay = proof(stage, root, stage_dir)
    check_frozen(root, frozen, published_sha)
    check_prior(root, stage, prior)
    if stage == "baseline_predict":
        check_staged_model_inputs(stage_dir, prior)
    outputs = {name: (target / name).resolve(strict=False)
               for name in replay["output_names"]}
    if any(root not in path.parents or path.exists() for path in outputs.values()):
        raise FileExistsError("A verified baseline output exists or escapes root")
    output_hashes = {name: sha256(stage_dir / name) for name in outputs}
    for name, path in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        os.link(stage_dir / name, path)
    for name, path in outputs.items():
        if sha256(path) != output_hashes[name]:
            raise ValueError("Promoted baseline output differs from saved-model proof")
    transaction = {
        "schema_version": 1, "stage": stage, "status": "complete",
        "adapter_sha256": published_sha,
        "source_and_raw_snapshot": frozen,
        "prior_snapshot": prior,
        "staged_model_input_sha256": staged_inputs,
        "commands": commands,
        "child_log_sha256": sha256(log_path),
        "proof": replay,
        "output_sha256": output_hashes,
        "exclusive_promotion_verified": True,
    }
    check_frozen(root, frozen, published_sha)
    check_prior(root, stage, prior)
    if stage == "baseline_predict":
        check_staged_model_inputs(stage_dir, prior)
    parent.exclusive_json(tx_path, transaction, root)
    return {"stage": stage, "status": "complete",
            "transaction_sha256": sha256(tx_path),
            "outputs": len(output_hashes)}


def plan() -> dict:
    require_source()
    return {"status": "prospective_source_only", "supported_stages": list(SUPPORTED),
            "blocked_before_child": list(BLOCKED),
            "full_v3_receipt_issued": False,
            "timestamp_and_gpu_source_chain_unblocked": False,
            "real_data_or_models_read": False}


def self_test() -> dict:
    require_source()
    if set(SUPPORTED) | set(BLOCKED) != set(parent.V3_STAGES):
        raise AssertionError("V3 stage closure differs")
    for stage in BLOCKED:
        try:
            proof(stage, Path("synthetic-root"), Path("synthetic-stage"))
        except RuntimeError as exc:
            if "refuses before child launch" not in str(exc):
                raise
        else:
            raise AssertionError("An unproved v3 stage was accepted")
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory).resolve()
        try:
            parent.isolated_root(SOURCE_ROOT)
        except ValueError:
            pass
        else:
            raise AssertionError("Source checkout was accepted as isolated root")
        try:
            parent.child(root, "../outside", existing=False)
        except ValueError:
            pass
        else:
            raise AssertionError("Private path traversal was allowed")
    if labels_sha([1, 2]) == labels_sha([2, 1]):
        raise AssertionError("Ordered label SHA lost row order")
    equal_metric({"x": [1]}, {"x": [1]}, "synthetic equality")
    try:
        equal_metric({"x": 2.0}, {"x": 3.0}, "synthetic tamper")
    except ValueError:
        pass
    else:
        raise AssertionError("Numerical report tamper was accepted")
    return {"source_and_spec_pin": "passed", "blocked_stage_refusal": "passed",
            "ordered_label_and_metric_tamper": "passed", "real_values_or_models_read": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("plan", "self-test", "run-stage", "verify-stage"),
                        default="plan")
    parser.add_argument("--stage", choices=parent.V3_STAGES)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--published-source-sha256")
    args = parser.parse_args()
    if args.mode == "plan":
        result = plan()
    elif args.mode == "self-test":
        result = self_test()
    else:
        if args.stage is None or args.run_root is None or args.published_source_sha256 is None:
            parser.error("Real modes require --stage, --run-root and --published-source-sha256")
        if args.mode == "run-stage":
            result = run_stage(args.run_root, args.stage, args.published_source_sha256)
        else:
            if args.stage not in SUPPORTED:
                raise RuntimeError(f"Unproved v3 stage {args.stage} refuses before value read")
            root = parent.isolated_root(args.run_root)
            parent.free_memory()
            frozen = frozen_inputs(root, args.published_source_sha256)
            tx = verify_transaction(root, args.stage, args.published_source_sha256, frozen)
            result = {"stage": args.stage, "status": "verified_actual_genesis",
                      "transaction_sha256": sha256(transaction_path(root, args.stage)),
                      "output_files": len(tx["output_sha256"])}
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
