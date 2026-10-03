"""Prospective actual-genesis proofs for the remaining original v3 stages.

No real mode is authorized until this source, its spec and the baseline proof
adapter are reviewed and published. Each stage uses only an isolated run root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np
import pandas as pd

import clean_parent_producers as parent
import replica_v3_proofs as baseline


SOURCE_ROOT = Path(__file__).resolve().parent
SPEC = SOURCE_ROOT / "reports/clean_replication_v3_remaining_spec.json"
STAGES = parent.V3_STAGES[2:]
RECOMPUTE_STAGES = ("grouped", "tail", "lirf", "ensemble", "lobt_blend")
MODEL_STAGES = ("missing", "weather", "airports", "lobt")
STAGE_BASE = "artifacts/_clean_v3_remaining_stage"
REPLAY_BASE = "artifacts/_clean_v3_remaining_replay"
TRANSACTION_BASE = "parents/transactions"
MIN_FREE_GIB = 10


def sha256(path: Path) -> str:
    return parent.sha256(path)


def bytes_sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def require_source(published_sha: str | None = None, *, git: bool = False) -> dict:
    spec = parent.read_json(SPEC)
    if (spec.get("schema_version") != 1 or spec.get("ordered_stages") != list(parent.V3_STAGES)
            or spec.get("supported_stages") != list(STAGES)):
        raise ValueError("Complete v3 stage order changed")
    expected = {
        "replica_v3_proofs.py": spec["baseline_proof_adapter_sha256"],
        "reports/clean_replication_v3_proof_spec.json": spec["baseline_proof_spec_sha256"],
        "clean_parent_producers.py": spec["clean_parent_producer_sha256"],
        "reports/clean_parent_producers_spec.json": spec["clean_parent_producer_spec_sha256"],
        parent.LEGACY_ERRATUM_REL: spec["legacy_validation_erratum_sha256"],
    }
    for name, digest in expected.items():
        parent.sha_required(digest, name)
        if sha256(SOURCE_ROOT / name) != digest:
            raise ValueError(f"Upstream proof source changed: {name}")
    baseline.require_source()
    parent.source_snapshot(spec["clean_parent_producer_sha256"])
    source = {**expected,
              "reports/clean_replication_v3_remaining_spec.json": sha256(SPEC),
              "replica_v3_remaining.py": sha256(Path(__file__).resolve())}
    if published_sha is not None:
        parent.sha_required(published_sha, "published remaining-v3 proof")
        if source["replica_v3_remaining.py"] != published_sha:
            raise ValueError("Remaining-v3 adapter differs from publication attestation")
    if git:
        if published_sha is None:
            raise ValueError("Real modes require a published source attestation")
        for name, digest in source.items():
            try:
                content = subprocess.run(["git", "-C", str(SOURCE_ROOT),
                                          "show", f"HEAD:{name}"],
                                         check=True, capture_output=True).stdout
            except subprocess.CalledProcessError as exc:
                raise ValueError(f"{name} is absent from published Git HEAD") from exc
            if bytes_sha(content) != digest:
                raise ValueError(f"{name} differs from published Git HEAD")
    return source


def frozen_inputs(root: Path, published_sha: str) -> dict:
    reviewed = parent.read_json(SOURCE_ROOT / "reports/clean_parent_producers_spec.json").get(
        "reviewed_v3_proof_adapter_sha256")
    require_reviewed_adapter(published_sha, reviewed)
    source = require_source(published_sha, git=True)
    spec = parent.read_json(SPEC)
    parent_raw = parent.snapshot(root, spec["clean_parent_producer_sha256"])
    commit = parent.published_git_commit(parent_raw)
    return {"source_sha256": source, "clean_raw_and_source": parent_raw,
            "scientific_source_commit": commit}


def check_frozen(root: Path, expected: dict, published_sha: str) -> None:
    if frozen_inputs(root, published_sha) != expected:
        raise ValueError("Remaining-v3 source/raw/weather changed")


def prior_snapshot(root: Path, stage: str, frozen: dict,
                   published_sha: str) -> dict:
    if stage not in STAGES:
        raise RuntimeError(f"Unproved v3 stage {stage} refuses")
    spec = parent.read_json(SPEC)
    baseline_frozen = baseline.frozen_inputs(root, spec["baseline_proof_adapter_sha256"])
    baseline.verify_transaction(root, "baseline_predict",
                                spec["baseline_proof_adapter_sha256"], baseline_frozen)
    result: dict = {}
    for prior in parent.V3_STAGES[:parent.V3_STAGES.index(stage)]:
        tx_path = parent.child(root, f"{TRANSACTION_BASE}/{prior}.json")
        tx = parent.read_json(tx_path)
        if (tx.get("stage") != prior or tx.get("status") != "complete"
                or tx.get("exclusive_promotion_verified") is not True):
            raise ValueError(f"Prior v3 {prior} actual-genesis transaction differs")
        if prior in STAGES:
            if (tx.get("source_and_raw_snapshot") != frozen
                    or tx.get("adapter_sha256") != published_sha
                    or tx.get("prior_snapshot") != result
                    or tx.get("commands") != parent.stage_commands(
                        prior, root / STAGE_BASE / prior)
                    or tx.get("proof", {}).get("replay_passed") is not True):
                raise ValueError(f"Prior v3 {prior} source, split or replay seal differs")
            if sha256(parent.child(root, f"{STAGE_BASE}/{prior}/child.log")) != tx.get(
                    "child_log_sha256"):
                raise ValueError(f"Prior v3 {prior} original child log changed")
        outputs = tx.get("output_sha256")
        if not isinstance(outputs, dict) or not outputs:
            raise ValueError(f"Prior v3 {prior} output map is absent")
        for filename, digest in outputs.items():
            parent.sha_required(digest, f"{prior} {filename}")
            if sha256(parent.child(root, f"{parent.OUTPUT_DIR[prior]}/{filename}")) != digest:
                raise ValueError(f"Prior v3 {prior} file changed: {filename}")
        result[prior] = {"transaction_sha256": sha256(tx_path),
                         "output_sha256": outputs}
    return result


def check_prior(root: Path, expected: dict) -> None:
    for stage, record in expected.items():
        if sha256(parent.child(root, f"{TRANSACTION_BASE}/{stage}.json")) != record[
                "transaction_sha256"]:
            raise ValueError(f"Prior v3 {stage} transaction changed")
        for filename, digest in record["output_sha256"].items():
            if sha256(parent.child(root, f"{parent.OUTPUT_DIR[stage]}/{filename}")) != digest:
                raise ValueError(f"Prior v3 {stage} output changed: {filename}")


def output_inventory(directory: Path, stage: str) -> dict[str, Path]:
    files = {}
    for file in directory.rglob("*"):
        if not file.is_file():
            continue
        relative = file.relative_to(directory).as_posix()
        if relative == "child.log":
            continue
        if file.suffix.lower() not in (".json", ".parquet", ".txt", ".cbm"):
            raise ValueError(f"Unrecognized original {stage} output type: {relative}")
        files[relative] = file
    if not set(parent.REQUIRED_OUTPUTS[stage]).issubset(files):
        raise ValueError(f"Original {stage} omitted a mandatory output")
    return files


def compare_recomputed_values(actual: Path, replay: Path, role: str) -> None:
    if actual.suffix == ".parquet":
        a = pd.read_parquet(actual)
        b = pd.read_parquet(replay)
        if (a.columns.tolist() != b.columns.tolist() or len(a) != len(b)
                or ("MVT_ID_mvt" in a and not a.MVT_ID_mvt.equals(b.MVT_ID_mvt))):
            raise ValueError(f"{role} replay ID/schema/coverage differs")
        for exact in ("target", "target_sec", "TAXITIME_SEC_mvt", "fold", "month", "time"):
            if exact in a and not a[exact].equals(b[exact]):
                raise ValueError(f"{role} replay {exact} metadata or label differs")
        try:
            pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-10,
                                          atol=1e-7, check_categorical=True)
        except AssertionError as exc:
            raise ValueError(f"{role} original-source replay values differ") from exc
    elif actual.suffix == ".json":
        baseline.equal_metric(parent.read_json(actual), parent.read_json(replay), role)
    else:
        if sha256(actual) != sha256(replay):
            raise ValueError(f"{role} deterministic source model bytes differ")


def prove_recomputed_stage(root: Path, stage: str, actual_dir: Path,
                           frozen: dict, prior: dict, published_sha: str) -> dict:
    """Second original-source execution only for stages without saved-model replay APIs."""
    if stage not in RECOMPUTE_STAGES:
        raise RuntimeError("A saved-model stage must replay its own models")
    replay_dir = (root / REPLAY_BASE / stage).resolve(strict=False)
    if root not in replay_dir.parents or replay_dir.exists():
        raise FileExistsError("Independent deterministic replay exists or escapes root")
    replay_dir.mkdir(parents=True, exist_ok=False)
    commands = parent.stage_commands(stage, replay_dir)
    with (replay_dir / "child.log").open("xb") as log:
        for command in commands:
            parent.free_memory()
            check_frozen(root, frozen, published_sha)
            check_prior(root, prior)
            result = subprocess.run(command, cwd=root, stdout=log,
                                    stderr=subprocess.STDOUT, check=False, shell=False)
            log.flush()
            if result.returncode != 0:
                raise RuntimeError(f"Independent {stage} replay exited {result.returncode}")
            check_frozen(root, frozen, published_sha)
            check_prior(root, prior)
    original = output_inventory(actual_dir, stage)
    replay = output_inventory(replay_dir, stage)
    if set(original) != set(replay):
        raise ValueError(f"Independent {stage} output inventory differs")
    for name in original:
        compare_recomputed_values(original[name], replay[name], f"{stage}:{name}")
    if stage == "lirf" and parent.read_json(actual_dir / "validation.json").get(
            "improves_existing_on_at_least_one_fold") is not True:
        raise ValueError("Original Rome specialist gate failed")
    if stage == "lobt_blend":
        report = parent.read_json(actual_dir / "validation.json")
        if (report.get("selected_valid_disagreement_weight") != 1.0 or
                report.get("invalid_aobt_lobt_fallback_weight") != 1.0 or
                report.get("valid_disagreement_threshold_sec") != 3600):
            raise ValueError("Accepted fixed LOBT policy was not recovered")
        template = pd.read_parquet(parent.child(root, parent.RAW["submission_template"]),
                                   columns=["MVT_ID_mvt"])
        rank = pd.read_parquet(actual_dir / "predictions.parquet")
        if (len(rank) != 344_841 or not rank.MVT_ID_mvt.equals(template.MVT_ID_mvt)
                or not np.isfinite(rank.TAXITIME_SEC_mvt).all()
                or (rank.TAXITIME_SEC_mvt < 0).any()):
            raise ValueError("Final v3 ranking coverage/template differs")
    return {"stage": stage, "replay_passed": True,
            "method": "second_pinned_original_cli_recomputation",
            "output_names": list(sorted(original)),
            "replay_output_sha256": {name: sha256(path) for name, path in replay.items()},
            "replay_log_sha256": sha256(replay_dir / "child.log")}


def verify_recomputed_stage(root: Path, stage: str, transaction: dict) -> dict:
    actual_dir = root / parent.OUTPUT_DIR[stage]
    replay_dir = root / REPLAY_BASE / stage
    actual = output_inventory(actual_dir, stage)
    replay = output_inventory(replay_dir, stage)
    if set(actual) != set(replay):
        raise ValueError(f"Saved independent {stage} replay inventory changed")
    for name in actual:
        compare_recomputed_values(actual[name], replay[name], f"{stage}:{name}")
    proof = transaction["proof"]
    if (proof.get("stage") != stage or proof.get("replay_passed") is not True
            or proof.get("method") != "second_pinned_original_cli_recomputation"
            or proof.get("output_names") != sorted(actual)
            or proof.get("replay_output_sha256") != {
                name: sha256(path) for name, path in replay.items()}
            or proof.get("replay_log_sha256") != sha256(replay_dir / "child.log")):
        raise ValueError(f"Saved independent {stage} replay receipt differs")
    return proof


def prove_model_stage(root: Path, stage: str, directory: Path) -> dict:
    """Dispatch saved-model OOF/ranking replay; no result is trusted from JSON alone."""
    if stage == "missing":
        return prove_missing(root, directory)
    if stage == "weather":
        return prove_weather(root, directory)
    if stage == "airports":
        return prove_airports(root, directory)
    if stage == "lobt":
        return prove_lobt(root, directory)
    raise RuntimeError(f"Unproved saved-model v3 stage {stage} refuses")


def checked_lgb(path: Path, features: pd.DataFrame, cap: int,
                expected_rounds: int | None = None) -> Any:
    import lightgbm as lgb
    model = lgb.Booster(model_file=str(path))
    count = model.num_trees()
    if (model.feature_name() != list(features.columns) or not 1 <= count <= cap
            or (expected_rounds is not None and count != expected_rounds)
            or not str(model.dump_model().get("objective", "")).startswith("regression")):
        raise ValueError(f"Saved original LightGBM model differs: {path.name}")
    return model


def prove_missing(root: Path, directory: Path) -> dict:
    import polars as pl
    import missing_expert as missing
    import solution

    parent.free_memory()
    base = root / "artifacts/baseline"
    rows = pd.read_parquet(base / "training_rows.parquet")
    missing_mask = ~np.isfinite(rows.proxy)
    raw_cols = ["MVT_ID_mvt", "FLIGHT_ID_mvt", "FLIGHT_mvt", "ADEP_mvt",
                "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"]
    raw = (pl.scan_parquet([str(parent.child(root, parent.RAW[f"raw_training_{m:02d}"]))
                           for m in range(1, 13)])
           .filter(pl.col("PHASE_mvt") == "DEP")
           .select(raw_cols).collect().to_pandas())
    if not np.array_equal(raw.MVT_ID_mvt.to_numpy(), rows.MVT_ID_mvt.to_numpy()):
        raise ValueError("Missing specialist raw/cache ID order differs")
    raw = raw.loc[missing_mask].reset_index(drop=True)
    x = pd.read_parquet(base / "features.parquet").loc[missing_mask].reset_index(drop=True)
    x = missing.augment(x, raw)
    rows = rows.loc[missing_mask].reset_index(drop=True)
    y = rows.target.to_numpy(dtype=np.float64)
    month = rows.month.to_numpy()
    schedule = x.schedule_proxy_unclipped.to_numpy(dtype=np.float64)
    offset = np.where(np.isfinite(schedule) & (schedule >= 0) & (schedule < 172800),
                      schedule, 900.)
    reports = {}
    rounds: dict[str, list[int]] = {"direct": [], "schedule_residual": []}
    fold_proofs = {}
    for name, held in baseline.FOLDS.items():
        valid_month = np.isin(month, held)
        train = ~valid_month & np.isfinite(y)
        valid = valid_month & np.isfinite(y)
        expected = rows.loc[valid].copy()
        for model_name in rounds:
            model = checked_lgb(directory / f"{name}_{model_name}.txt", x, 1400)
            model_offset = offset if model_name == "schedule_residual" else np.zeros(len(y))
            expected[model_name] = model.predict(x.loc[valid], num_threads=2) + model_offset[valid]
            rounds[model_name].append(model.num_trees())
        baseline.equal_frame(pd.read_parquet(directory / f"{name}_oof.parquet"), expected,
                             f"missing {name} model OOF")
        reports[name] = {col: {"rmse": solution._rmse(expected.target.to_numpy(),
                                                       expected[col].to_numpy()),
                               "by_airport": {str(a): solution._rmse(
                                   part.target.to_numpy(), part[col].to_numpy())
                                   for a, part in expected.groupby("airport")}}
                         for col in rounds}
        reports[name]["n"] = len(expected)
        fold_proofs[name] = {
            "fit_ids_sha256": parent.ids_sha(rows.MVT_ID_mvt[train]),
            "held_ids_sha256": parent.ids_sha(rows.MVT_ID_mvt[valid]),
            "fit_labels_sha256": baseline.labels_sha(y[train]),
            "held_labels_sha256": baseline.labels_sha(y[valid]),
        }
    baseline.equal_metric(parent.read_json(directory / "validation.json"), reports,
                          "missing specialist validation")
    final_rounds = {name: int(np.median(values)) for name, values in rounds.items()}
    metadata = parent.read_json(directory / "model.json")
    baseline.equal_metric(metadata, {"rounds": final_rounds,
                                    "features": list(x)}, "missing full-model metadata")
    for name, count in final_rounds.items():
        checked_lgb(directory / f"{name}.txt", x, 1400, count)
    ranking_deps, ranking_traffic = solution.load_data(
        [parent.child(root, parent.RAW["raw_ranking"])])
    rank_x, proxy = solution.build_features(ranking_deps, ranking_traffic)
    rank_valid = ~np.isfinite(proxy)
    rank_x = missing.augment(rank_x.loc[rank_valid].reset_index(drop=True),
                             ranking_deps.loc[rank_valid].reset_index(drop=True))
    rank_schedule = rank_x.schedule_proxy_unclipped.to_numpy(dtype=np.float64)
    rank_base = np.where(np.isfinite(rank_schedule) & (rank_schedule >= 0)
                         & (rank_schedule < 172800), rank_schedule, 900.)
    expected_rank = pd.DataFrame({"MVT_ID_mvt":
                                  ranking_deps.loc[rank_valid, "MVT_ID_mvt"].to_numpy()})
    for name in final_rounds:
        model = checked_lgb(directory / f"{name}.txt", rank_x, 1400, final_rounds[name])
        expected_rank[name] = (model.predict(rank_x, num_threads=2) +
                               (rank_base if name == "schedule_residual" else 0))
    baseline.equal_frame(pd.read_parquet(directory / "predictions.parquet"), expected_rank,
                         "missing specialist ranking model replay")
    model_files = {name for name in output_inventory(directory, "missing")
                   if name.endswith(".txt")}
    expected_models = {f"{fold}_{kind}.txt" for fold in baseline.FOLDS
                       for kind in final_rounds} | {f"{kind}.txt" for kind in final_rounds}
    if model_files != expected_models:
        raise ValueError("Missing specialist saved-model inventory differs")
    return {"stage": "missing", "replay_passed": True,
            "method": "saved_LightGBM_fold_and_ranking_replay",
            "feature_schema_sha256": parent.value_sha(parent.schema_of(x)),
            "folds": fold_proofs,
            "ranking_ids_sha256": parent.ids_sha(expected_rank.MVT_ID_mvt),
            "final_rounds": final_rounds,
            "output_names": list(sorted(output_inventory(directory, "missing")))}


def prove_weather(root: Path, directory: Path) -> dict:
    import weather_model as weather

    parent.free_memory()
    base = root / "artifacts/baseline"
    weather_file = parent.child(root, parent.RAW["noaa_weather"])
    features, rows = weather.load_cached_features(base, ranking=False)
    x = weather.add_weather(features, rows, weather_file)
    y = pd.to_numeric(rows.target, errors="coerce").to_numpy(dtype=np.float64)
    proxy = pd.to_numeric(rows.proxy, errors="coerce").to_numpy(dtype=np.float64)
    month = pd.to_numeric(rows.month, errors="coerce").to_numpy()
    eligible = np.isfinite(y) & (y >= 0) & (y <= 86400) & np.isfinite(proxy)
    if eligible.sum() < 100_000:
        raise ValueError("Weather full-2025 training coverage differs")
    metadata = parent.read_json(directory / "model.json")
    rounds = []
    fold_reports = {}
    fold_proofs = {}
    for fold, held in weather.FOLDS.items():
        train_mask = eligible & ~np.isin(month, held)
        all_hold = np.isfinite(y) & np.isfinite(proxy) & np.isin(month, held)
        model = checked_lgb(directory / f"{fold}_residual.txt", x, 1500)
        best = model.num_trees()
        rounds.append(best)
        residual = model.predict(x.loc[all_hold], num_threads=8)
        expected = rows.loc[all_hold,
                            ["MVT_ID_mvt", "target", "proxy", "month", "airport", "time"]].copy()
        expected["row_index"] = np.flatnonzero(all_hold)
        expected["weather_residual"] = residual.astype(np.float32)
        expected["weather_prediction"] = (proxy[all_hold] + residual).astype(np.float32)
        expected = weather.attach_baseline(expected, base, fold)
        baseline.equal_frame(pd.read_parquet(directory / f"{fold}_oof.parquet"),
                             expected, f"weather {fold} complete OOF")
        fold_reports[fold] = {**weather.fold_report(expected),
                              "best_round": best, "training_rows": int(train_mask.sum())}
        fold_proofs[fold] = {
            "fit_ids_sha256": parent.ids_sha(rows.MVT_ID_mvt[train_mask]),
            "early_ids_sha256": parent.ids_sha(rows.MVT_ID_mvt[eligible & np.isin(month, held)]),
            "held_all_ids_sha256": parent.ids_sha(rows.MVT_ID_mvt[all_hold]),
            "fit_labels_sha256": baseline.labels_sha(y[train_mask]),
            "held_all_labels_sha256": baseline.labels_sha(y[all_hold]),
        }
    final_rounds = int(np.median(rounds))
    checked_lgb(directory / "residual.txt", x, 1500, final_rounds)
    expected_metadata = {
        "features": list(x.columns),
        "weather_source": "NOAA NCEI GHCNh CC0-1.0",
        "weather_table": "data/external/weather.parquet",
        "folds": fold_reports,
        "final_rounds": final_rounds,
        "training_rows": int(eligible.sum()),
        "oof_coverage": "all finite target and proxy rows in each holdout month",
    }
    baseline.equal_metric(metadata, expected_metadata, "weather full-model metadata")
    ranking_features, ranking_rows = weather.load_cached_features(base, ranking=True)
    ranking_x = weather.add_weather(ranking_features, ranking_rows, weather_file)
    if list(ranking_x.columns) != metadata["features"]:
        raise ValueError("Weather ranking feature schema differs")
    rank_proxy = pd.to_numeric(ranking_rows.proxy, errors="coerce").to_numpy(dtype=np.float64)
    valid = np.isfinite(rank_proxy)
    model = checked_lgb(directory / "residual.txt", ranking_x, 1500, final_rounds)
    rank_residual = np.full(len(ranking_rows), np.nan, dtype=np.float64)
    rank_residual[valid] = model.predict(ranking_x.loc[valid], num_threads=8)
    expected_rank = ranking_rows[["MVT_ID_mvt", "proxy", "airport", "month", "time"]].copy()
    expected_rank["weather_residual"] = rank_residual
    expected_rank["weather_prediction"] = np.maximum(rank_proxy + rank_residual, 0)
    baseline.equal_frame(pd.read_parquet(directory / "ranking_predictions.parquet"),
                         expected_rank, "weather final model ranking")
    model_files = {name for name in output_inventory(directory, "weather")
                   if name.endswith(".txt")}
    if model_files != {"seasonal_jan_jul_residual.txt",
                       "forward_nov_dec_residual.txt", "residual.txt"}:
        raise ValueError("Weather saved-model inventory differs")
    return {"stage": "weather", "replay_passed": True,
            "method": "saved_LightGBM_fold_allfinite_and_ranking_replay",
            "feature_schema_sha256": parent.value_sha(parent.schema_of(x)),
            "folds": fold_proofs, "eligible_train_ids_sha256": parent.ids_sha(
                rows.MVT_ID_mvt[eligible]),
            "ranking_ids_sha256": parent.ids_sha(ranking_rows.MVT_ID_mvt),
            "final_rounds": final_rounds,
            "output_names": list(sorted(output_inventory(directory, "weather")))}


def prove_airports(root: Path, directory: Path) -> dict:
    import airport_models as airport_models
    import weather_model as weather

    parent.free_memory()
    base = root / "artifacts/baseline"
    weather_file = parent.child(root, parent.RAW["noaa_weather"])
    x, rows = weather.load_cached_features(base, ranking=False)
    x = weather.add_weather(x, rows, weather_file)
    raw_paths = [parent.child(root, parent.RAW[f"raw_training_{m:02d}"])
                 for m in range(1, 13)]
    x = airport_models.add_flight(x, rows, raw_paths)
    y = rows.target.to_numpy(dtype=float)
    proxy = rows.proxy.to_numpy(dtype=float)
    month = rows.month.to_numpy()
    airport = rows.airport.to_numpy()
    eligible = np.isfinite(y) & np.isfinite(proxy)
    report = parent.read_json(directory / "validation.json")
    expected_report = {}
    best_rounds: dict[str, list[int]] = {}
    fold_proofs = {}
    airports = sorted(pd.unique(airport))
    for fold, held in baseline.FOLDS.items():
        predictions = np.full(len(rows), np.nan)
        expected_report[fold] = {}
        fold_proofs[fold] = {}
        for code in airports:
            if not isinstance(code, str) or not re.fullmatch(r"[A-Z]{4}", code):
                raise ValueError("Airport model file code is not canonical")
            train = eligible & (airport == code) & ~np.isin(month, held)
            valid = eligible & (airport == code) & np.isin(month, held)
            count = int(report[fold][code]["best_round"])
            model = checked_lgb(directory / f"{fold}_{code}.txt", x, 1000, count)
            pred = proxy[valid] + model.predict(x.loc[valid], num_threads=6)
            predictions[valid] = pred
            best_rounds.setdefault(code, []).append(count)
            expected_report[fold][code] = {"rmse": weather.rmse(y[valid], pred),
                                           "best_round": count}
            fold_proofs[fold][code] = {
                "fit_ids_sha256": parent.ids_sha(rows.MVT_ID_mvt[train]),
                "held_ids_sha256": parent.ids_sha(rows.MVT_ID_mvt[valid]),
                "fit_labels_sha256": baseline.labels_sha(y[train]),
                "held_labels_sha256": baseline.labels_sha(y[valid]),
            }
        valid_all = eligible & np.isin(month, held)
        expected_oof = rows.loc[valid_all].copy()
        expected_oof["airport_prediction"] = predictions[valid_all]
        baseline.equal_frame(pd.read_parquet(directory / f"{fold}_oof.parquet"),
                             expected_oof, f"airport {fold} model OOF")
        expected_report[fold]["overall"] = weather.rmse(y[valid_all], predictions[valid_all])
    baseline.equal_metric(report, expected_report, "airport model validation")
    rounds = {code: int(np.median(values)) for code, values in best_rounds.items()}
    metadata = parent.read_json(directory / "model.json")
    baseline.equal_metric(metadata, {"rounds": rounds, "features": list(x)},
                          "airport final model metadata")
    for code, count in rounds.items():
        checked_lgb(directory / f"{code}.txt", x, 1000, count)
    xr, rank = weather.load_cached_features(base, ranking=True)
    xr = weather.add_weather(xr, rank, weather_file)
    xr = airport_models.add_flight(xr, rank, [parent.child(root, parent.RAW["raw_ranking"])])
    rank_pred = np.full(len(rank), np.nan)
    for code, count in rounds.items():
        use = np.isfinite(rank.proxy.to_numpy()) & rank.airport.eq(code).to_numpy()
        model = checked_lgb(directory / f"{code}.txt", xr, 1000, count)
        rank_pred[use] = (rank.loc[use, "proxy"].to_numpy() +
                          model.predict(xr.loc[use], num_threads=6))
    expected_rank = rank.copy()
    expected_rank["airport_prediction"] = rank_pred
    baseline.equal_frame(pd.read_parquet(directory / "ranking_predictions.parquet"),
                         expected_rank, "airport saved-model ranking")
    model_files = {name for name in output_inventory(directory, "airports")
                   if name.endswith(".txt")}
    expected_models = {f"{fold}_{code}.txt" for fold in baseline.FOLDS
                       for code in airports} | {f"{code}.txt" for code in airports}
    if model_files != expected_models:
        raise ValueError("Airport saved-model inventory differs")
    return {"stage": "airports", "replay_passed": True,
            "method": "saved_airport_LightGBM_fold_and_ranking_replay",
            "feature_schema_sha256": parent.value_sha(parent.schema_of(x)),
            "folds": fold_proofs,
            "ranking_ids_sha256": parent.ids_sha(rank.MVT_ID_mvt),
            "final_rounds": rounds,
            "output_names": list(sorted(output_inventory(directory, "airports")))}


def prove_lobt(root: Path, directory: Path) -> dict:
    import lobt_expert as lobt_expert

    parent.free_memory()
    base = root / "artifacts/baseline"
    ensemble_dir = root / "artifacts/ensemble"
    weather_file = parent.child(root, parent.RAW["noaa_weather"])
    rows, x = lobt_expert.load_features(root / "data", base, weather_file, False)
    y = pd.to_numeric(rows.target, errors="coerce").to_numpy(dtype=np.float64)
    aobt = pd.to_numeric(rows.proxy, errors="coerce").to_numpy(dtype=np.float64)
    lobt = rows.lobt_proxy_sec.to_numpy(dtype=np.float64)
    month = rows.month.to_numpy()
    eligible = np.isfinite(y) & np.isfinite(lobt) & (lobt >= 0) & (lobt <= 172800)
    if eligible.sum() < 100_000:
        raise ValueError("LOBT eligible ordinary training coverage differs")
    report = parent.read_json(directory / "validation.json")
    expected_report = {"training_rows": int(eligible.sum()), "folds": {},
                       "notes": "Labels only from 2025 training data; ranking uses released predictors."}
    best_rounds = []
    pooled = []
    fold_proofs = {}
    for fold, held in lobt_expert.FOLDS.items():
        holdout = np.isin(month, held)
        train = eligible & ~holdout
        test = eligible & holdout
        best = int(report["folds"][fold]["best_round"])
        model = checked_lgb(directory / f"{fold}_residual.txt", x, 900, best)
        best_rounds.append(best)
        pred = lobt[test] + model.predict(x.loc[test], num_threads=8, num_iteration=best)
        expected = rows.loc[test, ["MVT_ID_mvt", "target", "month", "airport", "time",
                                   "proxy", "lobt_proxy_sec"]].copy()
        expected["row_index"] = np.flatnonzero(test)
        expected["fold"] = fold
        expected["aobt_valid"] = np.isfinite(aobt[test])
        expected["aobt_lobt_abs_gap"] = np.where(
            np.isfinite(aobt[test]), np.abs(lobt[test] - aobt[test]), np.nan)
        expected["lobt_prediction"] = pred.astype(np.float32)
        reference = lobt_expert.reference_predictions(base, ensemble_dir, fold)
        expected = expected.merge(reference, on="MVT_ID_mvt", how="left", sort=False,
                                  validate="one_to_one")
        baseline.equal_frame(pd.read_parquet(directory / f"{fold}_oof.parquet"),
                             expected, f"LOBT {fold} saved model OOF")
        pooled.append(expected)
        expected_report["folds"][fold] = {"best_round": best,
                                           **lobt_expert.group_metrics(expected)}
        fold_proofs[fold] = {
            "fit_ids_sha256": parent.ids_sha(rows.MVT_ID_mvt[train]),
            "early_ids_sha256": parent.ids_sha(rows.MVT_ID_mvt[test]),
            "held_ids_sha256": parent.ids_sha(rows.MVT_ID_mvt[test]),
            "fit_labels_sha256": baseline.labels_sha(y[train]),
            "held_labels_sha256": baseline.labels_sha(y[test]),
        }
    expected_report["pooled"] = lobt_expert.group_metrics(pd.concat(pooled, ignore_index=True))
    segments = ("aobt_valid_gap_gt_3600", "aobt_valid_gap_gt_1800",
                "aobt_invalid_target_gt_7200")
    supported = bool(any(all(
        expected_report["folds"][fold][segment].get("n", 0) >= 5 and
        expected_report["folds"][fold][segment].get("blend_25pct_lobt", float("inf")) <
        expected_report["folds"][fold][segment].get("nested_ensemble", float("inf"))
        for fold in lobt_expert.FOLDS) for segment in segments))
    expected_report["final_fit_supported"] = supported
    if not supported:
        raise ValueError("Accepted LOBT final-fit scientific gate failed")
    final_rounds = int(np.median(best_rounds))
    checked_lgb(directory / "residual.txt", x, 900, final_rounds)
    ranking_rows, ranking_x = lobt_expert.load_features(root / "data", base,
                                                        weather_file, True)
    rank_lobt = ranking_rows.lobt_proxy_sec.to_numpy(dtype=np.float64)
    rank_aobt = ranking_rows.proxy.to_numpy(dtype=np.float64)
    valid = np.isfinite(rank_lobt) & (rank_lobt >= 0) & (rank_lobt <= 172800)
    model = checked_lgb(directory / "residual.txt", ranking_x, 900, final_rounds)
    rank_prediction = (rank_lobt[valid] +
                       model.predict(ranking_x.loc[valid], num_threads=8))
    expected_rank = ranking_rows.loc[valid,
                                     ["MVT_ID_mvt", "airport", "month", "proxy",
                                      "lobt_proxy_sec"]].copy()
    expected_rank["aobt_valid"] = np.isfinite(rank_aobt[valid])
    expected_rank["aobt_lobt_abs_gap"] = np.where(
        np.isfinite(rank_aobt[valid]), np.abs(rank_lobt[valid] - rank_aobt[valid]), np.nan)
    expected_rank["lobt_prediction"] = rank_prediction.astype(np.float32)
    baseline.equal_frame(pd.read_parquet(directory / "ranking_predictions.parquet"),
                         expected_rank, "LOBT full-model ranking")
    expected_report["final_rounds"] = final_rounds
    expected_report["ranking_rows"] = int(len(expected_rank))
    baseline.equal_metric(report, expected_report, "LOBT scientific validation")
    model_files = {name for name in output_inventory(directory, "lobt")
                   if name.endswith(".txt")}
    if model_files != {"seasonal_jan_jul_residual.txt",
                       "forward_nov_dec_residual.txt", "residual.txt"}:
        raise ValueError("LOBT saved-model inventory differs")
    return {"stage": "lobt", "replay_passed": True,
            "method": "saved_LightGBM_fold_and_ranking_replay",
            "feature_schema_sha256": parent.value_sha(parent.schema_of(x)),
            "folds": fold_proofs,
            "eligible_fit_ids_sha256": parent.ids_sha(rows.MVT_ID_mvt[eligible]),
            "ranking_ids_sha256": parent.ids_sha(expected_rank.MVT_ID_mvt),
            "final_rounds": final_rounds,
            "output_names": list(sorted(output_inventory(directory, "lobt")))}


def run_stage(root_path: Path, stage: str, published_sha: str) -> dict:
    if stage not in STAGES:
        raise RuntimeError(f"Unproved remaining-v3 stage {stage} refuses before child")
    root = parent.isolated_root(root_path)
    parent.free_memory()
    frozen = frozen_inputs(root, published_sha)
    prior = prior_snapshot(root, stage, frozen, published_sha)
    target = (root / parent.OUTPUT_DIR[stage]).resolve(strict=False)
    stage_dir = (root / STAGE_BASE / stage).resolve(strict=False)
    transaction_path = (root / TRANSACTION_BASE / f"{stage}.json").resolve(strict=False)
    if any(root not in path.parents or path.exists()
           for path in (target, stage_dir, transaction_path)):
        raise FileExistsError("Remaining-v3 output/transaction exists or escapes root")
    stage_dir.mkdir(parents=True, exist_ok=False)
    check_frozen(root, frozen, published_sha)
    check_prior(root, prior)
    commands = parent.stage_commands(stage, stage_dir)
    with (stage_dir / "child.log").open("xb") as log:
        for command in commands:
            parent.free_memory()
            check_frozen(root, frozen, published_sha)
            check_prior(root, prior)
            result = subprocess.run(command, cwd=root, stdout=log,
                                    stderr=subprocess.STDOUT, check=False, shell=False)
            log.flush()
            if result.returncode != 0:
                raise RuntimeError(f"Original {stage} child exited {result.returncode}; no resume")
            check_frozen(root, frozen, published_sha)
            check_prior(root, prior)
    proof = (prove_recomputed_stage(root, stage, stage_dir, frozen, prior, published_sha)
             if stage in RECOMPUTE_STAGES else prove_model_stage(root, stage, stage_dir))
    check_frozen(root, frozen, published_sha)
    check_prior(root, prior)
    outputs = output_inventory(stage_dir, stage)
    if set(outputs) != set(proof["output_names"]):
        raise ValueError("Independently proved output inventory differs")
    hashes = {name: sha256(path) for name, path in outputs.items()}
    for name, path in outputs.items():
        destination = (target / name).resolve(strict=False)
        if root not in destination.parents or destination.exists():
            raise FileExistsError("Canonical output exists or escapes root")
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.link(path, destination)
    for name, digest in hashes.items():
        if sha256(target / name) != digest:
            raise ValueError("Promoted remaining-v3 output byte differs")
    transaction = {"schema_version": 1, "stage": stage, "status": "complete",
                   "adapter_sha256": published_sha,
                   "source_and_raw_snapshot": frozen, "prior_snapshot": prior,
                   "commands": commands,
                   "child_log_sha256": sha256(stage_dir / "child.log"),
                   "proof": proof, "output_sha256": hashes,
                   "exclusive_promotion_verified": True}
    check_frozen(root, frozen, published_sha)
    check_prior(root, prior)
    parent.exclusive_json(transaction_path, transaction, root)
    return {"stage": stage, "status": "complete",
            "transaction_sha256": sha256(transaction_path),
            "output_files": len(hashes)}


def verify_stage(root: Path, stage: str, published_sha: str,
                 frozen: dict) -> dict:
    if stage not in STAGES:
        raise RuntimeError(f"Unsupported remaining-v3 stage {stage} refuses")
    parent.free_memory()
    transaction = parent.read_json(parent.child(
        root, f"{TRANSACTION_BASE}/{stage}.json"))
    expected_prior = prior_snapshot(root, stage, frozen, published_sha)
    if (transaction.get("schema_version") != 1 or transaction.get("stage") != stage
            or transaction.get("status") != "complete"
            or transaction.get("adapter_sha256") != published_sha
            or transaction.get("source_and_raw_snapshot") != frozen
            or transaction.get("prior_snapshot") != expected_prior
            or transaction.get("commands") != parent.stage_commands(
                stage, root / STAGE_BASE / stage)
            or transaction.get("exclusive_promotion_verified") is not True
            or transaction.get("child_log_sha256") != sha256(parent.child(
                root, f"{STAGE_BASE}/{stage}/child.log"))):
        raise ValueError(f"{stage} actual-genesis receipt changed")
    outputs = output_inventory(root / parent.OUTPUT_DIR[stage], stage)
    if transaction.get("output_sha256") != {name: sha256(path)
                                                  for name, path in outputs.items()}:
        raise ValueError(f"{stage} promoted output bytes changed")
    if stage in RECOMPUTE_STAGES:
        replay = verify_recomputed_stage(root, stage, transaction)
    else:
        replay = prove_model_stage(root, stage, root / parent.OUTPUT_DIR[stage])
        if replay != transaction.get("proof"):
            raise ValueError(f"{stage} saved-model OOF/ranking replay differs")
    if set(outputs) != set(replay["output_names"]):
        raise ValueError(f"{stage} verified output inventory differs")
    check_frozen(root, frozen, published_sha)
    check_prior(root, expected_prior)
    return transaction


def require_reviewed_adapter(published_sha: str, reviewed_sha: str | None) -> None:
    """Fail before values when no exact reviewed publication pin exists."""
    parent.sha_required(published_sha, "published remaining-v3 adapter")
    if reviewed_sha is None:
        raise RuntimeError("No reviewed remaining-v3 proof adapter is pinned; "
                           "the full receipt refuses before reading inputs")
    parent.sha_required(reviewed_sha, "reviewed remaining-v3 adapter")
    if reviewed_sha != published_sha:
        raise ValueError("Published remaining-v3 source differs from reviewed pin")


def emit_full_v3_receipt(root_path: Path, published_sha: str) -> dict:
    """Complete parent proof only after every original v3 stage actually ran."""
    reviewed = parent.read_json(SOURCE_ROOT / "reports/clean_parent_producers_spec.json").get(
        "reviewed_v3_proof_adapter_sha256")
    require_reviewed_adapter(published_sha, reviewed)
    require_source(published_sha, git=True)
    import replica_v4_timestamp as timestamp

    root = parent.isolated_root(root_path)
    parent.free_memory()
    frozen = frozen_inputs(root, published_sha)
    target = root / "parents/v3_producer_receipt.json"
    if target.exists():
        raise FileExistsError("Full v3 producer receipt already exists")
    baseline_spec = parent.read_json(SPEC)
    baseline_frozen = baseline.frozen_inputs(root, baseline_spec["baseline_proof_adapter_sha256"])
    baseline_tx = {stage: baseline.verify_transaction(
        root, stage, baseline_spec["baseline_proof_adapter_sha256"], baseline_frozen)
        for stage in baseline.SUPPORTED}
    transactions = {**baseline_tx,
                    **{stage: verify_stage(root, stage, published_sha, frozen)
                       for stage in STAGES}}
    if set(transactions) != set(parent.V3_STAGES):
        raise ValueError("Not all eleven original v3 transactions were proved")
    intermediate = {}
    model_files = {}
    output_sha = {}
    for stage in parent.V3_STAGES:
        transaction = transactions[stage]
        for filename, digest in transaction["output_sha256"].items():
            relative = f"{parent.OUTPUT_DIR[stage]}/{filename}"
            role = f"{stage}:{filename}"
            if role in intermediate:
                raise ValueError("Duplicate intermediate v3 output role")
            intermediate[role] = {"path": relative, "sha256": digest}
            if filename.endswith((".txt", ".cbm")):
                safe = re.sub(r"[^a-z0-9]+", "_", f"{stage}_{filename}".lower()).strip("_")
                model_role = f"v3_model_{safe}"
                if model_role in model_files:
                    raise ValueError("Duplicate saved v3 model role")
                model_files[model_role] = {"path": relative, "sha256": digest}
                output_sha[model_role] = digest
    if not model_files:
        raise ValueError("No actual saved v3 model bytes")
    for role in ("v3_validation_oof", "v3_ranking_predictions", "v3_validation_report"):
        output_sha[role] = sha256(parent.child(root, timestamp.CANONICAL[role]))
    input_sha = dict(frozen["clean_raw_and_source"]["raw_file_sha256"])
    for role in ("baseline_training_rows", "baseline_training_features",
                 "baseline_ranking_rows", "baseline_ranking_features"):
        input_sha[role] = sha256(parent.child(root, timestamp.CANONICAL[role]))
    oof = pd.read_parquet(parent.child(root, timestamp.CANONICAL["v3_validation_oof"]))
    training_rows = pd.read_parquet(parent.child(root, timestamp.CANONICAL[
        "baseline_training_rows"]), columns=["MVT_ID_mvt", "target", "month"])
    if (oof.MVT_ID_mvt.isna().any() or oof.MVT_ID_mvt.duplicated().any()
            or not np.isfinite(oof.target.to_numpy(dtype=float)).all()):
        raise ValueError("Final v3 OOF IDs/labels are incomplete")
    valid_month = training_rows.month.isin((1, 7, 11, 12))
    labeled = np.isfinite(training_rows.target.to_numpy(dtype=float))
    held = training_rows.loc[valid_month & labeled,
                             ["MVT_ID_mvt", "target", "month"]]
    if len(oof) != len(held) or set(oof.MVT_ID_mvt) != set(held.MVT_ID_mvt):
        raise ValueError("Final v3 OOF heldout ID set differs from sealed labels")
    matched = oof[["MVT_ID_mvt", "target", "fold"]].merge(
        held, on="MVT_ID_mvt", validate="one_to_one", suffixes=("", "_baseline"))
    if (not np.array_equal(matched.target.to_numpy(dtype=np.float32),
                           matched.target_baseline.to_numpy(dtype=np.float32))
            or not matched.fold.eq(np.where(matched.month.isin((1, 7)),
                                            "seasonal_jan_jul", "forward_nov_dec")).all()):
        raise ValueError("Final v3 OOF targets or heldout folds differ")
    rank = pd.read_parquet(parent.child(root, timestamp.CANONICAL["v3_ranking_predictions"]))
    template = pd.read_parquet(parent.child(root, timestamp.CANONICAL["submission_template"]),
                               columns=["MVT_ID_mvt"])
    if (len(rank) != 344_841 or not rank.MVT_ID_mvt.equals(template.MVT_ID_mvt)
            or not np.isfinite(rank.TAXITIME_SEC_mvt.to_numpy(dtype=float)).all()
            or (rank.TAXITIME_SEC_mvt.to_numpy(dtype=float) < 0).any()):
        raise ValueError("Final v3 ranking output is not exact template coverage")
    final_report = parent.read_json(parent.child(root, timestamp.CANONICAL[
        "v3_validation_report"]))
    if (final_report.get("selected_valid_disagreement_weight") != 1.0 or
            final_report.get("invalid_aobt_lobt_fallback_weight") != 1.0 or
            final_report.get("valid_disagreement_threshold_sec") != 3600):
        raise ValueError("Final v3 heldout-selected scientific route differs")
    source = {name: digest for name, digest in
              frozen["clean_raw_and_source"]["source_sha256"].items()
              if name.endswith(".py")}
    source.update({name: digest for name, digest in frozen["source_sha256"].items()
                   if name.endswith(".py")})
    receipt = {
        "name": "v3", "status": "complete",
        "v3_proof_adapter_sha256": published_sha,
        "source_commit": frozen["scientific_source_commit"],
        "source_sha256": source,
        "input_sha256": input_sha,
        "output_sha256": output_sha,
        "intermediate_transaction_sha256": {
            stage: sha256(parent.child(root, f"{TRANSACTION_BASE}/{stage}.json"))
            for stage in parent.V3_STAGES},
        "intermediate_files": intermediate,
        "model_files": model_files,
        "heldout_folds": {name: list(months) for name, months in baseline.FOLDS.items()},
        "legacy_validation_erratum_sha256": parent.LEGACY_ERRATUM_SHA256,
        **parent.read_json(SOURCE_ROOT / parent.LEGACY_ERRATUM_REL)[
            "legacy_v3_receipt_policy"],
        "independent_model_and_policy_replay_passed": True,
        "full_intermediate_replay_status": "passed",
        "published_choices": {
            "route": "lobt_ensemble",
            "selected_valid_disagreement_weight": 1.0,
            "invalid_aobt_lobt_fallback_weight": 1.0,
            "valid_disagreement_threshold_sec": 3600,
        },
        "ordered_validation_ids_sha256": parent.ids_sha(oof.MVT_ID_mvt),
        "ordered_ranking_ids_sha256": parent.ids_sha(template.MVT_ID_mvt),
        "feature_schema_sha256": baseline_tx["baseline_train"]["proof"][
            "feature_schema_sha256"],
    }
    check_frozen(root, frozen, published_sha)
    check_prior(root, {stage: {"transaction_sha256": receipt[
        "intermediate_transaction_sha256"][stage],
        "output_sha256": transactions[stage]["output_sha256"]}
        for stage in parent.V3_STAGES})
    parent.exclusive_json(target, receipt, root)
    parent.verify_external_v3(root, frozen["clean_raw_and_source"])
    return {"status": "complete", "v3_receipt_sha256": sha256(target),
            "verified_stages": len(transactions), "saved_models": len(model_files),
            "validation_rows": len(oof), "ranking_rows": len(rank)}


def plan() -> dict:
    require_source()
    return {"status": "prospective_only", "ordered_stages": list(parent.V3_STAGES),
            "current_recomputed_stages": list(RECOMPUTE_STAGES),
            "saved_model_stages": list(MODEL_STAGES),
            "full_v3_receipt_status": "refused_until_corrected_contract_and_adapter_published",
            "real_data_models_or_child_processes_read": False}


def self_test() -> dict:
    require_source()
    if set(RECOMPUTE_STAGES) | set(MODEL_STAGES) != set(STAGES):
        raise AssertionError("Remaining-v3 stage closure differs")
    if any(stage in baseline.SUPPORTED for stage in STAGES):
        raise AssertionError("Baseline and remaining stage adapters overlap")
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory).resolve()
        try:
            parent.child(root, "../outside", existing=False)
        except ValueError:
            pass
        else:
            raise AssertionError("Parent traversal was allowed")
        left = root / "left.parquet"
        right = root / "right.parquet"
        tiny = pd.DataFrame({"MVT_ID_mvt": [11, 12], "target": [1.0, 2.0]})
        tiny.to_parquet(left, index=False)
        tiny.to_parquet(right, index=False)
        compare_recomputed_values(left, right, "synthetic saved-policy replay")
        tiny.loc[1, "target"] = 7.0
        tiny.to_parquet(right, index=False)
        try:
            compare_recomputed_values(left, right, "synthetic tamper")
        except ValueError:
            pass
        else:
            raise AssertionError("Changed replay value was accepted")
    reviewed = parent.read_json(SOURCE_ROOT / "reports/clean_parent_producers_spec.json").get(
        "reviewed_v3_proof_adapter_sha256")
    wrong = "0" * 64 if reviewed != "0" * 64 else "1" * 64
    for real_entry in (frozen_inputs,
                       lambda root, digest: emit_full_v3_receipt(root, digest)):
        try:
            real_entry(Path("synthetic-unread-root"), wrong)
        except (RuntimeError, ValueError) as exc:
            expected = ("No reviewed remaining-v3" if reviewed is None else
                        "Published remaining-v3 source differs from reviewed pin")
            if expected not in str(exc):
                raise
        else:
            raise AssertionError("Unreviewed remaining-v3 real entry was accepted")
    require_reviewed_adapter("1" * 64, "1" * 64)
    try:
        require_reviewed_adapter("1" * 64, "2" * 64)
    except ValueError:
        pass
    else:
        raise AssertionError("Mismatched reviewed source pin was accepted")
    return {"source_and_spec_pin": "passed", "eleven_stage_closure": "passed",
            "stage_closure_and_path": "passed",
            "recomputed_value_tamper_and_real_mode_refusal": "passed",
            "real_values_or_models_read": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("plan", "self-test", "run-stage",
                                           "verify-stage", "emit-v3"), default="plan")
    parser.add_argument("--stage", choices=STAGES)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--published-source-sha256")
    args = parser.parse_args()
    if args.mode == "plan":
        result = plan()
    elif args.mode == "self-test":
        result = self_test()
    else:
        if args.run_root is None or args.published_source_sha256 is None:
            parser.error("Real modes require isolated root and published SHA")
        if args.mode == "emit-v3":
            result = emit_full_v3_receipt(args.run_root, args.published_source_sha256)
        else:
            if args.stage is None:
                parser.error("Stage execution/verification requires --stage")
            if args.mode == "run-stage":
                result = run_stage(args.run_root, args.stage, args.published_source_sha256)
            else:
                root = parent.isolated_root(args.run_root)
                parent.free_memory()
                frozen = frozen_inputs(root, args.published_source_sha256)
                tx = verify_stage(root, args.stage, args.published_source_sha256, frozen)
                result = {"stage": args.stage, "status": "verified_actual_genesis",
                          "transaction_sha256": sha256(root / TRANSACTION_BASE /
                                                        f"{args.stage}.json"),
                          "output_files": len(tx["output_sha256"])}
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
