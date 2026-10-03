"""Prospective isolated v3/GPU/source producer driver for clean timestamp parents.

Only ``plan`` and ``self-test`` may run before independent review/publication.
Every real mode requires an operator-attested published source SHA. The v3
scientific OOF/model replay adapters are deliberately pending; invoking those
stages refuses *before* launching a child process or reading private values.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SOURCE_ROOT = Path(__file__).resolve().parent
SPEC = SOURCE_ROOT / "reports/clean_parent_producers_spec.json"
LEGACY_ERRATUM_REL = "reports/clean_replication_legacy_validation_erratum.json"
LEGACY_ERRATUM_SHA256 = "209176e6e75453aa18d68406992f6ac17994622d9244b06a341738acbe99a4f3"
REVIEWED_V3_PROOF_REL = "replica_v3_remaining.py"
REVIEWED_V3_PROOF_SHA256 = "545e5ab1c077a219f2a74087b16cc3f4dada49091bae9ee3c1b3042a7f390806"
MIN_FREE_GIB = 10.0
FOLDS = {"seasonal_jan_jul": (1, 7), "forward_nov_dec": (11, 12)}
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
HEX40 = re.compile(r"[0-9a-f]{40}\Z")

SOURCE_CLOSURE = (
    "solution.py", "missing_expert.py", "grouped_proxy.py", "weather_model.py",
    "airport_models.py", "schedule_tail.py", "lirf_expert.py", "ensemble.py",
    "lobt_expert.py", "lobt_blend.py", "catboost_gpu.py",
    "catboost_source.py", "catboost_expert.py", "download_noaa_weather.py",
    "replica_v4_timestamp.py", "reports/clean_replication_timestamp_spec.json",
    REVIEWED_V3_PROOF_REL,
    "LICENSE", "requirements-lock.txt",
    "requirements-research.txt",
)


def month_path(month: int) -> str:
    end_year, end_month = (2026, 1) if month == 12 else (2025, month + 1)
    return f"data/training_2025-{month:02d}-01_{end_year}-{end_month:02d}-01.parquet"


RAW = {**{f"raw_training_{month:02d}": month_path(month) for month in range(1, 13)},
       "raw_ranking": "data/ranking.parquet",
       "submission_template": "data/submitting.parquet",
       "noaa_weather": "data/external/weather.parquet"}

REQUIRED_OUTPUTS = {
    "baseline_train": ("features.parquet", "training_rows.parquet",
                       "seasonal_jan_jul_oof.parquet", "forward_nov_dec_oof.parquet",
                       "validation.json", "model.json", "direct.txt"),
    "baseline_predict": ("ranking_features.parquet", "ranking_rows.parquet",
                         "ranking_predictions.parquet"),
    "missing": ("seasonal_jan_jul_oof.parquet", "forward_nov_dec_oof.parquet",
                "predictions.parquet", "validation.json", "model.json"),
    "grouped": ("validation_predictions.parquet", "forward_validation_predictions.parquet",
                "predictions.parquet", "validation.json", "forward_validation.json", "model.json"),
    "weather": ("seasonal_jan_jul_oof.parquet", "forward_nov_dec_oof.parquet",
                "ranking_predictions.parquet", "model.json", "residual.txt"),
    "airports": ("seasonal_jan_jul_oof.parquet", "forward_nov_dec_oof.parquet",
                 "ranking_predictions.parquet", "validation.json", "model.json"),
    "tail": ("oof.parquet", "ranking.parquet", "report.json"),
    "lirf": ("oof.parquet", "ranking.parquet", "validation.json"),
    "ensemble": ("validation_predictions.parquet", "predictions.parquet",
                 "validation.json", "model.json"),
    "lobt": ("seasonal_jan_jul_oof.parquet", "forward_nov_dec_oof.parquet",
             "ranking_predictions.parquet", "validation.json", "residual.txt"),
    "lobt_blend": ("validation_predictions.parquet", "predictions.parquet",
                   "validation.json", "model.json"),
    "gpu_fit": ("seasonal_jan_jul.cbm", "seasonal_jan_jul_oof.parquet",
                "seasonal_jan_jul_validation.json", "forward_nov_dec.cbm",
                "forward_nov_dec_oof.parquet", "forward_nov_dec_validation.json",
                "validation.json"),
    "gpu_final": ("full_2025.cbm", "ranking_expert.parquet",
                  "predictions.parquet", "final_report.json"),
    "source_fit": ("seasonal_jan_jul.cbm", "seasonal_jan_jul_oof.parquet",
                   "seasonal_jan_jul_validation.json", "forward_nov_dec.cbm",
                   "forward_nov_dec_oof.parquet", "forward_nov_dec_validation.json",
                   "validation.json", "validation_combined.json",
                   "validation_predictions.parquet", "validation_sequential_gpu.json"),
    "source_final": ("final.cbm", "ranking_source_probabilities.parquet",
                     "predictions.parquet", "manifest.json",
                     "sequential_predictions.parquet", "sequential_manifest.json"),
}
OPTIONAL_OUTPUTS = {"baseline_train": ("residual.txt",)}

OUTPUT_DIR = {
    "baseline_train": "artifacts/baseline", "baseline_predict": "artifacts/baseline",
    "missing": "artifacts/missing", "grouped": "artifacts/grouped",
    "weather": "artifacts/weather", "airports": "artifacts/airports",
    "tail": "artifacts/tail", "lirf": "artifacts/lirf",
    "ensemble": "artifacts/ensemble", "lobt": "artifacts/lobt",
    "lobt_blend": "artifacts/lobt_ensemble",
    "gpu_fit": "artifacts/catboost/gpu", "gpu_final": "artifacts/catboost/gpu",
    "source_fit": "artifacts/catboost/source",
    "source_final": "artifacts/catboost/source",
}

V3_STAGES = ("baseline_train", "baseline_predict", "missing", "grouped",
             "weather", "airports", "tail", "lirf", "ensemble", "lobt", "lobt_blend")
ORDER = V3_STAGES + ("gpu_fit", "source_fit", "gpu_final", "source_final")
PENDING_V3_PROOF = {
    "baseline_train": "saved direct/residual LightGBM fold replay and full cache/label reconstruction",
    "baseline_predict": "full baseline model ranking replay and exact template/proxy gate",
    "missing": "saved direct/schedule LightGBM fold and ranking replay",
    "grouped": "nested grouped proxy policy and target-free ranking replay",
    "weather": "saved weather fold/full LightGBM replay including complete-OOF expansion",
    "airports": "all airport fold/full LightGBM model and OOF replay",
    "tail": "deterministic schedule-tail fit-only statistics and OOF/ranking replay",
    "lirf": "Rome specialist split/model and OOF/ranking replay",
    "ensemble": "nested expert selection, OOF/ranking full formula replay",
    "lobt": "saved LOBT LightGBM fold/full model and OOF/ranking replay",
    "lobt_blend": "nested LOBT selection, accepted weight1 and OOF/ranking full formula replay",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def value_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def ids_sha(ids: Any) -> str:
    a = np.asarray(ids, dtype=np.int64)
    if a.ndim != 1:
        raise ValueError("ID order must be 1D")
    return hashlib.sha256(a.astype("<i8", copy=False).tobytes()).hexdigest()


def float_sha(values: Any) -> str:
    a = np.asarray(values, dtype="<f8")
    if a.ndim != 1 or not np.isfinite(a).all():
        raise ValueError("A sealed prediction vector must be finite")
    return hashlib.sha256(a.tobytes()).hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def sha_required(value: Any, name: str, *, commit: bool = False) -> None:
    if not isinstance(value, str) or not (HEX40 if commit else HEX64).fullmatch(value):
        raise ValueError(f"Invalid {name} SHA")


def isolated_root(path: Path) -> Path:
    root = Path(path).resolve(strict=True)
    source = SOURCE_ROOT.resolve(strict=True)
    if root == source or root in source.parents or source in root.parents:
        raise ValueError("Clean producer root must be disjoint from source checkout")
    return root


def child(root: Path, relative: str, *, existing: bool = True) -> Path:
    name = Path(relative)
    if not relative or name.is_absolute() or ".." in name.parts:
        raise ValueError("Expected run-root-relative path")
    path = (root / name).resolve(strict=existing)
    if root not in path.parents:
        raise ValueError("Private file escapes isolated root")
    if existing and not path.is_file():
        raise FileNotFoundError(path)
    return path


def source_snapshot(expected_published_sha: str | None = None) -> dict[str, str]:
    spec = read_json(SPEC)
    if spec.get("schema_version") != 1:
        raise ValueError("Clean producer spec version differs")
    actual = {name: sha256(SOURCE_ROOT / name) for name in SOURCE_CLOSURE}
    if spec.get("frozen_original_source_sha256") != actual:
        raise ValueError("A published original producer source changed")
    if (spec.get("legacy_validation_erratum_sha256") != LEGACY_ERRATUM_SHA256
            or sha256(SOURCE_ROOT / LEGACY_ERRATUM_REL) != LEGACY_ERRATUM_SHA256):
        raise ValueError("Published clean legacy-validation erratum changed")
    if (spec.get("reviewed_v3_proof_adapter_sha256") != REVIEWED_V3_PROOF_SHA256
            or spec.get("upstream_v3_gap", {}).get(
                "reviewed_v3_proof_adapter_sha256") != REVIEWED_V3_PROOF_SHA256):
        raise ValueError("Published reviewed full-v3 adapter pin changed")
    actual[LEGACY_ERRATUM_REL] = LEGACY_ERRATUM_SHA256
    actual["reports/clean_parent_producers_spec.json"] = sha256(SPEC)
    actual["clean_parent_producers.py"] = sha256(Path(__file__).resolve())
    if expected_published_sha is not None:
        sha_required(expected_published_sha, "operator-published producer")
        if actual["clean_parent_producers.py"] != expected_published_sha:
            raise ValueError("Producer adapter differs from published SHA attestation")
    return actual


def free_memory() -> None:
    import psutil
    available = psutil.virtual_memory().available / (1024 ** 3)
    if available < MIN_FREE_GIB:
        raise MemoryError(f"Need at least {MIN_FREE_GIB:g} GiB free RAM; observed {available:.2f} GiB")


def exclusive_bytes(path: Path, payload: bytes, root: Path) -> None:
    path = path.resolve(strict=False)
    if root not in path.parents or path.exists():
        raise FileExistsError("Output is outside isolated root or already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".clean-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def exclusive_json(path: Path, value: dict, root: Path) -> None:
    exclusive_bytes(path, json.dumps(value, indent=2, allow_nan=False).encode("utf-8"), root)


def json_sha(value: dict) -> str:
    """Hash the exact bytes written by exclusive_json, before publication."""
    return hashlib.sha256(json.dumps(value, indent=2, allow_nan=False).encode("utf-8")).hexdigest()


def raw_manifest(root: Path) -> tuple[dict, dict[str, Path]]:
    manifest_path = child(root, "parents/clean_raw_weather_inputs.json")
    value = read_json(manifest_path)
    if (value.get("schema_version") != 1 or value.get("status") != "complete"
            or not isinstance(value.get("files"), dict)
            or set(value["files"]) != set(RAW)):
        raise ValueError("Complete isolated raw/weather manifest required")
    if value.get("weather_license") != "NOAA GHCNh CC0-1.0":
        raise ValueError("Licensed independently prepared NOAA GHCNh weather provenance required")
    weather = value.get("weather_source_receipt")
    if not isinstance(weather, dict) or not weather.get("source_urls") or not weather.get("station_year_sha256"):
        raise ValueError("Weather station-year source URL and SHA receipt required")
    urls = weather["source_urls"]
    hashes = weather["station_year_sha256"]
    if (not isinstance(urls, list) or not isinstance(hashes, dict)
            or any(not isinstance(url, str) or not url.startswith(
                "https://www.ncei.noaa.gov/") for url in urls)):
        raise ValueError("Weather source URLs must identify official NOAA NCEI files")
    for station_year, digest in hashes.items():
        if not isinstance(station_year, str) or not station_year:
            raise ValueError("Weather station-year key missing")
        sha_required(digest, station_year)
    files = {}
    for role, canonical in RAW.items():
        item = value["files"][role]
        if item.get("path") != canonical:
            raise ValueError(f"Noncanonical isolated {role} path")
        sha_required(item.get("sha256"), role)
        path = child(root, canonical)
        if sha256(path) != item["sha256"]:
            raise ValueError(f"{role} changed from independent raw/weather receipt")
        files[role] = path
    discovered = set((root / "data").glob("training_2025-*.parquet"))
    if discovered != {files[f"raw_training_{m:02d}"] for m in range(1, 13)}:
        raise ValueError("Original loader would read an unsealed training month")
    return value, files


def snapshot(root: Path, published_sha: str) -> dict:
    source = source_snapshot(published_sha)
    raw, paths = raw_manifest(root)
    return {"source_sha256": source,
            "raw_manifest_sha256": sha256(child(root, "parents/clean_raw_weather_inputs.json")),
            "raw_file_sha256": {role: sha256(path) for role, path in paths.items()}}


def check_snapshot(root: Path, expected: dict, published_sha: str) -> None:
    if snapshot(root, published_sha) != expected:
        raise ValueError("Source/raw/weather changed during clean producer stage")


def stage_commands(stage: str, output: Path) -> list[list[str]]:
    """Exact original CLI arguments; all relative inputs resolve under isolated cwd."""
    if stage not in ORDER:
        raise ValueError("Unknown clean producer stage")
    output = output.resolve(strict=False)
    base = [sys.executable]
    def call(script: str, *arguments: str) -> list[str]:
        return base + [str(SOURCE_ROOT / script), *arguments]
    out = str(output)
    if stage == "baseline_train":
        return [call("solution.py", "train", "--data-dir", "data", "--output-dir", out,
                     "--threads", "8", "--rounds", "900")]
    if stage == "baseline_predict":
        return [call("solution.py", "predict", "--data-dir", "data", "--model-dir",
                     out, "--team", "merry-mushroom", "--version", "1",
                     "--output-dir", str(output / "proposal"), "--threads", "8")]
    if stage == "missing":
        return [call("missing_expert.py", "--data-dir", "data", "--cache-dir",
                     "artifacts/baseline", "--output-dir", out, "--threads", "2",
                     "--rounds", "1400")]
    if stage == "grouped":
        return [call("grouped_proxy.py", "--data-dir", "data", "--output-dir", out)]
    if stage == "weather":
        return [call("weather_model.py", mode, "--baseline-dir", "artifacts/baseline",
                     "--weather-file", "data/external/weather.parquet", "--output-dir",
                     out, "--threads", "8", "--rounds", "1500")
                for mode in ("train", "complete-oof", "predict-only")]
    if stage == "airports":
        return [call("airport_models.py", "--data-dir", "data", "--baseline-dir",
                     "artifacts/baseline", "--output-dir", out, "--threads", "6",
                     "--rounds", "1000")]
    if stage == "tail":
        return [call("schedule_tail.py", "--data-dir", "data", "--output-dir", out)]
    if stage == "lirf":
        return [call("lirf_expert.py", "--data-dir", "data", "--cache-dir",
                     "artifacts/baseline", "--comparison-dir", "artifacts/missing",
                     "--output-dir", out, "--threads", "2", "--rounds", "220")]
    if stage == "ensemble":
        return [call("ensemble.py", "--data-dir", "data", "--baseline-dir",
                     "artifacts/baseline", "--missing-dir", "artifacts/missing",
                     "--tail-dir", "artifacts/tail", "--lirf-dir", "artifacts/lirf",
                     "--weather-dir", "artifacts/weather", "--airports-dir", "artifacts/airports",
                     "--grouped-dir", "artifacts/grouped", "--baseline-ranking",
                     "artifacts/baseline/ranking_predictions.parquet", "--output-dir", out)]
    if stage == "lobt":
        return [call("lobt_expert.py", "--data-dir", "data", "--cache-dir",
                     "artifacts/baseline", "--weather-file", "data/external/weather.parquet",
                     "--ensemble-dir", "artifacts/ensemble", "--output-dir", out,
                     "--threads", "8", "--rounds", "900")]
    if stage == "lobt_blend":
        return [call("lobt_blend.py", "--ensemble-dir", "artifacts/ensemble",
                     "--lobt-dir", "artifacts/lobt", "--data-dir", "data",
                     "--output-dir", out)]
    common_gpu = ("--data-dir", "data", "--cache-dir", "artifacts/baseline",
                  "--weather-file", "data/external/weather.parquet", "--v3-dir",
                  "artifacts/lobt_ensemble", "--output-dir", out,
                  "--iterations", "1500", "--depth", "8", "--threads", "2",
                  "--seed", "2026")
    if stage == "gpu_fit":
        return [call("catboost_gpu.py", "--mode", "fit", *common_gpu)]
    if stage == "gpu_final":
        return [call("catboost_gpu.py", "--mode", "final-predict", *common_gpu)]
    common_source = ("--data-dir", "data", "--cache-dir", "artifacts/baseline",
                     "--v3-dir", "artifacts/lobt_ensemble", "--gpu-dir",
                     "artifacts/catboost/gpu", "--weather-file",
                     "data/external/weather.parquet", "--output-dir", out,
                     "--iterations", "600", "--depth", "7", "--threads", "4",
                     "--seed", "2026")
    if stage == "source_fit":
        return [call("catboost_source.py", "--fold", "both", *common_source),
                call("catboost_source.py", "--evaluate-only", *common_source),
                call("catboost_source.py", "--evaluate-sequential-gpu", *common_source)]
    return [call("catboost_source.py", "--fit-final-rank", *common_source),
            call("catboost_source.py", "--combine-gpu", *common_source)]


def plan() -> dict:
    """Static source-only plan; no raw/cache/parent value read."""
    example_stage = SOURCE_ROOT.parent / "CLEAN_RUNROOT_PLACEHOLDER" / "artifacts/_clean_parent_stage"
    return {"status": "prospective_only", "source_checkout": str(SOURCE_ROOT),
            "isolated_root_required": True,
            "inputs": RAW, "stage_order": list(ORDER),
            "canonical_output_dirs": OUTPUT_DIR,
            "required_outputs": {name: list(paths) for name, paths in REQUIRED_OUTPUTS.items()},
            "pending_v3_proofs": PENDING_V3_PROOF,
            "weather_downloader_refused": "download_noaa_weather.py writes beside __file__; independently prepared NOAA CC0 weather under isolated root is required",
            "gpu_and_source_commands": {name: stage_commands(name, example_stage / name)
                                        for name in ("gpu_fit", "gpu_final", "source_fit", "source_final")},
            "real_modes_require_published_source_and_complete_independent_inputs": True}


def schema_of(features: pd.DataFrame) -> dict:
    categories = {}
    for name in features.select_dtypes(include="category"):
        cat = features[name].cat
        categories[name] = value_sha({"values": [str(v) for v in cat.categories],
                                      "dtype": str(cat.categories.dtype),
                                      "ordered": bool(cat.ordered)})
    return {"columns": [{"name": str(name), "dtype": str(features[name].dtype)}
                        for name in features.columns],
            "categorical_columns": list(categories),
            "category_vocabulary_sha256": categories}


def canonical_prior(root: Path, stage: str) -> dict[str, Path]:
    """Read-only closure of already completed, independently proved inputs."""
    from replica_v4_timestamp import CANONICAL as TIMESTAMP_FILES
    raw, raw_files = raw_manifest(root)
    del raw
    named = dict(raw_files)
    index = ORDER.index(stage)
    for prior in ORDER[:index]:
        if prior in V3_STAGES:
            # The accepted v3 receipt must enumerate these intermediate bytes;
            # a directory listing alone is not enough scientific provenance.
            continue
        for filename in REQUIRED_OUTPUTS[prior]:
            path = child(root, f"{OUTPUT_DIR[prior]}/{filename}")
            named[f"{prior}:{filename}"] = path
        named[f"{prior}:transaction"] = child(root, f"parents/transactions/{prior}.json")
    for role in ("baseline_training_rows", "baseline_training_features",
                 "baseline_ranking_rows", "baseline_ranking_features",
                 "v3_validation_oof", "v3_ranking_predictions",
                 "v3_validation_report", "v3_producer_receipt"):
        named[role] = child(root, TIMESTAMP_FILES[role])
    if stage in ("source_fit", "gpu_final", "source_final"):
        for filename in REQUIRED_OUTPUTS["gpu_fit"]:
            named[f"gpu_fit:{filename}"] = child(root, f"{OUTPUT_DIR['gpu_fit']}/{filename}")
        named["gpu_fit:transaction"] = child(root, "parents/transactions/gpu_fit.json")
    if stage == "source_final":
        for filename in REQUIRED_OUTPUTS["source_fit"]:
            named[f"source_fit:{filename}"] = child(root, f"{OUTPUT_DIR['source_fit']}/{filename}")
        named["source_fit:transaction"] = child(root, "parents/transactions/source_fit.json")
        for filename in REQUIRED_OUTPUTS["gpu_final"]:
            named[f"gpu_final:{filename}"] = child(root, f"{OUTPUT_DIR['gpu_final']}/{filename}")
        named["gpu_final:transaction"] = child(root, "parents/transactions/gpu_final.json")
    return named


def check_legacy_v3_claim(receipt: dict) -> None:
    """Accept the source-derived legacy split disclosure, never an untouched-fold claim."""
    if sha256(SOURCE_ROOT / LEGACY_ERRATUM_REL) != LEGACY_ERRATUM_SHA256:
        raise ValueError("Published legacy-validation erratum changed")
    expected = read_json(SOURCE_ROOT / LEGACY_ERRATUM_REL)["legacy_v3_receipt_policy"]
    if (receipt.get("legacy_validation_erratum_sha256") != LEGACY_ERRATUM_SHA256
            or any(receipt.get(key) != expected[key] for key in (
                "fit_excludes_heldout", "early_stop_uses_heldout",
                "fit_and_early_exclude_heldout", "validation_interpretation"))
            or receipt.get("stage_validation_policy") != expected["stage_validation_policy"]):
        raise ValueError("Original v3 heldout/early-stop policy is misstated")


def verify_external_v3(root: Path, raw_snapshot: dict) -> dict:
    """Demand an independently generated full-v3 replay receipt; never infer it."""
    from replica_v4_timestamp import CANONICAL as TIMESTAMP_FILES
    path = child(root, TIMESTAMP_FILES["v3_producer_receipt"])
    receipt = read_json(path)
    approved = read_json(SPEC).get("reviewed_v3_proof_adapter_sha256")
    if approved != REVIEWED_V3_PROOF_SHA256:
        raise RuntimeError("Reviewed full-v3 adapter pin differs; "
                           "v3 receipt acceptance and all downstream fits refuse")
    sha_required(approved, "reviewed v3 proof adapter")
    if receipt.get("v3_proof_adapter_sha256") != approved:
        raise ValueError("v3 receipt did not come from the pinned reviewed replay adapter")
    if (receipt.get("name") != "v3" or receipt.get("status") != "complete"
            or receipt.get("published_choices") != {
                "route": "lobt_ensemble", "selected_valid_disagreement_weight": 1.0,
                "invalid_aobt_lobt_fallback_weight": 1.0,
                "valid_disagreement_threshold_sec": 3600}
            or receipt.get("heldout_folds") != {key: list(value) for key, value in FOLDS.items()}
            or receipt.get("independent_model_and_policy_replay_passed") is not True
            or receipt.get("full_intermediate_replay_status") != "passed"):
        raise ValueError("Reviewed independent full-v3 model/OOF/ranking proof is absent")
    check_legacy_v3_claim(receipt)
    proof = receipt.get("intermediate_transaction_sha256")
    if not isinstance(proof, dict) or set(proof) != set(V3_STAGES):
        raise ValueError("Full v3 transaction chain is absent")
    for stage, digest in proof.items():
        sha_required(digest, f"v3 {stage} transaction")
        if sha256(child(root, f"parents/transactions/{stage}.json")) != digest:
            raise ValueError(f"v3 {stage} transaction bytes changed")
    inputs = receipt.get("input_sha256")
    if not isinstance(inputs, dict):
        raise ValueError("v3 input SHA map missing")
    for role, digest in raw_snapshot["raw_file_sha256"].items():
        if inputs.get(role) != digest:
            raise ValueError(f"v3 raw/weather input {role} changed")
    for role in ("baseline_training_rows", "baseline_training_features",
                 "baseline_ranking_rows", "baseline_ranking_features"):
        actual = sha256(child(root, TIMESTAMP_FILES[role]))
        if inputs.get(role) != actual:
            raise ValueError(f"v3 baseline cache {role} is absent from receipt or changed")
    for role in ("v3_validation_oof", "v3_ranking_predictions", "v3_validation_report"):
        expected = receipt.get("output_sha256", {}).get(role)
        if expected != sha256(child(root, TIMESTAMP_FILES[role])):
            raise ValueError(f"v3 final output {role} changed")
    intermediate = receipt.get("intermediate_files")
    if not isinstance(intermediate, dict) or len(intermediate) < len(V3_STAGES):
        raise ValueError("Full v3 intermediate source/model/OOF byte inventory missing")
    for name, item in intermediate.items():
        if not isinstance(name, str) or not isinstance(item, dict):
            raise ValueError("Malformed v3 intermediate inventory")
        sha_required(item.get("sha256"), f"v3 intermediate {name}")
        if sha256(child(root, item.get("path", ""))) != item["sha256"]:
            raise ValueError(f"v3 intermediate {name} bytes changed")
    models = receipt.get("model_files")
    if not isinstance(models, dict) or not models:
        raise ValueError("v3 actual saved model inventory missing")
    for role, item in models.items():
        if not role.startswith("v3_model_") or not isinstance(item, dict):
            raise ValueError("v3 model role/path invalid")
        sha_required(item.get("sha256"), role)
        if sha256(child(root, item.get("path", ""))) != item["sha256"]:
            raise ValueError(f"v3 saved model {role} bytes changed")
        if receipt.get("output_sha256", {}).get(role) != item["sha256"]:
            raise ValueError(f"v3 model {role} missing from producer output map")
    source_commit = receipt.get("source_commit")
    sha_required(source_commit, "v3 source commit", commit=True)
    for key in ("ordered_validation_ids_sha256", "ordered_ranking_ids_sha256",
                "feature_schema_sha256"):
        sha_required(receipt.get(key), f"v3 {key}")
    return receipt


def prior_snapshot(root: Path, stage: str, raw_snapshot: dict) -> dict:
    verify_external_v3(root, raw_snapshot)
    paths = canonical_prior(root, stage)
    return {"source_and_raw_sha256": raw_snapshot,
            "prior_file_sha256": {role: sha256(path) for role, path in paths.items()},
            "v3_receipt_sha256": sha256(child(root, "parents/v3_producer_receipt.json"))}


def assert_prior(root: Path, stage: str, expected: dict, published_sha: str) -> None:
    current_raw = snapshot(root, published_sha)
    if prior_snapshot(root, stage, current_raw) != expected:
        raise ValueError("A stage input/source/parent changed during model fit")


def category_indices(features: pd.DataFrame) -> list[int]:
    return [features.columns.get_loc(name)
            for name in features.select_dtypes(include="category")]


def verify_catboost(path: Path, features: pd.DataFrame, expected: dict,
                    *, classifier: bool = False) -> Any:
    from catboost import CatBoostClassifier, CatBoostRegressor
    model = CatBoostClassifier() if classifier else CatBoostRegressor()
    model.load_model(str(path))
    if model.feature_names_ != list(features) or model.get_cat_feature_indices() != category_indices(features):
        raise ValueError("Saved CatBoost feature or category order differs")
    actual = model.get_all_params()
    for key, value in expected.items():
        found = actual.get(key)
        if isinstance(value, float):
            if found is None or not np.isclose(float(found), value, rtol=1e-8, atol=1e-8):
                raise ValueError(f"Saved CatBoost {key} differs")
        elif found != value:
            raise ValueError(f"Saved CatBoost {key} differs")
    return model


def fixed_catboost_params(*, classifier: bool, iterations: int) -> dict:
    return {"task_type": "CPU" if classifier else "GPU",
            "loss_function": "Logloss" if classifier else "RMSE",
            "iterations": iterations, "depth": 7 if classifier else 8,
            "random_seed": 2026, "learning_rate": .045 if classifier else .055,
            "l2_leaf_reg": 8 if classifier else 10,
            "border_count": 128, "max_ctr_complexity": 1,
            "one_hot_max_size": 20}


def prepare_stage_inputs(stage: str, root: Path, stage_dir: Path) -> None:
    """Only link newly produced, SHA-checked clean inputs into a fresh stage."""
    if stage == "baseline_predict":
        metadata = read_json(child(root, "artifacts/baseline/model.json"))
        filenames = ["model.json", "direct.txt"]
        if metadata.get("blend_alpha", 0) > 0 and metadata.get("residual_share", 0) > 0:
            filenames.append("residual.txt")
        for filename in filenames:
            original = child(root, f"artifacts/baseline/{filename}")
            destination = (stage_dir / filename).resolve(strict=False)
            if root not in destination.parents or destination.exists():
                raise FileExistsError("Baseline-predict input alias exists or escapes root")
            os.link(original, destination)
        return
    if stage != "source_final":
        return
    for filename in ("validation_combined.json", "seasonal_jan_jul_validation.json",
                     "forward_nov_dec_validation.json", "validation_sequential_gpu.json"):
        original = child(root, f"artifacts/catboost/source/{filename}")
        destination = (stage_dir / filename).resolve(strict=False)
        if root not in destination.parents or destination.exists():
            raise FileExistsError("Source-final staging input alias exists or escapes root")
        os.link(original, destination)


def prove_gpu_fit(root: Path, staged: Path) -> dict:
    """Replay every saved GPU original-fold model/OOF from sealed clean features."""
    import catboost_gpu as gpu
    p = argparse.Namespace(data_dir=root / "data", cache_dir=root / "artifacts/baseline",
                           weather_file=root / "data/external/weather.parquet",
                           v3_dir=root / "artifacts/lobt_ensemble", output_dir=staged,
                           iterations=1500, depth=8, threads=2, seed=2026)
    rows, features, v3, _ = gpu.load_inputs(p)
    if len(rows) != 2085047 or len(v3) != 672428:
        raise ValueError("GPU clean input row universe differs")
    schema = schema_of(features)
    v3_by_id = v3.set_index("MVT_ID_mvt", verify_integrity=True)
    validation = read_json(staged / "validation.json")
    proof = {"feature_schema_sha256": value_sha(schema), "feature_schema": schema,
             "folds": {}, "fixed_weight": .25}
    for fold, months in FOLDS.items():
        train_mask, test_mask = gpu.eligible_masks(rows, months)
        train = np.flatnonzero(train_mask)
        test = np.flatnonzero(test_mask)
        order = np.random.default_rng(2026 + (0 if fold == "seasonal_jan_jul" else 7)).permutation(train)
        early_n = max(20000, int(.06 * len(order)))
        early, fit = order[:early_n], order[early_n:]
        native = read_json(staged / f"{fold}_validation.json")
        saved = pd.read_parquet(staged / f"{fold}_oof.parquet")
        model = verify_catboost(staged / f"{fold}.cbm", features,
                                fixed_catboost_params(classifier=False, iterations=1500))
        if (native.get("fold") != fold or list(native.get("heldout_months", [])) != list(months)
                or int(model.tree_count_) != native.get("tree_count")
                or native.get("training_eligible_rows") != len(train)
                or native.get("fitted_rows") != len(fit)
                or native.get("internal_early_stop_rows") != len(early)
                or native.get("heldout_eligible_rows") != len(test)
                or not np.array_equal(saved.MVT_ID_mvt.to_numpy(), rows.MVT_ID_mvt.iloc[test].to_numpy())
                or not np.array_equal(saved.target.to_numpy(dtype=float), rows.target.iloc[test].to_numpy(dtype=float))
                or not np.array_equal(saved.proxy.to_numpy(dtype=float), rows.proxy.iloc[test].to_numpy(dtype=float))
                or not np.array_equal(saved.month.to_numpy(), rows.month.iloc[test].to_numpy())
                or not np.array_equal(saved.airport.to_numpy(), rows.airport.iloc[test].to_numpy())
                or not saved.fold.eq(fold).all() or not saved.a_valid.all()
                or not np.array_equal(saved.selected.to_numpy(dtype=float),
                                      v3_by_id.loc[saved.MVT_ID_mvt, "selected"].to_numpy(dtype=float))
                or int(model.tree_count_) != native.get("best_iteration", -2) + 1):
            raise ValueError(f"GPU {fold} split/label/OOF metadata differs")
        pred = rows.proxy.to_numpy(dtype=float)[test] + model.predict(features.iloc[test], thread_count=2)
        if not np.allclose(saved.gpu_prediction.to_numpy(dtype=float), pred, rtol=1e-11, atol=1e-8):
            raise ValueError(f"GPU {fold} saved model/OOF prediction replay differs")
        eligible_scores = {str(weight): gpu.rmse(saved.target.to_numpy(dtype=float),
            saved.selected.to_numpy(dtype=float) + weight * (
                saved.gpu_prediction.to_numpy(dtype=float) - saved.selected.to_numpy(dtype=float)))
            for weight in gpu.BLEND_WEIGHTS}
        if any(not np.isclose(native["scores_eligible"][key], score, rtol=1e-11, atol=1e-8)
               for key, score in eligible_scores.items()):
            raise ValueError(f"GPU {fold} eligible scores differ")
        if validation.get(fold) != native:
            raise ValueError(f"GPU {fold} saved report differs from validation selection")
        all_rows = gpu.all_row_score(v3, saved, .25, fold)
        reported = validation["seasonal_all_rows" if fold == "seasonal_jan_jul" else "forward_all_rows"]
        if any(not np.isclose(all_rows[key], reported[key], rtol=1e-11, atol=1e-8)
               for key in all_rows):
            raise ValueError(f"GPU {fold} all-finite score replay differs")
        proof["folds"][fold] = {"fit_ids_sha256": ids_sha(rows.MVT_ID_mvt.iloc[fit]),
                                "early_ids_sha256": ids_sha(rows.MVT_ID_mvt.iloc[early]),
                                "heldout_ids_sha256": ids_sha(rows.MVT_ID_mvt.iloc[test]),
                                "heldout_target_sha256": float_sha(rows.target.iloc[test]),
                                "tree_count": int(model.tree_count_),
                                "model_sha256": sha256(staged / f"{fold}.cbm"),
                                "oof_sha256": sha256(staged / f"{fold}_oof.parquet"),
                                "report_sha256": sha256(staged / f"{fold}_validation.json")}
    selected = min(gpu.BLEND_WEIGHTS,
                   key=lambda w: validation["seasonal_jan_jul"]["scores_eligible"][str(w)])
    if selected != .25 or validation.get("selected_weight") != .25:
        raise ValueError("GPU seasonal selected weight diverged from accepted 0.25")
    if validation["forward_all_rows"]["blend_rmse_sec"] >= validation["forward_all_rows"]["v3_rmse_sec"]:
        raise ValueError("GPU fixed 0.25 did not improve forward all-finite policy")
    ordered_gpu = pd.concat([pd.read_parquet(staged / f"{fold}_oof.parquet",
                                            columns=["MVT_ID_mvt"])["MVT_ID_mvt"]
                             for fold in FOLDS], ignore_index=True)
    proof["ordered_validation_ids_sha256"] = ids_sha(ordered_gpu)
    return proof


def prove_gpu_final(root: Path, staged: Path) -> dict:
    import catboost_gpu as gpu
    from catboost_expert import add_flight_weather, load_cache
    rows, features = load_cache(root / "artifacts/baseline", False)
    features = add_flight_weather(rows, features, root / "data",
                                  root / "data/external/weather.parquet", False)
    target = rows.target.to_numpy(dtype=float)
    proxy = rows.proxy.to_numpy(dtype=float)
    eligible = (np.isfinite(target) & (target >= 0) & (target <= 86400)
                & np.isfinite(proxy) & (proxy >= 0) & (proxy <= 7200))
    model = verify_catboost(staged / "full_2025.cbm", features,
                            fixed_catboost_params(classifier=False, iterations=1500))
    if int(model.tree_count_) != 1500:
        raise ValueError("GPU final saved tree count differs from fixed cap")
    rank_rows, rank_features = load_cache(root / "artifacts/baseline", True)
    rank_features = add_flight_weather(rank_rows, rank_features, root / "data",
                                       root / "data/external/weather.parquet", True)
    if schema_of(rank_features)["columns"] != schema_of(features)["columns"]:
        raise ValueError("GPU ranking feature schema/order differs from training")
    rank_proxy = rank_rows.proxy.to_numpy(dtype=float)
    valid = np.isfinite(rank_proxy) & (rank_proxy >= 0) & (rank_proxy <= 7200)
    expert = pd.read_parquet(staged / "ranking_expert.parquet")
    template = pd.read_parquet(root / "data/submitting.parquet", columns=["MVT_ID_mvt"])
    if (len(template) != 344841 or not rank_rows.MVT_ID_mvt.equals(template.MVT_ID_mvt)
            or not expert.MVT_ID_mvt.equals(template.MVT_ID_mvt)):
        raise ValueError("GPU ranking IDs/order differ from template")
    expected = np.full(len(rank_rows), np.nan, dtype=float)
    expected[valid] = rank_proxy[valid] + model.predict(rank_features.iloc[np.flatnonzero(valid)],
                                                        thread_count=2)
    if not np.array_equal(expert.gpu_prediction.to_numpy(dtype=float), expected, equal_nan=True):
        raise ValueError("GPU ranking expert does not replay from saved full model")
    v3 = pd.read_parquet(root / "artifacts/lobt_ensemble/predictions.parquet",
                         columns=["MVT_ID_mvt", "TAXITIME_SEC_mvt"])
    if not v3.MVT_ID_mvt.equals(template.MVT_ID_mvt):
        raise ValueError("v3 ranking parent order differs")
    base = v3.TAXITIME_SEC_mvt.to_numpy(dtype=float)
    prediction = base.copy()
    prediction[valid] += .25 * (expected[valid] - prediction[valid])
    prediction = np.maximum(prediction, 0)
    saved = pd.read_parquet(staged / "predictions.parquet")
    if (not saved.MVT_ID_mvt.equals(template.MVT_ID_mvt)
            or not np.array_equal(saved.TAXITIME_SEC_mvt.to_numpy(dtype=float), prediction)
            or not np.array_equal(prediction[~valid], base[~valid])
            or not np.isfinite(prediction).all()):
        raise ValueError("GPU fixed 0.25 ranking policy/readback differs")
    native = read_json(staged / "final_report.json")
    if (native.get("training_eligible_rows") != int(eligible.sum())
            or native.get("ranking_total_departures") != len(rank_rows)
            or native.get("ranking_gpu_eligible_rows") != int(valid.sum())
            or native.get("blend_weight_frozen_from_seasonal") != .25
            or native.get("depth") != 8 or native.get("iterations") != 1500):
        raise ValueError("GPU final report differs from saved model/ranking formula")
    return {"feature_schema_sha256": value_sha(schema_of(features)),
            "ranking_schema_sha256": value_sha(schema_of(rank_features)),
            "training_ids_sha256": ids_sha(rows.MVT_ID_mvt.iloc[np.flatnonzero(eligible)]),
            "ranking_ids_sha256": ids_sha(template.MVT_ID_mvt),
            "ranking_valid_rows": int(valid.sum()),
            "ranking_invalid_unchanged": int((~valid).sum()),
            "model_sha256": sha256(staged / "full_2025.cbm"),
            "ranking_expert_sha256": sha256(staged / "ranking_expert.parquet"),
            "predictions_sha256": sha256(staged / "predictions.parquet")}


def source_args(root: Path, staged: Path) -> argparse.Namespace:
    return argparse.Namespace(data_dir=root / "data", cache_dir=root / "artifacts/baseline",
                              v3_dir=root / "artifacts/lobt_ensemble",
                              gpu_dir=root / "artifacts/catboost/gpu",
                              weather_file=root / "data/external/weather.parquet",
                              output_dir=staged, iterations=600, depth=7,
                              threads=4, seed=2026)


def prove_source_fit(root: Path, staged: Path) -> dict:
    import catboost_source as source
    from catboost_expert import add_flight_weather, load_cache
    from sklearn.metrics import brier_score_loss, roc_auc_score
    args = source_args(root, staged)
    rows, features = load_cache(args.cache_dir, False)
    features = add_flight_weather(rows, features, args.data_dir, args.weather_file, False)
    if len(rows) != 2085047:
        raise ValueError("Source clean baseline row universe differs")
    schema = schema_of(features)
    candidate, schedule = source.candidate_mask(rows, features)
    y = rows.target.to_numpy(dtype=float)
    exact = np.abs(y - schedule) <= 60
    v3 = pd.read_parquet(args.v3_dir / "validation_predictions.parquet",
                         columns=["MVT_ID_mvt", "selected"])
    v3_indexed = v3.set_index("MVT_ID_mvt", verify_integrity=True)
    proof = {"feature_schema_sha256": value_sha(schema), "feature_schema": schema,
             "folds": {}, "fixed_scale": .5}
    for fold, months in FOLDS.items():
        held = rows.month.isin(months).to_numpy()
        train = np.flatnonzero(candidate & ~held & np.isfinite(y) & (y >= 0) & (y <= 86400))
        test = np.flatnonzero(candidate & held & np.isfinite(y))
        order = np.random.default_rng(2026).permutation(train)
        early_n = max(3000, int(.08 * len(order)))
        early, fit = order[:early_n], order[early_n:]
        native = read_json(staged / f"{fold}_validation.json")
        saved = pd.read_parquet(staged / f"{fold}_oof.parquet")
        model = verify_catboost(staged / f"{fold}.cbm", features,
                                fixed_catboost_params(classifier=True, iterations=600),
                                classifier=True)
        if (native.get("fold") != fold or list(native.get("months", [])) != list(months)
                or native.get("training_n") != len(train)
                or native.get("test_n") != len(test)
                or int(model.tree_count_) != native.get("best_iteration", -2) + 1
                or not np.array_equal(saved.MVT_ID_mvt.to_numpy(), rows.MVT_ID_mvt.iloc[test].to_numpy())
                or not np.array_equal(saved.target.to_numpy(dtype=float), y[test])
                or not np.array_equal(saved.proxy.to_numpy(dtype=float), rows.proxy.iloc[test].to_numpy(dtype=float))
                or not np.array_equal(saved.month.to_numpy(), rows.month.iloc[test].to_numpy())
                or not np.array_equal(saved.airport.to_numpy(), rows.airport.iloc[test].to_numpy())
                or not np.array_equal(saved.schedule_exact.to_numpy(dtype=bool), exact[test])
                or not np.allclose(saved.schedule_proxy_sec.to_numpy(dtype=float),
                                   schedule[test], rtol=0, atol=0)):
            raise ValueError(f"Source {fold} split/labels/schedule metadata differ")
        prob = model.predict_proba(features.iloc[test], thread_count=4)[:, 1]
        if not np.allclose(saved.p_schedule_exact.to_numpy(dtype=float), prob,
                           rtol=1e-11, atol=1e-9):
            raise ValueError(f"Source {fold} saved classifier probability OOF differs")
        base = v3_indexed.loc[saved.MVT_ID_mvt, "selected"].to_numpy(dtype=float)
        if not np.array_equal(saved.v3_prediction.to_numpy(dtype=float), base):
            raise ValueError(f"Source {fold} saved v3 OOF parent differs")
        checks = {"training_exact_rate": float(exact[train].mean()),
                  "test_exact_rate": float(exact[test].mean()),
                  "roc_auc": float(roc_auc_score(exact[test], prob)),
                  "brier": float(brier_score_loss(exact[test], prob))}
        if any(not np.isclose(native[key], value, rtol=1e-11, atol=1e-9)
               for key, value in checks.items()):
            raise ValueError(f"Source {fold} saved classifier metrics differ")
        scores = {str(scale): source.rmse(y[test], base + scale * prob * (schedule[test] - base))
                  for scale in source.SCALES}
        if any(not np.isclose(native["scores"][key], val, rtol=1e-11, atol=1e-8)
               for key, val in scores.items()):
            raise ValueError(f"Source {fold} candidate scores differ")
        proof["folds"][fold] = {"fit_ids_sha256": ids_sha(rows.MVT_ID_mvt.iloc[fit]),
                                "early_ids_sha256": ids_sha(rows.MVT_ID_mvt.iloc[early]),
                                "heldout_ids_sha256": ids_sha(rows.MVT_ID_mvt.iloc[test]),
                                "heldout_target_sha256": float_sha(y[test]),
                                "tree_count": int(model.tree_count_),
                                "model_sha256": sha256(staged / f"{fold}.cbm"),
                                "oof_sha256": sha256(staged / f"{fold}_oof.parquet"),
                                "report_sha256": sha256(staged / f"{fold}_validation.json")}
    # Recompute both published evaluation functions into a distinct, isolated
    # scratch directory. They can write there, and we compare every output to
    # the child's saved bytes before accepting the stage.
    scratch = staged / "_readback"
    if scratch.exists():
        raise FileExistsError("Source proof scratch already exists")
    scratch.mkdir()
    for fold in FOLDS:
        os.link(staged / f"{fold}_oof.parquet", scratch / f"{fold}_oof.parquet")
    replay_args = source_args(root, scratch)
    recomputed = source.evaluate_existing(replay_args)
    sequential = source.evaluate_sequential_gpu(replay_args)
    for name in ("validation_combined.json", "validation_predictions.parquet",
                 "validation_sequential_gpu.json"):
        if sha256(staged / name) != sha256(scratch / name):
            raise ValueError(f"Source original evaluator replay changed {name}")
    if recomputed.get("selected_scale") != .5:
        raise ValueError("Source seasonal scale differs from accepted 0.5")
    for key in ("seasonal_nested", "forward_transfer"):
        scores = recomputed[key]
        if scores["corrected"]["overall_rmse_sec"] >= scores["base"]["overall_rmse_sec"]:
            raise ValueError("Source fixed half-probability transfer gate failed")
    if sequential.get("rule") != "Fixed GPU .25 followed by source .5; no tuning on these combined OOF rows":
        raise ValueError("Source sequential GPU policy changed")
    for fold in FOLDS:
        (scratch / f"{fold}_oof.parquet").unlink()
    for name in ("validation_combined.json", "validation_predictions.parquet",
                 "validation_sequential_gpu.json"):
        (scratch / name).unlink()
    scratch.rmdir()
    ids = pd.concat([pd.read_parquet(staged / f"{fold}_oof.parquet",
                                     columns=["MVT_ID_mvt"])["MVT_ID_mvt"]
                     for fold in FOLDS], ignore_index=True)
    proof["ordered_validation_ids_sha256"] = ids_sha(ids)
    proof["selected_scale"] = recomputed["selected_scale"]
    proof["combined_report_sha256"] = sha256(staged / "validation_combined.json")
    proof["sequential_report_sha256"] = sha256(staged / "validation_sequential_gpu.json")
    return proof


def prove_source_final(root: Path, staged: Path) -> dict:
    import catboost_source as source
    from catboost_expert import add_flight_weather, load_cache
    args = source_args(root, staged)
    reports = [read_json(args.output_dir / f"{fold}_validation.json") for fold in FOLDS]
    iterations = int(np.median([part["best_iteration"] + 1 for part in reports]))
    rows, features = load_cache(args.cache_dir, False)
    features = add_flight_weather(rows, features, args.data_dir, args.weather_file, False)
    candidate, schedule = source.candidate_mask(rows, features)
    y = rows.target.to_numpy(dtype=float)
    train = candidate & np.isfinite(y) & (y >= 0) & (y <= 86400)
    model = verify_catboost(staged / "final.cbm", features,
                            fixed_catboost_params(classifier=True, iterations=iterations),
                            classifier=True)
    if int(model.tree_count_) != iterations:
        raise ValueError("Source full classifier saved tree count differs from original-fold median")
    rank_rows, rank_features = load_cache(args.cache_dir, True)
    rank_features = add_flight_weather(rank_rows, rank_features, args.data_dir,
                                       args.weather_file, True)
    if schema_of(rank_features)["columns"] != schema_of(features)["columns"]:
        raise ValueError("Source ranking feature names/dtypes differ")
    rank_candidate, rank_schedule = source.candidate_mask(rank_rows, rank_features)
    prob = pd.read_parquet(staged / "ranking_source_probabilities.parquet")
    template = pd.read_parquet(args.data_dir / "submitting.parquet", columns=["MVT_ID_mvt"])
    if (len(template) != 344841 or not rank_rows.MVT_ID_mvt.equals(template.MVT_ID_mvt)
            or not prob.MVT_ID_mvt.equals(template.MVT_ID_mvt)
            or not np.array_equal(prob.source_candidate.to_numpy(dtype=bool), rank_candidate)):
        raise ValueError("Source ranking candidate IDs/gate differ")
    expected_prob = np.full(len(rank_rows), np.nan, dtype=float)
    expected_prob[rank_candidate] = model.predict_proba(
        rank_features.iloc[np.flatnonzero(rank_candidate)], thread_count=4)[:, 1]
    if (not np.array_equal(prob.p_schedule_exact.to_numpy(dtype=float), expected_prob, equal_nan=True)
            or not np.allclose(prob.schedule_proxy_sec.to_numpy(dtype=float)[rank_candidate],
                               rank_schedule[rank_candidate], rtol=0, atol=0)):
        raise ValueError("Source final saved classifier ranking probabilities differ")
    v3 = pd.read_parquet(args.v3_dir / "predictions.parquet",
                         columns=["MVT_ID_mvt", "TAXITIME_SEC_mvt"])
    gpu = pd.read_parquet(args.gpu_dir / "predictions.parquet",
                          columns=["MVT_ID_mvt", "TAXITIME_SEC_mvt"])
    if not (v3.MVT_ID_mvt.equals(template.MVT_ID_mvt)
            and gpu.MVT_ID_mvt.equals(template.MVT_ID_mvt)):
        raise ValueError("Source ranking parent ID/template order differs")
    base = v3.TAXITIME_SEC_mvt.to_numpy(dtype=float)
    gpu_base = gpu.TAXITIME_SEC_mvt.to_numpy(dtype=float)
    expected_source = base.copy()
    expected_source[rank_candidate] += .5 * expected_prob[rank_candidate] * (
        rank_schedule[rank_candidate] - expected_source[rank_candidate])
    expected_source = np.maximum(expected_source, 0)
    expected_sequential = gpu_base.copy()
    expected_sequential[rank_candidate] += .5 * expected_prob[rank_candidate] * (
        rank_schedule[rank_candidate] - expected_sequential[rank_candidate])
    expected_sequential = np.maximum(expected_sequential, 0)
    for filename, expected in (("predictions.parquet", expected_source),
                               ("sequential_predictions.parquet", expected_sequential)):
        saved = pd.read_parquet(staged / filename)
        if (not saved.MVT_ID_mvt.equals(template.MVT_ID_mvt)
                or not np.array_equal(saved.TAXITIME_SEC_mvt.to_numpy(dtype=float), expected)
                or not np.array_equal(expected[~rank_candidate],
                                      (base if filename == "predictions.parquet" else gpu_base)[~rank_candidate])
                or not np.isfinite(expected).all()):
            raise ValueError(f"Source {filename} fixed policy/readback differs")
    native = read_json(staged / "manifest.json")
    if (native.get("iterations") != iterations or native.get("scale") != .5
            or native.get("training_rows") != int(train.sum())
            or native.get("ranking_rows") != len(template)
            or native.get("ranking_source_candidates") != int(rank_candidate.sum())):
        raise ValueError("Source final model report settings/coverage differ")
    return {"feature_schema_sha256": value_sha(schema_of(features)),
            "ranking_schema_sha256": value_sha(schema_of(rank_features)),
            "training_ids_sha256": ids_sha(rows.MVT_ID_mvt.iloc[np.flatnonzero(train)]),
            "ranking_ids_sha256": ids_sha(template.MVT_ID_mvt),
            "ranking_candidates": int(rank_candidate.sum()),
            "iterations": iterations,
            "model_sha256": sha256(staged / "final.cbm"),
            "probabilities_sha256": sha256(staged / "ranking_source_probabilities.parquet"),
            "sequential_sha256": sha256(staged / "sequential_predictions.parquet")}


def published_git_commit(snapshot_value: dict) -> str:
    result = subprocess.run(["git", "-C", str(SOURCE_ROOT), "rev-parse", "HEAD"],
                            check=True, capture_output=True, text=True)
    commit = result.stdout.strip()
    sha_required(commit, "published source commit", commit=True)
    for name, digest in snapshot_value["source_sha256"].items():
        if name == "reports/clean_parent_producers_spec.json":
            pass
        try:
            content = subprocess.run(["git", "-C", str(SOURCE_ROOT), "show",
                                      f"HEAD:{name}"], check=True, capture_output=True).stdout
        except subprocess.CalledProcessError as exc:
            raise ValueError(f"Source closure {name} is absent from published commit") from exc
        if hashlib.sha256(content).hexdigest() != digest:
            raise ValueError(f"Source closure {name} differs from published Git commit")
    return commit


def prove_stage(stage: str, root: Path, staged: Path) -> dict:
    if stage == "gpu_fit":
        return prove_gpu_fit(root, staged)
    if stage == "source_fit":
        return prove_source_fit(root, staged)
    if stage == "gpu_final":
        return prove_gpu_final(root, staged)
    if stage == "source_final":
        return prove_source_final(root, staged)
    raise RuntimeError(f"Missing reviewed replay adapter for v3 stage {stage}: {PENDING_V3_PROOF[stage]}")


def run_stage(args: argparse.Namespace) -> dict:
    if args.stage not in ORDER:
        raise ValueError("Unknown clean stage")
    if args.stage in PENDING_V3_PROOF:
        raise RuntimeError(f"V3 stage {args.stage} refuses before child execution: "
                           f"missing {PENDING_V3_PROOF[args.stage]}")
    root = isolated_root(args.run_root)
    free_memory()
    frozen = snapshot(root, args.published_source_sha256)
    commit = published_git_commit(frozen)
    prior = prior_snapshot(root, args.stage, frozen)
    target_dir = (root / OUTPUT_DIR[args.stage]).resolve(strict=False)
    if root not in target_dir.parents:
        raise ValueError("Stage canonical target escapes isolated root")
    staged = (root / "artifacts/_clean_parent_stage" / args.stage).resolve(strict=False)
    if root not in staged.parents or staged.exists():
        raise FileExistsError("Stage directory already exists or escapes root; no resume")
    transaction_path = (root / "parents/transactions" / f"{args.stage}.json").resolve(strict=False)
    if transaction_path.exists():
        raise FileExistsError("Stage transaction already exists")
    outputs = {filename: (target_dir / filename).resolve(strict=False)
               for filename in REQUIRED_OUTPUTS[args.stage]}
    optional = {filename: (target_dir / filename).resolve(strict=False)
                for filename in OPTIONAL_OUTPUTS.get(args.stage, ())}
    if any(root not in path.parents or path.exists()
           for path in (*outputs.values(), *optional.values())):
        raise FileExistsError("Canonical output exists or escapes root; no overwrite")
    staged.mkdir(parents=True, exist_ok=False)
    prepare_stage_inputs(args.stage, root, staged)
    assert_prior(root, args.stage, prior, args.published_source_sha256)
    commands = stage_commands(args.stage, staged)
    log_path = staged / "child.log"
    with log_path.open("xb") as log:
        for command in commands:
            free_memory()
            assert_prior(root, args.stage, prior, args.published_source_sha256)
            result = subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT,
                                    check=False, shell=False)
            log.flush()
            if result.returncode != 0:
                raise RuntimeError(f"Original {args.stage} CLI exited {result.returncode}; "
                                   "stage is terminal and cannot be resumed")
            assert_prior(root, args.stage, prior, args.published_source_sha256)
    for filename in outputs:
        candidate = (staged / filename).resolve(strict=True)
        if staged not in candidate.parents or not candidate.is_file():
            raise FileNotFoundError(f"Original {args.stage} did not write required {filename}")
    if args.stage == "baseline_train":
        metadata = read_json(staged / "model.json")
        needs_residual = (metadata.get("blend_alpha", 0) > 0
                          and metadata.get("residual_share", 0) > 0)
        if needs_residual and not (staged / "residual.txt").is_file():
            raise FileNotFoundError("Baseline model requires its saved residual.txt")
    for filename, target in optional.items():
        if (staged / filename).is_file():
            outputs[filename] = target
    proof = prove_stage(args.stage, root, staged)
    assert_prior(root, args.stage, prior, args.published_source_sha256)
    source_hash = sha256(SOURCE_ROOT / "clean_parent_producers.py")
    staged_hashes = {filename: sha256(staged / filename) for filename in outputs}
    for filename, target in outputs.items():
        target.parent.mkdir(parents=True, exist_ok=True)
        os.link(staged / filename, target)
    for filename in outputs:
        if sha256(outputs[filename]) != staged_hashes[filename]:
            raise ValueError("Promoted clean output bytes differ from verified stage")
        (staged / filename).unlink()
    transaction = {"schema_version": 1, "stage": args.stage, "status": "complete",
                   "scientific_source_commit": commit,
                   "producer_source_sha256": frozen["source_sha256"],
                   "producer_adapter_sha256": source_hash,
                   "input_snapshot": prior, "commands": commands,
                   "child_log_sha256": sha256(log_path),
                   "proof": proof, "output_sha256": staged_hashes,
                   "exclusive_promotion_verified": True}
    assert_prior(root, args.stage, prior, args.published_source_sha256)
    exclusive_json(transaction_path, transaction, root)
    return {"stage": args.stage, "status": "complete",
            "transaction_sha256": sha256(transaction_path),
            "output_files": len(outputs), "proof": proof}


def verified_transaction(root: Path, stage: str, published_sha: str,
                         frozen: dict) -> dict:
    path = child(root, f"parents/transactions/{stage}.json")
    transaction = read_json(path)
    if (transaction.get("schema_version") != 1 or transaction.get("stage") != stage
            or transaction.get("status") != "complete"
            or transaction.get("producer_source_sha256") != frozen["source_sha256"]
            or transaction.get("producer_adapter_sha256")
            != frozen["source_sha256"]["clean_parent_producers.py"]
            or transaction.get("commands") != stage_commands(
                stage, root / "artifacts/_clean_parent_stage" / stage)
            or transaction.get("exclusive_promotion_verified") is not True):
        raise ValueError(f"Clean {stage} actual-transaction source/command seal differs")
    sha_required(transaction.get("scientific_source_commit"), f"{stage} source commit", commit=True)
    if transaction["scientific_source_commit"] != published_git_commit(frozen):
        raise ValueError(f"{stage} scientific source commit changed")
    assert_prior(root, stage, transaction["input_snapshot"], published_sha)
    actual_outputs = set(transaction.get("output_sha256", {}))
    required_outputs = set(REQUIRED_OUTPUTS[stage])
    allowed_outputs = required_outputs | set(OPTIONAL_OUTPUTS.get(stage, ()))
    if not required_outputs.issubset(actual_outputs) or not actual_outputs.issubset(allowed_outputs):
        raise ValueError(f"Clean {stage} actual outputs are incomplete")
    for name, digest in transaction["output_sha256"].items():
        if sha256(child(root, f"{OUTPUT_DIR[stage]}/{name}")) != digest:
            raise ValueError(f"Clean {stage} output {name} changed")
    if not isinstance(transaction.get("proof"), dict):
        raise ValueError(f"Clean {stage} model/OOF proof missing")
    return transaction


def make_producer_receipt(name: str, root: Path, files: dict,
                          transactions: dict[str, dict], frozen: dict) -> dict:
    import replica_v4_timestamp as timestamp
    source = {path: digest for path, digest in frozen["source_sha256"].items()
              if path.endswith(".py")}
    data_roles = set(timestamp.DATA_ROLES)
    required_inputs = data_roles | {
        "v3_validation_oof", "v3_ranking_predictions", "v3_validation_report",
        "v3_producer_receipt"}
    if name == "source":
        required_inputs |= set(timestamp.PRODUCERS["gpu"]["outputs"]) | {"gpu_producer_receipt"}
    output_roles = set(timestamp.PRODUCERS[name]["outputs"])
    model_roles = {role for role in files if role.startswith(f"{name}_model_")}
    output_roles |= model_roles
    if not model_roles:
        raise ValueError(f"No actual saved {name} models in clean output inventory")
    if not required_inputs.issubset(files) or not output_roles.issubset(files):
        raise ValueError(f"{name} parent receipt cannot bind complete input/output roles")
    first, last = ("gpu_fit", "gpu_final") if name == "gpu" else ("source_fit", "source_final")
    a = transactions[first]["proof"]
    b = transactions[last]["proof"]
    if a.get("feature_schema_sha256") != b.get("feature_schema_sha256"):
        raise ValueError(f"{name} original fold and final model feature schema differ")
    if name == "gpu":
        if (a.get("fixed_weight") != .25 or b.get("ranking_invalid_unchanged", -1) < 1
                or b.get("ranking_ids_sha256") != ids_sha(pd.read_parquet(
                    child(root, "data/submitting.parquet"), columns=["MVT_ID_mvt"]).MVT_ID_mvt)):
            raise ValueError("GPU actual fixed science/ranking coverage differs")
        choices = {"gpu_weight": .25, "iterations": 1500, "depth": 8,
                   "seed": 2026, "final_iterations": 1500}
    else:
        if a.get("selected_scale") != .5 or b.get("iterations", 0) < 1:
            raise ValueError("Source actual seasonal scale/final model differs")
        choices = {"gpu_weight": .25, "source_scale": .5,
                   "iterations_cap": 600, "depth": 7, "seed": 2026,
                   "final_iterations_rule": "integer_median_original_best_iteration_plus_one"}
    if transactions[first]["scientific_source_commit"] != transactions[last]["scientific_source_commit"]:
        raise ValueError(f"{name} fold/final source commits differ")
    return {"name": name, "status": "complete",
            "source_commit": transactions[first]["scientific_source_commit"],
            "source_sha256": source,
            "input_sha256": {role: files[role]["sha256"] for role in sorted(required_inputs)},
            "output_sha256": {role: files[role]["sha256"] for role in sorted(output_roles)},
            "heldout_folds": {fold: list(months) for fold, months in FOLDS.items()},
             "fit_excludes_heldout": True,
             "early_stop_uses_heldout": False,
             "fit_and_early_exclude_heldout": True,
             "legacy_validation_erratum_sha256": LEGACY_ERRATUM_SHA256,
             "component_only_scope": read_json(SOURCE_ROOT / LEGACY_ERRATUM_REL)[
                 "new_component_receipt_policy"]["component_only_scope"],
             "legacy_parent_limit": read_json(SOURCE_ROOT / LEGACY_ERRATUM_REL)[
                 "new_component_receipt_policy"]["legacy_parent_limit"],
            "published_choices": choices,
            "ordered_validation_ids_sha256": a["ordered_validation_ids_sha256"],
            "ordered_ranking_ids_sha256": b["ranking_ids_sha256"],
            "feature_schema_sha256": a["feature_schema_sha256"],
            "independent_model_and_policy_replay_passed": True,
            "actual_stage_transaction_sha256": {
                first: sha256(child(root, f"parents/transactions/{first}.json")),
                last: sha256(child(root, f"parents/transactions/{last}.json"))}}


def emit_parent_manifest(args: argparse.Namespace) -> dict:
    """Emit exact timestamp-manifest producer receipts from verified transactions."""
    import replica_v4_timestamp as timestamp
    root = isolated_root(args.run_root)
    free_memory()
    frozen = snapshot(root, args.published_source_sha256)
    published_git_commit(frozen)
    v3 = verify_external_v3(root, frozen)
    transactions = {stage: verified_transaction(root, stage, args.published_source_sha256,
                                                 frozen)
                    for stage in ("gpu_fit", "source_fit", "gpu_final", "source_final")}
    target = root / "parents/v4_timestamp_parents.json"
    for file in (target, root / "parents/gpu_producer_receipt.json",
                 root / "parents/source_producer_receipt.json"):
        if file.exists():
            raise FileExistsError("Parent manifest/producer receipt exists; no overwrite")
    files = {}
    for role, relative in timestamp.CANONICAL.items():
        if role in ("gpu_producer_receipt", "source_producer_receipt"):
            continue
        path = child(root, relative)
        files[role] = {"path": relative, "sha256": sha256(path)}
    for role, item in v3["model_files"].items():
        path = child(root, item["path"])
        files[role] = {"path": item["path"], "sha256": sha256(path)}
    model_files = {
        "gpu_model_seasonal": "artifacts/catboost/gpu/seasonal_jan_jul.cbm",
        "gpu_model_forward": "artifacts/catboost/gpu/forward_nov_dec.cbm",
        "gpu_model_full": "artifacts/catboost/gpu/full_2025.cbm",
        "source_model_seasonal": "artifacts/catboost/source/seasonal_jan_jul.cbm",
        "source_model_forward": "artifacts/catboost/source/forward_nov_dec.cbm",
        "source_model_full": "artifacts/catboost/source/final.cbm",
    }
    for role, relative in model_files.items():
        files[role] = {"path": relative, "sha256": sha256(child(root, relative))}
    v3_declared = {**v3, "receipt_sha256": files["v3_producer_receipt"]["sha256"]}
    gpu = make_producer_receipt("gpu", root, files, transactions, frozen)
    files["gpu_producer_receipt"] = {
        "path": timestamp.CANONICAL["gpu_producer_receipt"],
        "sha256": json_sha(gpu)}
    source = make_producer_receipt("source", root, files, transactions, frozen)
    files["source_producer_receipt"] = {
        "path": timestamp.CANONICAL["source_producer_receipt"],
        "sha256": json_sha(source)}
    manifest = {"schema_version": 1, "status": "complete", "files": files,
                "legacy_validation_erratum_sha256": LEGACY_ERRATUM_SHA256,
                "heldout_folds": {name: list(months) for name, months in FOLDS.items()},
                "producers": {
                    "v3": v3_declared,
                    "gpu": {**gpu, "receipt_sha256": files["gpu_producer_receipt"]["sha256"]},
                    "source": {**source, "receipt_sha256": files["source_producer_receipt"]["sha256"]}}}
    timestamp.validate_manifest_metadata(manifest)
    for name in ("gpu", "source"):
        producer = manifest["producers"][name]
        receipt = gpu if name == "gpu" else source
        if (producer["receipt_sha256"] != json_sha(receipt)
                or receipt != {key: value for key, value in producer.items()
                               if key != "receipt_sha256"}):
            raise ValueError("Prospective producer receipt and manifest differ")
    if snapshot(root, args.published_source_sha256) != frozen:
        raise ValueError("Source/raw/weather changed while assembling parent manifest")
    exclusive_json(root / "parents/gpu_producer_receipt.json", gpu, root)
    exclusive_json(root / "parents/source_producer_receipt.json", source, root)
    for name in ("gpu", "source"):
        if sha256(child(root, timestamp.CANONICAL[timestamp.PRODUCERS[name]["receipt"]])) != files[
                f"{name}_producer_receipt"]["sha256"]:
            raise ValueError("Published producer receipt bytes differ from prevalidated bytes")
    exclusive_json(target, manifest, root)
    _, _, proof = timestamp.parent_snapshot(root, frozen["source_sha256"])
    if proof["manifest_sha256"] != sha256(target):
        raise ValueError("Timestamp adapter rejected clean producer manifest")
    return {"status": "complete", "manifest_sha256": sha256(target),
            "v3_receipt_sha256": files["v3_producer_receipt"]["sha256"],
            "gpu_receipt_sha256": files["gpu_producer_receipt"]["sha256"],
            "source_receipt_sha256": files["source_producer_receipt"]["sha256"]}


def synthetic_self_test() -> dict:
    """Only source/spec bytes and in-memory metadata; no private values/models."""
    source_snapshot()
    spec = read_json(SPEC)
    legacy_policy = read_json(SOURCE_ROOT / LEGACY_ERRATUM_REL)["legacy_v3_receipt_policy"]
    truthful_legacy = {**legacy_policy,
                       "legacy_validation_erratum_sha256": LEGACY_ERRATUM_SHA256}
    check_legacy_v3_claim(truthful_legacy)
    fabricated = {**truthful_legacy, "fit_and_early_exclude_heldout": True}
    try:
        check_legacy_v3_claim(fabricated)
    except ValueError:
        pass
    else:
        raise AssertionError("Fabricated original-v3 early-stop exclusion was accepted")
    if (spec.get("reviewed_v3_proof_adapter_sha256") != REVIEWED_V3_PROOF_SHA256
            or sha256(SOURCE_ROOT / REVIEWED_V3_PROOF_REL) != REVIEWED_V3_PROOF_SHA256):
        raise AssertionError("Reviewed v3 source or published pin differs")
    if len(ORDER) != 15 or set(ORDER) != set(OUTPUT_DIR) or set(ORDER) != set(REQUIRED_OUTPUTS):
        raise AssertionError("Stage closure registry differs")
    for stage in ORDER:
        commands = stage_commands(stage, Path("synthetic-stage"))
        if not commands or any(command[0] != sys.executable or
                               Path(command[1]).resolve().parent != SOURCE_ROOT or
                               "--output-dir" not in command for command in commands):
            raise AssertionError(f"{stage} original CLI is not explicit and source-bound")
        if stage in PENDING_V3_PROOF:
            try:
                prove_stage(stage, Path("synthetic-root"), Path("synthetic-stage"))
            except RuntimeError as exc:
                if "Missing reviewed replay adapter" not in str(exc):
                    raise
            else:
                raise AssertionError(f"Unproved v3 {stage} stage was allowed")
    try:
        isolated_root(SOURCE_ROOT)
    except ValueError:
        pass
    else:
        raise AssertionError("Source checkout was accepted as private run root")
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory).resolve()
        try:
            child(root, "../outside", existing=False)
        except ValueError:
            pass
        else:
            raise AssertionError("Parent path traversal was allowed")
        baseline = root / "artifacts/baseline"
        baseline.mkdir(parents=True)
        (baseline / "model.json").write_text(
            json.dumps({"blend_alpha": 1.0, "residual_share": 1.0}), encoding="utf-8")
        (baseline / "direct.txt").write_text("synthetic-direct", encoding="utf-8")
        (baseline / "residual.txt").write_text("synthetic-residual", encoding="utf-8")
        staged = root / "artifacts/staged-baseline-predict"
        staged.mkdir()
        prepare_stage_inputs("baseline_predict", root, staged)
        if any(sha256(staged / name) != sha256(baseline / name)
               for name in ("model.json", "direct.txt", "residual.txt")):
            raise AssertionError("Baseline predict did not link its conditional saved models")
        (baseline / "residual.txt").unlink()
        second = root / "artifacts/staged-missing-residual"
        second.mkdir()
        try:
            prepare_stage_inputs("baseline_predict", root, second)
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("Baseline predict accepted a missing required residual model")
        if json_sha({"proof": "synthetic"}) != hashlib.sha256(
                json.dumps({"proof": "synthetic"}, indent=2,
                           allow_nan=False).encode("utf-8")).hexdigest():
            raise AssertionError("Prospective receipt SHA differs from published bytes")
    synthetic = pd.DataFrame({"a": pd.Series(["A", "B"], dtype="category"),
                              "b": pd.Series([1.0, 2.0], dtype="float32")})
    schema = schema_of(synthetic)
    if schema["categorical_columns"] != ["a"] or [x["name"] for x in schema["columns"]] != ["a", "b"]:
        raise AssertionError("Feature/category schema proof order differs")
    if ids_sha([1, 2]) == ids_sha([2, 1]):
        raise AssertionError("Ordered ID hash lost order")
    for stage in ("gpu_fit", "source_fit", "gpu_final", "source_final"):
        if stage in PENDING_V3_PROOF:
            raise AssertionError("GPU/source proof stage unexpectedly missing")
    return {"source_and_spec_pin": "passed", "legacy_v3_false_exclusion": "refused",
            "all_stage_commands_explicit": "passed",
            "v3_unproved_stages": "refused_before_child",
            "source_root_and_path_escape": "refused",
            "schema_and_ordered_ID": "passed", "baseline_stage_alias_and_receipt_hash": "passed",
            "real_values_or_models_read": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("plan", "self-test", "verify-inputs",
                                           "run-stage", "emit-parents"), default="plan")
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--stage", choices=ORDER)
    parser.add_argument("--published-source-sha256")
    args = parser.parse_args()
    if args.mode == "plan":
        result = plan()
    elif args.mode == "self-test":
        result = synthetic_self_test()
    else:
        if args.run_root is None or args.published_source_sha256 is None:
            parser.error("Real modes require --run-root and --published-source-sha256")
        if args.mode == "verify-inputs":
            root = isolated_root(args.run_root)
            proof = snapshot(root, args.published_source_sha256)
            result = {"status": "input_bytes_verified_only",
                      "raw_manifest_sha256": proof["raw_manifest_sha256"],
                      "raw_weather_files": len(proof["raw_file_sha256"]),
                      "source_commit": published_git_commit(proof),
                      "v3_proof_adapter_pinned": read_json(SPEC).get(
                          "reviewed_v3_proof_adapter_sha256") is not None}
        elif args.mode == "run-stage":
            if args.stage is None:
                parser.error("run-stage requires --stage")
            result = run_stage(args)
        else:
            result = emit_parent_manifest(args)
    print(json.dumps(result, indent=2, allow_nan=False, default=str))


if __name__ == "__main__":
    main()
