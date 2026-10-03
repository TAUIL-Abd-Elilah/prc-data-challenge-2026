"""Prospective v14 event-sequence direct expert for the sealed v8 no-NM gate.

This source deliberately has no ranking or final-fit mode. The published
reports/event_sequence_missing_spec_v14.json is the scientific contract.
Real prepare/fit/score commands require immutable input receipts, and no
command silently changes an existing artifact.
"""

from __future__ import annotations

import argparse
from contextlib import closing
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch import nn
from torch.nn import functional as F

import movement_only_expert as movement
import compose_current_candidate as current


ROOT = Path(__file__).resolve().parent
SPEC = ROOT / "reports/event_sequence_missing_spec_v14.json"
SPEC_SHA256 = "26782bcd1c46662bb705525fa5a72c4d9eb1d57616bbca1ee42edcff1a50aabb"
INTEGRITY_ERRATUM = ROOT / "reports/event_sequence_integrity_erratum_v14.json"
INTEGRITY_ERRATUM_SHA256 = "5d3b166c2971558d85ca491356dfab3acd93ed2f88fac05aeb804cc7f0ba1293"
MOVEMENT_MANIFEST_SHA256 = "1066c5e1c47402a4858381055522f330d09dc9274da298277f40f4f8d6084825"
V8_OOF_SHA256 = "575f1b5e8046135b92c5d051c0272045416d01078da0f4372c379c86d79a7f92"
SEED = 20261017
FOLDS = {"seasonal_jan_jul": (1, 7), "forward_nov_dec": (11, 12),
         "fresh_april_october": (4, 10), "reserved_february_august": (2, 8)}
EXPECTED_GATE = {"seasonal_jan_jul": 4976, "forward_nov_dec": 2865}
SCHEMA_VERSION = 1
NUMERIC_COUNT = 66
CATEGORICAL_COUNT = 13
MAX_VOCAB = 4096
BATCH_SIZE = 4096
MAX_EPOCHS = 30
PATIENCE = 4


def sha256(path: Path) -> str:
    return movement.sha256(path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False) + "\n").encode("utf-8")


def digest_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def write_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as destination:
        destination.write(payload)


def write_json(path: Path, value: dict) -> None:
    write_exclusive(path, json.dumps(value, indent=2, ensure_ascii=False).encode("utf-8") + b"\n")


def publish_temp(temp: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(destination)
    os.link(temp, destination)
    temp.unlink()


def save_parquet_exclusive(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix="v14-", suffix=".parquet",
                                     dir=destination.parent, delete=False) as handle:
        temp = Path(handle.name)
    try:
        frame.to_parquet(temp, index=False)
        publish_temp(temp, destination)
    finally:
        temp.unlink(missing_ok=True)


def assert_min_memory() -> None:
    movement.require_memory(10.0)


def ids_digest(values: Any) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def values_digest(values: Any) -> str:
    a = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(a.dtype).encode("ascii"))
    digest.update(str(a.shape).encode("ascii"))
    digest.update(a.view(np.uint8))
    return digest.hexdigest()


def normalized_time_ns(values: Any) -> np.ndarray:
    time = pd.to_datetime(values, utc=True, errors="coerce")
    if time.isna().any():
        raise ValueError("Missing UTC movement timestamp")
    return time.dt.as_unit("ns").astype("int64").to_numpy()


def ordinary_split(rows: pd.DataFrame, held_months: tuple[int, int]) -> dict[str, np.ndarray]:
    y = rows.target.to_numpy(dtype=float)
    months = rows.month.to_numpy(dtype=np.int16)
    held = np.isin(months, held_months)
    ordinary = np.isfinite(y) & (y >= 0) & (y <= 7200) & ~held
    days = normalized_time_ns(rows.time) // 86_400_000_000_000
    early = ordinary & (days % 11 == 0)
    fit = ordinary & ~early
    if fit.sum() < 100_000 or early.sum() < 10_000:
        raise ValueError("Original movement fit/early eligibility changed")
    return {"fit": np.flatnonzero(fit), "early": np.flatnonzero(early),
            "held_all": np.flatnonzero(held & np.isfinite(y))}


def split_receipt(rows: pd.DataFrame, splits: dict[str, np.ndarray],
                  held_gate: np.ndarray) -> dict:
    result = {}
    for role, idx in {**splits, "held_gate": np.flatnonzero(held_gate)}.items():
        result[role] = {"rows": int(len(idx)),
                        "ordered_id_sha256": ids_digest(rows.MVT_ID_mvt.iloc[idx]),
                        "ordered_target_sha256": values_digest(rows.target.iloc[idx].to_numpy(dtype=float)),
                        "indices_sha256": values_digest(idx.astype(np.int64))}
    return result


def bootstrap_days(frame: pd.DataFrame, old: np.ndarray, new: np.ndarray) -> dict:
    if len(frame) != len(old) or len(old) != len(new):
        raise ValueError("Bootstrap arrays differ in length")
    dates = pd.to_datetime(frame.MVT_TIME_UTC_mvt, utc=True).dt.floor("D")
    group, unique = pd.factorize(dates, sort=True)
    if len(unique) < 20:
        raise ValueError("Too few UTC days for the frozen bootstrap")
    y = frame.target.to_numpy(dtype=float)
    counts = np.bincount(group, minlength=len(unique)).astype(float)
    old_sse = np.bincount(group, weights=(y - old) ** 2, minlength=len(unique))
    new_sse = np.bincount(group, weights=(y - new) ** 2, minlength=len(unique))
    rng = np.random.default_rng(SEED)
    drawn = rng.integers(0, len(unique), size=(1000, len(unique)))
    denom = counts[drawn].sum(axis=1)
    gain = np.sqrt(old_sse[drawn].sum(axis=1) / denom) - np.sqrt(new_sse[drawn].sum(axis=1) / denom)
    return {"days": int(len(unique)), "repeats": 1000, "seed": SEED,
            "gain_ci95_sec": np.quantile(gain, [0.025, 0.975]).tolist(),
            "observed_gain_sec": rmse(y, old) - rmse(y, new)}


def rmse(y: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y, dtype=float) -
                                  np.asarray(prediction, dtype=float)) ** 2)))


def transform_numeric(values: np.ndarray, median: float, iqr: float) -> np.ndarray:
    v = np.asarray(values, dtype=np.float64)
    out = np.full(len(v), -11.0, dtype=np.float32)
    good = np.isfinite(v)
    out[good] = np.clip((v[good] - median) / max(iqr, 1.0), -10, 10).astype(np.float32)
    return out


def canonical_category(value: Any) -> str | None:
    if pd.isna(value):
        return None
    token = str(value)
    # The sealed movement cache uses this explicit token for raw missing
    # categories. Treat it as missing rather than a fit-observed level.
    return None if token == "__MISSING__" else token


def fit_preprocessing(frame: pd.DataFrame, fit_idx: np.ndarray,
                      numeric: list[str], categorical: list[str]) -> dict:
    """Pure fit-row estimator, shared by disk streaming and synthetic checks."""
    if len(fit_idx) == 0 or len(np.unique(fit_idx)) != len(fit_idx):
        raise ValueError("Fit indices missing or duplicated")
    scalers = {}
    for name in numeric:
        values = pd.to_numeric(frame[name].iloc[fit_idx], errors="coerce").to_numpy(
            dtype=float, na_value=np.nan)
        finite = values[np.isfinite(values)]
        median = float(np.median(finite)) if len(finite) else 0.0
        iqr = float(np.quantile(finite, .75) - np.quantile(finite, .25)) if len(finite) else 1.0
        scalers[name] = {"median": median, "iqr": iqr}
    vocabs = {}
    for name in categorical:
        levels = frame[name].iloc[fit_idx].map(canonical_category)
        counts = levels.dropna().value_counts(sort=False)
        counts = counts[counts > 0]
        ordered = sorted(((str(level), int(count)) for level, count in counts.items()),
                         key=lambda pair: (-pair[1], pair[0].encode("utf-8")))[:MAX_VOCAB]
        vocabs[name] = [level for level, _ in ordered]
    return {"numeric": numeric, "categorical": categorical,
            "scalers": scalers, "vocabularies": vocabs,
            "vocabulary_sha256": digest_json(vocabs),
            "fit_indices_sha256": values_digest(fit_idx.astype(np.int64))}


def transform_frame(frame: pd.DataFrame, prep: dict) -> tuple[np.ndarray, np.ndarray]:
    numeric = np.empty((len(frame), len(prep["numeric"])), dtype=np.float32)
    categorical = np.empty((len(frame), len(prep["categorical"])), dtype=np.int16)
    for j, name in enumerate(prep["numeric"]):
        scaler = prep["scalers"][name]
        numeric[:, j] = transform_numeric(pd.to_numeric(frame[name], errors="coerce").to_numpy(
            dtype=float, na_value=np.nan),
                                          scaler["median"], scaler["iqr"])
    for j, name in enumerate(prep["categorical"]):
        mapping = {value: i + 2 for i, value in enumerate(prep["vocabularies"][name])}
        token = frame[name].map(canonical_category)
        categorical[:, j] = np.fromiter((0 if value is None else mapping.get(value, 1)
                                         for value in token), dtype=np.int16, count=len(frame))
    return numeric, categorical


class EventSequenceCNN(nn.Module):
    def __init__(self, vocab_sizes: list[int]):
        super().__init__()
        if len(vocab_sizes) != CATEGORICAL_COUNT:
            raise ValueError("Expected thirteen frozen movement categories")
        self.conv1 = nn.Conv1d(6, 32, 3, padding=1)
        self.conv2 = nn.Conv1d(32, 32, 3, padding=1)
        self.event_projection = nn.Linear(128, 64)
        self.embeddings = nn.ModuleList([nn.Embedding(size, 8) for size in vocab_sizes])
        self.tab1 = nn.Linear(170, 128)
        self.tab2 = nn.Linear(128, 64)
        self.head1 = nn.Linear(128, 64)
        self.head2 = nn.Linear(64, 1)

    def _half(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        active = mask.unsqueeze(1).to(values.dtype)
        x = values.transpose(1, 2) * active
        x = self.conv1(x * active) * active
        x = F.gelu(x) * active
        x = self.conv2(x * active) * active
        x = F.gelu(x) * active
        count = active.sum(dim=2)
        mean = x.sum(dim=2) / count.clamp_min(1.0)
        maximum = x.masked_fill(~mask.unsqueeze(1), -torch.inf).amax(dim=2)
        maximum = torch.where(count > 0, maximum, torch.zeros_like(maximum))
        return torch.cat((mean, maximum), dim=1)

    def forward(self, events: torch.Tensor, presence: torch.Tensor,
                numeric: torch.Tensor, categorical: torch.Tensor) -> torch.Tensor:
        if events.shape[1:] != (32, 6) or presence.shape[1:] != (32,):
            raise ValueError("Expected 32 by 6 events and 32-position mask")
        past = self._half(events[:, :16], presence[:, :16])
        future = self._half(events[:, 16:], presence[:, 16:])
        event = self.event_projection(torch.cat((past, future), dim=1))
        embedded = [layer(categorical[:, j]) for j, layer in enumerate(self.embeddings)]
        tab = F.gelu(self.tab1(torch.cat((numeric, *embedded), dim=1)))
        tab = F.gelu(self.tab2(tab))
        out = F.gelu(self.head1(torch.cat((event, tab), dim=1)))
        return self.head2(out).squeeze(1)


def configure_determinism() -> dict:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if not torch.cuda.is_available():
        raise RuntimeError("Frozen v14 architecture requires CUDA device 0")
    return {"seed": SEED, "torch": torch.__version__, "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"]}


def movement_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(output_dir=args.movement_dir, cache_dir=args.cache_dir,
                              arrival_cache=args.arrival_cache,
                              weather_file=args.weather_file,
                              v5_oof=args.v5_oof, data_dir=args.data_dir)


def composition_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(policy_path=ROOT / "reports/current_candidate_policy_protocol.json",
                              reserved_protocol=ROOT / "reports/reserved_guard_protocol.json",
                              data_dir=args.data_dir, cache_dir=args.cache_dir,
                              v5_oof=args.v5_oof,
                              v7_dir=ROOT / "artifacts/v7-runway-traffic",
                              v7_reserved_dir=ROOT / "artifacts/v8-reserved-v7",
                              v8_dir=ROOT / "artifacts/v8-lightgbm",
                              v8_combo_dir=ROOT / "artifacts/v8-combo",
                              v9b_dir=ROOT / "artifacts/v9b-movement-valid",
                              movement_dir=args.movement_dir,
                              guard_dir=ROOT / "artifacts/reserved-valid-guard",
                              neighbour_dir=ROOT / "artifacts/v6-neighbour",
                              runway_dir=ROOT / "artifacts/v6-runway-arrival",
                              weather_file=args.weather_file,
                              arrival_cache=args.arrival_cache,
                              ranking_arrival_cache=ROOT / "artifacts/v5-arrival-clean/ranking_arrival_features.parquet",
                              v6_ranking_reference=ROOT / "submissions/merry-mushroom_v6.parquet",
                              output_dir=args.v8_dir)


def builder_module():
    import event_sequence_features_v14 as builder
    return builder


def source_paths(args: argparse.Namespace) -> dict[str, Path]:
    from solution import _training_files
    static = {
        "v14_source": Path(__file__), "v14_spec": SPEC,
        "v14_integrity_erratum": INTEGRITY_ERRATUM,
        "event_builder_source": ROOT / "event_sequence_features_v14.py",
        "movement_source": ROOT / "movement_only_expert.py",
        "composition_source": ROOT / "compose_current_candidate.py",
        "solution_source": ROOT / "solution.py", "weather_source": ROOT / "weather_model.py",
        "movement_seal": args.movement_dir / "prepared_integrity.json",
        "movement_manifest": args.movement_dir / "features_manifest.json",
        "movement_features": args.movement_dir / "features.parquet",
        "movement_ids": args.movement_dir / "row_ids.parquet",
        "baseline_rows": args.cache_dir / "training_rows.parquet",
        "baseline_features": args.cache_dir / "features.parquet",
        "arrival_cache": args.arrival_cache, "weather_file": args.weather_file,
        "v5_oof": args.v5_oof,
        "v8_oof": args.v8_dir / "validation_predictions.parquet",
        "v8_composition_report": args.v8_dir / "composition_report.json",
        "v8_composition_protocol": args.v8_dir / "protocol.json",
        "v8_final_local": ROOT / "submissions/merry-mushroom_v8.parquet",
        "v8_final_manifest": ROOT / "submissions/merry-mushroom_v8.manifest.json",
        "event_protocol": args.feature_dir / "training_protocol.json",
        "event_build_report": args.feature_dir / "training_build.json",
        "event_sequences": args.feature_dir / "training_events.f16.memmap",
        "event_presence": args.feature_dir / "training_presence.u8.memmap",
        "event_ids": args.feature_dir / "training_ids.npy",
        "old_fresh_report": ROOT / "reports/movement_only_fresh_audit.json",
        "old_fresh_predictions": args.movement_dir / "april_october_predictions.parquet",
        "old_fresh_movement_model": args.movement_dir / "april_october_movement.txt",
        "old_fresh_reference_model": args.movement_dir / "april_october_reference.cbm",
        "old_reserved_report": ROOT / "reports/movement_only_reserved_audit.json",
        "old_reserved_predictions": args.movement_dir / "february_august_predictions.parquet",
        "old_reserved_movement_model": args.movement_dir / "february_august_movement.txt",
        "old_reserved_reference_model": args.movement_dir / "february_august_reference.cbm",
    }
    for path in _training_files(args.data_dir):
        static[f"released_2025_{path.name}"] = path
    # Include both upstream protocols' transitive sources in our own before/after
    # snapshot. Their verifiers run as well, but a check only at phase entry would
    # miss a changed source during a long GPU fit.
    composer_args = composition_args(args)
    composed_protocol = read_json(composer_args.output_dir / "protocol.json")
    for role, path in current.source_paths(
            composer_args, composed_protocol["valid_terminal"],
            composed_protocol["missing_terminal"]).items():
        static[f"composed_{role}"] = Path(path)
    builder = builder_module()
    inventory = builder.source_inventory(
        scope="training", data_dir=args.data_dir, cache_dir=args.cache_dir,
        movement_dir=args.movement_dir, directory=args.feature_dir)
    for role, item in inventory.items():
        static[f"event_input_{role}"] = Path(item["path"])
    return static


def source_snapshot(args: argparse.Namespace) -> dict[str, str]:
    return {role: sha256(path) for role, path in source_paths(args).items()}


def verify_sources(args: argparse.Namespace, expected: dict[str, str]) -> None:
    actual = source_snapshot(args)
    if actual != expected:
        changed = [key for key in expected if actual.get(key) != expected[key]]
        raise ValueError(f"Frozen v14 source bytes changed: {changed[:8]}")


def verify_existing_sources(args: argparse.Namespace, before: dict[str, str]) -> dict:
    verify_sources(args, before)
    if sha256(SPEC) != SPEC_SHA256:
        raise ValueError("Published v14 scientific spec changed")
    if (before["v14_integrity_erratum"] != INTEGRITY_ERRATUM_SHA256
            or sha256(INTEGRITY_ERRATUM) != INTEGRITY_ERRATUM_SHA256
            or read_json(INTEGRITY_ERRATUM).get("correct_manifest_sha256")
            != MOVEMENT_MANIFEST_SHA256):
        raise ValueError("Published v14 dependency-fingerprint erratum changed")
    if before["movement_manifest"] != MOVEMENT_MANIFEST_SHA256:
        raise ValueError("Original 79-field movement schema changed")
    if before["v8_oof"] != V8_OOF_SHA256:
        raise ValueError("Sealed v8 all-finite OOF changed")
    if before["v8_final_local"] != "8fc6519610a573dd77a4b5ca18f49ab816a564754db13d26d91f26b53d04b9e5":
        raise ValueError("Sealed local v8 ranking reference changed")
    integrity = movement.verify_prepared_integrity(movement_args(args))
    if integrity.get("method") != "independent_rebuild_exact":
        raise ValueError("Movement matrix needs an independent exact rebuild seal")
    _, valid, missing = current.require_composition_gate(composition_args(args))
    if valid["active"] not in ("v7", "v8_combo", "v9b") or not missing["active"]:
        raise ValueError("Frozen v8 component route has changed")
    builder = builder_module()
    built = builder.verify(scope="training", directory=args.feature_dir)
    if built.get("departure_labels_used") is not False:
        raise ValueError("Event cache label-free proof absent")
    verify_sources(args, before)
    return built


def load_rows_and_reference(args: argparse.Namespace) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    rows = movement.read_baseline_rows(args.cache_dir,
                                       ["MVT_ID_mvt", "target", "proxy", "month", "airport", "time"])
    row_ids = pd.read_parquet(args.movement_dir / "row_ids.parquet", columns=["MVT_ID_mvt"])
    if len(rows) != 2_085_047 or not np.array_equal(rows.MVT_ID_mvt.to_numpy(),
                                                     row_ids.MVT_ID_mvt.to_numpy()):
        raise ValueError("Prepared movement row order differs from baseline")
    gate = movement.read_gate(args.cache_dir, rows)
    base = pd.read_parquet(args.v8_dir / "validation_predictions.parquet",
                           columns=["MVT_ID_mvt", "target", "fold", "month",
                                    "MVT_TIME_UTC_mvt", "missing_gate", "combined"])
    if len(base) != 672_428 or base.MVT_ID_mvt.isna().any() or base.MVT_ID_mvt.duplicated().any():
        raise ValueError("Frozen v8 all-finite universe changed")
    pos = pd.Index(rows.MVT_ID_mvt).get_indexer(base.MVT_ID_mvt)
    if np.any(pos < 0):
        raise ValueError("Frozen v8 ID missing in prepared movement universe")
    if (not np.array_equal(base.target.to_numpy(dtype=float), rows.target.to_numpy(dtype=float)[pos])
            or not np.array_equal(base.month.to_numpy(dtype=int), rows.month.to_numpy(dtype=int)[pos])
            or not np.array_equal(normalized_time_ns(base.MVT_TIME_UTC_mvt),
                                  normalized_time_ns(rows.time.iloc[pos]))
            or not np.array_equal(base.missing_gate.to_numpy(dtype=bool), gate[pos])):
        raise ValueError("Frozen v8 ID/target/month/UTC/gate differs from released training")
    if (not np.isfinite(base[["target", "combined"]].to_numpy(dtype=float)).all()
            or (base.combined.to_numpy(dtype=float) < 0).any()
            or int(base.missing_gate.sum()) != 7841):
        raise ValueError("Frozen v8 policy is not all-finite and clipped")
    return rows, gate, base


def verify_event_alignment(args: argparse.Namespace, rows: pd.DataFrame,
                           built: dict) -> tuple[np.ndarray, np.memmap, np.memmap, dict]:
    builder = builder_module()
    ids, events, presence, manifest = builder.open_arrays(
        scope="training", directory=args.feature_dir, verified_receipt=built, mode="r")
    if (len(ids) != len(rows) or not np.array_equal(ids, rows.MVT_ID_mvt.to_numpy())
            or events.shape != (len(rows), 32, 6)
            or presence.shape != (len(rows), 32)
            or events.dtype != np.float16 or presence.dtype != np.uint8):
        raise ValueError("Event bank and prepared movement rows do not align exactly")
    return ids, events, presence, manifest


def protocol_path(args: argparse.Namespace) -> Path:
    return args.output_dir / "protocol.json"


def prepare(args: argparse.Namespace) -> dict:
    assert_min_memory()
    if protocol_path(args).exists():
        raise FileExistsError("v14 protocol already frozen")
    before = source_snapshot(args)
    built = verify_existing_sources(args, before)
    rows, gate, base = load_rows_and_reference(args)
    verify_event_alignment(args, rows, built)
    for fold, months in list(FOLDS.items())[:2]:
        subset = base.fold.eq(fold).to_numpy(dtype=bool)
        if not base.loc[subset, "month"].isin(months).all():
            raise ValueError(f"Frozen v8 {fold} month mapping differs")
        if int(np.sum(subset & base.missing_gate.to_numpy(dtype=bool))) != EXPECTED_GATE[fold]:
            raise ValueError(f"Frozen v8 {fold} gate count differs")
    result = {"schema_version": SCHEMA_VERSION, "scientific_spec_sha256": SPEC_SHA256,
              "source_sha256": before, "event_training_receipt_sha256": sha256(args.feature_dir / "training_build.json"),
              "movement_manifest_sha256": MOVEMENT_MANIFEST_SHA256,
              "v8_oof_sha256": V8_OOF_SHA256,
              "rows": int(len(rows)), "finite_oof_rows": int(len(base)),
              "finite_gate_rows": int(gate[np.isfinite(rows.target.to_numpy(dtype=float))].sum()),
              "architecture": {"events": [32, 6], "numeric": NUMERIC_COUNT,
                               "categorical": CATEGORICAL_COUNT, "seed": SEED,
                               "batch_size": BATCH_SIZE, "max_epochs": MAX_EPOCHS,
                               "patience": PATIENCE, "replacement_weight": 1.0}}
    verify_sources(args, before)
    write_json(protocol_path(args), result)
    verify_sources(args, before)
    return result


def check_protocol(args: argparse.Namespace) -> dict:
    value = read_json(protocol_path(args))
    if (value.get("scientific_spec_sha256") != SPEC_SHA256
            or value.get("source_sha256") != source_snapshot(args)
            or value.get("v8_oof_sha256") != V8_OOF_SHA256
            or value.get("event_training_receipt_sha256")
            != sha256(args.feature_dir / "training_build.json")
            or value.get("architecture", {}).get("replacement_weight") != 1.0):
        raise ValueError("Frozen v14 input/protocol bytes changed")
    return value


def fit_preprocessing_from_parquet(path: Path, fit_idx: np.ndarray,
                                   manifest: dict, fit_id_sha: str) -> dict:
    categorical = list(manifest["categorical"])
    names = list(manifest["features"])
    numeric = [name for name in names if name not in categorical]
    if len(numeric) != NUMERIC_COUNT or len(categorical) != CATEGORICAL_COUNT:
        raise ValueError("Frozen movement numeric/categorical schema differs")
    scalers: dict[str, dict[str, float]] = {}
    for name in numeric:
        s = pd.read_parquet(path, columns=[name])[name]
        x = pd.to_numeric(s.iloc[fit_idx], errors="coerce").to_numpy(dtype=float,
                                                                        na_value=np.nan)
        x = x[np.isfinite(x)]
        scalers[name] = {"median": float(np.median(x)) if len(x) else 0.0,
                         "iqr": float(np.quantile(x, .75) - np.quantile(x, .25)) if len(x) else 1.0}
        del s, x
    vocabs: dict[str, list[str]] = {}
    for name in categorical:
        s = pd.read_parquet(path, columns=[name])[name].iloc[fit_idx]
        token = s.map(canonical_category)
        counts = token.dropna().value_counts(sort=False)
        counts = counts[counts > 0]
        ordered = sorted(((str(key), int(count)) for key, count in counts.items()),
                         key=lambda item: (-item[1], item[0].encode("utf-8")))
        vocabs[name] = [key for key, _ in ordered[:MAX_VOCAB]]
        del s, token, counts, ordered
    return {"numeric": numeric, "categorical": categorical,
            "feature_order": names, "scalers": scalers,
            "vocabularies": vocabs, "vocabulary_sha256": digest_json(vocabs),
            "fit_ordered_id_sha256": fit_id_sha,
            "max_vocab": MAX_VOCAB, "missing_index": 0, "unseen_index": 1,
            "embedding_dim": 8, "numeric_missing_value": -11.0}


class RawMovementBatchView:
    """Keep only raw sealed columns; transform requested rows per minibatch."""

    def __init__(self, path: Path, prep: dict, nrows: int,
                 raw_frame: pd.DataFrame | None = None):
        self.path = path
        self.prep = prep
        self.frame = (pd.read_parquet(path, columns=prep["feature_order"])
                      if raw_frame is None else raw_frame)
        if len(self.frame) != nrows or list(self.frame) != prep["feature_order"]:
            raise ValueError("Raw prepared movement schema/order changed")
        self.source_sha256 = sha256(path)
        self.schema = {name: str(dtype) for name, dtype in self.frame.dtypes.items()}

    def take(self, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return transform_frame(self.frame.iloc[indices], self.prep)


def tensors_for(indices: np.ndarray, events: np.memmap, presence: np.memmap,
                movement_view: RawMovementBatchView,
                device: torch.device) -> tuple[torch.Tensor, ...]:
    # Memmaps remain on disk; only the requested minibatch enters CPU/GPU RAM.
    numeric, categories = movement_view.take(indices)
    return (torch.from_numpy(np.array(events[indices], dtype=np.float32, copy=True)).to(device),
            torch.from_numpy(np.array(presence[indices], dtype=np.bool_, copy=True)).to(device),
            torch.from_numpy(numeric).to(device),
            torch.from_numpy(categories.astype(np.int64, copy=False)).to(device))


@torch.no_grad()
def predict_indices(model: EventSequenceCNN, indices: np.ndarray,
                    events: np.memmap, presence: np.memmap,
                    movement_view: RawMovementBatchView,
                    device: torch.device) -> np.ndarray:
    model.eval()
    output = np.empty(len(indices), dtype=np.float64)
    for start in range(0, len(indices), BATCH_SIZE):
        stop = min(len(indices), start + BATCH_SIZE)
        part = indices[start:stop]
        tensors = tensors_for(part, events, presence, movement_view, device)
        output[start:stop] = (model(*tensors) * 3600).detach().cpu().numpy()
    if not np.isfinite(output).all():
        raise ValueError("Event model predicted nonfinite taxi time")
    return output


def train_model(rows: pd.DataFrame, splits: dict[str, np.ndarray],
                events: np.memmap, presence: np.memmap,
                movement_view: RawMovementBatchView,
                prep: dict) -> tuple[dict, dict]:
    import copy
    runtime = configure_determinism()
    device = torch.device("cuda:0")
    sizes = [len(prep["vocabularies"][name]) + 2 for name in prep["categorical"]]
    model = EventSequenceCNN(sizes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    y = rows.target.to_numpy(dtype=np.float32)
    best_rmse = math.inf
    best_epoch = 0
    history = []
    best_model = None
    best_optimizer = None
    fit_idx, early_idx = splits["fit"], splits["early"]
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        rng = np.random.default_rng(SEED + epoch)
        permuted = rng.permutation(fit_idx)
        train_sse = 0.0
        for start in range(0, len(permuted), BATCH_SIZE):
            part = permuted[start:start + BATCH_SIZE]
            tensors = tensors_for(part, events, presence, movement_view, device)
            label = torch.from_numpy(np.array(y[part] / 3600.0, dtype=np.float32,
                                              copy=True)).to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(*tensors)
            loss = F.mse_loss(prediction, label)
            loss.backward()
            optimizer.step()
            train_sse += float(loss.detach().cpu()) * len(part)
        early_prediction = predict_indices(model, early_idx, events, presence,
                                           movement_view, device)
        early_rmse = rmse(y[early_idx].astype(float), early_prediction)
        item = {"epoch": epoch, "fit_mse_target_div_3600": train_sse / len(fit_idx),
                "early_unclipped_rmse_sec": early_rmse}
        history.append(item)
        print(json.dumps({"v14_progress": item}), flush=True)
        if early_rmse < best_rmse:
            best_rmse = early_rmse
            best_epoch = epoch
            best_model = copy.deepcopy(model.state_dict())
            best_optimizer = copy.deepcopy(optimizer.state_dict())
        if epoch - best_epoch >= PATIENCE:
            break
    if best_model is None or best_optimizer is None:
        raise RuntimeError("No valid internal early-stop epoch")
    package = {"model_state": best_model, "optimizer_state": best_optimizer,
               "best_epoch": best_epoch, "best_early_unclipped_rmse_sec": best_rmse,
               "epochs_executed": len(history), "history": history,
               "vocab_sizes": sizes, "runtime": runtime,
               "requested": {"batch_size": BATCH_SIZE, "max_epochs": MAX_EPOCHS,
                             "patience": PATIENCE, "optimizer": "AdamW",
                             "learning_rate": 0.001, "weight_decay": 0.0001,
                             "loss": "mse_target_div_3600"}}
    report = {"best_epoch": best_epoch, "epochs_executed": len(history),
              "best_early_unclipped_rmse_sec": best_rmse,
              "runtime": runtime, "requested": package["requested"],
              "actual_optimizer": {"name": type(optimizer).__name__,
                                   "learning_rate": optimizer.param_groups[0]["lr"],
                                   "weight_decay": optimizer.param_groups[0]["weight_decay"]},
              "vocab_sizes": sizes, "history_sha256": digest_json(history)}
    return package, report


def save_model_exclusive(package: dict, destination: Path) -> None:
    with tempfile.NamedTemporaryFile(prefix="v14-model-", suffix=".pt",
                                     dir=destination.parent, delete=False) as handle:
        temp = Path(handle.name)
    try:
        torch.save(package, temp)
        publish_temp(temp, destination)
    finally:
        temp.unlink(missing_ok=True)


def readback_model(path: Path, fit_report: dict, prep: dict,
                   model_hash: str) -> EventSequenceCNN:
    if sha256(path) != model_hash:
        raise ValueError("Saved event model bytes changed")
    package = torch.load(path, map_location="cpu", weights_only=False)
    if (package["best_epoch"] != fit_report["best_epoch"]
            or package["epochs_executed"] != fit_report["epochs_executed"]
            or package["requested"] != fit_report["requested"]
            or package["runtime"] != fit_report["runtime"]
            or package["vocab_sizes"] != fit_report["vocab_sizes"]
            or digest_json(package["history"]) != fit_report["history_sha256"]
            or not 1 <= package["best_epoch"] <= MAX_EPOCHS
            or package["epochs_executed"] > MAX_EPOCHS):
        raise ValueError("Saved event checkpoint metadata does not match fit receipt")
    optimizer = package["optimizer_state"]
    groups = optimizer["param_groups"]
    if (len(groups) != 1 or groups[0]["lr"] != .001
            or groups[0]["weight_decay"] != .0001):
        raise ValueError("Saved AdamW state differs from frozen optimizer")
    expected_sizes = [len(prep["vocabularies"][name]) + 2 for name in prep["categorical"]]
    if package["vocab_sizes"] != expected_sizes:
        raise ValueError("Saved model category vocabulary sizes differ")
    model = EventSequenceCNN(expected_sizes)
    model.load_state_dict(package["model_state"], strict=True)
    return model


def old_matched_path(args: argparse.Namespace, fold: str) -> tuple[Path, Path, str]:
    if fold == "fresh_april_october":
        return (args.movement_dir / "april_october_predictions.parquet",
                ROOT / "reports/movement_only_fresh_audit.json", "april_october")
    if fold == "reserved_february_august":
        return (args.movement_dir / "february_august_predictions.parquet",
                ROOT / "reports/movement_only_reserved_audit.json", "february_august")
    raise ValueError("Original folds do not have a matched saved audit")


def verify_old_matched(args: argparse.Namespace, fold: str,
                       rows: pd.DataFrame, held_gate: np.ndarray) -> pd.DataFrame:
    path, report_path, stem = old_matched_path(args, fold)
    report = read_json(report_path)
    months = FOLDS[fold]
    if (report.get("heldout_months") != list(months)
            or not report.get("passed") or not report.get("coverage_verified")
            or float(report.get("fixed_weight", -1)) != 1.0
            or report.get("prediction_sha256") != sha256(path)
            or report.get("gate_rows") != int(held_gate.sum())):
        raise ValueError(f"Saved old {stem} matched audit changed or failed")
    for label, filename in (("movement", f"{stem}_movement.txt"),
                            ("catboost", f"{stem}_reference.cbm")):
        if report.get("model_sha256", {}).get(label) != sha256(args.movement_dir / filename):
            raise ValueError(f"Old {stem} model hash changed: {label}")
    movement.verify_audit_artifacts(args.movement_dir, args.cache_dir, report, stem, months)
    frame = pd.read_parquet(path)
    expected = rows.loc[held_gate]
    if (list(frame) != ["MVT_ID_mvt", "target", "airport", "month",
                           "MVT_TIME_UTC_mvt", "reference_direct",
                           "movement_direct", "fixed_blend"]
            or not np.array_equal(frame.MVT_ID_mvt.to_numpy(), expected.MVT_ID_mvt.to_numpy())
            or not np.array_equal(frame.target.to_numpy(dtype=float),
                                  expected.target.to_numpy(dtype=float))
            or not np.array_equal(normalized_time_ns(frame.MVT_TIME_UTC_mvt),
                                  normalized_time_ns(expected.time))
            or not np.isfinite(frame.fixed_blend.to_numpy(dtype=float)).all()):
        raise ValueError(f"Old {stem} held-out ID/target/time/reference differs")
    return frame


def fold_dir(args: argparse.Namespace, fold: str) -> Path:
    if fold not in FOLDS:
        raise ValueError("Unknown frozen v14 fold")
    return args.output_dir / fold


def verify_fold_receipt(args: argparse.Namespace, fold: str,
                        rows: pd.DataFrame | None = None,
                        gate: np.ndarray | None = None,
                        raw_frame: pd.DataFrame | None = None,
                        events: np.memmap | None = None,
                        presence: np.memmap | None = None) -> tuple[pd.DataFrame, dict]:
    if rows is None or gate is None:
        raise ValueError("Fold receipt replay requires exact baseline rows and gate")
    frozen = check_protocol(args)
    directory = fold_dir(args, fold)
    receipt_path = directory / "provenance.json"
    receipt = read_json(receipt_path)
    fit_path = directory / "fit.json"
    prep_path = directory / "preprocessing.json"
    model_path = directory / "best_model.pt"
    oof_path = directory / "oof.parquet"
    fit = read_json(fit_path)
    prep = read_json(prep_path)
    expected = {"protocol_sha256": sha256(protocol_path(args)),
                "source_sha256": frozen["source_sha256"],
                "fit_sha256": sha256(fit_path), "preprocessing_sha256": sha256(prep_path),
                "model_sha256": sha256(model_path), "oof_sha256": sha256(oof_path),
                "raw_movement_features_sha256": sha256(args.movement_dir / "features.parquet")}
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Frozen {fold} model/output receipt changed")
    if (receipt.get("fold") != fold or receipt.get("heldout_months") != list(FOLDS[fold])
            or fit.get("fold") != fold or prep.get("vocabulary_sha256")
            != digest_json(prep.get("vocabularies"))
            or prep.get("fit_ordered_id_sha256")
            != receipt["split"]["fit"]["ordered_id_sha256"]
            or fit.get("split") != receipt.get("split")
            or fit.get("preprocessing_sha256") != expected["preprocessing_sha256"]
            or fit.get("raw_movement_features_sha256") != expected["raw_movement_features_sha256"]):
        raise ValueError(f"Frozen {fold} fit/split/preprocessing receipt differs")
    model = readback_model(model_path, fit, prep, expected["model_sha256"])
    part = pd.read_parquet(oof_path)
    required = ["MVT_ID_mvt", "target", "airport", "month", "MVT_TIME_UTC_mvt",
                "fold", "expert"]
    if (list(part) != required or part.MVT_ID_mvt.isna().any()
            or part.MVT_ID_mvt.duplicated().any() or not part.fold.eq(fold).all()
            or not np.isfinite(part[["target", "expert"]].to_numpy(dtype=float)).all()
            or np.any(part.expert.to_numpy(dtype=float) < 0)):
        raise ValueError(f"Frozen {fold} OOF schema or values changed")
    if rows is not None:
        assert gate is not None
        held = (rows.month.isin(FOLDS[fold]).to_numpy(dtype=bool)
                & gate & np.isfinite(rows.target.to_numpy(dtype=float)))
        expected_rows = rows.loc[held]
        splits = ordinary_split(rows, FOLDS[fold])
        current_split = split_receipt(rows, splits, held)
        if (receipt["split"] != current_split
                or not np.array_equal(part.MVT_ID_mvt.to_numpy(),
                                      expected_rows.MVT_ID_mvt.to_numpy())
                or not np.array_equal(part.target.to_numpy(dtype=float),
                                      expected_rows.target.to_numpy(dtype=float))
                or not np.array_equal(part.month.to_numpy(dtype=int),
                                      expected_rows.month.to_numpy(dtype=int))
                or not np.array_equal(normalized_time_ns(part.MVT_TIME_UTC_mvt),
                                      normalized_time_ns(expected_rows.time))):
            raise ValueError(f"Frozen {fold} OOF held-out and split proof differs")
        if events is None or presence is None:
            built = builder_module().verify(scope="training", directory=args.feature_dir)
            _, events, presence, _ = verify_event_alignment(args, rows, built)
        view = RawMovementBatchView(args.movement_dir / "features.parquet", prep,
                                    len(rows), raw_frame=raw_frame)
        configure_determinism()
        model = model.to("cuda:0")
        raw = predict_indices(model, np.flatnonzero(held), events, presence,
                              view, torch.device("cuda:0"))
        if (receipt.get("readback_prediction_sha256") != values_digest(raw)
                or not np.array_equal(np.maximum(raw, 0),
                                      part.expert.to_numpy(dtype=float))):
            raise ValueError(f"Frozen {fold} checkpoint does not reproduce saved OOF")
    return part, receipt


def fit_fold(args: argparse.Namespace) -> dict:
    assert_min_memory()
    if args.fold not in FOLDS:
        raise ValueError("Unknown or non-frozen v14 fold")
    if args.fold == "fresh_april_october":
        require_original_pass(args)
    if args.fold == "reserved_february_august":
        require_fresh_pass(args)
    frozen = check_protocol(args)
    before = source_snapshot(args)
    built = verify_existing_sources(args, before)
    rows, gate, base = load_rows_and_reference(args)
    _, events, presence, _ = verify_event_alignment(args, rows, built)
    months = FOLDS[args.fold]
    splits = ordinary_split(rows, months)
    held_gate = (rows.month.isin(months).to_numpy(dtype=bool)
                 & gate & np.isfinite(rows.target.to_numpy(dtype=float)))
    if args.fold in EXPECTED_GATE and int(held_gate.sum()) != EXPECTED_GATE[args.fold]:
        raise ValueError("Original fold's exact no-NM gate count differs")
    if args.fold not in EXPECTED_GATE:
        verify_old_matched(args, args.fold, rows, held_gate)
    receipt_split = split_receipt(rows, splits, held_gate)
    directory = fold_dir(args, args.fold)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError("v14 fold outputs already exist; never overwrite")
    directory.mkdir(parents=True, exist_ok=True)
    movement_manifest = read_json(args.movement_dir / "features_manifest.json")
    prep = fit_preprocessing_from_parquet(args.movement_dir / "features.parquet",
                                          splits["fit"], movement_manifest,
                                          receipt_split["fit"]["ordered_id_sha256"])
    write_json(directory / "preprocessing.json", prep)
    raw_path = args.movement_dir / "features.parquet"
    metadata_rows = pq.ParquetFile(raw_path).metadata.num_rows
    if metadata_rows != len(rows):
        raise ValueError("Raw movement Parquet metadata row count differs")
    raw_ram_estimate = int(metadata_rows * (NUMERIC_COUNT * 8 + CATEGORICAL_COUNT * 8))
    view = RawMovementBatchView(raw_path, prep, len(rows))
    package, fit = train_model(rows, splits, events, presence, view, prep)
    fit.update({"fold": args.fold, "heldout_months": list(months),
                "split": receipt_split,
                "preprocessing_sha256": sha256(directory / "preprocessing.json"),
                "raw_movement_features_sha256": view.source_sha256,
                "raw_movement_frame_schema": view.schema,
                "metadata_only_raw_ram_estimate_bytes": raw_ram_estimate,
                "actual_raw_dataframe_memory_bytes": int(view.frame.memory_usage(deep=True).sum()),
                "normalized_full_matrix_materialized": False,
                "source_sha256": before,
                "protocol_sha256": sha256(protocol_path(args))})
    verify_sources(args, before)
    model_path = directory / "best_model.pt"
    save_model_exclusive(package, model_path)
    readback = readback_model(model_path, fit, prep, sha256(model_path)).to("cuda:0")
    held_idx = np.flatnonzero(held_gate)
    raw_prediction = predict_indices(readback, held_idx, events, presence,
                                     view, torch.device("cuda:0"))
    frame = rows.iloc[held_idx][["MVT_ID_mvt", "target", "airport", "month", "time"]].copy()
    frame = frame.rename(columns={"time": "MVT_TIME_UTC_mvt"})
    frame["fold"] = args.fold
    frame["expert"] = np.maximum(raw_prediction, 0)
    if not np.isfinite(frame.expert.to_numpy(dtype=float)).all():
        raise ValueError("Saved readback model produced nonfinite held-out prediction")
    # On originals, the base OOF is in a different order; prove exact ID coverage.
    if args.fold in EXPECTED_GATE:
        relevant = base.loc[base.fold.eq(args.fold) & base.missing_gate]
        if set(relevant.MVT_ID_mvt) != set(frame.MVT_ID_mvt):
            raise ValueError("Original expert coverage differs from sealed v8 gate")
    verify_sources(args, before)
    save_parquet_exclusive(frame, directory / "oof.parquet")
    write_json(directory / "fit.json", fit)
    receipt = {"schema_version": SCHEMA_VERSION, "fold": args.fold,
               "heldout_months": list(months), "split": receipt_split,
               "protocol_sha256": sha256(protocol_path(args)),
               "source_sha256": frozen["source_sha256"],
               "fit_sha256": sha256(directory / "fit.json"),
               "preprocessing_sha256": sha256(directory / "preprocessing.json"),
               "raw_movement_features_sha256": sha256(raw_path),
               "model_sha256": sha256(model_path),
               "oof_sha256": sha256(directory / "oof.parquet"),
               "readback_prediction_sha256": values_digest(raw_prediction),
               "readback_exact_model_state": True,
               "source_recheck_before_after": True}
    write_json(directory / "provenance.json", receipt)
    verify_sources(args, before)
    verify_fold_receipt(args, args.fold, rows, gate, raw_frame=view.frame,
                        events=events, presence=presence)
    return {"fold": args.fold, "best_epoch": fit["best_epoch"],
            "gate_rows": int(held_gate.sum()),
            "provenance_sha256": sha256(directory / "provenance.json")}


def error_concentration(y: np.ndarray, prediction: np.ndarray) -> dict:
    squared = np.square(np.asarray(y, dtype=float) - np.asarray(prediction, dtype=float))
    count = max(1, int(math.ceil(len(squared) * .01)))
    total = float(squared.sum())
    return {"top_one_percent_rows": count,
            "top_one_percent_sse_share": float(np.sort(squared)[-count:].sum() / total)
            if total > 0 else 0.0}


def score_original(args: argparse.Namespace) -> tuple[pd.DataFrame, dict]:
    check_protocol(args)
    before = source_snapshot(args)
    built = verify_existing_sources(args, before)
    rows, gate, base = load_rows_and_reference(args)
    _, events, presence, _ = verify_event_alignment(args, rows, built)
    movement_manifest = read_json(args.movement_dir / "features_manifest.json")
    raw_frame = pd.read_parquet(args.movement_dir / "features.parquet",
                                columns=movement_manifest["features"])
    parts = []
    sources = {}
    for fold in list(FOLDS)[:2]:
        part, receipt = verify_fold_receipt(args, fold, rows, gate,
                                            raw_frame=raw_frame,
                                            events=events, presence=presence)
        parts.append(part)
        sources[fold] = {"oof_sha256": receipt["oof_sha256"],
                         "model_sha256": receipt["model_sha256"],
                         "provenance_sha256": sha256(fold_dir(args, fold) / "provenance.json")}
    oof = pd.concat(parts, ignore_index=True)
    if oof.MVT_ID_mvt.duplicated().any() or len(oof) != 7841:
        raise ValueError("Original event expert ID coverage is not exact")
    aligned = oof.set_index("MVT_ID_mvt").reindex(base.MVT_ID_mvt)
    gate_oof = base.missing_gate.to_numpy(dtype=bool)
    expert = aligned.expert.to_numpy(dtype=float)
    if (not np.array_equal(np.isfinite(expert), gate_oof)
            or not np.array_equal(aligned.fold.to_numpy()[gate_oof],
                                  base.fold.to_numpy()[gate_oof])
            or not np.array_equal(aligned.target.to_numpy(dtype=float)[gate_oof],
                                  base.target.to_numpy(dtype=float)[gate_oof])
            or not np.array_equal(normalized_time_ns(aligned.MVT_TIME_UTC_mvt[gate_oof]),
                                  normalized_time_ns(base.MVT_TIME_UTC_mvt[gate_oof]))):
        raise ValueError("Event original expert ID/target/time/gate differs from v8")
    old = base.combined.to_numpy(dtype=float)
    candidate = old.copy()
    candidate[gate_oof] = np.maximum(expert[gate_oof], 0)
    if (not np.array_equal(candidate[~gate_oof], old[~gate_oof])
            or not np.isfinite(candidate).all() or (candidate < 0).any()):
        raise ValueError("Original candidate altered v8 outside exact no-NM gate")
    y = base.target.to_numpy(dtype=float)
    metrics = {}
    for fold, months in list(FOLDS.items())[:2]:
        fold_mask = base.fold.eq(fold).to_numpy(dtype=bool)
        if not base.loc[fold_mask, "month"].isin(months).all():
            raise ValueError("Original fold/month mapping changed")
        by_month = {}
        for month in months:
            subset = fold_mask & gate_oof & base.month.eq(month).to_numpy(dtype=bool)
            if not subset.any():
                raise ValueError("Original month has no finite no-NM gate rows")
            by_month[str(month)] = {"n": int(subset.sum()),
                                   "old_gate_rmse_sec": rmse(y[subset], old[subset]),
                                   "new_gate_rmse_sec": rmse(y[subset], candidate[subset]),
                                   "old_error_concentration": error_concentration(y[subset], old[subset]),
                                   "new_error_concentration": error_concentration(y[subset], candidate[subset])}
        stability = bootstrap_days(base.loc[fold_mask], old[fold_mask], candidate[fold_mask])
        old_all = rmse(y[fold_mask], old[fold_mask])
        new_all = rmse(y[fold_mask], candidate[fold_mask])
        passed = (all(item["new_gate_rmse_sec"] < item["old_gate_rmse_sec"]
                      for item in by_month.values())
                  and new_all < old_all and stability["gain_ci95_sec"][0] > 0)
        metrics[fold] = {"all_finite_n": int(fold_mask.sum()),
                         "gate_n": int((fold_mask & gate_oof).sum()),
                         "old_all_rmse_sec": old_all,
                         "new_all_rmse_sec": new_all,
                         "months": by_month, "day_bootstrap": stability,
                         "passed": bool(passed)}
        if metrics[fold]["gate_n"] != EXPECTED_GATE[fold]:
            raise ValueError("Original fold exact gate count differs")
    output = base[["MVT_ID_mvt", "target", "fold", "month",
                   "MVT_TIME_UTC_mvt", "missing_gate"]].copy()
    output["v8"] = old
    output["expert"] = expert
    output["selected"] = candidate
    verify_sources(args, before)
    report = {"schema_version": SCHEMA_VERSION, "stage": "original",
              "protocol_sha256": sha256(protocol_path(args)),
              "source_sha256": before,
              "fixed_weight": 1.0, "bootstrap_seed": SEED,
              "source_models": sources, "metrics": metrics,
              "all_finite_rows": int(len(base)), "gate_rows": 7841,
              "outside_gate_exact": True,
              "passed": bool(all(item["passed"] for item in metrics.values())),
              "decision": "passed_to_matched_audit" if all(item["passed"] for item in metrics.values())
              else "rejected_without_retuning"}
    return output, report


def evaluate_original(args: argparse.Namespace) -> dict:
    assert_min_memory()
    report_path = args.output_dir / "original_validation.json"
    output_path = args.output_dir / "original_predictions.parquet"
    if report_path.exists() or output_path.exists():
        raise FileExistsError("Original v14 validation already exists")
    output, report = score_original(args)
    before = report["source_sha256"]
    save_parquet_exclusive(output, output_path)
    report["predictions_sha256"] = sha256(output_path)
    verify_sources(args, before)
    write_json(report_path, report)
    verify_sources(args, before)
    return report


def require_original_pass(args: argparse.Namespace) -> dict:
    saved = read_json(args.output_dir / "original_validation.json")
    output, replay = score_original(args)
    prediction_path = args.output_dir / "original_predictions.parquet"
    if (not saved.get("passed") or saved.get("decision") != "passed_to_matched_audit"
            or saved.get("predictions_sha256") != sha256(prediction_path)
            or {key: value for key, value in saved.items() if key != "predictions_sha256"} != replay):
        raise ValueError("Original v14 gate or immutable validation report did not pass replay")
    stored = pd.read_parquet(prediction_path)
    if not stored.equals(output):
        raise ValueError("Saved original all-finite policy differs from fixed replay")
    return saved


def score_matched(args: argparse.Namespace, fold: str) -> tuple[pd.DataFrame, dict]:
    if fold == "fresh_april_october":
        require_original_pass(args)
    elif fold == "reserved_february_august":
        require_fresh_pass(args)
    else:
        raise ValueError("Matched score must be fresh or reserved")
    check_protocol(args)
    before = source_snapshot(args)
    built = verify_existing_sources(args, before)
    rows, gate, _ = load_rows_and_reference(args)
    _, events, presence, _ = verify_event_alignment(args, rows, built)
    raw_frame = pd.read_parquet(args.movement_dir / "features.parquet")
    held = (rows.month.isin(FOLDS[fold]).to_numpy(dtype=bool)
            & gate & np.isfinite(rows.target.to_numpy(dtype=float)))
    old = verify_old_matched(args, fold, rows, held)
    expert, receipt = verify_fold_receipt(args, fold, rows, gate,
                                          raw_frame=raw_frame,
                                          events=events, presence=presence)
    if (not np.array_equal(old.MVT_ID_mvt.to_numpy(), expert.MVT_ID_mvt.to_numpy())
            or not np.array_equal(old.target.to_numpy(dtype=float),
                                  expert.target.to_numpy(dtype=float))
            or not np.array_equal(normalized_time_ns(old.MVT_TIME_UTC_mvt),
                                  normalized_time_ns(expert.MVT_TIME_UTC_mvt))):
        raise ValueError("Matched event expert differs from saved old movement pair")
    y = old.target.to_numpy(dtype=float)
    baseline = old.fixed_blend.to_numpy(dtype=float)
    candidate = expert.expert.to_numpy(dtype=float)
    if not np.isfinite(candidate).all() or (candidate < 0).any():
        raise ValueError("Matched candidate has nonfinite or negative prediction")
    by_month = {}
    for month in FOLDS[fold]:
        subset = old.month.eq(month).to_numpy(dtype=bool)
        by_month[str(month)] = {"n": int(subset.sum()),
                                "old_gate_rmse_sec": rmse(y[subset], baseline[subset]),
                                "new_gate_rmse_sec": rmse(y[subset], candidate[subset]),
                                "old_error_concentration": error_concentration(y[subset], baseline[subset]),
                                "new_error_concentration": error_concentration(y[subset], candidate[subset])}
    stability = bootstrap_days(old, baseline, candidate)
    passed = (all(item["new_gate_rmse_sec"] < item["old_gate_rmse_sec"]
                  for item in by_month.values())
              and stability["gain_ci95_sec"][0] > 0)
    output = old[["MVT_ID_mvt", "target", "airport", "month", "MVT_TIME_UTC_mvt"]].copy()
    output["old_fixed_blend"] = baseline
    output["expert"] = candidate
    output["selected"] = candidate  # Frozen replacement weight is exactly one.
    report = {"schema_version": SCHEMA_VERSION, "stage": fold,
              "heldout_months": list(FOLDS[fold]),
              "protocol_sha256": sha256(protocol_path(args)),
              "source_sha256": before,
              "old_matched_report_sha256": sha256(old_matched_path(args, fold)[1]),
              "old_matched_predictions_sha256": sha256(old_matched_path(args, fold)[0]),
              "fold_provenance_sha256": sha256(fold_dir(args, fold) / "provenance.json"),
              "model_sha256": receipt["model_sha256"],
              "gate_rows": int(len(old)), "month_scores": by_month,
              "pooled_day_bootstrap": stability, "bootstrap_seed": SEED,
              "fixed_weight": 1.0, "passed": bool(passed),
              "decision": "passed_to_next_gate" if passed else "rejected_without_retuning"}
    verify_sources(args, before)
    return output, report


def evaluate_matched(args: argparse.Namespace, fold: str) -> dict:
    assert_min_memory()
    path = args.output_dir / f"{fold}_audit.json"
    prediction_path = args.output_dir / f"{fold}_predictions.parquet"
    if path.exists() or prediction_path.exists():
        raise FileExistsError(f"{fold} audit already exists")
    output, report = score_matched(args, fold)
    before = report["source_sha256"]
    save_parquet_exclusive(output, prediction_path)
    report["predictions_sha256"] = sha256(prediction_path)
    verify_sources(args, before)
    write_json(path, report)
    verify_sources(args, before)
    return report


def require_matched_pass(args: argparse.Namespace, fold: str) -> dict:
    saved = read_json(args.output_dir / f"{fold}_audit.json")
    output, replay = score_matched(args, fold)
    prediction_path = args.output_dir / f"{fold}_predictions.parquet"
    if (not saved.get("passed") or saved.get("decision") != "passed_to_next_gate"
            or saved.get("predictions_sha256") != sha256(prediction_path)
            or {key: value for key, value in saved.items() if key != "predictions_sha256"} != replay):
        raise ValueError(f"{fold} paired audit did not pass exact read-only replay")
    if not pd.read_parquet(prediction_path).equals(output):
        raise ValueError(f"Saved {fold} predictions differ from fixed replay")
    return saved


def require_fresh_pass(args: argparse.Namespace) -> dict:
    return require_matched_pass(args, "fresh_april_october")


def require_reserved_pass(args: argparse.Namespace) -> dict:
    return require_matched_pass(args, "reserved_february_august")


def freeze_terminal(args: argparse.Namespace) -> dict:
    assert_min_memory()
    terminal_path = args.output_dir / "terminal.json"
    if terminal_path.exists():
        raise FileExistsError("v14 terminal already frozen")
    original = require_original_pass(args)
    fresh = require_fresh_pass(args)
    reserved = require_reserved_pass(args)
    before = source_snapshot(args)
    result = {"schema_version": SCHEMA_VERSION, "passed": True,
              "route": "exact_v8_no_nm_non_lirf_only",
              "replacement_weight": 1.0,
              "scientific_spec_sha256": SPEC_SHA256,
              "protocol_sha256": sha256(protocol_path(args)),
              "source_snapshot_sha256": digest_json(before),
              "original_report_sha256": sha256(args.output_dir / "original_validation.json"),
              "fresh_report_sha256": sha256(args.output_dir / "fresh_april_october_audit.json"),
              "reserved_report_sha256": sha256(args.output_dir / "reserved_february_august_audit.json"),
              "original_best_epochs": [read_json(fold_dir(args, name) / "fit.json")["best_epoch"]
                                       for name in list(FOLDS)[:2]],
              "final_epochs_if_separately_authorized": int(math.floor(np.median(
                  [read_json(fold_dir(args, name) / "fit.json")["best_epoch"]
                   for name in list(FOLDS)[:2]]))),
              "original_gate_rows": original["gate_rows"],
              "fresh_gate_rows": fresh["gate_rows"],
              "reserved_gate_rows": reserved["gate_rows"],
              "ranking_and_final_modes": "not_implemented_pending_separate_extension"}
    verify_sources(args, before)
    write_json(terminal_path, result)
    verify_sources(args, before)
    return result


def require_terminal(output_dir: Path,
                     *, feature_dir: Path = Path("artifacts/v14-event-sequence/features"),
                     movement_dir: Path = Path("artifacts/v6-movement-only"),
                     cache_dir: Path = Path("artifacts/baseline")) -> dict:
    args = default_args()
    args.output_dir = Path(output_dir)
    args.feature_dir = Path(feature_dir)
    args.movement_dir = Path(movement_dir)
    args.cache_dir = Path(cache_dir)
    saved = read_json(args.output_dir / "terminal.json")
    original = require_original_pass(args)
    fresh = require_fresh_pass(args)
    reserved = require_reserved_pass(args)
    check_protocol(args)
    before = source_snapshot(args)
    required = {"passed": True, "route": "exact_v8_no_nm_non_lirf_only",
                "replacement_weight": 1.0,
                "scientific_spec_sha256": SPEC_SHA256,
                "protocol_sha256": sha256(protocol_path(args)),
                "source_snapshot_sha256": digest_json(before),
                "original_report_sha256": sha256(args.output_dir / "original_validation.json"),
                "fresh_report_sha256": sha256(args.output_dir / "fresh_april_october_audit.json"),
                "reserved_report_sha256": sha256(args.output_dir / "reserved_february_august_audit.json"),
                "original_gate_rows": original["gate_rows"],
                "fresh_gate_rows": fresh["gate_rows"],
                "reserved_gate_rows": reserved["gate_rows"]}
    if any(saved.get(key) != value for key, value in required.items()):
        raise ValueError("v14 terminal is absent, changed or failed")
    verify_sources(args, before)
    return saved


def self_test() -> dict:
    """Small CPU-only architecture/preprocessing checks; fits zero models."""
    torch.manual_seed(17)
    model = EventSequenceCNN([5] * CATEGORICAL_COUNT).eval()
    events = torch.zeros((3, 32, 6), dtype=torch.float32)
    mask = torch.zeros((3, 32), dtype=torch.bool)
    mask[1, 15] = True
    mask[2, 16] = True
    events[1, 15, 0] = -0.2
    events[2, 16, 0] = 0.2
    numeric = torch.zeros((3, NUMERIC_COUNT), dtype=torch.float32)
    categorical = torch.zeros((3, CATEGORICAL_COUNT), dtype=torch.long)
    with torch.no_grad():
        baseline = model(events, mask, numeric, categorical)
        altered = events.clone()
        altered[~mask] = 19.0
        padded = model(altered, mask, numeric, categorical)
        past_before = model._half(events[:, :16], mask[:, :16])
        future_before = model._half(events[:, 16:], mask[:, 16:])
        future_changed = events.clone()
        future_changed[:, 16:, :] = 7.0
        past_after = model._half(future_changed[:, :16], mask[:, :16])
    if (not torch.equal(baseline, padded)
            or not torch.equal(past_before, past_after)
            or not torch.equal(past_before[0], torch.zeros(64))
            or not torch.equal(future_before[0], torch.zeros(64))):
        raise AssertionError("Masked padding or half-isolation check failed")
    frame = pd.DataFrame({
        **{f"n{j}": [1.0, 3.0, 1_000_000.0] for j in range(NUMERIC_COUNT)},
        **{f"c{j}": pd.Categorical(
            ["common", "common", "heldout_only"],
            categories=["common", "heldout_only", "unused_global", "__MISSING__"])
           for j in range(CATEGORICAL_COUNT)}})
    prep = fit_preprocessing(frame, np.array([0, 1]),
                             [f"n{j}" for j in range(NUMERIC_COUNT)],
                             [f"c{j}" for j in range(CATEGORICAL_COUNT)])
    numeric_out, cat_out = transform_frame(frame, prep)
    if (prep["scalers"]["n0"]["median"] != 2.0
            or prep["vocabularies"]["c0"] != ["common"]
            or numeric_out[2, 0] != 10.0
            or cat_out[2, 0] != 1):
        raise AssertionError("Fit-only scaler/vocabulary or unseen encoding failed")
    with tempfile.TemporaryDirectory(prefix="v14-small-parquet-") as folder:
        path = Path(folder) / "categorical_fixture.parquet"
        frame.to_parquet(path, index=False)
        manifest = {"features": list(frame),
                    "categorical": [f"c{j}" for j in range(CATEGORICAL_COUNT)]}
        streamed = fit_preprocessing_from_parquet(path, np.array([0, 1]),
                                                   manifest, "synthetic_fit_ids")
        if streamed["vocabularies"]["c0"] != ["common"]:
            raise AssertionError("Global Parquet categorical levels leaked into fit vocabulary")
    return {"passed": True, "models_fitted": 0,
            "masked_padding_invariant": True,
            "past_future_seam_isolated": True,
            "empty_half_zero": True,
            "heldout_level_unseen": True,
            "heldout_numeric_not_fit": True}


def default_args() -> argparse.Namespace:
    return argparse.Namespace(
        data_dir=ROOT / "data", cache_dir=ROOT / "artifacts/baseline",
        movement_dir=ROOT / "artifacts/v6-movement-only",
        feature_dir=ROOT / "artifacts/v14-event-sequence/features",
        output_dir=ROOT / "artifacts/v14-event-sequence",
        v8_dir=ROOT / "artifacts/current-candidate",
        v5_oof=ROOT / "artifacts/v5-ensemble/validation_predictions.parquet",
        arrival_cache=ROOT / "artifacts/v5-arrival-clean/training_arrival_features.parquet",
        weather_file=ROOT / "data/external/weather.parquet",
        fold=None)


def main() -> None:
    defaults = default_args()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True,
                        choices=("show-spec", "self-test", "prepare", "fit-fold",
                                 "evaluate-original", "evaluate-fresh",
                                 "evaluate-reserved", "terminal", "verify-terminal",
                                 "final", "ranking"))
    parser.add_argument("--fold", choices=tuple(FOLDS))
    for name in ("data_dir", "cache_dir", "movement_dir", "feature_dir",
                 "output_dir", "v8_dir", "v5_oof", "arrival_cache", "weather_file"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path,
                            default=getattr(defaults, name))
    args = parser.parse_args()
    if args.mode in ("final", "ranking"):
        raise RuntimeError("Final and ranking operations require a separate published extension and guards")
    if args.mode == "show-spec":
        if sha256(SPEC) != SPEC_SHA256:
            raise ValueError("Published v14 spec SHA changed")
        result = {"scientific_spec": str(SPEC), "scientific_spec_sha256": SPEC_SHA256,
                  "frozen_folds": {name: list(months) for name, months in FOLDS.items()},
                  "fixed_weight": 1.0, "real_models_fitted": 0}
    elif args.mode == "self-test":
        result = self_test()
    elif args.mode == "prepare":
        result = prepare(args)
    elif args.mode == "fit-fold":
        if args.fold is None:
            parser.error("--fit-fold requires --fold")
        result = fit_fold(args)
    elif args.mode == "evaluate-original":
        result = evaluate_original(args)
    elif args.mode == "evaluate-fresh":
        result = evaluate_matched(args, "fresh_april_october")
    elif args.mode == "evaluate-reserved":
        result = evaluate_matched(args, "reserved_february_august")
    elif args.mode == "terminal":
        result = freeze_terminal(args)
    else:
        result = require_terminal(args.output_dir, feature_dir=args.feature_dir,
                                  movement_dir=args.movement_dir,
                                  cache_dir=args.cache_dir)
    print(json.dumps(result, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
