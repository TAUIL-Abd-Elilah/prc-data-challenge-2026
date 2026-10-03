"""Prospective 20,000-tree capacity audit for the frozen 184-feature v11 expert.

This module compares one longer CatBoost budget with the locally sealed v9
policy. It never fits a final model, predicts ranking rows, or changes the
existing policy. Original folds and the matched April/October audit are the
only executable model comparisons here; a separate component guard and
composition protocol would be required for any later use.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

import deep_arrival_expert as arrival
import deep_timestamp_expert as deep
import later_feature_final as final
import later_feature_portfolio as portfolio
import taxi_interval_flow_features as flow
import v11_taxi_interval_flow_expert as v11
from solution import _training_files


FOLDS = deep.FOLDS
FRESH_MONTHS = (4, 10)
EXPECTED_OOF_ROWS = 672428
EXPECTED_FRESH_ROWS = 357813
FEATURE_COUNT = 184
CATEGORICAL_COUNT = 24
PREFIX_TREES = 10000
MAX_TREES = 20000
FIXED_WEIGHT = 0.5
BOOTSTRAP_SEED = 20261016
EXPECTED_PORTFOLIO_SOURCE_SHA256 = "f0dde94e95479b0e0dae00d9d4b37b3437bcc7c8804f4dfd912cba2f78568a0d"
EXPECTED_V9_OOF_SHA256 = "2626c43410bc6c03c8cc3a91855bd9c8dbe3b117bb181903bf4afd6378a466d6"
EXPECTED_V9_SUBMISSION_SHA256 = "bc465ae7ff48deac5f93cd449a3799fee1a361a8021ec2ca03ff70baccc0f417"
REFERENCE_COLUMNS = ("MVT_ID_mvt", "target", "fold", "month",
                     "MVT_TIME_UTC_mvt", "a_valid", "selected", "old_v11_raw")
EXPERT_COLUMNS = ("MVT_ID_mvt", "full_expert", "prefix_10000_expert")
SPEC_PATH = Path(__file__).resolve().parent / "reports/long_budget_model_spec_v13.json"
EXPECTED_PARAMS = {**v11.EXPECTED_CATBOOST_PARAMS, "iterations": MAX_TREES}
PROMOTION_SPEC_PATH = Path(__file__).resolve().parent / "reports/long_budget_promotion_spec_v13.json"
EXPECTED_PROMOTION_SPEC_SHA256 = "bea09c4f944eaf66b48a6f4769bc052fd201ad3b587f6c151750694b92c9cf51"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_new(path: Path, value: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as output:
        json.dump(value, output, indent=2, allow_nan=False)
        output.write("\n")
    return sha256(path)


def write_parquet_new(path: Path, frame: pd.DataFrame) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".v13_", dir=path.parent) as work:
        staged = Path(work) / path.name
        frame.to_parquet(staged, index=False)
        digest = sha256(staged)
        os.link(staged, path)
    if sha256(path) != digest:
        raise ValueError("New V13 Parquet output changed after publication")
    return digest


def save_model_new(path: Path, model: CatBoostRegressor) -> str:
    """Publish a completed CatBoost model exclusively, without replacement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".v13_model_", dir=path.parent) as work:
        staged = Path(work) / path.name
        model.save_model(str(staged))
        digest = sha256(staged)
        os.link(staged, path)
    if sha256(path) != digest:
        raise ValueError("Published V13 model bytes changed")
    return digest


def exact_ids(actual: pd.Series, expected: pd.Series, context: str) -> None:
    left, right = pd.Index(actual), pd.Index(expected)
    if (len(left) != len(right) or left.has_duplicates or right.has_duplicates
            or left.isna().any() or right.isna().any()
            or not left.isin(right).all() or not right.isin(left).all()):
        raise ValueError(f"{context}: exact unique ID coverage failed")


def schema_of(features: pd.DataFrame) -> list[dict]:
    if (len(features.columns) != FEATURE_COUNT
            or features.columns.duplicated().any()
            or len(features.select_dtypes(include="category").columns)
               != CATEGORICAL_COUNT
            or any(name in features for name in
                   ("MVT_ID_mvt", "target", "BLOCK_TIME_UTC_mvt",
                    "TAXITIME_SEC_mvt"))):
        raise ValueError("V13 frozen 184-feature/24-category schema changed")
    return [{"name": str(name), "dtype": str(features[name].dtype)}
            for name in features]


def category_vocabulary_hashes(features: pd.DataFrame) -> dict[str, str]:
    result = {}
    for name in features.select_dtypes(include="category"):
        values = features[name].cat.categories.tolist()
        encoded = json.dumps({
            "ordered": bool(features[name].cat.ordered),
            "values": [(type(item).__name__, str(item)) for item in values]},
                             ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        result[str(name)] = hashlib.sha256(encoded).hexdigest()
    return result


def index_sha256(indices: np.ndarray) -> str:
    values = np.asarray(indices, dtype="<i8")
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def ordered_id_sha256(values: pd.Series) -> str:
    """Cryptographically bind the ordered movement IDs, including their dtype."""
    digest = hashlib.sha256()
    dtype = str(values.dtype).encode("utf-8")
    digest.update(len(dtype).to_bytes(4, "little"))
    digest.update(dtype)
    for value in values:
        if pd.isna(value):
            raise ValueError("Cannot seal a null movement ID")
        token = str(value).encode("utf-8")
        digest.update(len(token).to_bytes(4, "little"))
        digest.update(token)
    return digest.hexdigest()


def split_provenance(rows: pd.DataFrame,
                     split: dict[str, np.ndarray]) -> dict[str, str]:
    return {
        **{key + "_indices_sha256": index_sha256(indices)
           for key, indices in split.items()},
        **{key + "_ids_sha256": ordered_id_sha256(
            rows.MVT_ID_mvt.iloc[indices])
           for key, indices in split.items()},
    }


def fixed_replacement(base: np.ndarray, old_raw: np.ndarray,
                      new_raw: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Keep the locally selected 0.5 weight and all invalid rows exactly."""
    if (base.shape != old_raw.shape or base.shape != new_raw.shape
            or base.shape != valid.shape or valid.dtype != bool
            or not np.isfinite(base).all() or (base < 0).any()
            or not np.isfinite(old_raw[valid]).all()
            or not np.isfinite(new_raw[valid]).all()):
        raise ValueError("V13 replacement has nonfinite values or changed coverage")
    answer = base.copy()
    answer[valid] = np.maximum(
        base[valid] + FIXED_WEIGHT * (new_raw[valid] - old_raw[valid]), 0)
    if (not np.array_equal(answer[~valid], base[~valid])
            or not np.isfinite(answer).all() or (answer < 0).any()):
        raise ValueError("V13 changed a row outside the valid AOBT gate")
    return answer


def require_settings(args: argparse.Namespace) -> None:
    if (type(getattr(args, "iterations", None)) is not int
            or args.iterations != MAX_TREES
            or type(getattr(args, "depth", None)) is not int or args.depth != 10
            or type(getattr(args, "threads", None)) is not int or args.threads != 2
            or deep.params(args) != EXPECTED_PARAMS):
        raise ValueError("V13 changes only CatBoost's maximum to 20,000 trees")
    minimum = getattr(args, "min_free_gib", None)
    if (isinstance(minimum, bool)
            or not isinstance(minimum, (int, float, np.integer, np.floating))
            or not np.isfinite(minimum) or minimum < 10):
        raise ValueError("V13 requires a finite launch floor of at least 10 GiB")


def require_memory(args: argparse.Namespace) -> None:
    flow.require_memory(max(10.0, args.min_free_gib))


def protocol_spec() -> dict:
    return {
        "purpose": "One fixed 20,000-tree capacity comparison on the existing 184-feature v11 residual architecture",
        "reference": "locally sealed, unsubmitted v9 all-finite OOF policy and sealed v11 raw expert",
        "architecture": {
            "features": FEATURE_COUNT, "categorical_features": CATEGORICAL_COUNT,
            "target": "finite taxi target minus valid own AOBT proxy",
            "fit_target_range_sec": [0, 86400], "valid_proxy_range_sec": [0, 7200],
            "params": EXPECTED_PARAMS, "max_iterations": MAX_TREES,
            "only_change_from_v11": "CatBoost max iterations 10000 -> 20000",
            "internal_early_stop": "unchanged seed-2026 permutation; max(20000,6 percent) early rows; patience 200; use_best_model=True",
            "own_model_prefix": PREFIX_TREES,
            "memory_launch_floor_gib": 10,
        },
        "original_folds": {
            "selection_months": [1, 7], "forward_months": [11, 12],
            "candidate": "clip(v9 + 0.5*(new20k_raw-old_sealed_v11_raw),0) on valid proxy only",
            "prefix_diagnostic": "same formula with the same saved 20k model's exact ntree_end=10000 prediction",
            "gate": "each fold: actual trees >10000, all-finite v9-to-full RMSE improves and paired UTC-day CI lower >0; full beats own prefix with paired UTC-day CI lower >0",
            "bootstrap": {"seed": BOOTSTRAP_SEED, "repeats": 1000},
        },
        "fresh_audit": {
            "months": list(FRESH_MONTHS),
            "reference": "sealed matched v11 fresh_audit_predictions.v11_blend and fresh v11 raw expert",
            "formula": "clip(v11_blend + 0.5*(new20k_raw-old_v11_fresh_raw),0)",
            "gate": "actual trees >10000; both months improve against reference and own prefix; pooled paired-day lower gains >0 for both comparisons",
        },
        "after_fresh": "No final/ranking mode; separate frozen component guard and, if geometry succeeds, exact-composition compatibility required",
        "separate_promotion_protocol_sha256": EXPECTED_PROMOTION_SPEC_SHA256,
        "interpretation": "Repeated 2025 checks, not an untouched generalization estimate or evidence of a prize rank",
        "leaderboard_use": False,
    }


def v11_args(args: argparse.Namespace, *, fresh: bool = False) -> argparse.Namespace:
    copied = argparse.Namespace(**vars(args))
    copied.iterations = PREFIX_TREES
    copied.output_dir = args.v11_dir / "fresh_new" if fresh else args.v11_dir
    return copied


def source_inventory(args: argparse.Namespace) -> tuple[list[Path], dict[str, Path]]:
    """All own and inherited predictor inputs are checked before/after reads."""
    raw, paths = v11.source_inventory(v11_args(args))
    paths = {"v11_" + key: value for key, value in paths.items()}
    paths.update({
        "own_source": Path(__file__).resolve(),
        "own_spec": SPEC_PATH,
        "promotion_spec": PROMOTION_SPEC_PATH,
        "v11_source": Path(v11.__file__).resolve(),
        "v11_protocol": args.v11_dir / "protocol.json",
        "v11_validation": args.v11_dir / "validation.json",
        "v11_validation_predictions": args.v11_dir / "validation_predictions.parquet",
        "v11_fresh_report": args.v11_dir / "fresh_audit.json",
        "v11_fresh_predictions": args.v11_dir / "fresh_audit_predictions.parquet",
        "v11_fresh_raw": args.v11_dir / "fresh_new/fresh_apr_oct_oof.parquet",
        "v11_fresh_receipt": args.v11_dir / "fresh_new/fresh_apr_oct_provenance.json",
        "v11_fresh_model": args.v11_dir / "fresh_new/fresh_apr_oct.cbm",
        "v9_portfolio_source": Path(portfolio.__file__).resolve(),
        "v9_final_source": Path(final.__file__).resolve(),
        "v9_choice": args.portfolio_dir / "selection.json",
        "v9_portfolio_protocol": portfolio.PROTOCOL,
        "v9_evaluation": args.portfolio_dir / "evaluation_predictions.parquet",
        "v9_validation": args.portfolio_dir / "evaluation.json",
        "v9_guard_protocol": final.GUARD_DIR / "protocol.json",
        "v9_guard_terminal": final.GUARD_DIR / "terminal.json",
        "v9_guard_paired": final.GUARD_DIR / "paired_predictions.parquet",
        "v9_final_inputs_seal": final.OUT / "ranking_inputs.json",
        "v9_final_manifest": final.OUT / "ranking_manifest.json",
        "v9_final_predictions": final.OUT / "predictions.parquet",
        "v9_final_raw_expert": final.OUT / "ranking_expert.parquet",
        "v9_published_manifest": SPEC_PATH.parent / "submission_v9_finalized_manifest.json",
        "v9_sealed_submission": args.submission_v9,
    })
    for role in ("comparator", "replacement"):
        paths.update({
            f"v9_guard_{role}_model": final.GUARD_DIR / f"{role}.cbm",
            f"v9_guard_{role}_oof": final.GUARD_DIR / f"{role}_oof.parquet",
            f"v9_guard_{role}_fit": final.GUARD_DIR / f"{role}_fit.json",
            f"v9_guard_{role}_receipt": final.GUARD_DIR / f"{role}_receipt.json",
        })
    paths.update({"v9_rank_input_" + key: path
                  for key, path in final.ranking_input_paths("v11").items()})
    paths.update({"v9_final_source_" + key: path
                  for key, path in final.source_paths().items()})
    for name in FOLDS:
        paths.update({
            f"v11_{name}_raw": args.v11_dir / f"{name}_oof.parquet",
            f"v11_{name}_model": args.v11_dir / f"{name}.cbm",
            f"v11_{name}_report": args.v11_dir / f"{name}_validation.json",
            f"v11_{name}_receipt": args.v11_dir / f"{name}_provenance.json",
        })
    return raw, paths


def current_source_hashes(args: argparse.Namespace) -> tuple[dict[str, str],
                                                                dict[str, str]]:
    raw, paths = source_inventory(args)
    return ({name: sha256(path) for name, path in paths.items()},
            {path.name: sha256(path) for path in raw})


def assert_source_snapshot(args: argparse.Namespace, frozen: dict,
                           protocol_sha256: str) -> None:
    raw, paths = source_inventory(args)
    if (sha256(args.output_dir / "protocol.json") != protocol_sha256
            or json.loads((args.output_dir / "protocol.json")
                          .read_text(encoding="utf-8")) != frozen
            or {name: sha256(path) for name, path in paths.items()}
               != frozen["input_sha256"]
            or {path.name: sha256(path) for path in raw}
               != frozen["raw_training_sha256"]
            or sha256(args.output_dir / "frozen_v9_oof_reference.parquet")
               != frozen["frozen_reference_sha256"]):
        raise ValueError("V13 source, raw, cache, reference or protocol bytes changed")


def require_v9_choice(args: argparse.Namespace) -> dict:
    choice = portfolio.require_selection(
        output_dir=args.portfolio_dir,
        expected_source_sha256=EXPECTED_PORTFOLIO_SOURCE_SHA256)
    if (choice.get("selected_route") != "v11"
            or choice.get("selected_weight") != FIXED_WEIGHT
            or choice.get("evaluation_predictions_sha256") != EXPECTED_V9_OOF_SHA256
            or not final.require_guard(choice).get("passed")):
        raise ValueError("Locally sealed v9 selected-v11 policy changed")
    final.verify_ranking_inputs("v11")
    final_manifest = json.loads((final.OUT / "ranking_manifest.json")
                                .read_text(encoding="utf-8"))
    published = json.loads((SPEC_PATH.parent /
                            "submission_v9_finalized_manifest.json")
                           .read_text(encoding="utf-8"))
    if (final_manifest.get("predictions_sha256") !=
            EXPECTED_V9_SUBMISSION_SHA256
            or final_manifest.get("selected_route") != "v11"
            or final_manifest.get("selected_weight") != FIXED_WEIGHT
            or published.get("sha256") != EXPECTED_V9_SUBMISSION_SHA256
            or sha256(final.OUT / "predictions.parquet") !=
            EXPECTED_V9_SUBMISSION_SHA256
            or sha256(args.submission_v9) != EXPECTED_V9_SUBMISSION_SHA256):
        raise ValueError("Sealed local v9 ranking reference or receipt changed")
    return choice


def require_v11_reports(args: argparse.Namespace) -> tuple[dict, dict]:
    report = json.loads((args.v11_dir / "validation.json").read_text(encoding="utf-8"))
    fresh = json.loads((args.v11_dir / "fresh_audit.json").read_text(encoding="utf-8"))
    if (report.get("selected_weight") != FIXED_WEIGHT
            or not report.get("existing_folds_passed")
            or not report.get("fresh_audit_passed")
            or not fresh.get("passed") or fresh.get("weight") != FIXED_WEIGHT
            or fresh.get("months") != list(FRESH_MONTHS)
            or report.get("validation_predictions_sha256") !=
            sha256(args.v11_dir / "validation_predictions.parquet")
            or report.get("fresh_audit_sha256") !=
            sha256(args.v11_dir / "fresh_audit.json")
            or fresh.get("predictions_sha256") !=
            sha256(args.v11_dir / "fresh_audit_predictions.parquet")
            or fresh.get("v11_expert_oof_sha256") !=
            sha256(args.v11_dir / "fresh_new/fresh_apr_oct_oof.parquet")
            or fresh.get("fresh_fold_provenance_sha256") !=
            sha256(args.v11_dir / "fresh_new/fresh_apr_oct_provenance.json")):
        raise ValueError("Sealed v11 original/fresh reports or outputs changed")
    return report, fresh


def freeze_reference(args: argparse.Namespace) -> str:
    """Align sealed all-finite v9 policy and old raw v11 experts by exact ID."""
    require_v9_choice(args)
    require_v11_reports(args)
    path = args.portfolio_dir / "evaluation_predictions.parquet"
    if sha256(path) != EXPECTED_V9_OOF_SHA256:
        raise ValueError("Published v9 OOF bytes changed")
    base = pd.read_parquet(path, columns=list(REFERENCE_COLUMNS[:-1]))
    if (list(base) != list(REFERENCE_COLUMNS[:-1])
            or len(base) != EXPECTED_OOF_ROWS
            or base.MVT_ID_mvt.isna().any()
            or base.MVT_ID_mvt.duplicated().any()
            or set(base.fold.unique()) != set(FOLDS)
            or not np.isfinite(base[["target", "selected"]]
                               .to_numpy(dtype=float)).all()
            or (base.selected.to_numpy(dtype=float) < 0).any()):
        raise ValueError("Sealed v9 all-finite reference coverage is invalid")
    baseline = pd.read_parquet(args.cache_dir / "training_rows.parquet",
                               columns=["MVT_ID_mvt", "target", "proxy", "month",
                                        "time"])
    expected = baseline.loc[
        baseline.month.isin((1, 7, 11, 12)).to_numpy()
        & np.isfinite(baseline.target.to_numpy(dtype=float))]
    exact_ids(base.MVT_ID_mvt, expected.MVT_ID_mvt, "v9 all-finite OOF")
    checked = expected.merge(base, on="MVT_ID_mvt", how="left", sort=False,
                             validate="one_to_one", suffixes=("_baseline", ""))
    proxy = checked.proxy.to_numpy(dtype=float)
    valid = np.isfinite(proxy) & (proxy >= 0) & (proxy <= 7200)
    if (not np.array_equal(checked.target_baseline.to_numpy(dtype=float),
                           checked.target.to_numpy(dtype=float))
            or not np.array_equal(checked.month_baseline.to_numpy(),
                                  checked.month.to_numpy())
            or not np.array_equal(checked.a_valid.to_numpy(dtype=bool), valid)
            or not np.array_equal(
                pd.to_datetime(checked.time, utc=True).to_numpy(),
                pd.to_datetime(checked.MVT_TIME_UTC_mvt, utc=True).to_numpy())):
        raise ValueError("Sealed v9 labels, proxy gate or UTC time differ from baseline")
    old_ref = v11.load_reference(v11_args(args))
    pieces = []
    for name, months in FOLDS.items():
        held = base.loc[base.fold.eq(name)]
        if not held.month.isin(months).all():
            raise ValueError("Sealed v9 month-to-fold mapping changed")
        v11.verify_fold_provenance(v11_args(args), name,
                                   old_ref.loc[old_ref.fold.eq(name)])
        old = pd.read_parquet(args.v11_dir / f"{name}_oof.parquet")
        if list(old) != ["MVT_ID_mvt", "expert"]:
            raise ValueError("Sealed v11 raw expert schema changed")
        exact_ids(old.MVT_ID_mvt, held.loc[held.a_valid, "MVT_ID_mvt"],
                  f"{name} sealed v11 raw")
        renamed = old.rename(columns={"expert": "old_v11_raw"})
        subset = held.merge(renamed, on="MVT_ID_mvt", how="left", sort=False,
                            validate="one_to_one")
        if (not np.array_equal(subset.old_v11_raw.notna().to_numpy(),
                               held.a_valid.to_numpy(dtype=bool))
                or not np.isfinite(subset.loc[subset.a_valid, "old_v11_raw"]
                                   .to_numpy(dtype=float)).all()):
            raise ValueError("Sealed v11 raw has wrong valid-proxy coverage")
        pieces.append(subset)
    frozen = pd.concat(pieces, ignore_index=True)[list(REFERENCE_COLUMNS)]
    output = args.output_dir / "frozen_v9_oof_reference.parquet"
    if output.exists():
        if not pd.read_parquet(output).equals(frozen):
            raise ValueError("Previously frozen v9/v11 aligned reference differs")
    else:
        write_parquet_new(output, frozen)
    return sha256(output)


def prepare(args: argparse.Namespace) -> dict:
    require_settings(args)
    require_memory(args)
    if (not SPEC_PATH.exists()
            or json.loads(SPEC_PATH.read_text(encoding="utf-8")) != protocol_spec()):
        raise ValueError("Published V13 model spec differs from frozen source")
    if sha256(PROMOTION_SPEC_PATH) != EXPECTED_PROMOTION_SPEC_SHA256:
        raise ValueError("Published prospective V13 promotion protocol changed")
    if not (args.v11_dir / "protocol.json").exists():
        raise ValueError("Sealed v11 protocol is absent")
    before_inputs, before_raw = current_source_hashes(args)
    v11.protocol(v11_args(args))
    reference_sha = freeze_reference(args)
    after_inputs, after_raw = current_source_hashes(args)
    if before_inputs != after_inputs or before_raw != after_raw:
        raise ValueError("V13 source, raw or ranking input changed during prepare reads")
    source_receipt = json.loads((args.v11_dir / "seasonal_jan_jul_provenance.json")
                                .read_text(encoding="utf-8"))
    schema = source_receipt.get("feature_schema")
    if (not isinstance(schema, list) or len(schema) != FEATURE_COUNT
            or sum(item.get("dtype") == "category" for item in schema)
               != CATEGORICAL_COUNT
            or source_receipt.get("catboost_params") !=
               v11.EXPECTED_CATBOOST_PARAMS):
        raise ValueError("Sealed v11 feature schema or 10k model params changed")
    for name in FOLDS:
        other = json.loads((args.v11_dir / f"{name}_provenance.json")
                           .read_text(encoding="utf-8"))
        if other.get("feature_schema") != schema:
            raise ValueError("Sealed v11 original folds have different schemas")
    value = {
        "spec": protocol_spec(),
        "spec_sha256": sha256(SPEC_PATH),
        "frozen_reference_sha256": reference_sha,
        "feature_schema": schema,
        "input_sha256": before_inputs,
        "raw_training_sha256": before_raw,
    }
    target = args.output_dir / "protocol.json"
    if target.exists():
        if json.loads(target.read_text(encoding="utf-8")) != value:
            raise ValueError("Frozen V13 protocol or source hashes changed")
    else:
        write_json_new(target, value)
    assert_source_snapshot(args, value, sha256(target))
    return value


def require_protocol(args: argparse.Namespace) -> tuple[dict, str]:
    require_settings(args)
    path = args.output_dir / "protocol.json"
    frozen = json.loads(path.read_text(encoding="utf-8"))
    if (frozen.get("spec") != protocol_spec()
            or frozen.get("spec_sha256") != sha256(SPEC_PATH)
            or frozen.get("feature_schema") is None):
        raise ValueError("Frozen V13 protocol/spec/feature schema changed")
    digest = sha256(path)
    assert_source_snapshot(args, frozen, digest)
    require_v9_choice(args)
    require_v11_reports(args)
    return frozen, digest


def load_features(args: argparse.Namespace, frozen: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    rows, features = v11.load_features(v11_args(args))
    schema = schema_of(features)
    if schema != frozen["feature_schema"]:
        raise ValueError("V13 predictors differ from sealed 184-feature v11 schema")
    vocab = category_vocabulary_hashes(features)
    if len(vocab) != CATEGORICAL_COUNT:
        raise ValueError("V13 categorical vocabularies are incomplete")
    return rows, features, vocab


def split_indices(rows: pd.DataFrame, months: tuple[int, int]) -> dict[str, np.ndarray]:
    """Exact deep_timestamp_expert.fit_fold masks and seed-2026 split."""
    y = rows.target.to_numpy(dtype=float)
    proxy = rows.proxy.to_numpy(dtype=float)
    valid = np.isfinite(proxy) & (proxy >= 0) & (proxy <= 7200)
    held = rows.month.isin(months).to_numpy()
    train = np.flatnonzero(~held & valid & np.isfinite(y)
                           & (y >= 0) & (y <= 86400))
    test = np.flatnonzero(held & valid & np.isfinite(y))
    rng = np.random.default_rng(2026)
    order = rng.permutation(train)
    n_early = max(20000, int(.06 * len(order)))
    if n_early >= len(order) or not len(test):
        raise ValueError("V13 fit/early/test masks are unexpectedly empty")
    return {"fit": order[n_early:], "early": order[:n_early],
            "test": test, "held_all": np.flatnonzero(held & np.isfinite(y))}


def assert_test_alignment(rows: pd.DataFrame, test: np.ndarray,
                          expected: pd.DataFrame, name: str) -> None:
    """Pair baseline-order model rows to reference-order metadata by unique ID."""
    actual = rows.iloc[test][["MVT_ID_mvt", "target", "time"]]
    exact_ids(actual.MVT_ID_mvt, expected.MVT_ID_mvt, f"{name} held-out")
    aligned = actual.merge(expected, on="MVT_ID_mvt", how="left", sort=False,
                           validate="one_to_one", suffixes=("_baseline", ""))
    if (not np.array_equal(aligned.target_baseline.to_numpy(dtype=float),
                           aligned.target.to_numpy(dtype=float))
            or not np.array_equal(
                pd.to_datetime(aligned.time, utc=True).to_numpy(),
                pd.to_datetime(aligned.MVT_TIME_UTC_mvt, utc=True).to_numpy())):
        raise ValueError(f"{name} held-out ID, target or UTC time differs")


def verified_saved_model(path: Path, schema: list[dict],
                         expected_trees: int) -> CatBoostRegressor:
    model = CatBoostRegressor()
    model.load_model(str(path))
    names = [item["name"] for item in schema]
    categorical = [index for index, item in enumerate(schema)
                   if item["dtype"] == "category"]
    if (int(model.tree_count_) != expected_trees
            or expected_trees < 1 or expected_trees > MAX_TREES
            or list(model.feature_names_) != names
            or list(model.get_cat_feature_indices()) != categorical):
        raise ValueError("Saved V13 model tree, feature or category metadata differ")
    trained = model.get_all_params()
    exact = ("task_type", "loss_function", "eval_metric", "depth",
             "random_seed", "max_ctr_complexity", "one_hot_max_size",
             "border_count")
    numeric = ("learning_rate", "l2_leaf_reg", "random_strength",
               "bagging_temperature")
    if any(trained.get(key) != EXPECTED_PARAMS[key] for key in exact):
        raise ValueError("Saved V13 CatBoost core settings differ")
    for key in numeric:
        try:
            actual = float(trained[key])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Saved V13 CatBoost lacks {key}") from error
        if not np.isfinite(actual) or not np.isclose(
                actual, EXPECTED_PARAMS[key], rtol=0, atol=1e-6):
            raise ValueError(f"Saved V13 CatBoost changed {key}")
    return model


def paths_for_fold(args: argparse.Namespace, name: str) -> dict[str, Path]:
    root = args.output_dir / "fresh_new" if name == "fresh_apr_oct" else args.output_dir
    return {"model": root / f"{name}.cbm",
            "oof": root / f"{name}_oof.parquet",
            "fit": root / f"{name}_fit.json",
            "receipt": root / f"{name}_provenance.json"}


def fit_one(args: argparse.Namespace, name: str,
            rows: pd.DataFrame, features: pd.DataFrame,
            category_hashes: dict[str, str], expected: pd.DataFrame,
            frozen: dict, protocol_sha: str) -> dict:
    """Train one frozen fold without a weight grid or held-out metric lookup."""
    months = FRESH_MONTHS if name == "fresh_apr_oct" else FOLDS[name]
    paths = paths_for_fold(args, name)
    if any(path.exists() for path in paths.values()):
        raise FileExistsError(f"V13 {name} fit artifacts already exist")
    split = split_indices(rows, months)
    test = split["test"]
    assert_test_alignment(rows, test, expected, name)
    y = rows.target.to_numpy(dtype=float)
    proxy = rows.proxy.to_numpy(dtype=float)
    labels = y - proxy
    categories = features.select_dtypes(include="category").columns.tolist()
    if len(categories) != CATEGORICAL_COUNT:
        raise ValueError("V13 categorical feature count changed before fit")
    require_memory(args)
    assert_source_snapshot(args, frozen, protocol_sha)
    model = CatBoostRegressor(**deep.params(args))
    fit_pool = Pool(features.iloc[split["fit"]], label=labels[split["fit"]],
                    cat_features=categories)
    early_pool = Pool(features.iloc[split["early"]], label=labels[split["early"]],
                      cat_features=categories)
    started = time.monotonic()
    model.fit(fit_pool, eval_set=early_pool, early_stopping_rounds=200,
              use_best_model=True)
    elapsed = time.monotonic() - started
    del fit_pool, early_pool
    gc.collect()
    assert_source_snapshot(args, frozen, protocol_sha)
    model_sha = save_model_new(paths["model"], model)
    trained = verified_saved_model(paths["model"], frozen["feature_schema"],
                                   int(model.tree_count_))
    full = proxy[test] + trained.predict(features.iloc[test],
                                         thread_count=args.threads)
    if int(trained.tree_count_) > PREFIX_TREES:
        prefix = proxy[test] + trained.predict(
            features.iloc[test], ntree_end=PREFIX_TREES,
            thread_count=args.threads)
        exact_prefix = True
    else:
        # A capped run with <=10k saved trees cannot substantiate added capacity.
        # The absent exact prefix is recorded as NaN, never substituted by a fit.
        prefix = np.full(len(test), np.nan, dtype=float)
        exact_prefix = False
    if (not np.isfinite(full).all()
            or (exact_prefix and not np.isfinite(prefix).all())):
        raise ValueError("Saved V13 model gave a nonfinite held-out prediction")
    oof = pd.DataFrame({"MVT_ID_mvt": rows.MVT_ID_mvt.iloc[test].to_numpy(),
                        "full_expert": full,
                        "prefix_10000_expert": prefix})
    oof_sha = write_parquet_new(paths["oof"], oof)
    split_hashes = split_provenance(rows, split)
    report = {
        "fold": name, "heldout_months": list(months),
        "requested_iterations": MAX_TREES, "saved_trees": int(trained.tree_count_),
        "prefix_trees": PREFIX_TREES if exact_prefix else None,
        "exact_prefix_available": exact_prefix,
        "early_stopping_rounds": 200, "use_best_model": True,
        "fit_seconds": float(elapsed),
        "fit_rows": len(split["fit"]), "early_rows": len(split["early"]),
        "heldout_valid_rows": len(test),
        "heldout_all_finite_rows": len(split["held_all"]),
        **split_hashes,
        "model_sha256": model_sha,
        "oof_sha256": oof_sha,
    }
    write_json_new(paths["fit"], report)
    assert_source_snapshot(args, frozen, protocol_sha)
    receipt = {
        "schema_version": 1, "fold": name,
        "heldout_months": list(months),
        "heldout_scope": "valid_aobt_finite" if name == "fresh_apr_oct"
                           else "all_finite_targets",
        "protocol_sha256": protocol_sha,
        "model_params_requested": EXPECTED_PARAMS,
        "model_params_effective": trained.get_all_params(),
        "early_stopping_rounds": 200, "use_best_model": True,
        "feature_schema": frozen["feature_schema"],
        "categorical_feature_indices": list(trained.get_cat_feature_indices()),
        "categorical_vocabulary_sha256": category_hashes,
        "fit_rows": len(split["fit"]), "early_rows": len(split["early"]),
        "heldout_valid_rows": len(test),
        "heldout_all_finite_rows": len(split["held_all"]),
        **split_hashes,
        "saved_trees": int(trained.tree_count_),
        "prefix_trees": PREFIX_TREES if exact_prefix else None,
        "exact_prefix_available": exact_prefix,
        "model_sha256": model_sha,
        "oof_sha256": oof_sha,
        "fit_report_sha256": sha256(paths["fit"]),
    }
    write_json_new(paths["receipt"], receipt)
    assert_source_snapshot(args, frozen, protocol_sha)
    return receipt


def verify_fold_receipt(args: argparse.Namespace, name: str,
                        rows: pd.DataFrame, category_hashes: dict[str, str],
                        expected: pd.DataFrame, frozen: dict,
                        protocol_sha: str,
                        features: pd.DataFrame) -> dict:
    months = FRESH_MONTHS if name == "fresh_apr_oct" else FOLDS[name]
    paths = paths_for_fold(args, name)
    receipt = json.loads(paths["receipt"].read_text(encoding="utf-8"))
    report = json.loads(paths["fit"].read_text(encoding="utf-8"))
    split = split_indices(rows, months)
    test = split["test"]
    split_hashes = split_provenance(rows, split)
    assert_test_alignment(rows, test, expected, name)
    if (receipt.get("schema_version") != 1
            or receipt.get("fold") != name
            or receipt.get("heldout_months") != list(months)
            or receipt.get("heldout_scope") !=
               ("valid_aobt_finite" if name == "fresh_apr_oct"
                else "all_finite_targets")
            or receipt.get("protocol_sha256") != protocol_sha
            or receipt.get("model_params_requested") != EXPECTED_PARAMS
            or receipt.get("feature_schema") != frozen["feature_schema"]
            or receipt.get("categorical_vocabulary_sha256") != category_hashes
            or receipt.get("early_stopping_rounds") != 200
            or receipt.get("use_best_model") is not True
            or any(receipt.get(key) != value for key, value in split_hashes.items())
            or receipt.get("fit_rows") != len(split["fit"])
            or receipt.get("early_rows") != len(split["early"])
            or receipt.get("heldout_valid_rows") != len(test)
            or receipt.get("heldout_all_finite_rows") != len(split["held_all"])
            or receipt.get("model_sha256") != sha256(paths["model"])
            or receipt.get("oof_sha256") != sha256(paths["oof"])
            or receipt.get("fit_report_sha256") != sha256(paths["fit"])):
        raise ValueError(f"V13 {name} fold receipt or exact split changed")
    saved_trees = receipt.get("saved_trees")
    model = verified_saved_model(paths["model"], frozen["feature_schema"],
                                 saved_trees)
    if receipt.get("model_params_effective") != model.get_all_params():
        raise ValueError(f"V13 {name} saved effective model params changed")
    exact_prefix = saved_trees > PREFIX_TREES
    categories = [index for index, item in enumerate(frozen["feature_schema"])
                  if item["dtype"] == "category"]
    if (receipt.get("categorical_feature_indices") != categories
            or list(model.get_cat_feature_indices()) != categories
            or receipt.get("exact_prefix_available") is not exact_prefix
            or receipt.get("prefix_trees") !=
               (PREFIX_TREES if exact_prefix else None)
            or report.get("fold") != name
            or report.get("heldout_months") != list(months)
            or report.get("requested_iterations") != MAX_TREES
            or report.get("saved_trees") != saved_trees
            or report.get("exact_prefix_available") is not exact_prefix
            or report.get("prefix_trees") != receipt.get("prefix_trees")
            or report.get("early_stopping_rounds") != 200
            or report.get("use_best_model") is not True
            or report.get("model_sha256") != receipt["model_sha256"]
            or report.get("oof_sha256") != receipt["oof_sha256"]
            or report.get("fit_rows") != len(split["fit"])
            or report.get("early_rows") != len(split["early"])
            or report.get("heldout_valid_rows") != len(test)
            or report.get("heldout_all_finite_rows") != len(split["held_all"])
            or any(report.get(key) != value for key, value in split_hashes.items())):
        raise ValueError(f"V13 {name} model/fit metadata changed")
    oof = pd.read_parquet(paths["oof"])
    if list(oof) != list(EXPERT_COLUMNS):
        raise ValueError(f"V13 {name} OOF schema changed")
    exact_ids(oof.MVT_ID_mvt, expected.MVT_ID_mvt, f"V13 {name} OOF")
    if (not np.isfinite(oof.full_expert.to_numpy(dtype=float)).all()
            or (exact_prefix and not np.isfinite(
                oof.prefix_10000_expert.to_numpy(dtype=float)).all())
            or (not exact_prefix and not oof.prefix_10000_expert.isna().all())):
        raise ValueError(f"V13 {name} prefix/full predictions are invalid")
    proxy = rows.proxy.to_numpy(dtype=float)[test]
    full_from_model = proxy + model.predict(features.iloc[test],
                                            thread_count=args.threads)
    if not np.allclose(oof.full_expert.to_numpy(dtype=float), full_from_model,
                       rtol=0, atol=1e-6):
        raise ValueError(f"V13 {name} OOF differs from saved full model")
    if exact_prefix:
        prefix_from_model = proxy + model.predict(
            features.iloc[test], ntree_end=PREFIX_TREES,
            thread_count=args.threads)
        if not np.allclose(oof.prefix_10000_expert.to_numpy(dtype=float),
                           prefix_from_model, rtol=0, atol=1e-6):
            raise ValueError(f"V13 {name} OOF differs from own saved prefix")
    return receipt


def fit_fold(args: argparse.Namespace, name: str) -> None:
    if name not in FOLDS:
        raise ValueError("Only the two original V13 folds can use fit-fold")
    frozen, protocol_sha = require_protocol(args)
    require_memory(args)
    rows, features, category_hashes = load_features(args, frozen)
    ref = pd.read_parquet(args.output_dir / "frozen_v9_oof_reference.parquet")
    expected = ref.loc[ref.fold.eq(name) & ref.a_valid,
                       ["MVT_ID_mvt", "target", "MVT_TIME_UTC_mvt"]]
    fit_one(args, name, rows, features, category_hashes, expected,
            frozen, protocol_sha)
    del rows, features, ref
    gc.collect()


def evaluate(args: argparse.Namespace, *, write: bool = True) -> dict:
    """Replay both original gates; no candidate weight or budget search."""
    frozen, protocol_sha = require_protocol(args)
    require_memory(args)
    rows, features, category_hashes = load_features(args, frozen)
    ref = pd.read_parquet(args.output_dir / "frozen_v9_oof_reference.parquet")
    if (list(ref) != list(REFERENCE_COLUMNS) or len(ref) != EXPECTED_OOF_ROWS
            or ref.MVT_ID_mvt.isna().any() or ref.MVT_ID_mvt.duplicated().any()):
        raise ValueError("Frozen v9/v11 reference changed before V13 evaluation")
    report = {
        "protocol_sha256": protocol_sha,
        "frozen_reference_sha256": frozen["frozen_reference_sha256"],
        "fixed_weight": FIXED_WEIGHT,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_repeats": 1000,
        "folds": {}, "existing_folds_passed": False,
        "fresh_audit_passed": False, "ranking_authorized": False,
    }
    pieces = []
    for name in FOLDS:
        held = ref.loc[ref.fold.eq(name)]
        expected = held.loc[held.a_valid,
                            ["MVT_ID_mvt", "target", "MVT_TIME_UTC_mvt"]]
        receipt = verify_fold_receipt(args, name, rows, category_hashes,
                                      expected, frozen, protocol_sha,
                                      features)
        paths = paths_for_fold(args, name)
        raw = pd.read_parquet(paths["oof"])
        part = held.merge(raw, on="MVT_ID_mvt", how="left", sort=False,
                          validate="one_to_one")
        valid = part.a_valid.to_numpy(dtype=bool)
        if (not np.array_equal(part.full_expert.notna().to_numpy(), valid)
                or not np.isfinite(part.old_v11_raw.to_numpy(dtype=float)[valid]).all()):
            raise ValueError(f"V13 {name} raw expert coverage differs from proxy gate")
        base = part.selected.to_numpy(dtype=float)
        old = part.old_v11_raw.to_numpy(dtype=float)
        full = part.full_expert.to_numpy(dtype=float)
        candidate = fixed_replacement(base, old, full, valid)
        y = part.target.to_numpy(dtype=float)
        comparison = arrival.bootstrap(part, base, candidate,
                                       seed=BOOTSTRAP_SEED)
        exact_prefix = receipt["exact_prefix_available"]
        if exact_prefix:
            prefix = fixed_replacement(
                base, old, part.prefix_10000_expert.to_numpy(dtype=float), valid)
            prefix_comparison = arrival.bootstrap(part, prefix, candidate,
                                                  seed=BOOTSTRAP_SEED)
            prefix_rmse = deep.rmse(y, prefix)
            prefix_passed = bool(prefix_rmse > deep.rmse(y, candidate)
                                 and prefix_comparison["gain_ci95_sec"][0] > 0)
        else:
            prefix = np.full(len(part), np.nan, dtype=float)
            prefix_comparison = None
            prefix_rmse = None
            prefix_passed = False
        base_rmse = deep.rmse(y, base)
        full_rmse = deep.rmse(y, candidate)
        v9_passed = bool(full_rmse < base_rmse
                         and comparison["gain_ci95_sec"][0] > 0)
        fold_passed = bool(receipt["saved_trees"] > PREFIX_TREES
                           and v9_passed and prefix_passed)
        report["folds"][name] = {
            "rows_all_finite": len(part), "rows_valid_aobt": int(valid.sum()),
            "saved_trees": receipt["saved_trees"],
            "v9_rmse": base_rmse, "candidate_rmse": full_rmse,
            "prefix_candidate_rmse": prefix_rmse,
            "v9_to_candidate_day_bootstrap": comparison,
            "prefix_to_full_day_bootstrap": prefix_comparison,
            "v9_gate_passed": v9_passed,
            "own_prefix_gate_passed": prefix_passed,
            "passed": fold_passed,
            "expert_oof_sha256": receipt["oof_sha256"],
            "model_sha256": receipt["model_sha256"],
            "fit_report_sha256": receipt["fit_report_sha256"],
            "fold_provenance_sha256": sha256(paths["receipt"]),
        }
        part["candidate"] = candidate
        part["prefix_candidate"] = prefix
        pieces.append(part)
    report["existing_folds_passed"] = bool(
        all(info["passed"] for info in report["folds"].values()))
    combined = pd.concat(pieces, ignore_index=True)
    combined = ref[["MVT_ID_mvt"]].merge(
        combined, on="MVT_ID_mvt", how="left", sort=False,
        validate="one_to_one")
    exact_ids(combined.MVT_ID_mvt, ref.MVT_ID_mvt, "V13 original all-finite")
    if (not np.array_equal(combined.MVT_ID_mvt.to_numpy(),
                           ref.MVT_ID_mvt.to_numpy())
            or not np.array_equal(combined.target.to_numpy(dtype=float),
                                  ref.target.to_numpy(dtype=float))
            or not np.array_equal(combined.a_valid.to_numpy(dtype=bool),
                                  ref.a_valid.to_numpy(dtype=bool))
            or not np.array_equal(combined.candidate.to_numpy(dtype=float)
                                   [~combined.a_valid.to_numpy(dtype=bool)],
                                  combined.selected.to_numpy(dtype=float)
                                   [~combined.a_valid.to_numpy(dtype=bool)])):
        raise ValueError("V13 original OOF alignment or outside-gate preservation failed")
    ordered = [*REFERENCE_COLUMNS, "full_expert", "prefix_10000_expert",
               "candidate", "prefix_candidate"]
    combined = combined[ordered]
    output_path = args.output_dir / "validation_predictions.parquet"
    report_path = args.output_dir / "validation.json"
    assert_source_snapshot(args, frozen, protocol_sha)
    if write:
        if output_path.exists() or report_path.exists():
            raise FileExistsError("V13 original terminal outputs already exist")
        report["validation_predictions_sha256"] = write_parquet_new(
            output_path, combined)
        write_json_new(report_path, report)
    else:
        report["validation_predictions_sha256"] = sha256(output_path)
        if (not pd.read_parquet(output_path).equals(combined)
                or json.loads(report_path.read_text(encoding="utf-8")) != report):
            raise ValueError("V13 saved original scores, gates or OOF changed")
    assert_source_snapshot(args, frozen, protocol_sha)
    del rows, features, ref, pieces, combined
    gc.collect()
    return report


def fresh_reference(args: argparse.Namespace, rows: pd.DataFrame) -> pd.DataFrame:
    """Verify v11's matched 4/10 predictions against original labels and raw OOF."""
    _, fresh_report = require_v11_reports(args)
    path = args.v11_dir / "fresh_audit_predictions.parquet"
    paired = pd.read_parquet(path)
    needed = ("MVT_ID_mvt", "target", "month", "MVT_TIME_UTC_mvt",
              "fold", "expert", "v11_blend")
    if (len(paired) != EXPECTED_FRESH_ROWS
            or any(name not in paired for name in needed)
            or not paired.fold.eq("fresh_apr_oct").all()
            or not paired.month.isin(FRESH_MONTHS).all()
            or paired.MVT_ID_mvt.isna().any()
            or paired.MVT_ID_mvt.duplicated().any()
            or not np.isfinite(paired[["target", "expert", "v11_blend"]]
                               .to_numpy(dtype=float)).all()
            or (paired.v11_blend.to_numpy(dtype=float) < 0).any()
            or fresh_report["predictions_sha256"] != sha256(path)):
        raise ValueError("Sealed v11 fresh paired OOF coverage changed")
    v11.verify_fold_provenance(v11_args(args, fresh=True), "fresh_apr_oct",
                               paired)
    raw_path = args.v11_dir / "fresh_new/fresh_apr_oct_oof.parquet"
    old = pd.read_parquet(raw_path)
    if list(old) != ["MVT_ID_mvt", "expert"]:
        raise ValueError("Sealed v11 fresh raw OOF schema changed")
    exact_ids(old.MVT_ID_mvt, paired.MVT_ID_mvt, "v11 fresh raw")
    old_aligned = paired[["MVT_ID_mvt", "expert"]].merge(
        old, on="MVT_ID_mvt", how="left", sort=False, validate="one_to_one",
        suffixes=("_paired", "_oof"))
    if not np.array_equal(old_aligned.expert_paired.to_numpy(dtype=float),
                          old_aligned.expert_oof.to_numpy(dtype=float)):
        raise ValueError("v11 matched fresh raw expert differs from saved OOF")
    proxy = rows.proxy.to_numpy(dtype=float)
    y = rows.target.to_numpy(dtype=float)
    expected = rows.loc[
        rows.month.isin(FRESH_MONTHS).to_numpy()
        & np.isfinite(proxy) & (proxy >= 0) & (proxy <= 7200)
        & np.isfinite(y),
        ["MVT_ID_mvt", "target", "month", "time"]]
    exact_ids(paired.MVT_ID_mvt, expected.MVT_ID_mvt, "v11 fresh source labels")
    aligned = expected.merge(paired, on="MVT_ID_mvt", how="left", sort=False,
                             validate="one_to_one", suffixes=("_baseline", ""))
    if (not np.array_equal(aligned.target_baseline.to_numpy(dtype=float),
                           aligned.target.to_numpy(dtype=float))
            or not np.array_equal(aligned.month_baseline.to_numpy(),
                                  aligned.month.to_numpy())
            or not np.array_equal(pd.to_datetime(aligned.time, utc=True).to_numpy(),
                                  pd.to_datetime(aligned.MVT_TIME_UTC_mvt,
                                                 utc=True).to_numpy())):
        raise ValueError("v11 fresh source IDs, targets or UTC times differ")
    return paired


def fresh_audit(args: argparse.Namespace) -> dict:
    """One predeclared matched 4/10 fit, conditional on both original gates."""
    prior = evaluate(args, write=False)
    if not prior["existing_folds_passed"]:
        raise ValueError("V13 original capacity gates failed; no fresh fit allowed")
    frozen, protocol_sha = require_protocol(args)
    require_memory(args)
    if ((args.output_dir / "fresh_audit.json").exists()
            or (args.output_dir / "fresh_audit_predictions.parquet").exists()):
        raise FileExistsError("V13 fresh terminal outputs already exist")
    rows, features, category_hashes = load_features(args, frozen)
    old = fresh_reference(args, rows)
    expected = old[["MVT_ID_mvt", "target", "MVT_TIME_UTC_mvt"]]
    fit_one(args, "fresh_apr_oct", rows, features, category_hashes,
            expected, frozen, protocol_sha)
    receipt = verify_fold_receipt(args, "fresh_apr_oct", rows,
                                  category_hashes, expected, frozen,
                                  protocol_sha, features)
    paths = paths_for_fold(args, "fresh_apr_oct")
    raw = pd.read_parquet(paths["oof"])
    paired = old.merge(raw, on="MVT_ID_mvt", how="left", sort=False,
                       validate="one_to_one")
    if (len(paired) != EXPECTED_FRESH_ROWS
            or not np.array_equal(paired.MVT_ID_mvt.to_numpy(),
                                  old.MVT_ID_mvt.to_numpy())
            or not np.isfinite(paired.full_expert.to_numpy(dtype=float)).all()):
        raise ValueError("V13 fresh full prediction coverage changed")
    y = paired.target.to_numpy(dtype=float)
    base = paired.v11_blend.to_numpy(dtype=float)
    old_raw = paired.expert.to_numpy(dtype=float)
    valid = np.ones(len(paired), dtype=bool)
    full_candidate = fixed_replacement(
        base, old_raw, paired.full_expert.to_numpy(dtype=float), valid)
    exact_prefix = receipt["exact_prefix_available"]
    if exact_prefix:
        prefix_candidate = fixed_replacement(
            base, old_raw,
            paired.prefix_10000_expert.to_numpy(dtype=float), valid)
        own_prefix_bootstrap = arrival.bootstrap(
            paired, prefix_candidate, full_candidate, seed=BOOTSTRAP_SEED)
    else:
        prefix_candidate = np.full(len(paired), np.nan, dtype=float)
        own_prefix_bootstrap = None
    baseline_bootstrap = arrival.bootstrap(
        paired, base, full_candidate, seed=BOOTSTRAP_SEED)
    month = paired.month.to_numpy(dtype=int)
    monthly = {}
    for value in FRESH_MONTHS:
        mask = month == value
        monthly[str(value)] = {
            "rows": int(mask.sum()),
            "v11_blend_rmse": deep.rmse(y[mask], base[mask]),
            "candidate_rmse": deep.rmse(y[mask], full_candidate[mask]),
            "prefix_candidate_rmse": (deep.rmse(y[mask],
                                                prefix_candidate[mask])
                                      if exact_prefix else None),
        }
    baseline_passed = bool(all(
        monthly[str(value)]["candidate_rmse"]
        < monthly[str(value)]["v11_blend_rmse"] for value in FRESH_MONTHS)
        and baseline_bootstrap["gain_ci95_sec"][0] > 0)
    prefix_passed = bool(exact_prefix and all(
        monthly[str(value)]["candidate_rmse"]
        < monthly[str(value)]["prefix_candidate_rmse"]
        for value in FRESH_MONTHS)
        and own_prefix_bootstrap["gain_ci95_sec"][0] > 0)
    passed = bool(receipt["saved_trees"] > PREFIX_TREES
                  and baseline_passed and prefix_passed)
    output = paired[["MVT_ID_mvt", "target", "month",
                     "MVT_TIME_UTC_mvt"]].copy()
    output["old_v11_blend"] = base
    output["old_v11_raw"] = old_raw
    output["full_expert"] = paired.full_expert.to_numpy(dtype=float)
    output["prefix_10000_expert"] = paired.prefix_10000_expert.to_numpy(dtype=float)
    output["candidate"] = full_candidate
    output["prefix_candidate"] = prefix_candidate
    assert_source_snapshot(args, frozen, protocol_sha)
    output_sha = write_parquet_new(
        args.output_dir / "fresh_audit_predictions.parquet", output)
    report = {
        "protocol_sha256": protocol_sha,
        "original_validation_sha256": sha256(args.output_dir / "validation.json"),
        "original_predictions_sha256": sha256(
            args.output_dir / "validation_predictions.parquet"),
        "months": list(FRESH_MONTHS), "fixed_weight": FIXED_WEIGHT,
        "rows_valid_aobt_finite": len(output),
        "coverage_verified": True,
        "saved_trees": receipt["saved_trees"],
        "monthly_rmse": monthly,
        "v11_to_candidate_day_bootstrap": baseline_bootstrap,
        "prefix_to_full_day_bootstrap": own_prefix_bootstrap,
        "v11_gate_passed": baseline_passed,
        "own_prefix_gate_passed": prefix_passed,
        "passed": passed,
        "v11_fresh_report_sha256": sha256(args.v11_dir / "fresh_audit.json"),
        "v11_fresh_predictions_sha256": sha256(
            args.v11_dir / "fresh_audit_predictions.parquet"),
        "v11_fresh_raw_sha256": sha256(
            args.v11_dir / "fresh_new/fresh_apr_oct_oof.parquet"),
        "expert_oof_sha256": receipt["oof_sha256"],
        "model_sha256": receipt["model_sha256"],
        "fit_report_sha256": receipt["fit_report_sha256"],
        "fold_provenance_sha256": sha256(paths["receipt"]),
        "predictions_sha256": output_sha,
        "ranking_authorized": False,
    }
    write_json_new(args.output_dir / "fresh_audit.json", report)
    assert_source_snapshot(args, frozen, protocol_sha)
    del rows, features, old, paired, output
    gc.collect()
    return report


def verify_fresh(args: argparse.Namespace) -> dict:
    """Replay the immutable fresh component result without fitting or writing."""
    original = evaluate(args, write=False)
    if not original["existing_folds_passed"]:
        raise ValueError("V13 original gates failed; fresh output cannot be terminal")
    frozen, protocol_sha = require_protocol(args)
    require_memory(args)
    rows, features, category_hashes = load_features(args, frozen)
    prior = fresh_reference(args, rows)
    expected = prior[["MVT_ID_mvt", "target", "MVT_TIME_UTC_mvt"]]
    receipt = verify_fold_receipt(args, "fresh_apr_oct", rows,
                                  category_hashes, expected, frozen,
                                  protocol_sha, features)
    paths = paths_for_fold(args, "fresh_apr_oct")
    raw = pd.read_parquet(paths["oof"])
    aligned = prior.merge(raw, on="MVT_ID_mvt", how="left", sort=False,
                          validate="one_to_one")
    output_path = args.output_dir / "fresh_audit_predictions.parquet"
    output = pd.read_parquet(output_path)
    columns = ("MVT_ID_mvt", "target", "month", "MVT_TIME_UTC_mvt",
               "old_v11_blend", "old_v11_raw", "full_expert",
               "prefix_10000_expert", "candidate", "prefix_candidate")
    if (list(output) != list(columns)
            or len(output) != EXPECTED_FRESH_ROWS
            or not np.array_equal(output.MVT_ID_mvt.to_numpy(),
                                  prior.MVT_ID_mvt.to_numpy())
            or not np.array_equal(output.target.to_numpy(dtype=float),
                                  prior.target.to_numpy(dtype=float))
            or not np.array_equal(output.month.to_numpy(),
                                  prior.month.to_numpy())
            or not np.array_equal(
                pd.to_datetime(output.MVT_TIME_UTC_mvt, utc=True).to_numpy(),
                pd.to_datetime(prior.MVT_TIME_UTC_mvt, utc=True).to_numpy())
            or not np.array_equal(output.old_v11_blend.to_numpy(dtype=float),
                                  prior.v11_blend.to_numpy(dtype=float))
            or not np.array_equal(output.old_v11_raw.to_numpy(dtype=float),
                                  prior.expert.to_numpy(dtype=float))
            or not np.array_equal(output.full_expert.to_numpy(dtype=float),
                                  aligned.full_expert.to_numpy(dtype=float))):
        raise ValueError("V13 fresh paired IDs, labels, time or raw expert changed")
    exact_prefix = receipt["exact_prefix_available"]
    if exact_prefix:
        if not np.array_equal(output.prefix_10000_expert.to_numpy(dtype=float),
                              aligned.prefix_10000_expert.to_numpy(dtype=float)):
            raise ValueError("V13 fresh exact own-prefix raw prediction changed")
    elif not output.prefix_10000_expert.isna().all():
        raise ValueError("V13 ineligible fresh fold has a substituted prefix")
    y = prior.target.to_numpy(dtype=float)
    base = prior.v11_blend.to_numpy(dtype=float)
    old_raw = prior.expert.to_numpy(dtype=float)
    valid = np.ones(len(prior), dtype=bool)
    candidate = fixed_replacement(
        base, old_raw, aligned.full_expert.to_numpy(dtype=float), valid)
    if not np.array_equal(output.candidate.to_numpy(dtype=float), candidate):
        raise ValueError("V13 fresh fixed full formula changed")
    if exact_prefix:
        prefix = fixed_replacement(
            base, old_raw,
            aligned.prefix_10000_expert.to_numpy(dtype=float), valid)
        if not np.array_equal(output.prefix_candidate.to_numpy(dtype=float),
                              prefix):
            raise ValueError("V13 fresh own-prefix formula changed")
        prefix_bootstrap = arrival.bootstrap(prior, prefix, candidate,
                                             seed=BOOTSTRAP_SEED)
    else:
        prefix = np.full(len(prior), np.nan, dtype=float)
        if not output.prefix_candidate.isna().all():
            raise ValueError("V13 ineligible fresh fold has a prefix candidate")
        prefix_bootstrap = None
    baseline_bootstrap = arrival.bootstrap(prior, base, candidate,
                                           seed=BOOTSTRAP_SEED)
    month = prior.month.to_numpy(dtype=int)
    monthly = {}
    for value in FRESH_MONTHS:
        mask = month == value
        monthly[str(value)] = {
            "rows": int(mask.sum()),
            "v11_blend_rmse": deep.rmse(y[mask], base[mask]),
            "candidate_rmse": deep.rmse(y[mask], candidate[mask]),
            "prefix_candidate_rmse": (deep.rmse(y[mask], prefix[mask])
                                      if exact_prefix else None),
        }
    base_pass = bool(all(
        monthly[str(value)]["candidate_rmse"]
        < monthly[str(value)]["v11_blend_rmse"] for value in FRESH_MONTHS)
        and baseline_bootstrap["gain_ci95_sec"][0] > 0)
    prefix_pass = bool(exact_prefix and all(
        monthly[str(value)]["candidate_rmse"]
        < monthly[str(value)]["prefix_candidate_rmse"]
        for value in FRESH_MONTHS)
        and prefix_bootstrap["gain_ci95_sec"][0] > 0)
    value = {
        "protocol_sha256": protocol_sha,
        "original_validation_sha256": sha256(args.output_dir / "validation.json"),
        "original_predictions_sha256": sha256(
            args.output_dir / "validation_predictions.parquet"),
        "months": list(FRESH_MONTHS), "fixed_weight": FIXED_WEIGHT,
        "rows_valid_aobt_finite": len(output),
        "coverage_verified": True,
        "saved_trees": receipt["saved_trees"],
        "monthly_rmse": monthly,
        "v11_to_candidate_day_bootstrap": baseline_bootstrap,
        "prefix_to_full_day_bootstrap": prefix_bootstrap,
        "v11_gate_passed": base_pass,
        "own_prefix_gate_passed": prefix_pass,
        "passed": bool(receipt["saved_trees"] > PREFIX_TREES
                       and base_pass and prefix_pass),
        "v11_fresh_report_sha256": sha256(args.v11_dir / "fresh_audit.json"),
        "v11_fresh_predictions_sha256": sha256(
            args.v11_dir / "fresh_audit_predictions.parquet"),
        "v11_fresh_raw_sha256": sha256(
            args.v11_dir / "fresh_new/fresh_apr_oct_oof.parquet"),
        "expert_oof_sha256": receipt["oof_sha256"],
        "model_sha256": receipt["model_sha256"],
        "fit_report_sha256": receipt["fit_report_sha256"],
        "fold_provenance_sha256": sha256(paths["receipt"]),
        "predictions_sha256": sha256(output_path),
        "ranking_authorized": False,
    }
    if json.loads((args.output_dir / "fresh_audit.json")
                  .read_text(encoding="utf-8")) != value:
        raise ValueError("V13 saved fresh report, monthly scores or day CIs changed")
    assert_source_snapshot(args, frozen, protocol_sha)
    del rows, features, prior, aligned, output
    gc.collect()
    return value


def fit_final(_: argparse.Namespace) -> None:
    raise RuntimeError(
        "V13 final fit requires separate terminal component, compatibility and final protocols")


def final_predict(_: argparse.Namespace) -> None:
    raise RuntimeError(
        "V13 ranking prediction requires separate terminal component, compatibility and final protocols")


def synthetic_check() -> dict:
    """Tiny formula/coverage and frozen-interface checks; no model or data load."""
    base = np.array([10.0, 20.0, 30.0, 40.0])
    old = np.array([12.0, np.nan, 30.0, np.nan])
    new = np.array([16.0, np.nan, 26.0, np.nan])
    valid = np.array([True, False, True, False])
    result = fixed_replacement(base, old, new, valid)
    if not np.array_equal(result, np.array([12.0, 20.0, 28.0, 40.0])):
        raise AssertionError("Frozen 0.5 formula differs on synthetic rows")
    try:
        fixed_replacement(base, old, new, ~valid)
    except ValueError:
        pass
    else:
        raise AssertionError("Wrong eligibility mask was accepted")
    exact_ids(pd.Series(["one", "two"]), pd.Series(["two", "one"]),
              "synthetic")
    if ordered_id_sha256(pd.Series(["one", "two"])) == ordered_id_sha256(
            pd.Series(["two", "one"])):
        raise AssertionError("Ordered movement-ID hash ignored order")
    try:
        exact_ids(pd.Series(["one", "one"]), pd.Series(["one", "two"]),
                  "synthetic")
    except ValueError:
        pass
    else:
        raise AssertionError("Duplicate movement ID was accepted")
    fixture_rows = pd.DataFrame({
        "MVT_ID_mvt": [10, 20], "target": [11.0, 22.0],
        "time": pd.to_datetime(["2025-01-01", "2025-01-02"], utc=True)})
    fixture_expected = pd.DataFrame({
        "MVT_ID_mvt": [20, 10], "target": [22.0, 11.0],
        "MVT_TIME_UTC_mvt": pd.to_datetime(
            ["2025-01-02", "2025-01-01"], utc=True)})
    assert_test_alignment(fixture_rows, np.array([0, 1]), fixture_expected,
                          "synthetic permuted reference")
    fixture_expected.loc[0, "target"] = 23.0
    try:
        assert_test_alignment(fixture_rows, np.array([0, 1]), fixture_expected,
                              "synthetic altered target")
    except ValueError:
        pass
    else:
        raise AssertionError("Altered held-out target was accepted")
    namespace = argparse.Namespace(iterations=MAX_TREES, depth=10, threads=2,
                                   min_free_gib=10.0)
    require_settings(namespace)
    namespace.iterations = PREFIX_TREES
    try:
        require_settings(namespace)
    except ValueError:
        pass
    else:
        raise AssertionError("Old iteration cap was accepted as V13")
    return {"formula": "pass", "outside_gate_preserved": True,
            "exact_ids": "pass", "permuted_reference_alignment": "pass",
            "frozen_params_reject_mutation": True,
            "competition_rows_read": 0, "models_fitted": 0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("show-spec", "self-test", "prepare",
                                           "fit-fold", "evaluate", "fresh-audit",
                                           "verify-fresh",
                                           "fit-final", "final-predict"),
                        default="show-spec")
    parser.add_argument("--fold", choices=tuple(FOLDS),
                        default="seasonal_jan_jul")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--cache-dir", type=Path,
                        default=Path("artifacts/baseline"))
    parser.add_argument("--weather-file", type=Path,
                        default=Path("data/external/weather.parquet"))
    parser.add_argument("--arrival-dir", type=Path,
                        default=Path("artifacts/v5-arrival-clean"))
    parser.add_argument("--neighbour-dir", type=Path,
                        default=Path("artifacts/v6-neighbour"))
    parser.add_argument("--runway-dir", type=Path,
                        default=Path("artifacts/v6-runway-arrival"))
    parser.add_argument("--taxi-dir", type=Path,
                        default=Path("artifacts/v11-taxi-flow"))
    parser.add_argument("--v7-dir", type=Path,
                        default=Path("artifacts/v7-runway-traffic"))
    parser.add_argument("--v6-dir", type=Path,
                        default=Path("artifacts/v6-deep-arrival"))
    parser.add_argument("--v11-dir", type=Path,
                        default=Path("artifacts/v11-taxi-flow-expert"))
    parser.add_argument("--portfolio-dir", type=Path,
                        default=Path("artifacts/later-feature-portfolio"))
    parser.add_argument("--submission-v9", type=Path,
                        default=Path("submissions/merry-mushroom_v9.parquet"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("artifacts/v13-long-budget"))
    parser.add_argument("--min-free-gib", type=float, default=10.0)
    args = parser.parse_args()
    args.iterations, args.depth, args.threads = MAX_TREES, 10, 2
    if args.mode == "show-spec":
        print(json.dumps(protocol_spec(), indent=2))
    elif args.mode == "self-test":
        print(json.dumps(synthetic_check(), indent=2))
    elif args.mode == "prepare":
        frozen = prepare(args)
        print(json.dumps({"protocol": str(args.output_dir / "protocol.json"),
                          "feature_count": len(frozen["feature_schema"])}))
    elif args.mode == "fit-fold":
        fit_fold(args, args.fold)
    elif args.mode == "evaluate":
        print(json.dumps(evaluate(args), indent=2))
    elif args.mode == "fresh-audit":
        print(json.dumps(fresh_audit(args), indent=2))
    elif args.mode == "verify-fresh":
        print(json.dumps(verify_fresh(args), indent=2))
    elif args.mode == "fit-final":
        fit_final(args)
    else:
        final_predict(args)


if __name__ == "__main__":
    main()
