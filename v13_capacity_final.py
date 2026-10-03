"""Prospectively guarded v13 compatibility, component, and local final path.

The 184-field 20k expert, fixed 0.5 component formula, geometry branch rule,
and validation months come from the published promotion spec. No upload,
finalizer, alternate weight, or leaderboard feedback is implemented here.
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
import later_feature_final as v9_final
import later_reserved_guard as old_guard
import taxi_interval_flow_features as flow
import v11_taxi_interval_flow_expert as v11
import v12_geometry_final as geometry_guard
import v12_runway_geometry_expert as v12
import v13_long_budget_expert as v13


ROOT = Path(__file__).resolve().parent
SPEC = ROOT / "reports/long_budget_promotion_spec_v13.json"
SPEC_SHA256 = "bea09c4f944eaf66b48a6f4769bc052fd201ad3b587f6c151750694b92c9cf51"
V13_DIR = ROOT / "artifacts/v13-long-budget"
V12_DIR = ROOT / "artifacts/v12-runway-geometry-expert"
COMPAT_DIR = ROOT / "artifacts/v13-capacity-compatibility"
GUARD_DIR = ROOT / "artifacts/v13-capacity-feb-aug-guard"
FINAL_DIR = ROOT / "artifacts/v13-capacity-final"
V9_SUBMISSION = ROOT / "submissions/merry-mushroom_v9.parquet"
V9_SHA256 = "bc465ae7ff48deac5f93cd449a3799fee1a361a8021ec2ca03ff70baccc0f417"
TEMPLATE = ROOT / "data/submitting.parquet"
HELDOUT = (2, 8)
FOLDS = ("seasonal_jan_jul", "forward_nov_dec")
WEIGHT = 0.5
PREFIX_TREES = 10000
MAX_TREES = 20000
BOOTSTRAP_SEED = 20261016
BOOTSTRAP_REPEATS = 1000
TRAIN_ROWS = 2_085_047
RANK_ROWS = 344_841
VALID_RANK_ROWS = 339_377
FORBIDDEN = {"MVT_ID_mvt", "target", "BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt"}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    before = sha(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if sha(path) != before or not isinstance(value, dict):
        raise ValueError(f"JSON changed while reading: {path}")
    return value


def write_json_new(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as output:
        json.dump(value, output, indent=2, sort_keys=True, allow_nan=False)
        output.write("\n")


def publish_new(path: Path, writer) -> str:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".v13_capacity_", dir=path.parent) as directory:
        stage = Path(directory) / path.name
        writer(stage)
        expected = sha(stage)
        os.link(stage, path)
    if sha(path) != expected:
        raise ValueError(f"Published bytes changed: {path}")
    return expected


def hashes(paths: dict[str, Path]) -> dict[str, str]:
    if len(paths) != len(set(paths)):
        raise ValueError("Duplicate inventory key")
    return {name: sha(path) for name, path in sorted(paths.items())}


def check_hashes(paths: dict[str, Path], expected: dict[str, str], label: str) -> None:
    if hashes(paths) != expected:
        raise ValueError(f"{label} input/source bytes changed")


def path_map(paths: dict[str, Path]) -> dict[str, str]:
    result = {}
    for name, path in paths.items():
        try:
            result[name] = path.resolve().relative_to(ROOT).as_posix()
        except ValueError as error:
            raise ValueError(f"Input escapes competition workspace: {path}") from error
    return result


def paths_from_map(saved: dict[str, str]) -> dict[str, Path]:
    if not isinstance(saved, dict) or not saved:
        raise ValueError("Frozen source path map is absent")
    paths = {name: ROOT / relative for name, relative in saved.items()}
    if path_map(paths) != saved:
        raise ValueError("Frozen input path map escaped the workspace")
    return paths


def unique_paths(paths: dict[str, Path]) -> dict[str, Path]:
    """Keep one stable receipt key per physical byte stream."""
    unique = {}
    seen: set[Path] = set()
    for name, path in paths.items():
        resolved = path.resolve()
        if resolved not in seen:
            unique[name] = path
            seen.add(resolved)
    return unique


def published_source(expected_sha256: str) -> None:
    if (not isinstance(expected_sha256, str) or len(expected_sha256) != 64
            or expected_sha256 != sha(Path(__file__).resolve())):
        raise ValueError("Pass the exact published SHA256 of this source")
    if sha(SPEC) != SPEC_SHA256:
        raise ValueError("Published v13 promotion spec changed")


def require_memory() -> None:
    flow.require_memory(10.0)


def args_v13() -> argparse.Namespace:
    """Only the published v13 CLI defaults; no caller-selected data paths."""
    return argparse.Namespace(
        data_dir=ROOT / "data", cache_dir=ROOT / "artifacts/baseline",
        weather_file=ROOT / "data/external/weather.parquet",
        arrival_dir=ROOT / "artifacts/v5-arrival-clean",
        neighbour_dir=ROOT / "artifacts/v6-neighbour",
        runway_dir=ROOT / "artifacts/v6-runway-arrival",
        taxi_dir=ROOT / "artifacts/v11-taxi-flow",
        v7_dir=ROOT / "artifacts/v7-runway-traffic",
        v6_dir=ROOT / "artifacts/v6-deep-arrival",
        v11_dir=ROOT / "artifacts/v11-taxi-flow-expert",
        portfolio_dir=ROOT / "artifacts/later-feature-portfolio",
        output_dir=V13_DIR, min_free_gib=10.0,
        iterations=MAX_TREES, depth=10, threads=2,
    )


def args_v12() -> argparse.Namespace:
    return geometry_guard.args_for_v12()


def exact_ids(actual: pd.Series, expected: pd.Series, label: str) -> None:
    v13.exact_ids(actual, expected, label)


def schema(features: pd.DataFrame) -> list[dict]:
    if features.columns.duplicated().any() or FORBIDDEN.intersection(features.columns):
        raise ValueError("Feature matrix contains duplicate or forbidden target fields")
    return [{"name": str(name), "dtype": str(features[name].dtype)} for name in features]


def branch_from_status(original: bool | None, fresh: bool | None,
                       reserved: bool | None) -> str:
    """Verified failure retains v9; absent required terminal never counts as failure."""
    if any(item is not None and type(item) is not bool
           for item in (original, fresh, reserved)):
        raise ValueError("Geometry gate status must be an explicit Boolean")
    if original is None:
        raise ValueError("V12 original gate has no terminal result")
    if original is False:
        return "v9"
    if fresh is None:
        raise ValueError("V12 fresh gate has no terminal result")
    if fresh is False:
        return "v9"
    if reserved is None:
        raise ValueError("V12 February/August gate has no terminal result")
    return "geometry" if reserved is True else "v9"


def resolve_geometry_terminal() -> tuple[str, dict, dict[str, Path]]:
    """A verified failure chooses v9; a missing required report is unresolved."""
    args = args_v12()
    original_path = V12_DIR / "validation.json"
    if not original_path.is_file():
        raise FileNotFoundError("V12 original terminal is unresolved")
    original = v12.evaluate(args)
    paths: dict[str, Path] = {
        "v12_source": Path(v12.__file__).resolve(),
        "v12_spec": v12.SPEC_PATH,
        "v12_protocol": V12_DIR / "protocol.json",
        "v12_original_validation": original_path,
        "v12_original_predictions": V12_DIR / "validation_predictions.parquet",
    }
    for name in FOLDS:
        for suffix in (".cbm", "_oof.parquet", "_validation.json", "_provenance.json"):
            paths[f"v12_{name}{suffix}"] = V12_DIR / f"{name}{suffix}"
    status = {"original_passed": original["existing_folds_passed"],
              "fresh_passed": None, "feb_aug_passed": None}
    if original["existing_folds_passed"] is not True:
        return branch_from_status(False, None, None), status, paths
    fresh_path = V12_DIR / "fresh_audit.json"
    if not fresh_path.is_file():
        raise FileNotFoundError("V12 April/October terminal is unresolved")
    fresh = v12.verify_fresh(args)
    status["fresh_passed"] = fresh["passed"]
    paths.update({
        "v12_fresh_report": fresh_path,
        "v12_fresh_predictions": V12_DIR / "fresh_audit_predictions.parquet",
        "v12_fresh_model": V12_DIR / "fresh_new/fresh_apr_oct.cbm",
        "v12_fresh_oof": V12_DIR / "fresh_new/fresh_apr_oct_oof.parquet",
        "v12_fresh_fit": V12_DIR / "fresh_new/fresh_apr_oct_validation.json",
        "v12_fresh_receipt": V12_DIR / "fresh_new/fresh_apr_oct_provenance.json",
    })
    if fresh["passed"] is not True:
        return branch_from_status(True, False, None), status, paths
    guard_terminal = geometry_guard.GUARD_DIR / "terminal.json"
    if not guard_terminal.is_file():
        raise FileNotFoundError("V12 February/August terminal is unresolved")
    terminal = geometry_guard.verify_guard_terminal(sha(Path(geometry_guard.__file__)))
    status["feb_aug_passed"] = terminal["passed"]
    paths.update({
        "v12_guard_source": Path(geometry_guard.__file__).resolve(),
        "v12_guard_protocol": geometry_guard.GUARD_DIR / "protocol.json",
        "v12_guard_terminal": guard_terminal,
        "v12_guard_paired": geometry_guard.GUARD_DIR / "paired_predictions.parquet",
    })
    for name in ("comparator", "replacement"):
        paths.update({f"v12_guard_{name}_{key}": path
                      for key, path in geometry_guard.artifact_paths(name).items()})
    return branch_from_status(True, True, terminal["passed"]), status, paths


def require_capacity_original_fresh() -> tuple[dict, dict, dict]:
    args = args_v13()
    frozen, protocol_sha = v13.require_protocol(args)
    original = v13.evaluate(args, write=False)
    fresh = v13.verify_fresh(args)
    if (original.get("existing_folds_passed") is not True
            or fresh.get("passed") is not True
            or fresh.get("fixed_weight") != WEIGHT
            or fresh.get("original_validation_sha256") != sha(V13_DIR / "validation.json")):
        raise ValueError("V13 original or April/October capacity gate failed")
    v13.assert_source_snapshot(args, frozen, protocol_sha)
    return frozen, original, fresh


def compatibility_source_paths(geometry_status: str,
                               geometry_paths: dict[str, Path]) -> dict[str, Path]:
    args = args_v13()
    raw, inherited = v13.source_inventory(args)
    geometry_raw, geometry_inherited = v12.source_inventory(args_v12())
    paths = {f"v13_input_{name}": Path(path) for name, path in inherited.items()}
    paths.update({f"raw_{path.name}": path for path in raw})
    paths.update({f"v12_input_{name}": Path(path)
                  for name, path in geometry_inherited.items()})
    paths.update({f"v12_raw_{path.name}": path for path in geometry_raw})
    paths.update({
        "own_source": Path(__file__).resolve(),
        "promotion_spec": SPEC,
        "v13_source": Path(v13.__file__).resolve(),
        "v13_spec": v13.SPEC_PATH,
        "v13_protocol": V13_DIR / "protocol.json",
        "v13_original_validation": V13_DIR / "validation.json",
        "v13_original_predictions": V13_DIR / "validation_predictions.parquet",
        "v13_fresh_validation": V13_DIR / "fresh_audit.json",
        "v13_fresh_predictions": V13_DIR / "fresh_audit_predictions.parquet",
        "v13_fresh_model": V13_DIR / "fresh_new/fresh_apr_oct.cbm",
        "v13_fresh_oof": V13_DIR / "fresh_new/fresh_apr_oct_oof.parquet",
        "v13_fresh_fit": V13_DIR / "fresh_new/fresh_apr_oct_fit.json",
        "v13_fresh_receipt": V13_DIR / "fresh_new/fresh_apr_oct_provenance.json",
        "local_v9_submission": V9_SUBMISSION,
    })
    for name in FOLDS:
        for suffix in (".cbm", "_oof.parquet", "_fit.json", "_provenance.json"):
            paths[f"v13_{name}{suffix}"] = V13_DIR / f"{name}{suffix}"
    paths.update({f"geometry_{name}": path for name, path in geometry_paths.items()})
    if geometry_status not in ("v9", "geometry"):
        raise ValueError("Geometry terminal branch is unresolved")
    return unique_paths(paths)


def compatibility_protocol_path() -> Path:
    return COMPAT_DIR / "protocol.json"


def freeze_compatibility(expected_source_sha256: str) -> dict:
    published_source(expected_source_sha256)
    if compatibility_protocol_path().exists():
        return check_compatibility_frozen(expected_source_sha256)
    branch, status, geometry_paths = resolve_geometry_terminal()
    frozen, original, fresh = require_capacity_original_fresh()
    if sha(V9_SUBMISSION) != V9_SHA256:
        raise ValueError("Sealed local v9 submission changed")
    paths = compatibility_source_paths(branch, geometry_paths)
    before = hashes(paths)
    value = {
        "schema_version": 1,
        "promotion_spec_sha256": SPEC_SHA256,
        "published_source_sha256": expected_source_sha256,
        "v13_protocol_sha256": sha(V13_DIR / "protocol.json"),
        "v13_original_validation_sha256": sha(V13_DIR / "validation.json"),
        "v13_fresh_audit_sha256": sha(V13_DIR / "fresh_audit.json"),
        "v13_original_passed": original["existing_folds_passed"],
        "v13_fresh_passed": fresh["passed"],
        "geometry_branch": branch,
        "geometry_terminal_status": status,
        "fixed_weight": WEIGHT,
        "source_input_paths": path_map(paths),
        "source_input_sha256": before,
        "compatibility_labels_read_to_freeze": False,
    }
    v13.assert_source_snapshot(args_v13(), frozen, value["v13_protocol_sha256"])
    resolved_branch, resolved_status, _ = resolve_geometry_terminal()
    if (branch != resolved_branch or status != resolved_status
            or before != hashes(paths)):
        raise ValueError("Compatibility source or geometry terminal changed during freeze")
    write_json_new(compatibility_protocol_path(), value)
    return check_compatibility_frozen(expected_source_sha256)


def check_compatibility_frozen(expected_source_sha256: str) -> dict:
    published_source(expected_source_sha256)
    frozen = read_json(compatibility_protocol_path())
    branch, status, geometry_paths = resolve_geometry_terminal()
    source_frozen, original, fresh = require_capacity_original_fresh()
    if (frozen.get("schema_version") != 1
            or frozen.get("promotion_spec_sha256") != SPEC_SHA256
            or frozen.get("published_source_sha256") != expected_source_sha256
            or frozen.get("v13_protocol_sha256") != sha(V13_DIR / "protocol.json")
            or frozen.get("v13_original_validation_sha256") != sha(V13_DIR / "validation.json")
            or frozen.get("v13_fresh_audit_sha256") != sha(V13_DIR / "fresh_audit.json")
            or frozen.get("v13_original_passed") is not original["existing_folds_passed"]
            or frozen.get("v13_fresh_passed") is not fresh["passed"]
            or frozen.get("geometry_branch") != branch
            or frozen.get("geometry_terminal_status") != status
            or frozen.get("fixed_weight") != WEIGHT
            or frozen.get("compatibility_labels_read_to_freeze") is not False
            or sha(V9_SUBMISSION) != V9_SHA256):
        raise ValueError("Frozen v13 promotion branch, fixed weight or source changed")
    current_paths = compatibility_source_paths(branch, geometry_paths)
    if frozen.get("source_input_paths") != path_map(current_paths):
        raise ValueError("Compatibility input path map changed")
    check_hashes(current_paths, frozen["source_input_sha256"], "Compatibility")
    v13.assert_source_snapshot(args_v13(), source_frozen,
                               frozen["v13_protocol_sha256"])
    return frozen


def rehash_compatibility_sources(frozen: dict, protocol_sha256: str) -> None:
    """Byte-only recheck while paired predictions are resident in memory."""
    if (frozen.get("published_source_sha256") != sha(Path(__file__).resolve())
            or sha(SPEC) != SPEC_SHA256
            or sha(compatibility_protocol_path()) != protocol_sha256):
        raise ValueError("Published compatibility source or protocol changed")
    check_hashes(paths_from_map(frozen["source_input_paths"]),
                 frozen["source_input_sha256"], "Compatibility")


def aligned_frame(left: pd.DataFrame, right: pd.DataFrame,
                  label: str) -> pd.DataFrame:
    exact_ids(right.MVT_ID_mvt, left.MVT_ID_mvt, label)
    return right.set_index("MVT_ID_mvt").loc[left.MVT_ID_mvt.to_numpy()].reset_index()


def compatibility_pairs(branch: str) -> pd.DataFrame:
    """Construct exact paired rows only after the source/branch seal exists."""
    cap_original = pd.read_parquet(V13_DIR / "validation_predictions.parquet")
    cap_fresh = pd.read_parquet(V13_DIR / "fresh_audit_predictions.parquet")
    if (list(cap_original) != [*v13.REFERENCE_COLUMNS, "full_expert",
                               "prefix_10000_expert", "candidate", "prefix_candidate"]
            or len(cap_original) != v13.EXPECTED_OOF_ROWS
            or list(cap_fresh) != ["MVT_ID_mvt", "target", "month",
                                   "MVT_TIME_UTC_mvt", "old_v11_blend", "old_v11_raw",
                                   "full_expert", "prefix_10000_expert", "candidate",
                                   "prefix_candidate"]
            or len(cap_fresh) != v13.EXPECTED_FRESH_ROWS):
        raise ValueError("V13 original/fresh prediction schema or coverage changed")
    if branch == "geometry":
        geom_original = pd.read_parquet(V12_DIR / "validation_predictions.parquet")
        geom_fresh = pd.read_parquet(V12_DIR / "fresh_audit_predictions.parquet")
        geom_original = aligned_frame(cap_original, geom_original,
                                      "Original v13 versus geometry")
        geom_fresh = aligned_frame(cap_fresh, geom_fresh,
                                   "Fresh v13 versus geometry")
        original_fields = ("target", "month", "fold", "MVT_TIME_UTC_mvt", "a_valid")
        fresh_fields = ("target", "month", "MVT_TIME_UTC_mvt")
        for name in original_fields:
            left = cap_original[name].to_numpy()
            right = geom_original[name].to_numpy()
            if name == "MVT_TIME_UTC_mvt":
                left = pd.to_datetime(left, utc=True).to_numpy()
                right = pd.to_datetime(right, utc=True).to_numpy()
            if not np.array_equal(left, right):
                raise ValueError(f"Original geometry target/time/mask mismatch: {name}")
        for name in fresh_fields:
            left = cap_fresh[name].to_numpy()
            right = geom_fresh[name].to_numpy()
            if name == "MVT_TIME_UTC_mvt":
                left = pd.to_datetime(left, utc=True).to_numpy()
                right = pd.to_datetime(right, utc=True).to_numpy()
            if not np.array_equal(left, right):
                raise ValueError(f"Fresh geometry target/time mismatch: {name}")
        if (not np.array_equal(cap_original.selected.to_numpy(dtype=float),
                               geom_original.selected.to_numpy(dtype=float))
                or not np.array_equal(cap_fresh.old_v11_blend.to_numpy(dtype=float),
                                      geom_fresh.selected.to_numpy(dtype=float))):
            raise ValueError("Geometry and v13 use different sealed v9/v11 baselines")
        old_original = geom_original.candidate.to_numpy(dtype=float)
        old_fresh = geom_fresh.v12_blend.to_numpy(dtype=float)
    elif branch == "v9":
        old_original = cap_original.selected.to_numpy(dtype=float)
        old_fresh = cap_fresh.old_v11_blend.to_numpy(dtype=float)
    else:
        raise ValueError("Unresolved v12 promotion branch")
    valid = cap_original.a_valid.to_numpy(dtype=bool)
    cap_original_values = cap_original.candidate.to_numpy(dtype=float)
    if (not np.array_equal(cap_original_values[~valid],
                           cap_original.selected.to_numpy(dtype=float)[~valid])
            or not np.isfinite(cap_original_values).all()
            or not np.isfinite(old_original).all()
            or not np.isfinite(old_fresh).all()
            or not np.isfinite(cap_fresh.candidate.to_numpy(dtype=float)).all()):
        raise ValueError("Original invalid-proxy policy or comparison values differ")
    first = cap_original[["MVT_ID_mvt", "target", "month", "MVT_TIME_UTC_mvt"]].copy()
    first["stage"] = cap_original.fold.to_numpy(copy=True)
    first["a_valid"] = valid
    first["reference"] = old_original
    first["candidate"] = cap_original_values
    second = cap_fresh[["MVT_ID_mvt", "target", "month", "MVT_TIME_UTC_mvt"]].copy()
    second["stage"] = "fresh_apr_oct"
    second["a_valid"] = True
    second["reference"] = old_fresh
    second["candidate"] = cap_fresh.candidate.to_numpy(dtype=float)
    result = pd.concat([first, second], ignore_index=True)
    exact_ids(first.MVT_ID_mvt, cap_original.MVT_ID_mvt, "Original all-finite")
    exact_ids(second.MVT_ID_mvt, cap_fresh.MVT_ID_mvt, "Fresh valid-AOBT")
    if result.MVT_ID_mvt.isna().any() or result.MVT_ID_mvt.duplicated().any():
        raise ValueError("Original and fresh IDs unexpectedly overlap")
    return result[["MVT_ID_mvt", "target", "month", "MVT_TIME_UTC_mvt",
                   "stage", "a_valid", "reference", "candidate"]]


def score_compatibility(frame: pd.DataFrame) -> dict:
    required = ["MVT_ID_mvt", "target", "month", "MVT_TIME_UTC_mvt",
                "stage", "a_valid", "reference", "candidate"]
    if (list(frame) != required or frame.MVT_ID_mvt.isna().any()
            or frame.MVT_ID_mvt.duplicated().any()
            or len(frame) != v13.EXPECTED_OOF_ROWS + v13.EXPECTED_FRESH_ROWS
            or set(frame.stage.unique()) != {*FOLDS, "fresh_apr_oct"}
            or not np.isfinite(frame[["target", "reference", "candidate"]]
                               .to_numpy(dtype=float)).all()
            or (frame[["reference", "candidate"]].to_numpy(dtype=float) < 0).any()):
        raise ValueError("Compatibility paired frame is incomplete or nonfinite")
    times = pd.to_datetime(frame.MVT_TIME_UTC_mvt, utc=True, errors="coerce")
    if times.isna().any() or not np.array_equal(times.dt.month.to_numpy(),
                                                frame.month.to_numpy()):
        raise ValueError("Compatibility month/time provenance differs")
    scores = {}
    for name in FOLDS:
        part = frame.loc[frame.stage.eq(name)]
        if (not len(part) or set(part.month.unique()) != set(deep.FOLDS[name])):
            raise ValueError(f"{name} paired fold is incomplete")
        y = part.target.to_numpy(dtype=float)
        base = part.reference.to_numpy(dtype=float)
        new = part.candidate.to_numpy(dtype=float)
        bootstrap = arrival.bootstrap(part, base, new, seed=BOOTSTRAP_SEED)
        scores[name] = {
            "n": len(part), "reference_rmse": deep.rmse(y, base),
            "candidate_rmse": deep.rmse(y, new), "bootstrap": bootstrap,
            "passed": bool(deep.rmse(y, new) < deep.rmse(y, base)
                           and bootstrap["gain_ci95_sec"][0] > 0),
        }
    fresh = frame.loc[frame.stage.eq("fresh_apr_oct")]
    if (len(fresh) != v13.EXPECTED_FRESH_ROWS
            or sum(item["n"] for item in scores.values()) != v13.EXPECTED_OOF_ROWS
            or set(fresh.month.unique()) != {4, 10}
            or not fresh.a_valid.all()):
        raise ValueError("Matched April/October paired coverage changed")
    y = fresh.target.to_numpy(dtype=float)
    base = fresh.reference.to_numpy(dtype=float)
    new = fresh.candidate.to_numpy(dtype=float)
    month = fresh.month.to_numpy(dtype=int)
    monthly = {str(m): {
        "n": int((month == m).sum()),
        "reference_rmse": deep.rmse(y[month == m], base[month == m]),
        "candidate_rmse": deep.rmse(y[month == m], new[month == m]),
    } for m in (4, 10)}
    boot = arrival.bootstrap(fresh, base, new, seed=BOOTSTRAP_SEED)
    fresh_score = {"monthly": monthly, "bootstrap": boot,
                   "passed": bool(all(info["candidate_rmse"] < info["reference_rmse"]
                                      for info in monthly.values())
                                  and boot["gain_ci95_sec"][0] > 0)}
    return {"original": scores, "fresh": fresh_score,
            "passed": bool(all(info["passed"] for info in scores.values())
                           and fresh_score["passed"]),
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_repeats": BOOTSTRAP_REPEATS}


def compatibility_paired_path() -> Path:
    return COMPAT_DIR / "paired_predictions.parquet"


def compatibility_terminal_path() -> Path:
    return COMPAT_DIR / "terminal.json"


def compare_compatibility(expected_source_sha256: str) -> dict:
    frozen = check_compatibility_frozen(expected_source_sha256)
    if compatibility_paired_path().exists() or compatibility_terminal_path().exists():
        raise FileExistsError("Compatibility output is already complete or partial")
    before = dict(frozen["source_input_sha256"])
    branch, _, geometry_paths = resolve_geometry_terminal()
    paired = compatibility_pairs(branch)
    scores = score_compatibility(paired)
    rehash_compatibility_sources(frozen, sha(compatibility_protocol_path()))
    paired_sha = publish_new(compatibility_paired_path(),
                             lambda path: paired.to_parquet(path, index=False))
    terminal = {
        "schema_version": 1,
        "promotion_spec_sha256": SPEC_SHA256,
        "protocol_sha256": sha(compatibility_protocol_path()),
        "geometry_branch": branch,
        "fixed_weight": WEIGHT,
        "scores": scores,
        "passed": scores["passed"],
        "retained_policy": "v13" if scores["passed"] else branch,
        "paired_predictions_sha256": paired_sha,
        "source_input_sha256": before,
        "ranking_authorized": bool(scores["passed"]),
        "repeated_2025_stability_check": True,
    }
    rehash_compatibility_sources(frozen, sha(compatibility_protocol_path()))
    write_json_new(compatibility_terminal_path(), terminal)
    verify_compatibility_terminal(expected_source_sha256)
    return terminal


def verify_compatibility_terminal(expected_source_sha256: str) -> dict:
    frozen = check_compatibility_frozen(expected_source_sha256)
    terminal = read_json(compatibility_terminal_path())
    paired_path = compatibility_paired_path()
    branch = frozen["geometry_branch"]
    expected = compatibility_pairs(branch)
    actual = pd.read_parquet(paired_path)
    if not actual.equals(expected):
        raise ValueError("Compatibility paired file differs from receipt-bound OOFs")
    scores = score_compatibility(expected)
    if (terminal.get("schema_version") != 1
            or terminal.get("promotion_spec_sha256") != SPEC_SHA256
            or terminal.get("protocol_sha256") != sha(compatibility_protocol_path())
            or terminal.get("geometry_branch") != branch
            or terminal.get("fixed_weight") != WEIGHT
            or terminal.get("scores") != scores
            or terminal.get("passed") is not scores["passed"]
            or terminal.get("retained_policy") != ("v13" if scores["passed"] else branch)
            or terminal.get("paired_predictions_sha256") != sha(paired_path)
            or terminal.get("source_input_sha256") != frozen["source_input_sha256"]
            or terminal.get("ranking_authorized") is not scores["passed"]
            or terminal.get("repeated_2025_stability_check") is not True):
        raise ValueError("Fixed compatibility terminal or selected route changed")
    return terminal


def require_compatibility_pass(expected_source_sha256: str) -> dict:
    terminal = verify_compatibility_terminal(expected_source_sha256)
    if terminal["passed"] is not True or terminal["retained_policy"] != "v13":
        raise ValueError("Capacity route failed compatibility; retain prior policy")
    return terminal


def guard_protocol_path() -> Path:
    return GUARD_DIR / "protocol.json"


def guard_artifact_paths(name: str) -> dict[str, Path]:
    if name not in ("comparator", "replacement"):
        raise ValueError("Unknown matched February/August model")
    return {key: GUARD_DIR / f"{name}{suffix}" for key, suffix in (
        ("model", ".cbm"), ("oof", "_oof.parquet"),
        ("fit", "_fit.json"), ("receipt", "_receipt.json"))}


def guard_model_hashes(name: str) -> dict[str, str]:
    return hashes(guard_artifact_paths(name))


def guard_source_paths() -> dict[str, Path]:
    compatibility = read_json(compatibility_protocol_path())
    paths = paths_from_map(compatibility["source_input_paths"])
    paths.update({
        "compatibility_protocol": compatibility_protocol_path(),
        "compatibility_terminal": compatibility_terminal_path(),
        "compatibility_paired": compatibility_paired_path(),
    })
    return paths


def freeze_guard(expected_source_sha256: str) -> dict:
    published_source(expected_source_sha256)
    if guard_protocol_path().exists():
        return check_guard_frozen(expected_source_sha256)
    compatibility = require_compatibility_pass(expected_source_sha256)
    args = args_v13()
    frozen, _ = v13.require_protocol(args)
    feature_schema = frozen["feature_schema"]
    cats = [i for i, item in enumerate(feature_schema) if item["dtype"] == "category"]
    expected_long = {**v11.EXPECTED_CATBOOST_PARAMS, "iterations": MAX_TREES}
    if v13.EXPECTED_PARAMS != expected_long:
        raise ValueError("Matched model settings differ beyond maximum iterations")
    receipts = [read_json(V13_DIR / f"{name}_provenance.json") for name in FOLDS]
    vocabulary = receipts[0].get("categorical_vocabulary_sha256")
    if (len(feature_schema) != 184 or len(cats) != 24
            or any(receipt.get("feature_schema") != feature_schema
                   or receipt.get("categorical_feature_indices") != cats
                   or receipt.get("categorical_vocabulary_sha256") != vocabulary
                   or receipt.get("model_params_requested") != v13.EXPECTED_PARAMS
                   for receipt in receipts)
            or set(vocabulary or {}) != {feature_schema[i]["name"] for i in cats}):
        raise ValueError("Original v13 184-field/category/model architecture differs")
    before = hashes(guard_source_paths())
    value = {
        "schema_version": 1, "heldout_months": list(HELDOUT),
        "fixed_weight": WEIGHT, "prefix_trees": PREFIX_TREES,
        "comparator_params": v11.EXPECTED_CATBOOST_PARAMS,
        "replacement_params": v13.EXPECTED_PARAMS,
        "feature_schema": feature_schema,
        "categorical_indices": cats,
        "categorical_vocabulary_sha256": vocabulary,
        "promotion_spec_sha256": SPEC_SHA256,
        "published_source_sha256": expected_source_sha256,
        "compatibility_protocol_sha256": sha(compatibility_protocol_path()),
        "compatibility_terminal_sha256": sha(compatibility_terminal_path()),
        "geometry_branch": compatibility["geometry_branch"],
        "source_input_paths": path_map(guard_source_paths()),
        "source_input_sha256": before,
        "feb_aug_labels_read_to_freeze": False,
        "window_previously_inspected_for_v12": True,
    }
    check_hashes(guard_source_paths(), before, "Feb/Aug guard freeze")
    write_json_new(guard_protocol_path(), value)
    return check_guard_frozen(expected_source_sha256)


def check_guard_frozen(expected_source_sha256: str) -> dict:
    published_source(expected_source_sha256)
    fixed = read_json(guard_protocol_path())
    compatibility = require_compatibility_pass(expected_source_sha256)
    frozen, _ = v13.require_protocol(args_v13())
    feature_schema = frozen["feature_schema"]
    cats = [i for i, item in enumerate(feature_schema) if item["dtype"] == "category"]
    if v13.EXPECTED_PARAMS != {**v11.EXPECTED_CATBOOST_PARAMS,
                               "iterations": MAX_TREES}:
        raise ValueError("Matched model settings differ beyond maximum iterations")
    if (fixed.get("schema_version") != 1
            or fixed.get("heldout_months") != list(HELDOUT)
            or fixed.get("fixed_weight") != WEIGHT
            or fixed.get("prefix_trees") != PREFIX_TREES
            or fixed.get("comparator_params") != v11.EXPECTED_CATBOOST_PARAMS
            or fixed.get("replacement_params") != v13.EXPECTED_PARAMS
            or fixed.get("feature_schema") != feature_schema
            or fixed.get("categorical_indices") != cats
            or set(fixed.get("categorical_vocabulary_sha256", {})) !=
               {feature_schema[i]["name"] for i in cats}
            or fixed.get("promotion_spec_sha256") != SPEC_SHA256
            or fixed.get("published_source_sha256") != expected_source_sha256
            or fixed.get("compatibility_protocol_sha256") != sha(compatibility_protocol_path())
            or fixed.get("compatibility_terminal_sha256") != sha(compatibility_terminal_path())
            or fixed.get("geometry_branch") != compatibility["geometry_branch"]
            or fixed.get("feb_aug_labels_read_to_freeze") is not False
            or fixed.get("window_previously_inspected_for_v12") is not True):
        raise ValueError("Matched February/August guard settings changed")
    current_paths = guard_source_paths()
    if fixed.get("source_input_paths") != path_map(current_paths):
        raise ValueError("February/August source path map changed")
    check_hashes(current_paths, fixed["source_input_sha256"], "Feb/Aug guard")
    return fixed


def rehash_guard_sources(fixed: dict, protocol_sha256: str) -> None:
    """Byte check during a loaded fit without recursively loading feature values."""
    if (fixed.get("published_source_sha256") != sha(Path(__file__).resolve())
            or sha(SPEC) != SPEC_SHA256
            or sha(guard_protocol_path()) != protocol_sha256):
        raise ValueError("Published source, spec or frozen guard protocol changed")
    check_hashes(paths_from_map(fixed["source_input_paths"]),
                 fixed["source_input_sha256"], "Feb/Aug guard")


def fixed_split_receipt(rows: pd.DataFrame) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    if (len(rows) != TRAIN_ROWS or rows.MVT_ID_mvt.isna().any()
            or rows.MVT_ID_mvt.duplicated().any()):
        raise ValueError("Complete training row order or IDs changed")
    timestamps = pd.to_datetime(rows.time, utc=True, errors="coerce")
    if (timestamps.isna().any() or not timestamps.dt.year.eq(2025).all()
            or not np.array_equal(timestamps.dt.month.to_numpy(), rows.month.to_numpy())):
        raise ValueError("Training movement month or UTC time changed")
    split = v13.split_indices(rows, HELDOUT)
    fit, early, test = split["fit"], split["early"], split["test"]
    if (len(fit) < 1_500_000 or len(test) < 10_000
            or set(rows.month.iloc[test].unique()) != set(HELDOUT)
            or rows.month.iloc[np.r_[fit, early]].isin(HELDOUT).any()):
        raise ValueError("Reserved fit/early/heldout coverage changed")
    receipt = {
        "fit_rows": len(fit), "early_rows": len(early), "heldout_rows": len(test),
        "fit_ids_sha256": old_guard.id_hash(rows.MVT_ID_mvt.iloc[fit]),
        "early_ids_sha256": old_guard.id_hash(rows.MVT_ID_mvt.iloc[early]),
        "held_ids_sha256": old_guard.id_hash(rows.MVT_ID_mvt.iloc[test]),
        "fit_indices_sha256": v13.index_sha256(fit),
        "early_indices_sha256": v13.index_sha256(early),
        "held_indices_sha256": v13.index_sha256(test),
        "held_month_counts": {str(month): int((rows.month.iloc[test] == month).sum())
                              for month in HELDOUT},
    }
    return receipt, fit, early, test


def fit_guard_model(name: str, expected_source_sha256: str) -> dict:
    fixed = check_guard_frozen(expected_source_sha256)
    paths = guard_artifact_paths(name)
    if any(path.exists() for path in paths.values()):
        raise FileExistsError(f"{name} matched guard fit already exists or is partial")
    comparator_hashes = {}
    if name == "replacement":
        verify_guard_model("comparator", fixed, expected_source_sha256)
        comparator_hashes = guard_model_hashes("comparator")
    require_memory()
    args = args_v13()
    source_frozen, _ = v13.require_protocol(args)
    before = dict(fixed["source_input_sha256"])
    guard_protocol_sha = sha(guard_protocol_path())
    rows, features, vocabulary = v13.load_features(args, source_frozen)
    if (schema(features) != fixed["feature_schema"]
            or vocabulary != fixed["categorical_vocabulary_sha256"]):
        raise ValueError("Guard model schema or 24-category vocabularies changed")
    split_info, fit, early, test = fixed_split_receipt(rows)
    if name == "replacement":
        prior = read_json(guard_artifact_paths("comparator")["receipt"])
        if (any(prior.get(key) != value for key, value in split_info.items())
                or prior.get("categorical_vocabulary_sha256") != vocabulary):
            raise ValueError("20k replacement differs from matched 10k row/category split")
    rehash_guard_sources(fixed, guard_protocol_sha)
    y = rows.target.to_numpy(dtype=float)
    proxy = rows.proxy.to_numpy(dtype=float)
    categories = features.select_dtypes(include="category").columns.tolist()
    params = (v11.EXPECTED_CATBOOST_PARAMS if name == "comparator"
              else v13.EXPECTED_PARAMS)
    model = CatBoostRegressor(**params)
    fit_pool = Pool(features.iloc[fit], label=(y - proxy)[fit], cat_features=categories)
    early_pool = Pool(features.iloc[early], label=(y - proxy)[early], cat_features=categories)
    started = time.monotonic()
    model.fit(fit_pool, eval_set=early_pool,
              early_stopping_rounds=200, use_best_model=True)
    elapsed = time.monotonic() - started
    del fit_pool, early_pool
    gc.collect()
    rehash_guard_sources(fixed, guard_protocol_sha)
    if (name == "replacement"
            and comparator_hashes != guard_model_hashes("comparator")):
        raise ValueError("Matched comparator changed during replacement fit")
    model_sha = publish_new(paths["model"], lambda path: model.save_model(str(path)))
    trained = v13.verified_saved_model(paths["model"], fixed["feature_schema"],
                                       int(model.tree_count_))
    full = proxy[test] + trained.predict(features.iloc[test], thread_count=2)
    prefix_available = name == "replacement" and int(trained.tree_count_) > PREFIX_TREES
    prefix = (proxy[test] + trained.predict(features.iloc[test],
                                           ntree_end=PREFIX_TREES, thread_count=2)
              if prefix_available else np.full(len(test), np.nan, dtype=float))
    if (not np.isfinite(full).all()
            or (prefix_available and not np.isfinite(prefix).all())):
        raise ValueError("Matched guard raw/prefix expert predictions are nonfinite")
    held = rows.iloc[test]
    oof = pd.DataFrame({
        "MVT_ID_mvt": held.MVT_ID_mvt.to_numpy(copy=True),
        "target": y[test],
        "MVT_TIME_UTC_mvt": pd.to_datetime(held.time, utc=True).reset_index(drop=True),
        "full_expert": full, "prefix_10000_expert": prefix,
    })
    rehash_guard_sources(fixed, guard_protocol_sha)
    if (name == "replacement"
            and comparator_hashes != guard_model_hashes("comparator")):
        raise ValueError("Guard sources or comparator changed while fitting")
    oof_sha = publish_new(paths["oof"], lambda path: oof.to_parquet(path, index=False))
    report = {
        "name": name, "heldout_months": list(HELDOUT),
        "requested_params": params, "feature_schema": fixed["feature_schema"],
        "categorical_indices": fixed["categorical_indices"],
        "categorical_vocabulary_sha256": vocabulary,
        **split_info,
        "trees": int(trained.tree_count_),
        "prefix_trees": PREFIX_TREES if prefix_available else None,
        "exact_prefix_available": prefix_available,
        "best_iteration": int(model.get_best_iteration()),
        "fit_seconds": float(elapsed),
        "early_stopping_rounds": 200, "use_best_model": True,
        "feb_aug_excluded_from_fit_and_early": True,
        "protocol_sha256": sha(guard_protocol_path()),
    }
    write_json_new(paths["fit"], report)
    receipt = {
        "schema_version": 1, "name": name, "heldout_months": list(HELDOUT),
        "protocol_sha256": sha(guard_protocol_path()),
        "source_input_sha256": before,
        "requested_params": params,
        "effective_params": trained.get_all_params(),
        "feature_schema": fixed["feature_schema"],
        "categorical_indices": fixed["categorical_indices"],
        "categorical_vocabulary_sha256": vocabulary,
        **split_info,
        "trees": int(trained.tree_count_),
        "prefix_trees": PREFIX_TREES if prefix_available else None,
        "exact_prefix_available": prefix_available,
        "model_sha256": model_sha, "oof_sha256": oof_sha,
        "fit_report_sha256": sha(paths["fit"]),
        "paired_comparator_sha256": comparator_hashes,
    }
    rehash_guard_sources(fixed, guard_protocol_sha)
    if (name == "replacement"
            and comparator_hashes != guard_model_hashes("comparator")):
        raise ValueError("Guard source or comparator changed before fit receipt")
    write_json_new(paths["receipt"], receipt)
    del rows, features, held, oof, full, prefix, model, trained
    gc.collect()
    verify_guard_model(name, fixed, expected_source_sha256)
    return {"name": name, "trees": receipt["trees"],
            "heldout_rows": receipt["heldout_rows"],
            "model_sha256": model_sha}


def verify_guard_model(name: str, fixed: dict,
                       expected_source_sha256: str) -> pd.DataFrame:
    """Replay a saved matched model's exact masks, OOF, prefix, and receipts."""
    require_memory()
    protocol_sha = sha(guard_protocol_path())
    rehash_guard_sources(fixed, protocol_sha)
    paths = guard_artifact_paths(name)
    receipt = read_json(paths["receipt"])
    report = read_json(paths["fit"])
    params = (v11.EXPECTED_CATBOOST_PARAMS if name == "comparator"
              else v13.EXPECTED_PARAMS)
    paired = guard_model_hashes("comparator") if name == "replacement" else {}
    keys = ("name", "heldout_months", "protocol_sha256", "requested_params",
            "feature_schema", "categorical_indices",
            "categorical_vocabulary_sha256", "fit_rows", "early_rows",
            "heldout_rows", "fit_ids_sha256", "early_ids_sha256",
            "held_ids_sha256", "fit_indices_sha256", "early_indices_sha256",
            "held_indices_sha256", "held_month_counts", "trees",
            "prefix_trees", "exact_prefix_available")
    if (receipt.get("schema_version") != 1 or receipt.get("name") != name
            or receipt.get("heldout_months") != list(HELDOUT)
            or receipt.get("protocol_sha256") != protocol_sha
            or receipt.get("source_input_sha256") != fixed["source_input_sha256"]
            or receipt.get("requested_params") != params
            or receipt.get("feature_schema") != fixed["feature_schema"]
            or receipt.get("categorical_indices") != fixed["categorical_indices"]
            or receipt.get("categorical_vocabulary_sha256") !=
               fixed["categorical_vocabulary_sha256"]
            or receipt.get("model_sha256") != sha(paths["model"])
            or receipt.get("oof_sha256") != sha(paths["oof"])
            or receipt.get("fit_report_sha256") != sha(paths["fit"])
            or receipt.get("paired_comparator_sha256") != paired
            or any(receipt.get(key) != report.get(key) for key in keys)
            or report.get("early_stopping_rounds") != 200
            or report.get("use_best_model") is not True
            or report.get("feb_aug_excluded_from_fit_and_early") is not True
            or type(receipt.get("trees")) is not int
            or not 1 <= receipt["trees"] <= params["iterations"]):
        raise ValueError(f"{name} guard receipt/model/fit provenance differs")
    model = v13.verified_saved_model(paths["model"], fixed["feature_schema"],
                                     receipt["trees"])
    prefix_available = name == "replacement" and receipt["trees"] > PREFIX_TREES
    if (receipt.get("effective_params") != model.get_all_params()
            or receipt.get("exact_prefix_available") is not prefix_available
            or receipt.get("prefix_trees") !=
               (PREFIX_TREES if prefix_available else None)):
        raise ValueError(f"{name} actual saved model or exact own prefix differs")
    args = args_v13()
    source_frozen, source_protocol_sha = v13.require_protocol(args)
    rows, features, vocabulary = v13.load_features(args, source_frozen)
    split_info, _, _, test = fixed_split_receipt(rows)
    if (schema(features) != fixed["feature_schema"]
            or vocabulary != fixed["categorical_vocabulary_sha256"]
            or any(receipt.get(key) != value for key, value in split_info.items())):
        raise ValueError(f"{name} exact fit/early/held IDs or categories differ")
    y = rows.target.to_numpy(dtype=float)
    proxy = rows.proxy.to_numpy(dtype=float)
    expected_full = proxy[test] + model.predict(features.iloc[test], thread_count=2)
    expected_prefix = (proxy[test] + model.predict(features.iloc[test],
                                                   ntree_end=PREFIX_TREES,
                                                   thread_count=2)
                       if prefix_available else np.full(len(test), np.nan))
    frame = pd.read_parquet(paths["oof"])
    expected_time = pd.to_datetime(rows.time.iloc[test], utc=True).to_numpy()
    if (list(frame) != ["MVT_ID_mvt", "target", "MVT_TIME_UTC_mvt",
                        "full_expert", "prefix_10000_expert"]
            or len(frame) != len(test)
            or not np.array_equal(frame.MVT_ID_mvt.to_numpy(),
                                  rows.MVT_ID_mvt.iloc[test].to_numpy())
            or not np.array_equal(frame.target.to_numpy(dtype=float), y[test])
            or not np.array_equal(pd.to_datetime(frame.MVT_TIME_UTC_mvt,
                                                 utc=True).to_numpy(), expected_time)
            or not np.array_equal(frame.full_expert.to_numpy(dtype=float),
                                  expected_full)
            or not np.array_equal(frame.prefix_10000_expert.to_numpy(dtype=float),
                                  expected_prefix, equal_nan=True)
            or not np.isfinite(expected_full).all()
            or (prefix_available and not np.isfinite(expected_prefix).all())):
        raise ValueError(f"{name} saved raw/prefix OOF differs from actual model and labels")
    v13.assert_source_snapshot(args, source_frozen, source_protocol_sha)
    del rows, features, model, expected_full, expected_prefix, y, proxy
    gc.collect()
    rehash_guard_sources(fixed, protocol_sha)
    return frame


def guard_paired_path() -> Path:
    return GUARD_DIR / "paired_predictions.parquet"


def guard_terminal_path() -> Path:
    return GUARD_DIR / "terminal.json"


def guard_pair(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    exact_ids(right.MVT_ID_mvt, left.MVT_ID_mvt, "Matched February/August IDs")
    aligned = right.set_index("MVT_ID_mvt").loc[left.MVT_ID_mvt.to_numpy()].reset_index()
    if (not np.array_equal(left.target.to_numpy(dtype=float),
                           aligned.target.to_numpy(dtype=float))
            or not np.array_equal(pd.to_datetime(left.MVT_TIME_UTC_mvt,
                                                 utc=True).to_numpy(),
                                  pd.to_datetime(aligned.MVT_TIME_UTC_mvt,
                                                 utc=True).to_numpy())):
        raise ValueError("February/August comparator and replacement labels/times differ")
    return pd.DataFrame({
        "MVT_ID_mvt": left.MVT_ID_mvt.to_numpy(copy=True),
        "target": left.target.to_numpy(dtype=float, copy=True),
        "MVT_TIME_UTC_mvt": pd.to_datetime(left.MVT_TIME_UTC_mvt,
                                           utc=True).reset_index(drop=True),
        "old_raw": left.full_expert.to_numpy(dtype=float, copy=True),
        "full_raw": aligned.full_expert.to_numpy(dtype=float, copy=True),
        "own_prefix_raw": aligned.prefix_10000_expert.to_numpy(dtype=float, copy=True),
    })


def score_guard(paired: pd.DataFrame, exact_prefix_available: bool) -> tuple[dict, pd.DataFrame]:
    raw_columns = ["MVT_ID_mvt", "target", "MVT_TIME_UTC_mvt",
                   "old_raw", "full_raw", "own_prefix_raw"]
    if (list(paired) != raw_columns
            or paired.MVT_ID_mvt.isna().any() or paired.MVT_ID_mvt.duplicated().any()
            or not np.isfinite(paired[["target", "old_raw", "full_raw"]]
                               .to_numpy(dtype=float)).all()):
        raise ValueError("Matched February/August raw pairing differs")
    times = pd.to_datetime(paired.MVT_TIME_UTC_mvt, utc=True, errors="coerce")
    if times.isna().any() or set(times.dt.month.unique()) != set(HELDOUT):
        raise ValueError("Matched February/August timestamps differ")
    if ((exact_prefix_available and not np.isfinite(
            paired.own_prefix_raw.to_numpy(dtype=float)).all())
            or (not exact_prefix_available and not paired.own_prefix_raw.isna().all())):
        raise ValueError("Own-model 10000-tree prefix availability differs")
    y = paired.target.to_numpy(dtype=float)
    old_raw = paired.old_raw.to_numpy(dtype=float)
    full_raw = paired.full_raw.to_numpy(dtype=float)
    old = np.maximum(old_raw, 0)
    candidate = np.maximum(old_raw + WEIGHT * (full_raw - old_raw), 0)
    prefix = (np.maximum(old_raw + WEIGHT *
                         (paired.own_prefix_raw.to_numpy(dtype=float) - old_raw), 0)
              if exact_prefix_available else np.full(len(paired), np.nan))
    month = times.dt.month.to_numpy()
    monthly = {}
    for value in HELDOUT:
        take = month == value
        if int(take.sum()) < 1000:
            raise ValueError("Incomplete February/August month coverage")
        monthly[str(value)] = {
            "n": int(take.sum()),
            "old_rmse": deep.rmse(y[take], old[take]),
            "candidate_rmse": deep.rmse(y[take], candidate[take]),
            "prefix_candidate_rmse": (deep.rmse(y[take], prefix[take])
                                      if exact_prefix_available else None),
        }
    base_boot = arrival.bootstrap(paired, old, candidate, seed=BOOTSTRAP_SEED)
    prefix_boot = (arrival.bootstrap(paired, prefix, candidate, seed=BOOTSTRAP_SEED)
                   if exact_prefix_available else None)
    baseline_pass = (all(item["candidate_rmse"] < item["old_rmse"]
                         for item in monthly.values())
                     and base_boot["gain_ci95_sec"][0] > 0)
    prefix_pass = (exact_prefix_available
                   and all(item["candidate_rmse"] < item["prefix_candidate_rmse"]
                           for item in monthly.values())
                   and prefix_boot["gain_ci95_sec"][0] > 0)
    output = paired.copy()
    output["old_clipped"] = old
    output["fixed_candidate_clipped"] = candidate
    output["own_prefix_candidate_clipped"] = prefix
    return {
        "monthly": monthly,
        "old_to_candidate_day_bootstrap": base_boot,
        "prefix_to_full_day_bootstrap": prefix_boot,
        "exact_prefix_available": exact_prefix_available,
        "baseline_gate_passed": bool(baseline_pass),
        "own_prefix_gate_passed": bool(prefix_pass),
        "passed": bool(baseline_pass and prefix_pass),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_repeats": BOOTSTRAP_REPEATS,
        "pooled_old_rmse": deep.rmse(y, old),
        "pooled_candidate_rmse": deep.rmse(y, candidate),
    }, output


def evaluate_guard(expected_source_sha256: str) -> dict:
    fixed = check_guard_frozen(expected_source_sha256)
    if guard_paired_path().exists() or guard_terminal_path().exists():
        raise FileExistsError("Matched guard already evaluated or has a partial output")
    require_memory()
    before = dict(fixed["source_input_sha256"])
    left_hashes = guard_model_hashes("comparator")
    right_hashes = guard_model_hashes("replacement")
    left = verify_guard_model("comparator", fixed, expected_source_sha256)
    right = verify_guard_model("replacement", fixed, expected_source_sha256)
    paired = guard_pair(left, right)
    replacement = read_json(guard_artifact_paths("replacement")["receipt"])
    if (replacement["trees"] > PREFIX_TREES) is not replacement["exact_prefix_available"]:
        raise ValueError("Replacement saved trees do not support claimed exact prefix")
    score, output = score_guard(paired, replacement["exact_prefix_available"])
    rehash_guard_sources(fixed, sha(guard_protocol_path()))
    if (before != hashes(paths_from_map(fixed["source_input_paths"]))
            or left_hashes != guard_model_hashes("comparator")
            or right_hashes != guard_model_hashes("replacement")):
        raise ValueError("Model or frozen input changed during reserved scoring")
    paired_sha = publish_new(guard_paired_path(),
                             lambda path: output.to_parquet(path, index=False))
    terminal = {
        "schema_version": 1,
        "promotion_spec_sha256": SPEC_SHA256,
        "protocol_sha256": sha(guard_protocol_path()),
        "compatibility_terminal_sha256": sha(compatibility_terminal_path()),
        "geometry_branch": fixed["geometry_branch"],
        "heldout_months": list(HELDOUT), "fixed_weight": WEIGHT,
        "score": score, "passed": score["passed"],
        "retained_policy": "v13" if score["passed"] else fixed["geometry_branch"],
        "comparator_sha256": left_hashes,
        "replacement_sha256": right_hashes,
        "paired_predictions_sha256": paired_sha,
        "ranking_authorized": bool(score["passed"]),
        "reused_2025_window_not_untouched": True,
    }
    rehash_guard_sources(fixed, sha(guard_protocol_path()))
    if (left_hashes != guard_model_hashes("comparator")
            or right_hashes != guard_model_hashes("replacement")):
        raise ValueError("Matched guard models changed before terminal receipt")
    write_json_new(guard_terminal_path(), terminal)
    verify_guard_terminal(expected_source_sha256)
    return terminal


def verify_guard_terminal(expected_source_sha256: str) -> dict:
    fixed = check_guard_frozen(expected_source_sha256)
    require_memory()
    terminal = read_json(guard_terminal_path())
    left = verify_guard_model("comparator", fixed, expected_source_sha256)
    right = verify_guard_model("replacement", fixed, expected_source_sha256)
    paired = guard_pair(left, right)
    replacement = read_json(guard_artifact_paths("replacement")["receipt"])
    score, expected = score_guard(paired, replacement["exact_prefix_available"])
    actual = pd.read_parquet(guard_paired_path())
    if not actual.equals(expected):
        raise ValueError("Sealed guard paired outputs differ from both actual models")
    if (terminal.get("schema_version") != 1
            or terminal.get("promotion_spec_sha256") != SPEC_SHA256
            or terminal.get("protocol_sha256") != sha(guard_protocol_path())
            or terminal.get("compatibility_terminal_sha256") !=
               sha(compatibility_terminal_path())
            or terminal.get("geometry_branch") != fixed["geometry_branch"]
            or terminal.get("heldout_months") != list(HELDOUT)
            or terminal.get("fixed_weight") != WEIGHT
            or terminal.get("score") != score
            or terminal.get("passed") is not score["passed"]
            or terminal.get("retained_policy") !=
               ("v13" if score["passed"] else fixed["geometry_branch"])
            or terminal.get("comparator_sha256") != guard_model_hashes("comparator")
            or terminal.get("replacement_sha256") != guard_model_hashes("replacement")
            or terminal.get("paired_predictions_sha256") != sha(guard_paired_path())
            or terminal.get("ranking_authorized") is not score["passed"]
            or terminal.get("reused_2025_window_not_untouched") is not True):
        raise ValueError("Sealed February/August gate differs from fixed replay")
    return terminal


def require_all_gates_passed(expected_source_sha256: str) -> tuple[dict, dict]:
    fixed = check_guard_frozen(expected_source_sha256)
    terminal = verify_guard_terminal(expected_source_sha256)
    if (terminal["passed"] is not True
            or terminal["retained_policy"] != "v13"
            or terminal["ranking_authorized"] is not True):
        raise ValueError("Capacity component gate failed; retain prior policy")
    return fixed, terminal


def final_protocol_path() -> Path:
    return FINAL_DIR / "protocol.json"


def final_fit_source_paths() -> dict[str, Path]:
    fixed = read_json(guard_protocol_path())
    paths = paths_from_map(fixed["source_input_paths"])
    paths.update({
        "guard_protocol": guard_protocol_path(),
        "guard_terminal": guard_terminal_path(),
        "guard_paired": guard_paired_path(),
    })
    for name in ("comparator", "replacement"):
        paths.update({f"guard_{name}_{key}": path
                      for key, path in guard_artifact_paths(name).items()})
    return paths


def prepare_final(expected_source_sha256: str) -> dict:
    published_source(expected_source_sha256)
    if final_protocol_path().exists():
        return check_final_prepared(expected_source_sha256)
    fixed, terminal = require_all_gates_passed(expected_source_sha256)
    source_frozen, _ = v13.require_protocol(args_v13())
    paths = final_fit_source_paths()
    before = hashes(paths)
    value = {
        "schema_version": 1, "selected_route": "v13",
        "promotion_spec_sha256": SPEC_SHA256,
        "published_source_sha256": expected_source_sha256,
        "geometry_branch": fixed["geometry_branch"],
        "fixed_weight": WEIGHT,
        "v13_original_protocol_sha256": sha(V13_DIR / "protocol.json"),
        "compatibility_terminal_sha256": sha(compatibility_terminal_path()),
        "component_guard_terminal_sha256": sha(guard_terminal_path()),
        "feature_schema": source_frozen["feature_schema"],
        "categorical_vocabulary_sha256": fixed["categorical_vocabulary_sha256"],
        "source_input_paths": path_map(paths),
        "source_input_sha256": before,
        "status": "frozen_before_full_2025_fit",
    }
    if terminal["passed"] is not True:
        raise ValueError("Failed component gate cannot prepare final fit")
    check_hashes(paths, before, "Full 2025 fit freeze")
    write_json_new(final_protocol_path(), value)
    return check_final_prepared(expected_source_sha256)


def check_final_prepared(expected_source_sha256: str) -> dict:
    published_source(expected_source_sha256)
    fixed, terminal = require_all_gates_passed(expected_source_sha256)
    prepared = read_json(final_protocol_path())
    source_frozen, _ = v13.require_protocol(args_v13())
    paths = final_fit_source_paths()
    if (prepared.get("schema_version") != 1
            or prepared.get("selected_route") != "v13"
            or prepared.get("promotion_spec_sha256") != SPEC_SHA256
            or prepared.get("published_source_sha256") != expected_source_sha256
            or prepared.get("geometry_branch") != fixed["geometry_branch"]
            or prepared.get("fixed_weight") != WEIGHT
            or prepared.get("v13_original_protocol_sha256") != sha(V13_DIR / "protocol.json")
            or prepared.get("compatibility_terminal_sha256") !=
               sha(compatibility_terminal_path())
            or prepared.get("component_guard_terminal_sha256") != sha(guard_terminal_path())
            or prepared.get("feature_schema") != source_frozen["feature_schema"]
            or prepared.get("categorical_vocabulary_sha256") !=
               fixed["categorical_vocabulary_sha256"]
            or prepared.get("source_input_paths") != path_map(paths)
            or prepared.get("status") != "frozen_before_full_2025_fit"
            or terminal["retained_policy"] != "v13"):
        raise ValueError("Full fit selection, 184-field schema or source seal changed")
    check_hashes(paths, prepared["source_input_sha256"], "Full 2025 fit")
    return prepared


def rehash_final_sources(prepared: dict, protocol_sha256: str) -> None:
    if (sha(final_protocol_path()) != protocol_sha256
            or sha(Path(__file__).resolve()) != prepared["published_source_sha256"]
            or sha(SPEC) != SPEC_SHA256):
        raise ValueError("Published final source, spec or protocol changed")
    check_hashes(paths_from_map(prepared["source_input_paths"]),
                 prepared["source_input_sha256"], "Full 2025 fit")


def original_rounds(source_frozen: dict) -> tuple[int, list[int]]:
    trees = []
    vocab = None
    for name in FOLDS:
        receipt = read_json(V13_DIR / f"{name}_provenance.json")
        report = read_json(V13_DIR / f"{name}_fit.json")
        if (receipt.get("fold") != name
                or receipt.get("feature_schema") != source_frozen["feature_schema"]
                or receipt.get("model_params_requested") != v13.EXPECTED_PARAMS
                or receipt.get("saved_trees") != report.get("saved_trees")
                or receipt.get("model_sha256") != sha(V13_DIR / f"{name}.cbm")
                or receipt.get("oof_sha256") != sha(V13_DIR / f"{name}_oof.parquet")
                or receipt.get("fit_report_sha256") != sha(V13_DIR / f"{name}_fit.json")
                or type(receipt.get("saved_trees")) is not int
                or not PREFIX_TREES < receipt["saved_trees"] <= MAX_TREES
                or receipt.get("exact_prefix_available") is not True):
            raise ValueError(f"Original {name} full/prefix tree receipt changed")
        if vocab is None:
            vocab = receipt["categorical_vocabulary_sha256"]
        elif vocab != receipt["categorical_vocabulary_sha256"]:
            raise ValueError("Original fold category vocabularies differ")
        trees.append(receipt["saved_trees"])
    return int(np.median(trees)), trees


def full_model_path() -> Path:
    return FINAL_DIR / "full_2025.cbm"


def full_report_path() -> Path:
    return FINAL_DIR / "final_model.json"


def baseline_full_id_receipt() -> dict:
    rows = pd.read_parquet(args_v13().cache_dir / "training_rows.parquet",
                           columns=["MVT_ID_mvt", "target", "proxy", "month", "time"])
    if (len(rows) != TRAIN_ROWS or rows.MVT_ID_mvt.isna().any()
            or rows.MVT_ID_mvt.duplicated().any()):
        raise ValueError("Full 2025 baseline IDs changed")
    y = rows.target.to_numpy(dtype=float)
    proxy = rows.proxy.to_numpy(dtype=float)
    eligible = (np.isfinite(y) & (y >= 0) & (y <= 86400)
                & np.isfinite(proxy) & (proxy >= 0) & (proxy <= 7200))
    if int(eligible.sum()) < 1_500_000:
        raise ValueError("Too few eligible full 2025 rows")
    return {
        "training_rows": TRAIN_ROWS,
        "all_training_ids_sha256": old_guard.id_hash(rows.MVT_ID_mvt),
        "eligible_training_rows": int(eligible.sum()),
        "eligible_training_ids_sha256": old_guard.id_hash(rows.MVT_ID_mvt.loc[eligible]),
    }


def verify_full_model(expected_source_sha256: str) -> dict:
    prepared = check_final_prepared(expected_source_sha256)
    source_frozen, _ = v13.require_protocol(args_v13())
    rounds, original_trees = original_rounds(source_frozen)
    report = read_json(full_report_path())
    requested = dict(v13.EXPECTED_PARAMS)
    requested["iterations"] = rounds
    expected_ids = baseline_full_id_receipt()
    cats = [i for i, item in enumerate(source_frozen["feature_schema"])
            if item["dtype"] == "category"]
    if (report.get("selected_route") != "v13"
            or report.get("fixed_weight") != WEIGHT
            or report.get("iterations") != rounds
            or report.get("original_fold_trees") != dict(zip(FOLDS, original_trees))
            or report.get("requested_params") != requested
            or report.get("feature_schema") != source_frozen["feature_schema"]
            or report.get("categorical_indices") != cats
            or report.get("categorical_vocabulary_sha256") !=
               prepared["categorical_vocabulary_sha256"]
            or any(report.get(key) != value for key, value in expected_ids.items())
            or report.get("model_sha256") != sha(full_model_path())
            or report.get("final_protocol_sha256") != sha(final_protocol_path())
            or report.get("fit_input_sha256") != prepared["source_input_sha256"]
            or report.get("component_guard_terminal_sha256") !=
               sha(guard_terminal_path())
            or report.get("ranking_prediction_created") is not False):
        raise ValueError("Full model or eligible-ID provenance differs")
    model = v13.verified_saved_model(full_model_path(),
                                     source_frozen["feature_schema"], rounds)
    if (list(model.get_cat_feature_indices()) != cats
            or list(model.feature_names_) !=
               [item["name"] for item in source_frozen["feature_schema"]]):
        raise ValueError("Saved full model category or feature architecture differs")
    return report


def fit_final(expected_source_sha256: str) -> dict:
    require_memory()
    prepared = check_final_prepared(expected_source_sha256)
    if full_model_path().exists() or full_report_path().exists():
        raise FileExistsError("Full capacity model already exists or is partial")
    source_frozen, _ = v13.require_protocol(args_v13())
    rounds, original_trees = original_rounds(source_frozen)
    protocol_sha = sha(final_protocol_path())
    rows, features, vocabulary = v13.load_features(args_v13(), source_frozen)
    if (len(rows) != TRAIN_ROWS or schema(features) != source_frozen["feature_schema"]
            or vocabulary != prepared["categorical_vocabulary_sha256"]):
        raise ValueError("Full capacity model feature universe or categories changed")
    y = rows.target.to_numpy(dtype=float)
    proxy = rows.proxy.to_numpy(dtype=float)
    eligible = (np.isfinite(y) & (y >= 0) & (y <= 86400)
                & np.isfinite(proxy) & (proxy >= 0) & (proxy <= 7200))
    ids = {
        "training_rows": len(rows),
        "all_training_ids_sha256": old_guard.id_hash(rows.MVT_ID_mvt),
        "eligible_training_rows": int(eligible.sum()),
        "eligible_training_ids_sha256": old_guard.id_hash(rows.MVT_ID_mvt.loc[eligible]),
    }
    if ids != baseline_full_id_receipt():
        raise ValueError("Full model eligible IDs differ from frozen baseline")
    rehash_final_sources(prepared, protocol_sha)
    params = dict(v13.EXPECTED_PARAMS)
    params["iterations"] = rounds
    categories = features.select_dtypes(include="category").columns.tolist()
    model = CatBoostRegressor(**params)
    train = Pool(features.loc[eligible], label=(y - proxy)[eligible],
                 cat_features=categories)
    started = time.monotonic()
    model.fit(train)
    elapsed = time.monotonic() - started
    del train, rows, features
    gc.collect()
    if int(model.tree_count_) != rounds:
        raise ValueError("Final saved tree count differs from original-fold median")
    rehash_final_sources(prepared, protocol_sha)
    model_sha = publish_new(full_model_path(), lambda path: model.save_model(str(path)))
    saved = v13.verified_saved_model(full_model_path(),
                                     source_frozen["feature_schema"], rounds)
    cats = [i for i, item in enumerate(source_frozen["feature_schema"])
            if item["dtype"] == "category"]
    if list(saved.get_cat_feature_indices()) != cats:
        raise ValueError("Full 2025 saved category indices changed")
    report = {
        "selected_route": "v13", "fixed_weight": WEIGHT,
        "iterations": rounds, "original_fold_trees": dict(zip(FOLDS, original_trees)),
        **ids,
        "requested_params": params,
        "feature_schema": source_frozen["feature_schema"],
        "categorical_indices": cats,
        "categorical_vocabulary_sha256": vocabulary,
        "fit_seconds": float(elapsed), "model_sha256": model_sha,
        "final_protocol_sha256": protocol_sha,
        "fit_input_sha256": prepared["source_input_sha256"],
        "component_guard_terminal_sha256": sha(guard_terminal_path()),
        "ranking_prediction_created": False,
    }
    rehash_final_sources(prepared, protocol_sha)
    write_json_new(full_report_path(), report)
    verify_full_model(expected_source_sha256)
    return {key: report[key] for key in ("iterations", "original_fold_trees",
                                        "eligible_training_rows", "model_sha256")}


def ranking_source_paths() -> dict[str, Path]:
    prepared = read_json(final_protocol_path())
    paths = paths_from_map(prepared["source_input_paths"])
    paths.update({f"v9_rank_{name}": path for name, path in
                  v9_final.ranking_input_paths("v11").items()})
    paths.update({
        "raw_2026_ranking": args_v13().data_dir / "ranking.parquet",
        "template": TEMPLATE,
        "local_v9_submission": V9_SUBMISSION,
        "v9_saved_prediction": ROOT / "artifacts/later-feature-final/predictions.parquet",
        "v9_saved_ranking_expert": ROOT / "artifacts/later-feature-final/ranking_expert.parquet",
        "v9_saved_ranking_manifest": ROOT / "artifacts/later-feature-final/ranking_manifest.json",
        "full_capacity_model": full_model_path(),
        "full_capacity_model_report": full_report_path(),
        "capacity_final_protocol": final_protocol_path(),
        "capacity_guard_terminal": guard_terminal_path(),
        "capacity_compatibility_terminal": compatibility_terminal_path(),
    })
    return unique_paths(paths)


def ranking_seal_path() -> Path:
    return FINAL_DIR / "ranking_inputs.json"


def seal_ranking(expected_source_sha256: str) -> dict:
    published_source(expected_source_sha256)
    if ranking_seal_path().exists():
        return check_ranking_seal(expected_source_sha256)
    prepared = check_final_prepared(expected_source_sha256)
    final_report = verify_full_model(expected_source_sha256)
    if sha(V9_SUBMISSION) != V9_SHA256:
        raise ValueError("Sealed local v9 prediction changed")
    paths = ranking_source_paths()
    before = hashes(paths)
    value = {
        "schema_version": 1, "selected_route": "v13",
        "fixed_weight": WEIGHT,
        "promotion_spec_sha256": SPEC_SHA256,
        "published_source_sha256": expected_source_sha256,
        "final_protocol_sha256": sha(final_protocol_path()),
        "final_model_sha256": final_report["model_sha256"],
        "fit_input_sha256": prepared["source_input_sha256"],
        "ranking_input_paths": path_map(paths),
        "ranking_input_sha256": before,
        "status": "sealed_before_any_2026_ranking_feature_value_read",
    }
    check_hashes(paths, before, "Ranking seal")
    write_json_new(ranking_seal_path(), value)
    return check_ranking_seal(expected_source_sha256)


def check_ranking_seal(expected_source_sha256: str) -> dict:
    published_source(expected_source_sha256)
    prepared = check_final_prepared(expected_source_sha256)
    model_report = verify_full_model(expected_source_sha256)
    value = read_json(ranking_seal_path())
    paths = ranking_source_paths()
    if (value.get("schema_version") != 1
            or value.get("selected_route") != "v13"
            or value.get("fixed_weight") != WEIGHT
            or value.get("promotion_spec_sha256") != SPEC_SHA256
            or value.get("published_source_sha256") != expected_source_sha256
            or value.get("final_protocol_sha256") != sha(final_protocol_path())
            or value.get("final_model_sha256") != model_report["model_sha256"]
            or value.get("fit_input_sha256") != prepared["source_input_sha256"]
            or value.get("ranking_input_paths") != path_map(paths)
            or value.get("status") !=
               "sealed_before_any_2026_ranking_feature_value_read"
            or sha(V9_SUBMISSION) != V9_SHA256):
        raise ValueError("Capacity ranking seal, source or local v9 changed")
    check_hashes(paths, value["ranking_input_sha256"], "Ranking")
    return value


def rehash_ranking_sources(seal: dict, seal_sha256: str) -> None:
    if (sha(ranking_seal_path()) != seal_sha256
            or sha(Path(__file__).resolve()) != seal["published_source_sha256"]
            or sha(SPEC) != SPEC_SHA256):
        raise ValueError("Published ranking source/spec/seal changed")
    check_hashes(paths_from_map(seal["ranking_input_paths"]),
                 seal["ranking_input_sha256"], "Ranking")


def predict_ranking(expected_source_sha256: str) -> dict:
    require_memory()
    output_path = FINAL_DIR / "predictions.parquet"
    expert_path = FINAL_DIR / "ranking_expert.parquet"
    manifest_path = FINAL_DIR / "ranking_manifest.json"
    if any(path.exists() for path in (output_path, expert_path, manifest_path)):
        raise FileExistsError("Capacity ranking output already exists or is partial")
    seal = check_ranking_seal(expected_source_sha256)
    seal_sha = sha(ranking_seal_path())
    source_frozen, _ = v13.require_protocol(args_v13())
    final_report = verify_full_model(expected_source_sha256)
    rows, features = v9_final.load_ranking_features("v11")
    if (len(rows) != RANK_ROWS or schema(features) != source_frozen["feature_schema"]
            or rows.MVT_ID_mvt.isna().any() or rows.MVT_ID_mvt.duplicated().any()):
        raise ValueError("Saved 184-field ranking matrix or ID order changed")
    template = pd.read_parquet(TEMPLATE)
    local_v9 = pd.read_parquet(V9_SUBMISSION)
    v9_saved = pd.read_parquet(ROOT / "artifacts/later-feature-final/predictions.parquet")
    v11_raw = pd.read_parquet(ROOT / "artifacts/later-feature-final/ranking_expert.parquet")
    v7_saved = pd.read_parquet(ROOT / "artifacts/v7-runway-traffic/predictions.parquet")
    v8_saved = pd.read_parquet(ROOT / "artifacts/current-candidate/predictions.parquet")
    required = ["MVT_ID_mvt", "TAXITIME_SEC_mvt"]
    refs = (template, local_v9, v9_saved, v7_saved, v8_saved, rows, v11_raw)
    if (any(len(frame) != RANK_ROWS or frame.MVT_ID_mvt.isna().any()
            or frame.MVT_ID_mvt.duplicated().any() for frame in refs)
            or any(not np.array_equal(frame.MVT_ID_mvt.to_numpy(),
                                      template.MVT_ID_mvt.to_numpy()) for frame in refs)
            or any(list(frame) != required for frame in
                   (template, local_v9, v9_saved, v7_saved, v8_saved))
            or list(v11_raw) != ["MVT_ID_mvt", "a_valid", "raw_expert"]
            or not np.array_equal(local_v9.TAXITIME_SEC_mvt.to_numpy(dtype=float),
                                  v9_saved.TAXITIME_SEC_mvt.to_numpy(dtype=float))):
        raise ValueError("Sealed v9, saved v11 raw or template IDs/values changed")
    proxy = rows.proxy.to_numpy(dtype=float)
    valid = np.isfinite(proxy) & (proxy >= 0) & (proxy <= 7200)
    if (int(valid.sum()) != VALID_RANK_ROWS
            or int((~valid).sum()) != 5_464
            or not np.array_equal(v11_raw.a_valid.to_numpy(dtype=bool), valid)):
        raise ValueError("Exact 339377 valid-AOBT gate or saved v11 raw coverage changed")
    current = local_v9.TAXITIME_SEC_mvt.to_numpy(dtype=float, copy=True)
    old = v11_raw.raw_expert.to_numpy(dtype=float)
    v7_values = v7_saved.TAXITIME_SEC_mvt.to_numpy(dtype=float)
    v8_values = v8_saved.TAXITIME_SEC_mvt.to_numpy(dtype=float)
    if (not np.array_equal(current[valid],
                           np.maximum(v7_values[valid] + WEIGHT *
                                      (old[valid] - v7_values[valid]), 0))
            or not np.array_equal(current[~valid], v8_values[~valid])
            or int(np.count_nonzero(v8_values[~valid] != v7_values[~valid])) != 4_907):
        raise ValueError("Sealed v9 formula or 4907 missing-clock rows changed")
    rehash_ranking_sources(seal, seal_sha)
    rounds = final_report["iterations"]
    model = v13.verified_saved_model(full_model_path(),
                                     source_frozen["feature_schema"], rounds)
    raw = np.full(RANK_ROWS, np.nan, dtype=float)
    raw[valid] = proxy[valid] + model.predict(features.loc[valid], thread_count=2)
    output = v13.fixed_replacement(current, old, raw, valid)
    if (not np.array_equal(output[~valid], current[~valid])
            or not np.isfinite(output).all() or (output < 0).any()):
        raise ValueError("Fixed capacity formula changed nonvalid local v9 rows")
    rehash_ranking_sources(seal, seal_sha)
    expert = pd.DataFrame({
        "MVT_ID_mvt": template.MVT_ID_mvt.to_numpy(copy=True),
        "a_valid": valid,
        "old_v11_raw": old,
        "full_v13_raw": raw,
    })
    expert_sha = publish_new(expert_path,
                             lambda path: expert.to_parquet(path, index=False))
    result = pd.DataFrame({
        "MVT_ID_mvt": template.MVT_ID_mvt.to_numpy(copy=True),
        "TAXITIME_SEC_mvt": output,
    })
    result_sha = publish_new(output_path,
                             lambda path: result.to_parquet(path, index=False))
    readback = pd.read_parquet(output_path)
    if (list(readback) != required
            or not np.array_equal(readback.MVT_ID_mvt.to_numpy(),
                                  template.MVT_ID_mvt.to_numpy())
            or not np.array_equal(readback.TAXITIME_SEC_mvt.to_numpy(dtype=float), output)
            or not np.array_equal(output[~valid], current[~valid])):
        raise ValueError("Saved ranking ID order, formula or outside gate changed")
    rehash_ranking_sources(seal, seal_sha)
    manifest = {
        "schema_version": 1,
        "selected_route": "v13", "fixed_weight": WEIGHT,
        "rows": RANK_ROWS, "valid_aobt_rows": VALID_RANK_ROWS,
        "outside_valid_rows": 5_464,
        "missing_clock_preserved_rows": 4_907,
        "template_order_verified": True,
        "finite_nonnegative": True,
        "local_v9_sha256": V9_SHA256,
        "final_model_sha256": final_report["model_sha256"],
        "final_model_report_sha256": sha(full_report_path()),
        "compatibility_terminal_sha256": sha(compatibility_terminal_path()),
        "component_guard_terminal_sha256": sha(guard_terminal_path()),
        "ranking_seal_sha256": seal_sha,
        "ranking_expert_sha256": expert_sha,
        "predictions_sha256": result_sha,
        "prediction_bytes": output_path.stat().st_size,
        "uploaded": False,
    }
    write_json_new(manifest_path, manifest)
    rehash_ranking_sources(seal, seal_sha)
    return manifest


def synthetic() -> dict:
    """Small in-memory formula, branch, pairing, gate, and tamper checks."""
    base = np.array([10., 20., 30., 40.])
    old = np.array([12., np.nan, 30., np.nan])
    full = np.array([16., np.nan, 26., np.nan])
    valid = np.array([True, False, True, False])
    result = v13.fixed_replacement(base, old, full, valid)
    if not np.array_equal(result, np.array([12., 20., 28., 40.])):
        raise AssertionError("Fixed .5 ranking formula or outside-gate copy changed")
    for wrong in (np.array([True, True, True, False]),
                  np.array([True, False, True])):
        try:
            v13.fixed_replacement(base, old, full, wrong)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid or changed AOBT mask was accepted")
    if (branch_from_status(False, None, None) != "v9"
            or branch_from_status(True, False, None) != "v9"
            or branch_from_status(True, True, False) != "v9"
            or branch_from_status(True, True, True) != "geometry"):
        raise AssertionError("Terminal branch resolution changed")
    for missing in ((None, None, None), (True, None, None), (True, True, None)):
        try:
            branch_from_status(*missing)
        except ValueError:
            pass
        else:
            raise AssertionError("Missing required terminal was treated as a failure")
    left = pd.DataFrame({"MVT_ID_mvt": [1, 2], "target": [1., 2.]})
    right = pd.DataFrame({"MVT_ID_mvt": [2, 1], "target": [2., 1.]})
    if aligned_frame(left, right, "synthetic").MVT_ID_mvt.tolist() != [1, 2]:
        raise AssertionError("Exact ID alignment changed")
    for bad in (pd.Series([1, 1]), pd.Series([1, 3])):
        try:
            exact_ids(bad, left.MVT_ID_mvt, "synthetic")
        except ValueError:
            pass
        else:
            raise AssertionError("Duplicate or missing ID was accepted")
    times = pd.to_datetime(["2025-02-01T00:00:00Z", "2025-02-02T00:00:00Z",
                            "2025-08-01T00:00:00Z", "2025-08-02T00:00:00Z"])
    n = 1200
    paired = pd.DataFrame({
        "MVT_ID_mvt": np.arange(4 * n, dtype=np.int64),
        "target": np.full(4 * n, 10.),
        "MVT_TIME_UTC_mvt": times.repeat(n),
        "old_raw": np.full(4 * n, 20.),
        "full_raw": np.full(4 * n, 0.),
        "own_prefix_raw": np.full(4 * n, 8.),
    })
    score, output = score_guard(paired, True)
    if (not score["passed"] or score["bootstrap_seed"] != BOOTSTRAP_SEED
            or not np.array_equal(output.fixed_candidate_clipped.to_numpy(),
                                  np.full(4 * n, 10.))):
        raise AssertionError("Known matched capacity improvement failed")
    worse = paired.copy()
    worse.loc[worse.MVT_TIME_UTC_mvt.dt.month.eq(8), "full_raw"] = 60.
    if score_guard(worse, True)[0]["passed"]:
        raise AssertionError("Worse August was accepted")
    unavailable = paired.copy()
    unavailable["own_prefix_raw"] = np.nan
    if score_guard(unavailable, False)[0]["passed"]:
        raise AssertionError("Unavailable exact prefix was accepted")
    with tempfile.TemporaryDirectory(prefix="v13-capacity-synthetic-") as folder:
        root = Path(folder)
        path = root / "receipt.json"
        write_json_new(path, {"sealed": True})
        frozen = hashes({"receipt": path})
        try:
            write_json_new(path, {"sealed": False})
        except FileExistsError:
            pass
        else:
            raise AssertionError("Exclusive JSON receipt could be overwritten")
        path.write_text('{"sealed": false}', encoding="utf-8")
        try:
            check_hashes({"receipt": path}, frozen, "synthetic")
        except ValueError:
            pass
        else:
            raise AssertionError("Changed source receipt was accepted")
    return {"synthetic": "passed", "fixed_weight": WEIGHT,
            "reserved_months": list(HELDOUT), "bootstrap_seed": BOOTSTRAP_SEED,
            "real_data_read": False, "models_fitted": 0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=(
        "prepare", "compat-eval", "guard-prepare", "guardfit-comparator",
        "guardfit-replacement", "guard-eval", "finalprepare", "finalfit",
        "rankingseal", "predict", "synthetic"))
    parser.add_argument("--published-source-sha256", default="")
    args = parser.parse_args()
    if args.mode == "synthetic":
        report = synthetic()
    elif args.mode == "prepare":
        report = freeze_compatibility(args.published_source_sha256)
    elif args.mode == "compat-eval":
        report = compare_compatibility(args.published_source_sha256)
    elif args.mode == "guard-prepare":
        report = freeze_guard(args.published_source_sha256)
    elif args.mode == "guardfit-comparator":
        report = fit_guard_model("comparator", args.published_source_sha256)
    elif args.mode == "guardfit-replacement":
        report = fit_guard_model("replacement", args.published_source_sha256)
    elif args.mode == "guard-eval":
        report = evaluate_guard(args.published_source_sha256)
    elif args.mode == "finalprepare":
        report = prepare_final(args.published_source_sha256)
    elif args.mode == "finalfit":
        report = fit_final(args.published_source_sha256)
    elif args.mode == "rankingseal":
        report = seal_ranking(args.published_source_sha256)
    else:
        report = predict_ranking(args.published_source_sha256)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
