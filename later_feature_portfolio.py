"""Frozen local compatibility selector for the v10/v11 taxi-context experts.

This script only compares already-fitted, receipt-sealed 2025 OOF predictions.
It never fits a model, reads ranking values, runs the separate reserved-month
guard, or produces a ranking submission. The selected route is sealed before
that separate May/September guard is run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import v10_runway_taxi_expert as v10
import v11_taxi_interval_flow_expert as v11


ROOT = Path(__file__).resolve().parent
PROTOCOL = ROOT / "reports/later_feature_portfolio_protocol.json"
PROTOCOL_SHA256 = "c05422b1d010cb235d73c16d556613ca88c545cb6e6a81646da8dd6802160913"
OUT = ROOT / "artifacts/later-feature-portfolio"
CURRENT = ROOT / "artifacts/current-candidate"
V7 = ROOT / "artifacts/v7-runway-traffic"
CACHE = ROOT / "artifacts/baseline"
FOLDS = {"seasonal_jan_jul": (1, 7), "forward_nov_dec": (11, 12)}
ROUTES = ("current", "v10", "v11")
OOF_ROWS = 672_428
WEIGHTS = (0.0, 0.1, 0.25, 0.5, 1.0)
BOOTSTRAP_SEED = 20261014
ROUTE_CONFIG = {
    "v10": {"module": v10, "directory": ROOT / "artifacts/v10-runway-taxi",
            "builder": ROOT / "artifacts/v10-runway-arrival-taxi",
            "model_spec": ROOT / "reports/runway_taxi_model_spec_v10.json",
            "blend_column": "v10_blend"},
    "v11": {"module": v11, "directory": ROOT / "artifacts/v11-taxi-flow-expert",
            "builder": ROOT / "artifacts/v11-taxi-flow",
            "model_spec": ROOT / "reports/taxi_flow_model_spec_v11.json",
            "blend_column": "v11_blend"},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError as error:
        raise ValueError(f"Input/output path escapes the competition workspace: {path}") from error


def exact_ids(actual: pd.Series, expected: pd.Series, label: str) -> None:
    left, right = pd.Index(actual), pd.Index(expected)
    if (len(left) != len(right) or left.has_duplicates or right.has_duplicates
            or left.isna().any() or right.isna().any()
            or not left.isin(right).all() or not right.isin(left).all()):
        raise ValueError(f"{label}: exact unique ID coverage failed")


def align(frame: pd.DataFrame, ids: pd.Series, label: str) -> pd.DataFrame:
    exact_ids(frame.MVT_ID_mvt, ids, label)
    return frame.set_index("MVT_ID_mvt").loc[ids.to_numpy()].reset_index()


def same_metadata(left: pd.DataFrame, right: pd.DataFrame, label: str,
                  *, airport: bool = False, mask: bool = False) -> None:
    for field in ("target", "fold", "month"):
        if not np.array_equal(left[field].to_numpy(), right[field].to_numpy()):
            raise ValueError(f"{label}: {field} differs from common OOF")
    if not np.array_equal(pd.to_datetime(left.MVT_TIME_UTC_mvt, utc=True).to_numpy(),
                          pd.to_datetime(right.MVT_TIME_UTC_mvt, utc=True).to_numpy()):
        raise ValueError(f"{label}: UTC movement timestamps differ")
    if airport and not np.array_equal(left.airport.astype("string").to_numpy(),
                                      right.airport.astype("string").to_numpy()):
        raise ValueError(f"{label}: airports differ")
    if mask and not np.array_equal(left.a_valid.to_numpy(dtype=bool),
                                   right.a_valid.to_numpy(dtype=bool)):
        raise ValueError(f"{label}: valid-AOBT mask differs")


def rmse(target: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean((target - prediction) ** 2)))


def day_bootstrap(frame: pd.DataFrame, old: np.ndarray, new: np.ndarray,
                  seed: int = BOOTSTRAP_SEED) -> dict:
    """The predeclared paired UTC-day 1000-repeat RMSE-gain interval."""
    dates = pd.to_datetime(frame.MVT_TIME_UTC_mvt, utc=True).dt.floor("D")
    groups, days = pd.factorize(dates, sort=True)
    if len(days) == 0 or (groups < 0).any():
        raise ValueError("Paired UTC-day bootstrap has missing days")
    n = np.bincount(groups).astype(float)
    y = frame.target.to_numpy(dtype=float)
    a = np.bincount(groups, weights=(y - old) ** 2)
    b = np.bincount(groups, weights=(y - new) ** 2)
    samples = np.random.default_rng(seed).integers(
        0, len(days), size=(1000, len(days)))
    sampled_n = n[samples].sum(axis=1)
    gains = (np.sqrt(a[samples].sum(axis=1) / sampled_n)
             - np.sqrt(b[samples].sum(axis=1) / sampled_n))
    return {"days": len(days), "repeats": 1000,
            "gain_ci95_sec": np.quantile(gains, [.025, .975]).tolist(),
            "fraction_positive": float((gains > 0).mean()),
            "observed_gain_sec": rmse(y, old) - rmse(y, new)}


def frozen_file_hashes(mapping: dict[str, str]) -> dict[str, str]:
    if not isinstance(mapping, dict) or len(mapping) != 25:
        raise ValueError("Portfolio must bind the twenty-five published inputs")
    result = {}
    for name, expected in mapping.items():
        if (not isinstance(name, str) or name.startswith("/")
                or "\\" in name or ".." in Path(name).parts
                or not re.fullmatch(r"[0-9a-f]{64}", expected)):
            raise ValueError("Frozen portfolio mapping contains an invalid path/hash")
        path = ROOT / name
        if relative(path) != name or not path.is_file():
            raise ValueError(f"Frozen portfolio input path differs: {name}")
        result[name] = sha256(path)
    if result != mapping:
        changed = [name for name in mapping if result[name] != mapping[name]]
        raise ValueError(f"Frozen portfolio inputs changed: {changed}")
    return result


def verify_protocol() -> tuple[dict, str]:
    actual = sha256(PROTOCOL)
    if actual != PROTOCOL_SHA256:
        raise ValueError("Published later-feature portfolio protocol bytes changed")
    spec = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if (spec.get("compatibility_bootstrap") != {"repeats": 1000,
                                                 "seed": BOOTSTRAP_SEED}
            or spec.get("reserved_guard", {}).get("months") != [5, 9]
            or spec.get("alternatives") != [
                "Unchanged current policy", "v10 same-runway ARR taxi expert",
                "v11 released departure interval-flow expert"]):
        raise ValueError("Published portfolio rules differ from frozen implementation")
    frozen_file_hashes(spec["frozen_input_sha256"])
    return spec, actual


class Snapshot:
    def __init__(self, protocol: dict, source_sha: str):
        self.protocol = protocol
        self.source_sha = source_sha
        self.dynamic: dict[str, str] = {}

    def track(self, path: Path) -> str:
        name = relative(path)
        current = sha256(path)
        old = self.dynamic.setdefault(name, current)
        if old != current:
            raise ValueError(f"Input changed between reads: {name}")
        return current

    def json(self, path: Path) -> dict:
        before = self.track(path)
        result = json.loads(path.read_text(encoding="utf-8"))
        if sha256(path) != before:
            raise ValueError(f"JSON changed while reading: {path}")
        return result

    def parquet(self, path: Path, columns: list[str]) -> pd.DataFrame:
        before = self.track(path)
        result = pd.read_parquet(path, columns=columns)
        if sha256(path) != before:
            raise ValueError(f"Parquet changed while reading: {path}")
        return result

    def check(self) -> None:
        verify_protocol()
        if sha256(Path(__file__).resolve()) != self.source_sha:
            raise ValueError("Portfolio implementation source changed during comparison")
        for name, expected in self.dynamic.items():
            if sha256(ROOT / name) != expected:
                raise ValueError(f"Input changed after read: {name}")


def load_current(snapshot: Snapshot) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    selected_policy = snapshot.json(ROOT / "reports/current_selected_policy.json")
    report = snapshot.json(ROOT / "reports/current_composition_validation.json")
    composition = snapshot.json(CURRENT / "protocol.json")
    v7_report = snapshot.json(V7 / "validation.json")
    current_path = CURRENT / "validation_predictions.parquet"
    v7_path = V7 / "validation_predictions.parquet"
    if (selected_policy.get("selected_route") != "v7"
            or report.get("passed") is not True
            or report.get("rows_all_finite") != OOF_ROWS
            or report.get("valid_route", {}).get("active") != "v7"
            or report.get("missing_route", {}).get("active") is not True
            or report.get("missing_route", {}).get("weight") != 1.0
            or report.get("validation_predictions_sha256") != sha256(current_path)
            or report.get("protocol_sha256") != sha256(CURRENT / "protocol.json")
            or report.get("source_sha256") != composition.get("source_sha256")
            or v7_report.get("promoted") is not True
            or v7_report.get("validation_predictions_sha256") != sha256(v7_path)):
        raise ValueError("Current v7 plus guarded missing-clock policy is not frozen")
    current = snapshot.parquet(current_path, [
        "MVT_ID_mvt", "target", "fold", "month", "MVT_TIME_UTC_mvt",
        "a_valid", "missing_gate", "v7", "valid_component",
        "missing_component", "combined"])
    v7 = snapshot.parquet(v7_path, [
        "MVT_ID_mvt", "target", "fold", "month", "MVT_TIME_UTC_mvt",
        "a_valid", "airport", "candidate"])
    if (len(current) != OOF_ROWS or current.MVT_ID_mvt.isna().any()
            or current.MVT_ID_mvt.duplicated().any()
            or set(current.fold.unique()) != set(FOLDS)
            or not np.isfinite(current[["target", "v7", "combined"]]
                               .to_numpy(dtype=float)).all()
            or (current[["v7", "combined"]].to_numpy(dtype=float) < 0).any()):
        raise ValueError("Current all-finite OOF coverage or values differ")
    v7 = align(v7, current.MVT_ID_mvt, "v7 OOF")
    same_metadata(current, v7, "v7", mask=True)
    if not np.array_equal(current.v7.to_numpy(dtype=float),
                          v7.candidate.to_numpy(dtype=float)):
        raise ValueError("Current v7 prediction differs from accepted v7 OOF")
    for name, months in FOLDS.items():
        part = current.loc[current.fold.eq(name)]
        if (not part.month.isin(months).all()
                or not np.array_equal(
                    pd.to_datetime(part.MVT_TIME_UTC_mvt, utc=True)
                    .dt.month.to_numpy(), part.month.to_numpy())):
            raise ValueError(f"Current {name} month mapping changed")
    baseline_path = CACHE / "training_rows.parquet"
    features_path = CACHE / "features.parquet"
    if (report["source_sha256"]["baseline_rows"] != sha256(baseline_path)
            or report["source_sha256"]["baseline_features"] !=
               sha256(features_path)):
        raise ValueError("Baseline rows or missing-clock flags changed")
    base = snapshot.parquet(baseline_path,
                            ["MVT_ID_mvt", "target", "proxy", "month",
                             "airport", "time"])
    flags = snapshot.parquet(features_path,
                             ["AOBT_3_flt_missing", "LOBT_flt_missing"])
    if len(base) != len(flags):
        raise ValueError("Baseline rows and clock flags differ in length")
    base["nm_aobt_missing"] = flags.AOBT_3_flt_missing.to_numpy(dtype=bool)
    base["nm_lobt_missing"] = flags.LOBT_flt_missing.to_numpy(dtype=bool)
    base = base.loc[base.month.isin((1, 7, 11, 12)).to_numpy()
                    & np.isfinite(base.target.to_numpy(dtype=float))]
    base = align(base, current.MVT_ID_mvt, "baseline finite OOF")
    if (not np.array_equal(base.target.to_numpy(dtype=float),
                           current.target.to_numpy(dtype=float))
            or not np.array_equal(base.month.to_numpy(), current.month.to_numpy())
            or not np.array_equal(pd.to_datetime(base.time, utc=True).to_numpy(),
                                  pd.to_datetime(current.MVT_TIME_UTC_mvt,
                                                 utc=True).to_numpy())
            or not np.array_equal(base.airport.astype("string").to_numpy(),
                                  v7.airport.astype("string").to_numpy())):
        raise ValueError("Common OOF labels, month, airport or UTC times differ")
    proxy = base.proxy.to_numpy(dtype=float)
    valid = np.isfinite(proxy) & (proxy >= 0) & (proxy <= 7200)
    missing = (~np.isfinite(proxy)
               & base.nm_aobt_missing.to_numpy(dtype=bool)
               & base.nm_lobt_missing.to_numpy(dtype=bool)
               & ~base.airport.eq("LIRF").to_numpy(dtype=bool))
    current_value = current.combined.to_numpy(dtype=float)
    v7_value = current.v7.to_numpy(dtype=float)
    if (np.any(valid & missing)
            or not np.array_equal(current.a_valid.to_numpy(dtype=bool), valid)
            or not np.array_equal(current.missing_gate.to_numpy(dtype=bool), missing)
            or not np.array_equal(current.valid_component.to_numpy(dtype=float),
                                  v7_value)
            or not np.array_equal(current_value[~missing], v7_value[~missing])
            or not np.array_equal(current_value[missing],
                                  current.missing_component.to_numpy(dtype=float)[missing])
            or int(valid.sum()) != report["n_valid"]
            or int(missing.sum()) != report["n_missing_clock"]):
        raise ValueError("Current valid/missing masks or unchanged rows differ")
    current["airport"] = v7.airport.to_numpy(copy=True)
    return current, valid, missing


def alternative_args(route: str) -> argparse.Namespace:
    config = ROUTE_CONFIG[route]
    return argparse.Namespace(
        data_dir=ROOT / "data", cache_dir=CACHE,
        weather_file=ROOT / "data/external/weather.parquet",
        arrival_dir=ROOT / "artifacts/v5-arrival-clean",
        neighbour_dir=ROOT / "artifacts/v6-neighbour",
        runway_dir=ROOT / "artifacts/v6-runway-arrival",
        taxi_dir=config["builder"], v7_dir=V7,
        v6_dir=ROOT / "artifacts/v6-deep-arrival",
        output_dir=config["directory"],
        iterations=10000, depth=10, threads=2, min_free_gib=10.0)


def track_original_inputs(route: str, args: argparse.Namespace,
                          frozen: dict, snapshot: Snapshot) -> None:
    """Bind every original fit source to the final and downstream snapshots."""
    module = ROUTE_CONFIG[route]["module"]
    if route == "v10":
        raw = module._training_files(args.data_dir)
        paths = module.input_paths(args)
    else:
        raw, paths = module.source_inventory(args)
    if (len(raw) != 12 or len({path.name for path in raw}) != 12
            or set(paths) != set(frozen["input_sha256"])
            or {path.name for path in raw} != set(frozen["raw_training_sha256"])):
        raise ValueError(f"{route} original source inventory differs from frozen protocol")
    for name, path in paths.items():
        if snapshot.track(path) != frozen["input_sha256"][name]:
            raise ValueError(f"{route} original input changed: {name}")
    for path in raw:
        if snapshot.track(path) != frozen["raw_training_sha256"][path.name]:
            raise ValueError(f"{route} original training source changed: {path.name}")


def original_terminal(route: str, snapshot: Snapshot,
                      current: pd.DataFrame) -> tuple[dict, dict, pd.DataFrame | None]:
    """Verify both saved fold receipts; fresh remains unneeded if original fails."""
    config = ROUTE_CONFIG[route]
    module = config["module"]
    args = alternative_args(route)
    root = config["directory"]
    protocol_path = root / "protocol.json"
    validation_path = root / "validation.json"
    oof_path = root / "validation_predictions.parquet"
    required = [protocol_path, validation_path, oof_path,
                root / "frozen_v7_oof_reference.parquet"]
    for name in FOLDS:
        required += [root / f"{name}_oof.parquet", root / f"{name}.cbm",
                     root / f"{name}_validation.json",
                     root / f"{name}_provenance.json"]
    if any(not path.is_file() for path in required):
        missing = [relative(path) for path in required if not path.is_file()]
        raise FileNotFoundError(f"{route} original fit/receipt is pending: {missing}")
    frozen = snapshot.json(protocol_path)
    track_original_inputs(route, args, frozen, snapshot)
    for path in required:
        snapshot.track(path)
    validation = snapshot.json(validation_path)
    if route == "v10":
        module.assert_fixed_args(args)
    else:
        module.require_frozen_settings(args)
    if frozen.get("spec") != module.protocol_spec():
        raise ValueError(f"{route} frozen model specification changed")
    published_spec = snapshot.json(config["model_spec"])
    for key in ("base", "depth", "max_iterations", "random_seed",
                "additional_features", "training_labels", "scoring_labels"):
        if frozen["spec"]["architecture"].get(key) != published_spec["architecture"].get(key):
            raise ValueError(f"{route} model architecture differs from published spec: {key}")
    for key in ("existing_folds", "fresh_matched_audit"):
        if frozen["spec"].get(key) != published_spec.get(key):
            raise ValueError(f"{route} validation or fresh audit policy changed: {key}")
    if route == "v10":
        module.assert_frozen_inputs(args, frozen, snapshot.track(protocol_path))
    else:
        module.verify_source_snapshot(args, frozen, snapshot.track(protocol_path))
    if (validation.get("protocol_sha256") != snapshot.track(protocol_path)
            or validation.get("frozen_reference_sha256") !=
               frozen["references"]["frozen_reference_sha256"]
            or validation.get("validation_predictions_sha256") !=
               snapshot.track(oof_path)):
        raise ValueError(f"{route} original validation source/report hash changed")
    v7_ref = current[["MVT_ID_mvt", "target", "fold", "month",
                      "MVT_TIME_UTC_mvt", "a_valid", "v7"]].rename(
                          columns={"v7": "selected"})
    fold_data: dict[str, tuple[pd.DataFrame, np.ndarray, np.ndarray]] = {}
    for name in FOLDS:
        held = v7_ref.loc[v7_ref.fold.eq(name)]
        module.verify_fold_provenance(args, name, held)
        entry = validation["folds"][name]
        if (entry["expert_oof_sha256"] != snapshot.track(root / f"{name}_oof.parquet")
                or entry["model_sha256"] != snapshot.track(root / f"{name}.cbm")
                or entry["fit_report_sha256"] != snapshot.track(
                    root / f"{name}_validation.json")
                or entry["fold_provenance_sha256"] != snapshot.track(
                    root / f"{name}_provenance.json")
                or entry["rows_all_finite"] != len(held)
                or entry["rows_valid_aobt"] != int(held.a_valid.sum())):
            raise ValueError(f"{route} {name} fold report/receipt differs")
        expert = snapshot.parquet(root / f"{name}_oof.parquet",
                                  ["MVT_ID_mvt", "expert"])
        valid_held = held.loc[held.a_valid]
        expert = align(expert, valid_held.MVT_ID_mvt,
                       f"{route} {name} sealed expert")
        raw = expert.expert.to_numpy(dtype=float)
        if not np.isfinite(raw).all():
            raise ValueError(f"{route} {name} sealed expert is nonfinite")
        base = held.selected.to_numpy(dtype=float)
        alternate = base.copy()
        alternate[held.a_valid.to_numpy(dtype=bool)] = raw
        target = held.target.to_numpy(dtype=float)
        scores = {str(weight): rmse(target, np.maximum(
            base + weight * (alternate - base), 0)) for weight in WEIGHTS}
        if scores != entry["scores_all_finite_rmse"]:
            raise ValueError(f"{route} {name} reported weight scores differ")
        fold_data[name] = (held, base, alternate)
    seasonal = validation["folds"]["seasonal_jan_jul"]["scores_all_finite_rmse"]
    selected = min(WEIGHTS, key=lambda value: (seasonal[str(value)], value))
    if validation.get("selected_weight") != selected:
        raise ValueError(f"{route} selected weight differs from January/July rule")
    for name, (held, base, alternate) in fold_data.items():
        candidate = np.maximum(base + selected * (alternate - base), 0)
        interval = day_bootstrap(held, base, candidate,
                                 seed=module.BOOTSTRAP_SEED)
        if interval != validation["folds"][name]["day_bootstrap"]:
            raise ValueError(f"{route} {name} reported day interval differs")
    passed = bool(selected > 0 and all(
        validation["folds"][name]["scores_all_finite_rmse"][str(selected)]
        < validation["folds"][name]["scores_all_finite_rmse"]["0.0"]
        and validation["folds"][name]["day_bootstrap"]["gain_ci95_sec"][0] > 0
        for name in FOLDS))
    if validation.get("existing_folds_passed") is not passed:
        raise ValueError(f"{route} original pass flag conflicts with fixed gate")
    status = {"route": route, "original_passed": passed,
              "fresh_passed": None, "compatibility_passed": False,
              "selected_weight": float(selected),
              "original_validation_sha256": snapshot.track(validation_path),
              "original_protocol_sha256": snapshot.track(protocol_path),
              "reason": "original Jan/Jul or Nov/Dec fixed gate failed" if not passed else None}
    if not passed:
        return status, validation, None
    candidate = snapshot.parquet(oof_path, [
        "MVT_ID_mvt", "target", "fold", "month", "MVT_TIME_UTC_mvt",
        "a_valid", "airport", "selected", "expert", "candidate"])
    return status, validation, candidate


def fresh_terminal(route: str, snapshot: Snapshot, status: dict,
                   current: pd.DataFrame) -> bool:
    """Verify saved April/October evidence without invoking mutable evaluate()."""
    config = ROUTE_CONFIG[route]
    module = config["module"]
    args = alternative_args(route)
    root = config["directory"]
    fresh = root / "fresh_new"
    audit_path = root / "fresh_audit.json"
    paired_path = root / "fresh_audit_predictions.parquet"
    expert_path = fresh / "fresh_apr_oct_oof.parquet"
    receipt_path = fresh / "fresh_apr_oct_provenance.json"
    required = [audit_path, paired_path, expert_path, receipt_path,
                fresh / "fresh_apr_oct.cbm", fresh / "fresh_apr_oct_validation.json",
                V7 / "fresh_audit_predictions.parquet", V7 / "fresh_audit.json",
                V7 / "fresh_new/fresh_apr_oct.cbm",
                args.v6_dir / "fresh_new/fresh_apr_oct.cbm"]
    if any(not path.is_file() for path in required):
        missing = [relative(path) for path in required if not path.is_file()]
        raise FileNotFoundError(f"{route} fresh fit/receipt is pending: {missing}")
    for path in required:
        snapshot.track(path)
    audit = snapshot.json(audit_path)
    validation = snapshot.json(root / "validation.json")
    if (audit.get("months") != [4, 10]
            or audit.get("weight") != status["selected_weight"]
            or audit.get("coverage_verified") is not True
            or audit.get("protocol_sha256") != status["original_protocol_sha256"]
            or audit.get("fresh_fold_provenance_sha256") != snapshot.track(receipt_path)
            or audit.get("v7_paired_sha256") != snapshot.track(
                V7 / "fresh_audit_predictions.parquet")
            or audit.get("v7_fresh_report_sha256") != snapshot.track(
                V7 / "fresh_audit.json")
            or audit.get("v7_fresh_model_sha256") != snapshot.track(
                V7 / "fresh_new/fresh_apr_oct.cbm")
            or audit.get("v6_fresh_model_sha256") != snapshot.track(
                args.v6_dir / "fresh_new/fresh_apr_oct.cbm")
            or audit.get("v11_expert_oof_sha256" if route == "v11"
                         else "v10_expert_oof_sha256") != snapshot.track(expert_path)
            or audit.get("v11_model_sha256" if route == "v11"
                         else "v10_model_sha256") != snapshot.track(
                             fresh / "fresh_apr_oct.cbm")):
        raise ValueError(f"{route} fresh audit/receipt source changed")
    frame = snapshot.parquet(paired_path, [
        "MVT_ID_mvt", "target", "fold", "month", "MVT_TIME_UTC_mvt",
        "selected", "expert", config["blend_column"]])
    prior = snapshot.parquet(V7 / "fresh_audit_predictions.parquet", [
        "MVT_ID_mvt", "target", "MVT_TIME_UTC_mvt", "candidate"])
    prior = align(prior, frame.MVT_ID_mvt, f"{route} fresh v7 comparator")
    if (frame.MVT_ID_mvt.isna().any() or frame.MVT_ID_mvt.duplicated().any()
            or not frame.fold.eq("fresh_apr_oct").all()
            or not frame.month.isin((4, 10)).all()
            or not np.array_equal(pd.to_datetime(frame.MVT_TIME_UTC_mvt,
                                                 utc=True).dt.month.to_numpy(),
                                  frame.month.to_numpy())
            or not np.array_equal(frame.target.to_numpy(dtype=float),
                                  prior.target.to_numpy(dtype=float))
            or not np.array_equal(frame.selected.to_numpy(dtype=float),
                                  prior.candidate.to_numpy(dtype=float))
            or not np.array_equal(pd.to_datetime(frame.MVT_TIME_UTC_mvt,
                                                 utc=True).to_numpy(),
                                  pd.to_datetime(prior.MVT_TIME_UTC_mvt,
                                                 utc=True).to_numpy())):
        raise ValueError(f"{route} fresh matched IDs/labels/times differ")
    fresh_args = argparse.Namespace(**vars(args))
    fresh_args.output_dir = fresh
    module.verify_fold_provenance(fresh_args, "fresh_apr_oct", frame)
    expert = snapshot.parquet(expert_path, ["MVT_ID_mvt", "expert"])
    expert = align(expert, frame.MVT_ID_mvt, f"{route} fresh expert")
    y = frame.target.to_numpy(dtype=float)
    base = frame.selected.to_numpy(dtype=float)
    raw = expert.expert.to_numpy(dtype=float)
    replacement = np.maximum(base + status["selected_weight"] * (raw - base), 0)
    if (audit.get("rows_valid_aobt_finite") != len(frame)
            or audit.get("predictions_sha256") != snapshot.track(paired_path)
            or not np.isfinite(frame[["target", "selected", "expert",
                                     config["blend_column"]]].to_numpy(dtype=float)).all()
            or not np.array_equal(frame.expert.to_numpy(dtype=float), raw)
            or not np.array_equal(frame[config["blend_column"]].to_numpy(dtype=float),
                                  replacement)):
        raise ValueError(f"{route} fresh fixed-weight prediction formula changed")
    month = frame.month.to_numpy(dtype=int)
    scores = {str(m): {"n": int((month == m).sum()),
                       "v7_rmse": rmse(y[month == m], base[month == m]),
                       f"{route}_blend_rmse": rmse(y[month == m],
                                                     replacement[month == m])}
              for m in (4, 10)}
    bootstrap = day_bootstrap(frame, base, replacement,
                              seed=module.BOOTSTRAP_SEED)
    passed = (all(scores[str(m)][f"{route}_blend_rmse"] <
                  scores[str(m)]["v7_rmse"] for m in (4, 10))
              and bootstrap["gain_ci95_sec"][0] > 0)
    if (audit.get("scores") != scores or audit.get("bootstrap") != bootstrap
            or audit.get("passed") is not bool(passed)
            or validation.get("fresh_audit_passed") is not bool(passed)
            or validation.get("fresh_audit_sha256") != snapshot.track(audit_path)):
        raise ValueError(f"{route} fresh scores or terminal gate changed")
    status["fresh_passed"] = bool(passed)
    status["fresh_audit_sha256"] = snapshot.track(audit_path)
    if not passed:
        status["reason"] = "matched April/October fixed audit failed"
    return bool(passed)


def compose_alternative(current: pd.DataFrame, valid: np.ndarray,
                        alternate: pd.DataFrame, weight: float,
                        route: str) -> np.ndarray:
    """Apply only the preselected valid-AOBT expert; copy all other rows."""
    frame = align(alternate, current.MVT_ID_mvt, f"{route} original OOF")
    same_metadata(current, frame, route, airport=True, mask=True)
    base = current.v7.to_numpy(dtype=float)
    old = current.combined.to_numpy(dtype=float)
    expert = frame.expert.to_numpy(dtype=float)
    frozen_candidate = frame.candidate.to_numpy(dtype=float)
    if (not np.array_equal(frame.selected.to_numpy(dtype=float), base)
            or not np.array_equal(np.isfinite(expert), valid)
            or not np.isfinite(frozen_candidate).all()
            or not np.array_equal(frozen_candidate[~valid], base[~valid])):
        raise ValueError(f"{route} original OOF reference, expert mask or fallback changed")
    replacement = np.maximum(base[valid] + weight * (expert[valid] - base[valid]), 0)
    if not np.array_equal(frozen_candidate[valid], replacement):
        raise ValueError(f"{route} original OOF changed fixed Jan/Jul blend")
    result = old.copy()
    result[valid] = replacement
    if (not np.array_equal(result[~valid], old[~valid])
            or not np.isfinite(result).all() or (result < 0).any()):
        raise ValueError(f"{route} altered missing/outside rows or produced invalid values")
    return result


def score_compatibility(current: pd.DataFrame,
                        candidate: np.ndarray) -> tuple[dict, bool]:
    old = current.combined.to_numpy(dtype=float)
    scores = {}
    for name, months in FOLDS.items():
        mask = current.fold.eq(name).to_numpy()
        part = current.loc[mask]
        if not part.month.isin(months).all():
            raise ValueError(f"{name} common OOF month map changed")
        target = part.target.to_numpy(dtype=float)
        before = rmse(target, old[mask])
        after = rmse(target, candidate[mask])
        interval = day_bootstrap(part, old[mask], candidate[mask])
        scores[name] = {"rows": len(part), "current_rmse": before,
                        "candidate_rmse": after, "day_bootstrap": interval,
                        "passed": bool(after < before and
                                       interval["gain_ci95_sec"][0] > 0)}
    return scores, all(item["passed"] for item in scores.values())


def _write_json_exclusive(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as output:
        json.dump(value, output, indent=2)
        output.write("\n")


def _write_parquet_exclusive(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".later_feature_", dir=path.parent) as work:
        stage = Path(work) / "candidate.parquet"
        frame.to_parquet(stage, index=False)
        os.link(stage, path)


def select_route(scores: dict[str, dict]) -> str:
    """Use common-universe Jan/Jul RMSE, with published current/v10/v11 tie order."""
    eligible = [name for name in ROUTES if scores[name]["eligible"]]
    if "current" not in eligible:
        raise ValueError("Unchanged current policy must remain eligible")
    return min(eligible, key=lambda name: (
        scores[name]["seasonal_all_finite_rmse"], ROUTES.index(name)))


def evaluate_and_select(expected_source_sha256: str,
                        output_dir: Path = OUT) -> dict:
    if not re.fullmatch(r"[0-9a-f]{64}", expected_source_sha256):
        raise ValueError("Expected published implementation SHA256 is required")
    source_sha = sha256(Path(__file__).resolve())
    if source_sha != expected_source_sha256:
        raise ValueError("Published implementation source SHA256 differs")
    output_dir = output_dir.resolve()
    relative(output_dir)
    outputs = (output_dir / "evaluation_predictions.parquet",
               output_dir / "evaluation.json", output_dir / "selection.json")
    if any(path.exists() for path in outputs):
        raise FileExistsError("Later-feature portfolio output already exists; no overwrite")
    protocol, protocol_sha = verify_protocol()
    snapshot = Snapshot(protocol, source_sha)
    current, valid, missing = load_current(snapshot)
    y = current.target.to_numpy(dtype=float)
    baseline = current.combined.to_numpy(dtype=float)
    seasonal_mask = current.fold.eq("seasonal_jan_jul").to_numpy()
    forward_mask = current.fold.eq("forward_nov_dec").to_numpy()
    terminal = {}
    candidate_values: dict[str, np.ndarray | None] = {"v10": None, "v11": None}
    scores = {"current": {
        "eligible": True,
        "seasonal_all_finite_rmse": rmse(y[seasonal_mask], baseline[seasonal_mask]),
        "forward_all_finite_rmse": rmse(y[forward_mask], baseline[forward_mask]),
        "selected_weight": 0.0}}
    for route in ("v10", "v11"):
        status, _, original = original_terminal(route, snapshot, current)
        terminal[route] = status
        if not status["original_passed"]:
            scores[route] = {"eligible": False, "reason": status["reason"]}
            continue
        if not fresh_terminal(route, snapshot, status, current):
            scores[route] = {"eligible": False, "reason": status["reason"]}
            continue
        if original is None:
            raise AssertionError("Terminal original pass has no sealed common OOF")
        candidate = compose_alternative(current, valid, original,
                                        status["selected_weight"], route)
        fold_scores, passed = score_compatibility(current, candidate)
        status["compatibility_passed"] = bool(passed)
        status["compatibility_scores"] = fold_scores
        if not passed:
            status["reason"] = "fixed-weight compatibility RMSE/day-CI gate failed"
            scores[route] = {"eligible": False, "reason": status["reason"],
                             "compatibility_scores": fold_scores}
            continue
        candidate_values[route] = candidate
        scores[route] = {
            "eligible": True, "selected_weight": status["selected_weight"],
            "seasonal_all_finite_rmse": fold_scores["seasonal_jan_jul"]["candidate_rmse"],
            "forward_all_finite_rmse": fold_scores["forward_nov_dec"]["candidate_rmse"],
            "compatibility_scores": fold_scores}
    chosen = select_route(scores)
    chosen_weight = float(scores[chosen]["selected_weight"])
    selected = (baseline if chosen == "current" else candidate_values[chosen])
    if selected is None or not np.isfinite(selected).all() or (selected < 0).any():
        raise ValueError("Selected compatible OOF is incomplete or invalid")
    if (not np.array_equal(selected[~valid], baseline[~valid])
            or not np.array_equal(selected[missing], baseline[missing])):
        raise ValueError("Selected route changed the guarded missing-clock policy")
    frame = pd.DataFrame({
        "MVT_ID_mvt": current.MVT_ID_mvt,
        "target": current.target,
        "fold": current.fold,
        "month": current.month,
        "MVT_TIME_UTC_mvt": current.MVT_TIME_UTC_mvt,
        "a_valid": valid, "missing_gate": missing,
        "current": baseline,
        "v10": (candidate_values["v10"] if candidate_values["v10"] is not None
                else np.full(len(current), np.nan)),
        "v11": (candidate_values["v11"] if candidate_values["v11"] is not None
                else np.full(len(current), np.nan)),
        "selected": selected,
    })
    snapshot.check()
    _write_parquet_exclusive(outputs[0], frame)
    snapshot.check()
    evaluation = {
        "protocol_sha256": protocol_sha,
        "implementation_sha256": source_sha,
        "frozen_input_sha256": protocol["frozen_input_sha256"],
        "dynamic_input_sha256": snapshot.dynamic,
        "rows_all_finite": len(current),
        "valid_aobt_rows": int(valid.sum()),
        "missing_clock_rows": int(missing.sum()),
        "months_scored": [1, 7, 11, 12],
        "fresh_eligibility_months": [4, 10],
        "reserved_months_unscored": [5, 9],
        "february_august_not_used_for_selection": True,
        "scores": scores,
        "terminal_routes": terminal,
        "selected_route": chosen,
        "selected_weight": chosen_weight,
        "evaluation_predictions_sha256": sha256(outputs[0]),
        "decision": "sealed_for_separate_may_september_guard",
        "ranking_authorized": False,
    }
    _write_json_exclusive(outputs[1], evaluation)
    snapshot.check()
    selection = {
        "selected_route": chosen,
        "selected_weight": chosen_weight,
        "tie_order": list(ROUTES),
        "portfolio_protocol_sha256": protocol_sha,
        "portfolio_source_sha256": source_sha,
        "evaluation_sha256": sha256(outputs[1]),
        "evaluation_predictions_sha256": sha256(outputs[0]),
        "current_oof_sha256": protocol["frozen_input_sha256"][
            "artifacts/current-candidate/validation_predictions.parquet"],
        "selected_original_validation_sha256": (
            terminal[chosen]["original_validation_sha256"]
            if chosen != "current" else None),
        "selected_fresh_audit_sha256": (
            terminal[chosen]["fresh_audit_sha256"]
            if chosen != "current" else None),
        "terminal_routes": terminal,
        "input_sha256": snapshot.dynamic,
        "reserved_months_unscored": [5, 9],
        "guard_status": "pending_separate_frozen_may_september_guard",
        "ranking_authorized": False,
    }
    snapshot.check()
    _write_json_exclusive(outputs[2], selection)
    return selection


def require_selection(output_dir: Path = OUT,
                      expected_source_sha256: str | None = None) -> dict:
    """Read-only downstream gate for a published, immutable route decision."""
    protocol, protocol_sha = verify_protocol()
    output_dir = output_dir.resolve()
    relative(output_dir)
    source_sha = sha256(Path(__file__).resolve())
    if (expected_source_sha256 is not None
            and source_sha != expected_source_sha256):
        raise ValueError("Published selector source changed")
    evaluation_path = output_dir / "evaluation.json"
    prediction_path = output_dir / "evaluation_predictions.parquet"
    selection_path = output_dir / "selection.json"
    selection_sha = sha256(selection_path)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    chosen = selection.get("selected_route")
    if (chosen not in ROUTES
            or selection.get("selected_weight") !=
               evaluation["scores"][chosen]["selected_weight"]
            or selection.get("tie_order") != list(ROUTES)
            or selection.get("portfolio_protocol_sha256") != protocol_sha
            or selection.get("portfolio_source_sha256") != source_sha
            or selection.get("evaluation_sha256") != sha256(evaluation_path)
            or selection.get("evaluation_predictions_sha256") !=
               sha256(prediction_path)
            or selection.get("current_oof_sha256") !=
               protocol["frozen_input_sha256"][
                   "artifacts/current-candidate/validation_predictions.parquet"]
            or selection.get("terminal_routes") != evaluation.get("terminal_routes")
            or selection.get("input_sha256") != evaluation.get("dynamic_input_sha256")
            or selection.get("reserved_months_unscored") != [5, 9]
            or selection.get("guard_status") !=
               "pending_separate_frozen_may_september_guard"
            or selection.get("ranking_authorized") is not False
            or evaluation.get("protocol_sha256") != protocol_sha
            or evaluation.get("implementation_sha256") != source_sha
            or evaluation.get("frozen_input_sha256") !=
               protocol["frozen_input_sha256"]
            or evaluation.get("evaluation_predictions_sha256") !=
               sha256(prediction_path)
            or evaluation.get("rows_all_finite") != OOF_ROWS
            or evaluation.get("months_scored") != [1, 7, 11, 12]
            or evaluation.get("reserved_months_unscored") != [5, 9]
            or evaluation.get("decision") !=
               "sealed_for_separate_may_september_guard"
            or evaluation.get("ranking_authorized") is not False
            or select_route(evaluation["scores"]) != chosen):
        raise ValueError("Sealed later-feature route/report/OOF contract changed")
    terminal = evaluation["terminal_routes"]
    if set(terminal) != {"v10", "v11"}:
        raise ValueError("Both later-feature alternatives must have terminal outcomes")
    for route in ("v10", "v11"):
        item = terminal[route]
        score = evaluation["scores"].get(route)
        if (item.get("route") != route
                or not isinstance(item.get("original_passed"), bool)
                or not isinstance(item.get("compatibility_passed"), bool)
                or item.get("fresh_passed") not in (None, True, False)
                or score is None
                or score.get("eligible") is not bool(
                    item["original_passed"] and item["fresh_passed"]
                    and item["compatibility_passed"])):
            raise ValueError(f"{route} terminal status differs from eligibility")
        if (item["original_passed"] and not isinstance(item.get("original_validation_sha256"), str)):
            raise ValueError(f"{route} original audit hash is missing")
        if item["original_passed"] and item["fresh_passed"] is None:
            raise ValueError(f"{route} fresh audit is still pending")
        if not item["original_passed"] and item["fresh_passed"] is not None:
            raise ValueError(f"{route} fresh audit followed a rejected original fit")
        if item["fresh_passed"] and not isinstance(item.get("fresh_audit_sha256"), str):
            raise ValueError(f"{route} fresh audit hash is missing")
        if item["compatibility_passed"] and not item["fresh_passed"]:
            raise ValueError(f"{route} compatibility lacks fresh eligibility")
        if score["eligible"] and (score.get("selected_weight") !=
                                  item.get("selected_weight")):
            raise ValueError(f"{route} eligible weight changed")
    if chosen == "current":
        if (selection.get("selected_original_validation_sha256") is not None
                or selection.get("selected_fresh_audit_sha256") is not None):
            raise ValueError("Unchanged route must have no replacement model audit")
    else:
        item = terminal[chosen]
        if (item.get("original_passed") is not True
                or item.get("fresh_passed") is not True
                or item.get("compatibility_passed") is not True
                or selection.get("selected_original_validation_sha256") !=
                   item["original_validation_sha256"]
                or selection.get("selected_fresh_audit_sha256") !=
                   item["fresh_audit_sha256"]):
            raise ValueError("Selected replacement lacks all local terminal gates")
    for name, expected in evaluation["dynamic_input_sha256"].items():
        path = ROOT / name
        if relative(path) != name or sha256(path) != expected:
            raise ValueError(f"Selected route input changed after freezing: {name}")
    if (sha256(evaluation_path) != selection["evaluation_sha256"]
            or sha256(prediction_path) != selection["evaluation_predictions_sha256"]
            or sha256(selection_path) != selection_sha):
        raise ValueError("Sealed portfolio changed during downstream verification")
    return selection


def show_contract() -> dict:
    return {"portfolio_protocol_sha256": PROTOCOL_SHA256,
            "input_binding_count": 25,
            "routes": list(ROUTES),
            "folds": {name: list(months) for name, months in FOLDS.items()},
            "fresh_eligibility_months": [4, 10],
            "bootstrap": {"repeats": 1000, "seed": BOOTSTRAP_SEED},
            "selected_output": relative(OUT / "selection.json"),
            "forbidden_modes": ["may-september-guard", "fit-final", "ranking"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("show-contract", "evaluate-select",
                                           "may-september-guard", "fit-final",
                                           "ranking"), default="show-contract")
    parser.add_argument("--expected-source-sha256", type=str)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    if args.mode == "show-contract":
        print(json.dumps(show_contract(), indent=2))
    elif args.mode == "evaluate-select":
        if args.expected_source_sha256 is None:
            parser.error("--expected-source-sha256 must pin the published selector source")
        print(json.dumps(evaluate_and_select(args.expected_source_sha256,
                                             args.output_dir), indent=2))
    else:
        raise RuntimeError("May/September guard, final fit and ranking require a separately frozen implementation")


if __name__ == "__main__":
    main()
