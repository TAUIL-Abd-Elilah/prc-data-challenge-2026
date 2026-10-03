"""Prospective v12 comparison of 16 label-free runway geometry covariates.

The reference is the sealed v9 policy, including its guarded missing-clock
route. This module only compares an added 200-feature residual expert on 2025
held-out months. It cannot fit a final model or predict ranking rows until a
separately predeclared February/August component guard is implemented.
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
from catboost import CatBoostRegressor

import airport_models
import catboost_expert
import deep_arrival_expert as arrival
import deep_timestamp_expert as deep
import proxy_neighbour_expert
import runway_arrival_features
import solution
import later_feature_final as final
import later_feature_portfolio as portfolio
import runway_geometry_features as geometry
import traffic_deep_expert as traffic
import v11_taxi_interval_flow_expert as v11
import v4_reference
import weather_model
from solution import _training_files


FOLDS = deep.FOLDS
WEIGHTS = (0.0, 0.1, 0.25, 0.5, 1.0)
FRESH_MONTHS = (4, 10)
REFERENCE_COLUMNS = ("MVT_ID_mvt", "target", "fold", "month",
                     "MVT_TIME_UTC_mvt", "a_valid", "selected")
EXPECTED_OOF_ROWS = 672428
EXPECTED_FRESH_ROWS = 357813
EXPECTED_TRAINING_ROWS = 2085047
BOOTSTRAP_SEED = 20261015
GEOMETRY_FEATURES = geometry.FEATURES
EXPECTED_GEOMETRY_FEATURES = (
    "rgeom_heading_sin", "rgeom_heading_cos", "rgeom_strip_length_m",
    "rgeom_headwind_mps", "rgeom_tailwind_positive_mps", "rgeom_crosswind_abs_mps",
    "rgeom_intersecting_other_strip_count", "rgeom_near_parallel_other_strip_count",
    "rgeom_intersecting_arr_past15_count", "rgeom_intersecting_arr_past60_count",
    "rgeom_intersecting_dep_past15_count", "rgeom_intersecting_dep_past60_count",
    "rgeom_near_parallel_arr_past15_count", "rgeom_near_parallel_arr_past60_count",
    "rgeom_near_parallel_dep_past15_count", "rgeom_near_parallel_dep_past60_count",
)
EXPECTED_CATBOOST_PARAMS = v11.EXPECTED_CATBOOST_PARAMS
EXPECTED_PORTFOLIO_SOURCE_SHA256 = "f0dde94e95479b0e0dae00d9d4b37b3437bcc7c8804f4dfd912cba2f78568a0d"
EXPECTED_V9_OOF_SHA256 = "2626c43410bc6c03c8cc3a91855bd9c8dbe3b117bb181903bf4afd6378a466d6"
EXPECTED_V9_SUBMISSION_SHA256 = "bc465ae7ff48deac5f93cd449a3799fee1a361a8021ec2ca03ff70baccc0f417"
EXPECTED_BUILDER_SHA256 = "2e4429306cfabad8c6b4ee7990718e4350fb952daf08c325367a48ff381c1059"
EXPECTED_GEOMETRY_PROTOCOL_SHA256 = "a901a9d442ff01b0f8e1c5931cd1ea56c4246f89bfc12d4b8f35ed966089d502"
EXPECTED_GEOMETRY_TRAIN_SHA256 = "8cab44bb942976d12c2c75407c1aee4200a0087a3fd19da5a1924916bdc47005"
EXPECTED_GEOMETRY_RANK_SHA256 = "b790d2afbc9e0964bc0c8df71d3c41d18edfa447e6f77e7c691a90677281315c"
SPEC_PATH = Path(__file__).resolve().parent / "reports/runway_geometry_model_spec_v12.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_new(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as output:
        json.dump(value, output, indent=2, allow_nan=False)
        output.write("\n")


def write_parquet_new(path: Path, frame: pd.DataFrame) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".v12_", dir=path.parent) as work:
        staged = Path(work) / path.name
        frame.to_parquet(staged, index=False)
        digest = sha256(staged)
        os.link(staged, path)
    if sha256(path) != digest:
        raise ValueError("Published v12 Parquet bytes changed")
    return digest


def exact_ids(actual: pd.Series, expected: pd.Series, label: str) -> None:
    left, right = pd.Index(actual), pd.Index(expected)
    if (len(left) != len(right) or left.has_duplicates or right.has_duplicates
            or left.isna().any() or right.isna().any()
            or not left.isin(right).all() or not right.isin(left).all()):
        raise ValueError(f"{label}: exact unique ID coverage failed")


def fixed_blend(base: np.ndarray, raw: np.ndarray,
                eligible: np.ndarray, weight: float) -> np.ndarray:
    """Apply the frozen weight only to valid-proxy IDs, preserving all others."""
    if (weight not in WEIGHTS or base.shape != raw.shape
            or base.shape != eligible.shape or eligible.dtype != bool
            or not np.isfinite(base).all() or (base < 0).any()
            or not np.isfinite(raw[eligible]).all()):
        raise ValueError("V12 fixed blend inputs, weight or valid coverage differ")
    result = base.copy()
    result[eligible] = np.maximum(
        base[eligible] + weight * (raw[eligible] - base[eligible]), 0)
    if (not np.array_equal(result[~eligible], base[~eligible])
            or not np.isfinite(result).all() or (result < 0).any()):
        raise ValueError("V12 fixed blend changed an ineligible row")
    return result


def require_frozen_settings(args: argparse.Namespace) -> None:
    """Reject direct calls that bypass the CLI's frozen fit arguments."""
    if (type(getattr(args, "depth", None)) is not int or args.depth != 10
            or type(getattr(args, "iterations", None)) is not int
            or args.iterations != 10000
            or type(getattr(args, "threads", None)) is not int
            or args.threads != 2):
        raise ValueError("V12 requires depth=10, iterations=10000, threads=2")
    minimum = getattr(args, "min_free_gib", None)
    if (isinstance(minimum, bool)
            or not isinstance(minimum, (int, float, np.integer, np.floating))):
        raise ValueError("V12 memory floor must be finite and at least 10 GiB")
    if not np.isfinite(minimum) or minimum < 10:
        raise ValueError("V12 memory floor must be finite and at least 10 GiB")
    if deep.params(args) != EXPECTED_CATBOOST_PARAMS:
        raise ValueError("V12 CatBoost trainer params or seed changed")
    if tuple(GEOMETRY_FEATURES) != EXPECTED_GEOMETRY_FEATURES:
        raise ValueError("V12 exact 16-field geometry schema changed")


def verify_saved_params(trained: dict) -> None:
    """Check effective core training settings embedded in a saved CatBoost model."""
    exact = ("task_type", "loss_function", "eval_metric", "depth",
             "random_seed", "max_ctr_complexity", "one_hot_max_size",
             "border_count")
    numeric = ("learning_rate", "l2_leaf_reg", "random_strength",
               "bagging_temperature")
    if any(trained.get(key) != EXPECTED_CATBOOST_PARAMS[key] for key in exact):
        raise ValueError("Saved CatBoost model has changed core settings")
    for key in numeric:
        try:
            actual = float(trained[key])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Saved CatBoost model lacks {key}") from error
        if (not np.isfinite(actual)
                or not np.isclose(actual, EXPECTED_CATBOOST_PARAMS[key],
                                  rtol=0, atol=1e-6)):
            raise ValueError(f"Saved CatBoost model changed {key}")


def protocol_spec() -> dict:
    """Prospective policy fixed before any v12 fit or held-out comparison."""
    return {
        "purpose": "Assess exactly 16 original runway geometry, wind and other-strip flow features",
        "architecture": {
            "base": "locally sealed v11 184-feature CatBoost residual to own AOBT proxy",
            "trainer": "deep_timestamp_expert.fit_fold with unchanged split and CatBoost parameters",
            "base_feature_count": 184, "added_feature_count": 16,
            "feature_count": 200, "categorical_feature_count": 24,
            "additional_features": list(GEOMETRY_FEATURES),
            "depth": 10, "max_iterations": 10000, "random_seed": 2026,
            "params": EXPECTED_CATBOOST_PARAMS,
            "fit_labels": "finite 2025 targets in [0,86400] with AOBT proxy in [0,7200]",
            "score_labels": "all finite targets, including negative and extreme values",
            "early_stop": "same complement-month RNG permutation, six percent or 20,000, 200 patience",
            "forbidden": ["own departure BLOCK", "own departure TAXITIME",
                          "opaque movement IDs as predictors", "ranking labels"],
        },
        "reference": {
            "all_finite_oof": "sealed v9 evaluation_predictions.selected, 672428 exact IDs",
            "selected_route": "v11", "selected_weight": 0.5,
            "unsubmitted_ranking_sha256": EXPECTED_V9_SUBMISSION_SHA256,
            "outside_valid_proxy": "copy locally sealed, unsubmitted v9 without change",
            "prior_may_september_guard": "verify its already sealed v9 terminal as provenance only; those labels do not select any v12 weight or feature",
        },
        "original_folds": {
            "selection": [1, 7], "forward": [11, 12],
            "fit_and_early_stop": "complementary 2025 months only; held months absent from both",
            "weights": list(WEIGHTS),
            "selection_rule": "minimum clipped Jan/Jul all-finite RMSE; ties favor smaller weight",
            "formula": "clip(v9 + w * (new_raw_200 - v9), lower=0) only on valid AOBT proxy",
            "forward_rule": "same fixed Jan/Jul weight without adjustment",
            "gate": "nonzero weight; all-finite RMSE improves and paired UTC-day bootstrap 95 percent lower gain >0 in both folds",
            "bootstrap": {"repeats": 1000, "seed": BOOTSTRAP_SEED},
        },
        "fresh_april_october": {
            "months": [4, 10], "expected_valid_ids": EXPECTED_FRESH_ROWS,
            "reference": "sealed v11 fresh_audit_predictions.v11_blend with both components excluding 4/10",
            "replacement": "same 200-feature residual architecture refit excluding 4/10 from fit and early stop",
            "formula": "clip(v11_blend + original selected weight * (new_raw_200 - v11_blend), lower=0)",
            "gate": "exact ID/target/UTC-time coverage; each month RMSE improves and pooled paired UTC-day CI lower gain >0",
        },
        "separate_guard": {
            "months": [2, 8],
            "comparator": "fresh v11 raw residual model excluding 2/8 from fit and early stop",
            "replacement": "fresh v12 raw residual model with identical exclusions and frozen weight",
            "status": "prospective separate protocol required; neither refit nor score is implemented here",
        },
        "builder": {
            "source": "pre-2025 pinned public-domain OurAirports endpoints and released movements/NOAA wind",
            "features": list(GEOMETRY_FEATURES),
            "batch_caveat": "Counts use complete released batches. January/July 2026 batch boundaries may lack prior-window history present in full-year 2025 training.",
            "availability": "retrospective released-batch context, not complete live traffic",
        },
        "final_and_ranking": "refuse until separate February/August component guard is frozen, run and passed; missing policy remains unchanged",
        "interpretation": "Repeated 2025 component comparison, not an untouched generalization estimate or guaranteed prize rank",
        "leaderboard_use": False,
    }


def freeze_reference(args: argparse.Namespace) -> dict:
    """Seal v9 all-finite OOF and verify IDs/labels/proxy gate against cache."""
    choice = portfolio.require_selection(
        output_dir=args.portfolio_dir,
        expected_source_sha256=EXPECTED_PORTFOLIO_SOURCE_SHA256)
    if (choice.get("selected_route") != "v11"
            or choice.get("selected_weight") != 0.5
            or choice.get("evaluation_predictions_sha256") !=
            EXPECTED_V9_OOF_SHA256):
        raise ValueError("Sealed v9 OOF policy, weight or SHA changed")
    selected_choice, _ = final.selected_route()
    if selected_choice != choice or not final.require_guard(choice).get("passed"):
        raise ValueError("Sealed v9 route lacks its completed May/September guard")
    final.verify_ranking_inputs("v11")
    prior_schema, prior_rounds, _ = final.original_schema_and_rounds("v11")
    final.verify_model("v11", prior_schema, prior_rounds)
    final_manifest_path = args.v9_final_dir / "ranking_manifest.json"
    final_manifest = json.loads(final_manifest_path.read_text(encoding="utf-8"))
    published = json.loads((args.reports_dir / "submission_v9_finalized_manifest.json")
                           .read_text(encoding="utf-8"))
    if (final_manifest.get("predictions_sha256") != EXPECTED_V9_SUBMISSION_SHA256
            or final_manifest.get("selected_route") != "v11"
            or final_manifest.get("selected_weight") != 0.5
            or published.get("sha256") != EXPECTED_V9_SUBMISSION_SHA256
            or published.get("rows") != 344841
            or sha256(args.v9_final_dir / "predictions.parquet") !=
            EXPECTED_V9_SUBMISSION_SHA256
            or sha256(args.submission_v9) != EXPECTED_V9_SUBMISSION_SHA256):
        raise ValueError("Sealed unsubmitted v9 ranking identity changed")
    source_path = args.portfolio_dir / "evaluation_predictions.parquet"
    if sha256(source_path) != EXPECTED_V9_OOF_SHA256:
        raise ValueError("Sealed v9 all-finite OOF bytes changed")
    source = pd.read_parquet(source_path, columns=list(REFERENCE_COLUMNS))
    if (list(source) != list(REFERENCE_COLUMNS)
            or len(source) != EXPECTED_OOF_ROWS
            or source.MVT_ID_mvt.isna().any()
            or source.MVT_ID_mvt.duplicated().any()
            or set(source.fold.unique()) != set(FOLDS)
            or any(not source.loc[source.fold.eq(name), "month"].isin(months).all()
                   for name, months in FOLDS.items())
            or pd.to_datetime(source.MVT_TIME_UTC_mvt, utc=True,
                              errors="coerce").isna().any()
            or not np.array_equal(
                pd.to_datetime(source.MVT_TIME_UTC_mvt, utc=True)
                .dt.month.to_numpy(), source.month.to_numpy())
            or not np.isfinite(source[["target", "selected"]]
                               .to_numpy(dtype=float)).all()
            or (source.selected.to_numpy(dtype=float) < 0).any()):
        raise ValueError("Sealed v9 all-finite OOF reference is invalid")
    expected = pd.read_parquet(args.cache_dir / "training_rows.parquet",
                               columns=["MVT_ID_mvt", "target", "proxy", "month", "time"])
    expected = expected.loc[
        expected.month.isin((1, 7, 11, 12)).to_numpy()
        & np.isfinite(expected.target.to_numpy(dtype=float))]
    exact_ids(source.MVT_ID_mvt, expected.MVT_ID_mvt, "v9 frozen reference")
    aligned = expected.merge(source, on="MVT_ID_mvt", how="left", sort=False,
                             validate="one_to_one", suffixes=("_baseline", ""))
    valid = np.isfinite(aligned.proxy.to_numpy(dtype=float)) & (
        aligned.proxy.to_numpy(dtype=float) >= 0) & (
        aligned.proxy.to_numpy(dtype=float) <= 7200)
    if (not np.array_equal(aligned.target_baseline.to_numpy(dtype=float),
                           aligned.target.to_numpy(dtype=float))
            or not np.array_equal(aligned.month_baseline.to_numpy(),
                                  aligned.month.to_numpy())
            or not np.array_equal(aligned.a_valid.to_numpy(dtype=bool), valid)
            or not np.array_equal(pd.to_datetime(aligned.time, utc=True).to_numpy(),
                                  pd.to_datetime(aligned.MVT_TIME_UTC_mvt,
                                                 utc=True).to_numpy())):
        raise ValueError("v9 reference IDs, labels, month, proxy gate or time disagree")
    frozen = source
    frozen_path = args.output_dir / "frozen_v9_oof_reference.parquet"
    if frozen_path.exists():
        if not pd.read_parquet(frozen_path).equals(frozen):
            raise ValueError("Previously frozen v9 OOF reference differs")
    else:
        frozen_path.parent.mkdir(parents=True, exist_ok=True)
        write_parquet_new(frozen_path, frozen)
    return {"source_reference_sha256": sha256(source_path),
            "frozen_reference_sha256": sha256(frozen_path)}


def source_inventory(args: argparse.Namespace) -> tuple[list[Path], dict[str, Path]]:
    """Paths whose exact bytes must stay fixed across every model fit."""
    raw = _training_files(args.data_dir)
    if len(raw) != 12 or len({path.name for path in raw}) != 12:
        raise ValueError("Expected twelve canonical 2025 source files")
    _, inherited = v11.source_inventory(args)
    paths = {f"v11_input_{name}": path for name, path in inherited.items()}
    paths.update({
        "own_script": Path(__file__).resolve(),
        "own_spec": SPEC_PATH,
        "v11_script": Path(v11.__file__).resolve(),
        "portfolio_source": Path(portfolio.__file__).resolve(),
        "final_source": Path(final.__file__).resolve(),
        "geometry_builder_source": Path(geometry.__file__).resolve(),
        "geometry_feature_spec": args.geometry_source_dir / "feature_spec.json",
        "geometry_feature_protocol": args.geometry_dir / "protocol.json",
        "geometry_build_receipt": args.geometry_dir / "build_receipt.json",
        "geometry_training_cache": args.geometry_dir / "training_runway_geometry_features.parquet",
        "geometry_ranking_cache": args.geometry_dir / "ranking_runway_geometry_features.parquet",
        "v11_protocol": args.v11_dir / "protocol.json",
        "v11_frozen_reference": args.v11_dir / "frozen_v7_oof_reference.parquet",
        "v11_validation": args.v11_dir / "validation.json",
        "v11_validation_predictions": args.v11_dir / "validation_predictions.parquet",
        "v11_fresh_audit": args.v11_dir / "fresh_audit.json",
        "v11_fresh_predictions": args.v11_dir / "fresh_audit_predictions.parquet",
        "v11_fresh_model": args.v11_dir / "fresh_new/fresh_apr_oct.cbm",
        "v11_fresh_receipt": args.v11_dir / "fresh_new/fresh_apr_oct_provenance.json",
        "v11_fresh_oof": args.v11_dir / "fresh_new/fresh_apr_oct_oof.parquet",
        "v9_portfolio_protocol": args.reports_dir / "later_feature_portfolio_protocol.json",
        "v9_selection": args.portfolio_dir / "selection.json",
        "v9_evaluation": args.portfolio_dir / "evaluation.json",
        "v9_evaluation_predictions": args.portfolio_dir / "evaluation_predictions.parquet",
        "v9_guard_protocol": args.guard_dir / "protocol.json",
        "v9_guard_terminal": args.guard_dir / "terminal.json",
        "v9_final_protocol": args.v9_final_dir / "protocol.json",
        "v9_final_model_report": args.v9_final_dir / "final_model.json",
        "v9_final_model": args.v9_final_dir / "full_2025.cbm",
        "v9_final_ranking_manifest": args.v9_final_dir / "ranking_manifest.json",
        "v9_final_ranking_inputs": args.v9_final_dir / "ranking_inputs.json",
        "v9_final_ranking_expert": args.v9_final_dir / "ranking_expert.parquet",
        "v9_submission_manifest": args.reports_dir / "submission_v9_finalized_manifest.json",
        "v9_unsubmitted_ranking": args.submission_v9,
        "v9_final_predictions": args.v9_final_dir / "predictions.parquet",
        "v7_loader": Path(traffic.__file__).resolve(),
        "residual_trainer": Path(deep.__file__).resolve(),
        "arrival_loader_bootstrap": Path(arrival.__file__).resolve(),
        "flight_weather_loader": Path(catboost_expert.__file__).resolve(),
        "flight_join": Path(airport_models.__file__).resolve(),
        "baseline_loader": Path(solution.__file__).resolve(),
        "weather_join": Path(weather_model.__file__).resolve(),
        "neighbour_feature_names": Path(proxy_neighbour_expert.__file__).resolve(),
        "runway_feature_names_normalizer": Path(runway_arrival_features.__file__).resolve(),
        "trainer_import_reference": Path(v4_reference.__file__).resolve(),
        "baseline_rows": args.cache_dir / "training_rows.parquet",
        "baseline_features": args.cache_dir / "features.parquet",
        "arrival_cache": args.arrival_dir / "training_arrival_features.parquet",
        "neighbour_cache": args.neighbour_dir / "training_neighbour_features.parquet",
        "runway_sequence_cache": args.runway_dir / "training_runway_arrival_features.parquet",
        "weather": args.weather_file,
    })
    for fold in FOLDS:
        for suffix in (".cbm", "_oof.parquet", "_validation.json", "_provenance.json"):
            paths[f"v11_{fold}{suffix}"] = args.v11_dir / f"{fold}{suffix}"
    if len(paths) != len(set(paths)) or any(not path.is_file() for path in paths.values()):
        missing = [name for name, path in paths.items() if not path.is_file()]
        raise FileNotFoundError(f"V12 prospective source inventory incomplete: {missing}")
    return raw, paths


def verify_source_snapshot(args: argparse.Namespace, frozen: dict,
                           protocol_sha256: str) -> None:
    """Rehash the frozen inputs without loading dataframes or applying a RAM gate."""
    raw, paths = source_inventory(args)
    if (sha256(args.output_dir / "protocol.json") != protocol_sha256
            or json.loads((args.output_dir / "protocol.json")
                          .read_text(encoding="utf-8")) != frozen
            or {name: sha256(path) for name, path in paths.items()}
               != frozen["input_sha256"]
            or {path.name: sha256(path) for path in raw}
               != frozen["raw_training_sha256"]
            or sha256(args.output_dir / "frozen_v9_oof_reference.parquet")
               != frozen["references"]["frozen_reference_sha256"]):
        raise ValueError("V12 source, cache, reference or protocol changed during fit")


def verify_prior_v11(args: argparse.Namespace) -> None:
    """Recheck locally sealed original and fresh component proofs without refit."""
    choice = portfolio.require_selection(
        output_dir=args.portfolio_dir,
        expected_source_sha256=EXPECTED_PORTFOLIO_SOURCE_SHA256)
    terminal = choice.get("terminal_routes", {}).get("v11", {})
    validation_path = args.v11_dir / "validation.json"
    fresh_path = args.v11_dir / "fresh_audit.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    fresh = json.loads(fresh_path.read_text(encoding="utf-8"))
    if (choice.get("selected_route") != "v11"
            or choice.get("selected_weight") != 0.5
            or terminal.get("original_passed") is not True
            or terminal.get("fresh_passed") is not True
            or terminal.get("compatibility_passed") is not True
            or terminal.get("original_validation_sha256") != sha256(validation_path)
            or terminal.get("fresh_audit_sha256") != sha256(fresh_path)
            or validation.get("existing_folds_passed") is not True
            or fresh.get("passed") is not True
            or validation.get("selected_weight") != 0.5
            or fresh.get("weight") != 0.5):
        raise ValueError("Sealed v11 original/fresh terminal proof changed")
    inherited = argparse.Namespace(**vars(args))
    inherited.output_dir = args.v11_dir
    frozen = json.loads((args.v11_dir / "protocol.json").read_text(encoding="utf-8"))
    v11.verify_source_snapshot(inherited, frozen, sha256(args.v11_dir / "protocol.json"))


def verify_geometry_builder(args: argparse.Namespace) -> None:
    """Validate the label-free builder protocol and both exact cache receipts."""
    if sha256(Path(geometry.__file__).resolve()) != EXPECTED_BUILDER_SHA256:
        raise ValueError("Published geometry builder source changed")
    built_args = argparse.Namespace(source_dir=args.geometry_source_dir,
                                    output_dir=args.geometry_dir,
                                    data_dir=args.data_dir,
                                    baseline_dir=args.cache_dir,
                                    weather_file=args.weather_file,
                                    min_free_gib=4.0)
    geometry.require_prepared(built_args)
    report = json.loads((args.geometry_dir / "build_receipt.json")
                        .read_text(encoding="utf-8"))
    outputs = report.get("outputs", [])
    if (sha256(args.geometry_dir / "protocol.json") !=
            EXPECTED_GEOMETRY_PROTOCOL_SHA256
            or report.get("protocol_sha256") != EXPECTED_GEOMETRY_PROTOCOL_SHA256
            or report.get("source_script_sha256") != EXPECTED_BUILDER_SHA256
            or report.get("features") != list(GEOMETRY_FEATURES)
            or report.get("departure_labels_used") is not False
            or report.get("ranking_labels_used") is not False
            or len(outputs) != 2
            or outputs[0].get("name") != "training"
            or outputs[0].get("rows") != EXPECTED_TRAINING_ROWS
            or outputs[0].get("sha256") != EXPECTED_GEOMETRY_TRAIN_SHA256
            or outputs[1].get("name") != "ranking"
            or outputs[1].get("rows") != 344841
            or outputs[1].get("sha256") != EXPECTED_GEOMETRY_RANK_SHA256
            or sha256(args.geometry_dir / "training_runway_geometry_features.parquet")
            != EXPECTED_GEOMETRY_TRAIN_SHA256
            or sha256(args.geometry_dir / "ranking_runway_geometry_features.parquet")
            != EXPECTED_GEOMETRY_RANK_SHA256):
        raise ValueError("Label-free geometry builder or cache receipt changed")


def protocol(args: argparse.Namespace) -> dict:
    require_frozen_settings(args)
    geometry.require_memory(args.min_free_gib)
    published_spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    if published_spec != protocol_spec():
        raise ValueError("Published prospective v12 model spec differs from source")
    verify_prior_v11(args)
    verify_geometry_builder(args)
    raw, paths = source_inventory(args)
    inputs = {name: sha256(path) for name, path in paths.items()}
    original = {path.name: sha256(path) for path in raw}
    references = freeze_reference(args)
    if (inputs != {name: sha256(path) for name, path in paths.items()}
            or original != {path.name: sha256(path) for path in raw}):
        raise ValueError("V12 source or input changed while freezing v9 reference")
    value = {
        "spec": protocol_spec(), "references": references,
        "input_sha256": inputs,
        "raw_training_sha256": original,
    }
    target = args.output_dir / "protocol.json"
    if target.exists():
        if json.loads(target.read_text(encoding="utf-8")) != value:
            raise ValueError("Frozen v12 protocol or source/input hashes changed")
    else:
        write_json_new(target, value)
    verify_source_snapshot(args, value, sha256(target))
    return value


def require_prepared(args: argparse.Namespace) -> tuple[dict, str]:
    """Read-only verification of the pre-fit frozen policy and source bytes."""
    require_frozen_settings(args)
    path = args.output_dir / "protocol.json"
    if not path.is_file():
        raise FileNotFoundError("Run v12 --mode prepare before model fitting")
    frozen = json.loads(path.read_text(encoding="utf-8"))
    protocol_sha = sha256(path)
    if frozen.get("spec") != protocol_spec() or frozen["references"].get(
            "source_reference_sha256") != EXPECTED_V9_OOF_SHA256:
        raise ValueError("V12 prospective model policy or sealed v9 reference changed")
    verify_source_snapshot(args, frozen, protocol_sha)
    verify_prior_v11(args)
    verify_geometry_builder(args)
    return frozen, protocol_sha


def load_features(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, features = v11.load_features(args)
    if len(rows) != EXPECTED_TRAINING_ROWS or len(features.columns) != 184:
        raise ValueError("Sealed v11 184-column matrix or row universe changed")
    cache_path = args.geometry_dir / "training_runway_geometry_features.parquet"
    added = pd.read_parquet(cache_path)
    if (list(added) != ["MVT_ID_mvt", *GEOMETRY_FEATURES]
            or len(added) != len(rows)
            or not np.array_equal(added.MVT_ID_mvt.to_numpy(),
                                  rows.MVT_ID_mvt.to_numpy())
            or any(name in features for name in GEOMETRY_FEATURES)
            or any(added[name].dtype != np.dtype("float32")
                   for name in GEOMETRY_FEATURES)):
        raise ValueError("Sixteen fixed runway geometry fields lack exact ID/schema alignment")
    geometry.verify_output(added, rows[["MVT_ID_mvt"]])
    features = pd.concat([features.reset_index(drop=True),
                          added[list(GEOMETRY_FEATURES)].reset_index(drop=True)], axis=1)
    if (len(features.columns) != 200
            or len(features.select_dtypes(include="category").columns) != 24
            or features.columns.duplicated().any() or "MVT_ID_mvt" in features
            or "target" in features or "BLOCK_TIME_UTC_mvt" in features
            or "TAXITIME_SEC_mvt" in features):
        raise ValueError("v12 model predictor schema or forbidden own fields changed")
    return rows, features


def load_reference(args: argparse.Namespace) -> pd.DataFrame:
    ref = pd.read_parquet(args.output_dir / "frozen_v9_oof_reference.parquet")
    if (list(ref) != list(REFERENCE_COLUMNS) or len(ref) != EXPECTED_OOF_ROWS
            or ref.MVT_ID_mvt.isna().any() or ref.MVT_ID_mvt.duplicated().any()
            or set(ref.fold.unique()) != set(FOLDS)
            or pd.to_datetime(ref.MVT_TIME_UTC_mvt, utc=True,
                              errors="coerce").isna().any()
            or not np.array_equal(
                pd.to_datetime(ref.MVT_TIME_UTC_mvt, utc=True)
                .dt.month.to_numpy(), ref.month.to_numpy())
            or not np.isfinite(ref[["target", "selected"]]
                               .to_numpy(dtype=float)).all()
            or (ref.selected.to_numpy(dtype=float) < 0).any()):
        raise ValueError("Frozen v9 policy reference schema or coverage changed")
    return ref


def verify_expert(args: argparse.Namespace, name: str,
                  expected: pd.Series) -> pd.DataFrame:
    path = args.output_dir / f"{name}_oof.parquet"
    expert = pd.read_parquet(path)
    if list(expert) != ["MVT_ID_mvt", "expert"]:
        raise ValueError(f"{name} expert OOF has unexpected schema")
    exact_ids(expert.MVT_ID_mvt, expected, f"{name} v12 expert")
    if not np.isfinite(expert.expert.to_numpy(dtype=float)).all():
        raise ValueError(f"{name} v12 expert has nonfinite predictions")
    return expert


def fold_provenance(args: argparse.Namespace, name: str,
                    held: pd.DataFrame, schema: list[dict]) -> dict:
    """Bind each saved fold to its frozen source, feature schema and model."""
    if name == "fresh_apr_oct":
        months = FRESH_MONTHS
        protocol_path = args.output_dir.parent / "protocol.json"
        if args.output_dir.name != "fresh_new" or "a_valid" in held:
            raise ValueError("Fresh fold must use valid-only heldout rows and parent protocol")
        expected = held.MVT_ID_mvt
    elif name in FOLDS:
        months = FOLDS[name]
        protocol_path = args.output_dir / "protocol.json"
        if "a_valid" not in held:
            raise ValueError("Existing fold lacks valid-AOBT flags")
        expected = held.loc[held.a_valid, "MVT_ID_mvt"]
    else:
        raise ValueError("Unknown V12 fold")
    if (not held.fold.eq(name).all()
            or not held.month.isin(months).all()
            or held.MVT_ID_mvt.isna().any()
            or held.MVT_ID_mvt.duplicated().any()
            or not np.isfinite(held.target.to_numpy(dtype=float)).all()
            or not np.array_equal(
                pd.to_datetime(held.MVT_TIME_UTC_mvt, utc=True)
                .dt.month.to_numpy(), held.month.to_numpy())):
        raise ValueError("Fold provenance has an unexpected held-out universe")
    expert = verify_expert(args, name, expected)
    report_path = args.output_dir / f"{name}_validation.json"
    model_path = args.output_dir / f"{name}.cbm"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    names = [item["name"] for item in schema]
    if (report.get("fold") != name or report.get("n_all_finite") != len(held)
            or report.get("n_eligible") != len(expert)
            or report.get("features") != names
            or len(names) != 200
            or names[-len(GEOMETRY_FEATURES):] != list(GEOMETRY_FEATURES)
            or len(names) != len(set(names))
            or report.get("trees", 0) < 1
            or report["trees"] > EXPECTED_CATBOOST_PARAMS["iterations"]):
        raise ValueError("Fold report differs from the frozen valid-AOBT feature universe")
    model = CatBoostRegressor()
    model.load_model(str(model_path))
    trained = model.get_all_params()
    categorical_indices = [position for position, item in enumerate(schema)
                           if item["dtype"] == "category"]
    if (len(categorical_indices) != 24
            or int(model.tree_count_) != report["trees"]
            or list(model.feature_names_) != names
            or list(model.get_cat_feature_indices()) != categorical_indices):
        raise ValueError("Saved CatBoost model metadata differs from frozen architecture")
    verify_saved_params(trained)
    return {
        "schema_version": 1,
        "fold": name,
        "heldout_months": list(months),
        "heldout_scope": ("valid_aobt_finite" if name == "fresh_apr_oct"
                          else "all_finite_targets"),
        "protocol_sha256": sha256(protocol_path),
        "catboost_params": EXPECTED_CATBOOST_PARAMS,
        "feature_schema": schema,
        "categorical_feature_indices": categorical_indices,
        "heldout_all_finite_rows": len(held),
        "heldout_valid_aobt_rows": len(expert),
        "trees": report["trees"],
        "oof_sha256": sha256(args.output_dir / f"{name}_oof.parquet"),
        "model_sha256": sha256(model_path),
        "fit_report_sha256": sha256(report_path),
    }


def seal_fold(args: argparse.Namespace, name: str,
              held: pd.DataFrame, features: pd.DataFrame) -> None:
    schema = [{"name": str(column), "dtype": str(features[column].dtype)}
              for column in features]
    value = fold_provenance(args, name, held, schema)
    path = args.output_dir / f"{name}_provenance.json"
    with path.open("x", encoding="utf-8") as output:
        json.dump(value, output, indent=2)
        output.write("\n")


def verify_fold_provenance(args: argparse.Namespace, name: str,
                           held: pd.DataFrame) -> None:
    path = args.output_dir / f"{name}_provenance.json"
    saved = json.loads(path.read_text(encoding="utf-8"))
    schema = saved.get("feature_schema")
    if (not isinstance(schema, list)
            or any(not isinstance(item, dict)
                   or set(item) != {"name", "dtype"} for item in schema)):
        raise ValueError("Fold feature schema receipt is invalid")
    if saved != fold_provenance(args, name, held, schema):
        raise ValueError("Fold OOF/model/report or frozen protocol provenance changed")


def fit_folds(args: argparse.Namespace) -> None:
    frozen, protocol_sha = require_prepared(args)
    geometry.require_memory(args.min_free_gib)
    for name in FOLDS:
        if any((args.output_dir / f"{name}{suffix}").exists() for suffix in
               ("_oof.parquet", ".cbm", "_validation.json",
                "_provenance.json")):
            raise FileExistsError(f"v12 {name} artifacts already exist; verify before reuse")
    rows, features = load_features(args)
    ref = load_reference(args)
    for name, months in FOLDS.items():
        verify_source_snapshot(args, frozen, protocol_sha)
        deep.fit_fold(name, months, rows, features, ref, args)
        verify_source_snapshot(args, frozen, protocol_sha)
        held = ref.loc[ref.fold.eq(name)]
        verify_expert(args, name, held.loc[held.a_valid, "MVT_ID_mvt"])
        seal_fold(args, name, held, features)
        verify_source_snapshot(args, frozen, protocol_sha)
    del rows, features, ref, held
    gc.collect()
    evaluate(args, persist=True)


def evaluate(args: argparse.Namespace, persist: bool = False) -> dict:
    """Recompute both original folds; write once or verify saved immutable output."""
    frozen, protocol_sha = require_prepared(args)
    ref = load_reference(args)
    report: dict = {
        "protocol_sha256": protocol_sha,
        "frozen_reference_sha256": frozen["references"]["frozen_reference_sha256"],
        "folds": {}, "selected_weight": None,
        "existing_folds_passed": False,
        "fresh_audit_status": "separate_april_october_audit",
        "ranking_authorized": False,
    }
    pieces = []
    for name in FOLDS:
        held = ref.loc[ref.fold.eq(name)]
        verify_fold_provenance(args, name, held)
        expert = verify_expert(args, name, held.loc[held.a_valid, "MVT_ID_mvt"])
        part = held.merge(expert, on="MVT_ID_mvt", how="left", sort=False,
                          validate="one_to_one")
        present = part.expert.notna().to_numpy()
        if (len(part) != len(held)
                or not np.array_equal(present, part.a_valid.to_numpy(dtype=bool))):
            raise ValueError(f"{name} v12 expert lacks exact valid-proxy coverage")
        y = part.target.to_numpy(dtype=float)
        base = part.selected.to_numpy(dtype=float)
        alternate = part.expert.fillna(part.selected).to_numpy(dtype=float)
        scores = {str(weight): deep.rmse(y, np.maximum(
            base + weight * (alternate - base), 0)) for weight in WEIGHTS}
        info_path = args.output_dir / f"{name}_validation.json"
        info = json.loads(info_path.read_text(encoding="utf-8"))
        if info.get("scores_all_finite") != scores:
            raise ValueError(f"{name} CatBoost saved validation scores differ")
        if name == "seasonal_jan_jul":
            report["selected_weight"] = min(
                WEIGHTS, key=lambda weight: (scores[str(weight)], weight))
        weight = report["selected_weight"]
        candidate = fixed_blend(base, part.expert.to_numpy(dtype=float),
                                present, weight)
        if not np.array_equal(candidate[~present], base[~present]):
            raise ValueError(f"{name} changed the v9 invalid-proxy policy")
        report["folds"][name] = {
            "rows_all_finite": len(part), "rows_valid_aobt": int(present.sum()),
            "scores_all_finite_rmse": scores,
            "day_bootstrap": arrival.bootstrap(part, base, candidate,
                                                 seed=BOOTSTRAP_SEED),
            "expert_oof_sha256": sha256(args.output_dir / f"{name}_oof.parquet"),
            "model_sha256": sha256(args.output_dir / f"{name}.cbm"),
            "fit_report_sha256": sha256(info_path),
            "fold_provenance_sha256": sha256(
                args.output_dir / f"{name}_provenance.json"),
            "trees": info["trees"],
        }
        part["candidate"] = candidate
        pieces.append(part)
    weight = report["selected_weight"]
    report["existing_folds_passed"] = bool(weight > 0 and all(
        fold["scores_all_finite_rmse"][str(weight)]
        < fold["scores_all_finite_rmse"]["0.0"]
        and fold["day_bootstrap"]["gain_ci95_sec"][0] > 0
        for fold in report["folds"].values()))
    result = pd.concat(pieces, ignore_index=True)
    exact_ids(result.MVT_ID_mvt, ref.MVT_ID_mvt, "v12 all-finite OOF")
    if len(result) != EXPECTED_OOF_ROWS:
        raise ValueError("V12 all-finite OOF row count changed")
    verify_source_snapshot(args, frozen, protocol_sha)
    output_path = args.output_dir / "validation_predictions.parquet"
    report_path = args.output_dir / "validation.json"
    if persist:
        if output_path.exists() or report_path.exists():
            raise FileExistsError("V12 original evaluation already sealed")
        report["validation_predictions_sha256"] = write_parquet_new(
            output_path, result)
        verify_source_snapshot(args, frozen, protocol_sha)
        write_json_new(report_path, report)
    else:
        saved = pd.read_parquet(output_path)
        if (list(saved) != list(result) or not saved.equals(result)
                or report_path.is_file() is False):
            raise ValueError("Saved v12 all-finite OOF differs from fixed formula")
        report["validation_predictions_sha256"] = sha256(output_path)
        prior = json.loads(report_path.read_text(encoding="utf-8"))
        if prior != report:
            raise ValueError("Saved v12 original scores, receipt hashes or gate differ")
    verify_source_snapshot(args, frozen, protocol_sha)
    if report["validation_predictions_sha256"] != sha256(output_path):
        raise ValueError("V12 OOF changed after evaluation")
    return report

def verify_fresh(args: argparse.Namespace) -> dict:
    """Replay the sealed April/October formula and all matched ID proofs."""
    local = evaluate(args)
    if not local["existing_folds_passed"]:
        raise ValueError("V12 original folds failed before fresh audit")
    frozen, protocol_sha = require_prepared(args)
    audit_path = args.output_dir / "fresh_audit.json"
    paired_path = args.output_dir / "fresh_audit_predictions.parquet"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (audit.get("months") != list(FRESH_MONTHS)
            or audit.get("weight") != local["selected_weight"]
            or audit.get("protocol_sha256") != protocol_sha
            or audit.get("original_validation_sha256") !=
            sha256(args.output_dir / "validation.json")
            or audit.get("v11_fresh_report_sha256") !=
            sha256(args.v11_dir / "fresh_audit.json")
            or audit.get("v11_paired_sha256") !=
            sha256(args.v11_dir / "fresh_audit_predictions.parquet")
            or audit.get("v11_fresh_model_sha256") !=
            sha256(args.v11_dir / "fresh_new/fresh_apr_oct.cbm")
            or audit.get("v12_expert_oof_sha256") !=
            sha256(args.output_dir / "fresh_new/fresh_apr_oct_oof.parquet")
            or audit.get("v12_model_sha256") !=
            sha256(args.output_dir / "fresh_new/fresh_apr_oct.cbm")
            or audit.get("fresh_fold_provenance_sha256") !=
            sha256(args.output_dir / "fresh_new/fresh_apr_oct_provenance.json")
            or audit.get("predictions_sha256") != sha256(paired_path)
            or audit.get("ranking_authorized") is not False):
        raise ValueError("V12 fresh report/model/reference seals changed")
    frame = pd.read_parquet(paired_path)
    columns = ["MVT_ID_mvt", "target", "month", "MVT_TIME_UTC_mvt",
               "selected", "fold", "expert", "v12_blend"]
    if (list(frame) != columns or len(frame) != EXPECTED_FRESH_ROWS
            or frame.MVT_ID_mvt.isna().any() or frame.MVT_ID_mvt.duplicated().any()
            or not frame.fold.eq("fresh_apr_oct").all()
            or set(frame.month.unique()) != set(FRESH_MONTHS)
            or not np.isfinite(frame[["target", "selected", "expert", "v12_blend"]]
                               .to_numpy(dtype=float)).all()):
        raise ValueError("V12 fresh paired schema, coverage or numeric values changed")
    baseline = pd.read_parquet(args.cache_dir / "training_rows.parquet",
                               columns=["MVT_ID_mvt", "target", "proxy", "month", "time"])
    proxy = baseline.proxy.to_numpy(dtype=float)
    target = baseline.target.to_numpy(dtype=float)
    mask = (baseline.month.isin(FRESH_MONTHS).to_numpy()
            & np.isfinite(target) & np.isfinite(proxy)
            & (proxy >= 0) & (proxy <= 7200))
    expected = baseline.loc[mask]
    exact_ids(frame.MVT_ID_mvt, expected.MVT_ID_mvt, "V12 fresh baseline valid IDs")
    aligned = expected.set_index("MVT_ID_mvt").loc[frame.MVT_ID_mvt.to_numpy()]
    if (len(expected) != EXPECTED_FRESH_ROWS
            or not np.array_equal(frame.target.to_numpy(dtype=float),
                                  aligned.target.to_numpy(dtype=float))
            or not np.array_equal(frame.month.to_numpy(), aligned.month.to_numpy())
            or not np.array_equal(
                pd.to_datetime(frame.MVT_TIME_UTC_mvt, utc=True).to_numpy(),
                pd.to_datetime(aligned.time, utc=True).to_numpy())):
        raise ValueError("V12 fresh IDs, labels or timestamps differ from held-out baseline")
    prior = pd.read_parquet(args.v11_dir / "fresh_audit_predictions.parquet",
                            columns=["MVT_ID_mvt", "target", "MVT_TIME_UTC_mvt",
                                     "v11_blend"])
    exact_ids(frame.MVT_ID_mvt, prior.MVT_ID_mvt, "V12 matched v11 fresh comparator")
    old = prior.set_index("MVT_ID_mvt").loc[frame.MVT_ID_mvt.to_numpy()]
    if (not np.array_equal(frame.target.to_numpy(dtype=float),
                           old.target.to_numpy(dtype=float))
            or not np.array_equal(frame.selected.to_numpy(dtype=float),
                                  old.v11_blend.to_numpy(dtype=float))
            or not np.array_equal(
                pd.to_datetime(frame.MVT_TIME_UTC_mvt, utc=True).to_numpy(),
                pd.to_datetime(old.MVT_TIME_UTC_mvt, utc=True).to_numpy())):
        raise ValueError("V12 fresh reference differs from sealed v11 matched model")
    new_args = argparse.Namespace(**vars(args))
    new_args.output_dir = args.output_dir / "fresh_new"
    verify_fold_provenance(new_args, "fresh_apr_oct", frame)
    expert = verify_expert(new_args, "fresh_apr_oct", frame.MVT_ID_mvt)
    raw = expert.set_index("MVT_ID_mvt").loc[
        frame.MVT_ID_mvt.to_numpy(), "expert"].to_numpy(dtype=float)
    base = frame.selected.to_numpy(dtype=float)
    candidate = fixed_blend(base, raw, np.ones(len(base), dtype=bool),
                            local["selected_weight"])
    if (not np.array_equal(frame.expert.to_numpy(dtype=float), raw)
            or not np.array_equal(frame.v12_blend.to_numpy(dtype=float), candidate)):
        raise ValueError("Saved V12 fresh expert or fixed blend differs")
    y = frame.target.to_numpy(dtype=float)
    month = frame.month.to_numpy(dtype=int)
    scores = {str(value): {
        "n": int((month == value).sum()),
        "v11_rmse": deep.rmse(y[month == value], base[month == value]),
        "v12_blend_rmse": deep.rmse(y[month == value], candidate[month == value]),
    } for value in FRESH_MONTHS}
    bootstrap = arrival.bootstrap(frame, base, candidate, seed=BOOTSTRAP_SEED)
    passed = (all(scores[str(value)]["v12_blend_rmse"] <
                  scores[str(value)]["v11_rmse"] for value in FRESH_MONTHS)
              and bootstrap["gain_ci95_sec"][0] > 0)
    if (audit.get("scores") != scores or audit.get("bootstrap") != bootstrap
            or audit.get("passed") is not bool(passed)
            or audit.get("rows_valid_aobt_finite") != len(frame)):
        raise ValueError("Saved V12 fresh scores, interval or terminal gate changed")
    verify_source_snapshot(args, frozen, protocol_sha)
    return audit


def fresh_audit(args: argparse.Namespace) -> None:
    """One matched April/October refit at the frozen original-fold weight."""
    local = evaluate(args)
    if not local["existing_folds_passed"]:
        raise ValueError("Original January/July or November/December gate failed")
    fresh_path = args.output_dir / "fresh_audit.json"
    paired_path = args.output_dir / "fresh_audit_predictions.parquet"
    new_dir = args.output_dir / "fresh_new"
    artifacts = [fresh_path, paired_path]
    artifacts += [new_dir / f"fresh_apr_oct{suffix}" for suffix in
                  ("_oof.parquet", ".cbm", "_validation.json", "_provenance.json")]
    if any(path.exists() for path in artifacts):
        raise FileExistsError("V12 fresh artifact already exists; never refit or overwrite")
    frozen, protocol_sha = require_prepared(args)
    geometry.require_memory(args.min_free_gib)
    rows, features = load_features(args)
    proxy = rows.proxy.to_numpy(dtype=float)
    y = rows.target.to_numpy(dtype=float)
    held = rows.month.isin(FRESH_MONTHS).to_numpy()
    valid = np.isfinite(proxy) & (proxy >= 0) & (proxy <= 7200) & np.isfinite(y)
    expected = rows.loc[held & valid,
                        ["MVT_ID_mvt", "target", "month", "time"]].copy()
    expected.rename(columns={"time": "MVT_TIME_UTC_mvt"}, inplace=True)
    if (len(expected) != EXPECTED_FRESH_ROWS
            or expected.MVT_ID_mvt.isna().any()
            or expected.MVT_ID_mvt.duplicated().any()
            or set(expected.month.unique()) != set(FRESH_MONTHS)
            or not np.array_equal(
                pd.to_datetime(expected.MVT_TIME_UTC_mvt, utc=True)
                .dt.month.to_numpy(), expected.month.to_numpy())):
        raise ValueError("V12 fresh held-out universe or month/time mapping changed")
    prior_path = args.v11_dir / "fresh_audit_predictions.parquet"
    prior_report = json.loads((args.v11_dir / "fresh_audit.json")
                              .read_text(encoding="utf-8"))
    if (prior_report.get("months") != list(FRESH_MONTHS)
            or prior_report.get("passed") is not True
            or prior_report.get("weight") != 0.5
            or prior_report.get("predictions_sha256") != sha256(prior_path)):
        raise ValueError("Sealed v11 fresh comparator changed")
    prior = pd.read_parquet(prior_path,
                            columns=["MVT_ID_mvt", "target", "MVT_TIME_UTC_mvt",
                                     "v11_blend"])
    exact_ids(prior.MVT_ID_mvt, expected.MVT_ID_mvt,
              "V12 fresh matched v11 comparator")
    paired = expected.merge(prior, on="MVT_ID_mvt", how="left", sort=False,
                            validate="one_to_one", suffixes=("", "_v11"))
    if (not np.array_equal(paired.target.to_numpy(dtype=float),
                           paired.target_v11.to_numpy(dtype=float))
            or not np.array_equal(
                pd.to_datetime(paired.MVT_TIME_UTC_mvt, utc=True).to_numpy(),
                pd.to_datetime(paired.MVT_TIME_UTC_mvt_v11, utc=True).to_numpy())
            or not np.isfinite(paired.v11_blend.to_numpy(dtype=float)).all()
            or (paired.v11_blend.to_numpy(dtype=float) < 0).any()):
        raise ValueError("V12 fresh v11 labels, timestamps or predictions differ")
    reference = paired[["MVT_ID_mvt", "target", "month", "MVT_TIME_UTC_mvt",
                        "v11_blend"]].rename(columns={"v11_blend": "selected"})
    reference["fold"] = "fresh_apr_oct"
    new_args = argparse.Namespace(**vars(args))
    new_args.output_dir = new_dir
    verify_source_snapshot(args, frozen, protocol_sha)
    deep.fit_fold("fresh_apr_oct", FRESH_MONTHS, rows, features, reference,
                  new_args)
    verify_source_snapshot(args, frozen, protocol_sha)
    seal_fold(new_args, "fresh_apr_oct", reference, features)
    expert = verify_expert(new_args, "fresh_apr_oct", expected.MVT_ID_mvt)
    paired = reference.merge(expert, on="MVT_ID_mvt", how="left", sort=False,
                             validate="one_to_one")
    if (len(paired) != EXPECTED_FRESH_ROWS
            or not np.array_equal(paired.MVT_ID_mvt.to_numpy(),
                                  expected.MVT_ID_mvt.to_numpy())
            or not np.isfinite(paired[["target", "selected", "expert"]]
                               .to_numpy(dtype=float)).all()):
        raise ValueError("V12 fresh model lacks exact matched prediction coverage")
    base = paired.selected.to_numpy(dtype=float)
    candidate = fixed_blend(base, paired.expert.to_numpy(dtype=float),
                            np.ones(len(base), dtype=bool),
                            local["selected_weight"])
    paired["v12_blend"] = candidate
    month = paired.month.to_numpy(dtype=int)
    target = paired.target.to_numpy(dtype=float)
    scores = {str(value): {
        "n": int((month == value).sum()),
        "v11_rmse": deep.rmse(target[month == value], base[month == value]),
        "v12_blend_rmse": deep.rmse(target[month == value], candidate[month == value]),
    } for value in FRESH_MONTHS}
    bootstrap = arrival.bootstrap(paired, base, candidate, seed=BOOTSTRAP_SEED)
    passed = (all(scores[str(value)]["v12_blend_rmse"] <
                  scores[str(value)]["v11_rmse"] for value in FRESH_MONTHS)
              and bootstrap["gain_ci95_sec"][0] > 0)
    verify_source_snapshot(args, frozen, protocol_sha)
    paired_hash = write_parquet_new(paired_path, paired)
    report = {
        "months": list(FRESH_MONTHS), "weight": local["selected_weight"],
        "scores": scores, "bootstrap": bootstrap, "passed": bool(passed),
        "coverage_verified": True, "rows_valid_aobt_finite": len(paired),
        "protocol_sha256": protocol_sha,
        "original_validation_sha256": sha256(args.output_dir / "validation.json"),
        "v11_fresh_report_sha256": sha256(args.v11_dir / "fresh_audit.json"),
        "v11_paired_sha256": sha256(prior_path),
        "v11_fresh_model_sha256": sha256(args.v11_dir / "fresh_new/fresh_apr_oct.cbm"),
        "v12_expert_oof_sha256": sha256(new_dir / "fresh_apr_oct_oof.parquet"),
        "v12_model_sha256": sha256(new_dir / "fresh_apr_oct.cbm"),
        "fresh_fold_provenance_sha256": sha256(
            new_dir / "fresh_apr_oct_provenance.json"),
        "predictions_sha256": paired_hash,
        "ranking_authorized": False,
        "scope": "Repeated 2025 matched component comparison; not untouched generalization",
    }
    verify_source_snapshot(args, frozen, protocol_sha)
    write_json_new(fresh_path, report)
    del rows, features, proxy, y, held, valid, expected, prior
    del paired, reference, expert, base, candidate, month, target
    gc.collect()
    verify_fresh(args)

def fit_final(_: argparse.Namespace) -> None:
    raise RuntimeError("V12 final training requires a separately frozen and passed February/August matched component guard")


def final_predict(_: argparse.Namespace) -> None:
    raise RuntimeError("V12 ranking prediction requires a separately frozen and passed February/August matched component guard")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("show-spec", "prepare", "fit-folds",
                                          "evaluate", "fresh-audit", "fit-final",
                                          "final-predict"), default="show-spec")
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
    parser.add_argument("--geometry-source-dir", type=Path,
                        default=Path("artifacts/prospective-runway-geometry"))
    parser.add_argument("--geometry-dir", type=Path,
                        default=Path("artifacts/prospective-runway-geometry/features"))
    parser.add_argument("--v11-dir", type=Path,
                        default=Path("artifacts/v11-taxi-flow-expert"))
    parser.add_argument("--portfolio-dir", type=Path,
                        default=Path("artifacts/later-feature-portfolio"))
    parser.add_argument("--guard-dir", type=Path,
                        default=Path("artifacts/later-reserved-guard"))
    parser.add_argument("--v9-final-dir", type=Path,
                        default=Path("artifacts/later-feature-final"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--submission-v9", type=Path,
                        default=Path("submissions/merry-mushroom_v9.parquet"))
    parser.add_argument("--v7-dir", type=Path,
                        default=Path("artifacts/v7-runway-traffic"))
    parser.add_argument("--v6-dir", type=Path,
                        default=Path("artifacts/v6-deep-arrival"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("artifacts/v12-runway-geometry-expert"))
    parser.add_argument("--min-free-gib", type=float, default=10.0)
    args = parser.parse_args()
    args.iterations, args.depth, args.threads = 10000, 10, 2
    if args.mode == "show-spec":
        print(json.dumps(protocol_spec(), indent=2))
    elif args.mode == "prepare":
        value = protocol(args)
        print(json.dumps({"protocol": str(args.output_dir / "protocol.json"),
                          "new_features": len(value["spec"]["architecture"]
                                             ["additional_features"])}))
    elif args.mode == "fit-folds":
        fit_folds(args)
    elif args.mode == "evaluate":
        evaluate(args)
    elif args.mode == "fresh-audit":
        fresh_audit(args)
    elif args.mode == "fit-final":
        fit_final(args)
    else:
        final_predict(args)


if __name__ == "__main__":
    main()
