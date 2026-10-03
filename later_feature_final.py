"""Guarded final fit and ranking output for the frozen later v10/v11 portfolio.

This is an extension of the original comparison scripts, whose final modes
intentionally refuse.  No route, weight, feature, or tree count is selected
here.  The unchanged-current route only verifies the existing v8 prediction.
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

import deep_timestamp_expert as deep
import later_feature_portfolio as portfolio
import later_reserved_guard as reserved
import traffic_deep_expert as traffic
import v10_runway_taxi_expert as v10
import v11_taxi_interval_flow_expert as v11


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "artifacts/later-feature-final"
SELECTION_DIR = ROOT / "artifacts/later-feature-portfolio"
GUARD_DIR = ROOT / "artifacts/later-reserved-guard"
CURRENT_DIR = ROOT / "artifacts/current-candidate"
V7_DIR = ROOT / "artifacts/v7-runway-traffic"
TEMPLATE = ROOT / "data/submitting.parquet"
ROUTES = ("current", "v10", "v11")
FOLDS = ("seasonal_jan_jul", "forward_nov_dec")
WEIGHTS = (0.0, 0.1, 0.25, 0.5, 1.0)
EXPECTED_TRAINING_ROWS = 2_085_047
EXPECTED_RANKING_ROWS = 344_841
EXPECTED_VALID_RANKING = 339_377
EXPECTED_FEATURES = 184
FAMILY = {"v10": v10, "v11": v11}


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
        raise ValueError(f"Path escapes the fixed competition workspace: {path}") from error


def read_json(path: Path) -> dict:
    before = sha256(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if sha256(path) != before or not isinstance(value, dict):
        raise ValueError(f"JSON changed during read or is not an object: {path}")
    return value


def json_exclusive(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as output:
        json.dump(value, output, indent=2)
        output.write("\n")


def parquet_exclusive(path: Path, frame: pd.DataFrame) -> str:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".later_final_", dir=path.parent) as work:
        stage = Path(work) / path.name
        frame.to_parquet(stage, index=False)
        expected = sha256(stage)
        os.link(stage, path)  # Same-volume exclusive publish, never overwrite.
    if sha256(path) != expected:
        raise ValueError(f"Published Parquet differs from staged bytes: {path}")
    return expected


def file_hashes(paths: dict[str, Path]) -> dict[str, str]:
    if len(paths) != len(set(paths)):
        raise ValueError("Duplicate input inventory key")
    return {name: sha256(path) for name, path in sorted(paths.items())}


def verify_hashes(paths: dict[str, Path], expected: dict[str, str],
                  label: str) -> None:
    if file_hashes(paths) != expected:
        raise ValueError(f"{label} source or input bytes changed")


def exact_ids(actual: pd.Series, expected: pd.Series, label: str) -> None:
    left, right = pd.Index(actual), pd.Index(expected)
    if (len(left) != len(right) or left.has_duplicates or right.has_duplicates
            or left.isna().any() or right.isna().any()
            or not left.isin(right).all() or not right.isin(left).all()):
        raise ValueError(f"{label}: exact unique ID coverage failed")


def fixed_blend(current: np.ndarray, v7: np.ndarray, raw: np.ndarray,
                valid: np.ndarray, weight: float) -> np.ndarray:
    """Apply the already selected weight only on the supplied valid-AOBT mask."""
    if (current.shape != v7.shape or current.shape != raw.shape
            or current.shape != valid.shape or valid.dtype != bool
            or weight not in WEIGHTS or weight <= 0
            or not np.isfinite(current).all() or not np.isfinite(v7).all()
            or not np.isfinite(raw[valid]).all()
            or (current < 0).any() or (v7 < 0).any()
            or not np.array_equal(current[valid], v7[valid])):
        raise ValueError("Selected ranking formula inputs or gate changed")
    result = current.copy()
    result[valid] = np.maximum(v7[valid]
                               + weight * (raw[valid] - v7[valid]), 0)
    if (not np.array_equal(result[~valid], current[~valid])
            or not np.isfinite(result).all() or (result < 0).any()):
        raise ValueError("Selected ranking formula changed nonvalid rows")
    return result


def family_args(route: str) -> argparse.Namespace:
    if route not in FAMILY:
        raise ValueError("Only v10 and v11 have a replacement model")
    args = portfolio.alternative_args(route)
    args.min_free_gib = 10.0
    return args


def selected_route() -> tuple[dict, dict]:
    """Verify sealed portfolio choice without rescoring held-out labels."""
    evaluation_path = SELECTION_DIR / "evaluation.json"
    predictions_path = SELECTION_DIR / "evaluation_predictions.parquet"
    choice = portfolio.require_selection(
        output_dir=SELECTION_DIR,
        expected_source_sha256=sha256(ROOT / "later_feature_portfolio.py"))
    evaluation = read_json(evaluation_path)
    route = choice.get("selected_route")
    weight = choice.get("selected_weight")
    terminal = choice.get("terminal_routes", {})
    if (route not in ROUTES or weight not in WEIGHTS
            or (route == "current") != (weight == 0.0)
            or choice.get("tie_order") != list(ROUTES)
            or choice.get("reserved_months_unscored") != [5, 9]
            or choice.get("guard_status") !=
               "pending_separate_frozen_may_september_guard"
            or choice.get("ranking_authorized") is not False
            or choice.get("portfolio_source_sha256") !=
               sha256(ROOT / "later_feature_portfolio.py")
            or choice.get("portfolio_protocol_sha256") !=
               sha256(ROOT / "reports/later_feature_portfolio_protocol.json")
            or choice.get("evaluation_sha256") != sha256(evaluation_path)
            or choice.get("evaluation_predictions_sha256") !=
               sha256(predictions_path)
            or choice.get("current_oof_sha256") !=
               sha256(CURRENT_DIR / "validation_predictions.parquet")
            or evaluation.get("selected_route") != route
            or evaluation.get("selected_weight") != weight
            or evaluation.get("evaluation_predictions_sha256") !=
               sha256(predictions_path)
            or evaluation.get("terminal_routes") != terminal):
        raise ValueError("Frozen later portfolio choice or evaluation changed")
    if route != "current":
        info = terminal.get(route, {})
        directory = portfolio.ROUTE_CONFIG[route]["directory"]
        if (info.get("original_passed") is not True
                or info.get("fresh_passed") is not True
                or info.get("compatibility_passed") is not True
                or info.get("selected_weight") != weight
                or choice.get("selected_original_validation_sha256") !=
                   sha256(directory / "validation.json")
                or choice.get("selected_fresh_audit_sha256") !=
                   sha256(directory / "fresh_audit.json")):
            raise ValueError("Selected replacement did not pass original/fresh/composition gates")
    elif (choice.get("selected_original_validation_sha256") is not None
          or choice.get("selected_fresh_audit_sha256") is not None):
        raise ValueError("Current policy selection unexpectedly binds a replacement")
    return choice, evaluation


def verify_paired_guard(left: pd.DataFrame, right: pd.DataFrame,
                        paired: pd.DataFrame, terminal: dict,
                        weight: float) -> None:
    """Replay only the frozen May/September formula and paired audit."""
    raw_columns = list(reserved.RAW_COLUMNS)
    if list(left) != raw_columns or list(right) != raw_columns:
        raise ValueError("Reserved raw OOF schemas differ")
    exact_ids(right.MVT_ID_mvt, left.MVT_ID_mvt, "Reserved replacement OOF")
    right = right.set_index("MVT_ID_mvt").loc[
        left.MVT_ID_mvt.to_numpy()].reset_index()
    if (not np.array_equal(left.target.to_numpy(dtype=float),
                           right.target.to_numpy(dtype=float))
            or not np.array_equal(
                pd.to_datetime(left.MVT_TIME_UTC_mvt, utc=True).to_numpy(),
                pd.to_datetime(right.MVT_TIME_UTC_mvt, utc=True).to_numpy())):
        raise ValueError("Reserved OOF labels or UTC times differ by ID")
    raw = left.rename(columns={"expert": "v7_raw"}).copy()
    raw["replacement_raw"] = right.expert.to_numpy(dtype=float)
    report, expected = reserved.fixed_gate(raw, weight)
    if (report["passed"] is not True
            or terminal.get("scores") != report["scores"]
            or terminal.get("bootstrap") != report["bootstrap"]
            or terminal.get("pooled_v7_rmse") != report["pooled_v7_rmse"]
            or terminal.get("pooled_fixed_blend_rmse") !=
               report["pooled_fixed_blend_rmse"]):
        raise ValueError("Saved reserved pass, monthly scores or day interval differ")
    if (list(paired) != list(expected)
            or not np.array_equal(paired.MVT_ID_mvt.to_numpy(),
                                  expected.MVT_ID_mvt.to_numpy())
            or not np.array_equal(
                pd.to_datetime(paired.MVT_TIME_UTC_mvt, utc=True).to_numpy(),
                pd.to_datetime(expected.MVT_TIME_UTC_mvt, utc=True).to_numpy())
            or any(not np.array_equal(paired[name].to_numpy(dtype=float),
                                      expected[name].to_numpy(dtype=float))
                   for name in ("target", "v7_raw", "replacement_raw",
                                "v7_clipped", "fixed_blend_clipped"))):
        raise ValueError("Saved reserved paired predictions differ from fixed formula")


def verify_guard_receipt_artifacts(name: str, directory: Path, terminal: dict,
                                   protocol_sha: str, selection_sha: str,
                                   source_sha: dict) -> dict:
    """Bind each saved guard model, OOF and fit report to its receipt."""
    if name not in ("comparator", "replacement"):
        raise ValueError("Unknown reserved model receipt")
    receipt_path = directory / f"{name}_receipt.json"
    receipt = read_json(receipt_path)
    model_path = directory / f"{name}.cbm"
    oof_path = directory / f"{name}_oof.parquet"
    fit_path = directory / f"{name}_fit.json"
    if (terminal.get(f"{name}_receipt_sha256") != sha256(receipt_path)
            or terminal.get(f"{name}_model_sha256") != sha256(model_path)
            or receipt.get("model_sha256") != sha256(model_path)
            or receipt.get("oof_sha256") != sha256(oof_path)
            or receipt.get("fit_report_sha256") != sha256(fit_path)
            or receipt.get("protocol_sha256") != protocol_sha
            or receipt.get("portfolio_selection_sha256") != selection_sha
            or receipt.get("heldout_months") != [5, 9]
            or receipt.get("source_sha256") != source_sha):
        raise ValueError(f"Reserved {name} receipt/model/OOF/fit report differs")
    return receipt


def require_guard(choice: dict) -> dict:
    """Require the separate May/September guard for a replacement route."""
    if choice["selected_route"] == "current":
        path = GUARD_DIR / "terminal.json"
        guard = read_json(path)
        protocol_path = GUARD_DIR / "protocol.json"
        frozen = read_json(protocol_path)
        guard_args = family_args("v10")
        guard_args.output_dir = GUARD_DIR
        if (reserved.check_frozen(guard_args) != frozen
                or guard.get("schema_version") != 1
                or guard.get("route") != "current"
                or guard.get("weight") != 0.0
                or guard.get("heldout_months") != [5, 9]
                or guard.get("passed") is not True
                or guard.get("replacement_fit") is not False
                or guard.get("retained_policy") != "current"
                or guard.get("protocol_sha256") != sha256(protocol_path)
                or guard.get("selection_sha256") !=
                   sha256(SELECTION_DIR / "selection.json")):
            raise ValueError("Unchanged policy lacks its frozen terminal guard")
        return guard
    path = GUARD_DIR / "terminal.json"
    guard = read_json(path)
    protocol_path = GUARD_DIR / "protocol.json"
    protocol = read_json(protocol_path)
    guard_args = family_args(choice["selected_route"])
    guard_args.output_dir = GUARD_DIR
    if reserved.check_frozen(guard_args) != protocol:
        raise ValueError("Frozen May/September sources differ from guard protocol")
    if (guard.get("schema_version") != 1
            or guard.get("route") != choice["selected_route"]
            or guard.get("weight") != choice["selected_weight"]
            or guard.get("selection_sha256") != sha256(SELECTION_DIR / "selection.json")
            or guard.get("heldout_months") != [5, 9]
            or guard.get("passed") is not True
            or guard.get("retained_policy") != choice["selected_route"]
            or guard.get("architecture_comparison_only") is not True
            or guard.get("protocol_sha256") != sha256(protocol_path)
            or protocol.get("selected_route") != choice["selected_route"]
            or protocol.get("selected_weight") != choice["selected_weight"]
            or protocol.get("portfolio_selection_sha256") !=
               sha256(SELECTION_DIR / "selection.json")
            or protocol.get("reserved_labels_used_to_freeze") is not False):
        raise ValueError("Selected May/September guard is absent, changed, or failed")
    for name in ("comparator", "replacement"):
        verify_guard_receipt_artifacts(
            name, GUARD_DIR, guard, sha256(protocol_path),
            sha256(SELECTION_DIR / "selection.json"),
            protocol.get("frozen_sources_sha256"))
    if (guard.get("paired_predictions_sha256") !=
            sha256(GUARD_DIR / "paired_predictions.parquet")
            or set(guard.get("scores", {})) != {"5", "9"}
            or any(guard["scores"][month]["fixed_blend_rmse"] >=
                   guard["scores"][month]["v7_rmse"] for month in ("5", "9"))
            or guard.get("bootstrap", {}).get("gain_ci95_sec", [0])[0] <= 0):
        raise ValueError("Reserved month scores, day interval or paired output changed")
    left = reserved.verify_fit(guard_args, "comparator", protocol)
    right = reserved.verify_fit(guard_args, "replacement", protocol)
    paired_path = GUARD_DIR / "paired_predictions.parquet"
    paired = pd.read_parquet(paired_path)
    if guard["paired_predictions_sha256"] != sha256(paired_path):
        raise ValueError("Reserved paired output changed during read")
    verify_paired_guard(left, right, paired, guard,
                        float(choice["selected_weight"]))
    return guard


def verify_current_prediction() -> dict:
    """The unchanged route creates no new model or ranking file."""
    choice, _ = selected_route()
    if choice["selected_route"] != "current":
        raise ValueError("Current-only verification called for a replacement route")
    require_guard(choice)
    manifest = read_json(CURRENT_DIR / "ranking_manifest.json")
    finalized = read_json(ROOT / "reports/submission_v8_finalized_manifest.json")
    current_path = CURRENT_DIR / "predictions.parquet"
    if (manifest.get("predictions_sha256") != sha256(current_path)
            or finalized.get("sha256") != sha256(current_path)
            or manifest.get("rows") != EXPECTED_RANKING_ROWS
            or manifest.get("valid_route", {}).get("active") != "v7"
            or manifest.get("missing_route", {}).get("terminal") != "passed"):
        raise ValueError("The frozen current v8 ranking policy changed")
    return {"selected_route": "current", "verified_current_sha256":
            sha256(current_path), "new_model_fitted": False,
            "new_ranking_output_created": False}


def family_input_paths(route: str) -> dict[str, Path]:
    args = family_args(route)
    module = FAMILY[route]
    if route == "v10":
        paths = module.input_paths(args)
        raw = module._training_files(args.data_dir)
    else:
        raw, paths = module.source_inventory(args)
    result = {f"family_{name}": Path(path) for name, path in paths.items()}
    result.update({f"training_{path.name}": path for path in raw})
    output = args.output_dir
    for name in ("protocol.json", "validation.json", "validation_predictions.parquet",
                 "fresh_audit.json", "fresh_audit_predictions.parquet",
                 "seasonal_jan_jul_validation.json",
                 "forward_nov_dec_validation.json",
                 "seasonal_jan_jul_provenance.json",
                 "forward_nov_dec_provenance.json",
                 "seasonal_jan_jul_oof.parquet",
                 "forward_nov_dec_oof.parquet",
                 "seasonal_jan_jul.cbm", "forward_nov_dec.cbm",
                 "fresh_new/fresh_apr_oct_validation.json",
                 "fresh_new/fresh_apr_oct_provenance.json",
                 "fresh_new/fresh_apr_oct_oof.parquet",
                 "fresh_new/fresh_apr_oct.cbm"):
        result[f"family_output_{name}"] = output / name
    return result


def source_paths() -> dict[str, Path]:
    return {
        "own_final_source": Path(__file__).resolve(),
        "portfolio_source": ROOT / "later_feature_portfolio.py",
        "guard_source": ROOT / "later_reserved_guard.py",
        "v10_source": ROOT / "v10_runway_taxi_expert.py",
        "v11_source": ROOT / "v11_taxi_interval_flow_expert.py",
        "v10_builder_source": ROOT / "runway_arrival_taxi_features.py",
        "v11_builder_source": ROOT / "taxi_interval_flow_features.py",
        "base_loader_source": ROOT / "traffic_deep_expert.py",
        "residual_trainer_source": ROOT / "deep_timestamp_expert.py",
    }


def fit_input_paths(route: str) -> dict[str, Path]:
    paths = family_input_paths(route)
    paths.update({
        "portfolio_protocol": ROOT / "reports/later_feature_portfolio_protocol.json",
        "portfolio_selection": SELECTION_DIR / "selection.json",
        "portfolio_evaluation": SELECTION_DIR / "evaluation.json",
        "portfolio_evaluation_predictions":
            SELECTION_DIR / "evaluation_predictions.parquet",
        "guard_terminal": GUARD_DIR / "terminal.json",
        "guard_protocol": GUARD_DIR / "protocol.json",
        "guard_comparator_receipt": GUARD_DIR / "comparator_receipt.json",
        "guard_replacement_receipt": GUARD_DIR / "replacement_receipt.json",
        "guard_comparator_model": GUARD_DIR / "comparator.cbm",
        "guard_replacement_model": GUARD_DIR / "replacement.cbm",
        "guard_comparator_oof": GUARD_DIR / "comparator_oof.parquet",
        "guard_replacement_oof": GUARD_DIR / "replacement_oof.parquet",
        "guard_comparator_fit": GUARD_DIR / "comparator_fit.json",
        "guard_replacement_fit": GUARD_DIR / "replacement_fit.json",
        "guard_paired_predictions": GUARD_DIR / "paired_predictions.parquet",
        "current_validation": CURRENT_DIR / "composition_report.json",
        "current_ranking": CURRENT_DIR / "predictions.parquet",
        "current_ranking_manifest": CURRENT_DIR / "ranking_manifest.json",
        "current_ranking_sources": CURRENT_DIR / "ranking_sources.json",
        "v7_ranking": V7_DIR / "predictions.parquet",
        "v7_ranking_manifest": V7_DIR / "manifest.json",
    })
    return paths


def prepared_path() -> Path:
    return OUT / "protocol.json"


def assert_family_frozen(route: str) -> None:
    """Bind the original family's named source/cache map before our own seal."""
    args = family_args(route)
    family = FAMILY[route]
    protocol_path = args.output_dir / "protocol.json"
    frozen = read_json(protocol_path)
    protocol_sha = sha256(protocol_path)
    if frozen.get("spec") != family.protocol_spec():
        raise ValueError("Selected original architecture protocol changed")
    if route == "v10":
        family.assert_frozen_inputs(args, frozen, protocol_sha)
    else:
        family.verify_source_snapshot(args, frozen, protocol_sha)


def prepare() -> dict:
    choice, _ = selected_route()
    route = choice["selected_route"]
    if route == "current":
        return verify_current_prediction()
    guard = require_guard(choice)
    assert_family_frozen(route)
    value = {
        "selected_route": route, "selected_weight": choice["selected_weight"],
        "selection_sha256": sha256(SELECTION_DIR / "selection.json"),
        "guard_sha256": sha256(GUARD_DIR / "terminal.json"),
        "source_sha256": file_hashes(source_paths()),
        "fit_input_sha256": file_hashes(fit_input_paths(route)),
        "status": "frozen_before_final_fit",
    }
    path = prepared_path()
    if path.exists():
        if read_json(path) != value:
            raise ValueError("Prepared final source/input seal differs")
    else:
        json_exclusive(path, value)
    verify_prepared(route)
    return value


def verify_prepared(route: str) -> dict:
    value = read_json(prepared_path())
    choice, _ = selected_route()
    require_guard(choice)
    assert_family_frozen(route)
    if (value.get("selected_route") != route
            or value.get("selected_weight") != choice["selected_weight"]
            or value.get("selection_sha256") !=
               sha256(SELECTION_DIR / "selection.json")
            or value.get("guard_sha256") != sha256(GUARD_DIR / "terminal.json")
            or value.get("status") != "frozen_before_final_fit"):
        raise ValueError("Final fit source seal or selected guard changed")
    verify_hashes(source_paths(), value["source_sha256"], "Final code")
    verify_hashes(fit_input_paths(route), value["fit_input_sha256"],
                  "Final training")
    return value


def original_schema_and_rounds(route: str) -> tuple[list[dict], int, list[int]]:
    output = family_args(route).output_dir
    receipts = [read_json(output / f"{fold}_provenance.json") for fold in FOLDS]
    reports = [read_json(output / f"{fold}_validation.json") for fold in FOLDS]
    schema = receipts[0].get("feature_schema")
    if (not isinstance(schema, list) or len(schema) != EXPECTED_FEATURES
            or schema != receipts[1].get("feature_schema")
            or len({item["name"] for item in schema}) != EXPECTED_FEATURES
            or any(item["name"] in {"MVT_ID_mvt", "target", "BLOCK_TIME_UTC_mvt",
                                     "TAXITIME_SEC_mvt"} for item in schema)):
        raise ValueError("Original 184-feature schema changed")
    cats = [i for i, item in enumerate(schema) if item["dtype"] == "category"]
    guard_protocol = read_json(GUARD_DIR / "protocol.json")
    guard_receipt = read_json(GUARD_DIR / "replacement_receipt.json")
    if (not cats
            or guard_protocol.get("replacement_feature_schema") != schema
            or guard_receipt.get("feature_schema") != schema
            or guard_receipt.get("categorical_feature_indices") != cats
            or set(guard_receipt.get("categorical_vocabularies", {})) !=
               {schema[i]["name"] for i in cats}):
        raise ValueError("Original and reserved categorical schemas differ")
    family = FAMILY[route]
    if schema[-10:] != [{"name": name, "dtype": "float32"}
                        for name in family.TAXI_FEATURES]:
        raise ValueError("Original ten taxi feature names or dtypes changed")
    trees = []
    for fold, receipt, report in zip(FOLDS, receipts, reports):
        model = output / f"{fold}.cbm"
        if (receipt.get("fold") != fold or receipt.get("heldout_months") !=
            list(deep.FOLDS[fold]) or receipt.get("trees") != report.get("trees")
            or receipt.get("fit_report_sha256") !=
               sha256(output / f"{fold}_validation.json")
            or receipt.get("model_sha256") != sha256(model)
            or receipt.get("categorical_feature_indices") != cats
            or receipt.get("catboost_params") != family.EXPECTED_CATBOOST_PARAMS
            or report.get("features") != [item["name"] for item in schema]
            or not isinstance(report.get("trees"), int)
            or not 1 <= report["trees"] <= 10000):
            raise ValueError(f"{fold} original model provenance changed")
        trees.append(report["trees"])
    return schema, int(np.median(trees)), trees


def check_feature_schema(features: pd.DataFrame, schema: list[dict]) -> list[str]:
    actual = [{"name": str(name), "dtype": str(features[name].dtype)}
              for name in features]
    if actual != schema:
        raise ValueError("Final feature names, order or dtypes differ from original folds")
    return [item["name"] for item in schema if item["dtype"] == "category"]


def model_report_path() -> Path:
    return OUT / "final_model.json"


def model_path() -> Path:
    return OUT / "full_2025.cbm"


def verify_model(route: str, schema: list[dict], rounds: int) -> dict:
    report = read_json(model_report_path())
    guard_receipt = read_json(GUARD_DIR / "replacement_receipt.json")
    if (report.get("selected_route") != route
            or report.get("iterations") != rounds
            or report.get("feature_schema") != schema
            or report.get("model_sha256") != sha256(model_path())
            or report.get("prepared_protocol_sha256") != sha256(prepared_path())
            or report.get("selection_sha256") !=
               sha256(SELECTION_DIR / "selection.json")
            or report.get("guard_sha256") != sha256(GUARD_DIR / "terminal.json")
            or report.get("fit_input_sha256") !=
               read_json(prepared_path())["fit_input_sha256"]
            or report.get("categorical_vocabularies") !=
               guard_receipt.get("categorical_vocabularies")):
        raise ValueError("Final model report or source binding differs")
    model = CatBoostRegressor()
    model.load_model(str(model_path()))
    cats = [i for i, item in enumerate(schema) if item["dtype"] == "category"]
    if (int(model.tree_count_) != rounds
            or list(model.feature_names_) != [item["name"] for item in schema]
            or list(model.get_cat_feature_indices()) != cats):
        raise ValueError("Final CatBoost model metadata differs from frozen architecture")
    FAMILY[route].verify_saved_params(model.get_all_params())
    return report


def fit_final(min_free_gib: float) -> dict:
    choice, _ = selected_route()
    route = choice["selected_route"]
    if route == "current":
        return verify_current_prediction()
    if route == "v10":
        v10.require_memory(max(10.0, min_free_gib))
    else:
        v11.flow.require_memory(max(10.0, min_free_gib))
    if model_path().exists() or model_report_path().exists():
        raise FileExistsError("Later final model or report already exists")
    verify_prepared(route)
    schema, rounds, original_trees = original_schema_and_rounds(route)
    args = family_args(route)
    rows, features = FAMILY[route].load_features(args)
    cats = check_feature_schema(features, schema)
    categories_frozen = reserved.category_hashes(features)
    if categories_frozen != read_json(
            GUARD_DIR / "replacement_receipt.json").get("categorical_vocabularies"):
        raise ValueError("Full training category vocabularies differ from paired guard")
    if (len(rows) != EXPECTED_TRAINING_ROWS or len(features) != len(rows)
            or rows.MVT_ID_mvt.isna().any()
            or rows.MVT_ID_mvt.duplicated().any()):
        raise ValueError("Full 2025 training feature/ID universe changed")
    y = rows.target.to_numpy(dtype=float)
    proxy = rows.proxy.to_numpy(dtype=float)
    eligible = (np.isfinite(y) & (y >= 0) & (y <= 86400)
                & np.isfinite(proxy) & (proxy >= 0) & (proxy <= 7200))
    if eligible.sum() < 1_500_000:
        raise ValueError("Unexpectedly few eligible 2025 training departures")
    verify_prepared(route)
    params = dict(FAMILY[route].EXPECTED_CATBOOST_PARAMS)
    params["iterations"] = rounds
    model = CatBoostRegressor(**params)
    train = Pool(features.loc[eligible], label=(y - proxy)[eligible],
                 cat_features=cats)
    start = time.monotonic()
    model.fit(train)
    fit_seconds = time.monotonic() - start
    del train, rows, features
    gc.collect()
    if (int(model.tree_count_) != rounds
            or list(model.feature_names_) != [item["name"] for item in schema]
            or list(model.get_cat_feature_indices()) !=
               [i for i, item in enumerate(schema) if item["dtype"] == "category"]):
        raise ValueError("Full 2025 model differs from original architecture")
    FAMILY[route].verify_saved_params(model.get_all_params())
    verify_prepared(route)
    OUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".later_model_", dir=OUT) as work:
        stage = Path(work) / "full_2025.cbm"
        model.save_model(str(stage))
        model_sha = sha256(stage)
        verify_prepared(route)
        os.link(stage, model_path())
    if sha256(model_path()) != model_sha:
        raise ValueError("Published model bytes differ from staged model")
    report = {
        "selected_route": route, "selected_weight": choice["selected_weight"],
        "iterations": rounds, "original_fold_trees": dict(zip(FOLDS, original_trees)),
        "training_rows": EXPECTED_TRAINING_ROWS,
        "eligible_training_rows": int(eligible.sum()),
        "feature_schema": schema,
        "categorical_feature_indices":
            [i for i, item in enumerate(schema) if item["dtype"] == "category"],
        "categorical_vocabularies": categories_frozen,
        "fixed_catboost_params": params,
        "fit_seconds": fit_seconds, "model_sha256": model_sha,
        "prepared_protocol_sha256": sha256(prepared_path()),
        "selection_sha256": sha256(SELECTION_DIR / "selection.json"),
        "guard_sha256": sha256(GUARD_DIR / "terminal.json"),
        "fit_input_sha256": read_json(prepared_path())["fit_input_sha256"],
        "ranking_prediction_created": False,
    }
    json_exclusive(model_report_path(), report)
    verify_prepared(route)
    verify_model(route, schema, rounds)
    return {key: report[key] for key in ("selected_route", "iterations",
                                        "original_fold_trees", "training_rows",
                                        "eligible_training_rows", "model_sha256")}


def ranking_input_paths(route: str) -> dict[str, Path]:
    args = family_args(route)
    taxi_name = ("ranking_runway_arrival_taxi_features.parquet" if route == "v10"
                 else "ranking_taxi_interval_flow_features.parquet")
    paths = {
        "raw_ranking": args.data_dir / "ranking.parquet",
        "template": TEMPLATE,
        "baseline_ranking_rows": args.cache_dir / "ranking_rows.parquet",
        "baseline_ranking_features": args.cache_dir / "ranking_features.parquet",
        "weather": args.weather_file,
        "arrival_ranking": args.arrival_dir / "ranking_arrival_features.parquet",
        "neighbour_ranking": args.neighbour_dir / "ranking_neighbour_features.parquet",
        "runway_sequence_ranking": args.runway_dir / "ranking_runway_arrival_features.parquet",
        "taxi_ranking": args.taxi_dir / taxi_name,
        "taxi_builder_protocol": args.taxi_dir / "protocol.json",
        "taxi_builder_manifest": args.taxi_dir / "feature_build.json",
        "v7_ranking": V7_DIR / "predictions.parquet",
        "v7_ranking_manifest": V7_DIR / "manifest.json",
        "current_ranking": CURRENT_DIR / "predictions.parquet",
        "current_ranking_manifest": CURRENT_DIR / "ranking_manifest.json",
        "current_ranking_sources": CURRENT_DIR / "ranking_sources.json",
        "final_model": model_path(),
        "final_model_report": model_report_path(),
        "final_protocol": prepared_path(),
        "selection": SELECTION_DIR / "selection.json",
        "selection_evaluation": SELECTION_DIR / "evaluation.json",
        "guard_protocol": GUARD_DIR / "protocol.json",
        "guard_terminal": GUARD_DIR / "terminal.json",
    }
    return paths


def ranking_input_path() -> Path:
    return OUT / "ranking_inputs.json"


def freeze_ranking_inputs() -> dict:
    choice, _ = selected_route()
    route = choice["selected_route"]
    if route == "current":
        return verify_current_prediction()
    verify_prepared(route)
    schema, rounds, _ = original_schema_and_rounds(route)
    verify_model(route, schema, rounds)
    paths = ranking_input_paths(route)
    value = {
        "selected_route": route,
        "selected_weight": choice["selected_weight"],
        "prepared_protocol_sha256": sha256(prepared_path()),
        "source_sha256": file_hashes(source_paths()),
        "ranking_input_sha256": file_hashes(paths),
        "status": "sealed_before_any_ranking_feature_read",
    }
    path = ranking_input_path()
    if path.exists():
        if read_json(path) != value:
            raise ValueError("Previously frozen ranking inputs differ")
    else:
        json_exclusive(path, value)
    verify_ranking_inputs(route)
    return value


def verify_ranking_inputs(route: str) -> dict:
    value = read_json(ranking_input_path())
    choice, _ = selected_route()
    if (value.get("selected_route") != route
            or value.get("selected_weight") != choice["selected_weight"]
            or value.get("prepared_protocol_sha256") != sha256(prepared_path())
            or value.get("status") != "sealed_before_any_ranking_feature_read"):
        raise ValueError("Frozen ranking source seal differs")
    verify_hashes(source_paths(), value["source_sha256"], "Ranking code")
    verify_hashes(ranking_input_paths(route), value["ranking_input_sha256"],
                  "Ranking source")
    verify_prepared(route)
    return value


def load_ranking_features(route: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reuse the accepted 174-field loader and append exactly ten new fields."""
    args = family_args(route)
    rows, features = traffic.load_features(args, ranking=True)
    names = list(FAMILY[route].TAXI_FEATURES)
    taxi_file = (args.taxi_dir / "ranking_runway_arrival_taxi_features.parquet"
                 if route == "v10" else
                 args.taxi_dir / "ranking_taxi_interval_flow_features.parquet")
    built = read_json(args.taxi_dir / "feature_build.json")
    expected_hash = (built["ranking"]["output_sha256"] if route == "v10"
                     else built["ranking"]["sha256"])
    if expected_hash != sha256(taxi_file):
        raise ValueError("Selected label-free ranking taxi cache changed")
    taxi = pd.read_parquet(taxi_file)
    if (list(taxi) != ["MVT_ID_mvt", *names]
            or len(taxi) != len(rows)
            or not np.array_equal(taxi.MVT_ID_mvt.to_numpy(),
                                  rows.MVT_ID_mvt.to_numpy())
            or any(name in features for name in names)
            or any(taxi[name].dtype != np.dtype("float32") for name in names)
            or np.isinf(taxi[names].to_numpy(dtype=float)).any()):
        raise ValueError("Selected ten taxi ranking features lack exact schema/ID alignment")
    features = pd.concat([features.reset_index(drop=True),
                          taxi[names].reset_index(drop=True)], axis=1)
    if (len(features) != EXPECTED_RANKING_ROWS
            or features.columns.duplicated().any()
            or any(name in features for name in ("MVT_ID_mvt", "target",
                                                 "BLOCK_TIME_UTC_mvt",
                                                 "TAXITIME_SEC_mvt"))):
        raise ValueError("Ranking predictor matrix changed or contains forbidden fields")
    return rows, features


def verify_current_template() -> dict:
    result = verify_current_prediction()
    template = pd.read_parquet(TEMPLATE)
    current = pd.read_parquet(CURRENT_DIR / "predictions.parquet")
    if (list(template) != ["MVT_ID_mvt", "TAXITIME_SEC_mvt"]
            or list(current) != list(template)
            or len(template) != EXPECTED_RANKING_ROWS
            or not np.array_equal(current.MVT_ID_mvt.to_numpy(),
                                  template.MVT_ID_mvt.to_numpy())
            or template.MVT_ID_mvt.isna().any()
            or template.MVT_ID_mvt.duplicated().any()
            or not np.isfinite(current.TAXITIME_SEC_mvt.to_numpy(dtype=float)).all()
            or (current.TAXITIME_SEC_mvt.to_numpy(dtype=float) < 0).any()):
        raise ValueError("Current route is not the exact valid template prediction")
    return result


def final_predict(min_free_gib: float) -> dict:
    choice, _ = selected_route()
    route = choice["selected_route"]
    if route == "current":
        return verify_current_template()
    if route == "v10":
        v10.require_memory(max(10.0, min_free_gib))
    else:
        v11.flow.require_memory(max(10.0, min_free_gib))
    output_path = OUT / "predictions.parquet"
    expert_path = OUT / "ranking_expert.parquet"
    manifest_path = OUT / "ranking_manifest.json"
    if any(path.exists() for path in (output_path, expert_path, manifest_path)):
        raise FileExistsError("Final ranking output already exists; never overwrite")
    verify_ranking_inputs(route)
    schema, rounds, _ = original_schema_and_rounds(route)
    final = verify_model(route, schema, rounds)
    rows, features = load_ranking_features(route)
    check_feature_schema(features, schema)
    if (len(rows) != EXPECTED_RANKING_ROWS
            or rows.MVT_ID_mvt.isna().any()
            or rows.MVT_ID_mvt.duplicated().any()):
        raise ValueError("Ranking row IDs or feature coverage changed")
    template = pd.read_parquet(TEMPLATE)
    v7_ref = pd.read_parquet(V7_DIR / "predictions.parquet")
    current_ref = pd.read_parquet(CURRENT_DIR / "predictions.parquet")
    required = ["MVT_ID_mvt", "TAXITIME_SEC_mvt"]
    if (list(template) != required or list(v7_ref) != required
            or list(current_ref) != required
            or len(template) != EXPECTED_RANKING_ROWS
            or template.MVT_ID_mvt.isna().any()
            or template.MVT_ID_mvt.duplicated().any()
            or any(not np.array_equal(item.MVT_ID_mvt.to_numpy(),
                                      template.MVT_ID_mvt.to_numpy())
                   for item in (rows, v7_ref, current_ref))):
        raise ValueError("Ranking feature/reference IDs lack exact template order")
    proxy = rows.proxy.to_numpy(dtype=float)
    valid = np.isfinite(proxy) & (proxy >= 0) & (proxy <= 7200)
    if int(valid.sum()) != EXPECTED_VALID_RANKING:
        raise ValueError("Valid-AOBT ranking gate differs from frozen current policy")
    current = current_ref.TAXITIME_SEC_mvt.to_numpy(dtype=float, copy=True)
    v7_values = v7_ref.TAXITIME_SEC_mvt.to_numpy(dtype=float, copy=True)
    if (not np.isfinite(current).all() or (current < 0).any()
            or not np.isfinite(v7_values).all() or (v7_values < 0).any()
            or not np.array_equal(current[valid], v7_values[valid])):
        raise ValueError("Accepted current/v7 valid-row prediction policy changed")
    verify_ranking_inputs(route)
    model = CatBoostRegressor()
    model.load_model(str(model_path()))
    FAMILY[route].verify_saved_params(model.get_all_params())
    if (int(model.tree_count_) != rounds
            or list(model.feature_names_) != [item["name"] for item in schema]
            or list(model.get_cat_feature_indices()) !=
               [i for i, item in enumerate(schema) if item["dtype"] == "category"]):
        raise ValueError("Saved full model differs from original feature/category architecture")
    raw = np.full(len(rows), np.nan, dtype=float)
    raw[valid] = proxy[valid] + model.predict(features.loc[valid], thread_count=2)
    if not np.isfinite(raw[valid]).all():
        raise ValueError("Selected model lacks finite coverage on valid AOBT")
    output = fixed_blend(current, v7_values, raw, valid,
                         float(choice["selected_weight"]))
    verify_ranking_inputs(route)
    expert = pd.DataFrame({"MVT_ID_mvt": template.MVT_ID_mvt.to_numpy(copy=True),
                           "a_valid": valid, "raw_expert": raw})
    expert_sha = parquet_exclusive(expert_path, expert)
    result = pd.DataFrame({"MVT_ID_mvt": template.MVT_ID_mvt.to_numpy(copy=True),
                           "TAXITIME_SEC_mvt": output})
    prediction_sha = parquet_exclusive(output_path, result)
    readback = pd.read_parquet(output_path)
    if (list(readback) != required
            or not np.array_equal(readback.MVT_ID_mvt.to_numpy(),
                                  template.MVT_ID_mvt.to_numpy())
            or not np.array_equal(readback.TAXITIME_SEC_mvt.to_numpy(dtype=float),
                                  output)
            or not np.array_equal(output[~valid], current[~valid])
            or not np.isfinite(output).all() or (output < 0).any()):
        raise ValueError("Final ranking readback, nonvalid rows or values differ")
    verify_ranking_inputs(route)
    report = {
        "selected_route": route, "selected_weight": choice["selected_weight"],
        "rows": len(result), "valid_aobt_rows": int(valid.sum()),
        "outside_valid_unchanged": True, "template_order_verified": True,
        "finite_nonnegative": True,
        "current_missing_clock_policy_preserved": True,
        "final_model_sha256": final["model_sha256"],
        "final_model_report_sha256": sha256(model_report_path()),
        "selection_sha256": sha256(SELECTION_DIR / "selection.json"),
        "guard_sha256": sha256(GUARD_DIR / "terminal.json"),
        "ranking_inputs_sha256": sha256(ranking_input_path()),
        "ranking_input_sha256": read_json(ranking_input_path())["ranking_input_sha256"],
        "ranking_expert_sha256": expert_sha,
        "predictions_sha256": prediction_sha,
        "prediction_bytes": output_path.stat().st_size,
        "uploaded": False,
    }
    json_exclusive(manifest_path, report)
    verify_ranking_inputs(route)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True,
                        choices=("prepare", "fit-final", "freeze-ranking-inputs",
                                 "final-predict"))
    parser.add_argument("--min-free-gib", type=float, default=10.0)
    args = parser.parse_args()
    if not np.isfinite(args.min_free_gib) or args.min_free_gib < 10.0:
        raise ValueError("Later full fit and ranking require at least 10 GiB free")
    if args.mode == "prepare":
        result = prepare()
    elif args.mode == "fit-final":
        result = fit_final(args.min_free_gib)
    elif args.mode == "freeze-ranking-inputs":
        result = freeze_ranking_inputs()
    else:
        result = final_predict(args.min_free_gib)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
