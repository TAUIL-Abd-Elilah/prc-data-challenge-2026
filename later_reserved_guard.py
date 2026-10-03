"""Fixed May/September architecture guard for the later-feature portfolio.

Freeze a single previously selected route and weight before reading either
reserved month. The two optional refits use identical complementary 2025
training rows and the unchanged v7 CatBoost residual recipe. This module never
creates ranking predictions or chooses a second alternative after a failure.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import tempfile
import uuid
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
from catboost import CatBoostRegressor, Pool

import deep_timestamp_expert as deep
import traffic_deep_expert as traffic
import v10_runway_taxi_expert as v10
import v11_taxi_interval_flow_expert as v11
from solution import _training_files


ROOT = Path(__file__).resolve().parent
PORTFOLIO = ROOT / "reports/later_feature_portfolio_protocol.json"
SELECTION = ROOT / "artifacts/later-feature-portfolio/selection.json"
DEFAULT_OUT = ROOT / "artifacts/later-reserved-guard"
HELDOUT = (5, 9)
WEIGHTS = (0.0, 0.1, 0.25, 0.5, 1.0)
BOOTSTRAP_SEED = 20261014
BOOTSTRAP_REPEATS = 1000
MIN_FREE_GIB = 10.0
RAW_COLUMNS = ("MVT_ID_mvt", "target", "MVT_TIME_UTC_mvt", "expert")
ROUTE_MODULES = {"v10": v10, "v11": v11}
ROUTE_DIRS = {"v10": ROOT / "artifacts/v10-runway-taxi",
              "v11": ROOT / "artifacts/v11-taxi-flow-expert"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def relative(path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(ROOT).as_posix()
    except ValueError as error:
        raise ValueError(f"Source path escapes workspace: {path}") from error


def json_file(path: Path) -> dict:
    result = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return result


def write_new_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def publish_temp_exclusive(target: Path, producer) -> None:
    """Finish a fully written artifact before linking it at an unused path."""
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + f".{uuid.uuid4().hex}.tmp")
    try:
        producer(temporary)
        os.link(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def id_hash(values: pd.Series) -> str:
    if values.isna().any() or values.duplicated().any():
        raise ValueError("ID receipt requires exact unique nonmissing IDs")
    array = np.ascontiguousarray(values.to_numpy(copy=True))
    if array.dtype.kind not in "iuf":
        raise ValueError("ID receipt requires numeric movement IDs")
    header = f"{array.dtype}:{len(array)}:".encode("ascii")
    return hashlib.sha256(header + array.tobytes()).hexdigest()


def assert_hashes(mapping: dict[str, str]) -> None:
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("Frozen input mapping is empty")
    for name, digest in mapping.items():
        if (not isinstance(name, str) or name.startswith("/") or "\\" in name
                or ".." in Path(name).parts or relative(ROOT / name) != name
                or not isinstance(digest, str)
                or not re.fullmatch(r"[a-f0-9]{64}", digest)
                or sha256(ROOT / name) != digest):
            raise ValueError(f"Frozen source/input changed: {name}")


def source_paths(args: argparse.Namespace, route: str) -> dict[str, Path]:
    paths = {
        "guard_source": Path(__file__).resolve(),
        "portfolio_protocol": PORTFOLIO,
        "portfolio_selector": ROOT / "later_feature_portfolio.py",
        "portfolio_selection": SELECTION,
        "deep_trainer": Path(deep.__file__).resolve(),
        "traffic_loader": Path(traffic.__file__).resolve(),
        "arrival_loader": ROOT / "deep_arrival_expert.py",
        "cache_loader": ROOT / "catboost_expert.py",
        "base_features": ROOT / "solution.py",
        "weather_loader": ROOT / "weather_model.py",
        "flight_loader": ROOT / "airport_models.py",
        "baseline_rows": args.cache_dir / "training_rows.parquet",
        "baseline_features": args.cache_dir / "features.parquet",
        "arrival_features": args.arrival_dir / "training_arrival_features.parquet",
        "neighbour_features": args.neighbour_dir / "training_neighbour_features.parquet",
        "runway_features": args.runway_dir / "training_runway_arrival_features.parquet",
        "weather": args.weather_file,
        "v7_protocol": ROOT / "artifacts/v7-runway-traffic/protocol.json",
        "v7_validation": ROOT / "artifacts/v7-runway-traffic/validation.json",
        "v7_manifest": ROOT / "artifacts/v7-runway-traffic/manifest.json",
        "v7_original_oof": ROOT / "artifacts/v7-runway-traffic/validation_predictions.parquet",
        "v7_fresh_report": ROOT / "artifacts/v7-runway-traffic/fresh_audit.json",
        "current_policy": ROOT / "reports/current_selected_policy.json",
        "current_composition": ROOT / "reports/current_composition_validation.json",
        "current_oof": ROOT / "artifacts/current-candidate/validation_predictions.parquet",
    }
    if route != "current":
        module = ROUTE_MODULES[route]
        directory = ROUTE_DIRS[route]
        paths.update({
            "replacement_source": Path(module.__file__).resolve(),
            "replacement_protocol": directory / "protocol.json",
            "replacement_validation": directory / "validation.json",
            "replacement_fresh_report": directory / "fresh_audit.json",
            "replacement_original_oof": directory / "validation_predictions.parquet",
            "replacement_fresh_oof": directory / "fresh_audit_predictions.parquet",
            "replacement_seasonal_receipt": directory / "seasonal_jan_jul_provenance.json",
            "replacement_forward_receipt": directory / "forward_nov_dec_provenance.json",
            "replacement_fresh_receipt": directory / "fresh_new/fresh_apr_oct_provenance.json",
            "replacement_builder_source": ROOT / (
                "runway_arrival_taxi_features.py" if route == "v10"
                else "taxi_interval_flow_features.py"),
            "replacement_feature_cache": args.taxi_dir /
                ("training_runway_arrival_taxi_features.parquet" if route == "v10"
                 else "training_taxi_interval_flow_features.parquet"),
            "replacement_build_protocol": args.taxi_dir / "protocol.json",
            "replacement_build_manifest": args.taxi_dir / "feature_build.json",
        })
    return paths


def source_hashes(args: argparse.Namespace, route: str) -> dict[str, str]:
    raw = _training_files(args.data_dir)
    if len(raw) != 12 or len({path.name for path in raw}) != 12:
        raise ValueError("Exactly twelve canonical 2025 training files are required")
    paths = {relative(path): path for path in source_paths(args, route).values()}
    for path in raw:
        paths[relative(path)] = path
    return {key: sha256(path) for key, path in sorted(paths.items())}


def fit_artifact_hashes(args: argparse.Namespace, names: tuple[str, ...]) -> dict[str, str]:
    return {f"{name}{suffix}": sha256(args.output_dir / f"{name}{suffix}")
            for name in names
            for suffix in (".cbm", "_oof.parquet", "_fit.json", "_receipt.json")}


def fixed_params(args: argparse.Namespace) -> dict:
    if (args.depth != 10 or args.iterations != 10000 or args.threads != 2
            or not np.isfinite(args.min_free_gib) or args.min_free_gib < 10):
        raise ValueError("Guard requires depth 10, 10000 rounds, two threads, and a 10 GiB fit floor")
    params = deep.params(args)
    if params != v10.EXPECTED_CATBOOST_PARAMS or params != v11.EXPECTED_CATBOOST_PARAMS:
        raise ValueError("Frozen v7/v10/v11 CatBoost parameters diverged")
    return params


def verify_selected_inputs(args: argparse.Namespace, route: str) -> None:
    """Bind the new refit to the exact bytes used by its selected original models."""
    if route == "current":
        return
    selected = argparse.Namespace(**vars(args))
    selected.output_dir = ROUTE_DIRS[route]
    selected.v7_dir = ROOT / "artifacts/v7-runway-traffic"
    selected.v6_dir = ROOT / "artifacts/v6-deep-arrival"
    protocol_path = selected.output_dir / "protocol.json"
    protocol_sha = sha256(protocol_path)
    frozen = json_file(protocol_path)
    if route == "v10":
        v10.assert_frozen_inputs(selected, frozen, protocol_sha)
    else:
        v11.verify_source_snapshot(selected, frozen, protocol_sha)
    if sha256(protocol_path) != protocol_sha:
        raise ValueError("Selected original model protocol changed during verification")


def require_memory(args: argparse.Namespace) -> None:
    needed = max(MIN_FREE_GIB, float(args.min_free_gib))
    free = psutil.virtual_memory().available / (1024 ** 3)
    if free < needed:
        raise MemoryError(f"Free RAM {free:.2f} GiB is below fixed {needed:.2f} GiB fit floor")


def schema(rows: pd.DataFrame) -> list[dict[str, str]]:
    return [{"name": str(column), "dtype": str(rows[column].dtype)}
            for column in rows]


def schema_hash(value: list[dict[str, str]]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def category_hashes(x: pd.DataFrame) -> dict[str, dict[str, object]]:
    """Bind the exact ordered category vocabulary, not just category dtype."""
    result = {}
    for name in x.select_dtypes(include="category"):
        index = x[name].cat.categories
        labels = [None if pd.isna(item) else [type(item).__name__, str(item)]
                  for item in index.tolist()]
        if len(labels) != len({json.dumps(item, ensure_ascii=False) for item in labels}):
            raise ValueError(f"Ambiguous categorical feature levels: {name}")
        encoded = json.dumps(labels, ensure_ascii=False, separators=(",", ":"))
        result[name] = {"count": len(labels),
                        "ordered": bool(x[name].cat.ordered),
                        "sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest()}
    return result


def exact_ids(actual: pd.Series, expected: pd.Series, label: str) -> None:
    a, b = pd.Index(actual), pd.Index(expected)
    if (len(a) != len(b) or a.has_duplicates or b.has_duplicates
            or a.isna().any() or b.isna().any()
            or not a.isin(b).all() or not b.isin(a).all()):
        raise ValueError(f"{label}: exact unique ID coverage failed")


def fixed_split(rows: pd.DataFrame, x: pd.DataFrame,
                expected_schema: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray,
                                                                  np.ndarray, np.ndarray]:
    if (not {"MVT_ID_mvt", "target", "proxy", "month", "time"}.issubset(rows)
            or len(rows) != len(x) or rows.MVT_ID_mvt.isna().any()
            or rows.MVT_ID_mvt.duplicated().any()
            or schema(x) != expected_schema or x.columns.duplicated().any()
            or any(name in x for name in ("target", "MVT_ID_mvt",
                                           "BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt"))):
        raise ValueError("Held-out rows or predictor schema differ from frozen input")
    time_value = pd.to_datetime(rows.time, utc=True, errors="coerce")
    if (time_value.isna().any() or not time_value.dt.year.eq(2025).all()
            or not np.array_equal(time_value.dt.month.to_numpy(),
                                  rows.month.to_numpy())):
        raise ValueError("Month and UTC movement timestamp disagree")
    y = rows.target.to_numpy(dtype=float)
    proxy = rows.proxy.to_numpy(dtype=float)
    valid = np.isfinite(proxy) & (proxy >= 0) & (proxy <= 7200)
    held = rows.month.isin(HELDOUT).to_numpy()
    train = np.flatnonzero(~held & valid & np.isfinite(y)
                           & (y >= 0) & (y <= 86400))
    test = np.flatnonzero(held & valid & np.isfinite(y))
    if (len(train) < 100000 or len(test) < 10000
            or np.intersect1d(train, test).size
            or set(rows.month.iloc[test].unique()) != set(HELDOUT)):
        raise ValueError("Fixed training or complete held-out mask is invalid")
    return train, test, y, proxy


def selection_fields(selection: dict) -> tuple[str, float]:
    route, weight = selection.get("selected_route"), selection.get("selected_weight")
    if route not in ("current", "v10", "v11"):
        raise ValueError("Selector has no single terminal route")
    if (not isinstance(weight, (int, float)) or isinstance(weight, bool)
            or not np.isfinite(weight) or float(weight) not in WEIGHTS
            or (route == "current") != (float(weight) == 0.0)):
        raise ValueError("Selector weight differs from prospectively frozen route")
    return route, float(weight)


def verify_selection(selection: dict) -> tuple[str, float]:
    """Cross-check selector's terminal report before any reserved label access."""
    import later_feature_portfolio as portfolio

    verified = portfolio.require_selection(
        expected_source_sha256=sha256(Path(portfolio.__file__)))
    if verified != selection:
        raise ValueError("Selector verification returned different sealed data")
    _, portfolio_sha = portfolio.verify_protocol()
    route, weight = selection_fields(selection)
    if (selection.get("portfolio_protocol_sha256") != portfolio_sha
            or selection.get("portfolio_source_sha256") != sha256(Path(portfolio.__file__))
            or selection.get("evaluation_sha256") != sha256(
                ROOT / "artifacts/later-feature-portfolio/evaluation.json")
            or selection.get("evaluation_predictions_sha256") != sha256(
                ROOT / "artifacts/later-feature-portfolio/evaluation_predictions.parquet")
            or selection.get("current_oof_sha256") != sha256(
                ROOT / "artifacts/current-candidate/validation_predictions.parquet")):
        raise ValueError("Selected portfolio input, code, or terminal result changed")
    if route != "current":
        selected_dir = ROUTE_DIRS[route]
        original = json_file(selected_dir / "validation.json")
        fresh = json_file(selected_dir / "fresh_audit.json")
        if (not original.get("existing_folds_passed")
                or not original.get("fresh_audit_passed")
                or not fresh.get("passed") or not fresh.get("coverage_verified")
                or original.get("selected_weight") != weight
                or fresh.get("weight") != weight
                or selection.get("selected_original_validation_sha256") !=
                   sha256(selected_dir / "validation.json")
                or selection.get("selected_fresh_audit_sha256") !=
                   sha256(selected_dir / "fresh_audit.json")):
            raise ValueError("Selected route lacks fixed original/fresh gates")
        item = selection.get("terminal_routes", {}).get(route, {})
        if (item.get("original_passed") is not True
                or item.get("fresh_passed") is not True
                or item.get("compatibility_passed") is not True):
            raise ValueError("Selected route lacks terminal compatibility gate")
    else:
        if (selection.get("selected_original_validation_sha256") is not None
                or selection.get("selected_fresh_audit_sha256") is not None):
            raise ValueError("Current route must not bind replacement gates")
    return route, weight


def freeze(args: argparse.Namespace) -> dict:
    """Create an immutable source/selection seal without reading reserved labels."""
    fixed_params(args)
    if args.output_dir.resolve() != DEFAULT_OUT.resolve():
        raise ValueError("Reserved guard output must use its published artifact directory")
    portfolio = json_file(PORTFOLIO)
    published = portfolio["frozen_input_sha256"]
    assert_hashes(published)
    if (portfolio.get("reserved_guard", {}).get("months") != list(HELDOUT)
            or portfolio["reserved_guard"]["bootstrap"] !=
               {"repeats": BOOTSTRAP_REPEATS, "seed": BOOTSTRAP_SEED}):
        raise ValueError("Published May/September guard specification changed")
    selection_sha = sha256(SELECTION)
    selection = json_file(SELECTION)
    route, weight = verify_selection(selection)
    verify_selected_inputs(args, route)
    if sha256(SELECTION) != selection_sha:
        raise ValueError("Portfolio selection changed while freezing")
    feature_names = list(json_file(ROOT / "artifacts/v7-runway-traffic/manifest.json")
                         ["feature_names"])
    if len(feature_names) != 174 or len(set(feature_names)) != 174:
        raise ValueError("Frozen v7 must have exactly 174 distinct features")
    if route == "current":
        replacement = []
        v7_schema = []
        replacement_schema = []
    else:
        module = ROUTE_MODULES[route]
        replacement = feature_names + list(module.TAXI_FEATURES)
        if (len(replacement) != 184 or len(set(replacement)) != 184
                or json_file(ROUTE_DIRS[route] / "protocol.json")["spec"] !=
                   module.protocol_spec()):
            raise ValueError("Selected replacement is not the frozen 184-feature architecture")
        receipts = [json_file(ROUTE_DIRS[route] / f"{fold}_provenance.json")
                    for fold in ("seasonal_jan_jul", "forward_nov_dec")]
        receipts.append(json_file(ROUTE_DIRS[route] /
                                  "fresh_new/fresh_apr_oct_provenance.json"))
        replacement_schema = receipts[0]["feature_schema"]
        if (any(receipt.get("feature_schema") != replacement_schema
                or receipt.get("catboost_params") != fixed_params(args)
                for receipt in receipts)
                or [item["name"] for item in replacement_schema] != replacement):
            raise ValueError("Original/fresh selected model schemas or params differ")
        v7_schema = replacement_schema[:174]
    hashes = source_hashes(args, route)
    assert_hashes(published)
    if (sha256(SELECTION) != selection_sha
            or hashes != source_hashes(args, route)):
        raise ValueError("Freeze inputs changed while hashing")
    verify_selected_inputs(args, route)
    value = {
        "schema_version": 1,
        "heldout_months": list(HELDOUT),
        "selected_route": route,
        "selected_weight": weight,
        "portfolio_protocol_sha256": sha256(PORTFOLIO),
        "portfolio_selection_sha256": selection_sha,
        "portfolio_source_sha256": selection["portfolio_source_sha256"],
        "published_inputs_sha256": published,
        "frozen_sources_sha256": hashes,
        "catboost_params": fixed_params(args),
        "v7_feature_names": feature_names,
        "v7_feature_schema": v7_schema,
        "replacement_feature_names": replacement,
        "replacement_feature_schema": replacement_schema,
        "training_exclusion": "May and September excluded from both fit and internal early stopping",
        "reserved_labels_used_to_freeze": False,
    }
    path = args.output_dir / "protocol.json"
    if path.exists():
        if json_file(path) != value:
            raise ValueError("Frozen reserved protocol conflicts with current sources")
    else:
        write_new_json(path, value)
    if route == "current":
        terminal = {"schema_version": 1, "route": "current",
                    "weight": 0.0, "heldout_months": list(HELDOUT),
                    "passed": True, "replacement_fit": False,
                    "protocol_sha256": sha256(path),
                    "selection_sha256": selection_sha,
                    "retained_policy": "current"}
        terminal_path = args.output_dir / "terminal.json"
        if terminal_path.exists():
            if json_file(terminal_path) != terminal:
                raise ValueError("Existing current-route terminal seal differs")
        else:
            write_new_json(terminal_path, terminal)
    return value


def check_frozen(args: argparse.Namespace) -> dict:
    if args.output_dir.resolve() != DEFAULT_OUT.resolve():
        raise ValueError("Reserved guard output must use its published artifact directory")
    fixed = json_file(args.output_dir / "protocol.json")
    if (fixed.get("heldout_months") != list(HELDOUT)
            or fixed.get("reserved_labels_used_to_freeze") is not False
            or fixed.get("catboost_params") != fixed_params(args)
            or sha256(PORTFOLIO) != fixed.get("portfolio_protocol_sha256")
            or sha256(SELECTION) != fixed.get("portfolio_selection_sha256")):
        raise ValueError("Reserved protocol or selected route changed")
    assert_hashes(fixed["published_inputs_sha256"])
    assert_hashes(fixed["frozen_sources_sha256"])
    route, weight = verify_selection(json_file(SELECTION))
    verify_selected_inputs(args, route)
    if (route != fixed["selected_route"] or weight != fixed["selected_weight"]
            or (route != "current" and
                (len(fixed["replacement_feature_names"]) != 184
                 or len(fixed["replacement_feature_schema"]) != 184
                 or len(fixed["v7_feature_schema"]) != 174))):
        raise ValueError("Selected route or feature architecture changed")
    return fixed


def fit_one(args: argparse.Namespace, name: str) -> None:
    fixed = check_frozen(args)
    route = fixed["selected_route"]
    if route == "current":
        raise ValueError("Current route has no reserved model fits")
    if name == "replacement" and not (args.output_dir / "comparator_receipt.json").exists():
        raise FileNotFoundError("The comparator must be fitted and sealed first")
    if name == "replacement":
        verify_fit(args, "comparator", fixed)
        paired_comparator = fit_artifact_hashes(args, ("comparator",))
    else:
        paired_comparator = {}
    require_memory(args)
    for suffix in ("_oof.parquet", ".cbm", "_fit.json", "_receipt.json"):
        if (args.output_dir / f"{name}{suffix}").exists():
            raise FileExistsError(f"{name} already has an output; never silently refit")
    before = dict(fixed["frozen_sources_sha256"])
    rows, x = (traffic.load_features(args) if name == "comparator"
               else ROUTE_MODULES[route].load_features(args))
    names = fixed["v7_feature_names"] if name == "comparator" else fixed["replacement_feature_names"]
    if list(x) != names:
        raise ValueError("Loaded predictor order differs from frozen architecture")
    expected_schema = (fixed["v7_feature_schema"] if name == "comparator"
                       else fixed["replacement_feature_schema"])
    train, test, y, proxy = fixed_split(rows, x, expected_schema)
    feature_schema = schema(x)
    categories_frozen = category_hashes(x)
    check_frozen(args)
    if before != source_hashes(args, route):
        raise ValueError("Frozen inputs changed during feature loading")
    rng = np.random.default_rng(2026)
    order = rng.permutation(train)
    n_early = max(20000, int(.06 * len(order)))
    early, fit_idx = order[:n_early], order[n_early:]
    if (len(fit_idx) == 0 or len(early) == 0
            or rows.month.iloc[np.r_[fit_idx, early]].isin(HELDOUT).any()):
        raise ValueError("Reserved labels entered fit or early stopping")
    categories = x.select_dtypes(include="category").columns.tolist()
    cat_indices = [x.columns.get_loc(column) for column in categories]
    fit_ids_hash = id_hash(rows.MVT_ID_mvt.iloc[fit_idx])
    early_ids_hash = id_hash(rows.MVT_ID_mvt.iloc[early])
    heldout_ids_hash = id_hash(rows.MVT_ID_mvt.iloc[test])
    if name == "replacement":
        prior = json_file(args.output_dir / "comparator_receipt.json")
        if (prior.get("fit_ids_sha256") != fit_ids_hash
                or prior.get("early_stop_ids_sha256") != early_ids_hash
                or prior.get("heldout_ids_sha256") != heldout_ids_hash
                or prior.get("categorical_vocabularies") != categories_frozen):
            raise ValueError("Replacement differs from paired comparator rows/categories")
    residual = y - proxy
    model = CatBoostRegressor(**fixed_params(args))
    fit_pool = Pool(x.iloc[fit_idx], label=residual[fit_idx], cat_features=categories)
    early_pool = Pool(x.iloc[early], label=residual[early], cat_features=categories)
    start = time.monotonic()
    model.fit(fit_pool, eval_set=early_pool, early_stopping_rounds=200,
              use_best_model=True)
    elapsed = time.monotonic() - start
    del fit_pool, early_pool
    gc.collect()
    prediction = proxy[test] + model.predict(x.iloc[test], thread_count=args.threads)
    if not np.isfinite(prediction).all():
        raise ValueError("Held-out raw expert predictions contain nonfinite values")
    held = rows.iloc[test]
    oof = pd.DataFrame({
        "MVT_ID_mvt": held.MVT_ID_mvt.to_numpy(copy=True),
        "target": y[test],
        "MVT_TIME_UTC_mvt": pd.to_datetime(held.time, utc=True).reset_index(drop=True),
        "expert": prediction,
    })
    if (list(oof) != list(RAW_COLUMNS) or oof.MVT_ID_mvt.duplicated().any()
            or oof.MVT_ID_mvt.isna().any()
            or not np.isfinite(oof[["target", "expert"]].to_numpy(dtype=float)).all()
            or set(oof.MVT_TIME_UTC_mvt.dt.month.unique()) != set(HELDOUT)):
        raise ValueError("Held-out OOF coverage, labels, times, or predictions are invalid")
    check_frozen(args)
    if before != source_hashes(args, route):
        raise ValueError("Frozen inputs changed during model fit")
    if name == "replacement" and paired_comparator != fit_artifact_hashes(
            args, ("comparator",)):
        raise ValueError("Comparator artifacts changed during replacement fit")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / f"{name}.cbm"
    oof_path = args.output_dir / f"{name}_oof.parquet"
    report_path = args.output_dir / f"{name}_fit.json"
    receipt_path = args.output_dir / f"{name}_receipt.json"
    publish_temp_exclusive(model_path, lambda path: model.save_model(str(path)))
    publish_temp_exclusive(oof_path, lambda path: oof.to_parquet(path, index=False))
    report = {
        "name": name, "route": route, "heldout_months": list(HELDOUT),
        "fit_rows": len(fit_idx), "early_stop_rows": len(early),
        "heldout_rows": len(test),
        "heldout_ids_sha256": heldout_ids_hash,
        "fit_ids_sha256": fit_ids_hash,
        "early_stop_ids_sha256": early_ids_hash,
        "heldout_month_counts": {str(month): int((pd.to_datetime(held.time, utc=True)
                                                   .dt.month == month).sum())
                                 for month in HELDOUT},
        "best_iteration": int(model.get_best_iteration()),
        "trees": int(model.tree_count_), "fit_seconds": float(elapsed),
        "feature_schema": feature_schema,
        "categorical_feature_indices": cat_indices,
        "categorical_vocabularies": categories_frozen,
        "catboost_params": fixed_params(args),
        "source_protocol_sha256": sha256(args.output_dir / "protocol.json"),
        "labels_excluded_from_fit_and_early_stop": True,
    }
    write_new_json(report_path, report)
    receipt = {
        "schema_version": 1, "name": name, "route": route,
        "heldout_months": list(HELDOUT),
        "protocol_sha256": sha256(args.output_dir / "protocol.json"),
        "portfolio_selection_sha256": sha256(SELECTION),
        "feature_schema_sha256": schema_hash(feature_schema),
        "feature_schema": feature_schema,
        "categorical_feature_indices": cat_indices,
        "categorical_vocabularies": categories_frozen,
        "catboost_params": fixed_params(args),
        "fit_rows": len(fit_idx), "early_stop_rows": len(early),
        "heldout_rows": len(test),
        "heldout_ids_sha256": heldout_ids_hash,
        "fit_ids_sha256": fit_ids_hash,
        "early_stop_ids_sha256": early_ids_hash,
        "heldout_month_counts": report["heldout_month_counts"],
        "trees": int(model.tree_count_),
        "model_sha256": sha256(model_path),
        "oof_sha256": sha256(oof_path),
        "fit_report_sha256": sha256(report_path),
        "source_sha256": before,
        "paired_comparator_sha256": paired_comparator,
    }
    check_frozen(args)
    if before != source_hashes(args, route):
        raise ValueError("Frozen inputs changed during artifact serialization")
    if name == "replacement" and paired_comparator != fit_artifact_hashes(
            args, ("comparator",)):
        raise ValueError("Comparator artifacts changed during replacement serialization")
    write_new_json(receipt_path, receipt)
    check_frozen(args)
    if before != source_hashes(args, route):
        raise ValueError("Frozen inputs changed before fit receipt verification")
    del rows, x, model, prediction, oof, held, residual, y, proxy
    gc.collect()
    verify_fit(args, name, fixed)
    print(json.dumps({"name": name, "trees": receipt["trees"],
                      "heldout_rows": len(test), "fit_seconds": elapsed}), flush=True)


def verify_fit(args: argparse.Namespace, name: str, fixed: dict) -> pd.DataFrame:
    """Reopen the sealed OOF/model and verify predictor/model/fit provenance."""
    receipt_path = args.output_dir / f"{name}_receipt.json"
    receipt = json_file(receipt_path)
    report_path = args.output_dir / f"{name}_fit.json"
    model_path = args.output_dir / f"{name}.cbm"
    oof_path = args.output_dir / f"{name}_oof.parquet"
    report = json_file(report_path)
    route = fixed["selected_route"]
    names = fixed["v7_feature_names"] if name == "comparator" else fixed["replacement_feature_names"]
    if (receipt.get("schema_version") != 1 or receipt.get("name") != name
            or receipt.get("route") != route or receipt.get("heldout_months") != list(HELDOUT)
            or receipt.get("protocol_sha256") != sha256(args.output_dir / "protocol.json")
            or receipt.get("portfolio_selection_sha256") != sha256(SELECTION)
            or receipt.get("catboost_params") != fixed_params(args)
            or receipt.get("source_sha256") != fixed["frozen_sources_sha256"]
            or receipt.get("paired_comparator_sha256") !=
               (fit_artifact_hashes(args, ("comparator",))
                if name == "replacement" else {})
            or receipt.get("model_sha256") != sha256(model_path)
            or receipt.get("oof_sha256") != sha256(oof_path)
            or receipt.get("fit_report_sha256") != sha256(report_path)
            or report.get("name") != name or report.get("route") != route
            or report.get("heldout_months") != list(HELDOUT)
            or report.get("source_protocol_sha256") != receipt["protocol_sha256"]
            or report.get("labels_excluded_from_fit_and_early_stop") is not True
            or report.get("catboost_params") != fixed_params(args)
            or report.get("feature_schema") != receipt.get("feature_schema")
            or report.get("categorical_vocabularies") !=
               receipt.get("categorical_vocabularies")
            or report.get("trees") != receipt.get("trees")
            or [item["name"] for item in receipt["feature_schema"]] != names
            or receipt.get("feature_schema_sha256") != schema_hash(receipt["feature_schema"])
            or receipt.get("feature_schema") !=
               (fixed["v7_feature_schema"] if name == "comparator"
                else fixed["replacement_feature_schema"])
            or receipt.get("heldout_month_counts") != report.get("heldout_month_counts")
            or receipt.get("fit_rows") != report.get("fit_rows")
            or receipt.get("early_stop_rows") != report.get("early_stop_rows")
            or receipt.get("heldout_rows") != report.get("heldout_rows")
            or receipt.get("heldout_ids_sha256") !=
               report.get("heldout_ids_sha256")
            or receipt.get("fit_ids_sha256") != report.get("fit_ids_sha256")
            or receipt.get("early_stop_ids_sha256") !=
               report.get("early_stop_ids_sha256")
            or receipt.get("categorical_feature_indices") !=
               report.get("categorical_feature_indices")
            or not 1 <= receipt.get("trees", 0) <= 10000):
        raise ValueError(f"{name} receipt, fixed schema or model/report hashes differ")
    model = CatBoostRegressor()
    model.load_model(str(model_path))
    (v10.verify_saved_params if route == "v10" else v11.verify_saved_params)(model.get_all_params())
    categorical_indices = [i for i, item in enumerate(receipt["feature_schema"])
                           if item["dtype"] == "category"]
    if (receipt.get("categorical_feature_indices") != categorical_indices
            or set(receipt.get("categorical_vocabularies", {})) !=
               {names[i] for i in categorical_indices}):
        raise ValueError(f"{name} categorical schema receipt differs")
    if (model.tree_count_ != receipt["trees"]
            or list(model.feature_names_) != names
            or list(model.get_cat_feature_indices()) != categorical_indices
            or receipt["feature_schema"] != report["feature_schema"]):
        raise ValueError(f"{name} saved CatBoost feature/category/tree metadata differs")
    frame = pd.read_parquet(oof_path)
    if (list(frame) != list(RAW_COLUMNS) or len(frame) != receipt["heldout_rows"]
            or frame.MVT_ID_mvt.isna().any() or frame.MVT_ID_mvt.duplicated().any()
            or not np.isfinite(frame[["target", "expert"]].to_numpy(dtype=float)).all()
            or pd.to_datetime(frame.MVT_TIME_UTC_mvt, utc=True).isna().any()
            or set(pd.to_datetime(frame.MVT_TIME_UTC_mvt, utc=True).dt.month.unique())
               != set(HELDOUT)):
        raise ValueError(f"{name} saved raw held-out output differs")
    if id_hash(frame.MVT_ID_mvt) != receipt["heldout_ids_sha256"]:
        raise ValueError(f"{name} saved held-out ID receipt differs")
    counts = {str(m): int((pd.to_datetime(frame.MVT_TIME_UTC_mvt, utc=True)
                           .dt.month == m).sum()) for m in HELDOUT}
    if counts != receipt["heldout_month_counts"]:
        raise ValueError(f"{name} month counts differ")
    return frame


def paired_day_bootstrap(target: np.ndarray, old: np.ndarray,
                         new: np.ndarray, times: pd.Series) -> dict:
    dates = pd.to_datetime(times, utc=True).dt.floor("D")
    groups, days = pd.factorize(dates, sort=True)
    if len(days) < 2 or (groups < 0).any():
        raise ValueError("Reserved paired UTC-day bootstrap has missing or too few days")
    n = np.bincount(groups).astype(float)
    left = np.bincount(groups, weights=(target - old) ** 2)
    right = np.bincount(groups, weights=(target - new) ** 2)
    draws = np.random.default_rng(BOOTSTRAP_SEED).integers(
        0, len(days), size=(BOOTSTRAP_REPEATS, len(days)))
    sampled_n = n[draws].sum(axis=1)
    gains = np.sqrt(left[draws].sum(axis=1) / sampled_n) - np.sqrt(
        right[draws].sum(axis=1) / sampled_n)
    return {"days": len(days), "repeats": BOOTSTRAP_REPEATS,
            "seed": BOOTSTRAP_SEED,
            "gain_ci95_sec": np.quantile(gains, [.025, .975]).tolist()}


def rmse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - p) ** 2)))


def fixed_gate(frame: pd.DataFrame, weight: float) -> tuple[dict, pd.DataFrame]:
    if (list(frame) != ["MVT_ID_mvt", "target", "MVT_TIME_UTC_mvt",
                        "v7_raw", "replacement_raw"]
            or frame.MVT_ID_mvt.isna().any()
            or frame.MVT_ID_mvt.duplicated().any()
            or not np.isfinite(frame[["target", "v7_raw", "replacement_raw"]]
                               .to_numpy(dtype=float)).all()):
        raise ValueError("Paired held-out raw expert frame is invalid")
    time_value = pd.to_datetime(frame.MVT_TIME_UTC_mvt, utc=True, errors="coerce")
    if time_value.isna().any() or set(time_value.dt.month.unique()) != set(HELDOUT):
        raise ValueError("Paired held-out timestamps disagree with reserved months")
    y = frame.target.to_numpy(dtype=float)
    v7_raw = frame.v7_raw.to_numpy(dtype=float)
    new_raw = frame.replacement_raw.to_numpy(dtype=float)
    old = np.maximum(v7_raw, 0)
    candidate = np.maximum(v7_raw + weight * (new_raw - v7_raw), 0)
    month = time_value.dt.month.to_numpy()
    scores = {}
    for m in HELDOUT:
        take = month == m
        scores[str(m)] = {"n": int(take.sum()),
                          "v7_rmse": rmse(y[take], old[take]),
                          "fixed_blend_rmse": rmse(y[take], candidate[take])}
    boot = paired_day_bootstrap(y, old, candidate, time_value)
    passed = (all(item["fixed_blend_rmse"] < item["v7_rmse"]
                  for item in scores.values())
              and boot["gain_ci95_sec"][0] > 0)
    result = frame.copy()
    result["v7_clipped"] = old
    result["fixed_blend_clipped"] = candidate
    return {"scores": scores, "bootstrap": boot,
            "pooled_v7_rmse": rmse(y, old),
            "pooled_fixed_blend_rmse": rmse(y, candidate),
            "passed": bool(passed)}, result


def evaluate(args: argparse.Namespace) -> dict:
    fixed = check_frozen(args)
    route = fixed["selected_route"]
    if route == "current":
        terminal = json_file(args.output_dir / "terminal.json")
        if (terminal.get("route") != "current"
                or terminal.get("weight") != 0.0
                or terminal.get("heldout_months") != list(HELDOUT)
                or terminal.get("passed") is not True
                or terminal.get("retained_policy") != "current"
                or terminal.get("replacement_fit") is not False
                or terminal.get("protocol_sha256") !=
                   sha256(args.output_dir / "protocol.json")
                or terminal.get("selection_sha256") != sha256(SELECTION)):
            raise ValueError("Current route terminal seal differs")
        return terminal
    paired_artifacts = fit_artifact_hashes(args, ("comparator", "replacement"))
    left = verify_fit(args, "comparator", fixed)
    right = verify_fit(args, "replacement", fixed)
    exact_ids(right.MVT_ID_mvt, left.MVT_ID_mvt, "Reserved paired OOF")
    right = right.set_index("MVT_ID_mvt").loc[left.MVT_ID_mvt.to_numpy()].reset_index()
    if (not np.array_equal(left.target.to_numpy(dtype=float),
                           right.target.to_numpy(dtype=float))
            or not np.array_equal(pd.to_datetime(left.MVT_TIME_UTC_mvt, utc=True).to_numpy(),
                                  pd.to_datetime(right.MVT_TIME_UTC_mvt, utc=True).to_numpy())):
        raise ValueError("Reserved paired held-out labels or timestamps differ")
    before = dict(fixed["frozen_sources_sha256"])
    baseline = pd.read_parquet(args.cache_dir / "training_rows.parquet",
                               columns=["MVT_ID_mvt", "target", "proxy", "month", "time"])
    if (baseline.MVT_ID_mvt.isna().any() or baseline.MVT_ID_mvt.duplicated().any()
            or pd.to_datetime(baseline.time, utc=True, errors="coerce").isna().any()):
        raise ValueError("Baseline held-out ID/time provenance differs")
    y = baseline.target.to_numpy(dtype=float)
    proxy = baseline.proxy.to_numpy(dtype=float)
    take = (baseline.month.isin(HELDOUT).to_numpy() & np.isfinite(y)
            & np.isfinite(proxy) & (proxy >= 0) & (proxy <= 7200))
    expected = baseline.loc[take]
    exact_ids(left.MVT_ID_mvt, expected.MVT_ID_mvt, "All finite valid-AOBT May/September")
    aligned = expected.set_index("MVT_ID_mvt").loc[left.MVT_ID_mvt.to_numpy()]
    if (not np.array_equal(left.target.to_numpy(dtype=float),
                           aligned.target.to_numpy(dtype=float))
            or not np.array_equal(pd.to_datetime(left.MVT_TIME_UTC_mvt, utc=True).to_numpy(),
                                  pd.to_datetime(aligned.time, utc=True).to_numpy())):
        raise ValueError("Reserved predictions differ from exact baseline labels/times")
    paired = left.rename(columns={"expert": "v7_raw"})
    paired["replacement_raw"] = right.expert.to_numpy(dtype=float)
    report, output = fixed_gate(paired, fixed["selected_weight"])
    check_frozen(args)
    if before != source_hashes(args, route):
        raise ValueError("Sources changed during May/September evaluation")
    if paired_artifacts != fit_artifact_hashes(args, ("comparator", "replacement")):
        raise ValueError("Model, OOF or fit receipts changed during evaluation")
    prediction_path = args.output_dir / "paired_predictions.parquet"
    terminal_path = args.output_dir / "terminal.json"
    if prediction_path.exists() or terminal_path.exists():
        raise FileExistsError("Reserved guard already evaluated; never overwrite it")
    publish_temp_exclusive(prediction_path,
                           lambda path: output.to_parquet(path, index=False))
    if paired_artifacts != fit_artifact_hashes(args, ("comparator", "replacement")):
        raise ValueError("Model, OOF or fit receipts changed during output serialization")
    terminal = {
        "schema_version": 1, "route": route, "weight": fixed["selected_weight"],
        "passed": report["passed"],
        "retained_policy": route if report["passed"] else "current",
        "heldout_months": list(HELDOUT),
        "scores": report["scores"], "bootstrap": report["bootstrap"],
        "pooled_v7_rmse": report["pooled_v7_rmse"],
        "pooled_fixed_blend_rmse": report["pooled_fixed_blend_rmse"],
        "protocol_sha256": sha256(args.output_dir / "protocol.json"),
        "selection_sha256": sha256(SELECTION),
        "comparator_receipt_sha256": sha256(args.output_dir / "comparator_receipt.json"),
        "replacement_receipt_sha256": sha256(args.output_dir / "replacement_receipt.json"),
        "comparator_model_sha256": sha256(args.output_dir / "comparator.cbm"),
        "replacement_model_sha256": sha256(args.output_dir / "replacement.cbm"),
        "paired_predictions_sha256": sha256(prediction_path),
        "architecture_comparison_only": True,
    }
    check_frozen(args)
    if paired_artifacts != fit_artifact_hashes(args, ("comparator", "replacement")):
        raise ValueError("Model, OOF or fit receipts changed before terminal seal")
    write_new_json(terminal_path, terminal)
    print(json.dumps({"route": route, "passed": report["passed"],
                      "retained_policy": terminal["retained_policy"]}), flush=True)
    return terminal


def synthetic() -> dict:
    """Small metadata/gate proof with no competition data access."""
    if selection_fields({"selected_route": "current", "selected_weight": 0}) != ("current", 0.0):
        raise AssertionError("Current route schema failed")
    for item in ({"selected_route": "v10", "selected_weight": 0.0},
                 {"selected_route": "current", "selected_weight": .25},
                 {"selected_route": "v11", "selected_weight": float("nan")},
                 {"selected_route": "other", "selected_weight": .25}):
        try:
            selection_fields(item)
        except ValueError:
            pass
        else:
            raise AssertionError("Tampered selector route/weight was accepted")
    times = pd.to_datetime(["2025-05-01T00:00:00Z", "2025-05-02T00:00:00Z",
                            "2025-09-01T00:00:00Z", "2025-09-02T00:00:00Z"])
    sample = pd.DataFrame({"MVT_ID_mvt": [1, 2, 3, 4], "target": [10., 10., 10., 10.],
                           "MVT_TIME_UTC_mvt": times,
                           "v7_raw": [20., 20., 20., 20.],
                           "replacement_raw": [10., 10., 10., 10.]})
    result, _ = fixed_gate(sample, 1.0)
    if not result["passed"] or result["bootstrap"]["gain_ci95_sec"][0] <= 0:
        raise AssertionError("Known paired improvement failed")
    result, _ = fixed_gate(sample, .25)
    if not result["passed"]:
        raise AssertionError("Frozen partial blend failed")
    worse = sample.copy()
    worse.loc[2:, "replacement_raw"] = 40.
    result, _ = fixed_gate(worse, 1.0)
    if result["passed"]:
        raise AssertionError("Nonimproving September was accepted")
    if schema_hash(schema(pd.DataFrame({"a": pd.Series(["x"], dtype="category")}))) == \
            schema_hash(schema(pd.DataFrame({"a": pd.Series([1.], dtype="float64")}))):
        raise AssertionError("Category schema tampering was not detected")
    if (id_hash(pd.Series([1., 2., 3.])) == id_hash(pd.Series([2., 1., 3.]))
            or category_hashes(pd.DataFrame({"a": pd.Categorical(["x", "y"])})) ==
               category_hashes(pd.DataFrame({"a": pd.Categorical(
                   ["x", "y"], categories=["y", "x"])}))):
        raise AssertionError("ID order or category-level order was not bound")
    try:
        fixed_gate(sample.iloc[:2].copy(), 1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("Incomplete May/September coverage was accepted")
    with tempfile.TemporaryDirectory(prefix="later-guard-synthetic-") as folder:
        target = Path(folder) / "receipt.json"
        write_new_json(target, {"synthetic": True})
        digest = sha256(target)
        try:
            write_new_json(target, {"synthetic": False})
        except FileExistsError:
            pass
        else:
            raise AssertionError("Existing provenance receipt was overwritten")
        if sha256(target) != digest:
            raise AssertionError("Synthetic immutable receipt changed")
    return {"synthetic": "passed", "heldout_months": list(HELDOUT),
            "bootstrap_repeats": BOOTSTRAP_REPEATS,
            "ranking_or_competition_data_read": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True,
                        choices=("freeze", "fit-comparator", "fit-replacement",
                                 "evaluate", "synthetic"))
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "artifacts/baseline")
    parser.add_argument("--weather-file", type=Path,
                        default=ROOT / "data/external/weather.parquet")
    parser.add_argument("--arrival-dir", type=Path,
                        default=ROOT / "artifacts/v5-arrival-clean")
    parser.add_argument("--neighbour-dir", type=Path,
                        default=ROOT / "artifacts/v6-neighbour")
    parser.add_argument("--runway-dir", type=Path,
                        default=ROOT / "artifacts/v6-runway-arrival")
    parser.add_argument("--taxi-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--min-free-gib", type=float, default=MIN_FREE_GIB)
    args = parser.parse_args()
    args.depth, args.iterations, args.threads = 10, 10000, 2
    if args.mode == "synthetic":
        print(json.dumps(synthetic(), indent=2))
        return
    selected = json_file(SELECTION)
    route, _ = selection_fields(selected)
    if args.taxi_dir is None:
        args.taxi_dir = ROOT / ("artifacts/v10-runway-arrival-taxi" if route == "v10"
                                else "artifacts/v11-taxi-flow")
    if args.mode == "freeze":
        value = freeze(args)
        print(json.dumps({"protocol": str(args.output_dir / "protocol.json"),
                          "selected_route": value["selected_route"]}, indent=2))
    elif args.mode == "fit-comparator":
        fit_one(args, "comparator")
    elif args.mode == "fit-replacement":
        fit_one(args, "replacement")
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
