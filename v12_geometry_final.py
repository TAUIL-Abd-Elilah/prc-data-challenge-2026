"""Frozen February/August component guard and conditional v12 final prediction.

This extends the original v12 comparison without changing its folds or weight.
The guard compares matched v11 and v12 refits; only a passed guard permits a
full 2025 v12 fit and a local ranking artifact. Nothing here uploads a file.
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
import later_feature_final as v9_final
import later_reserved_guard as old_guard
import runway_geometry_features as geometry
import v11_taxi_interval_flow_expert as v11
import v12_runway_geometry_expert as v12


ROOT = Path(__file__).resolve().parent
V12_DIR = ROOT / "artifacts/v12-runway-geometry-expert"
GUARD_DIR = ROOT / "artifacts/v12-geometry-feb-aug-guard"
FINAL_DIR = ROOT / "artifacts/v12-geometry-final"
V9_SUBMISSION = ROOT / "submissions/merry-mushroom_v9.parquet"
V9_SHA256 = "bc465ae7ff48deac5f93cd449a3799fee1a361a8021ec2ca03ff70baccc0f417"
TEMPLATE = ROOT / "data/submitting.parquet"
HELDOUT = (2, 8)
SEED = 20261015
REPEATS = 1000
EXPECTED_TRAIN = 2_085_047
EXPECTED_RANK = 344_841
EXPECTED_VALID = 339_377
FOLDS = ("seasonal_jan_jul", "forward_nov_dec")
FORBIDDEN = {"MVT_ID_mvt", "target", "BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt"}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    digest = sha(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if sha(path) != digest or not isinstance(value, dict):
        raise ValueError(f"JSON changed while reading: {path}")
    return value


def write_json_new(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as output:
        json.dump(value, output, indent=2, sort_keys=True, allow_nan=False)
        output.write("\n")


def publish_new(path: Path, writer) -> str:
    """Publish a completed file with exclusive same-volume hard-link semantics."""
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".v12_geometry_", dir=path.parent) as folder:
        stage = Path(folder) / path.name
        writer(stage)
        digest = sha(stage)
        os.link(stage, path)
    if sha(path) != digest:
        raise ValueError(f"Published bytes changed: {path}")
    return digest


def hashes(paths: dict[str, Path]) -> dict[str, str]:
    if len(paths) != len(set(paths)):
        raise ValueError("Input inventory contains duplicate keys")
    return {name: sha(path) for name, path in sorted(paths.items())}


def check_hashes(paths: dict[str, Path], expected: dict[str, str], label: str) -> None:
    if hashes(paths) != expected:
        raise ValueError(f"{label} source or input bytes changed")


def args_for_v12() -> argparse.Namespace:
    """Exactly the published v12 CLI defaults, never caller-selected paths."""
    return argparse.Namespace(
        data_dir=ROOT / "data", cache_dir=ROOT / "artifacts/baseline",
        weather_file=ROOT / "data/external/weather.parquet",
        arrival_dir=ROOT / "artifacts/v5-arrival-clean",
        neighbour_dir=ROOT / "artifacts/v6-neighbour",
        runway_dir=ROOT / "artifacts/v6-runway-arrival",
        taxi_dir=ROOT / "artifacts/v11-taxi-flow",
        geometry_source_dir=ROOT / "artifacts/prospective-runway-geometry",
        geometry_dir=ROOT / "artifacts/prospective-runway-geometry/features",
        v11_dir=ROOT / "artifacts/v11-taxi-flow-expert",
        portfolio_dir=ROOT / "artifacts/later-feature-portfolio",
        guard_dir=ROOT / "artifacts/later-reserved-guard",
        v9_final_dir=ROOT / "artifacts/later-feature-final",
        reports_dir=ROOT / "reports", submission_v9=V9_SUBMISSION,
        v7_dir=ROOT / "artifacts/v7-runway-traffic",
        v6_dir=ROOT / "artifacts/v6-deep-arrival", output_dir=V12_DIR,
        min_free_gib=10.0, iterations=10000, depth=10, threads=2,
    )


def require_published_source(expected: str) -> None:
    if len(expected) != 64 or expected != sha(Path(__file__).resolve()):
        raise ValueError("Pass the exact published SHA256 of this source")


def require_memory() -> None:
    geometry.require_memory(10.0)


def schema(frame: pd.DataFrame) -> list[dict[str, str]]:
    if frame.columns.duplicated().any() or FORBIDDEN.intersection(frame.columns):
        raise ValueError("Predictor names contain duplicates or forbidden fields")
    return [{"name": name, "dtype": str(frame[name].dtype)} for name in frame]


def original_gates() -> tuple[float, list[dict], dict]:
    """Replay the original v12 folds and April/October before any reserved labels."""
    args = args_for_v12()
    original = v12.evaluate(args)
    fresh = v12.verify_fresh(args)
    weight = original.get("selected_weight")
    if (original.get("existing_folds_passed") is not True
            or fresh.get("passed") is not True
            or fresh.get("weight") != weight
            or type(weight) not in (int, float)
            or float(weight) not in v12.WEIGHTS or weight <= 0):
        raise ValueError("Original v12 or April/October fixed-weight gate failed")
    receipts = [read_json(V12_DIR / f"{name}_provenance.json") for name in FOLDS]
    feature_schema = receipts[0].get("feature_schema")
    if not isinstance(feature_schema, list):
        raise ValueError("Original v12 model has no feature schema")
    cats = [i for i, item in enumerate(feature_schema) if item["dtype"] == "category"]
    if (len(feature_schema) != 200
            or feature_schema != receipts[1].get("feature_schema")
            or len(cats) != 24 or len({item["name"] for item in feature_schema}) != 200
            or feature_schema[-16:] != [{"name": name, "dtype": "float32"}
                                       for name in geometry.FEATURES]
            or any(receipt.get("categorical_feature_indices") != cats
                   or receipt.get("catboost_params") != v12.EXPECTED_CATBOOST_PARAMS
                   for receipt in receipts)):
        raise ValueError("Original v12 model schemas or parameters disagree")
    return float(weight), feature_schema, original


def source_paths() -> dict[str, Path]:
    """All transitive original inputs plus the exact terminal OOF/model proofs."""
    args = args_for_v12()
    raw, inherited = v12.source_inventory(args)
    paths = {f"v12_source_{name}": Path(path) for name, path in inherited.items()}
    paths.update({f"raw_{path.name}": path for path in raw})
    paths.update({
        "own_source": Path(__file__).resolve(),
        "old_guard_helper": Path(old_guard.__file__).resolve(),
        "v11_source": Path(v11.__file__).resolve(),
        "v12_source": Path(v12.__file__).resolve(),
        "v9_final_helper": Path(v9_final.__file__).resolve(),
        "v12_protocol": V12_DIR / "protocol.json",
        "v12_original_validation": V12_DIR / "validation.json",
        "v12_original_predictions": V12_DIR / "validation_predictions.parquet",
        "v12_fresh_audit": V12_DIR / "fresh_audit.json",
        "v12_fresh_predictions": V12_DIR / "fresh_audit_predictions.parquet",
        "v12_fresh_model": V12_DIR / "fresh_new/fresh_apr_oct.cbm",
        "v12_fresh_receipt": V12_DIR / "fresh_new/fresh_apr_oct_provenance.json",
        "v12_fresh_oof": V12_DIR / "fresh_new/fresh_apr_oct_oof.parquet",
        "v12_fresh_fit": V12_DIR / "fresh_new/fresh_apr_oct_validation.json",
        "v11_original_protocol": args.v11_dir / "protocol.json",
    })
    for name in FOLDS:
        for suffix in (".cbm", "_oof.parquet", "_validation.json", "_provenance.json"):
            paths[f"v12_{name}{suffix}"] = V12_DIR / f"{name}{suffix}"
            paths[f"v11_{name}{suffix}"] = args.v11_dir / f"{name}{suffix}"
    # The v12 inventory already names several helpers that this extension
    # also names for clarity. Hash each byte stream once while keeping the
    # complete transitive closure.
    unique: dict[str, Path] = {}
    seen: set[Path] = set()
    for name, path in paths.items():
        resolved = path.resolve()
        if resolved not in seen:
            unique[name] = path
            seen.add(resolved)
    return unique


def guard_protocol_path() -> Path:
    return GUARD_DIR / "protocol.json"


def freeze_guard(expected_source_sha256: str) -> dict:
    require_published_source(expected_source_sha256)
    if guard_protocol_path().exists():
        return check_guard_frozen(expected_source_sha256)
    weight, feature_schema, _ = original_gates()
    args = args_for_v12()
    frozen, protocol_sha = v12.require_prepared(args)
    v12.verify_source_snapshot(args, frozen, protocol_sha)
    if sha(V9_SUBMISSION) != V9_SHA256:
        raise ValueError("Frozen local v9 ranking bytes changed")
    original = read_json(V12_DIR / "validation.json")
    fresh = read_json(V12_DIR / "fresh_audit.json")
    inputs = hashes(source_paths())
    value = {
        "schema_version": 1,
        "heldout_months": list(HELDOUT), "selected_weight": weight,
        "source_sha256": expected_source_sha256,
        "source_input_sha256": inputs,
        "v12_protocol_sha256": protocol_sha,
        "v12_original_validation_sha256": sha(V12_DIR / "validation.json"),
        "v12_fresh_audit_sha256": sha(V12_DIR / "fresh_audit.json"),
        "v12_original_passed": original["existing_folds_passed"],
        "v12_fresh_passed": fresh["passed"],
        "v11_schema": feature_schema[:184],
        "v12_schema": feature_schema,
        "catboost_params": v12.EXPECTED_CATBOOST_PARAMS,
        "training_exclusion": "February and August excluded from fit and internal early stopping",
        "reserved_labels_read_to_freeze": False,
    }
    check_hashes(source_paths(), inputs, "Guard freeze")
    write_json_new(guard_protocol_path(), value)
    return check_guard_frozen(expected_source_sha256)


def check_guard_frozen(expected_source_sha256: str) -> dict:
    require_published_source(expected_source_sha256)
    frozen = read_json(guard_protocol_path())
    weight, feature_schema, _ = original_gates()
    args = args_for_v12()
    v12_frozen, protocol_sha = v12.require_prepared(args)
    v12.verify_source_snapshot(args, v12_frozen, protocol_sha)
    if (frozen.get("schema_version") != 1
            or frozen.get("heldout_months") != list(HELDOUT)
            or frozen.get("selected_weight") != weight
            or frozen.get("source_sha256") != expected_source_sha256
            or frozen.get("v12_protocol_sha256") != protocol_sha
            or frozen.get("v12_original_validation_sha256") != sha(V12_DIR / "validation.json")
            or frozen.get("v12_fresh_audit_sha256") != sha(V12_DIR / "fresh_audit.json")
            or frozen.get("v12_original_passed") is not True
            or frozen.get("v12_fresh_passed") is not True
            or frozen.get("v11_schema") != feature_schema[:184]
            or frozen.get("v12_schema") != feature_schema
            or frozen.get("catboost_params") != v12.EXPECTED_CATBOOST_PARAMS
            or frozen.get("reserved_labels_read_to_freeze") is not False
            or sha(V9_SUBMISSION) != V9_SHA256):
        raise ValueError("Prospective guard protocol, selected weight or source changed")
    check_hashes(source_paths(), frozen["source_input_sha256"], "Guard")
    return frozen


def exact_ids(actual: pd.Series, expected: pd.Series, label: str) -> None:
    v12.exact_ids(actual, expected, label)


def split(rows: pd.DataFrame, features: pd.DataFrame,
          expected_schema: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if (len(rows) != EXPECTED_TRAIN or len(features) != len(rows)
            or schema(features) != expected_schema
            or rows.MVT_ID_mvt.isna().any() or rows.MVT_ID_mvt.duplicated().any()):
        raise ValueError("Guard training universe or exact feature schema changed")
    times = pd.to_datetime(rows.time, utc=True, errors="coerce")
    if (times.isna().any() or not times.dt.year.eq(2025).all()
            or not np.array_equal(times.dt.month.to_numpy(), rows.month.to_numpy())):
        raise ValueError("Guard movement time/month provenance differs")
    y = rows.target.to_numpy(dtype=float)
    proxy = rows.proxy.to_numpy(dtype=float)
    valid = np.isfinite(proxy) & (proxy >= 0) & (proxy <= 7200)
    held = rows.month.isin(HELDOUT).to_numpy()
    train = np.flatnonzero(~held & valid & np.isfinite(y) & (y >= 0) & (y <= 86400))
    test = np.flatnonzero(held & valid & np.isfinite(y))
    if (len(train) < 1_500_000 or len(test) < 10_000
            or set(rows.month.iloc[test].unique()) != set(HELDOUT)
            or np.intersect1d(train, test).size):
        raise ValueError("February/August fit or all-finite heldout mask differs")
    return train, test, y, proxy


def baseline_split_receipt() -> tuple[dict, pd.DataFrame]:
    """Recreate fixed fit/early/held masks from sealed baseline row order."""
    base = pd.read_parquet(args_for_v12().cache_dir / "training_rows.parquet",
                           columns=["MVT_ID_mvt", "target", "proxy", "month", "time"])
    timestamps = pd.to_datetime(base.time, utc=True, errors="coerce")
    if (len(base) != EXPECTED_TRAIN or base.MVT_ID_mvt.isna().any()
            or base.MVT_ID_mvt.duplicated().any() or timestamps.isna().any()
            or not timestamps.dt.year.eq(2025).all()
            or not np.array_equal(timestamps.dt.month.to_numpy(), base.month.to_numpy())):
        raise ValueError("Baseline movement IDs, order, or 2025 month/time provenance changed")
    y = base.target.to_numpy(dtype=float)
    proxy = base.proxy.to_numpy(dtype=float)
    valid = np.isfinite(proxy) & (proxy >= 0) & (proxy <= 7200)
    held = base.month.isin(HELDOUT).to_numpy()
    training = np.flatnonzero(~held & valid & np.isfinite(y) & (y >= 0) & (y <= 86400))
    testing = np.flatnonzero(held & valid & np.isfinite(y))
    order = np.random.default_rng(2026).permutation(training)
    n_early = max(20_000, int(.06 * len(order)))
    early, fitting = order[:n_early], order[n_early:]
    if (len(fitting) < 1_500_000 or len(testing) < 10_000
            or set(base.month.iloc[testing].unique()) != set(HELDOUT)):
        raise ValueError("Fixed baseline February/August fit or held mask changed")
    receipt = {
        "fit_rows": len(fitting), "early_rows": len(early),
        "heldout_rows": len(testing),
        "fit_ids_sha256": old_guard.id_hash(base.MVT_ID_mvt.iloc[fitting]),
        "early_ids_sha256": old_guard.id_hash(base.MVT_ID_mvt.iloc[early]),
        "held_ids_sha256": old_guard.id_hash(base.MVT_ID_mvt.iloc[testing]),
        "held_month_counts": {
            str(month): int((base.month.iloc[testing] == month).sum())
            for month in HELDOUT},
    }
    return receipt, base.iloc[testing].copy()


def artifact_paths(name: str) -> dict[str, Path]:
    if name not in ("comparator", "replacement"):
        raise ValueError("Unknown fixed guard model")
    return {key: GUARD_DIR / f"{name}{suffix}" for key, suffix in (
        ("model", ".cbm"), ("oof", "_oof.parquet"),
        ("fit", "_fit.json"), ("receipt", "_receipt.json"))}


def guard_artifact_hashes(name: str) -> dict[str, str]:
    return hashes(artifact_paths(name))


def fit_guard(name: str, expected_source_sha256: str) -> dict:
    fixed = check_guard_frozen(expected_source_sha256)
    if name not in ("comparator", "replacement"):
        raise ValueError("Guard fits have only comparator and replacement modes")
    paths = artifact_paths(name)
    if any(path.exists() for path in paths.values()):
        raise FileExistsError(f"Partial or complete {name} outputs already exist")
    comparator_hashes = {}
    if name == "replacement":
        verify_guard_fit("comparator", fixed, expected_source_sha256)
        comparator_hashes = guard_artifact_hashes("comparator")
    require_memory()
    args = args_for_v12()
    before = dict(fixed["source_input_sha256"])
    rows, features = (v11.load_features(args) if name == "comparator"
                      else v12.load_features(args))
    expected_schema = fixed["v11_schema"] if name == "comparator" else fixed["v12_schema"]
    train, test, y, proxy = split(rows, features, expected_schema)
    categories = features.select_dtypes(include="category").columns.tolist()
    cat_idx = [features.columns.get_loc(column) for column in categories]
    vocab = old_guard.category_hashes(features)
    if len(cat_idx) != 24:
        raise ValueError("Fixed 24-category architecture changed")
    rng = np.random.default_rng(2026)
    order = rng.permutation(train)
    n_early = max(20_000, int(.06 * len(order)))
    early, fit = order[:n_early], order[n_early:]
    if (len(fit) == 0 or len(early) == 0
            or rows.month.iloc[np.r_[fit, early]].isin(HELDOUT).any()):
        raise ValueError("Reserved February/August labels entered training")
    fit_ids = old_guard.id_hash(rows.MVT_ID_mvt.iloc[fit])
    early_ids = old_guard.id_hash(rows.MVT_ID_mvt.iloc[early])
    held_ids = old_guard.id_hash(rows.MVT_ID_mvt.iloc[test])
    if name == "replacement":
        prior = read_json(artifact_paths("comparator")["receipt"])
        if (prior.get("fit_ids_sha256") != fit_ids
                or prior.get("early_ids_sha256") != early_ids
                or prior.get("held_ids_sha256") != held_ids
                or prior.get("category_vocabularies") != vocab):
            raise ValueError("Paired comparator and replacement rows/categories differ")
    check_guard_frozen(expected_source_sha256)
    if before != hashes(source_paths()):
        raise ValueError("Guard sources changed during feature assembly")
    model = CatBoostRegressor(**v12.EXPECTED_CATBOOST_PARAMS)
    fit_pool = Pool(features.iloc[fit], label=(y - proxy)[fit], cat_features=categories)
    early_pool = Pool(features.iloc[early], label=(y - proxy)[early], cat_features=categories)
    started = time.monotonic()
    model.fit(fit_pool, eval_set=early_pool, early_stopping_rounds=200,
              use_best_model=True)
    elapsed = time.monotonic() - started
    del fit_pool, early_pool
    gc.collect()
    raw = proxy[test] + model.predict(features.iloc[test], thread_count=2)
    if not np.isfinite(raw).all():
        raise ValueError("Reserved raw expert predictions contain nonfinite values")
    held = rows.iloc[test]
    oof = pd.DataFrame({
        "MVT_ID_mvt": held.MVT_ID_mvt.to_numpy(copy=True),
        "target": y[test],
        "MVT_TIME_UTC_mvt": pd.to_datetime(held.time, utc=True).reset_index(drop=True),
        "expert": raw,
    })
    if (list(oof) != ["MVT_ID_mvt", "target", "MVT_TIME_UTC_mvt", "expert"]
            or oof.MVT_ID_mvt.isna().any() or oof.MVT_ID_mvt.duplicated().any()
            or set(oof.MVT_TIME_UTC_mvt.dt.month.unique()) != set(HELDOUT)):
        raise ValueError("Reserved OOF IDs or month coverage changed")
    check_guard_frozen(expected_source_sha256)
    if before != hashes(source_paths()):
        raise ValueError("Guard sources changed while fitting")
    if name == "replacement" and comparator_hashes != guard_artifact_hashes("comparator"):
        raise ValueError("Comparator changed during replacement fit")
    model_sha = publish_new(paths["model"], lambda path: model.save_model(str(path)))
    oof_sha = publish_new(paths["oof"], lambda path: oof.to_parquet(path, index=False))
    report = {
        "name": name, "heldout_months": list(HELDOUT),
        "fit_rows": len(fit), "early_rows": len(early), "heldout_rows": len(test),
        "fit_ids_sha256": fit_ids, "early_ids_sha256": early_ids,
        "held_ids_sha256": held_ids,
        "held_month_counts": {str(month): int((oof.MVT_TIME_UTC_mvt.dt.month == month).sum())
                              for month in HELDOUT},
        "feature_schema": expected_schema, "categorical_indices": cat_idx,
        "category_vocabularies": vocab, "catboost_params": v12.EXPECTED_CATBOOST_PARAMS,
        "best_iteration": int(model.get_best_iteration()),
        "trees": int(model.tree_count_), "fit_seconds": float(elapsed),
        "excluded_reserved_from_fit_and_early": True,
        "protocol_sha256": sha(guard_protocol_path()),
    }
    write_json_new(paths["fit"], report)
    receipt = {
        "schema_version": 1, "name": name, "heldout_months": list(HELDOUT),
        "protocol_sha256": sha(guard_protocol_path()),
        "source_input_sha256": before,
        "feature_schema": expected_schema,
        "feature_schema_sha256": old_guard.schema_hash(expected_schema),
        "categorical_indices": cat_idx, "category_vocabularies": vocab,
        "catboost_params": v12.EXPECTED_CATBOOST_PARAMS,
        "fit_rows": len(fit), "early_rows": len(early), "heldout_rows": len(test),
        "fit_ids_sha256": fit_ids, "early_ids_sha256": early_ids,
        "held_ids_sha256": held_ids,
        "held_month_counts": report["held_month_counts"],
        "trees": int(model.tree_count_), "model_sha256": model_sha,
        "oof_sha256": oof_sha, "fit_sha256": sha(paths["fit"]),
        "paired_comparator_sha256": comparator_hashes,
    }
    check_guard_frozen(expected_source_sha256)
    if before != hashes(source_paths()):
        raise ValueError("Guard sources changed before fit receipt")
    if name == "replacement" and comparator_hashes != guard_artifact_hashes("comparator"):
        raise ValueError("Comparator changed before replacement receipt")
    write_json_new(paths["receipt"], receipt)
    del rows, features, model, oof, held, raw, y, proxy
    gc.collect()
    verify_guard_fit(name, fixed, expected_source_sha256)
    return {"name": name, "trees": receipt["trees"],
            "heldout_rows": receipt["heldout_rows"], "model_sha256": model_sha}


def verify_guard_fit(name: str, fixed: dict, expected_source_sha256: str) -> pd.DataFrame:
    paths = artifact_paths(name)
    receipt = read_json(paths["receipt"])
    report = read_json(paths["fit"])
    expected_schema = fixed["v11_schema"] if name == "comparator" else fixed["v12_schema"]
    cats = [i for i, item in enumerate(expected_schema) if item["dtype"] == "category"]
    paired = guard_artifact_hashes("comparator") if name == "replacement" else {}
    if (receipt.get("schema_version") != 1 or receipt.get("name") != name
            or receipt.get("heldout_months") != list(HELDOUT)
            or receipt.get("protocol_sha256") != sha(guard_protocol_path())
            or receipt.get("source_input_sha256") != fixed["source_input_sha256"]
            or receipt.get("feature_schema") != expected_schema
            or receipt.get("feature_schema_sha256") != old_guard.schema_hash(expected_schema)
            or receipt.get("categorical_indices") != cats
            or set(receipt.get("category_vocabularies", {})) !=
               {expected_schema[i]["name"] for i in cats}
            or receipt.get("catboost_params") != v12.EXPECTED_CATBOOST_PARAMS
            or receipt.get("model_sha256") != sha(paths["model"])
            or receipt.get("oof_sha256") != sha(paths["oof"])
            or receipt.get("fit_sha256") != sha(paths["fit"])
            or receipt.get("paired_comparator_sha256") != paired
            or any(report.get(key) != receipt.get(key) for key in (
                "name", "heldout_months", "protocol_sha256", "feature_schema",
                "categorical_indices", "category_vocabularies", "catboost_params",
                "fit_rows", "early_rows", "heldout_rows", "fit_ids_sha256",
                "early_ids_sha256", "held_ids_sha256", "held_month_counts", "trees"))
            or report.get("excluded_reserved_from_fit_and_early") is not True
            or type(receipt.get("trees")) is not int
            or not 1 <= receipt["trees"] <= 10000):
        raise ValueError(f"{name} guard receipt or fit/model hashes changed")
    baseline_receipt, expected_held = baseline_split_receipt()
    if any(receipt.get(key) != value for key, value in baseline_receipt.items()):
        raise ValueError(f"{name} fit, early or complete heldout IDs differ from baseline")
    model = CatBoostRegressor()
    model.load_model(str(paths["model"]))
    v12.verify_saved_params(model.get_all_params())
    if (int(model.tree_count_) != receipt["trees"]
            or list(model.feature_names_) != [item["name"] for item in expected_schema]
            or list(model.get_cat_feature_indices()) != cats):
        raise ValueError(f"{name} saved CatBoost metadata differs")
    frame = pd.read_parquet(paths["oof"])
    times = pd.to_datetime(frame.MVT_TIME_UTC_mvt, utc=True, errors="coerce")
    if (list(frame) != ["MVT_ID_mvt", "target", "MVT_TIME_UTC_mvt", "expert"]
            or len(frame) != receipt["heldout_rows"]
            or frame.MVT_ID_mvt.isna().any() or frame.MVT_ID_mvt.duplicated().any()
            or not np.isfinite(frame[["target", "expert"]].to_numpy(dtype=float)).all()
            or times.isna().any() or set(times.dt.month.unique()) != set(HELDOUT)
            or old_guard.id_hash(frame.MVT_ID_mvt) != receipt["held_ids_sha256"]
            or {str(month): int((times.dt.month == month).sum()) for month in HELDOUT}
               != receipt["held_month_counts"]):
        raise ValueError(f"{name} guarded OOF IDs, values or months differ")
    exact_ids(frame.MVT_ID_mvt, expected_held.MVT_ID_mvt, f"{name} complete heldout")
    aligned = expected_held.set_index("MVT_ID_mvt").loc[frame.MVT_ID_mvt.to_numpy()]
    if (not np.array_equal(frame.target.to_numpy(dtype=float),
                           aligned.target.to_numpy(dtype=float))
            or not np.array_equal(times.to_numpy(),
                                  pd.to_datetime(aligned.time, utc=True).to_numpy())):
        raise ValueError(f"{name} OOF labels or UTC times differ from baseline")
    check_guard_frozen(expected_source_sha256)
    return frame


def paired_gate(paired: pd.DataFrame, weight: float) -> tuple[dict, pd.DataFrame]:
    needed = ["MVT_ID_mvt", "target", "MVT_TIME_UTC_mvt", "v11_raw", "v12_raw"]
    if (list(paired) != needed or paired.MVT_ID_mvt.isna().any()
            or paired.MVT_ID_mvt.duplicated().any() or weight not in v12.WEIGHTS
            or weight <= 0
            or not np.isfinite(paired[["target", "v11_raw", "v12_raw"]]
                               .to_numpy(dtype=float)).all()):
        raise ValueError("Fixed paired guard input or weight differs")
    times = pd.to_datetime(paired.MVT_TIME_UTC_mvt, utc=True, errors="coerce")
    if times.isna().any() or set(times.dt.month.unique()) != set(HELDOUT):
        raise ValueError("Fixed paired guard timestamps differ")
    y = paired.target.to_numpy(dtype=float)
    old_raw = paired.v11_raw.to_numpy(dtype=float)
    new_raw = paired.v12_raw.to_numpy(dtype=float)
    old = np.maximum(old_raw, 0)
    candidate = np.maximum(old_raw + weight * (new_raw - old_raw), 0)
    months = times.dt.month.to_numpy()
    scores = {str(month): {
        "n": int((months == month).sum()),
        "v11_rmse": old_guard.rmse(y[months == month], old[months == month]),
        "fixed_blend_rmse": old_guard.rmse(y[months == month], candidate[months == month]),
    } for month in HELDOUT}
    if any(item["n"] < 1000 for item in scores.values()):
        raise ValueError("Incomplete February/August heldout coverage")
    days = times.dt.floor("D")
    group, uniques = pd.factorize(days, sort=True)
    if len(uniques) < 2 or (group < 0).any():
        raise ValueError("Paired UTC-day bootstrap lacks complete dates")
    counts = np.bincount(group).astype(float)
    left = np.bincount(group, weights=(y - old) ** 2)
    right = np.bincount(group, weights=(y - candidate) ** 2)
    draws = np.random.default_rng(SEED).integers(0, len(uniques), size=(REPEATS, len(uniques)))
    n = counts[draws].sum(axis=1)
    gains = np.sqrt(left[draws].sum(axis=1) / n) - np.sqrt(right[draws].sum(axis=1) / n)
    bootstrap = {"days": len(uniques), "repeats": REPEATS, "seed": SEED,
                 "gain_ci95_sec": np.quantile(gains, [.025, .975]).tolist()}
    passed = (all(item["fixed_blend_rmse"] < item["v11_rmse"]
                  for item in scores.values()) and bootstrap["gain_ci95_sec"][0] > 0)
    output = paired.copy()
    output["v11_clipped"] = old
    output["fixed_blend_clipped"] = candidate
    return {"scores": scores, "bootstrap": bootstrap,
            "pooled_v11_rmse": old_guard.rmse(y, old),
            "pooled_fixed_blend_rmse": old_guard.rmse(y, candidate),
            "passed": bool(passed)}, output


def evaluate_guard(expected_source_sha256: str) -> dict:
    fixed = check_guard_frozen(expected_source_sha256)
    output_path = GUARD_DIR / "paired_predictions.parquet"
    terminal_path = GUARD_DIR / "terminal.json"
    if output_path.exists() or terminal_path.exists():
        raise FileExistsError("Fixed guard already evaluated or has a partial output")
    before = dict(fixed["source_input_sha256"])
    comparator_hashes = guard_artifact_hashes("comparator")
    replacement_hashes = guard_artifact_hashes("replacement")
    left = verify_guard_fit("comparator", fixed, expected_source_sha256)
    right = verify_guard_fit("replacement", fixed, expected_source_sha256)
    exact_ids(right.MVT_ID_mvt, left.MVT_ID_mvt, "February/August paired OOF")
    right = right.set_index("MVT_ID_mvt").loc[left.MVT_ID_mvt.to_numpy()].reset_index()
    if (not np.array_equal(left.target.to_numpy(dtype=float), right.target.to_numpy(dtype=float))
            or not np.array_equal(pd.to_datetime(left.MVT_TIME_UTC_mvt, utc=True).to_numpy(),
                                  pd.to_datetime(right.MVT_TIME_UTC_mvt, utc=True).to_numpy())):
        raise ValueError("Paired models disagree on heldout target/time")
    baseline = pd.read_parquet(args_for_v12().cache_dir / "training_rows.parquet",
                               columns=["MVT_ID_mvt", "target", "proxy", "month", "time"])
    y = baseline.target.to_numpy(dtype=float)
    proxy = baseline.proxy.to_numpy(dtype=float)
    held = (baseline.month.isin(HELDOUT).to_numpy() & np.isfinite(y)
            & np.isfinite(proxy) & (proxy >= 0) & (proxy <= 7200))
    expected = baseline.loc[held]
    exact_ids(left.MVT_ID_mvt, expected.MVT_ID_mvt, "Complete February/August finite valid proxy")
    aligned = expected.set_index("MVT_ID_mvt").loc[left.MVT_ID_mvt.to_numpy()]
    if (not np.array_equal(left.target.to_numpy(dtype=float), aligned.target.to_numpy(dtype=float))
            or not np.array_equal(pd.to_datetime(left.MVT_TIME_UTC_mvt, utc=True).to_numpy(),
                                  pd.to_datetime(aligned.time, utc=True).to_numpy())):
        raise ValueError("Paired guard differs from complete baseline labels or times")
    paired = left.rename(columns={"expert": "v11_raw"})
    paired["v12_raw"] = right.expert.to_numpy(dtype=float)
    report, output = paired_gate(paired, fixed["selected_weight"])
    check_guard_frozen(expected_source_sha256)
    if (before != hashes(source_paths())
            or comparator_hashes != guard_artifact_hashes("comparator")
            or replacement_hashes != guard_artifact_hashes("replacement")):
        raise ValueError("Frozen sources or model proofs changed during guard scoring")
    paired_sha = publish_new(output_path, lambda path: output.to_parquet(path, index=False))
    terminal = {
        "schema_version": 1, "heldout_months": list(HELDOUT),
        "selected_weight": fixed["selected_weight"],
        "passed": report["passed"],
        "retained_policy": "v12" if report["passed"] else "current_v9",
        "scores": report["scores"], "bootstrap": report["bootstrap"],
        "pooled_v11_rmse": report["pooled_v11_rmse"],
        "pooled_fixed_blend_rmse": report["pooled_fixed_blend_rmse"],
        "protocol_sha256": sha(guard_protocol_path()),
        "comparator_sha256": comparator_hashes,
        "replacement_sha256": replacement_hashes,
        "paired_predictions_sha256": paired_sha,
        "architecture_comparison_only": True,
        "ranking_authorized": bool(report["passed"]),
        "outside_current_v9_unchanged_on_failure": True,
    }
    check_guard_frozen(expected_source_sha256)
    if (before != hashes(source_paths())
            or comparator_hashes != guard_artifact_hashes("comparator")
            or replacement_hashes != guard_artifact_hashes("replacement")):
        raise ValueError("Frozen inputs changed before terminal guard receipt")
    write_json_new(terminal_path, terminal)
    verify_guard_terminal(expected_source_sha256)
    return terminal


def verify_guard_terminal(expected_source_sha256: str) -> dict:
    fixed = check_guard_frozen(expected_source_sha256)
    terminal = read_json(GUARD_DIR / "terminal.json")
    left = verify_guard_fit("comparator", fixed, expected_source_sha256)
    right = verify_guard_fit("replacement", fixed, expected_source_sha256)
    exact_ids(right.MVT_ID_mvt, left.MVT_ID_mvt, "Sealed guard paired IDs")
    right = right.set_index("MVT_ID_mvt").loc[left.MVT_ID_mvt.to_numpy()].reset_index()
    paired_path = GUARD_DIR / "paired_predictions.parquet"
    paired = pd.read_parquet(paired_path)
    expected_raw = left.rename(columns={"expert": "v11_raw"})
    expected_raw["v12_raw"] = right.expert.to_numpy(dtype=float)
    raw_cols = list(expected_raw)
    if (list(paired) != raw_cols + ["v11_clipped", "fixed_blend_clipped"]
            or not paired[raw_cols].equals(expected_raw)):
        raise ValueError("Guard paired output differs from both receipt-bound OOFs")
    recomputed, expected_output = paired_gate(expected_raw, fixed["selected_weight"])
    if not paired.equals(expected_output):
        raise ValueError("Guard fixed clipped formula differs from sealed output")
    if (terminal.get("schema_version") != 1
            or terminal.get("heldout_months") != list(HELDOUT)
            or terminal.get("selected_weight") != fixed["selected_weight"]
            or terminal.get("passed") is not recomputed["passed"]
            or terminal.get("retained_policy") !=
               ("v12" if recomputed["passed"] else "current_v9")
            or terminal.get("scores") != recomputed["scores"]
            or terminal.get("bootstrap") != recomputed["bootstrap"]
            or terminal.get("pooled_v11_rmse") != recomputed["pooled_v11_rmse"]
            or terminal.get("pooled_fixed_blend_rmse") != recomputed["pooled_fixed_blend_rmse"]
            or terminal.get("protocol_sha256") != sha(guard_protocol_path())
            or terminal.get("comparator_sha256") != guard_artifact_hashes("comparator")
            or terminal.get("replacement_sha256") != guard_artifact_hashes("replacement")
            or terminal.get("paired_predictions_sha256") != sha(paired_path)
            or terminal.get("architecture_comparison_only") is not True
            or terminal.get("ranking_authorized") is not recomputed["passed"]
            or terminal.get("outside_current_v9_unchanged_on_failure") is not True):
        raise ValueError("Terminal February/August gate differs from fixed replay")
    return terminal


def require_passed_guard(expected_source_sha256: str) -> tuple[dict, dict]:
    frozen = check_guard_frozen(expected_source_sha256)
    terminal = verify_guard_terminal(expected_source_sha256)
    if (terminal["passed"] is not True or terminal["retained_policy"] != "v12"
            or terminal["ranking_authorized"] is not True):
        raise ValueError("Fixed February/August guard failed; retain current v9")
    return frozen, terminal


def final_fit_input_paths() -> dict[str, Path]:
    paths = source_paths()
    paths.update({
        "new_guard_protocol": GUARD_DIR / "protocol.json",
        "new_guard_terminal": GUARD_DIR / "terminal.json",
        "new_guard_paired": GUARD_DIR / "paired_predictions.parquet",
    })
    for name in ("comparator", "replacement"):
        paths.update({f"new_guard_{name}_{key}": path
                      for key, path in artifact_paths(name).items()})
    return paths


def final_protocol_path() -> Path:
    return FINAL_DIR / "protocol.json"


def prepare_final(expected_source_sha256: str) -> dict:
    if final_protocol_path().exists():
        return check_final_prepared(expected_source_sha256)
    fixed, terminal = require_passed_guard(expected_source_sha256)
    before = hashes(final_fit_input_paths())
    value = {
        "schema_version": 1, "selected_route": "v12",
        "selected_weight": fixed["selected_weight"],
        "source_sha256": expected_source_sha256,
        "guard_protocol_sha256": sha(guard_protocol_path()),
        "guard_terminal_sha256": sha(GUARD_DIR / "terminal.json"),
        "v12_original_validation_sha256": sha(V12_DIR / "validation.json"),
        "v12_fresh_audit_sha256": sha(V12_DIR / "fresh_audit.json"),
        "feature_schema": fixed["v12_schema"],
        "fit_input_sha256": before,
        "status": "frozen_before_full_2025_fit",
    }
    if terminal["retained_policy"] != "v12":
        raise ValueError("Failed guard cannot prepare a final model")
    check_hashes(final_fit_input_paths(), before, "Final fit prepare")
    write_json_new(final_protocol_path(), value)
    return check_final_prepared(expected_source_sha256)


def check_final_prepared(expected_source_sha256: str) -> dict:
    fixed, _ = require_passed_guard(expected_source_sha256)
    prepared = read_json(final_protocol_path())
    if (prepared.get("schema_version") != 1
            or prepared.get("selected_route") != "v12"
            or prepared.get("selected_weight") != fixed["selected_weight"]
            or prepared.get("source_sha256") != expected_source_sha256
            or prepared.get("guard_protocol_sha256") != sha(guard_protocol_path())
            or prepared.get("guard_terminal_sha256") != sha(GUARD_DIR / "terminal.json")
            or prepared.get("v12_original_validation_sha256") != sha(V12_DIR / "validation.json")
            or prepared.get("v12_fresh_audit_sha256") != sha(V12_DIR / "fresh_audit.json")
            or prepared.get("feature_schema") != fixed["v12_schema"]
            or prepared.get("status") != "frozen_before_full_2025_fit"):
        raise ValueError("Final fit guard/protocol/input seal differs")
    check_hashes(final_fit_input_paths(), prepared["fit_input_sha256"], "Final fit")
    return prepared


def original_rounds_and_schema(frozen: dict) -> tuple[int, list[int], list[dict]]:
    """Use only the two original complementary-fold tree counts."""
    receipts = [read_json(V12_DIR / f"{name}_provenance.json") for name in FOLDS]
    reports = [read_json(V12_DIR / f"{name}_validation.json") for name in FOLDS]
    feature_schema = frozen["v12_schema"]
    cats = [i for i, item in enumerate(feature_schema) if item["dtype"] == "category"]
    trees = []
    for name, receipt, report in zip(FOLDS, receipts, reports):
        if (receipt.get("fold") != name
                or receipt.get("heldout_months") != list(deep.FOLDS[name])
                or receipt.get("feature_schema") != feature_schema
                or receipt.get("categorical_feature_indices") != cats
                or receipt.get("catboost_params") != v12.EXPECTED_CATBOOST_PARAMS
                or receipt.get("model_sha256") != sha(V12_DIR / f"{name}.cbm")
                or receipt.get("oof_sha256") != sha(V12_DIR / f"{name}_oof.parquet")
                or receipt.get("fit_report_sha256") != sha(V12_DIR / f"{name}_validation.json")
                or report.get("trees") != receipt.get("trees")
                or report.get("features") != [item["name"] for item in feature_schema]
                or type(report.get("trees")) is not int
                or not 1 <= report["trees"] <= 10000):
            raise ValueError(f"Original {name} architecture or tree receipt changed")
        trees.append(report["trees"])
    return int(np.median(trees)), trees, feature_schema


def model_path() -> Path:
    return FINAL_DIR / "full_2025.cbm"


def model_report_path() -> Path:
    return FINAL_DIR / "final_model.json"


def verify_final_model(expected_source_sha256: str) -> dict:
    prepared = check_final_prepared(expected_source_sha256)
    frozen = check_guard_frozen(expected_source_sha256)
    rounds, original_trees, feature_schema = original_rounds_and_schema(frozen)
    report = read_json(model_report_path())
    cats = [i for i, item in enumerate(feature_schema) if item["dtype"] == "category"]
    vocab = read_json(artifact_paths("replacement")["receipt"])["category_vocabularies"]
    expected_params = dict(v12.EXPECTED_CATBOOST_PARAMS)
    expected_params["iterations"] = rounds
    if (report.get("iterations") != rounds
            or report.get("original_fold_trees") != dict(zip(FOLDS, original_trees))
            or report.get("feature_schema") != feature_schema
            or report.get("categorical_indices") != cats
            or report.get("category_vocabularies") != vocab
            or report.get("fixed_catboost_params") != expected_params
            or report.get("model_sha256") != sha(model_path())
            or report.get("final_protocol_sha256") != sha(final_protocol_path())
            or report.get("fit_input_sha256") != prepared["fit_input_sha256"]
            or report.get("guard_terminal_sha256") != sha(GUARD_DIR / "terminal.json")
            or report.get("selected_weight") != frozen["selected_weight"]
            or report.get("ranking_prediction_created") is not False
            or report.get("training_rows") != EXPECTED_TRAIN
            or type(report.get("eligible_training_rows")) is not int
            or report["eligible_training_rows"] < 1_500_000):
        raise ValueError("Full v12 model report or source/category seal differs")
    model = CatBoostRegressor()
    model.load_model(str(model_path()))
    v12.verify_saved_params(model.get_all_params())
    if (int(model.tree_count_) != rounds
            or list(model.feature_names_) != [item["name"] for item in feature_schema]
            or list(model.get_cat_feature_indices()) != cats):
        raise ValueError("Full model saved feature/category/tree architecture differs")
    return report


def fit_final(expected_source_sha256: str) -> dict:
    require_memory()
    prepared = check_final_prepared(expected_source_sha256)
    if model_path().exists() or model_report_path().exists():
        raise FileExistsError("Full v12 model has complete or partial prior outputs")
    frozen = check_guard_frozen(expected_source_sha256)
    rounds, original_trees, expected_schema = original_rounds_and_schema(frozen)
    rows, features = v12.load_features(args_for_v12())
    if (len(rows) != EXPECTED_TRAIN or len(features) != EXPECTED_TRAIN
            or schema(features) != expected_schema
            or rows.MVT_ID_mvt.isna().any() or rows.MVT_ID_mvt.duplicated().any()):
        raise ValueError("Full 2025 v12 feature universe differs")
    categories = features.select_dtypes(include="category").columns.tolist()
    vocab = old_guard.category_hashes(features)
    if (len(categories) != 24
            or vocab != read_json(artifact_paths("replacement")["receipt"])
               ["category_vocabularies"]):
        raise ValueError("Full 2025 categorical vocabulary differs from matched guard")
    y = rows.target.to_numpy(dtype=float)
    proxy = rows.proxy.to_numpy(dtype=float)
    eligible = (np.isfinite(y) & (y >= 0) & (y <= 86400)
                & np.isfinite(proxy) & (proxy >= 0) & (proxy <= 7200))
    if int(eligible.sum()) < 1_500_000:
        raise ValueError("Unexpectedly few complete eligible 2025 training rows")
    check_final_prepared(expected_source_sha256)
    params = dict(v12.EXPECTED_CATBOOST_PARAMS)
    params["iterations"] = rounds
    model = CatBoostRegressor(**params)
    train = Pool(features.loc[eligible], label=(y - proxy)[eligible],
                 cat_features=categories)
    started = time.monotonic()
    model.fit(train)
    elapsed = time.monotonic() - started
    del train, rows, features
    gc.collect()
    cats = [i for i, item in enumerate(expected_schema) if item["dtype"] == "category"]
    if (int(model.tree_count_) != rounds
            or list(model.feature_names_) != [item["name"] for item in expected_schema]
            or list(model.get_cat_feature_indices()) != cats):
        raise ValueError("Full fit differs from original v12 architecture")
    v12.verify_saved_params(model.get_all_params())
    check_final_prepared(expected_source_sha256)
    model_sha = publish_new(model_path(), lambda path: model.save_model(str(path)))
    report = {
        "selected_route": "v12", "selected_weight": frozen["selected_weight"],
        "iterations": rounds, "original_fold_trees": dict(zip(FOLDS, original_trees)),
        "training_rows": EXPECTED_TRAIN, "eligible_training_rows": int(eligible.sum()),
        "feature_schema": expected_schema, "categorical_indices": cats,
        "category_vocabularies": vocab, "fixed_catboost_params": params,
        "fit_seconds": float(elapsed), "model_sha256": model_sha,
        "final_protocol_sha256": sha(final_protocol_path()),
        "fit_input_sha256": prepared["fit_input_sha256"],
        "guard_terminal_sha256": sha(GUARD_DIR / "terminal.json"),
        "ranking_prediction_created": False,
    }
    check_final_prepared(expected_source_sha256)
    write_json_new(model_report_path(), report)
    verify_final_model(expected_source_sha256)
    return {key: report[key] for key in ("iterations", "original_fold_trees",
                                        "eligible_training_rows", "model_sha256")}


def ranking_input_paths() -> dict[str, Path]:
    args = args_for_v12()
    paths = {f"v9_{name}": path for name, path in
             v9_final.ranking_input_paths("v11").items()}
    paths.update({
        "raw_2026_ranking": args.data_dir / "ranking.parquet",
        "submission_template": TEMPLATE,
        "frozen_v9_submission": V9_SUBMISSION,
        "v9_final_predictions": args.v9_final_dir / "predictions.parquet",
        "v9_final_manifest": args.v9_final_dir / "ranking_manifest.json",
        "v9_final_expert": args.v9_final_dir / "ranking_expert.parquet",
        "geometry_ranking_cache": args.geometry_dir / "ranking_runway_geometry_features.parquet",
        "geometry_training_cache": args.geometry_dir / "training_runway_geometry_features.parquet",
        "geometry_protocol": args.geometry_dir / "protocol.json",
        "geometry_build_receipt": args.geometry_dir / "build_receipt.json",
        "geometry_source_receipt": args.geometry_source_dir / "feature_spec.json",
        "full_v12_model": model_path(),
        "full_v12_report": model_report_path(),
        "v12_final_protocol": final_protocol_path(),
        "feb_aug_guard_terminal": GUARD_DIR / "terminal.json",
    })
    return paths


def ranking_seal_path() -> Path:
    return FINAL_DIR / "ranking_inputs.json"


def seal_ranking(expected_source_sha256: str) -> dict:
    if ranking_seal_path().exists():
        return check_ranking_seal(expected_source_sha256)
    prepared = check_final_prepared(expected_source_sha256)
    final_report = verify_final_model(expected_source_sha256)
    if sha(V9_SUBMISSION) != V9_SHA256:
        raise ValueError("Sealed local v9 prediction changed")
    paths = ranking_input_paths()
    before = hashes(paths)
    value = {
        "schema_version": 1, "selected_route": "v12",
        "selected_weight": prepared["selected_weight"],
        "source_sha256": expected_source_sha256,
        "final_protocol_sha256": sha(final_protocol_path()),
        "final_model_sha256": final_report["model_sha256"],
        "ranking_input_sha256": before,
        "fit_input_sha256": prepared["fit_input_sha256"],
        "status": "sealed_before_any_2026_feature_value_read",
    }
    check_hashes(paths, before, "Ranking input freeze")
    write_json_new(ranking_seal_path(), value)
    return check_ranking_seal(expected_source_sha256)


def check_ranking_seal(expected_source_sha256: str) -> dict:
    prepared = check_final_prepared(expected_source_sha256)
    value = read_json(ranking_seal_path())
    model_report = verify_final_model(expected_source_sha256)
    if (value.get("schema_version") != 1
            or value.get("selected_route") != "v12"
            or value.get("selected_weight") != prepared["selected_weight"]
            or value.get("source_sha256") != expected_source_sha256
            or value.get("final_protocol_sha256") != sha(final_protocol_path())
            or value.get("final_model_sha256") != model_report["model_sha256"]
            or value.get("fit_input_sha256") != prepared["fit_input_sha256"]
            or value.get("status") != "sealed_before_any_2026_feature_value_read"
            or sha(V9_SUBMISSION) != V9_SHA256):
        raise ValueError("Ranking seal, final model or sealed local v9 changed")
    check_hashes(ranking_input_paths(), value["ranking_input_sha256"], "Ranking")
    return value


def fixed_ranking_blend(current: np.ndarray, raw: np.ndarray,
                        valid: np.ndarray, weight: float) -> np.ndarray:
    if (current.shape != raw.shape or current.shape != valid.shape
            or valid.dtype != bool or weight not in v12.WEIGHTS or weight <= 0
            or not np.isfinite(current).all() or (current < 0).any()
            or not np.isfinite(raw[valid]).all()):
        raise ValueError("V12 fixed ranking formula inputs or valid-AOBT gate differ")
    output = current.copy()
    output[valid] = np.maximum(current[valid] + weight * (raw[valid] - current[valid]), 0)
    if (not np.array_equal(output[~valid], current[~valid])
            or not np.isfinite(output).all() or (output < 0).any()):
        raise ValueError("V12 fixed formula changed nonvalid v9 predictions")
    return output


def load_ranking_features() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Append the already sealed 16-field geometry cache to exact v11 order."""
    rows, base = v9_final.load_ranking_features("v11")
    if len(base.columns) != 184 or len(rows) != EXPECTED_RANK:
        raise ValueError("Sealed v11 ranking feature matrix changed")
    cache = args_for_v12().geometry_dir / "ranking_runway_geometry_features.parquet"
    built = pd.read_parquet(cache)
    if (list(built) != ["MVT_ID_mvt", *geometry.FEATURES]
            or len(built) != len(rows)
            or not np.array_equal(built.MVT_ID_mvt.to_numpy(), rows.MVT_ID_mvt.to_numpy())
            or any(name in base for name in geometry.FEATURES)
            or any(built[name].dtype != np.dtype("float32") for name in geometry.FEATURES)):
        raise ValueError("Geometry ranking cache lacks exact ID/order/float32 schema")
    geometry.verify_output(built, rows[["MVT_ID_mvt"]])
    result = pd.concat([base.reset_index(drop=True),
                        built[list(geometry.FEATURES)].reset_index(drop=True)], axis=1)
    if (len(result.columns) != 200 or result.columns.duplicated().any()
            or FORBIDDEN.intersection(result.columns)):
        raise ValueError("V12 200-field ranking matrix contains forbidden or duplicate fields")
    return rows, result


def predict_ranking(expected_source_sha256: str) -> dict:
    require_memory()
    output_path = FINAL_DIR / "predictions.parquet"
    expert_path = FINAL_DIR / "ranking_expert.parquet"
    manifest_path = FINAL_DIR / "ranking_manifest.json"
    if any(path.exists() for path in (output_path, expert_path, manifest_path)):
        raise FileExistsError("Final v12 ranking artifacts already exist or are partial")
    ranking_seal = check_ranking_seal(expected_source_sha256)
    frozen = check_guard_frozen(expected_source_sha256)
    rounds, _, expected_schema = original_rounds_and_schema(frozen)
    model_report = verify_final_model(expected_source_sha256)
    rows, features = load_ranking_features()
    if (schema(features) != expected_schema
            or rows.MVT_ID_mvt.isna().any() or rows.MVT_ID_mvt.duplicated().any()):
        raise ValueError("Ranking predictor schema or exact IDs differ from original v12")
    template = pd.read_parquet(TEMPLATE)
    current = pd.read_parquet(V9_SUBMISSION)
    sealed_v9 = pd.read_parquet(args_for_v12().v9_final_dir / "predictions.parquet")
    columns = ["MVT_ID_mvt", "TAXITIME_SEC_mvt"]
    if (list(template) != columns or list(current) != columns
            or list(sealed_v9) != columns or len(template) != EXPECTED_RANK
            or any(item.MVT_ID_mvt.isna().any() or item.MVT_ID_mvt.duplicated().any()
                   for item in (template, current, sealed_v9, rows))
            or any(not np.array_equal(item.MVT_ID_mvt.to_numpy(),
                                      template.MVT_ID_mvt.to_numpy())
                   for item in (current, sealed_v9, rows))
            or not np.array_equal(current.TAXITIME_SEC_mvt.to_numpy(dtype=float),
                                  sealed_v9.TAXITIME_SEC_mvt.to_numpy(dtype=float))):
        raise ValueError("Sealed local v9 ranking reference/template ID order or values changed")
    proxy = rows.proxy.to_numpy(dtype=float)
    valid = np.isfinite(proxy) & (proxy >= 0) & (proxy <= 7200)
    if int(valid.sum()) != EXPECTED_VALID or int((~valid).sum()) != 5_464:
        raise ValueError("Exact valid-AOBT ranking gate differs from frozen v9")
    current_values = current.TAXITIME_SEC_mvt.to_numpy(dtype=float, copy=True)
    v8_local = pd.read_parquet(ROOT / "artifacts/current-candidate/predictions.parquet")
    v7_local = pd.read_parquet(ROOT / "artifacts/v7-runway-traffic/predictions.parquet")
    if (list(v8_local) != columns or list(v7_local) != columns
            or any(not np.array_equal(item.MVT_ID_mvt.to_numpy(),
                                      template.MVT_ID_mvt.to_numpy())
                   for item in (v8_local, v7_local))
            or not np.array_equal(current_values[~valid],
                                  v8_local.TAXITIME_SEC_mvt.to_numpy(dtype=float)[~valid])
            or int(np.count_nonzero(
                v8_local.TAXITIME_SEC_mvt.to_numpy(dtype=float)[~valid]
                != v7_local.TAXITIME_SEC_mvt.to_numpy(dtype=float)[~valid])) != 4_907):
        raise ValueError("Sealed v9 outside-valid and 4,907 missing-clock rows changed")
    check_ranking_seal(expected_source_sha256)
    model = CatBoostRegressor()
    model.load_model(str(model_path()))
    v12.verify_saved_params(model.get_all_params())
    cats = [i for i, item in enumerate(expected_schema) if item["dtype"] == "category"]
    if (int(model.tree_count_) != rounds
            or list(model.feature_names_) != [item["name"] for item in expected_schema]
            or list(model.get_cat_feature_indices()) != cats):
        raise ValueError("Saved full v12 model no longer matches original architecture")
    raw = np.full(len(rows), np.nan, dtype=float)
    raw[valid] = proxy[valid] + model.predict(features.loc[valid], thread_count=2)
    output = fixed_ranking_blend(current_values, raw, valid, frozen["selected_weight"])
    check_ranking_seal(expected_source_sha256)
    expert = pd.DataFrame({"MVT_ID_mvt": template.MVT_ID_mvt.to_numpy(copy=True),
                           "a_valid": valid, "raw_expert": raw})
    expert_sha = publish_new(expert_path, lambda path: expert.to_parquet(path, index=False))
    result = pd.DataFrame({"MVT_ID_mvt": template.MVT_ID_mvt.to_numpy(copy=True),
                           "TAXITIME_SEC_mvt": output})
    result_sha = publish_new(output_path, lambda path: result.to_parquet(path, index=False))
    readback = pd.read_parquet(output_path)
    if (list(readback) != columns
            or not np.array_equal(readback.MVT_ID_mvt.to_numpy(), template.MVT_ID_mvt.to_numpy())
            or not np.array_equal(readback.TAXITIME_SEC_mvt.to_numpy(dtype=float), output)
            or not np.array_equal(output[~valid], current_values[~valid])
            or not np.isfinite(output).all() or (output < 0).any()):
        raise ValueError("Published ranking readback, mask or finite values differ")
    check_ranking_seal(expected_source_sha256)
    report = {
        "schema_version": 1, "selected_route": "v12",
        "selected_weight": frozen["selected_weight"],
        "rows": EXPECTED_RANK, "valid_aobt_rows": EXPECTED_VALID,
        "outside_valid_rows": 5_464, "missing_clock_preserved_rows": 4_907,
        "nonvalid_v9_unchanged": True,
        "current_missing_clock_policy_preserved": True,
        "template_order_verified": True, "finite_nonnegative": True,
        "v9_submission_sha256": V9_SHA256,
        "final_model_sha256": model_report["model_sha256"],
        "final_model_report_sha256": sha(model_report_path()),
        "guard_terminal_sha256": sha(GUARD_DIR / "terminal.json"),
        "ranking_seal_sha256": sha(ranking_seal_path()),
        "ranking_input_sha256": ranking_seal["ranking_input_sha256"],
        "ranking_expert_sha256": expert_sha,
        "predictions_sha256": result_sha,
        "prediction_bytes": output_path.stat().st_size,
        "uploaded": False,
    }
    write_json_new(manifest_path, report)
    check_ranking_seal(expected_source_sha256)
    return report


def synthetic() -> dict:
    """In-memory mask, fixed-formula, date gate and exclusive-receipt checks."""
    current = np.array([10., 20., 30., 40.])
    raw = np.array([8., np.nan, -10., np.nan])
    valid = np.array([True, False, True, False])
    out = fixed_ranking_blend(current, raw, valid, .5)
    if not np.array_equal(out, np.array([9., 20., 10., 40.])):
        raise AssertionError("Fixed ranking blend changed nonvalid rows or clipping")
    for bad in (np.array([True, True, True, False]),
                np.array([False, False, False, False])):
        if bad[1]:
            try:
                fixed_ranking_blend(current, raw, bad, .5)
            except ValueError:
                pass
            else:
                raise AssertionError("Nonfinite raw valid prediction was accepted")
    for bad_weight in (0., .3, float("nan")):
        try:
            fixed_ranking_blend(current, raw, valid, bad_weight)
        except ValueError:
            pass
        else:
            raise AssertionError("An unselected ranking weight was accepted")
    times = pd.to_datetime(["2025-02-01T00:00:00Z", "2025-02-02T00:00:00Z",
                            "2025-08-01T00:00:00Z", "2025-08-02T00:00:00Z"])
    # The fixed gate requires realistic complete month counts. Repeat four
    # independent UTC days without touching competition targets.
    n = 1200
    test = pd.DataFrame({
        "MVT_ID_mvt": np.arange(4 * n, dtype=np.int64),
        "target": np.full(4 * n, 10.),
        "MVT_TIME_UTC_mvt": times.repeat(n),
        "v11_raw": np.full(4 * n, 20.),
        "v12_raw": np.full(4 * n, 10.),
    })
    report, output = paired_gate(test, 1.0)
    if (not report["passed"] or report["bootstrap"]["seed"] != SEED
            or not np.array_equal(output.fixed_blend_clipped.to_numpy(),
                                  np.full(4 * n, 10.))):
        raise AssertionError("Known paired February/August improvement failed")
    worse = test.copy()
    worse.loc[worse.MVT_TIME_UTC_mvt.dt.month.eq(8), "v12_raw"] = 40.
    if paired_gate(worse, 1.0)[0]["passed"]:
        raise AssertionError("Worse August was accepted")
    if old_guard.id_hash(pd.Series([1, 2, 3])) == old_guard.id_hash(pd.Series([2, 1, 3])):
        raise AssertionError("ID order is not sealed")
    vocabulary_a = pd.DataFrame({"airport": pd.Categorical(
        ["a", "b"], categories=["a", "b"])})
    vocabulary_b = pd.DataFrame({"airport": pd.Categorical(
        ["a", "b"], categories=["b", "a"])})
    if old_guard.category_hashes(vocabulary_a) == old_guard.category_hashes(vocabulary_b):
        raise AssertionError("Category vocabulary order is not sealed")
    with tempfile.TemporaryDirectory(prefix="v12-final-synthetic-") as directory:
        path = Path(directory) / "receipt.json"
        write_json_new(path, {"sealed": True})
        expected = hashes({"receipt": path})
        try:
            write_json_new(path, {"sealed": False})
        except FileExistsError:
            pass
        else:
            raise AssertionError("Exclusive receipt could be overwritten")
        path.write_text('{"sealed": false}', encoding="utf-8")
        try:
            check_hashes({"receipt": path}, expected, "Synthetic tamper")
        except ValueError:
            pass
        else:
            raise AssertionError("Changed receipt bytes were accepted")
    return {"synthetic": "passed", "heldout_months": list(HELDOUT),
            "bootstrap_seed": SEED, "real_data_read": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=(
        "prepare", "guardfit-comparator", "guardfit-replacement", "guard-eval",
        "finalprepare", "finalfit", "rankingseal", "predict", "synthetic"))
    parser.add_argument("--published-source-sha256", default="")
    args = parser.parse_args()
    if args.mode == "synthetic":
        result = synthetic()
    elif args.mode == "prepare":
        result = freeze_guard(args.published_source_sha256)
    elif args.mode == "guardfit-comparator":
        result = fit_guard("comparator", args.published_source_sha256)
    elif args.mode == "guardfit-replacement":
        result = fit_guard("replacement", args.published_source_sha256)
    elif args.mode == "guard-eval":
        result = evaluate_guard(args.published_source_sha256)
    elif args.mode == "finalprepare":
        result = prepare_final(args.published_source_sha256)
    elif args.mode == "finalfit":
        result = fit_final(args.published_source_sha256)
    elif args.mode == "rankingseal":
        result = seal_ranking(args.published_source_sha256)
    else:
        result = predict_ranking(args.published_source_sha256)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
