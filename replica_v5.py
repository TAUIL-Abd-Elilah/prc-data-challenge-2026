"""Receipt-bound isolated reproduction of the fixed, clean v5 taxi-out chain.

Only ``plan`` and ``self-test`` are intended before source/spec publication.
Every real mode requires an independently rebuilt and replayed replica-v4 root.
It trains fresh missing-clock and ARR experts; original competition artifacts
are neither imported as model bytes nor accepted as parent substitutes.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import replica_v4_timestamp as v4


HERE = Path(__file__).resolve().parent
SPEC = HERE / "reports/clean_replication_v5_spec.json"
FOLDS = {"seasonal_jan_jul": (1, 7), "forward_nov_dec": (11, 12)}
EXPECTED_VALIDATION = 672_428
EXPECTED_RANKING = 344_841
EXPECTED_RANK_VALID = 339_377
MIN_FREE_GIB = 10.0
MISSING_WEIGHT = 0.5
ARRIVAL_WEIGHT = 0.25
ARRIVAL_GRID = (0.0, 0.1, 0.25, 0.5, 1.0)
LEGACY_V3_ERRATUM_SHA256 = "209176e6e75453aa18d68406992f6ac17994622d9244b06a341738acbe99a4f3"

# These are public source bytes, never substitutes for a newly trained model.
PINNED_SOURCES = {
    "deep_timestamp_expert.py": "8edf1eb0b00b065c2d6b5fbb1d7b85b013ba861b12c55a574d53dc8ead14ad82",
    "missing_catboost.py": "952a1419095033b5ae4a8d6f7e8d0e597999370c50679bd718c6bbe3d7bb97ef",
    "arrival_residual_expert.py": "09aec3bf3c04280b80e378e4bde71ab4926242b0c345ef8b356204553822ceb4",
    "arrival_features.py": "f52164dccf311cd90163c720390768b660207eb6f45099de39797763258c249a",
    "v5_ensemble.py": "373b4251ed795266206737586f8456f868c73581e313c1185dfd5d10e84233ab",
    "replica_v4_timestamp.py": "7f8f5fd0cfe7b54f10e110e80aeddf0ca6e84966978de6a6f0e44604d2d96a11",
    "catboost_expert.py": "1e1ecfec5acfe41524f028339984ae56fdb1b8c6e360fde704acdd64258345f2",
    "solution.py": "13848cd8483737c1e5db0ac4e1e90c0b64eead8ce289cc17870944f58715aa11",
    "weather_model.py": "6dbfe4f41550b97da604e698c00c3fbbafc0ef6489ea6b47e4099e5658b7c1ec",
    "v4_reference.py": "5bf163d64863cc695f0a3fdfab379e2d76b6c056332622f9ffa9d9ee5eeddaa0",
}

V4_FILES = {
    "protocol": "protocol.json",
    "reference": "v4/validation_predictions.parquet",
    "reference_receipt": "v4/reference.json",
    "deep_validation": "deep/validation.json",
    "deep_validation_predictions": "deep/validation_predictions.parquet",
    "deep_seasonal_model": "deep/seasonal_jan_jul.cbm",
    "deep_seasonal_oof": "deep/seasonal_jan_jul_oof.parquet",
    "deep_seasonal_receipt": "deep/seasonal_jan_jul_receipt.json",
    "deep_forward_model": "deep/forward_nov_dec.cbm",
    "deep_forward_oof": "deep/forward_nov_dec_oof.parquet",
    "deep_forward_receipt": "deep/forward_nov_dec_receipt.json",
    "deep_final_model": "deep/full_2025.cbm",
    "deep_final_receipt": "deep/full_2025_receipt.json",
    "deep_ranking_seal": "deep/ranking_inputs.json",
    "deep_ranking_expert": "deep/ranking_expert.parquet",
    "deep_ranking_prediction": "deep/predictions.parquet",
    "deep_ranking_manifest": "deep/ranking_manifest.json",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def digest_ids(values: Any) -> str:
    return v4.id_hash(values)


def digest_float(values: Any) -> str:
    return v4.float_hash(values)


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def require_sha(value: str | None, name: str, actual: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or value.lower() != actual:
        raise ValueError(f"Published {name} attestation missing or different")


def source_snapshot() -> dict[str, str]:
    result = {}
    for name, expected in PINNED_SOURCES.items():
        result[name] = sha256(HERE / name)
        if result[name] != expected:
            raise ValueError(f"Frozen scientific source changed: {name}")
    result["replica_v5.py"] = sha256(Path(__file__).resolve())
    result["reports/clean_replication_v5_spec.json"] = sha256(SPEC)
    return result


def check_publication(args: argparse.Namespace) -> dict[str, str]:
    sources = source_snapshot()
    require_sha(args.published_source_sha256, "v5 adapter source",
                sources["replica_v5.py"])
    require_sha(args.published_spec_sha256, "v5 adapter spec",
                sources["reports/clean_replication_v5_spec.json"])
    return sources


def require_memory() -> None:
    v4.require_memory()


def paths_and_snapshot(args: argparse.Namespace) -> tuple[Path, Path, Path, dict, dict, dict]:
    sources = check_publication(args)
    v4args = argparse.Namespace(run_root=args.run_root,
                                published_source_sha256=args.published_v4_sha256)
    root, v4out, manifest, parents, upstream = v4.real_context(v4args)
    from solution import _training_files
    expected_training = [parents[f"raw_training_{m:02d}"] for m in range(1, 13)]
    actual_training = _training_files(parents["raw_ranking"].parent)
    if [p.resolve() for p in actual_training] != [p.resolve() for p in expected_training]:
        raise ValueError("Raw training discovery differs from twelve sealed canonical months")
    if sources["replica_v4_timestamp.py"] != upstream["source_sha256"]["replica_v4_timestamp.py"]:
        raise ValueError("v4 adapter source lineage differs")
    if manifest.get("legacy_validation_erratum_sha256") != LEGACY_V3_ERRATUM_SHA256:
        raise ValueError("Inherited v3 early-stop disclosure changed")
    out = v4.strict_child(root, "artifacts/replica-v5", existing=False)
    fingerprint = {"source_sha256": sources,
                   "v4_source_sha256": upstream["source_sha256"],
                   "v4_parent_manifest_sha256": upstream["manifest_sha256"],
                   "legacy_validation_erratum_sha256": manifest[
                       "legacy_validation_erratum_sha256"],
                   "v4_parent_file_sha256": upstream["parent_file_sha256"],
                   "v4_output_sha256": {name: sha256(v4out / relative)
                                        for name, relative in V4_FILES.items()},
                   "raw_and_cache_sha256": {name: sha256(path)
                                             for name, path in parents.items()}}
    return root, out, v4out, parents, manifest, fingerprint


def assert_snapshot(args: argparse.Namespace, expected: dict) -> None:
    _, _, _, _, _, current = paths_and_snapshot(args)
    if current != expected:
        raise ValueError("Frozen source, raw/cache parent or v4 producer byte changed")


def exclusive_file(root: Path, target: Path, write: Any, suffix: str) -> None:
    target = target.resolve(strict=False)
    if root not in target.parents or target.exists():
        raise FileExistsError(f"Replica output already exists or escapes run root: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_name = tempfile.mkstemp(prefix=".replica-v5-", suffix=suffix,
                                    dir=target.parent)
    os.close(fd)
    stage = Path(raw_name)
    try:
        write(stage)
        os.link(stage, target)
    finally:
        stage.unlink(missing_ok=True)


def exclusive_json(root: Path, target: Path, value: dict) -> None:
    payload = json.dumps(value, indent=2, allow_nan=False).encode("utf-8")
    exclusive_file(root, target, lambda p: p.write_bytes(payload), ".json")


def exclusive_parquet(root: Path, target: Path, value: pd.DataFrame) -> None:
    exclusive_file(root, target, lambda p: value.to_parquet(p, index=False), ".parquet")


def exclusive_saved_model(root: Path, target: Path, model: Any) -> None:
    exclusive_file(root, target, lambda p: model.save_model(str(p)), target.suffix)


def protocol_value(snapshot: dict) -> dict:
    return {"schema_version": 1, "status": "frozen_before_v5_values",
            "source_sha256": snapshot["source_sha256"],
            "upstream": {k: v for k, v in snapshot.items() if k != "source_sha256"},
            "folds": {k: list(v) for k, v in FOLDS.items()},
            "missing_weight": MISSING_WEIGHT, "arrival_weight": ARRIVAL_WEIGHT,
            "missing_direct_iterations": 700, "missing_direct_depth": 6,
            "arrival_max_rounds": 1200, "minimum_free_gib": MIN_FREE_GIB,
            "new_v5_components_fit_and_early_exclude_held_months": True,
            "inherited_legacy_v3_early_stop_used_held_month_labels": True,
            "full_policy_folds_untouched": False,
            "legacy_arrival_included": False}


def require_protocol(root: Path, out: Path, snapshot: dict,
                     *, create: bool = False) -> dict:
    value = protocol_value(snapshot)
    path = out / "protocol.json"
    if not path.exists():
        if not create:
            raise FileNotFoundError("Publish the isolated v5 protocol before fitting")
        exclusive_json(root, path, value)
    elif read_json(path) != value:
        raise ValueError("Frozen v5 protocol differs")
    return value


def prepare(args: argparse.Namespace) -> dict:
    root, out, v4out, parents, parent_manifest, snapshot = paths_and_snapshot(args)
    receipt_path = out / "prepared.json"
    if receipt_path.exists():
        raise FileExistsError("V5 prepare receipt exists; no implicit reprepare")
    value = require_protocol(root, out, snapshot, create=True)
    # Full verification recursively replays reference and both original folds.
    v4snap = v4.parent_snapshot(root, v4.publication_check(args.published_v4_sha256))[2]
    v4.require_final(root, v4out, parents, v4snap)
    v4.require_ranking_seal(v4out, v4snap)
    verify_v4_ranking(v4out, parents, parent_manifest)
    reference = clean_reference(v4out)
    if len(reference) != EXPECTED_VALIDATION:
        raise ValueError("Clean v4 all-finite reference coverage differs")
    assert_snapshot(args, snapshot)
    receipt = {"status": "complete", "protocol_sha256": sha256(out / "protocol.json"),
               "source_input_sha256": snapshot,
               "upstream_model_and_ranking_replay": True,
               "all_finite_rows": len(reference)}
    exclusive_json(root, receipt_path, receipt)
    assert_snapshot(args, snapshot)
    return {"status": "complete", "protocol_status": value["status"],
            "protocol_sha256": sha256(out / "protocol.json"),
            "v4_parent_genesis": "independent receipts and saved-model replays passed",
            "all_finite_rows": len(reference), "prepared_receipt_sha256": sha256(receipt_path)}


def verify_v4_ranking(v4out: Path, parents: dict, parent_manifest: dict) -> None:
    template, v4base = v4.verify_v4_rank(parents, parent_manifest)
    deep = pd.read_parquet(v4out / "deep/ranking_expert.parquet")
    final = pd.read_parquet(v4out / "deep/predictions.parquet")
    if (len(template) != EXPECTED_RANKING or list(deep) != ["MVT_ID_mvt", "expert"]
            or list(final) != ["MVT_ID_mvt", "TAXITIME_SEC_mvt"]
            or not deep.MVT_ID_mvt.equals(template.MVT_ID_mvt)
            or not final.MVT_ID_mvt.equals(template.MVT_ID_mvt)):
        raise ValueError("Timestamp ranking output coverage/order differs")
    proxy = pd.read_parquet(parents["baseline_ranking_rows"], columns=["proxy"]).proxy.to_numpy(dtype=float)
    gate = np.isfinite(proxy) & (proxy >= 0) & (proxy <= 7200)
    expert = deep.expert.to_numpy(dtype=float)
    expected = v4base.copy()
    if not np.array_equal(np.isfinite(expert), gate):
        raise ValueError("Timestamp ranking expert gate differs")
    expected[gate] = np.maximum(v4base[gate] + .5 * (expert[gate] - v4base[gate]), 0)
    if not np.array_equal(expected, final.TAXITIME_SEC_mvt.to_numpy(dtype=float)):
        raise ValueError("Timestamp ranking parent fixed formula differs")
    receipt = read_json(v4out / "deep/ranking_manifest.json")
    if (receipt.get("status") != "complete"
            or receipt.get("expert_sha256") != sha256(v4out / "deep/ranking_expert.parquet")
            or receipt.get("predictions_sha256") != sha256(v4out / "deep/predictions.parquet")
            or receipt.get("final_model_sha256") != sha256(v4out / "deep/full_2025.cbm")
            or receipt.get("ranking_seal_sha256") != sha256(v4out / "deep/ranking_inputs.json")
            or receipt.get("ordered_ids_sha256") != digest_ids(template.MVT_ID_mvt)):
        raise ValueError("Timestamp ranking model/output manifest differs")
    from deep_timestamp_expert import load_features
    p = v4.adapter_args(parents, v4out / "deep")
    rank_rows, rank_x = load_features(p, ranking=True)
    if not rank_rows.MVT_ID_mvt.equals(template.MVT_ID_mvt):
        raise ValueError("Timestamp ranking model feature IDs differ")
    final_report = read_json(v4out / "deep/full_2025_receipt.json")
    model = v4.verify_model(v4out / "deep/full_2025.cbm", rank_x,
                            requested_iterations=int(final_report["iterations"]),
                            expected_trees=int(final_report["actual_trees"]))
    replay = np.full(len(template), np.nan)
    replay[gate] = proxy[gate] + model.predict(rank_x.loc[gate], thread_count=2)
    if not np.allclose(replay[gate], expert[gate], rtol=1e-11, atol=1e-8):
        raise ValueError("Timestamp saved full model does not replay ranking expert")


def adapter_args(parents: dict, output: Path) -> argparse.Namespace:
    return argparse.Namespace(data_dir=parents["raw_ranking"].parent,
                              cache_dir=parents["baseline_training_rows"].parent,
                              weather_file=parents["noaa_weather"], output_dir=output,
                              threads=4, seed=2026, direct_iterations=700,
                              long_iterations=400, max_rounds=1200,
                              v4_dir=output.parent / "unused_v4",
                              v4_ranking=output.parent / "unused_v4_ranking")


def ensure_base_rows(rows: pd.DataFrame, reference: pd.DataFrame) -> None:
    if len(reference) != EXPECTED_VALIDATION or reference.MVT_ID_mvt.duplicated().any():
        raise ValueError("Invalid all-finite clean reference")
    if rows.MVT_ID_mvt.isna().any() or rows.MVT_ID_mvt.duplicated().any():
        raise ValueError("Training movement IDs are not unique")
    if not pd.Index(reference.MVT_ID_mvt).isin(rows.MVT_ID_mvt).all():
        raise ValueError("Reference IDs absent from independently built baseline rows")
    aligned = rows.set_index("MVT_ID_mvt").reindex(reference.MVT_ID_mvt)
    if (not np.array_equal(aligned.target.to_numpy(dtype=float), reference.target.to_numpy(dtype=float))
            or not np.array_equal(aligned.month.to_numpy(dtype=int), reference.month.to_numpy(dtype=int))):
        raise ValueError("Reference labels/months differ from independently built rows")


def clean_reference(v4out: Path) -> pd.DataFrame:
    frame = pd.read_parquet(v4out / "v4/validation_predictions.parquet",
                            columns=["MVT_ID_mvt", "target", "fold", "a_valid", "selected",
                                     "airport", "month", "MVT_TIME_UTC_mvt"])
    if len(frame) != EXPECTED_VALIDATION or not np.isfinite(frame.target).all():
        raise ValueError("Clean v4 reference coverage/labels differ")
    return frame


def rmse(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(y - pred))))


def real_stage(args: argparse.Namespace) -> tuple[Path, Path, Path, dict, dict, dict]:
    result = paths_and_snapshot(args)
    root, out, _, _, _, snapshot = result
    require_protocol(root, out, snapshot)
    prepared = read_json(out / "prepared.json")
    if (prepared.get("status") != "complete"
            or prepared.get("protocol_sha256") != sha256(out / "protocol.json")
            or prepared.get("source_input_sha256") != snapshot
            or prepared.get("all_finite_rows") != EXPECTED_VALIDATION
            or prepared.get("upstream_model_and_ranking_replay") is not True):
        raise ValueError("Clean v4 parent replay/prepare receipt is missing or changed")
    return result


def missing_fold_paths(out: Path, fold: str) -> dict[str, Path]:
    base = out / "missing"
    return {"model": base / f"{fold}.cbm",
            "oof": base / f"{fold}_oof.parquet",
            "receipt": base / f"{fold}_receipt.json"}


def arrival_fold_paths(out: Path, fold: str) -> dict[str, Path]:
    base = out / "arrival"
    return {"model": base / f"{fold}.txt",
            "oof": base / f"{fold}_oof.parquet",
            "receipt": base / f"{fold}_receipt.json"}


def split_missing(rows: pd.DataFrame, features: pd.DataFrame,
                  fold: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import missing_catboost as missing
    mask = missing.masks(rows, features, FOLDS[fold])
    ordinary = np.flatnonzero(mask["ordinary_train"])
    held = np.flatnonzero(mask["direct_test"])
    if len(ordinary) < 1000:
        raise ValueError("Too few complementary ordinary missing-clock training rows")
    shuffled = np.random.default_rng(2026 + len("ordinary_direct")).permutation(ordinary)
    early_n = max(500, int(.1 * len(ordinary)))
    early_n = min(early_n, max(1, len(ordinary) // 3))
    fit, early = shuffled[early_n:], shuffled[:early_n]
    if (rows.month.iloc[fit].isin(FOLDS[fold]).any()
            or rows.month.iloc[early].isin(FOLDS[fold]).any()
            or not rows.month.iloc[held].isin(FOLDS[fold]).all()):
        raise ValueError("Missing-clock fit/early/held month separation differs")
    return fit, early, held


def schema(features: pd.DataFrame) -> dict:
    return v4.schema_of(features)


def verify_missing_model(model_path: Path, features: pd.DataFrame,
                         *, expected_trees: int) -> Any:
    from catboost import CatBoostRegressor
    model = CatBoostRegressor()
    model.load_model(str(model_path))
    if model.feature_names_ != list(features) or int(model.tree_count_) != expected_trees:
        raise ValueError("Saved missing-clock model feature order/tree count differs")
    cats = [features.columns.get_loc(col) for col in features.select_dtypes(include="category")]
    if list(model.get_cat_feature_indices()) != cats:
        raise ValueError("Saved missing-clock model categorical positions differ")
    actual = model.get_all_params()
    expected = {"task_type": "CPU", "depth": 6, "random_seed": 2026,
                "loss_function": "RMSE", "learning_rate": .045,
                "l2_leaf_reg": 20, "border_count": 128,
                "max_ctr_complexity": 1, "one_hot_max_size": 20,
                "random_strength": .5, "bagging_temperature": .5}
    for key, value in expected.items():
        found = actual.get(key)
        if isinstance(value, float):
            if found is None or not np.isclose(float(found), value, rtol=1e-7, atol=1e-7):
                raise ValueError(f"Saved missing-clock model {key} differs")
        elif found != value:
            raise ValueError(f"Saved missing-clock model {key} differs")
    if not 1 <= expected_trees <= 700:
        raise ValueError("Missing-clock model exceeded original 700-tree cap")
    return model


def fit_missing_fold(args: argparse.Namespace) -> dict:
    if args.fold not in FOLDS:
        raise ValueError("Select one original fold")
    root, out, v4out, parents, _, snap = real_stage(args)
    dest = missing_fold_paths(out, args.fold)
    if any(p.exists() for p in dest.values()):
        raise FileExistsError("Missing-clock fold output exists; no implicit resume")
    import missing_catboost as missing
    p = adapter_args(parents, out / "missing")
    rows, features = missing.load_inputs(p)
    reference = clean_reference(v4out)
    ensure_base_rows(rows, reference)
    fit, early, held = split_missing(rows, features, args.fold)
    names, cats = missing.columns(features, long=False)
    if not names or not cats:
        raise ValueError("Original missing-clock feature schema is incomplete")
    selected = features[names]
    y = rows.target.to_numpy(dtype=float)
    # Original train_model shuffles ordinary internally with its name seed.
    mask = missing.masks(rows, features, FOLDS[args.fold])
    ordinary = np.flatnonzero(mask["ordinary_train"])
    model, native = missing.train_model(p, "ordinary_direct", "regression",
                                        features, ordinary, y, names, cats, 700, 6)
    if (native["fit_rows"] != len(fit) or native["internal_early_rows"] != len(early)
            or native["train_rows"] != len(ordinary)):
        raise ValueError("Original missing-clock internal split changed")
    pred = model.predict(selected.iloc[held], thread_count=4)
    oof = pd.DataFrame({"MVT_ID_mvt": rows.MVT_ID_mvt.iloc[held].to_numpy(),
                        "target": y[held], "direct_candidate": pred})
    if not np.isfinite(pred).all() or len(oof) != len(held):
        raise ValueError("Missing-clock fold produced nonfinite/incomplete OOF")
    assert_snapshot(args, snap)
    exclusive_saved_model(root, dest["model"], model)
    saved = verify_missing_model(dest["model"], selected,
                                 expected_trees=int(model.tree_count_))
    replay = saved.predict(selected.iloc[held], thread_count=4)
    if not np.allclose(pred, replay, rtol=1e-11, atol=1e-8):
        raise ValueError("Saved missing-clock model does not replay fold predictions")
    exclusive_parquet(root, dest["oof"], oof)
    if not pd.read_parquet(dest["oof"]).equals(oof):
        raise ValueError("Missing-clock OOF readback differs")
    receipt = {"status": "complete", "fold": args.fold,
               "heldout_months": list(FOLDS[args.fold]),
               "fit_and_early_exclude_held_months": True,
               "protocol_sha256": sha256(out / "protocol.json"),
               "source_input_sha256": snap, "model_sha256": sha256(dest["model"]),
               "oof_sha256": sha256(dest["oof"]),
               "schema": schema(selected), "requested_iterations": 700,
               "best_iteration": int(model.get_best_iteration()),
               "trees": int(model.tree_count_),
               "fit_ids_sha256": digest_ids(rows.MVT_ID_mvt.iloc[fit]),
               "early_ids_sha256": digest_ids(rows.MVT_ID_mvt.iloc[early]),
               "held_ids_sha256": digest_ids(rows.MVT_ID_mvt.iloc[held]),
               "fit_target_sha256": digest_float(y[fit]),
               "early_target_sha256": digest_float(y[early]),
               "held_target_sha256": digest_float(y[held]),
               "held_prediction_sha256": digest_float(replay),
               "training_eligible": len(ordinary), "held_rows": len(held)}
    assert_snapshot(args, snap)
    exclusive_json(root, dest["receipt"], receipt)
    return receipt


def verify_missing_fold(args: argparse.Namespace, fold: str,
                        rows: pd.DataFrame, features: pd.DataFrame,
                        out: Path, snapshot: dict) -> tuple[dict, pd.DataFrame]:
    import missing_catboost as missing
    dest = missing_fold_paths(out, fold)
    receipt = read_json(dest["receipt"])
    fit, early, held = split_missing(rows, features, fold)
    names, _ = missing.columns(features, long=False)
    selected = features[names]
    y = rows.target.to_numpy(dtype=float)
    if (receipt.get("status") != "complete" or receipt.get("fold") != fold
            or receipt.get("heldout_months") != list(FOLDS[fold])
            or receipt.get("fit_and_early_exclude_held_months") is not True
            or receipt.get("protocol_sha256") != sha256(out / "protocol.json")
            or receipt.get("source_input_sha256") != snapshot
            or receipt.get("model_sha256") != sha256(dest["model"])
            or receipt.get("oof_sha256") != sha256(dest["oof"])
            or receipt.get("schema") != schema(selected)
            or receipt.get("requested_iterations") != 700
            or receipt.get("fit_ids_sha256") != digest_ids(rows.MVT_ID_mvt.iloc[fit])
            or receipt.get("early_ids_sha256") != digest_ids(rows.MVT_ID_mvt.iloc[early])
            or receipt.get("held_ids_sha256") != digest_ids(rows.MVT_ID_mvt.iloc[held])
            or receipt.get("fit_target_sha256") != digest_float(y[fit])
            or receipt.get("early_target_sha256") != digest_float(y[early])
            or receipt.get("held_target_sha256") != digest_float(y[held])
            or receipt.get("training_eligible") != len(fit) + len(early)
            or receipt.get("held_rows") != len(held)):
        raise ValueError(f"{fold} missing-clock producer receipt differs")
    trees = int(receipt["trees"])
    if receipt.get("best_iteration") != trees - 1:
        raise ValueError("Missing-clock saved best iteration/tree count differs")
    model = verify_missing_model(dest["model"], selected, expected_trees=trees)
    saved = pd.read_parquet(dest["oof"])
    replay = model.predict(selected.iloc[held], thread_count=4)
    if (list(saved) != ["MVT_ID_mvt", "target", "direct_candidate"]
            or not np.array_equal(saved.MVT_ID_mvt.to_numpy(),
                                  rows.MVT_ID_mvt.iloc[held].to_numpy())
            or not np.array_equal(saved.target.to_numpy(dtype=float), y[held])
            or not np.allclose(saved.direct_candidate.to_numpy(dtype=float), replay,
                               rtol=1e-11, atol=1e-8)
            or receipt.get("held_prediction_sha256") != digest_float(replay)):
        raise ValueError("Saved missing-clock model→OOF replay differs")
    return receipt, saved


def build_arrival_features(args: argparse.Namespace, scope: str) -> dict:
    if scope not in ("training", "ranking"):
        raise ValueError("Unknown ARR feature scope")
    root, out, _, parents, _, snap = real_stage(args)
    if scope == "ranking":
        require_ranking_seal(args, root, out, snap)
    target = out / "arrival" / f"{scope}_arrival_features.parquet"
    receipt_path = out / "arrival" / f"{scope}_features_receipt.json"
    if target.exists() or receipt_path.exists():
        raise FileExistsError("ARR feature cache exists; no implicit rebuild")
    import arrival_residual_expert as arrival
    raw = ([parents[f"raw_training_{m:02d}"] for m in range(1, 13)]
           if scope == "training" else [parents["raw_ranking"]])
    row_file = parents[f"baseline_{scope}_rows"]
    def write(stage: Path) -> None:
        arrival._build_one(raw, row_file, stage)
    assert_snapshot(args, snap)
    exclusive_file(root, target, write, ".parquet")
    row_ids = pd.read_parquet(row_file, columns=["MVT_ID_mvt"])
    built = pd.read_parquet(target)
    if (len(built) != len(row_ids)
            or not built.MVT_ID_mvt.equals(row_ids.MVT_ID_mvt)
            or built.MVT_ID_mvt.duplicated().any()):
        raise ValueError("ARR feature cache IDs/order differ from clean baseline")
    receipt = {"status": "complete", "scope": scope,
               "protocol_sha256": sha256(out / "protocol.json"),
               "ranking_seal_sha256": sha256(out / "ranking_inputs.json")
               if scope == "ranking" else None,
               "source_input_sha256": snap,
               "cache_sha256": sha256(target), "ordered_ids_sha256": digest_ids(row_ids.MVT_ID_mvt),
               "rows": len(built), "feature_schema": schema(built)}
    assert_snapshot(args, snap)
    exclusive_json(root, receipt_path, receipt)
    return receipt


def require_arrival_cache(args: argparse.Namespace, out: Path,
                          parents: dict, snapshot: dict, scope: str) -> dict:
    target = out / "arrival" / f"{scope}_arrival_features.parquet"
    receipt_path = out / "arrival" / f"{scope}_features_receipt.json"
    receipt = read_json(receipt_path)
    if (receipt.get("status") != "complete" or receipt.get("scope") != scope
            or receipt.get("protocol_sha256") != sha256(out / "protocol.json")
            or receipt.get("source_input_sha256") != snapshot
            or receipt.get("cache_sha256") != sha256(target)
            or receipt.get("ranking_seal_sha256") != (
                sha256(out / "ranking_inputs.json") if scope == "ranking" else None)):
        raise ValueError("ARR feature cache receipt differs")
    row_ids = pd.read_parquet(parents[f"baseline_{scope}_rows"], columns=["MVT_ID_mvt"])
    built = pd.read_parquet(target)
    if (receipt.get("rows") != len(row_ids) or len(built) != len(row_ids)
            or receipt.get("ordered_ids_sha256") != digest_ids(row_ids.MVT_ID_mvt)
            or not built.MVT_ID_mvt.equals(row_ids.MVT_ID_mvt)
            or receipt.get("feature_schema") != schema(built)):
        raise ValueError("ARR feature cache source ID/schema replay differs")
    return receipt


def split_arrival(rows: pd.DataFrame, fold: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import arrival_residual_expert as arrival
    train_mask, held_mask = arrival._masks(rows, FOLDS[fold])
    train = np.flatnonzero(train_mask)
    held = np.flatnonzero(held_mask)
    shuffled = np.random.default_rng(2026 if fold == "seasonal_jan_jul" else 2033).permutation(train)
    early_n = max(30000, int(.06 * len(shuffled)))
    if early_n <= 0 or early_n >= len(shuffled):
        raise ValueError("ARR early split is empty or consumes training")
    fit, early = shuffled[early_n:], shuffled[:early_n]
    if (rows.month.iloc[fit].isin(FOLDS[fold]).any()
            or rows.month.iloc[early].isin(FOLDS[fold]).any()
            or not rows.month.iloc[held].isin(FOLDS[fold]).all()):
        raise ValueError("ARR fit/early/held month separation differs")
    return fit, early, held


def verify_arrival_model(path: Path, features: pd.DataFrame,
                         *, expected_trees: int) -> Any:
    import lightgbm as lgb
    model = lgb.Booster(model_file=str(path))
    cats = [features.columns.get_loc(name) for name in features.select_dtypes(include="category")]
    if (model.feature_name() != list(features) or model.num_trees() != expected_trees
            or not 1 <= expected_trees <= 1200):
        raise ValueError("Saved ARR LightGBM feature order/tree count differs")
    dumped = model.dump_model()
    # LightGBM persists categorical splits and pandas category vocabularies.
    if len(dumped.get("pandas_categorical", []) or []) != len(cats):
        raise ValueError("Saved ARR LightGBM categorical metadata differs")
    if not model.params:
        raise ValueError("Saved ARR LightGBM lacks native parameter metadata")
    expected = {"num_leaves": 63, "min_data_in_leaf": 100,
                "bagging_freq": 1, "max_cat_threshold": 64, "seed": 2026,
                "feature_fraction_seed": 2026, "bagging_seed": 2026,
                "learning_rate": .045, "feature_fraction": .85,
                "bagging_fraction": .85, "lambda_l2": 12, "cat_smooth": 20}
    for key, value in expected.items():
        found = model.params.get(key)
        if found is None:
            raise ValueError(f"Saved ARR LightGBM lacks {key}")
        if not np.isclose(float(found), float(value), rtol=1e-7, atol=1e-7):
            raise ValueError(f"Saved ARR LightGBM {key} differs")
    if str(model.params.get("objective", "")).lower() not in ("regression", "regression_l2"):
        raise ValueError("Saved ARR LightGBM objective differs")
    return model


def fit_arrival_fold(args: argparse.Namespace) -> dict:
    if args.fold not in FOLDS:
        raise ValueError("Select one original fold")
    root, out, v4out, parents, _, snap = real_stage(args)
    require_arrival_cache(args, out, parents, snap, "training")
    dest = arrival_fold_paths(out, args.fold)
    if any(p.exists() for p in dest.values()):
        raise FileExistsError("ARR fold output exists; no implicit resume")
    import arrival_residual_expert as arrival
    import lightgbm as lgb
    p = adapter_args(parents, out / "arrival")
    rows, features = arrival._load_all(p, False)
    reference = clean_reference(v4out)
    ensure_base_rows(rows, reference)
    fit, early, held = split_arrival(rows, args.fold)
    y = rows.target.to_numpy(dtype=float)
    proxy = rows.proxy.to_numpy(dtype=float)
    residual = y - proxy
    cats = [name for name in features if isinstance(features[name].dtype, pd.CategoricalDtype)]
    train_set = lgb.Dataset(features.iloc[fit], label=residual[fit],
                            categorical_feature=cats, free_raw_data=True)
    early_set = lgb.Dataset(features.iloc[early], label=residual[early],
                            categorical_feature=cats, reference=train_set,
                            free_raw_data=True)
    model = lgb.train(arrival._params(4), train_set, num_boost_round=1200,
                      valid_sets=[early_set], callbacks=[lgb.early_stopping(100, verbose=False),
                                                        lgb.log_evaluation(100)])
    expert = proxy[held] + model.predict(features.iloc[held], num_threads=4)
    if not np.isfinite(expert).all():
        raise ValueError("ARR fold expert has nonfinite held-out values")
    oof = pd.DataFrame({"MVT_ID_mvt": rows.MVT_ID_mvt.iloc[held].to_numpy(),
                        "target": y[held], "arrival_direct_expert": expert})
    assert_snapshot(args, snap)
    exclusive_saved_model(root, dest["model"], model)
    saved = verify_arrival_model(dest["model"], features,
                                 expected_trees=int(model.best_iteration))
    replay = proxy[held] + saved.predict(features.iloc[held], num_threads=4)
    if not np.allclose(replay, expert, rtol=1e-11, atol=1e-8):
        raise ValueError("Saved ARR LightGBM does not replay OOF")
    exclusive_parquet(root, dest["oof"], oof)
    if not pd.read_parquet(dest["oof"]).equals(oof):
        raise ValueError("ARR OOF Parquet readback differs")
    receipt = {"status": "complete", "fold": args.fold,
               "heldout_months": list(FOLDS[args.fold]),
               "fit_and_early_exclude_held_months": True,
               "protocol_sha256": sha256(out / "protocol.json"),
               "training_arrival_features_sha256": sha256(out / "arrival/training_arrival_features.parquet"),
               "source_input_sha256": snap, "model_sha256": sha256(dest["model"]),
               "oof_sha256": sha256(dest["oof"]), "schema": schema(features),
               "params": arrival._params(4), "requested_rounds": 1200,
               "best_iteration": int(model.best_iteration),
               "trees": int(saved.num_trees()),
               "fit_ids_sha256": digest_ids(rows.MVT_ID_mvt.iloc[fit]),
               "early_ids_sha256": digest_ids(rows.MVT_ID_mvt.iloc[early]),
               "held_ids_sha256": digest_ids(rows.MVT_ID_mvt.iloc[held]),
               "fit_target_sha256": digest_float(y[fit]),
               "early_target_sha256": digest_float(y[early]),
               "held_target_sha256": digest_float(y[held]),
               "fit_residual_label_sha256": digest_float(residual[fit]),
               "early_residual_label_sha256": digest_float(residual[early]),
               "held_proxy_sha256": digest_float(proxy[held]),
               "held_prediction_sha256": digest_float(replay),
               "training_eligible": len(fit) + len(early), "held_rows": len(held)}
    assert_snapshot(args, snap)
    exclusive_json(root, dest["receipt"], receipt)
    del train_set, early_set, model, saved, features, rows
    gc.collect()
    return receipt


def verify_arrival_fold(args: argparse.Namespace, fold: str,
                        rows: pd.DataFrame, features: pd.DataFrame,
                        out: Path, snapshot: dict) -> tuple[dict, pd.DataFrame]:
    import arrival_residual_expert as arrival
    dest = arrival_fold_paths(out, fold)
    receipt = read_json(dest["receipt"])
    fit, early, held = split_arrival(rows, fold)
    y = rows.target.to_numpy(dtype=float)
    proxy = rows.proxy.to_numpy(dtype=float)
    if (receipt.get("status") != "complete" or receipt.get("fold") != fold
            or receipt.get("heldout_months") != list(FOLDS[fold])
            or receipt.get("fit_and_early_exclude_held_months") is not True
            or receipt.get("protocol_sha256") != sha256(out / "protocol.json")
            or receipt.get("source_input_sha256") != snapshot
            or receipt.get("training_arrival_features_sha256") != sha256(
                out / "arrival/training_arrival_features.parquet")
            or receipt.get("model_sha256") != sha256(dest["model"])
            or receipt.get("oof_sha256") != sha256(dest["oof"])
            or receipt.get("schema") != schema(features)
            or receipt.get("params") != arrival._params(4)
            or receipt.get("requested_rounds") != 1200
            or receipt.get("fit_ids_sha256") != digest_ids(rows.MVT_ID_mvt.iloc[fit])
            or receipt.get("early_ids_sha256") != digest_ids(rows.MVT_ID_mvt.iloc[early])
            or receipt.get("held_ids_sha256") != digest_ids(rows.MVT_ID_mvt.iloc[held])
            or receipt.get("fit_target_sha256") != digest_float(y[fit])
            or receipt.get("early_target_sha256") != digest_float(y[early])
            or receipt.get("held_target_sha256") != digest_float(y[held])
            or receipt.get("fit_residual_label_sha256") != digest_float((y - proxy)[fit])
            or receipt.get("early_residual_label_sha256") != digest_float((y - proxy)[early])
            or receipt.get("held_proxy_sha256") != digest_float(proxy[held])
            or receipt.get("training_eligible") != len(fit) + len(early)
            or receipt.get("held_rows") != len(held)):
        raise ValueError(f"{fold} ARR producer receipt differs")
    trees = int(receipt["trees"])
    if receipt.get("best_iteration") != trees:
        raise ValueError("ARR LightGBM best iteration/tree count differs")
    model = verify_arrival_model(dest["model"], features, expected_trees=trees)
    saved = pd.read_parquet(dest["oof"])
    replay = proxy[held] + model.predict(features.iloc[held], num_threads=4)
    if (list(saved) != ["MVT_ID_mvt", "target", "arrival_direct_expert"]
            or not np.array_equal(saved.MVT_ID_mvt.to_numpy(),
                                  rows.MVT_ID_mvt.iloc[held].to_numpy())
            or not np.array_equal(saved.target.to_numpy(dtype=float), y[held])
            or not np.allclose(saved.arrival_direct_expert.to_numpy(dtype=float), replay,
                               rtol=1e-11, atol=1e-8)
            or receipt.get("held_prediction_sha256") != digest_float(replay)):
        raise ValueError("Saved ARR LightGBM→OOF replay differs")
    return receipt, saved


def align(base: pd.DataFrame, source: pd.DataFrame, field: str,
          *, full: bool = False) -> np.ndarray:
    if source.MVT_ID_mvt.isna().any() or source.MVT_ID_mvt.duplicated().any():
        raise ValueError("Expert OOF has null/duplicate IDs")
    if not pd.Index(source.MVT_ID_mvt).isin(base.MVT_ID_mvt).all():
        raise ValueError("Expert OOF has ID outside clean base")
    if full and len(source) != len(base):
        raise ValueError("Expert OOF lacks full base coverage")
    aligned = source.set_index("MVT_ID_mvt")[field].reindex(base.MVT_ID_mvt)
    return aligned.to_numpy(dtype=float)


def strict_clean_v4_oof(v4out: Path, reference: pd.DataFrame) -> np.ndarray:
    deep = pd.read_parquet(v4out / "deep/validation_predictions.parquet")
    if not {"MVT_ID_mvt", "timestamp_selected"}.issubset(deep):
        raise ValueError("Clean timestamp parent OOF lacks selected prediction")
    selected = align(reference, deep, "timestamp_selected", full=True)
    base = reference.selected.to_numpy(dtype=float)
    valid = reference.a_valid.to_numpy(dtype=bool)
    raw = align(reference, deep, "expert", full=True)
    expected = base.copy()
    expected[valid] = np.maximum(base[valid] + .5 * (raw[valid] - base[valid]), 0)
    if (not np.array_equal(selected, expected)
            or not np.array_equal(selected[~valid], base[~valid])):
        raise ValueError("Clean timestamp OOF fixed blend differs")
    return selected


def score_arrival_grid(reference: pd.DataFrame, raw: np.ndarray,
                       fold: str) -> dict[str, float]:
    use = reference.fold.eq(fold).to_numpy(dtype=bool)
    valid = reference.a_valid.to_numpy(dtype=bool)
    if not np.array_equal(np.isfinite(raw[use]), valid[use]):
        raise ValueError("ARR OOF must cover exactly valid AOBT held-out rows")
    base = reference.selected.to_numpy(dtype=float)
    y = reference.target.to_numpy(dtype=float)
    scores = {}
    for weight in ARRIVAL_GRID:
        pred = base.copy()
        gate = use & valid
        pred[gate] = np.maximum(base[gate] + weight * (raw[gate] - base[gate]), 0)
        scores[str(weight)] = rmse(y[use], pred[use])
    return scores


def day_gain(frame: pd.DataFrame, old: np.ndarray, new: np.ndarray,
             fold: str, seed: int) -> dict:
    mask = frame.fold.eq(fold).to_numpy(dtype=bool)
    dates = pd.to_datetime(frame.loc[mask, "MVT_TIME_UTC_mvt"], utc=True).dt.floor("D")
    codes, days = pd.factorize(dates, sort=True)
    y = frame.target.to_numpy(dtype=float)[mask]
    counts = np.bincount(codes, minlength=len(days)).astype(float)
    old_sse = np.bincount(codes, weights=(y - old[mask]) ** 2, minlength=len(days))
    new_sse = np.bincount(codes, weights=(y - new[mask]) ** 2, minlength=len(days))
    draw = np.random.default_rng(seed).integers(0, len(days), size=(1000, len(days)))
    gain = np.sqrt(old_sse[draw].sum(axis=1) / counts[draw].sum(axis=1)) - np.sqrt(
        new_sse[draw].sum(axis=1) / counts[draw].sum(axis=1))
    return {"days": len(days), "observed_rmse_gain_sec": rmse(y, old[mask]) - rmse(y, new[mask]),
            "bootstrap_gain_95pct_interval_sec": [float(v) for v in np.quantile(gain, [.025, .975])]}


def compose_validation(reference: pd.DataFrame, deep: np.ndarray,
                       direct: np.ndarray, arrival: np.ndarray) -> tuple[dict, pd.DataFrame]:
    y = reference.target.to_numpy(dtype=float)
    base = reference.selected.to_numpy(dtype=float)
    valid = reference.a_valid.to_numpy(dtype=bool)
    if (not np.isfinite(y).all() or not np.isfinite(base).all()
            or not np.isfinite(deep).all() or np.any(base < 0)):
        raise ValueError("Nonfinite/negative clean v4 or deep OOF")
    if np.any(np.isfinite(direct) & valid):
        raise ValueError("Missing direct expert overlaps valid AOBT")
    if np.any(np.isfinite(direct) & reference.airport.eq("LIRF").to_numpy()):
        raise ValueError("Missing direct expert changes LIRF")
    if not np.array_equal(np.isfinite(arrival), valid):
        raise ValueError("Clean ARR OOF gate differs from valid AOBT")
    missing = deep.copy()
    gate = np.isfinite(direct)
    missing_vs_v4 = {}
    for fold in FOLDS:
        use = reference.fold.eq(fold).to_numpy(dtype=bool)
        option_scores = {}
        for weight in (0.0, 0.5, 1.0):
            option = base.copy()
            option[gate] = np.maximum(base[gate] + weight *
                                      (direct[gate] - base[gate]), 0)
            option_scores[str(weight)] = rmse(y[use], option[use])
        missing_vs_v4[fold] = option_scores
    if min((0.0, 0.5, 1.0), key=lambda w: missing_vs_v4[
            "seasonal_jan_jul"][str(w)]) != MISSING_WEIGHT:
        raise ValueError("Original Jan/Jul missing-clock choice no longer selects 0.5")
    if not missing_vs_v4["forward_nov_dec"]["0.5"] < missing_vs_v4[
            "forward_nov_dec"]["0.0"]:
        raise ValueError("Original missing-clock 0.5 does not improve Nov/Dec")
    missing[gate] = np.maximum(.5 * missing[gate] + .5 * direct[gate], 0)
    final = missing.copy()
    final[valid] = np.maximum(final[valid] + ARRIVAL_WEIGHT *
                              (arrival[valid] - final[valid]), 0)
    grid = {fold: score_arrival_grid(reference, arrival, fold) for fold in FOLDS}
    chosen = min(ARRIVAL_GRID, key=lambda w: (grid["seasonal_jan_jul"][str(w)], w))
    if chosen != ARRIVAL_WEIGHT:
        raise ValueError("Original clean ARR Jan/Jul selection no longer chooses 0.25")
    if not grid["forward_nov_dec"][str(chosen)] < grid["forward_nov_dec"]["0.0"]:
        raise ValueError("Clean ARR fixed 0.25 does not transfer to Nov/Dec")
    scores = {}
    for fold in FOLDS:
        mask = reference.fold.eq(fold).to_numpy(dtype=bool)
        scores[fold] = {"n": int(mask.sum()), "v4": rmse(y[mask], base[mask]),
                        "deep": rmse(y[mask], deep[mask]),
                        "deep_missing": rmse(y[mask], missing[mask]),
                        "deep_missing_clean_arrival": rmse(y[mask], final[mask]),
                        "arrival_grid_against_v4": grid[fold],
                        "missing_grid_against_v4": missing_vs_v4[fold],
                        "missing_day_gain": day_gain(reference, deep, missing, fold, 20261002),
                        "arrival_day_gain": day_gain(reference, missing, final, fold, 20261003)}
        if not (scores[fold]["deep_missing"] < scores[fold]["deep"]
                and scores[fold]["deep_missing_clean_arrival"] < scores[fold]["deep_missing"]):
            raise ValueError(f"Fixed v5 stages fail original all-finite {fold} gate")
    if (not np.isfinite(final).all() or np.any(final < 0)
            or not np.array_equal(final[~(valid | gate)], base[~(valid | gate)])):
        raise ValueError("Fixed v5 composition changes outside gate or is invalid")
    output = reference[["MVT_ID_mvt", "target", "fold", "a_valid", "airport", "month",
                        "MVT_TIME_UTC_mvt"]].copy()
    output["v4"] = base
    output["deep"] = deep
    output["deep_missing"] = missing
    output["selected"] = final
    report = {"status": "passed", "selected_stage": "deep_missing_clean_arrival",
              "fixed_weights": {"deep": .5, "missing_direct": .5, "clean_arrival": .25},
              "legacy_arrival_included": False, "all_finite_rows": len(reference),
              "folds": scores, "gate_rows": {"valid_aobt": int(valid.sum()),
                                              "ordinary_missing_direct": int(gate.sum())},
              "validation_limit": "Repeated 2025 local comparisons; not untouched final estimates"}
    return report, output


def evaluate(args: argparse.Namespace) -> dict:
    root, out, v4out, parents, _, snap = real_stage(args)
    require_arrival_cache(args, out, parents, snap, "training")
    report_path = out / "validation.json"
    oof_path = out / "validation_predictions.parquet"
    if report_path.exists() or oof_path.exists():
        raise FileExistsError("V5 validation output exists; no implicit rescore")
    import missing_catboost as missing
    import arrival_residual_expert as arrival
    reference = clean_reference(v4out)
    deep = strict_clean_v4_oof(v4out, reference)
    p = adapter_args(parents, out / "missing")
    missing_rows, missing_x = missing.load_inputs(p)
    ensure_base_rows(missing_rows, reference)
    missing_receipts, missing_parts = {}, []
    expected_direct = []
    for fold in FOLDS:
        missing_receipts[fold], part = verify_missing_fold(
            args, fold, missing_rows, missing_x, out, snap)
        missing_parts.append(part)
        held = np.flatnonzero(missing.masks(missing_rows, missing_x,
                                          FOLDS[fold])["direct_test"])
        expected_direct.extend(missing_rows.MVT_ID_mvt.iloc[held].tolist())
    missing_oof = pd.concat(missing_parts, ignore_index=True)
    if (len(missing_oof) != len(expected_direct)
            or not np.array_equal(missing_oof.MVT_ID_mvt.to_numpy(), np.asarray(expected_direct))
            or missing_oof.MVT_ID_mvt.duplicated().any()):
        raise ValueError("Missing direct OOF gate does not match original mask")
    direct = align(reference, missing_oof, "direct_candidate")
    del missing_rows, missing_x
    gc.collect()
    p.output_dir = out / "arrival"
    arrival_rows, arrival_x = arrival._load_all(p, False)
    ensure_base_rows(arrival_rows, reference)
    arrival_receipts, arrival_parts = {}, []
    expected_arrival = []
    for fold in FOLDS:
        arrival_receipts[fold], part = verify_arrival_fold(
            args, fold, arrival_rows, arrival_x, out, snap)
        arrival_parts.append(part)
        held = split_arrival(arrival_rows, fold)[2]
        expected_arrival.extend(arrival_rows.MVT_ID_mvt.iloc[held].tolist())
    arrival_oof = pd.concat(arrival_parts, ignore_index=True)
    if (len(arrival_oof) != len(expected_arrival)
            or not np.array_equal(arrival_oof.MVT_ID_mvt.to_numpy(), np.asarray(expected_arrival))
            or arrival_oof.MVT_ID_mvt.duplicated().any()):
        raise ValueError("Clean ARR OOF gate does not match original mask")
    arrival_raw = align(reference, arrival_oof, "arrival_direct_expert")
    value, frame = compose_validation(reference, deep, direct, arrival_raw)
    value.update({"protocol_sha256": sha256(out / "protocol.json"),
                  "prepared_receipt_sha256": sha256(out / "prepared.json"),
                  "source_input_sha256": snap,
                  "v4_timestamp_validation_sha256": sha256(
                      v4out / "deep/validation_predictions.parquet"),
                  "missing_fold_receipts_sha256": {k: sha256(missing_fold_paths(out, k)["receipt"])
                                                   for k in FOLDS},
                  "arrival_fold_receipts_sha256": {k: sha256(arrival_fold_paths(out, k)["receipt"])
                                                   for k in FOLDS},
                  "arrival_feature_receipt_sha256": sha256(
                      out / "arrival/training_features_receipt.json"),
                  "ordered_ids_sha256": digest_ids(frame.MVT_ID_mvt)})
    assert_snapshot(args, snap)
    exclusive_parquet(root, oof_path, frame)
    if not pd.read_parquet(oof_path).equals(frame):
        raise ValueError("V5 validation prediction readback differs")
    value["validation_predictions_sha256"] = sha256(oof_path)
    assert_snapshot(args, snap)
    exclusive_json(root, report_path, value)
    return value


def require_validation(args: argparse.Namespace) -> tuple[dict, pd.DataFrame]:
    root, out, v4out, parents, _, snap = real_stage(args)
    require_arrival_cache(args, out, parents, snap, "training")
    reference = clean_reference(v4out)
    deep = strict_clean_v4_oof(v4out, reference)
    import missing_catboost as missing
    import arrival_residual_expert as arrival
    p = adapter_args(parents, out / "missing")
    missing_rows, missing_x = missing.load_inputs(p)
    ensure_base_rows(missing_rows, reference)
    missing_parts = []
    for fold in FOLDS:
        _, miss = verify_missing_fold(args, fold, missing_rows, missing_x, out, snap)
        missing_parts.append(miss)
    miss = pd.concat(missing_parts, ignore_index=True)
    expected_miss = np.concatenate([
        missing_rows.MVT_ID_mvt.iloc[np.flatnonzero(missing.masks(
            missing_rows, missing_x, FOLDS[f])["direct_test"])].to_numpy()
        for f in FOLDS])
    if not np.array_equal(miss.MVT_ID_mvt.to_numpy(), expected_miss):
        raise ValueError("Saved missing-clock fold prediction gate/order differs")
    direct = align(reference, miss, "direct_candidate")
    del missing_rows, missing_x
    gc.collect()
    p.output_dir = out / "arrival"
    arrival_rows, arrival_x = arrival._load_all(p, False)
    ensure_base_rows(arrival_rows, reference)
    arrival_parts = []
    for fold in FOLDS:
        _, part = verify_arrival_fold(args, fold, arrival_rows, arrival_x, out, snap)
        arrival_parts.append(part)
    arr = pd.concat(arrival_parts, ignore_index=True)
    expected_arr = np.concatenate([
        arrival_rows.MVT_ID_mvt.iloc[split_arrival(arrival_rows, f)[2]].to_numpy()
        for f in FOLDS])
    if not np.array_equal(arr.MVT_ID_mvt.to_numpy(), expected_arr):
        raise ValueError("Saved ARR fold prediction gate/order differs")
    raw = align(reference, arr, "arrival_direct_expert")
    del arrival_rows, arrival_x
    gc.collect()
    value, rebuilt = compose_validation(reference, deep, direct, raw)
    report = read_json(out / "validation.json")
    saved = pd.read_parquet(out / "validation_predictions.parquet")
    if (report.get("status") != "passed" or report.get("selected_stage") != value["selected_stage"]
            or report.get("fixed_weights") != value["fixed_weights"]
            or report.get("folds") != value["folds"]
            or report.get("gate_rows") != value["gate_rows"]
            or report.get("protocol_sha256") != sha256(out / "protocol.json")
            or report.get("prepared_receipt_sha256") != sha256(out / "prepared.json")
            or report.get("source_input_sha256") != snap
            or report.get("ordered_ids_sha256") != digest_ids(rebuilt.MVT_ID_mvt)
            or report.get("validation_predictions_sha256") != sha256(
                out / "validation_predictions.parquet")
            or report.get("missing_fold_receipts_sha256") != {
                f: sha256(missing_fold_paths(out, f)["receipt"]) for f in FOLDS}
            or report.get("arrival_fold_receipts_sha256") != {
                f: sha256(arrival_fold_paths(out, f)["receipt"]) for f in FOLDS}
            or not saved.equals(rebuilt)):
        raise ValueError("Saved v5 validation receipt or fixed-policy replay differs")
    assert_snapshot(args, snap)
    return report, saved


def fit_missing_final(args: argparse.Namespace) -> dict:
    root, out, _, parents, _, snap = real_stage(args)
    model_path = out / "missing/final_ordinary_direct.cbm"
    receipt_path = out / "missing/final_receipt.json"
    if model_path.exists() or receipt_path.exists():
        raise FileExistsError("Final missing-clock model exists")
    import missing_catboost as missing
    from catboost import CatBoostRegressor, Pool
    p = adapter_args(parents, out / "missing")
    require_validation(args)
    rows, features = missing.load_inputs(p)
    y = rows.target.to_numpy(dtype=float)
    proxy = rows.proxy.to_numpy(dtype=float)
    no_nm = (features.AOBT_3_flt_missing.to_numpy(dtype=bool)
             & features.LOBT_flt_missing.to_numpy(dtype=bool))
    eligible = no_nm & ~np.isfinite(proxy) & np.isfinite(y) & (y >= 0) & (y <= 7200)
    train = np.flatnonzero(eligible)
    names, cats = missing.columns(features, long=False)
    selected = features[names]
    best = [read_json(missing_fold_paths(out, f)["receipt"])["best_iteration"] + 1
            for f in FOLDS]
    iterations = int(np.median(best))
    model = CatBoostRegressor(**missing.params(p, "regression", iterations, 6))
    model.fit(Pool(selected.iloc[train], label=y[train], cat_features=cats))
    if int(model.tree_count_) != iterations:
        raise ValueError("Full missing-clock tree count differs from original-fold median")
    probe = train[:min(256, len(train))]
    in_memory = model.predict(selected.iloc[probe], thread_count=4)
    assert_snapshot(args, snap)
    exclusive_saved_model(root, model_path, model)
    saved = verify_missing_model(model_path, selected,
                                 expected_trees=iterations)
    replay = saved.predict(selected.iloc[probe], thread_count=4)
    if not np.allclose(in_memory, replay, rtol=1e-11, atol=1e-8):
        raise ValueError("Saved full missing-clock model probe differs")
    receipt = {"status": "complete", "protocol_sha256": sha256(out / "protocol.json"),
               "source_input_sha256": snap,
               "validation_sha256": sha256(out / "validation.json"),
               "fold_receipts_sha256": {f: sha256(missing_fold_paths(out, f)["receipt"])
                                        for f in FOLDS},
               "fold_best_plus_one": best, "iterations": iterations,
               "actual_trees": int(saved.tree_count_), "training_rows": len(train),
               "training_ids_sha256": digest_ids(rows.MVT_ID_mvt.iloc[train]),
               "training_target_sha256": digest_float(y[train]),
               "probe_ids_sha256": digest_ids(rows.MVT_ID_mvt.iloc[probe]),
               "probe_prediction_sha256": digest_float(replay),
               "schema": schema(selected), "model_sha256": sha256(model_path)}
    assert_snapshot(args, snap)
    exclusive_json(root, receipt_path, receipt)
    return receipt


def require_missing_final(args: argparse.Namespace, rows: pd.DataFrame,
                          features: pd.DataFrame, out: Path, snapshot: dict) -> Any:
    import missing_catboost as missing
    model_path = out / "missing/final_ordinary_direct.cbm"
    receipt_path = out / "missing/final_receipt.json"
    receipt = read_json(receipt_path)
    y = rows.target.to_numpy(dtype=float)
    proxy = rows.proxy.to_numpy(dtype=float)
    no_nm = (features.AOBT_3_flt_missing.to_numpy(dtype=bool)
             & features.LOBT_flt_missing.to_numpy(dtype=bool))
    train = np.flatnonzero(no_nm & ~np.isfinite(proxy) & np.isfinite(y)
                           & (y >= 0) & (y <= 7200))
    probe = train[:min(256, len(train))]
    names, _ = missing.columns(features, long=False)
    selected = features[names]
    best = [read_json(missing_fold_paths(out, f)["receipt"])["best_iteration"] + 1
            for f in FOLDS]
    iterations = int(np.median(best))
    if (receipt.get("status") != "complete"
            or receipt.get("protocol_sha256") != sha256(out / "protocol.json")
            or receipt.get("source_input_sha256") != snapshot
            or receipt.get("validation_sha256") != sha256(out / "validation.json")
            or receipt.get("fold_receipts_sha256") != {
                f: sha256(missing_fold_paths(out, f)["receipt"]) for f in FOLDS}
            or receipt.get("fold_best_plus_one") != best
            or receipt.get("iterations") != iterations
            or receipt.get("actual_trees") != iterations
            or receipt.get("training_rows") != len(train)
            or receipt.get("training_ids_sha256") != digest_ids(rows.MVT_ID_mvt.iloc[train])
            or receipt.get("training_target_sha256") != digest_float(y[train])
            or receipt.get("probe_ids_sha256") != digest_ids(rows.MVT_ID_mvt.iloc[probe])
            or receipt.get("schema") != schema(selected)
            or receipt.get("model_sha256") != sha256(model_path)):
        raise ValueError("Full missing-clock model receipt differs")
    model = verify_missing_model(model_path, selected, expected_trees=iterations)
    if receipt.get("probe_prediction_sha256") != digest_float(
            model.predict(selected.iloc[probe], thread_count=4)):
        raise ValueError("Full missing-clock saved-model probe differs")
    return model


def fit_arrival_final(args: argparse.Namespace) -> dict:
    root, out, _, parents, _, snap = real_stage(args)
    require_arrival_cache(args, out, parents, snap, "training")
    model_path = out / "arrival/final.txt"
    receipt_path = out / "arrival/final_receipt.json"
    if model_path.exists() or receipt_path.exists():
        raise FileExistsError("Final ARR model exists")
    import arrival_residual_expert as arrival
    import lightgbm as lgb
    p = adapter_args(parents, out / "arrival")
    require_validation(args)
    rows, features = arrival._load_all(p, False)
    y = rows.target.to_numpy(dtype=float)
    proxy = rows.proxy.to_numpy(dtype=float)
    core = (np.isfinite(y) & (y >= 0) & (y <= 86400)
            & np.isfinite(proxy) & (proxy >= 0) & (proxy <= 7200))
    train = np.flatnonzero(core)
    rounds = int(np.median([read_json(arrival_fold_paths(out, f)["receipt"])["best_iteration"]
                            for f in FOLDS]))
    cats = [name for name in features if isinstance(features[name].dtype, pd.CategoricalDtype)]
    train_set = lgb.Dataset(features.iloc[train], label=(y - proxy)[train],
                            categorical_feature=cats, free_raw_data=True)
    model = lgb.train(arrival._params(4), train_set, num_boost_round=rounds,
                      callbacks=[lgb.log_evaluation(100)])
    if model.num_trees() != rounds:
        raise ValueError("Full ARR model tree count differs from original-fold median")
    probe = train[:min(256, len(train))]
    in_memory = model.predict(features.iloc[probe], num_threads=4)
    assert_snapshot(args, snap)
    exclusive_saved_model(root, model_path, model)
    saved = verify_arrival_model(model_path, features, expected_trees=rounds)
    replay = saved.predict(features.iloc[probe], num_threads=4)
    if not np.allclose(in_memory, replay, rtol=1e-11, atol=1e-8):
        raise ValueError("Saved full ARR model probe differs")
    receipt = {"status": "complete", "protocol_sha256": sha256(out / "protocol.json"),
               "source_input_sha256": snap,
               "training_arrival_features_sha256": sha256(
                   out / "arrival/training_arrival_features.parquet"),
               "validation_sha256": sha256(out / "validation.json"),
               "fold_receipts_sha256": {f: sha256(arrival_fold_paths(out, f)["receipt"])
                                        for f in FOLDS},
               "iterations": rounds, "actual_trees": saved.num_trees(),
               "params": arrival._params(4),
               "training_rows": len(train),
               "training_ids_sha256": digest_ids(rows.MVT_ID_mvt.iloc[train]),
               "training_target_sha256": digest_float(y[train]),
               "training_residual_label_sha256": digest_float((y - proxy)[train]),
               "probe_ids_sha256": digest_ids(rows.MVT_ID_mvt.iloc[probe]),
               "probe_prediction_sha256": digest_float(replay),
               "schema": schema(features), "model_sha256": sha256(model_path)}
    assert_snapshot(args, snap)
    exclusive_json(root, receipt_path, receipt)
    del train_set, model, saved, features, rows
    gc.collect()
    return receipt


def require_arrival_final(args: argparse.Namespace, rows: pd.DataFrame,
                          features: pd.DataFrame, out: Path,
                          snapshot: dict) -> Any:
    import arrival_residual_expert as arrival
    model_path = out / "arrival/final.txt"
    receipt = read_json(out / "arrival/final_receipt.json")
    y = rows.target.to_numpy(dtype=float)
    proxy = rows.proxy.to_numpy(dtype=float)
    train = np.flatnonzero(np.isfinite(y) & (y >= 0) & (y <= 86400)
                           & np.isfinite(proxy) & (proxy >= 0) & (proxy <= 7200))
    probe = train[:min(256, len(train))]
    rounds = int(np.median([read_json(arrival_fold_paths(out, f)["receipt"])["best_iteration"]
                            for f in FOLDS]))
    if (receipt.get("status") != "complete"
            or receipt.get("protocol_sha256") != sha256(out / "protocol.json")
            or receipt.get("source_input_sha256") != snapshot
            or receipt.get("training_arrival_features_sha256") != sha256(
                out / "arrival/training_arrival_features.parquet")
            or receipt.get("validation_sha256") != sha256(out / "validation.json")
            or receipt.get("fold_receipts_sha256") != {
                f: sha256(arrival_fold_paths(out, f)["receipt"]) for f in FOLDS}
            or receipt.get("iterations") != rounds
            or receipt.get("actual_trees") != rounds
            or receipt.get("params") != arrival._params(4)
            or receipt.get("training_rows") != len(train)
            or receipt.get("training_ids_sha256") != digest_ids(rows.MVT_ID_mvt.iloc[train])
            or receipt.get("training_target_sha256") != digest_float(y[train])
            or receipt.get("training_residual_label_sha256") != digest_float((y - proxy)[train])
            or receipt.get("probe_ids_sha256") != digest_ids(rows.MVT_ID_mvt.iloc[probe])
            or receipt.get("schema") != schema(features)
            or receipt.get("model_sha256") != sha256(model_path)):
        raise ValueError("Full ARR model receipt differs")
    model = verify_arrival_model(model_path, features, expected_trees=rounds)
    if receipt.get("probe_prediction_sha256") != digest_float(
            model.predict(features.iloc[probe], num_threads=4)):
        raise ValueError("Full ARR saved-model probe differs")
    return model


def ranking_seal_value(out: Path, snapshot: dict) -> dict:
    return {"status": "sealed_before_v5_ranking_values",
            "protocol_sha256": sha256(out / "protocol.json"),
            "prepared_receipt_sha256": sha256(out / "prepared.json"),
            "source_input_sha256": snapshot,
            "validation_sha256": sha256(out / "validation.json"),
            "validation_predictions_sha256": sha256(out / "validation_predictions.parquet"),
            "missing_model_sha256": sha256(out / "missing/final_ordinary_direct.cbm"),
            "missing_final_receipt_sha256": sha256(out / "missing/final_receipt.json"),
            "missing_fold_sha256": {fold: {
                key: sha256(path) for key, path in missing_fold_paths(out, fold).items()}
                for fold in FOLDS},
            "arrival_model_sha256": sha256(out / "arrival/final.txt"),
            "arrival_final_receipt_sha256": sha256(out / "arrival/final_receipt.json"),
            "arrival_fold_sha256": {fold: {
                key: sha256(path) for key, path in arrival_fold_paths(out, fold).items()}
                for fold in FOLDS},
            "training_arrival_cache_sha256": sha256(
                out / "arrival/training_arrival_features.parquet"),
            "training_arrival_cache_receipt_sha256": sha256(
                out / "arrival/training_features_receipt.json"),
            "v4_timestamp_ranking_sha256": snapshot["v4_output_sha256"]["deep_ranking_prediction"],
            "ranking_raw_sha256": snapshot["raw_and_cache_sha256"]["raw_ranking"],
            "template_sha256": snapshot["raw_and_cache_sha256"]["submission_template"],
            "ranking_rows_sha256": snapshot["raw_and_cache_sha256"]["baseline_ranking_rows"],
            "ranking_features_sha256": snapshot["raw_and_cache_sha256"]["baseline_ranking_features"]}


def ranking_seal(args: argparse.Namespace) -> dict:
    root, out, v4out, parents, parent_manifest, snap = real_stage(args)
    target = out / "ranking_inputs.json"
    if target.exists():
        raise FileExistsError("V5 ranking input seal already exists")
    require_validation(args)
    # Full models are replayed on fixed training probes before ranking reads.
    import missing_catboost as missing
    import arrival_residual_expert as arrival
    p = adapter_args(parents, out / "missing")
    miss_rows, miss_x = missing.load_inputs(p)
    require_missing_final(args, miss_rows, miss_x, out, snap)
    del miss_rows, miss_x
    gc.collect()
    p.output_dir = out / "arrival"
    arr_rows, arr_x = arrival._load_all(p, False)
    require_arrival_final(args, arr_rows, arr_x, out, snap)
    del arr_rows, arr_x
    gc.collect()
    # Independently rebuilt parent ranking bytes also replay the fixed v4 formula.
    verify_v4_ranking(v4out, parents, parent_manifest)
    value = ranking_seal_value(out, snap)
    assert_snapshot(args, snap)
    exclusive_json(root, target, value)
    return value


def require_ranking_seal(args: argparse.Namespace, root: Path,
                         out: Path, snapshot: dict) -> dict:
    expected = ranking_seal_value(out, snapshot)
    saved = read_json(out / "ranking_inputs.json")
    if saved != expected:
        raise ValueError("V5 ranking input seal, source or model changed")
    return saved


def rank_schema_compatible(training: dict, ranking: dict) -> bool:
    a = training["columns"]
    b = ranking["columns"]
    return (a == b
            and [v["name"] for v in a if v["name"] in training["category_vocabulary_sha256"]]
                == [v["name"] for v in b if v["name"] in ranking["category_vocabulary_sha256"]])


def predict(args: argparse.Namespace) -> dict:
    root, out, v4out, parents, parent_manifest, snap = real_stage(args)
    require_ranking_seal(args, root, out, snap)
    require_arrival_cache(args, out, parents, snap, "ranking")
    outputs = {"expert": out / "ranking_experts.parquet",
               "prediction": out / "predictions.parquet",
               "manifest": out / "ranking_manifest.json"}
    if any(path.exists() for path in outputs.values()):
        raise FileExistsError("V5 ranking output exists; no implicit resume")
    import missing_catboost as missing
    import arrival_residual_expert as arrival
    p = adapter_args(parents, out / "missing")
    miss_train_rows, miss_train_x = missing.load_inputs(p)
    miss_model = require_missing_final(args, miss_train_rows, miss_train_x, out, snap)
    miss_names, _ = missing.columns(miss_train_x, long=False)
    miss_training_schema = schema(miss_train_x[miss_names])
    del miss_train_rows, miss_train_x
    gc.collect()
    p.output_dir = out / "arrival"
    arr_train_rows, arr_train_x = arrival._load_all(p, False)
    arr_model = require_arrival_final(args, arr_train_rows, arr_train_x, out, snap)
    arr_training_schema = schema(arr_train_x)
    del arr_train_rows, arr_train_x
    gc.collect()
    template, _ = v4.verify_v4_rank(parents, parent_manifest)
    parent_deep = pd.read_parquet(v4out / "deep/predictions.parquet",
                                  columns=["MVT_ID_mvt", "TAXITIME_SEC_mvt"])
    if (not parent_deep.MVT_ID_mvt.equals(template.MVT_ID_mvt)
            or len(template) != EXPECTED_RANKING):
        raise ValueError("Timestamp ranking parent/template IDs differ")
    old = parent_deep.TAXITIME_SEC_mvt.to_numpy(dtype=float)
    if not np.isfinite(old).all() or np.any(old < 0):
        raise ValueError("Timestamp ranking parent invalid")
    p.output_dir = out / "missing"
    miss_rows, miss_x = missing.load_inputs(p, ranking=True)
    if not miss_rows.MVT_ID_mvt.equals(template.MVT_ID_mvt):
        raise ValueError("Missing-clock ranking covariate IDs differ")
    miss_names, _ = missing.columns(miss_x, long=False)
    if not rank_schema_compatible(miss_training_schema, schema(miss_x[miss_names])):
        raise ValueError("Missing-clock training/ranking feature order or types differ")
    proxy = miss_rows.proxy.to_numpy(dtype=float)
    valid = np.isfinite(proxy) & (proxy >= 0) & (proxy <= 7200)
    if int(valid.sum()) != EXPECTED_RANK_VALID:
        raise ValueError("Own-proxy valid ranking gate changed")
    no_nm = (miss_x.AOBT_3_flt_missing.to_numpy(dtype=bool)
             & miss_x.LOBT_flt_missing.to_numpy(dtype=bool))
    miss_gate = no_nm & ~np.isfinite(proxy) & ~miss_rows.airport.eq("LIRF").to_numpy(dtype=bool)
    if np.any(miss_gate & valid):
        raise ValueError("Missing direct gate overlaps timestamp/ARR gate")
    miss_raw = np.full(len(template), np.nan, dtype=float)
    miss_raw[miss_gate] = miss_model.predict(miss_x.loc[miss_gate, miss_names], thread_count=4)
    if not np.isfinite(miss_raw[miss_gate]).all():
        raise ValueError("Missing direct ranking output nonfinite")
    del miss_rows, miss_x, miss_model
    gc.collect()
    p.output_dir = out / "arrival"
    arr_rows, arr_x = arrival._load_all(p, True)
    if (not arr_rows.MVT_ID_mvt.equals(template.MVT_ID_mvt)
            or not rank_schema_compatible(arr_training_schema, schema(arr_x))):
        raise ValueError("ARR training/ranking schema or movement order differs")
    if not np.array_equal(valid, np.isfinite(arr_rows.proxy.to_numpy(dtype=float))
                          & (arr_rows.proxy.to_numpy(dtype=float) >= 0)
                          & (arr_rows.proxy.to_numpy(dtype=float) <= 7200)):
        raise ValueError("ARR and missing own-proxy ranking gates differ")
    arrival_raw = np.full(len(template), np.nan, dtype=float)
    arrival_raw[valid] = proxy[valid] + arr_model.predict(
        arr_x.loc[valid], num_threads=4)
    if not np.isfinite(arrival_raw[valid]).all():
        raise ValueError("ARR direct ranking output nonfinite")
    del arr_rows, arr_x, arr_model
    gc.collect()
    pred = old.copy()
    pred[miss_gate] = np.maximum(.5 * pred[miss_gate] + .5 * miss_raw[miss_gate], 0)
    pred[valid] = np.maximum(pred[valid] + ARRIVAL_WEIGHT *
                             (arrival_raw[valid] - pred[valid]), 0)
    if (not np.isfinite(pred).all() or np.any(pred < 0)
            or not np.array_equal(pred[~(valid | miss_gate)], old[~(valid | miss_gate)])):
        raise ValueError("V5 ranking prediction invalid or changed outside fixed gates")
    expert = pd.DataFrame({"MVT_ID_mvt": template.MVT_ID_mvt,
                           "missing_direct": miss_raw, "arrival_direct": arrival_raw,
                           "missing_gate": miss_gate, "valid_aobt_gate": valid})
    result = pd.DataFrame({"MVT_ID_mvt": template.MVT_ID_mvt,
                           "TAXITIME_SEC_mvt": pred})
    assert_snapshot(args, snap)
    require_ranking_seal(args, root, out, snap)
    exclusive_parquet(root, outputs["expert"], expert)
    exclusive_parquet(root, outputs["prediction"], result)
    expert_saved = pd.read_parquet(outputs["expert"])
    saved = pd.read_parquet(outputs["prediction"])
    if (not saved.equals(result) or not expert_saved.equals(expert)
            or not saved.MVT_ID_mvt.equals(template.MVT_ID_mvt)):
        raise ValueError("V5 immutable ranking Parquet readback differs")
    manifest = {"status": "complete", "selected_stage": "deep_missing_clean_arrival",
                "source_input_sha256": snap,
                "protocol_sha256": sha256(out / "protocol.json"),
                "ranking_seal_sha256": sha256(out / "ranking_inputs.json"),
                "validation_sha256": sha256(out / "validation.json"),
                "missing_final_receipt_sha256": sha256(out / "missing/final_receipt.json"),
                "arrival_final_receipt_sha256": sha256(out / "arrival/final_receipt.json"),
                "arrival_ranking_features_receipt_sha256": sha256(
                    out / "arrival/ranking_features_receipt.json"),
                "expert_sha256": sha256(outputs["expert"]),
                "predictions_sha256": sha256(outputs["prediction"]),
                "ordered_ids_sha256": digest_ids(template.MVT_ID_mvt),
                "ranking_rows": len(template), "valid_proxy_rows": int(valid.sum()),
                "missing_direct_rows": int(miss_gate.sum()),
                "outside_gate_unchanged_rows": int((~(valid | miss_gate)).sum()),
                "fixed_weights": {"deep": .5, "missing_direct": .5, "clean_arrival": .25},
                "legacy_arrival_included": False,
                "finite_nonnegative_and_template_order_verified": True,
                "no_upload_performed": True}
    assert_snapshot(args, snap)
    require_ranking_seal(args, root, out, snap)
    exclusive_json(root, outputs["manifest"], manifest)
    return manifest


def plan() -> dict:
    """No private values, model bytes or upstream output metadata read."""
    return {"status": "prospective_only", "spec": str(SPEC),
            "run_root": "operator-provided, disjoint from source repository",
            "upstream": "replica_v4_timestamp.py complete v3/GPU/source/timestamp genesis and replay",
            "real_modes": ["prepare", "build-arrival-train", "fit-missing-fold",
                           "fit-arrival-fold", "evaluate", "fit-missing-final",
                           "fit-arrival-final", "ranking-seal", "build-arrival-rank",
                           "predict"],
            "original_model_byte_substitution": False,
            "real_values_or_models_read": False}


def synthetic_self_test() -> dict:
    root = pd.DataFrame({"MVT_ID_mvt": [1, 2, 3, 4],
                         "target": [20., 30., 40., 50.],
                         "fold": ["seasonal_jan_jul", "seasonal_jan_jul",
                                  "forward_nov_dec", "forward_nov_dec"],
                         "a_valid": [True, False, True, False],
                         "airport": ["EGLL", "EGLL", "EHAM", "EHAM"],
                         "month": [1, 7, 11, 12],
                         "MVT_TIME_UTC_mvt": pd.to_datetime([
                             "2025-01-01", "2025-07-01", "2025-11-01", "2025-12-01"], utc=True),
                         "selected": [10., 20., 30., 40.]})
    direct = np.array([np.nan, 40., np.nan, 60.])
    arrival = np.array([50., np.nan, 70., np.nan])
    deep = np.array([15., 20., 35., 40.])
    report, frame = compose_validation(root, deep, direct, arrival)
    if (report["selected_stage"] != "deep_missing_clean_arrival"
            or not np.allclose(frame.selected.to_numpy(), [23.75, 30., 43.75, 50.])):
        raise AssertionError("Fixed disjoint v5 routing differs")
    invalid_direct = direct.copy()
    invalid_direct[0] = 30.
    try:
        compose_validation(root, deep, invalid_direct, arrival)
    except ValueError:
        pass
    else:
        raise AssertionError("Overlapping missing-clock gate accepted")
    try:
        v4.strict_root(HERE)
    except ValueError:
        pass
    else:
        raise AssertionError("Source-tree alias accepted as isolated run root")
    current = source_snapshot()
    if current["replica_v4_timestamp.py"] != PINNED_SOURCES["replica_v4_timestamp.py"]:
        raise AssertionError("Clean v4 producer source lineage drifted")
    return {"status": "passed", "fixed_disjoint_blend": "passed",
            "source_tree_alias": "rejected", "pinned_original_sources": len(PINNED_SOURCES),
            "real_values_or_models_read": False, "models_fitted": 0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("plan", "self-test", "prepare",
                                           "build-arrival-train", "fit-missing-fold",
                                           "fit-arrival-fold", "evaluate",
                                           "fit-missing-final", "fit-arrival-final",
                                           "ranking-seal", "build-arrival-rank", "predict"),
                        default="plan")
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--fold", choices=tuple(FOLDS))
    parser.add_argument("--published-source-sha256")
    parser.add_argument("--published-spec-sha256")
    parser.add_argument("--published-v4-sha256")
    args = parser.parse_args()
    if args.mode == "plan":
        value = plan()
    elif args.mode == "self-test":
        value = synthetic_self_test()
    else:
        if args.run_root is None:
            parser.error("Real modes require --run-root")
        if args.mode in ("fit-missing-fold", "fit-arrival-fold") and args.fold is None:
            parser.error("Fold fit requires --fold")
        actions = {"prepare": prepare,
                   "build-arrival-train": lambda a: build_arrival_features(a, "training"),
                   "fit-missing-fold": fit_missing_fold,
                   "fit-arrival-fold": fit_arrival_fold,
                   "evaluate": evaluate,
                   "fit-missing-final": fit_missing_final,
                   "fit-arrival-final": fit_arrival_final,
                   "ranking-seal": ranking_seal,
                   "build-arrival-rank": lambda a: build_arrival_features(a, "ranking"),
                   "predict": predict}
        value = actions[args.mode](args)
    print(json.dumps(value, indent=2, allow_nan=False, default=str))


if __name__ == "__main__":
    main()
