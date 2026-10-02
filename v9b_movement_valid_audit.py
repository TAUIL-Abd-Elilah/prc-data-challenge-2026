"""Prospective valid-AOBT audit of the frozen movement-only architecture.

Reuse the movement fold models. Only after both fold gates pass, fit a paired
April/October model if the original movement route did not make one. A final
all-ordinary model is similarly conditional on the complete local audit.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

import deep_arrival_expert as arrival
import deep_timestamp_expert as deep
import movement_only_expert as movement


FOLDS = {"seasonal_jan_jul": (1, 7), "forward_nov_dec": (11, 12)}
WEIGHTS = (0.0, 0.1, 0.25, 0.5)
FRESH_MONTHS = (4, 10)
EXPECTED_OOF_ROWS = 672428


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def assert_ids(actual: pd.Series, expected: pd.Series, label: str) -> None:
    a, e = pd.Index(actual), pd.Index(expected)
    if (a.has_duplicates or e.has_duplicates or a.isna().any() or e.isna().any()
            or len(a) != len(e) or not a.isin(e).all() or not e.isin(a).all()):
        raise ValueError(f"{label}: exact unique ID coverage failed")


def protocol(args: argparse.Namespace) -> dict:
    movement_manifest = json.loads((args.movement_dir / "features_manifest.json")
                                   .read_text(encoding="utf-8"))
    if (movement_manifest["arrival_cache_sha256"] != sha256(args.arrival_cache)
            or movement_manifest["reference_sha256"] != sha256(args.v5_oof)):
        raise ValueError("Movement prepared feature sources differ from their frozen hashes")
    value = {
        "purpose": "Separate prospective v9b valid-AOBT route, preserving the frozen v9 history",
        "training": "Reuse saved movement-only complement-fold models without changes; conditional fresh and full refits use exactly the frozen movement architecture",
        "reference": "Locally promoted clipped v7 candidate on all finite 2025 OOF rows",
        "source_sha256": {
            "movement_protocol": sha256(args.movement_dir / "protocol.json"),
            "frozen_v9_protocol": sha256(args.v9_protocol),
            "movement_features_manifest": sha256(args.movement_dir / "features_manifest.json"),
            "movement_row_ids": sha256(args.movement_dir / "row_ids.parquet"),
            "v7_protocol": sha256(args.v7_dir / "protocol.json"),
            "baseline_training_rows": sha256(args.cache_dir / "training_rows.parquet"),
        },
        "prediction_gate": "Held-out month, all finite target and own AOBT proxy in [0,7200] seconds",
        "prediction_policy": "Saved model best_round, direct prediction clipped >=0; all finite valid-AOBT IDs; no model retraining",
        "existing_fold_selection": {
            "months": {name: list(months) for name, months in FOLDS.items()},
            "weights": list(WEIGHTS),
            "formula": "clip(v7_candidate + weight*(movement_direct-v7_candidate), lower=0) on valid-AOBT only",
            "selection": "minimum January/July all-finite RMSE, tie to smaller weight",
            "forward": "same selected weight, no tuning",
            "gate": "nonzero weight, RMSE gain and UTC-day bootstrap 95% lower gain >0 separately in both folds",
        },
        "fresh_audit": {
            "months": list(FRESH_MONTHS),
            "movement_model": "reuse saved movement-only April/October refit; if absent and valid-AOBT existing-fold gates pass, fit exactly once in the separate v9b namespace with unchanged movement-only architecture",
            "fallback_fit_params": movement.model_params(3),
            "fallback_training": "same movement-only ordinary target [0,7200], same calendar-day modulo-11 early stop, with April/October excluded from fit and early stop",
            "comparator": "v7 fixed fresh blend of depth10 ARR prior and v7 fresh expert",
            "weight": "same January/July-selected movement weight",
            "gate": "exact valid-AOBT IDs, both April and October point RMSE gains, and pooled UTC-day bootstrap 95% lower gain >0",
        },
        "final_model": {
            "reuse": "if original no-NM movement route passes, reuse its saved full ordinary model",
            "fallback": "if original no-NM route fails but v9b passes, one v9b full ordinary fit with unchanged movement-only params and median existing-fold best rounds",
            "restart": "saved model requires matching model/fit-manifest hashes and feature schema",
        },
        "ranking": "blend only valid-AOBT rows of an exact-order v7 ranking reference; every invalid-AOBT value remains unchanged",
        "decision": "accept only after both existing folds and fresh audit pass; otherwise reject without retuning",
        "forbidden_uses": ["flight/movement IDs as predictor values", "departure BLOCK/TAXITIME as predictors", "official result or ranking labels"],
        "limitations": "Repeated 2025 comparisons; this estimates a model difference, not untouched 2026 accuracy.",
    }
    path = args.output_dir / "protocol.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise ValueError("Frozen v9 movement-valid protocol or source hash changed")
    else:
        write_json(path, value)
    return value


def load_movement_features(args: argparse.Namespace):
    source_args = argparse.Namespace(**vars(args))
    source_args.output_dir = args.movement_dir
    features, rows, manifest = movement.load_prepared(source_args)
    if (list(features) != manifest["features"]
            or rows.MVT_ID_mvt.isna().any()
            or rows.MVT_ID_mvt.duplicated().any()):
        raise ValueError("Prepared movement matrix differs from exact baseline rows")
    return rows, features, manifest


def original_fresh_model_info(args: argparse.Namespace) -> tuple[Path, int, Path, str] | None:
    original_model = args.movement_dir / "april_october_movement.txt"
    fit_path = args.movement_dir / "fresh_audit.json"
    # The original pipeline saves its movement model before its CatBoost
    # comparator finishes. An orphan model has no validated fit report.
    if not original_model.exists() or not fit_path.exists():
        return None
    fit = json.loads(fit_path.read_text(encoding="utf-8"))
    if (fit["heldout_months"] != list(FRESH_MONTHS)
            or fit["model_sha256"]["movement"] != sha256(original_model)
            or fit["features_manifest_sha256"] !=
               sha256(args.movement_dir / "features_manifest.json")):
        raise ValueError("Original movement April/October model differs")
    return (original_model, int(fit["movement_training"]["best_round"]),
            fit_path, "original_movement")


def fresh_model_info(args: argparse.Namespace) -> tuple[Path, int, Path, str]:
    own_model = args.output_dir / "april_october_movement.txt"
    fit_path = args.output_dir / "fresh_fit_manifest.json"
    if own_model.exists() or fit_path.exists():
        if not own_model.exists() or not fit_path.exists():
            raise ValueError("v9b April/October model and fit manifest are incomplete")
        fit = json.loads(fit_path.read_text(encoding="utf-8"))
        if (fit.get("status") != "complete"
                or fit["heldout_months"] != list(FRESH_MONTHS)
                or fit["model_sha256"] != sha256(own_model)
                or fit["features_manifest_sha256"] !=
                   sha256(args.movement_dir / "features_manifest.json")
                or fit["prepared_features_sha256"] !=
                   sha256(args.movement_dir / "features.parquet")
                or fit["arrival_cache_sha256"] != sha256(args.arrival_cache)
                or fit["v5_oof_sha256"] != sha256(args.v5_oof)
                or fit["existing_fold_validation_sha256"] !=
                   sha256(args.output_dir / "existing_fold_validation.json")
                or fit["protocol_sha256"] != sha256(args.output_dir / "protocol.json")):
            raise ValueError("v9b fallback April/October model provenance differs")
        return own_model, int(fit["best_round"]), fit_path, "v9b_fallback"
    original = original_fresh_model_info(args)
    if original is not None:
        return original
    raise FileNotFoundError("No validated April/October movement model; run v9b fit-fresh after its fold gates")


def predict(args: argparse.Namespace, name: str, months: tuple[int, int]) -> None:
    protocol(args)
    rows, features, manifest = load_movement_features(args)
    proxy = rows.proxy.to_numpy(dtype=float)
    y = rows.target.to_numpy(dtype=float)
    held = rows.month.isin(months).to_numpy()
    valid = held & np.isfinite(y) & np.isfinite(proxy) & (proxy >= 0) & (proxy <= 7200)
    if name == "fresh_apr_oct":
        model_path, rounds, fit_path, source = fresh_model_info(args)
    else:
        model_path = args.movement_dir / f"{name}.txt"
        fit_path = args.movement_dir / f"{name}_fit.json"
        fit = json.loads(fit_path.read_text(encoding="utf-8"))
        if (fit["heldout_months"] != list(months)
                or fit["features_manifest_sha256"] !=
                   sha256(args.movement_dir / "features_manifest.json")):
            raise ValueError("Saved movement fold or feature manifest differs")
        rounds = int(fit["best_round"])
        source = "original_movement"
    model = lgb.Booster(model_file=str(model_path))
    if (not 1 <= rounds <= model.num_trees()
            or model.feature_name() != manifest["features"]):
        raise ValueError("Saved movement model rounds or feature names differ")
    expert = np.maximum(model.predict(features.loc[valid],
                                      num_iteration=rounds, num_threads=3), 0)
    if len(expert) != int(valid.sum()) or not np.isfinite(expert).all():
        raise ValueError("Saved movement model lacks finite valid-AOBT coverage")
    output = pd.DataFrame({"MVT_ID_mvt": rows.loc[valid, "MVT_ID_mvt"].to_numpy(),
                           "target": y[valid],
                           "MVT_TIME_UTC_mvt": rows.loc[valid, "time"].to_numpy(),
                           "expert": expert})
    if output.MVT_ID_mvt.duplicated().any():
        raise ValueError("Saved movement valid-AOBT IDs repeat")
    path = args.output_dir / f"{name}_valid_oof.parquet"
    output.to_parquet(path, index=False)
    write_json(args.output_dir / f"{name}_prediction_manifest.json", {
        "months": list(months), "n_valid_finite": len(output),
        "model_source": source, "model_path": str(model_path),
        "model_sha256": sha256(model_path),
        "fit_report_sha256": sha256(fit_path),
        "features_manifest_sha256": sha256(args.movement_dir /
                                            "features_manifest.json"),
        "prepared_features_sha256": sha256(args.movement_dir /
                                            "features.parquet"),
        "arrival_cache_sha256": sha256(args.arrival_cache),
        "v5_oof_sha256": sha256(args.v5_oof),
        "output_sha256": sha256(path),
    })
    print(json.dumps({"scope": name, "n_valid_finite": len(output),
                      "output": str(path)}, indent=2))


def verify_prediction(args: argparse.Namespace, name: str,
                      expected_ids: pd.Series, months: tuple[int, int]) -> pd.DataFrame:
    path = args.output_dir / f"{name}_valid_oof.parquet"
    manifest = json.loads((args.output_dir /
                           f"{name}_prediction_manifest.json")
                          .read_text(encoding="utf-8"))
    source = manifest["model_source"]
    if name == "fresh_apr_oct":
        if source == "original_movement":
            model_path = args.movement_dir / "april_october_movement.txt"
            fit_path = args.movement_dir / "fresh_audit.json"
        elif source == "v9b_fallback":
            model_path = args.output_dir / "april_october_movement.txt"
            fit_path = args.output_dir / "fresh_fit_manifest.json"
        else:
            raise ValueError("Unknown April/October model source")
    else:
        if source != "original_movement":
            raise ValueError("Fold model must come from original movement fit")
        model_path = args.movement_dir / f"{name}.txt"
        fit_path = args.movement_dir / f"{name}_fit.json"
    if (manifest["months"] != list(months)
            or manifest["model_path"] != str(model_path)
            or manifest["output_sha256"] != sha256(path)
            or manifest["fit_report_sha256"] != sha256(fit_path)
            or manifest["features_manifest_sha256"] !=
               sha256(args.movement_dir / "features_manifest.json")
            or manifest["prepared_features_sha256"] !=
               sha256(args.movement_dir / "features.parquet")
            or manifest["arrival_cache_sha256"] != sha256(args.arrival_cache)
            or manifest["v5_oof_sha256"] != sha256(args.v5_oof)
            or manifest["model_sha256"] != sha256(model_path)):
        raise ValueError(f"{name} saved movement prediction inputs changed")
    frame = pd.read_parquet(path)
    if (list(frame) != ["MVT_ID_mvt", "target", "MVT_TIME_UTC_mvt", "expert"]
            or manifest["n_valid_finite"] != len(frame)
            or not np.isfinite(frame[["target", "expert"]]
                               .to_numpy(dtype=float)).all()):
        raise ValueError(f"{name} prediction schema or values invalid")
    assert_ids(frame.MVT_ID_mvt, expected_ids, f"{name} valid AOBT")
    return frame


def fold_audit(args: argparse.Namespace, v7: pd.DataFrame) -> tuple[dict,
                                                                      pd.DataFrame,
                                                                      float]:
    parts = []
    for name, months in FOLDS.items():
        expected = v7.loc[v7.fold.eq(name) & v7.a_valid, "MVT_ID_mvt"]
        expert = verify_prediction(args, name, expected, months)
        expert = expert.rename(columns={"target": "target_movement",
                                        "MVT_TIME_UTC_mvt": "time_movement"})
        part = v7.loc[v7.fold.eq(name)].merge(
            expert, on="MVT_ID_mvt", how="left", sort=False,
            validate="one_to_one")
        gate = part.a_valid.to_numpy(dtype=bool)
        if (not np.array_equal(part.expert.notna().to_numpy(), gate)
                or not np.allclose(part.loc[gate, "target"],
                                   part.loc[gate, "target_movement"],
                                   rtol=0, atol=1e-6)
                or not np.array_equal(
                    pd.to_datetime(part.loc[gate, "MVT_TIME_UTC_mvt"],
                                   utc=True).to_numpy(),
                    pd.to_datetime(part.loc[gate, "time_movement"],
                                   utc=True).to_numpy())):
            raise ValueError(f"{name} movement expert labels or times differ")
        parts.append(part)
    frame = pd.concat(parts, ignore_index=True)
    scores = {}
    for name in FOLDS:
        part = frame.loc[frame.fold.eq(name)]
        base = part.candidate.to_numpy(dtype=float)
        alternative = part.expert.fillna(part.candidate).to_numpy(dtype=float)
        y = part.target.to_numpy(dtype=float)
        scores[name] = {str(weight): deep.rmse(y, np.maximum(
            base + weight*(alternative-base), 0)) for weight in WEIGHTS}
    selected = min(WEIGHTS, key=lambda weight: (
        scores["seasonal_jan_jul"][str(weight)], weight))
    report = {}
    output_parts = []
    for name in FOLDS:
        part = frame.loc[frame.fold.eq(name)].copy()
        base = part.candidate.to_numpy(dtype=float)
        alternative = part.expert.fillna(part.candidate).to_numpy(dtype=float)
        selected_prediction = np.maximum(base + selected*(alternative-base), 0)
        if not np.array_equal(selected_prediction[~part.a_valid.to_numpy(dtype=bool)],
                              base[~part.a_valid.to_numpy(dtype=bool)]):
            raise ValueError("Movement routing changed invalid-AOBT rows")
        part["movement_valid_candidate"] = selected_prediction
        bootstrap = arrival.bootstrap(part, base, selected_prediction,
                                      seed=20261010)
        y = part.target.to_numpy(dtype=float)
        report[name] = {"n_all_finite": len(part),
                        "n_valid_aobt": int(part.a_valid.sum()),
                        "scores": scores[name],
                        "bootstrap": bootstrap,
                        "passed": (selected > 0
                                   and deep.rmse(y, selected_prediction)
                                   < deep.rmse(y, base)
                                   and bootstrap["gain_ci95_sec"][0] > 0)}
        output_parts.append(part)
    return report, pd.concat(output_parts, ignore_index=True), selected


def fresh_audit(args: argparse.Namespace, selected: float,
                v7_weight: float) -> tuple[dict, pd.DataFrame]:
    fresh_path = args.v7_dir / "fresh_audit_predictions.parquet"
    fresh_report_path = args.v7_dir / "fresh_audit.json"
    fresh_report = json.loads(fresh_report_path.read_text(encoding="utf-8"))
    if (not fresh_report.get("passed")
            or not fresh_report.get("coverage_verified")
            or fresh_report.get("months") != list(FRESH_MONTHS)
            or float(fresh_report.get("weight", -1)) != v7_weight
            or fresh_report.get("paired_predictions_sha256") != sha256(fresh_path)):
        raise ValueError("v7 fresh comparator report, weight or file hash differs")
    fresh = pd.read_parquet(fresh_path,
                            columns=["MVT_ID_mvt", "target", "MVT_TIME_UTC_mvt",
                                     "prior", "new", "candidate"])
    rows = pd.read_parquet(args.cache_dir / "training_rows.parquet",
                           columns=["MVT_ID_mvt", "target", "proxy", "month"])
    proxy = rows.proxy.to_numpy(dtype=float)
    y = rows.target.to_numpy(dtype=float)
    valid = (rows.month.isin(FRESH_MONTHS).to_numpy() & np.isfinite(y)
             & np.isfinite(proxy) & (proxy >= 0) & (proxy <= 7200))
    assert_ids(fresh.MVT_ID_mvt, rows.loc[valid, "MVT_ID_mvt"],
               "v7 fresh valid-AOBT reference")
    movement = verify_prediction(args, "fresh_apr_oct", fresh.MVT_ID_mvt,
                                 FRESH_MONTHS)
    movement.rename(columns={"target": "target_movement",
                             "MVT_TIME_UTC_mvt": "time_movement"}, inplace=True)
    frame = fresh.merge(movement, on="MVT_ID_mvt", how="left",
                        sort=False, validate="one_to_one")
    clipped_prior = np.maximum(frame.prior.to_numpy(dtype=float), 0)
    expected_v7 = np.maximum(clipped_prior + v7_weight *
                             (frame.new.to_numpy(dtype=float)-clipped_prior), 0)
    if (len(frame) != len(fresh)
            or not frame.MVT_ID_mvt.equals(fresh.MVT_ID_mvt)
            or not np.isfinite(frame[["target", "prior", "new", "candidate",
                                      "target_movement", "expert"]]
                               .to_numpy(dtype=float)).all()
            or not np.allclose(frame.target, frame.target_movement,
                               rtol=0, atol=1e-6)
            or not np.allclose(frame.candidate, expected_v7,
                               rtol=0, atol=1e-6)
            or not np.array_equal(
                pd.to_datetime(frame.MVT_TIME_UTC_mvt, utc=True).to_numpy(),
                pd.to_datetime(frame.time_movement, utc=True).to_numpy())):
        raise ValueError("Fresh movement prediction or fixed v7 comparator differs")
    base = frame.candidate.to_numpy(dtype=float)
    expert = frame.expert.to_numpy(dtype=float)
    combined = np.maximum(base + selected*(expert-base), 0)
    frame["movement_valid_candidate"] = combined
    bootstrap = arrival.bootstrap(frame, base, combined, seed=20261011)
    month = pd.to_datetime(frame.MVT_TIME_UTC_mvt, utc=True).dt.month
    month_scores = {str(m): {
        "n": int((month == m).sum()),
        "v7_rmse": deep.rmse(frame.target.to_numpy(dtype=float)[month == m],
                             base[month == m]),
        "candidate_rmse": deep.rmse(frame.target.to_numpy(dtype=float)[month == m],
                                     combined[month == m])} for m in FRESH_MONTHS}
    passed = (all(month_scores[str(m)]["candidate_rmse"]
                  < month_scores[str(m)]["v7_rmse"] for m in FRESH_MONTHS)
              and bootstrap["gain_ci95_sec"][0] > 0)
    return {"n": len(frame), "month_scores": month_scores,
            "bootstrap": bootstrap, "passed": passed}, frame


def load_v7_reference(args: argparse.Namespace) -> tuple[pd.DataFrame, dict, Path]:
    protocol(args)
    v7_report = json.loads((args.v7_dir / "validation.json")
                           .read_text(encoding="utf-8"))
    if not v7_report.get("promoted"):
        raise ValueError("Valid-AOBT routing requires locally promoted v7")
    v7_path = args.v7_dir / "validation_predictions.parquet"
    if v7_report["validation_predictions_sha256"] != sha256(v7_path):
        raise ValueError("v7 OOF differs from its local validation report")
    v7 = pd.read_parquet(v7_path,
                         columns=["MVT_ID_mvt", "target", "fold", "month",
                                  "MVT_TIME_UTC_mvt", "a_valid", "candidate"])
    reference = pd.read_parquet(args.v7_dir / "frozen_v6_oof_reference.parquet",
                                columns=["MVT_ID_mvt", "target", "fold",
                                         "a_valid"])
    assert_ids(v7.MVT_ID_mvt, reference.MVT_ID_mvt, "v7 all-finite OOF")
    if len(v7) != EXPECTED_OOF_ROWS or not np.isfinite(
            v7[["target", "candidate"]].to_numpy(dtype=float)).all():
        raise ValueError("v7 OOF is incomplete or nonfinite")
    aligned = v7.merge(reference, on="MVT_ID_mvt", how="left", sort=False,
                       validate="one_to_one", suffixes=("", "_reference"))
    if (not np.allclose(aligned.target, aligned.target_reference,
                        rtol=0, atol=1e-6)
            or not np.array_equal(aligned.fold.to_numpy(),
                                  aligned.fold_reference.to_numpy())
            or not np.array_equal(aligned.a_valid.to_numpy(dtype=bool),
                                  aligned.a_valid_reference.to_numpy(dtype=bool))):
        raise ValueError("v7 OOF labels, folds or valid-AOBT gate differ")
    return v7, v7_report, v7_path


def evaluate_folds(args: argparse.Namespace) -> None:
    v7, _, v7_path = load_v7_reference(args)
    fold_report, fold_rows, selected = fold_audit(args, v7)
    fold_path = args.output_dir / "validation_predictions.parquet"
    fold_rows[["MVT_ID_mvt", "target", "fold", "month", "MVT_TIME_UTC_mvt",
               "a_valid", "candidate", "expert",
               "movement_valid_candidate"]].to_parquet(fold_path, index=False)
    report = {
        "selected_weight": selected, "folds": fold_report,
        "existing_folds_passed": all(part["passed"] for part in fold_report.values()),
        "protocol_sha256": sha256(args.output_dir / "protocol.json"),
        "v7_validation_sha256": sha256(args.v7_dir / "validation.json"),
        "v7_oof_sha256": sha256(v7_path),
        "movement_fold_oof_sha256": {
            name: sha256(args.output_dir / f"{name}_valid_oof.parquet") for name in FOLDS},
        "validation_predictions_sha256": sha256(fold_path),
    }
    write_json(args.output_dir / "existing_fold_validation.json", report)
    print(json.dumps({"selected_weight": selected, "folds": fold_report,
                      "existing_folds_passed": report["existing_folds_passed"]}, indent=2))


def require_fold_gates(args: argparse.Namespace) -> dict:
    protocol(args)
    path = args.output_dir / "existing_fold_validation.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    if (not report.get("existing_folds_passed")
            or float(report.get("selected_weight", 0)) not in WEIGHTS[1:]
            or not all(report["folds"][name]["passed"] for name in FOLDS)
            or report["protocol_sha256"] != sha256(args.output_dir / "protocol.json")
            or report["v7_validation_sha256"] != sha256(args.v7_dir / "validation.json")
            or report["v7_oof_sha256"] !=
               sha256(args.v7_dir / "validation_predictions.parquet")
            or report["validation_predictions_sha256"] !=
               sha256(args.output_dir / "validation_predictions.parquet")
            or any(report["movement_fold_oof_sha256"][name] !=
                   sha256(args.output_dir / f"{name}_valid_oof.parquet")
                   for name in FOLDS)):
        raise ValueError("v9b existing-fold gates or source hashes failed")
    return report


def fit_fresh(args: argparse.Namespace) -> None:
    fold_report = require_fold_gates(args)
    own_model = args.output_dir / "april_october_movement.txt"
    own_manifest = args.output_dir / "fresh_fit_manifest.json"
    if own_model.exists() or own_manifest.exists():
        path, rounds, fit_path, source = fresh_model_info(args)
        print(json.dumps({"source": source, "model": str(path),
                          "rounds": rounds, "fit_report": str(fit_path)}, indent=2))
        return
    original = original_fresh_model_info(args)
    if original is not None:
        path, rounds, fit_path, source = original
        print(json.dumps({"source": source, "model": str(path),
                          "rounds": rounds, "fit_report": str(fit_path)}, indent=2))
        return
    movement.require_memory(args.min_free_gib)
    source_args = argparse.Namespace(**vars(args))
    source_args.output_dir = args.movement_dir
    features, rows, manifest = movement.load_prepared(source_args)
    model, fit = movement.fit_movement_model(
        source_args, features, rows, manifest["categorical"], FRESH_MONTHS)
    own_model.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(own_model))
    write_json(own_manifest, {
        "status": "complete", "heldout_months": list(FRESH_MONTHS),
        "best_round": int(fit["best_round"]), "fit": fit,
        "model_sha256": sha256(own_model),
        "features_manifest_sha256": sha256(args.movement_dir / "features_manifest.json"),
        "prepared_features_sha256": sha256(args.movement_dir / "features.parquet"),
        "arrival_cache_sha256": sha256(args.arrival_cache),
        "v5_oof_sha256": sha256(args.v5_oof),
        "existing_fold_validation_sha256": sha256(args.output_dir / "existing_fold_validation.json"),
        "protocol_sha256": sha256(args.output_dir / "protocol.json"),
        "model_params": movement.model_params(args.threads),
        "selected_weight": fold_report["selected_weight"],
    })
    print(json.dumps({"source": "v9b_fallback", "model": str(own_model),
                      "rounds": fit["best_round"]}, indent=2))


def audit(args: argparse.Namespace) -> None:
    fixed = protocol(args)
    fold_report = require_fold_gates(args)
    _, v7_report, v7_path = load_v7_reference(args)
    selected = float(fold_report["selected_weight"])
    fresh_report, fresh_rows = fresh_audit(args, selected,
                                          float(v7_report["selected_weight"]))
    fresh_path = args.output_dir / "fresh_audit_predictions.parquet"
    fresh_rows[["MVT_ID_mvt", "target", "MVT_TIME_UTC_mvt", "candidate",
                "expert", "movement_valid_candidate"]].to_parquet(
                    fresh_path, index=False)
    report = {"protocol": fixed, "selected_weight": selected,
              "folds": fold_report["folds"], "fresh_audit": fresh_report,
              "accepted": bool(fresh_report["passed"]),
              "source_sha256": {
                  "v7_validation": sha256(args.v7_dir / "validation.json"),
                  "v7_oof": sha256(v7_path),
                  "v7_fresh_report": sha256(args.v7_dir / "fresh_audit.json"),
                  "v7_fresh": sha256(args.v7_dir / "fresh_audit_predictions.parquet"),
                  "existing_fold_validation": sha256(args.output_dir / "existing_fold_validation.json"),
                  "movement_seasonal": sha256(args.output_dir /
                                               "seasonal_jan_jul_valid_oof.parquet"),
                  "movement_forward": sha256(args.output_dir /
                                              "forward_nov_dec_valid_oof.parquet"),
                  "movement_fresh": sha256(args.output_dir /
                                            "fresh_apr_oct_valid_oof.parquet")},
              "output_sha256": {"fold_oof": sha256(args.output_dir / "validation_predictions.parquet"),
                                "fresh_oof": sha256(fresh_path)}}
    write_json(args.output_dir / "audit.json", report)
    print(json.dumps({"selected_weight": selected, "folds": fold_report["folds"],
                      "fresh_audit": fresh_report,
                      "accepted": report["accepted"]}, indent=2))


def require_accepted_audit(args: argparse.Namespace) -> dict:
    fold = require_fold_gates(args)
    path = args.output_dir / "audit.json"
    audit_report = json.loads(path.read_text(encoding="utf-8"))
    fresh = audit_report["fresh_audit"]
    if (not audit_report.get("accepted") or not fresh.get("passed")
            or float(audit_report["selected_weight"]) != float(fold["selected_weight"])
            or fresh["bootstrap"]["gain_ci95_sec"][0] <= 0
            or not all(fresh["month_scores"][str(month)]["candidate_rmse"] <
                       fresh["month_scores"][str(month)]["v7_rmse"]
                       for month in FRESH_MONTHS)
            or audit_report["source_sha256"]["existing_fold_validation"] !=
               sha256(args.output_dir / "existing_fold_validation.json")
            or audit_report["source_sha256"]["v7_fresh_report"] !=
               sha256(args.v7_dir / "fresh_audit.json")
            or audit_report["source_sha256"]["v7_fresh"] !=
               sha256(args.v7_dir / "fresh_audit_predictions.parquet")
            or audit_report["output_sha256"]["fold_oof"] !=
               sha256(args.output_dir / "validation_predictions.parquet")
            or audit_report["output_sha256"]["fresh_oof"] !=
               sha256(args.output_dir / "fresh_audit_predictions.parquet")):
        raise ValueError("v9b complete local audit did not pass or inputs changed")
    return audit_report


def original_final_info(args: argparse.Namespace) -> tuple[Path, int, Path] | None:
    model_file = args.movement_dir / "final_movement_only.txt"
    report_file = args.movement_dir / "final_model.json"
    if not model_file.exists() and not report_file.exists():
        return None
    if not model_file.exists() or not report_file.exists():
        raise ValueError("Original movement final model/report are incomplete")
    movement.require_all_gates(args.movement_dir, args.cache_dir)
    report = json.loads(report_file.read_text(encoding="utf-8"))
    if (report["model_sha256"] != sha256(model_file)
            or report["features_manifest_sha256"] !=
               sha256(args.movement_dir / "features_manifest.json")):
        raise ValueError("Original movement final model provenance differs")
    return model_file, int(report["final_rounds"]), report_file


def original_route_passed(args: argparse.Namespace) -> bool:
    validation_path = args.movement_dir / "validation.json"
    fresh_path = args.movement_dir / "fresh_audit.json"
    reserved_path = args.movement_dir / "reserved_audit.json"
    if not all(path.exists() for path in (validation_path, fresh_path,
                                          reserved_path)):
        return False
    fresh = json.loads(fresh_path.read_text(encoding="utf-8"))
    reserved = json.loads(reserved_path.read_text(encoding="utf-8"))
    if not fresh.get("passed") or not reserved.get("passed"):
        return False
    # A completed passing report with damaged hashes or coverage is an error,
    # rather than a reason to silently fit a different full model.
    movement.require_all_gates(args.movement_dir, args.cache_dir)
    return True


def own_final_info(args: argparse.Namespace) -> tuple[Path, int, Path]:
    model_file = args.output_dir / "full_movement_valid.txt"
    report_file = args.output_dir / "full_fit_manifest.json"
    if not model_file.exists() or not report_file.exists():
        raise FileNotFoundError("v9b full model is absent; run fit-final after gates")
    report = json.loads(report_file.read_text(encoding="utf-8"))
    if (report.get("status") != "complete"
            or report["model_sha256"] != sha256(model_file)
            or report["audit_sha256"] != sha256(args.output_dir / "audit.json")
            or report["reserved_guard_sha256"] !=
               sha256(args.reserved_guard_dir / "v9b" / "guard_report.json")
            or report["protocol_sha256"] != sha256(args.output_dir / "protocol.json")
            or report["features_manifest_sha256"] !=
               sha256(args.movement_dir / "features_manifest.json")
            or report["prepared_features_sha256"] !=
               sha256(args.movement_dir / "features.parquet")
            or report["arrival_cache_sha256"] != sha256(args.arrival_cache)
            or report["v5_oof_sha256"] != sha256(args.v5_oof)):
        raise ValueError("v9b full model provenance differs")
    return model_file, int(report["final_rounds"]), report_file


def fit_final(args: argparse.Namespace) -> None:
    from reserved_valid_guard import require_route_guard

    audit_report = require_accepted_audit(args)
    require_route_guard(args.reserved_guard_dir, "v9b")
    original = original_final_info(args)
    if original:
        print(json.dumps({"model_source": "original_movement",
                          "model": str(original[0]), "rounds": original[1]}, indent=2))
        return
    if original_route_passed(args):
        raise ValueError("Original movement route passed; finish and reuse its full ordinary model")
    own_model = args.output_dir / "full_movement_valid.txt"
    own_manifest = args.output_dir / "full_fit_manifest.json"
    if own_model.exists() or own_manifest.exists():
        model_file, rounds, _ = own_final_info(args)
        print(json.dumps({"model_source": "v9b_full", "model": str(model_file),
                          "rounds": rounds}, indent=2))
        return
    movement.require_memory(args.min_free_gib)
    source_args = argparse.Namespace(**vars(args))
    source_args.output_dir = args.movement_dir
    features, rows, manifest = movement.load_prepared(source_args)
    fold_reports = {name: json.loads((args.movement_dir / f"{name}_fit.json")
                                     .read_text(encoding="utf-8")) for name in FOLDS}
    rounds = int(np.median([int(report["best_round"])
                            for report in fold_reports.values()]))
    if (not 1 <= rounds <= 1200
            or any(report["heldout_months"] != list(FOLDS[name])
                   or report["features_manifest_sha256"] !=
                   sha256(args.movement_dir / "features_manifest.json")
                   for name, report in fold_reports.items())):
        raise ValueError("Saved movement fold rounds or feature provenance differ")
    y = rows.target.to_numpy(dtype=np.float32)
    ordinary = np.isfinite(y) & (y >= 0) & (y <= 7200)
    if int(ordinary.sum()) < 1_000_000:
        raise ValueError("Too few ordinary all-2025 movement rows")
    train = lgb.Dataset(features.loc[ordinary], label=y[ordinary],
                        categorical_feature=manifest["categorical"],
                        free_raw_data=True)
    model = lgb.train(movement.model_params(args.threads), train,
                      num_boost_round=rounds)
    own_model.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(own_model))
    write_json(own_manifest, {
        "status": "complete", "model_sha256": sha256(own_model),
        "final_rounds": rounds, "training_rows": int(ordinary.sum()),
        "fold_best_rounds": {name: int(report["best_round"])
                              for name, report in fold_reports.items()},
        "fold_fit_sha256": {name: sha256(args.movement_dir / f"{name}_fit.json")
                              for name in FOLDS},
        "model_params": movement.model_params(args.threads),
        "audit_sha256": sha256(args.output_dir / "audit.json"),
        "reserved_guard_sha256": sha256(args.reserved_guard_dir / "v9b" /
                                          "guard_report.json"),
        "protocol_sha256": sha256(args.output_dir / "protocol.json"),
        "features_manifest_sha256": sha256(args.movement_dir / "features_manifest.json"),
        "prepared_features_sha256": sha256(args.movement_dir / "features.parquet"),
        "arrival_cache_sha256": sha256(args.arrival_cache),
        "v5_oof_sha256": sha256(args.v5_oof),
        "selected_weight": float(audit_report["selected_weight"]),
    })
    print(json.dumps({"model_source": "v9b_full", "model": str(own_model),
                      "rounds": rounds}, indent=2))


def final_predict(args: argparse.Namespace) -> None:
    from reserved_valid_guard import require_route_guard

    audit_report = require_accepted_audit(args)
    require_route_guard(args.reserved_guard_dir, "v9b")
    original = original_final_info(args)
    if original is None and original_route_passed(args):
        raise ValueError("Original movement route passed; use its pending full ordinary model")
    model_file, rounds, fit_path = original or own_final_info(args)
    source = "original_movement" if original else "v9b_full"
    manifest = json.loads((args.movement_dir / "features_manifest.json")
                          .read_text(encoding="utf-8"))
    if sha256(args.weather_file) != manifest["weather_file_sha256"]:
        raise ValueError("Weather source changed after movement feature preparation")
    ranking_inputs = movement.freeze_ranking_inputs(args)
    features, rows = movement.build_ranking_features(args, manifest)
    template = pd.read_parquet(args.data_dir / "submitting.parquet")
    reference_path = args.v7_dir / "predictions.parquet"
    v7_manifest_path = args.v7_dir / "manifest.json"
    v7_manifest = json.loads(v7_manifest_path.read_text(encoding="utf-8"))
    if v7_manifest["predictions_sha256"] != sha256(reference_path):
        raise ValueError("v7 ranking reference differs from accepted manifest")
    reference = pd.read_parquet(reference_path)
    required = ["MVT_ID_mvt", "TAXITIME_SEC_mvt"]
    if list(template) != required or list(reference) != required:
        raise ValueError("Submission template or v7 reference schema differs")
    ids = template.MVT_ID_mvt.to_numpy()
    if (len(rows) != len(ids) or pd.Index(ids).has_duplicates
            or not np.array_equal(rows.MVT_ID_mvt.to_numpy(), ids)
            or not np.array_equal(reference.MVT_ID_mvt.to_numpy(), ids)):
        raise ValueError("Ranking movement, template or v7 ID order differs")
    old = reference.TAXITIME_SEC_mvt.to_numpy(dtype=np.float64, copy=True)
    proxy = rows.proxy.to_numpy(dtype=float)
    valid = np.isfinite(proxy) & (proxy >= 0) & (proxy <= 7200)
    if (int(valid.sum()) != int(v7_manifest["valid_aobt"])
            or not np.isfinite(old).all() or np.any(old < 0)):
        raise ValueError("v7 ranking values or valid-AOBT coverage differ")
    model = lgb.Booster(model_file=str(model_file))
    if model.feature_name() != manifest["features"] or not 1 <= rounds <= model.num_trees():
        raise ValueError("Final movement model feature names or rounds differ")
    expert = np.maximum(model.predict(features.loc[valid], num_iteration=rounds,
                                      num_threads=args.threads), 0)
    if len(expert) != int(valid.sum()) or not np.isfinite(expert).all():
        raise ValueError("Movement valid-AOBT ranking expert lacks finite coverage")
    selected = float(audit_report["selected_weight"])
    output = old.copy()
    output[valid] = np.maximum(old[valid] + selected*(expert-old[valid]), 0)
    if (not np.array_equal(output[~valid], old[~valid])
            or not np.isfinite(output).all() or np.any(output < 0)):
        raise ValueError("v9b final predictions changed invalid rows or are nonfinite")
    expert_full = np.full(len(output), np.nan, dtype=np.float32)
    expert_full[valid] = expert.astype(np.float32)
    movement.verify_ranking_inputs(args)
    expert_path = args.output_dir / "ranking_expert.parquet"
    pd.DataFrame({"MVT_ID_mvt": ids, "expert": expert_full}).to_parquet(
        expert_path, index=False)
    output_path = args.output_dir / "predictions.parquet"
    pd.DataFrame({"MVT_ID_mvt": ids, "TAXITIME_SEC_mvt": output}).to_parquet(
        output_path, index=False)
    readback = pd.read_parquet(output_path)
    if (not np.array_equal(readback.MVT_ID_mvt.to_numpy(), ids)
            or not np.array_equal(readback.TAXITIME_SEC_mvt.to_numpy(), output)):
        raise ValueError("v9b prediction readback differs from template order")
    write_json(args.output_dir / "ranking_manifest.json", {
        "model_source": source, "model_sha256": sha256(model_file),
        "model_fit_sha256": sha256(fit_path), "model_rounds": rounds,
        "arrival_cache_sha256": sha256(args.arrival_cache),
        "v5_oof_sha256": sha256(args.v5_oof),
        "ranking_inputs_sha256": sha256(args.output_dir / "ranking_inputs.json"),
        "ranking_input_sha256": {name: item["sha256"]
                                   for name, item in ranking_inputs["inputs"].items()},
        "audit_sha256": sha256(args.output_dir / "audit.json"),
        "reserved_guard_sha256": sha256(args.reserved_guard_dir / "v9b" /
                                          "guard_report.json"),
        "v7_manifest_sha256": sha256(v7_manifest_path),
        "v7_reference_sha256": sha256(reference_path),
        "ranking_rows": len(ids), "valid_aobt": int(valid.sum()),
        "invalid_unchanged": True, "selected_weight": selected,
        "ranking_expert_sha256": sha256(expert_path),
        "predictions_sha256": sha256(output_path),
        "finite_nonnegative": True, "uploaded": False,
    })
    print(json.dumps({"predictions": str(output_path), "rows": len(ids),
                      "valid_aobt": int(valid.sum()), "model_source": source}, indent=2))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("prepare", "predict-fold", "evaluate-folds",
                                      "fit-fresh", "predict-fresh", "audit",
                                      "fit-final", "final-predict"), default="prepare")
    p.add_argument("--fold", choices=tuple(FOLDS))
    p.add_argument("--v9-protocol", type=Path,
                   default=Path("artifacts/v9-movement-valid/protocol.json"))
    p.add_argument("--movement-dir", type=Path,
                   default=Path("artifacts/v6-movement-only"))
    p.add_argument("--v7-dir", type=Path,
                   default=Path("artifacts/v7-runway-traffic"))
    p.add_argument("--cache-dir", type=Path, default=Path("artifacts/baseline"))
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--weather-file", type=Path,
                   default=Path("data/external/weather.parquet"))
    p.add_argument("--arrival-cache", type=Path,
                   default=Path("artifacts/v5-arrival-clean/training_arrival_features.parquet"))
    p.add_argument("--v5-oof", type=Path,
                   default=Path("artifacts/v5-ensemble/validation_predictions.parquet"))
    p.add_argument("--ranking-arrival-cache", type=Path,
                   default=Path("artifacts/v5-arrival-clean/ranking_arrival_features.parquet"))
    p.add_argument("--threads", type=int, default=3)
    p.add_argument("--min-free-gib", type=float, default=10.0)
    p.add_argument("--reserved-guard-dir", type=Path,
                   default=Path("artifacts/reserved-valid-guard"))
    p.add_argument("--output-dir", type=Path,
                   default=Path("artifacts/v9b-movement-valid"))
    args = p.parse_args()
    if args.threads != 3:
        p.error("v9b uses the frozen movement protocol's 3 CPU threads")
    if args.mode == "prepare":
        value = protocol(args)
        print(json.dumps({"protocol_path": str(args.output_dir / "protocol.json"),
                          "source_sha256": value["source_sha256"]}, indent=2))
    elif args.mode == "predict-fold":
        if not args.fold:
            p.error("predict-fold requires --fold")
        predict(args, args.fold, FOLDS[args.fold])
    elif args.mode == "evaluate-folds":
        evaluate_folds(args)
    elif args.mode == "fit-fresh":
        fit_fresh(args)
    elif args.mode == "predict-fresh":
        require_fold_gates(args)
        predict(args, "fresh_apr_oct", FRESH_MONTHS)
    elif args.mode == "audit":
        audit(args)
    elif args.mode == "fit-final":
        fit_final(args)
    elif args.mode == "final-predict":
        final_predict(args)


if __name__ == "__main__":
    main()
