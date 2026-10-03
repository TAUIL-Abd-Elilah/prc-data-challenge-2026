"""Fixed February/August paired guard for prospective valid-AOBT replacements.

All paired models exclude February and August from fitting and internal early
stopping. The v7 traffic comparator is provided by reserved_v7_comparator.py.
This module freezes previously selected weights, fits only a previously
accepted replacement architecture, and scores the paired held-out months.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


MONTHS = (2, 8)
SEED = 20261002
ROUTES = ("v8_combo", "v9b")
POLICY_ORDER = ("v7", *ROUTES)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def exact_ids(actual: pd.Series, expected: pd.Series, label: str) -> None:
    a, e = pd.Index(actual), pd.Index(expected)
    if (len(a) != len(e) or a.has_duplicates or e.has_duplicates
            or a.isna().any() or e.isna().any() or not a.isin(e).all()
            or not e.isin(a).all()):
        raise ValueError(f"{label}: exact unique ID coverage failed")


def select_policy(args: argparse.Namespace) -> dict:
    """Freeze one valid-AOBT policy before any reserved model is fitted."""
    policy_path = args.output_dir / "selected_policy.json"
    v8_validation_path = args.v8_dir / "validation.json"
    v9b_folds_path = args.v9b_dir / "existing_fold_validation.json"
    if not v8_validation_path.exists() or not v9b_folds_path.exists():
        raise ValueError("Both v8 and v9b existing-fold decisions must finish before portfolio selection")
    v8_validation = json.loads(v8_validation_path.read_text(encoding="utf-8"))
    v9b_folds = json.loads(v9b_folds_path.read_text(encoding="utf-8"))
    if (v8_validation.get("existing_folds_passed")
            and "fresh_audit" not in v8_validation):
        raise ValueError("v8 April/October audit is pending")
    if (v8_validation.get("promoted")
            and not (args.v8_combo_dir / "audit.json").exists()):
        raise ValueError("v8 fixed v7 combo audit is pending")
    if (v9b_folds.get("existing_folds_passed")
            and not (args.v9b_dir / "audit.json").exists()):
        raise ValueError("v9b April/October audit is pending")
    v7_path = args.v7_dir / "validation_predictions.parquet"
    v7_report_path = args.v7_dir / "validation.json"
    v7_report = json.loads(v7_report_path.read_text(encoding="utf-8"))
    if (not v7_report.get("promoted")
            or v7_report["validation_predictions_sha256"] != sha256(v7_path)):
        raise ValueError("Accepted v7 all-finite reference is not intact")
    columns = ["MVT_ID_mvt", "target", "fold", "month",
               "MVT_TIME_UTC_mvt", "a_valid", "candidate"]
    baseline = pd.read_parquet(args.cache_dir / "training_rows.parquet",
                               columns=["MVT_ID_mvt", "target", "proxy",
                                        "month", "time"])
    expected = baseline.loc[
        baseline.month.isin((1, 7, 11, 12)).to_numpy()
        & np.isfinite(baseline.target.to_numpy(dtype=float))]
    v7 = pd.read_parquet(v7_path, columns=columns)
    exact_ids(v7.MVT_ID_mvt, expected.MVT_ID_mvt, "portfolio v7 all-finite OOF")
    v7 = expected[["MVT_ID_mvt", "target", "proxy", "month", "time"]].merge(
        v7, on="MVT_ID_mvt", how="left", sort=False, validate="one_to_one",
        suffixes=("_baseline", ""))
    proxy = v7.proxy.to_numpy(dtype=float)
    valid = np.isfinite(proxy) & (proxy >= 0) & (proxy <= 7200)
    if (not np.allclose(v7.target_baseline, v7.target, rtol=0, atol=1e-6)
            or not np.array_equal(v7.month_baseline.to_numpy(), v7.month.to_numpy())
            or not np.array_equal(v7.a_valid.to_numpy(dtype=bool), valid)
            or not np.array_equal(pd.to_datetime(v7.time, utc=True).to_numpy(),
                                  pd.to_datetime(v7.MVT_TIME_UTC_mvt,
                                                 utc=True).to_numpy())
            or not np.isfinite(v7.candidate.to_numpy(dtype=float)).all()
            or np.any(v7.candidate.to_numpy(dtype=float) < 0)):
        raise ValueError("Portfolio v7 labels, time, gate or prediction differ")
    fold_months = {"seasonal_jan_jul": (1, 7), "forward_nov_dec": (11, 12)}
    if (set(v7.fold.dropna().unique()) != set(fold_months)
            or any(not v7.loc[v7.fold.eq(name), "month"].isin(months).all()
                   for name, months in fold_months.items())):
        raise ValueError("Portfolio v7 fold/month mapping differs")
    candidates = {
        "v7": {"path": v7_path, "prediction_column": "candidate",
               "report": v7_report_path, "eligible": True,
               "protocol": args.v7_dir / "protocol.json"},
        "v8_combo": {"path": args.v8_combo_dir / "validation_predictions.parquet",
                     "prediction_column": "combined",
                     "report": args.v8_combo_dir / "audit.json",
                     "protocol": args.v8_combo_dir / "protocol.json"},
        "v9b": {"path": args.v9b_dir / "validation_predictions.parquet",
                "prediction_column": "movement_valid_candidate",
                "report": args.v9b_dir / "audit.json",
                "protocol": args.v9b_dir / "protocol.json"},
    }
    y = v7.target.to_numpy(dtype=float)
    season = v7.fold.eq("seasonal_jan_jul").to_numpy()
    scores = {}
    sources = {
        "current_candidate_policy": args.current_policy,
        "master_reserved_guard": args.master_protocol,
        "baseline_training_rows": args.cache_dir / "training_rows.parquet",
        "v7_protocol": args.v7_dir / "protocol.json",
        "v7_validation": v7_report_path,
        "v7_oof": v7_path,
        "v8_validation": v8_validation_path,
        "v9b_existing_folds": v9b_folds_path,
    }
    for name, item in candidates.items():
        path = item["path"]
        report_path = item["report"]
        if name != "v7" and not report_path.exists():
            scores[name] = {"eligible": False, "reason": "existing local audit absent"}
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        eligible = (True if name == "v7" else
                    bool(report.get("accepted") and (
                        v8_validation.get("promoted") if name == "v8_combo"
                        else v9b_folds.get("existing_folds_passed"))))
        if name != "v7" and not path.exists():
            if eligible:
                raise ValueError(f"{name} accepted audit lacks its OOF")
            scores[name] = {"eligible": False, "reason": "existing local OOF absent"}
            sources[f"{name}_audit"] = report_path
            continue
        if name != "v7":
            if (name == "v8_combo" and
                    report["source_sha256"]["v8_validation"] !=
                    sha256(v8_validation_path)):
                raise ValueError("v8 combo audit references a different v8 validation")
            if (name == "v9b" and
                    report["source_sha256"]["existing_fold_validation"] !=
                    sha256(v9b_folds_path)):
                raise ValueError("v9b audit references a different fold decision")
            key = "existing_oof" if name == "v8_combo" else "fold_oof"
            if report["output_sha256"][key] != sha256(path):
                raise ValueError(f"{name} local audit does not bind its OOF")
            frame = pd.read_parquet(path, columns=[*columns[:-1],
                                                   item["prediction_column"]])
            exact_ids(frame.MVT_ID_mvt, v7.MVT_ID_mvt, f"portfolio {name} OOF")
            # The v7 master was aligned to chronological baseline row order;
            # saved expert OOFs use fold-concatenated order. After exact unique
            # ID-set proof, align by ID before comparing every metadata field.
            frame = v7[["MVT_ID_mvt"]].merge(
                frame, on="MVT_ID_mvt", how="left", sort=False,
                validate="one_to_one")
            if (not frame.MVT_ID_mvt.equals(v7.MVT_ID_mvt)
                    or not np.allclose(frame.target, v7.target, rtol=0, atol=1e-6)
                    or not np.array_equal(frame.fold.to_numpy(), v7.fold.to_numpy())
                    or not np.array_equal(frame.month.to_numpy(), v7.month.to_numpy())
                    or not np.array_equal(frame.a_valid.to_numpy(dtype=bool), valid)
                    or not np.array_equal(
                        pd.to_datetime(frame.MVT_TIME_UTC_mvt, utc=True).to_numpy(),
                        pd.to_datetime(v7.MVT_TIME_UTC_mvt, utc=True).to_numpy())):
                raise ValueError(f"Portfolio {name} OOF metadata or target differs")
            prediction = frame[item["prediction_column"]].to_numpy(dtype=float)
            if (not np.isfinite(prediction).all() or np.any(prediction < 0)
                    or not np.array_equal(
                        prediction[~valid],
                        v7.candidate.to_numpy(dtype=float)[~valid])):
                raise ValueError(f"Portfolio {name} nonfinite or changed invalid row")
            sources[f"{name}_oof"] = path
            sources[f"{name}_audit"] = report_path
            sources[f"{name}_protocol"] = item["protocol"]
        else:
            prediction = v7.candidate.to_numpy(dtype=float)
        scores[name] = {
            "eligible": eligible,
            "seasonal_all_finite_rmse": float(np.sqrt(np.mean(
                (y[season] - prediction[season])**2))),
            "forward_all_finite_rmse": float(np.sqrt(np.mean(
                (y[~season] - prediction[~season])**2))),
        }
    selected = min((name for name in POLICY_ORDER if scores[name]["eligible"]),
                   key=lambda name: (scores[name]["seasonal_all_finite_rmse"],
                                     POLICY_ORDER.index(name)))
    report = {
        "selected_route": selected, "tie_order": list(POLICY_ORDER),
        "selection_metric": "all-finite January/July RMSE among routes that passed existing Jan/Jul, Nov/Dec and Apr/Oct gates",
        "all_finite_rows": len(v7), "n_valid_aobt": int(valid.sum()),
        "scores": scores,
        "source_sha256": {name: sha256(path) for name, path in sources.items()},
        "guard_months_not_scored_for_selection": list(MONTHS),
    }
    if policy_path.exists():
        if json.loads(policy_path.read_text(encoding="utf-8")) != report:
            raise ValueError("Frozen valid-route portfolio selection or sources changed")
    else:
        write_json(policy_path, report)
    print(json.dumps({"selected_route": selected, "scores": scores}, indent=2))
    return report


def require_selected_route(args: argparse.Namespace, route: str) -> dict:
    path = args.output_dir / "selected_policy.json"
    selection = json.loads(path.read_text(encoding="utf-8"))
    if selection["selected_route"] != route:
        raise ValueError(f"{route} was not the frozen selected valid route")
    paths = {
        "current_candidate_policy": args.current_policy,
        "master_reserved_guard": args.master_protocol,
        "baseline_training_rows": args.cache_dir / "training_rows.parquet",
        "v7_protocol": args.v7_dir / "protocol.json",
        "v7_validation": args.v7_dir / "validation.json",
        "v7_oof": args.v7_dir / "validation_predictions.parquet",
        "v8_validation": args.v8_dir / "validation.json",
        "v9b_existing_folds": args.v9b_dir / "existing_fold_validation.json",
        "v8_combo_oof": args.v8_combo_dir / "validation_predictions.parquet",
        "v8_combo_audit": args.v8_combo_dir / "audit.json",
        "v8_combo_protocol": args.v8_combo_dir / "protocol.json",
        "v9b_oof": args.v9b_dir / "validation_predictions.parquet",
        "v9b_audit": args.v9b_dir / "audit.json",
        "v9b_protocol": args.v9b_dir / "protocol.json",
    }
    if any(name not in paths or sha256(paths[name]) != digest
           for name, digest in selection["source_sha256"].items()):
        raise ValueError("Frozen valid-route portfolio source hashes changed")
    return selection


def route_spec(args: argparse.Namespace, route: str) -> dict:
    selection = require_selected_route(args, route)
    master_path = args.master_protocol
    if route == "v8_combo":
        validation_path = args.v8_dir / "validation.json"
        combo_path = args.v8_combo_dir / "audit.json"
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        combo = json.loads(combo_path.read_text(encoding="utf-8"))
        if (not validation.get("promoted") or not combo.get("accepted")
                or float(combo["weight"]) !=
                   float(validation["selected_weight"])):
            raise ValueError("v8 and its fixed v7 combo must pass existing local gates")
        weight = float(validation["selected_weight"])
        sources = {
            "v8_protocol": args.v8_dir / "protocol.json",
            "v8_validation": validation_path,
            "v8_combo_audit": combo_path,
            "v8_combo_oof": args.v8_combo_dir / "validation_predictions.parquet",
            "v8_combo_fresh": args.v8_combo_dir / "fresh_audit_predictions.parquet",
        }
    elif route == "v9b":
        audit_path = args.v9b_dir / "audit.json"
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if not audit.get("accepted"):
            raise ValueError("v9b valid-AOBT route must pass existing local gates")
        movement_manifest = json.loads((args.movement_dir / "features_manifest.json")
                                       .read_text(encoding="utf-8"))
        if (movement_manifest["arrival_cache_sha256"] != sha256(args.arrival_cache)
                or movement_manifest["reference_sha256"] != sha256(args.v5_oof)):
            raise ValueError("Reserved movement sources differ from frozen feature manifest")
        weight = float(audit["selected_weight"])
        sources = {
            "v9b_protocol": args.v9b_dir / "protocol.json",
            "v9b_audit": audit_path,
            "v9b_oof": args.v9b_dir / "validation_predictions.parquet",
            "v9b_fresh": args.v9b_dir / "fresh_audit_predictions.parquet",
            "movement_protocol": args.movement_dir / "protocol.json",
            "movement_features_manifest": args.movement_dir / "features_manifest.json",
            "movement_arrival_cache": args.arrival_cache,
            "movement_v5_oof": args.v5_oof,
        }
    else:
        raise ValueError(f"Unknown valid-AOBT reserved route: {route}")
    if weight <= 0 or weight > 1:
        raise ValueError("Selected existing-fold weight is invalid")
    sources.update({
        "selected_policy": args.output_dir / "selected_policy.json",
        "current_candidate_policy": args.current_policy,
        "master_guard": master_path,
        "v7_protocol": args.v7_dir / "protocol.json",
        "v7_validation": args.v7_dir / "validation.json",
        "v7_fresh_report": args.v7_dir / "fresh_audit.json",
        "baseline_rows": args.cache_dir / "training_rows.parquet",
    })
    return {
        "route": route, "heldout_months": list(MONTHS),
        "selected_weight": weight,
        "portfolio_selected_route": selection["selected_route"],
        "source_sha256": {name: sha256(path) for name, path in sources.items()},
        "comparator": "fresh depth-10 v7 traffic residual expert, with February/August excluded from fit and early stop",
        "replacement": ("fresh fixed v8 LightGBM residual expert" if route == "v8_combo"
                        else "fresh frozen movement-only direct expert"),
        "formula": "clip(v7_raw_expert + selected_weight*(replacement_raw_expert-v7_raw_expert), lower=0)",
        "scoring_gate": "all finite 2025 labels, own AOBT proxy in [0,7200], months February or August",
        "pass_rule": "RMSE improves in each month and pooled paired UTC-day bootstrap 95% lower gain >0, 1000 resamples seed 20261002",
        "failure": "exclude this replacement without tuning weight, predictors or routing gate",
    }


def prepare(args: argparse.Namespace, route: str) -> dict:
    spec = route_spec(args, route)
    path = args.output_dir / route / "protocol.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != spec:
            raise ValueError(f"Frozen {route} reserved protocol or source hashes changed")
    else:
        write_json(path, spec)
    return spec


def fit_v8(args: argparse.Namespace) -> None:
    """Refit the exact v8 architecture with February/August held out."""
    import v8_lightgbm_residual as v8

    spec = prepare(args, "v8_combo")
    out = args.output_dir / "v8_combo"
    manifest_path = out / "replacement_fit_manifest.json"
    model_path = out / "feb_aug.txt"
    oof_path = out / "feb_aug_oof.parquet"
    if any(path.exists() for path in (manifest_path, model_path, oof_path)):
        verify_v8_fit(args, spec)
        print(json.dumps({"status": "reused", "model": str(model_path)}, indent=2))
        return
    source = argparse.Namespace(**vars(args))
    source.output_dir = out
    rows, features = v8.load_features(source)
    fit = v8.fit_fold("feb_aug", MONTHS, rows, features, source)
    write_json(manifest_path, {
        "route": "v8_combo", "heldout_months": list(MONTHS),
        "protocol_sha256": sha256(out / "protocol.json"),
        "v8_protocol_sha256": sha256(args.v8_dir / "protocol.json"),
        "model_sha256": sha256(model_path), "oof_sha256": sha256(oof_path),
        "fit_report_sha256": sha256(out / "feb_aug_fit.json"),
        "best_iteration": int(fit["best_iteration"]),
        "feature_names": fit["feature_names"],
        "params": v8.fixed_params(),
    })
    print(json.dumps({"status": "complete", "model": str(model_path),
                      "best_iteration": fit["best_iteration"]}, indent=2))


def verify_v8_fit(args: argparse.Namespace, spec: dict) -> Path:
    out = args.output_dir / "v8_combo"
    manifest = json.loads((out / "replacement_fit_manifest.json")
                          .read_text(encoding="utf-8"))
    model = out / "feb_aug.txt"
    oof = out / "feb_aug_oof.parquet"
    fit_report = out / "feb_aug_fit.json"
    if (manifest["route"] != "v8_combo"
            or manifest["heldout_months"] != list(MONTHS)
            or manifest["protocol_sha256"] != sha256(out / "protocol.json")
            or manifest["v8_protocol_sha256"] != spec["source_sha256"]["v8_protocol"]
            or manifest["model_sha256"] != sha256(model)
            or manifest["oof_sha256"] != sha256(oof)
            or manifest["fit_report_sha256"] != sha256(fit_report)):
        raise ValueError("v8 reserved replacement model or input provenance differs")
    return oof


def expected_rows(args: argparse.Namespace) -> pd.DataFrame:
    rows = pd.read_parquet(args.cache_dir / "training_rows.parquet",
                           columns=["MVT_ID_mvt", "target", "proxy", "month", "time"])
    proxy = rows.proxy.to_numpy(dtype=float)
    target = rows.target.to_numpy(dtype=float)
    gate = (rows.month.isin(MONTHS).to_numpy() & np.isfinite(target)
            & np.isfinite(proxy) & (proxy >= 0) & (proxy <= 7200))
    held = rows.loc[gate, ["MVT_ID_mvt", "target", "month", "time"]].copy()
    held.rename(columns={"time": "MVT_TIME_UTC_mvt"}, inplace=True)
    if held.MVT_ID_mvt.isna().any() or held.MVT_ID_mvt.duplicated().any():
        raise ValueError("February/August expected held-out IDs repeat")
    return held


def load_comparator(args: argparse.Namespace, expected: pd.DataFrame) -> pd.DataFrame:
    root = args.v7_reserved_dir
    oof_path = root / "feb_aug_oof.parquet"
    manifest = json.loads((root / "feb_aug_manifest.json")
                          .read_text(encoding="utf-8"))
    model_path = root / "feb_aug.cbm"
    fit_path = root / "feb_aug_fit.json"
    comparator_protocol = root / "protocol.json"
    if (manifest["heldout_months"] != list(MONTHS)
            or manifest["oof_sha256"] != sha256(oof_path)
            or manifest["comparator_model_sha256"] != sha256(model_path)
            or manifest["fit_report_sha256"] != sha256(fit_path)
            or manifest["protocol_sha256"] != sha256(comparator_protocol)
            or manifest["frozen_v7_protocol_sha256"] !=
               sha256(args.v7_dir / "protocol.json")
            or manifest["reserved_guard_protocol_sha256"] !=
               sha256(args.master_protocol)
            or manifest["selected_valid_policy_sha256"] !=
               sha256(args.output_dir / "selected_policy.json")
            or manifest["selected_valid_route"] !=
               json.loads((args.output_dir / "selected_policy.json")
                          .read_text(encoding="utf-8"))["selected_route"]):
        raise ValueError("v7 reserved comparator provenance differs")
    comparator = pd.read_parquet(oof_path,
                                  columns=["MVT_ID_mvt", "target",
                                           "MVT_TIME_UTC_mvt", "expert"])
    exact_ids(comparator.MVT_ID_mvt, expected.MVT_ID_mvt,
              "v7 reserved comparator")
    comparator = expected.merge(comparator, on="MVT_ID_mvt", how="left",
                                sort=False, validate="one_to_one",
                                suffixes=("", "_comparator"))
    if (not np.allclose(comparator.target, comparator.target_comparator,
                        rtol=0, atol=1e-6)
            or not np.array_equal(
                pd.to_datetime(comparator.MVT_TIME_UTC_mvt, utc=True).to_numpy(),
                pd.to_datetime(comparator.MVT_TIME_UTC_mvt_comparator,
                               utc=True).to_numpy())
            or not np.isfinite(comparator.expert.to_numpy(dtype=float)).all()):
        raise ValueError("v7 comparator labels, times or predictions differ")
    return comparator[["MVT_ID_mvt", "target", "month",
                       "MVT_TIME_UTC_mvt", "expert"]].rename(
                           columns={"expert": "v7_expert"})


def shared_movement_model(args: argparse.Namespace) -> tuple[Path, int, Path] | None:
    import movement_only_expert as movement

    model_path = args.movement_dir / "february_august_movement.txt"
    report_path = args.movement_dir / "reserved_audit.json"
    # The movement pipeline may save its model before a longer paired audit
    # completes. That orphan is never a validated source for this route.
    if not model_path.exists() or not report_path.exists():
        return None
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (report["heldout_months"] != list(MONTHS)
            or report["model_sha256"]["movement"] != sha256(model_path)
            or report["features_manifest_sha256"] !=
               sha256(args.movement_dir / "features_manifest.json")
            or report["prepared_features_sha256"] !=
               sha256(args.movement_dir / "features.parquet")
            or report["row_ids_sha256"] !=
               sha256(args.movement_dir / "row_ids.parquet")):
        raise ValueError("Shared February/August movement model provenance differs")
    movement.verify_audit_artifacts(args.movement_dir, args.cache_dir, report,
                                    "february_august", MONTHS)
    return model_path, int(report["movement_training"]["best_round"]), report_path


def own_movement_model(args: argparse.Namespace) -> tuple[Path, int, Path] | None:
    out = args.output_dir / "v9b"
    model_path = out / "feb_aug_movement.txt"
    report_path = out / "movement_fit_manifest.json"
    if not model_path.exists() and not report_path.exists():
        return None
    if not model_path.exists() or not report_path.exists():
        raise ValueError("v9b reserved movement model/report are incomplete")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (report["heldout_months"] != list(MONTHS)
            or report["model_sha256"] != sha256(model_path)
            or report["features_manifest_sha256"] !=
               sha256(args.movement_dir / "features_manifest.json")
            or report["prepared_features_sha256"] !=
               sha256(args.movement_dir / "features.parquet")
            or report["arrival_cache_sha256"] != sha256(args.arrival_cache)
            or report["v5_oof_sha256"] != sha256(args.v5_oof)
            or report["route_protocol_sha256"] != sha256(out / "protocol.json")):
        raise ValueError("v9b reserved movement fit provenance differs")
    return model_path, int(report["best_round"]), report_path


def fit_movement(args: argparse.Namespace) -> None:
    """Reuse a validated shared movement refit or fit the same frozen model."""
    import movement_only_expert as movement

    spec = prepare(args, "v9b")
    out = args.output_dir / "v9b"
    oof_path = out / "feb_aug_oof.parquet"
    fit_manifest_path = out / "replacement_fit_manifest.json"
    if oof_path.exists() or fit_manifest_path.exists():
        verify_movement_fit(args, spec, expected_rows(args))
        print(json.dumps({"status": "reused", "oof": str(oof_path)}, indent=2))
        return
    source = own_movement_model(args) or shared_movement_model(args)
    source_args = argparse.Namespace(**vars(args))
    source_args.output_dir = args.movement_dir
    if source is None:
        movement.require_memory(args.min_free_gib)
        source_args.threads = 3
        features, rows, manifest = movement.load_prepared(source_args)
        model, fit = movement.fit_movement_model(
            source_args, features, rows, manifest["categorical"], MONTHS)
        model_path = out / "feb_aug_movement.txt"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save_model(str(model_path))
        own_report_path = out / "movement_fit_manifest.json"
        write_json(own_report_path, {
            "heldout_months": list(MONTHS), "best_round": int(fit["best_round"]),
            "fit": fit, "model_sha256": sha256(model_path),
            "features_manifest_sha256": sha256(args.movement_dir / "features_manifest.json"),
            "prepared_features_sha256": sha256(args.movement_dir / "features.parquet"),
            "arrival_cache_sha256": sha256(args.arrival_cache),
            "v5_oof_sha256": sha256(args.v5_oof),
            "route_protocol_sha256": sha256(out / "protocol.json"),
            "params": movement.model_params(3),
        })
        source = model_path, int(fit["best_round"]), own_report_path
    model_path, rounds, source_report_path = source
    features, rows, manifest = movement.load_prepared(source_args)
    expected = expected_rows(args)
    proxy = rows.proxy.to_numpy(dtype=float)
    target = rows.target.to_numpy(dtype=float)
    valid = (rows.month.isin(MONTHS).to_numpy() & np.isfinite(target)
             & np.isfinite(proxy) & (proxy >= 0) & (proxy <= 7200))
    exact_ids(rows.loc[valid, "MVT_ID_mvt"], expected.MVT_ID_mvt,
              "v9b reserved prediction gate")
    model = lgb.Booster(model_file=str(model_path))
    if (model.feature_name() != manifest["features"]
            or not 1 <= rounds <= model.num_trees()):
        raise ValueError("Reserved movement model features or best round differ")
    expert = np.maximum(model.predict(features.loc[valid],
                                      num_iteration=rounds, num_threads=3), 0)
    if len(expert) != len(expected) or not np.isfinite(expert).all():
        raise ValueError("Reserved movement predictions lack finite coverage")
    pd.DataFrame({"MVT_ID_mvt": rows.loc[valid, "MVT_ID_mvt"].to_numpy(),
                  "expert": expert}).to_parquet(oof_path, index=False)
    write_json(fit_manifest_path, {
        "route": "v9b", "heldout_months": list(MONTHS),
        "route_protocol_sha256": sha256(out / "protocol.json"),
        "model_source": "v9b_own" if model_path.parent == out else "shared_movement",
        "model_sha256": sha256(model_path),
        "source_fit_report_sha256": sha256(source_report_path),
        "features_manifest_sha256": sha256(args.movement_dir / "features_manifest.json"),
        "prepared_features_sha256": sha256(args.movement_dir / "features.parquet"),
        "arrival_cache_sha256": sha256(args.arrival_cache),
        "v5_oof_sha256": sha256(args.v5_oof),
        "model_best_round": rounds, "oof_sha256": sha256(oof_path),
        "n_heldout_valid_finite": len(expected),
    })
    print(json.dumps({"status": "complete", "model": str(model_path),
                      "oof": str(oof_path), "n": len(expected)}, indent=2))


def verify_movement_fit(args: argparse.Namespace, spec: dict,
                        expected: pd.DataFrame) -> Path:
    out = args.output_dir / "v9b"
    manifest = json.loads((out / "replacement_fit_manifest.json")
                          .read_text(encoding="utf-8"))
    if manifest["model_source"] == "v9b_own":
        source = own_movement_model(args)
    elif manifest["model_source"] == "shared_movement":
        source = shared_movement_model(args)
    else:
        raise ValueError("Unknown reserved movement model source")
    if source is None:
        raise ValueError("Reserved movement source model/report is unavailable")
    model_path, rounds, fit_path = source
    oof_path = out / "feb_aug_oof.parquet"
    if (manifest["route"] != "v9b"
            or manifest["heldout_months"] != list(MONTHS)
            or manifest["route_protocol_sha256"] != sha256(out / "protocol.json")
            or manifest["model_sha256"] != sha256(model_path)
            or manifest["source_fit_report_sha256"] != sha256(fit_path)
            or manifest["features_manifest_sha256"] !=
               sha256(args.movement_dir / "features_manifest.json")
            or manifest["prepared_features_sha256"] !=
               sha256(args.movement_dir / "features.parquet")
            or manifest["arrival_cache_sha256"] != sha256(args.arrival_cache)
            or manifest["v5_oof_sha256"] != sha256(args.v5_oof)
            or manifest["model_best_round"] != rounds
            or manifest["oof_sha256"] != sha256(oof_path)
            or manifest["n_heldout_valid_finite"] != len(expected)):
        raise ValueError("v9b reserved movement model or OOF provenance differs")
    frame = pd.read_parquet(oof_path)
    if list(frame) != ["MVT_ID_mvt", "expert"] or not np.isfinite(
            frame.expert.to_numpy(dtype=float)).all():
        raise ValueError("Reserved movement OOF schema or predictions differ")
    exact_ids(frame.MVT_ID_mvt, expected.MVT_ID_mvt,
              "v9b reserved movement OOF")
    return oof_path


def score(args: argparse.Namespace, route: str) -> None:
    import deep_arrival_expert as arrival
    import deep_timestamp_expert as deep

    spec = prepare(args, route)
    expected = expected_rows(args)
    comparator = load_comparator(args, expected)
    if route == "v8_combo":
        candidate_path = verify_v8_fit(args, spec)
    else:
        candidate_path = verify_movement_fit(args, spec, expected)
    candidate = pd.read_parquet(candidate_path,
                                columns=["MVT_ID_mvt", "expert"])
    exact_ids(candidate.MVT_ID_mvt, expected.MVT_ID_mvt,
              f"{route} reserved replacement")
    frame = comparator.merge(candidate, on="MVT_ID_mvt", how="left",
                             sort=False, validate="one_to_one")
    if (len(frame) != len(expected) or
            not frame.MVT_ID_mvt.equals(expected.MVT_ID_mvt) or
            not np.isfinite(frame.expert.to_numpy(dtype=float)).all()):
        raise ValueError("Reserved replacement coverage or finite values differ")
    base = np.maximum(frame.v7_expert.to_numpy(dtype=float), 0)
    alternate = frame.expert.to_numpy(dtype=float)
    selected = np.maximum(base + spec["selected_weight"]*(alternate-base), 0)
    frame["comparator"] = base
    frame["candidate"] = selected
    month_scores = {}
    for month in MONTHS:
        mask = frame.month.to_numpy(dtype=int) == month
        y = frame.target.to_numpy(dtype=float)[mask]
        month_scores[str(month)] = {
            "n": int(mask.sum()), "comparator_rmse": deep.rmse(y, base[mask]),
            "candidate_rmse": deep.rmse(y, selected[mask]),
        }
    stability = arrival.bootstrap(frame, base, selected, seed=SEED)
    passed = (all(month_scores[str(month)]["candidate_rmse"] <
                  month_scores[str(month)]["comparator_rmse"] for month in MONTHS)
              and stability["gain_ci95_sec"][0] > 0)
    out = args.output_dir / route
    paired_path = out / "paired_predictions.parquet"
    frame[["MVT_ID_mvt", "target", "month", "MVT_TIME_UTC_mvt",
           "v7_expert", "expert", "comparator", "candidate"]].to_parquet(
               paired_path, index=False)
    report = {
        "route": route, "heldout_months": list(MONTHS),
        "selected_weight": spec["selected_weight"],
        "month_scores": month_scores, "bootstrap": stability,
        "passed": bool(passed), "coverage_verified": True,
        "protocol_sha256": sha256(out / "protocol.json"),
        "master_guard_sha256": sha256(args.master_protocol),
        "selected_policy_sha256": sha256(args.output_dir / "selected_policy.json"),
        "comparator_manifest_sha256": sha256(args.v7_reserved_dir / "feb_aug_manifest.json"),
        "comparator_oof_path": str(args.v7_reserved_dir / "feb_aug_oof.parquet"),
        "comparator_oof_sha256": sha256(args.v7_reserved_dir / "feb_aug_oof.parquet"),
        "replacement_oof_sha256": sha256(candidate_path),
        "paired_predictions_sha256": sha256(paired_path),
        "source": "2025 released labels only; no leaderboard feedback",
    }
    write_json(out / "guard_report.json", report)
    print(json.dumps({"route": route, "month_scores": month_scores,
                      "bootstrap": stability, "passed": passed}, indent=2))


def require_route_guard(output_dir: Path, route: str) -> dict:
    """Final model/prediction entry points call this before any ranking work."""
    if route not in ROUTES:
        raise ValueError("Unknown reserved route")
    out = output_dir / route
    report_path = out / "guard_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    route_protocol = json.loads((out / "protocol.json").read_text(encoding="utf-8"))
    selection = json.loads((output_dir / "selected_policy.json")
                           .read_text(encoding="utf-8"))
    comparator_path = Path(report["comparator_oof_path"])
    comparator_manifest_path = comparator_path.parent / "feb_aug_manifest.json"
    comparator_manifest = json.loads(comparator_manifest_path.read_text(
        encoding="utf-8"))
    if (report.get("route") != route or not report.get("passed")
            or not report.get("coverage_verified")
            or report.get("heldout_months") != list(MONTHS)
            or selection.get("selected_route") != route
            or route_protocol.get("portfolio_selected_route") != route
            or float(report["selected_weight"]) !=
               float(route_protocol["selected_weight"])
            or route_protocol["source_sha256"]["selected_policy"] !=
               sha256(output_dir / "selected_policy.json")
            or report["selected_policy_sha256"] !=
               sha256(output_dir / "selected_policy.json")
            or report["protocol_sha256"] != sha256(out / "protocol.json")
            or report["paired_predictions_sha256"] !=
               sha256(out / "paired_predictions.parquet")
            or report["replacement_oof_sha256"] !=
               sha256(out / "feb_aug_oof.parquet")
            or report["comparator_manifest_sha256"] !=
               sha256(comparator_manifest_path)
            or report["comparator_oof_sha256"] !=
               sha256(comparator_path)
            or comparator_manifest["oof_sha256"] != sha256(comparator_path)
            or comparator_manifest["comparator_model_sha256"] !=
               sha256(comparator_path.parent / "feb_aug.cbm")
            or report["master_guard_sha256"] !=
               sha256(Path("reports/reserved_guard_protocol.json"))
            or any(report["month_scores"][str(m)]["candidate_rmse"] >=
                   report["month_scores"][str(m)]["comparator_rmse"]
                   for m in MONTHS)
            or report["bootstrap"]["gain_ci95_sec"][0] <= 0):
        raise ValueError(f"{route} fixed February/August guard is pending or failed")
    return report


def final_v8_combo(args: argparse.Namespace) -> None:
    """Produce only the fixed, guarded v7-based LightGBM ranking candidate."""
    import v8_lightgbm_residual as v8

    guard = require_route_guard(args.output_dir, "v8_combo")
    spec = prepare(args, "v8_combo")
    validation_path = args.v8_dir / "validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    combo_path = args.v8_combo_dir / "audit.json"
    combo = json.loads(combo_path.read_text(encoding="utf-8"))
    if (not validation.get("promoted") or not combo.get("accepted")
            or float(validation["selected_weight"]) != spec["selected_weight"]
            or float(combo["weight"]) != spec["selected_weight"]
            or validation["validation_predictions_sha256"] !=
               sha256(args.v8_dir / "validation_predictions.parquet")
            or combo["source_sha256"]["v8_validation"] != sha256(validation_path)
            or combo["output_sha256"]["existing_oof"] !=
               sha256(args.v8_combo_dir / "validation_predictions.parquet")):
        raise ValueError("Frozen v8 combo local gates or source hashes differ")
    rounds = int(np.median([validation["folds"][name]["fit"]["best_iteration"]
                            for name in ("seasonal_jan_jul", "forward_nov_dec")]))
    v8_args = argparse.Namespace(**vars(args))
    v8_args.output_dir = args.v8_dir
    v8_args.ranking_reference = args.v8_ranking_reference
    v8_args.prior_fresh_oof = args.prior_fresh_oof
    v8.protocol(v8_args)
    model, fit_manifest = v8.load_or_fit_full(v8_args, validation, rounds)
    rank_rows, rank_features = v8.load_features(v8_args, ranking=True)
    schema = v8.feature_schema(rank_features)
    if (schema["columns"] != fit_manifest["train_feature_schema"]["columns"]
            or schema["kinds"] != fit_manifest["train_feature_schema"]["kinds"]
            or schema["categorical_columns"] !=
               fit_manifest["train_feature_schema"]["categorical_columns"]):
        raise ValueError("v8 full model ranking feature schema differs")
    template = pd.read_parquet(args.data_dir / "submitting.parquet")
    reference_path = args.v7_dir / "predictions.parquet"
    v7_manifest_path = args.v7_dir / "manifest.json"
    v7_manifest = json.loads(v7_manifest_path.read_text(encoding="utf-8"))
    reference = pd.read_parquet(reference_path)
    expected_columns = ["MVT_ID_mvt", "TAXITIME_SEC_mvt"]
    if (list(template) != expected_columns or list(reference) != expected_columns
            or v7_manifest["predictions_sha256"] != sha256(reference_path)
            or len(rank_rows) != len(template)
            or template.MVT_ID_mvt.isna().any()
            or template.MVT_ID_mvt.duplicated().any()
            or not rank_rows.MVT_ID_mvt.equals(template.MVT_ID_mvt)
            or not reference.MVT_ID_mvt.equals(template.MVT_ID_mvt)):
        raise ValueError("v8 combo ranking reference, features or template differ")
    old = reference.TAXITIME_SEC_mvt.to_numpy(dtype=float, copy=True)
    proxy = rank_rows.proxy.to_numpy(dtype=float)
    valid = np.isfinite(proxy) & (proxy >= 0) & (proxy <= 7200)
    if (int(valid.sum()) != int(v7_manifest["valid_aobt"])
            or not np.isfinite(old).all() or np.any(old < 0)):
        raise ValueError("v8 combo v7 ranking gate or values differ")
    raw = proxy[valid] + model.predict(rank_features.loc[valid], num_threads=3)
    if len(raw) != int(valid.sum()) or not np.isfinite(raw).all():
        raise ValueError("v8 combo raw expert lacks finite valid-AOBT coverage")
    selected = old.copy()
    selected[valid] = np.maximum(old[valid] + spec["selected_weight"] *
                                 (raw - old[valid]), 0)
    if (not np.array_equal(selected[~valid], old[~valid])
            or not np.isfinite(selected).all() or np.any(selected < 0)):
        raise ValueError("v8 combo changed invalid rows or produced invalid values")
    expert = np.full(len(selected), np.nan, dtype=float)
    expert[valid] = raw
    raw_path = args.v8_combo_dir / "ranking_expert.parquet"
    prediction_path = args.v8_combo_dir / "predictions.parquet"
    pd.DataFrame({"MVT_ID_mvt": template.MVT_ID_mvt,
                  "expert": expert}).to_parquet(raw_path, index=False)
    pd.DataFrame({"MVT_ID_mvt": template.MVT_ID_mvt,
                  "TAXITIME_SEC_mvt": selected}).to_parquet(prediction_path,
                                                              index=False)
    readback = pd.read_parquet(prediction_path)
    if (not readback.MVT_ID_mvt.equals(template.MVT_ID_mvt)
            or not np.array_equal(readback.TAXITIME_SEC_mvt.to_numpy(dtype=float),
                                  selected)):
        raise ValueError("v8 combo ranking readback differs from exact template")
    write_json(args.v8_combo_dir / "ranking_manifest.json", {
        "rows": len(selected), "valid_aobt": int(valid.sum()),
        "selected_weight": spec["selected_weight"],
        "source_v7_ranking_sha256": sha256(reference_path),
        "v7_manifest_sha256": sha256(v7_manifest_path),
        "v8_full_model_sha256": fit_manifest["model_sha256"],
        "v8_full_fit_manifest_sha256": sha256(args.v8_dir / "full_fit_manifest.json"),
        "v8_validation_sha256": sha256(validation_path),
        "v8_combo_audit_sha256": sha256(combo_path),
        "reserved_guard_report_sha256": sha256(args.output_dir / "v8_combo" /
                                                 "guard_report.json"),
        "selected_policy_sha256": sha256(args.output_dir / "selected_policy.json"),
        "raw_expert_sha256": sha256(raw_path),
        "predictions_sha256": sha256(prediction_path),
        "invalid_aobt_unchanged": True, "finite_nonnegative": True,
        "uploaded": False,
    })
    print(json.dumps({"route": "v8_combo", "rows": len(selected),
                      "valid_aobt": int(valid.sum()),
                      "predictions": str(prediction_path),
                      "guard_passed": guard["passed"]}, indent=2))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("select-policy", "prepare", "fit-v8", "fit-movement",
                                      "score", "verify", "final-v8-combo"), required=True)
    p.add_argument("--route", choices=ROUTES)
    p.add_argument("--master-protocol", type=Path,
                   default=Path("reports/reserved_guard_protocol.json"))
    p.add_argument("--current-policy", type=Path,
                   default=Path("reports/current_candidate_policy_protocol.json"))
    p.add_argument("--output-dir", type=Path,
                   default=Path("artifacts/reserved-valid-guard"))
    p.add_argument("--v7-reserved-dir", type=Path,
                   default=Path("artifacts/v8-reserved-v7"))
    p.add_argument("--v7-dir", type=Path,
                   default=Path("artifacts/v7-runway-traffic"))
    p.add_argument("--v8-dir", type=Path,
                   default=Path("artifacts/v8-lightgbm"))
    p.add_argument("--v8-combo-dir", type=Path,
                   default=Path("artifacts/v8-combo"))
    p.add_argument("--v9b-dir", type=Path,
                   default=Path("artifacts/v9b-movement-valid"))
    p.add_argument("--movement-dir", type=Path,
                   default=Path("artifacts/v6-movement-only"))
    p.add_argument("--cache-dir", type=Path,
                   default=Path("artifacts/baseline"))
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--weather-file", type=Path,
                   default=Path("data/external/weather.parquet"))
    p.add_argument("--arrival-cache", type=Path,
                   default=Path("artifacts/v5-arrival-clean/training_arrival_features.parquet"))
    p.add_argument("--v5-oof", type=Path,
                   default=Path("artifacts/v5-ensemble/validation_predictions.parquet"))
    p.add_argument("--arrival-dir", type=Path,
                   default=Path("artifacts/v5-arrival-clean"))
    p.add_argument("--neighbour-dir", type=Path,
                   default=Path("artifacts/v6-neighbour"))
    p.add_argument("--runway-dir", type=Path,
                   default=Path("artifacts/v6-runway-arrival"))
    p.add_argument("--v8-ranking-reference", type=Path,
                   default=Path("submissions/merry-mushroom_v6.parquet"))
    p.add_argument("--prior-fresh-oof", type=Path,
                   default=Path("artifacts/v6-deep-arrival/fresh_new/fresh_apr_oct_oof.parquet"))
    p.add_argument("--source-reference", type=Path,
                   default=Path("artifacts/v6-deep-arrival/validation_predictions.parquet"))
    p.add_argument("--min-free-gib", type=float, default=10.0)
    args = p.parse_args()
    if args.mode in ("prepare", "score", "verify") and not args.route:
        p.error(f"{args.mode} requires --route")
    if args.mode == "select-policy":
        select_policy(args)
    elif args.mode == "prepare":
        result = prepare(args, args.route)
        print(json.dumps({"route": args.route,
                          "protocol": str(args.output_dir / args.route / "protocol.json"),
                          "selected_weight": result["selected_weight"]}, indent=2))
    elif args.mode == "fit-v8":
        fit_v8(args)
    elif args.mode == "fit-movement":
        fit_movement(args)
    elif args.mode == "score":
        score(args, args.route)
    elif args.mode == "final-v8-combo":
        final_v8_combo(args)
    else:
        print(json.dumps(require_route_guard(args.output_dir, args.route), indent=2))


if __name__ == "__main__":
    main()
