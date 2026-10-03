"""Guard and compose the frozen current-candidate policy, without model fitting.

``validate`` applies terminal 2025 component OOF corrections to the common
all-finite v7 universe and records the fixed two-fold composition gate.
``assemble`` uses only hash-bound final component ranking predictions after
that gate passes. Neither mode submits a file or reads leaderboard feedback.
The default ``contract`` mode only prints the required artifact schema.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import deep_arrival_expert as arrival
import deep_timestamp_expert as deep
import movement_only_expert as movement
import reserved_valid_guard as reserved


POLICY_SHA256 = "9dca07f04ae0cf4f9da4484fce42fc962bbec54fc4ee80cfc7e41f4882488c7c"
V7_RANKING_SHA256 = "314e5dcc537b46954744ddc02018c7e2a65cd1dfe10be8e13da313c69eb0b0eb"
OOF_ROWS = 672428
RANKING_ROWS = 344841
FOLDS = {"seasonal_jan_jul": (1, 7), "forward_nov_dec": (11, 12)}
BOOTSTRAP_SEED = 20261003
RANK_COLUMNS = ["MVT_ID_mvt", "TAXITIME_SEC_mvt"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_new_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as out:
        out.write(json.dumps(value, indent=2) + "\n")


def exact_ids(actual: pd.Series, expected: pd.Series, label: str) -> None:
    left, right = pd.Index(actual), pd.Index(expected)
    if (len(left) != len(right) or left.has_duplicates or right.has_duplicates
            or left.isna().any() or right.isna().any()
            or not left.isin(right).all() or not right.isin(left).all()):
        raise ValueError(f"{label}: exact unique ID coverage failed")


def aligned(frame: pd.DataFrame, ids: pd.Series, label: str) -> pd.DataFrame:
    exact_ids(frame.MVT_ID_mvt, ids, label)
    result = pd.DataFrame({"MVT_ID_mvt": ids.to_numpy(copy=True)}).merge(
        frame, on="MVT_ID_mvt", how="left", sort=False,
        validate="one_to_one")
    if not np.array_equal(result.MVT_ID_mvt.to_numpy(), ids.to_numpy()):
        raise ValueError(f"{label}: ID order changed on alignment")
    return result


def same_metadata(reference: pd.DataFrame, other: pd.DataFrame,
                  label: str, *, airport: bool = False,
                  valid: bool = False) -> None:
    fields = ["target", "fold", "month", "MVT_TIME_UTC_mvt"]
    if airport:
        fields.append("airport")
    if valid:
        fields.append("a_valid")
    for field in fields:
        if field == "MVT_TIME_UTC_mvt":
            left = pd.to_datetime(reference[field], utc=True).to_numpy()
            right = pd.to_datetime(other[field], utc=True).to_numpy()
        else:
            left, right = reference[field].to_numpy(), other[field].to_numpy()
        if not np.array_equal(left, right):
            raise ValueError(f"{label}: {field} differs from common OOF reference")


def policy_spec() -> dict:
    return {
        "prepublished_policy_sha256": POLICY_SHA256,
        "reference": "Accepted v7 all-finite OOF and exact 344841-row ranking output",
        "valid_route": "Frozen selected v7, v8_combo or v9b; use replacement only after its terminal February/August guard passes",
        "missing_route": "Use original movement correction only after both original folds, April/October and February/August guards pass",
        "masks": "valid own proxy [0,7200] and invalid no-NM non-LIRF are disjoint",
        "composition": "Apply each fixed component exactly once to v7; every other row equals v7 bitwise",
        "selection": "No weight, airport or alternative route tuning in this assembler",
        "OOF_gate": "all-finite RMSE improves and paired UTC-day 95% gain lower bound >0 separately in Jan/Jul and Nov/Dec",
        "bootstrap": {"repeats": 1000, "seed": BOOTSTRAP_SEED},
        "ranking": "Only hash-bound terminal model, source, cache and ranking-input manifests; exact template order; no overwrite",
        "submission": "This script writes an internal candidate artifact only; no upload or leaderboard calls",
    }


def guard_args(args: argparse.Namespace) -> argparse.Namespace:
    value = argparse.Namespace(**vars(args))
    value.output_dir = args.guard_dir
    value.current_policy = args.policy_path
    value.master_protocol = args.reserved_protocol
    return value


def check_policy(args: argparse.Namespace) -> dict:
    if sha256(args.policy_path) != POLICY_SHA256:
        raise ValueError("Prepublished current-candidate policy changed")
    policy = read_json(args.policy_path)
    if "v7" not in policy["baseline"] or policy["leaderboard_use"] != (
            "Ranking scores are recorded only after policy and file freezing. They never choose these models, weights, routes or checks."):
        raise ValueError("Current-candidate policy text differs from the pinned source")
    return policy


def verify_failed_valid_guard(args: argparse.Namespace, route: str,
                              report: dict) -> None:
    gargs = guard_args(args)
    spec = reserved.route_spec(gargs, route)
    protocol_path = args.guard_dir / route / "protocol.json"
    if read_json(protocol_path) != spec:
        raise ValueError("Failed valid-route guard protocol or input sources changed")
    paired_path = args.guard_dir / route / "paired_predictions.parquet"
    comparator_path = args.v7_reserved_dir / "feb_aug_oof.parquet"
    candidate_path = (args.guard_dir / route / "feb_aug_oof.parquet")
    if (report.get("route") != route or report.get("heldout_months") != [2, 8]
            or report.get("coverage_verified") is not True
            or report.get("passed") is not False
            or report["protocol_sha256"] != sha256(protocol_path)
            or report["selected_policy_sha256"] != sha256(
                args.guard_dir / "selected_policy.json")
            or report["paired_predictions_sha256"] != sha256(paired_path)
            or report["comparator_oof_sha256"] != sha256(comparator_path)
            or report["replacement_oof_sha256"] != sha256(candidate_path)
            or report["master_guard_sha256"] != sha256(args.reserved_protocol)
            or float(report["selected_weight"]) != float(spec["selected_weight"])):
        raise ValueError("Failed valid-route guard lacks terminal provenance")
    actual_pass = (all(report["month_scores"][str(month)]["candidate_rmse"]
                       < report["month_scores"][str(month)]["comparator_rmse"]
                       for month in (2, 8))
                   and report["bootstrap"]["gain_ci95_sec"][0] > 0)
    if actual_pass:
        raise ValueError("Valid-route report says failed despite passing its fixed gate")


def verify_v8_combo_audit_outputs(args: argparse.Namespace) -> None:
    """Bind both saved combo OOFs to their local audit, including the fresh one."""
    audit = read_json(args.v8_combo_dir / "audit.json")
    expected_outputs = {
        "existing_oof": args.v8_combo_dir / "validation_predictions.parquet",
        "fresh_oof": args.v8_combo_dir / "fresh_audit_predictions.parquet",
    }
    expected_sources = {
        "v7_fresh": args.v7_dir / "fresh_audit_predictions.parquet",
        "v8_fresh": args.v8_dir / "fresh_audit_predictions.parquet",
    }
    if (audit.get("accepted") is not True
            or any(audit.get("output_sha256", {}).get(name) != sha256(path)
                   for name, path in expected_outputs.items())
            or any(audit.get("source_sha256", {}).get(name) != sha256(path)
                   for name, path in expected_sources.items())):
        raise ValueError("v8 combo existing/fresh OOF or audited source hash changed")


def terminal_valid_route(args: argparse.Namespace) -> dict:
    selection_path = args.guard_dir / "selected_policy.json"
    selection = read_json(selection_path)
    chosen = selection.get("selected_route")
    if chosen not in ("v7", "v8_combo", "v9b"):
        raise ValueError("Frozen selected valid route is unknown")
    reserved.require_selected_route(guard_args(args), chosen)
    if chosen == "v8_combo":
        verify_v8_combo_audit_outputs(args)
    if chosen == "v7":
        return {"chosen": chosen, "active": "v7", "terminal": "unchanged",
                "selected_policy_sha256": sha256(selection_path)}
    report_path = args.guard_dir / chosen / "guard_report.json"
    if not report_path.exists():
        raise FileNotFoundError("Chosen valid route has no terminal reserved guard")
    report = read_json(report_path)
    if report.get("passed"):
        reserved.require_route_guard(args.guard_dir, chosen)
        active = chosen
        terminal = "passed"
    else:
        verify_failed_valid_guard(args, chosen, report)
        active, terminal = "v7", "failed_retained_v7"
    return {"chosen": chosen, "active": active, "terminal": terminal,
            "selected_policy_sha256": sha256(selection_path),
            "guard_report_sha256": sha256(report_path)}


def verify_saved_movement_audit(args: argparse.Namespace, report: dict,
                                stem: str, months: tuple[int, int]) -> bool:
    movement.verify_audit_artifacts(args.movement_dir, args.cache_dir,
                                    report, stem, months)
    model_names = (("april_october_reference.cbm", "april_october_movement.txt")
                   if stem == "april_october" else
                   ("february_august_reference.cbm", "february_august_movement.txt"))
    if (report["model_sha256"]["catboost"] !=
            sha256(args.movement_dir / model_names[0])
            or report["model_sha256"]["movement"] !=
            sha256(args.movement_dir / model_names[1])):
        raise ValueError(f"{stem} movement audit model source changed")
    if stem == "february_august" and report.get("reserved_protocol_sha256") != (
            sha256(args.reserved_protocol)):
        raise ValueError("Movement February/August guard protocol changed")
    passed = (all(report["month_scores"][str(month)]["candidate_rmse_sec"]
                  < report["month_scores"][str(month)]["reference_rmse_sec"]
                  for month in months)
              and report["pooled_day_stability"]["gain_ci95_sec"][0] > 0)
    if bool(report.get("passed")) != bool(passed):
        raise ValueError(f"{stem} movement audit pass flag conflicts with metrics")
    return bool(passed)


def terminal_missing_route(args: argparse.Namespace) -> dict:
    validation_path = args.movement_dir / "validation.json"
    validation = read_json(validation_path)
    if (validation.get("reference_sha256") != movement.REFERENCE_SHA256
            or validation.get("features_manifest_sha256") !=
               sha256(args.movement_dir / "features_manifest.json")):
        raise ValueError("Original missing-route local validation changed")
    result = {"active": False, "terminal": "local_failed",
              "validation_sha256": sha256(validation_path)}
    if not validation.get("both_existing_folds_passed"):
        return result
    fresh_path = args.movement_dir / "fresh_audit.json"
    if not fresh_path.exists():
        raise FileNotFoundError("Original missing route April/October audit pending")
    fresh = read_json(fresh_path)
    fresh_passed = verify_saved_movement_audit(args, fresh,
                                               "april_october", (4, 10))
    result["fresh_audit_sha256"] = sha256(fresh_path)
    if not fresh_passed:
        result["terminal"] = "fresh_failed"
        return result
    reserved_path = args.movement_dir / "reserved_audit.json"
    if not reserved_path.exists():
        raise FileNotFoundError("Original missing route February/August guard pending")
    held = read_json(reserved_path)
    held_passed = verify_saved_movement_audit(args, held,
                                              "february_august", (2, 8))
    result["reserved_audit_sha256"] = sha256(reserved_path)
    if not held_passed:
        result["terminal"] = "reserved_failed"
        return result
    weight, _ = movement.require_all_gates(args.movement_dir, args.cache_dir)
    result.update(active=True, terminal="passed", weight=float(weight))
    return result


def source_paths(args: argparse.Namespace, valid: dict,
                 missing: dict) -> dict[str, Path]:
    paths = {
        "current_policy": args.policy_path,
        "selected_policy": args.guard_dir / "selected_policy.json",
        "reserved_protocol": args.reserved_protocol,
        "v5_oof": args.v5_oof,
        "v7_oof": args.v7_dir / "validation_predictions.parquet",
        "v7_validation": args.v7_dir / "validation.json",
        "v7_manifest": args.v7_dir / "manifest.json",
        "baseline_rows": args.cache_dir / "training_rows.parquet",
        "baseline_features": args.cache_dir / "features.parquet",
        "own_script": Path(__file__).resolve(),
    }
    if valid["chosen"] != "v7":
        chosen = valid["chosen"]
        chosen_dir = args.v8_combo_dir if chosen == "v8_combo" else args.v9b_dir
        paths.update(valid_guard_report=args.guard_dir / chosen / "guard_report.json",
                     chosen_oof=chosen_dir / "validation_predictions.parquet",
                     chosen_audit=chosen_dir / "audit.json")
        if chosen == "v8_combo":
            paths.update(chosen_fresh_oof=chosen_dir /
                         "fresh_audit_predictions.parquet",
                         v7_fresh_oof=args.v7_dir /
                         "fresh_audit_predictions.parquet",
                         v8_fresh_oof=args.v8_dir /
                         "fresh_audit_predictions.parquet")
    if missing["active"]:
        paths.update(movement_oof=args.movement_dir /
                     "validation_predictions.parquet",
                     movement_validation=args.movement_dir / "validation.json",
                     movement_fresh=args.movement_dir / "fresh_audit.json",
                     movement_reserved=args.movement_dir / "reserved_audit.json")
    return paths


def frozen_protocol(args: argparse.Namespace, valid: dict,
                    missing: dict) -> dict:
    paths = source_paths(args, valid, missing)
    value = {"spec": policy_spec(), "valid_terminal": valid,
             "missing_terminal": missing,
             "source_sha256": {name: sha256(path) for name, path in paths.items()}}
    path = args.output_dir / "protocol.json"
    if path.exists():
        if read_json(path) != value:
            raise ValueError("Frozen composition protocol or source hashes changed")
    else:
        write_new_json(path, value)
    return value


def load_common_oof(args: argparse.Namespace) -> tuple[pd.DataFrame,
                                                       np.ndarray, np.ndarray]:
    v7_path = args.v7_dir / "validation_predictions.parquet"
    v7_report = read_json(args.v7_dir / "validation.json")
    if (not v7_report.get("promoted") or
            v7_report["validation_predictions_sha256"] != sha256(v7_path)):
        raise ValueError("Accepted v7 all-finite OOF source changed")
    names = ["MVT_ID_mvt", "target", "fold", "airport", "month",
             "MVT_TIME_UTC_mvt", "a_valid", "candidate"]
    v7 = pd.read_parquet(v7_path, columns=names)
    if (len(v7) != OOF_ROWS or v7.MVT_ID_mvt.isna().any()
            or v7.MVT_ID_mvt.duplicated().any()
            or set(v7.fold.unique()) != set(FOLDS)
            or not np.isfinite(v7[["target", "candidate"]]
                               .to_numpy(dtype=float)).all()
            or (v7.candidate.to_numpy(dtype=float) < 0).any()):
        raise ValueError("v7 common OOF universe or values are invalid")
    for name, months in FOLDS.items():
        if not v7.loc[v7.fold.eq(name), "month"].isin(months).all():
            raise ValueError("v7 fold/month mapping changed")
    base = pd.read_parquet(args.cache_dir / "training_rows.parquet",
                           columns=["MVT_ID_mvt", "target", "proxy", "airport",
                                    "month", "time"])
    flags = pd.read_parquet(args.cache_dir / "features.parquet",
                            columns=["AOBT_3_flt_missing", "LOBT_flt_missing"])
    if len(base) != len(flags) or base.MVT_ID_mvt.isna().any() or base.MVT_ID_mvt.duplicated().any():
        raise ValueError("Baseline rows and missing-clock flags differ")
    base["nm_aobt_missing"] = flags.AOBT_3_flt_missing.to_numpy(dtype=bool)
    base["nm_lobt_missing"] = flags.LOBT_flt_missing.to_numpy(dtype=bool)
    base = base.loc[base.month.isin((1, 7, 11, 12)).to_numpy()
                    & np.isfinite(base.target.to_numpy(dtype=float))]
    base = aligned(base, v7.MVT_ID_mvt, "baseline finite OOF")
    if (not np.array_equal(v7.target.to_numpy(dtype=float),
                           base.target.to_numpy(dtype=float))
            or not np.array_equal(v7.month.to_numpy(), base.month.to_numpy())
            or not np.array_equal(v7.airport.astype("string").to_numpy(),
                                  base.airport.astype("string").to_numpy())
            or not np.array_equal(pd.to_datetime(v7.MVT_TIME_UTC_mvt, utc=True).to_numpy(),
                                  pd.to_datetime(base.time, utc=True).to_numpy())):
        raise ValueError("v7 labels, airport, month or timestamps differ from baseline")
    proxy = base.proxy.to_numpy(dtype=float)
    valid = np.isfinite(proxy) & (proxy >= 0) & (proxy <= 7200)
    missing = (~np.isfinite(proxy) & base.nm_aobt_missing.to_numpy(dtype=bool)
               & base.nm_lobt_missing.to_numpy(dtype=bool)
               & ~base.airport.eq("LIRF").to_numpy(dtype=bool))
    if (not np.array_equal(valid, v7.a_valid.to_numpy(dtype=bool))
            or np.any(valid & missing)):
        raise ValueError("Fixed valid and missing-clock masks differ or overlap")
    v5 = pd.read_parquet(args.v5_oof,
                         columns=["MVT_ID_mvt", "target", "fold", "airport",
                                  "month", "MVT_TIME_UTC_mvt", "a_valid", "selected"])
    v5 = aligned(v5, v7.MVT_ID_mvt, "v5 common OOF")
    same_metadata(v7, v5, "v5", airport=True, valid=True)
    if not np.array_equal(v5.selected.to_numpy(dtype=float)[missing],
                          v7.candidate.to_numpy(dtype=float)[missing]):
        raise ValueError("v5 and v7 reference predictions differ on missing route")
    v7["missing_gate"] = missing
    v7["v5"] = v5.selected.to_numpy(dtype=float)
    return v7, valid, missing


def valid_oof_component(args: argparse.Namespace, base: pd.DataFrame,
                        valid_mask: np.ndarray, status: dict) -> np.ndarray:
    reference = base.candidate.to_numpy(dtype=float)
    if status["active"] == "v7":
        return reference.copy()
    route = status["active"]
    path = ((args.v8_combo_dir if route == "v8_combo" else args.v9b_dir)
            / "validation_predictions.parquet")
    audit_path = ((args.v8_combo_dir if route == "v8_combo" else args.v9b_dir)
                  / "audit.json")
    audit = read_json(audit_path)
    selected = ("combined" if route == "v8_combo" else
                "movement_valid_candidate")
    fields = ["MVT_ID_mvt", "target", "fold", "month",
              "MVT_TIME_UTC_mvt", "a_valid", "candidate", "expert", selected]
    component = aligned(pd.read_parquet(path, columns=fields),
                        base.MVT_ID_mvt, f"{route} OOF")
    same_metadata(base, component, route, valid=True)
    if not np.array_equal(component.candidate.to_numpy(dtype=float), reference):
        raise ValueError(f"{route} OOF used a different v7 candidate")
    if route == "v8_combo":
        if (not audit.get("accepted") or audit["output_sha256"]["existing_oof"]
                != sha256(path)):
            raise ValueError("v8 combo audit or OOF hash changed")
        weight = float(audit["weight"])
    else:
        if (not audit.get("accepted") or audit["output_sha256"]["fold_oof"]
                != sha256(path)):
            raise ValueError("v9b audit or OOF hash changed")
        weight = float(audit["selected_weight"])
    expert = component.expert.to_numpy(dtype=float)
    output = component[selected].to_numpy(dtype=float)
    if (not np.isfinite(expert[valid_mask]).all()
            or not np.isfinite(output).all() or (output < 0).any()
            or not np.array_equal(output[~valid_mask], reference[~valid_mask])
            or not np.allclose(output[valid_mask], np.maximum(
                reference[valid_mask] + weight*(expert[valid_mask]
                                                - reference[valid_mask]), 0),
                               rtol=0, atol=1e-6)):
        raise ValueError(f"{route} OOF does not apply its fixed valid formula once")
    return output


def missing_oof_component(args: argparse.Namespace, base: pd.DataFrame,
                          missing_mask: np.ndarray, status: dict) -> np.ndarray:
    reference = base.candidate.to_numpy(dtype=float)
    if not status["active"]:
        return reference.copy()
    path = args.movement_dir / "validation_predictions.parquet"
    columns = ["MVT_ID_mvt", "target", "fold", "airport", "month",
               "MVT_TIME_UTC_mvt", "gate", "v5", "expert",
               "selected_candidate"]
    source = aligned(pd.read_parquet(path, columns=columns),
                     base.MVT_ID_mvt, "original movement OOF")
    same_metadata(base, source, "original movement", airport=True)
    if (not np.array_equal(source.gate.to_numpy(dtype=bool), missing_mask)
            or not np.array_equal(source.v5.to_numpy(dtype=float),
                                  base.v5.to_numpy(dtype=float))):
        raise ValueError("Original movement OOF gate or v5 base differs")
    expert = source.expert.to_numpy(dtype=float)
    output = source.selected_candidate.to_numpy(dtype=float)
    weight = float(status["weight"])
    if (not np.isfinite(expert[missing_mask]).all()
            or not np.isfinite(output).all() or (output < 0).any()
            or not np.allclose(output[missing_mask], np.maximum(
                reference[missing_mask] + weight*(expert[missing_mask]
                                                  - reference[missing_mask]), 0),
                               rtol=0, atol=1e-6)):
        raise ValueError("Original missing-clock OOF fixed formula differs")
    return output


def validate(args: argparse.Namespace) -> dict:
    check_policy(args)
    report_path = args.output_dir / "composition_report.json"
    oof_path = args.output_dir / "validation_predictions.parquet"
    if report_path.exists() or oof_path.exists():
        raise FileExistsError("Composition validation already exists; no overwrite")
    valid_status = terminal_valid_route(args)
    missing_status = terminal_missing_route(args)
    frozen_protocol(args, valid_status, missing_status)
    base, valid_mask, missing_mask = load_common_oof(args)
    valid_component = valid_oof_component(args, base, valid_mask, valid_status)
    missing_component = missing_oof_component(args, base, missing_mask,
                                               missing_status)
    reference = base.candidate.to_numpy(dtype=float)
    combined = reference.copy()
    if valid_status["active"] != "v7":
        combined[valid_mask] = valid_component[valid_mask]
    if missing_status["active"]:
        combined[missing_mask] = missing_component[missing_mask]
    outside = ~(valid_mask | missing_mask)
    if (np.any(valid_mask & missing_mask)
            or not np.array_equal(combined[outside], reference[outside])
            or not np.isfinite(combined).all() or (combined < 0).any()):
        raise ValueError("Composed OOF altered outside masks or is invalid")
    scores = {}
    for name, months in FOLDS.items():
        idx = base.fold.eq(name).to_numpy()
        part = base.loc[idx]
        if not part.month.isin(months).all():
            raise ValueError(f"Composed OOF {name} month mapping differs")
        y = part.target.to_numpy(dtype=float)
        old, new = reference[idx], combined[idx]
        bootstrap = arrival.bootstrap(part, old, new,
                                      seed=BOOTSTRAP_SEED)
        before, after = deep.rmse(y, old), deep.rmse(y, new)
        scores[name] = {"rows": len(part), "v7_rmse": before,
                        "combined_rmse": after, "bootstrap": bootstrap,
                        "passed": bool(after < before and
                                       bootstrap["gain_ci95_sec"][0] > 0)}
    passed = all(item["passed"] for item in scores.values())
    output = {"MVT_ID_mvt": base.MVT_ID_mvt, "target": base.target,
              "fold": base.fold, "month": base.month,
              "MVT_TIME_UTC_mvt": base.MVT_TIME_UTC_mvt,
              "a_valid": valid_mask, "missing_gate": missing_mask,
              "v7": reference, "valid_component": valid_component,
              "missing_component": missing_component, "combined": combined}
    report = {"passed": bool(passed), "scores": scores,
              "rows_all_finite": len(base),
              "valid_route": valid_status, "missing_route": missing_status,
              "n_valid": int(valid_mask.sum()),
              "n_missing_clock": int(missing_mask.sum()),
              "disjoint_masks_verified": True,
              "outside_masks_unchanged": True,
              "v5_v7_missing_equal": True,
              "protocol_sha256": sha256(args.output_dir / "protocol.json"),
              "source_sha256": read_json(args.output_dir / "protocol.json")
                               ["source_sha256"],
              "bootstrap_seed": BOOTSTRAP_SEED,
              "decision": "accepted" if passed else "rejected_without_retuning"}
    if passed:
        frame = pd.DataFrame(output)
        frame.to_parquet(oof_path, index=False)
        report["validation_predictions_sha256"] = sha256(oof_path)
    write_new_json(report_path, report)
    print(json.dumps({"passed": passed, "scores": scores,
                      "valid_route": valid_status["active"],
                      "missing_route": missing_status["terminal"]},
                     indent=2), flush=True)
    return report


def require_composition_gate(args: argparse.Namespace) -> tuple[dict, dict, dict]:
    check_policy(args)
    report_path = args.output_dir / "composition_report.json"
    report = read_json(report_path)
    if not report.get("passed") or report.get("decision") != "accepted":
        raise ValueError("Fixed two-fold composition gate did not pass")
    valid = terminal_valid_route(args)
    missing = terminal_missing_route(args)
    protocol_path = args.output_dir / "protocol.json"
    frozen = read_json(protocol_path)
    paths = source_paths(args, valid, missing)
    if (frozen.get("spec") != policy_spec()
            or frozen.get("valid_terminal") != valid
            or frozen.get("missing_terminal") != missing
            or frozen.get("source_sha256") !=
               {name: sha256(path) for name, path in paths.items()}
            or report.get("protocol_sha256") != sha256(protocol_path)
            or report.get("source_sha256") != frozen["source_sha256"]
            or report.get("valid_route") != valid
            or report.get("missing_route") != missing
            or report.get("rows_all_finite") != OOF_ROWS
            or not report.get("disjoint_masks_verified")
            or not report.get("outside_masks_unchanged")
            or not report.get("v5_v7_missing_equal")):
        raise ValueError("Frozen composition sources, terminal routes or coverage changed")
    for name in FOLDS:
        item = report["scores"][name]
        if (not item.get("passed") or
                not item["combined_rmse"] < item["v7_rmse"] or
                item["bootstrap"]["gain_ci95_sec"][0] <= 0):
            raise ValueError(f"Composition {name} RMSE/day-CI gate failed")
    oof_path = args.output_dir / "validation_predictions.parquet"
    if report.get("validation_predictions_sha256") != sha256(oof_path):
        raise ValueError("Accepted composition OOF changed")
    return report, valid, missing


def verify_rank_inputs(args: argparse.Namespace, output_dir: Path) -> dict:
    local = argparse.Namespace(**vars(args))
    local.output_dir = output_dir
    local.ranking_arrival_cache = args.ranking_arrival_cache
    return movement.verify_ranking_inputs(local)


def load_ranking_base(args: argparse.Namespace) -> tuple[pd.DataFrame,
                                                        np.ndarray, np.ndarray]:
    template = pd.read_parquet(args.data_dir / "submitting.parquet")
    rank_rows = pd.read_parquet(args.cache_dir / "ranking_rows.parquet",
                                columns=["MVT_ID_mvt", "proxy", "airport"])
    flags = pd.read_parquet(args.cache_dir / "ranking_features.parquet",
                            columns=["AOBT_3_flt_missing", "LOBT_flt_missing"])
    reference_path = args.v7_dir / "predictions.parquet"
    reference = pd.read_parquet(reference_path)
    manifest = read_json(args.v7_dir / "manifest.json")
    if (list(template) != RANK_COLUMNS or list(reference) != RANK_COLUMNS
            or len(template) != RANKING_ROWS or len(rank_rows) != RANKING_ROWS
            or len(flags) != RANKING_ROWS or template.MVT_ID_mvt.isna().any()
            or template.MVT_ID_mvt.duplicated().any()
            or not np.array_equal(template.MVT_ID_mvt.to_numpy(),
                                  rank_rows.MVT_ID_mvt.to_numpy())
            or not np.array_equal(template.MVT_ID_mvt.to_numpy(),
                                  reference.MVT_ID_mvt.to_numpy())):
        raise ValueError("Ranking template, cache and v7 reference ID/order differ")
    if (sha256(reference_path) != V7_RANKING_SHA256
            or manifest["predictions_sha256"] != V7_RANKING_SHA256
            or manifest["rows"] != RANKING_ROWS
            or manifest["ranking_rows_sha256"] !=
               sha256(args.cache_dir / "ranking_rows.parquet")
            or manifest["ranking_arrival_features_sha256"] !=
               sha256(args.ranking_arrival_cache)
            or manifest["model_sha256"] !=
               sha256(args.v7_dir / "full_2025.cbm")
            or manifest["validation_sha256"] !=
               sha256(args.v7_dir / "validation.json")
            or manifest["fresh_audit_sha256"] !=
               sha256(args.v7_dir / "fresh_audit.json")
            or manifest["ranking_expert_sha256"] !=
               sha256(args.v7_dir / "ranking_expert.parquet")
            or manifest["ranking_reference_sha256"] !=
               sha256(args.v6_ranking_reference)):
        raise ValueError("Accepted v7 ranking model/source/cache manifest changed")
    old = reference.TAXITIME_SEC_mvt.to_numpy(dtype=float, copy=True)
    if not np.isfinite(old).all() or (old < 0).any():
        raise ValueError("Accepted v7 ranking reference must be finite/nonnegative")
    proxy = rank_rows.proxy.to_numpy(dtype=float)
    valid = np.isfinite(proxy) & (proxy >= 0) & (proxy <= 7200)
    missing = (~np.isfinite(proxy)
               & flags.AOBT_3_flt_missing.to_numpy(dtype=bool)
               & flags.LOBT_flt_missing.to_numpy(dtype=bool)
               & ~rank_rows.airport.eq("LIRF").to_numpy(dtype=bool))
    if (int(valid.sum()) != int(manifest["valid_aobt"])
            or np.any(valid & missing)):
        raise ValueError("Ranking component masks differ or overlap")
    return reference, valid, missing


def read_component_prediction(path: Path, expert_path: Path,
                              base: pd.DataFrame, mask: np.ndarray,
                              weight: float, *, missing: bool = False
                              ) -> tuple[np.ndarray, dict]:
    frame = pd.read_parquet(path)
    raw = pd.read_parquet(expert_path)
    if (list(frame) != RANK_COLUMNS or len(frame) != len(base)
            or not np.array_equal(frame.MVT_ID_mvt.to_numpy(),
                                  base.MVT_ID_mvt.to_numpy())
            or not np.array_equal(raw.MVT_ID_mvt.to_numpy(),
                                  base.MVT_ID_mvt.to_numpy())):
        raise ValueError(f"{path}: terminal ranking component ID/order/schema differs")
    if missing:
        if (list(raw) != ["MVT_ID_mvt", "gate", "movement_expert"]
                or not np.array_equal(raw.gate.to_numpy(dtype=bool), mask)):
            raise ValueError("Missing-clock ranking expert gate/schema differs")
        expert = raw.movement_expert.to_numpy(dtype=float)
    else:
        if list(raw) != ["MVT_ID_mvt", "expert"]:
            raise ValueError("Valid-route ranking expert schema differs")
        expert = raw.expert.to_numpy(dtype=float)
    selected = frame.TAXITIME_SEC_mvt.to_numpy(dtype=float)
    old = base.TAXITIME_SEC_mvt.to_numpy(dtype=float)
    formula = np.maximum(old[mask] + weight*(expert[mask]-old[mask]), 0)
    # Expert files may store float32 while final predictions were calculated
    # from float64 before the expert cast. Source manifests are checked too.
    if (not np.isfinite(expert[mask]).all()
            or not np.isfinite(selected).all() or (selected < 0).any()
            or not np.array_equal(selected[~mask], old[~mask])
            or not np.allclose(selected[mask], formula, rtol=1e-7, atol=1e-3)):
        raise ValueError(f"{path}: fixed ranking formula or unchanged mask differs")
    return selected, {"predictions_sha256": sha256(path),
                      "expert_sha256": sha256(expert_path),
                      "rows_changed": int(np.count_nonzero(selected != old)),
                      "gate_rows": int(mask.sum())}


def check_v8_ranking_inputs(args: argparse.Namespace) -> None:
    source = read_json(args.v8_dir / "protocol.json")["source_sha256"]
    expected = {
        "baseline_ranking_rows": args.cache_dir / "ranking_rows.parquet",
        "baseline_ranking_features": args.cache_dir / "ranking_features.parquet",
        "weather": args.weather_file,
        "arrival_ranking": args.ranking_arrival_cache,
        "neighbour_ranking": args.neighbour_dir / "ranking_neighbour_features.parquet",
        "runway_ranking": args.runway_dir / "ranking_runway_arrival_features.parquet",
        "raw_ranking": args.data_dir / "ranking.parquet",
        "submission_template": args.data_dir / "submitting.parquet",
        "ranking_reference": args.v6_ranking_reference,
    }
    if any(source[name] != sha256(path) for name, path in expected.items()):
        raise ValueError("v8 frozen ranking raw/cache/weather inputs changed")


def valid_ranking_component(args: argparse.Namespace, base: pd.DataFrame,
                            mask: np.ndarray, status: dict
                            ) -> tuple[np.ndarray, dict]:
    old = base.TAXITIME_SEC_mvt.to_numpy(dtype=float, copy=True)
    if status["active"] == "v7":
        return old, {"active": "v7"}
    route = status["active"]
    root = args.v8_combo_dir if route == "v8_combo" else args.v9b_dir
    manifest_path = root / "ranking_manifest.json"
    manifest = read_json(manifest_path)
    output_path = root / "predictions.parquet"
    expert_path = root / "ranking_expert.parquet"
    guard_path = args.guard_dir / route / "guard_report.json"
    selected_path = args.guard_dir / "selected_policy.json"
    v7_manifest_path = args.v7_dir / "manifest.json"
    if route == "v8_combo":
        fit_path = args.v8_dir / "full_fit_manifest.json"
        fit = read_json(fit_path)
        model_path = args.v8_dir / "full_2025.txt"
        check_v8_ranking_inputs(args)
        if (fit.get("status") != "complete"
                or fit["model_sha256"] != sha256(model_path)
                or fit["protocol_sha256"] != sha256(args.v8_dir / "protocol.json")
                or fit["validation_sha256"] != sha256(args.v8_dir / "validation.json")
                or fit["fresh_audit_sha256"] != sha256(args.v8_dir / "fresh_audit.json")
                or fit["ranking_reference_sha256"] != sha256(args.v6_ranking_reference)
                or float(fit["weight"]) != float(manifest["selected_weight"])
                or manifest["source_v7_ranking_sha256"] != V7_RANKING_SHA256
                or manifest["v7_manifest_sha256"] != sha256(v7_manifest_path)
                or manifest["v8_full_model_sha256"] != sha256(model_path)
                or manifest["v8_full_fit_manifest_sha256"] != sha256(fit_path)
                or manifest["v8_validation_sha256"] != sha256(args.v8_dir / "validation.json")
                or manifest["v8_combo_audit_sha256"] != sha256(root / "audit.json")
                or manifest["reserved_guard_report_sha256"] != sha256(guard_path)
                or manifest["selected_policy_sha256"] != sha256(selected_path)
                or manifest["raw_expert_sha256"] != sha256(expert_path)
                or manifest["predictions_sha256"] != sha256(output_path)
                or manifest["valid_aobt"] != int(mask.sum())
                or not manifest["invalid_aobt_unchanged"]
                or not manifest["finite_nonnegative"]):
            raise ValueError("v8 combo terminal model/source/ranking manifests differ")
        weight = float(manifest["selected_weight"])
    else:
        ranking_inputs = verify_rank_inputs(args, root)
        model_source = manifest["model_source"]
        if model_source == "original_movement":
            model_path = args.movement_dir / "final_movement_only.txt"
            fit_path = args.movement_dir / "final_model.json"
            movement.require_all_gates(args.movement_dir, args.cache_dir)
        elif model_source == "v9b_full":
            model_path = root / "full_movement_valid.txt"
            fit_path = root / "full_fit_manifest.json"
        else:
            raise ValueError("v9b terminal model source is unknown")
        fit = read_json(fit_path)
        if model_source == "original_movement":
            if (fit["features_manifest_sha256"] != sha256(
                    args.movement_dir / "features_manifest.json")
                    or fit["validation_sha256"] != sha256(
                        args.movement_dir / "validation.json")
                    or fit["fresh_audit_sha256"] != sha256(
                        args.movement_dir / "fresh_audit.json")
                    or fit["reserved_audit_sha256"] != sha256(
                        args.movement_dir / "reserved_audit.json")):
                raise ValueError("Original movement full-fit sources changed")
        else:
            if (fit.get("status") != "complete"
                    or fit["audit_sha256"] != sha256(root / "audit.json")
                    or fit["reserved_guard_sha256"] != sha256(guard_path)
                    or fit["protocol_sha256"] != sha256(root / "protocol.json")
                    or fit["features_manifest_sha256"] != sha256(
                        args.movement_dir / "features_manifest.json")
                    or fit["prepared_features_sha256"] != sha256(
                        args.movement_dir / "features.parquet")
                    or fit["arrival_cache_sha256"] != sha256(args.arrival_cache)
                    or fit["v5_oof_sha256"] != sha256(args.v5_oof)):
                raise ValueError("v9b own full-fit sources changed")
        if (fit["model_sha256"] != sha256(model_path)
                or manifest["model_sha256"] != sha256(model_path)
                or manifest["model_fit_sha256"] != sha256(fit_path)
                or manifest["v7_reference_sha256"] != V7_RANKING_SHA256
                or manifest["v7_manifest_sha256"] != sha256(v7_manifest_path)
                or manifest["ranking_inputs_sha256"] != sha256(root / "ranking_inputs.json")
                or manifest["ranking_input_sha256"] !=
                   {name: item["sha256"] for name, item in
                    ranking_inputs["inputs"].items()}
                or manifest["audit_sha256"] != sha256(root / "audit.json")
                or manifest["reserved_guard_sha256"] != sha256(guard_path)
                or manifest["arrival_cache_sha256"] != sha256(args.arrival_cache)
                or manifest["v5_oof_sha256"] != sha256(args.v5_oof)
                or manifest["ranking_expert_sha256"] != sha256(expert_path)
                or manifest["predictions_sha256"] != sha256(output_path)
                or manifest["valid_aobt"] != int(mask.sum())
                or not manifest["invalid_unchanged"]
                or not manifest["finite_nonnegative"]):
            raise ValueError("v9b terminal model/source/ranking manifests differ")
        weight = float(manifest["selected_weight"])
    spec = read_json(args.guard_dir / route / "protocol.json")
    row_count = manifest["ranking_rows"] if route == "v9b" else manifest["rows"]
    if weight != float(spec["selected_weight"]) or int(row_count) != RANKING_ROWS:
        raise ValueError("Valid-route ranking weight or row count differs")
    selected, details = read_component_prediction(output_path, expert_path,
                                                  base, mask, weight)
    details.update(active=route, manifest_sha256=sha256(manifest_path),
                   model_sha256=sha256(model_path), weight=weight)
    return selected, details


def missing_ranking_component(args: argparse.Namespace, base: pd.DataFrame,
                              mask: np.ndarray, status: dict
                              ) -> tuple[np.ndarray, dict]:
    old = base.TAXITIME_SEC_mvt.to_numpy(dtype=float, copy=True)
    if not status["active"]:
        return old, {"active": False, "terminal": status["terminal"]}
    movement.require_all_gates(args.movement_dir, args.cache_dir)
    ranking_inputs = verify_rank_inputs(args, args.movement_dir)
    manifest_path = args.movement_dir / "ranking_manifest.json"
    manifest = read_json(manifest_path)
    fit_path = args.movement_dir / "final_model.json"
    fit = read_json(fit_path)
    model_path = args.movement_dir / "final_movement_only.txt"
    prediction_path = args.movement_dir / "predictions.parquet"
    expert_path = args.movement_dir / "ranking_expert.parquet"
    if (fit["model_sha256"] != sha256(model_path)
            or fit["validation_sha256"] != sha256(args.movement_dir / "validation.json")
            or fit["fresh_audit_sha256"] != sha256(args.movement_dir / "fresh_audit.json")
            or fit["reserved_audit_sha256"] != sha256(args.movement_dir / "reserved_audit.json")
            or manifest["model_sha256"] != sha256(model_path)
            or manifest["ranking_reference_sha256"] != V7_RANKING_SHA256
            or manifest["ranking_inputs_sha256"] !=
               sha256(args.movement_dir / "ranking_inputs.json")
            or manifest["ranking_input_sha256"] !=
               {name: item["sha256"] for name, item in
                ranking_inputs["inputs"].items()}
            or manifest["prediction_sha256"] != sha256(prediction_path)
            or float(manifest["selected_weight"]) != float(status["weight"])
            or int(manifest["gate_rows"]) != int(mask.sum())
            or int(manifest["ranking_rows"]) != RANKING_ROWS
            or not manifest["template_order_verified"]
            or not manifest["non_gate_unchanged"]
            or not manifest["finite_nonnegative"]):
        raise ValueError("Original missing-route terminal model/source/ranking manifests differ")
    selected, details = read_component_prediction(prediction_path, expert_path,
                                                  base, mask, float(status["weight"]),
                                                  missing=True)
    details.update(active=True, manifest_sha256=sha256(manifest_path),
                   model_sha256=sha256(model_path), weight=status["weight"])
    return selected, details


def assembly_source_files(args: argparse.Namespace, valid_status: dict,
                          missing_status: dict) -> dict[str, Path]:
    """Enumerate every ranking input before loading any prediction values."""
    files = {
        "composition_report": args.output_dir / "composition_report.json",
        "composition_oof": args.output_dir / "validation_predictions.parquet",
        "composition_protocol": args.output_dir / "protocol.json",
        "current_policy": args.policy_path,
        "selected_policy": args.guard_dir / "selected_policy.json",
        "template": args.data_dir / "submitting.parquet",
        "raw_ranking": args.data_dir / "ranking.parquet",
        "baseline_ranking_rows": args.cache_dir / "ranking_rows.parquet",
        "baseline_ranking_features": args.cache_dir / "ranking_features.parquet",
        "weather": args.weather_file,
        "ranking_arrival_cache": args.ranking_arrival_cache,
        "ranking_neighbour_cache": args.neighbour_dir / "ranking_neighbour_features.parquet",
        "ranking_runway_cache": args.runway_dir / "ranking_runway_arrival_features.parquet",
        "v6_ranking_reference": args.v6_ranking_reference,
        "v7_ranking": args.v7_dir / "predictions.parquet",
        "v7_expert": args.v7_dir / "ranking_expert.parquet",
        "v7_model": args.v7_dir / "full_2025.cbm",
        "v7_manifest": args.v7_dir / "manifest.json",
        "v7_protocol": args.v7_dir / "protocol.json",
        "v7_validation": args.v7_dir / "validation.json",
        "v7_fresh_audit": args.v7_dir / "fresh_audit.json",
    }
    if valid_status["chosen"] == "v8_combo":
        # Keep the local fresh architecture audit sealed even if the reserved
        # guard later retained v7 for ranking. These paths are rehashed before
        # and after ranking reads and again before the final manifest.
        files.update(v8_combo_existing_oof=args.v8_combo_dir /
                     "validation_predictions.parquet",
                     v8_combo_fresh_oof=args.v8_combo_dir /
                     "fresh_audit_predictions.parquet",
                     v8_combo_audit=args.v8_combo_dir / "audit.json",
                     v7_fresh_oof=args.v7_dir /
                     "fresh_audit_predictions.parquet",
                     v8_fresh_oof=args.v8_dir /
                     "fresh_audit_predictions.parquet")
    if valid_status["active"] != "v7":
        route = valid_status["active"]
        root = args.v8_combo_dir if route == "v8_combo" else args.v9b_dir
        files.update(valid_predictions=root / "predictions.parquet",
                     valid_expert=root / "ranking_expert.parquet",
                     valid_manifest=root / "ranking_manifest.json",
                     valid_oof=root / "validation_predictions.parquet",
                     valid_audit=root / "audit.json",
                     valid_protocol=root / "protocol.json",
                     valid_guard=args.guard_dir / route / "guard_report.json")
        if route == "v8_combo":
            files.update(valid_model=args.v8_dir / "full_2025.txt",
                         valid_fit=args.v8_dir / "full_fit_manifest.json",
                         valid_source_protocol=args.v8_dir / "protocol.json",
                         valid_source_validation=args.v8_dir / "validation.json",
                         valid_source_fresh=args.v8_dir / "fresh_audit.json")
        else:
            manifest = read_json(root / "ranking_manifest.json")
            if manifest["model_source"] == "original_movement":
                files.update(valid_model=args.movement_dir / "final_movement_only.txt",
                             valid_fit=args.movement_dir / "final_model.json")
            elif manifest["model_source"] == "v9b_full":
                files.update(valid_model=root / "full_movement_valid.txt",
                             valid_fit=root / "full_fit_manifest.json")
            else:
                raise ValueError("Unknown terminal v9b model source")
            files.update(valid_ranking_inputs=root / "ranking_inputs.json",
                         valid_movement_features=args.movement_dir / "features.parquet",
                         valid_movement_manifest=args.movement_dir / "features_manifest.json",
                         valid_training_arrival_cache=args.arrival_cache)
    if missing_status["active"]:
        files.update(missing_predictions=args.movement_dir / "predictions.parquet",
                     missing_expert=args.movement_dir / "ranking_expert.parquet",
                     missing_manifest=args.movement_dir / "ranking_manifest.json",
                     missing_model=args.movement_dir / "final_movement_only.txt",
                     missing_fit=args.movement_dir / "final_model.json",
                     missing_ranking_inputs=args.movement_dir / "ranking_inputs.json",
                     missing_features=args.movement_dir / "features.parquet",
                     missing_features_manifest=args.movement_dir / "features_manifest.json",
                     missing_validation=args.movement_dir / "validation.json",
                     missing_fresh=args.movement_dir / "fresh_audit.json",
                     missing_reserved=args.movement_dir / "reserved_audit.json")
    return files


def verify_source_snapshot(files: dict[str, Path], expected: dict[str, str]) -> None:
    actual = {name: sha256(path) for name, path in files.items()}
    if actual != expected:
        changed = [name for name in files if actual.get(name) != expected.get(name)]
        raise ValueError(f"Ranking source changed during assembly: {changed}")


def assemble(args: argparse.Namespace) -> dict:
    report, valid_status, missing_status = require_composition_gate(args)
    output_path = args.output_dir / "predictions.parquet"
    manifest_path = args.output_dir / "ranking_manifest.json"
    sources_path = args.output_dir / "ranking_sources.json"
    temp = output_path.with_suffix(".parquet.tmp")
    if (output_path.exists() or manifest_path.exists() or sources_path.exists()
            or temp.exists()):
        raise FileExistsError("Current-candidate ranking assembly already exists; no overwrite")
    source_files = assembly_source_files(args, valid_status, missing_status)
    source_hashes = {name: sha256(path) for name, path in source_files.items()}
    confirmed_report, confirmed_valid, confirmed_missing = require_composition_gate(args)
    if (confirmed_report != report or confirmed_valid != valid_status
            or confirmed_missing != missing_status):
        raise ValueError("Composition gate changed before ranking inputs were read")
    verify_source_snapshot(source_files, source_hashes)
    reference, valid_mask, missing_mask = load_ranking_base(args)
    valid_prediction, valid_info = valid_ranking_component(
        args, reference, valid_mask, valid_status)
    missing_prediction, missing_info = missing_ranking_component(
        args, reference, missing_mask, missing_status)
    old = reference.TAXITIME_SEC_mvt.to_numpy(dtype=float, copy=True)
    final = old.copy()
    if valid_status["active"] != "v7":
        final[valid_mask] = valid_prediction[valid_mask]
    if missing_status["active"]:
        final[missing_mask] = missing_prediction[missing_mask]
    other = ~(valid_mask | missing_mask)
    if (np.any(valid_mask & missing_mask)
            or not np.array_equal(final[other], old[other])
            or not np.isfinite(final).all() or (final < 0).any()):
        raise ValueError("Final candidate changed outside disjoint masks or is invalid")
    verify_source_snapshot(source_files, source_hashes)
    sources = {"files_sha256": source_hashes,
               "valid_route": valid_status, "missing_route": missing_status,
               "valid_component": valid_info, "missing_component": missing_info}
    write_new_json(sources_path, sources)
    pd.DataFrame({"MVT_ID_mvt": reference.MVT_ID_mvt.to_numpy(copy=True),
                  "TAXITIME_SEC_mvt": final}).to_parquet(temp, index=False)
    if output_path.exists():
        raise FileExistsError("Current-candidate ranking output appeared during assembly")
    temp.rename(output_path)
    readback = pd.read_parquet(output_path)
    if (list(readback) != RANK_COLUMNS or len(readback) != RANKING_ROWS
            or not np.array_equal(readback.MVT_ID_mvt.to_numpy(),
                                  reference.MVT_ID_mvt.to_numpy())
            or not np.array_equal(readback.TAXITIME_SEC_mvt.to_numpy(dtype=float),
                                  final)):
        raise ValueError("Final candidate readback differs from exact template/order")
    verify_source_snapshot(source_files, source_hashes)
    manifest = {"rows": len(final), "valid_gate_rows": int(valid_mask.sum()),
                "missing_gate_rows": int(missing_mask.sum()),
                "valid_route": valid_status, "missing_route": missing_status,
                "component_oof_gate_passed": bool(report["passed"]),
                "composition_report_sha256": sha256(args.output_dir /
                                                    "composition_report.json"),
                "ranking_sources_sha256": sha256(sources_path),
                "template_order_verified": True,
                "outside_masks_unchanged": True,
                "finite_nonnegative": True,
                "predictions_sha256": sha256(output_path),
                "prediction_bytes": output_path.stat().st_size,
                "uploaded": False}
    write_new_json(manifest_path, manifest)
    print(json.dumps({"candidate": str(output_path),
                      "rows": len(final), "sha256": manifest["predictions_sha256"],
                      "valid_route": valid_status["active"],
                      "missing_route": missing_status["terminal"]},
                     indent=2), flush=True)
    return manifest


def contract() -> dict:
    return {"modes": ["contract", "validate", "assemble"],
            "validation_sources": ["prepublished current policy",
                                   "frozen selected valid policy and terminal reserved guard",
                                   "original movement full local/fresh/reserved guards",
                                   "v5/v7/selected-valid/movement all-finite OOF",
                                   "baseline rows and missing-clock flags"],
            "validation_output": ["composition_report.json",
                                  "validation_predictions.parquet only if both folds pass"],
            "assembly_sources": ["accepted composition report and OOF",
                                 "v7 ranking file/manifest/model",
                                 "selected guarded valid-route ranking file/expert/model/manifests if active",
                                 "original guarded missing-route ranking file/expert/model/manifests if active",
                                 "raw ranking/cache/template IDs and input hashes"],
            "assembly_output": ["internal predictions.parquet",
                                "ranking_sources.json", "ranking_manifest.json"],
            "never": ["model fitting", "feature building", "ranking label reading",
                      "leaderboard calls", "submission", "file overwrite"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("contract", "validate", "assemble"),
                        default="contract")
    parser.add_argument("--policy-path", type=Path,
                        default=Path("reports/current_candidate_policy_protocol.json"))
    parser.add_argument("--reserved-protocol", type=Path,
                        default=Path("reports/reserved_guard_protocol.json"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--cache-dir", type=Path,
                        default=Path("artifacts/baseline"))
    parser.add_argument("--v5-oof", type=Path,
                        default=Path("artifacts/v5-ensemble/validation_predictions.parquet"))
    parser.add_argument("--v7-dir", type=Path,
                        default=Path("artifacts/v7-runway-traffic"))
    parser.add_argument("--v7-reserved-dir", type=Path,
                        default=Path("artifacts/v8-reserved-v7"))
    parser.add_argument("--v8-dir", type=Path,
                        default=Path("artifacts/v8-lightgbm"))
    parser.add_argument("--v8-combo-dir", type=Path,
                        default=Path("artifacts/v8-combo"))
    parser.add_argument("--v9b-dir", type=Path,
                        default=Path("artifacts/v9b-movement-valid"))
    parser.add_argument("--movement-dir", type=Path,
                        default=Path("artifacts/v6-movement-only"))
    parser.add_argument("--guard-dir", type=Path,
                        default=Path("artifacts/reserved-valid-guard"))
    parser.add_argument("--neighbour-dir", type=Path,
                        default=Path("artifacts/v6-neighbour"))
    parser.add_argument("--runway-dir", type=Path,
                        default=Path("artifacts/v6-runway-arrival"))
    parser.add_argument("--weather-file", type=Path,
                        default=Path("data/external/weather.parquet"))
    parser.add_argument("--arrival-cache", type=Path,
                        default=Path("artifacts/v5-arrival-clean/training_arrival_features.parquet"))
    parser.add_argument("--ranking-arrival-cache", type=Path,
                        default=Path("artifacts/v5-arrival-clean/ranking_arrival_features.parquet"))
    parser.add_argument("--v6-ranking-reference", type=Path,
                        default=Path("submissions/merry-mushroom_v6.parquet"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("artifacts/current-candidate"))
    args = parser.parse_args()
    if args.mode == "contract":
        print(json.dumps(contract(), indent=2))
    elif args.mode == "validate":
        validate(args)
    else:
        assemble(args)


if __name__ == "__main__":
    main()
