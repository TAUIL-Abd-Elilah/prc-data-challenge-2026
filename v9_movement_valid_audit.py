"""Independent valid-AOBT routing audit for saved movement-only direct models.

The movement models and their training protocol are unchanged. This script can
save held-out predictions from those models, then compare a fixed coarse blend
against the locally promoted v7 candidate. No model fit or leaderboard read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

import deep_arrival_expert as arrival
import deep_timestamp_expert as deep


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
    value = {
        "purpose": "Assess saved movement-only direct expert on valid-AOBT departures",
        "training": "No fitting or hyperparameter changes; reuse saved movement-only complement-fold and April/October models",
        "reference": "Locally promoted clipped v7 candidate on all finite 2025 OOF rows",
        "source_sha256": {
            "movement_protocol": sha256(args.movement_dir / "protocol.json"),
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
            "movement_model": "saved movement-only April/October refit trained and early-stopped without those months",
            "comparator": "v7 fixed fresh blend of depth10 ARR prior and v7 fresh expert",
            "weight": "same January/July-selected movement weight",
            "gate": "exact valid-AOBT IDs, both April and October point RMSE gains, and pooled UTC-day bootstrap 95% lower gain >0",
        },
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
    manifest = json.loads((args.movement_dir / "features_manifest.json")
                          .read_text(encoding="utf-8"))
    ids = pd.read_parquet(args.movement_dir / "row_ids.parquet",
                          columns=["MVT_ID_mvt"])
    rows = pd.read_parquet(args.cache_dir / "training_rows.parquet",
                           columns=["MVT_ID_mvt", "target", "proxy", "month",
                                    "time"])
    features = pd.read_parquet(args.movement_dir / "features.parquet")
    if (len(features) != len(rows) or len(ids) != len(rows)
            or list(features) != manifest["features"]
            or not ids.MVT_ID_mvt.equals(rows.MVT_ID_mvt)
            or rows.MVT_ID_mvt.isna().any()
            or rows.MVT_ID_mvt.duplicated().any()
            or sha256(args.cache_dir / "training_rows.parquet") !=
               manifest["baseline_rows_sha256"]):
        raise ValueError("Prepared movement matrix differs from exact baseline rows")
    return rows, features, manifest


def predict(args: argparse.Namespace, name: str, months: tuple[int, int]) -> None:
    protocol(args)
    rows, features, manifest = load_movement_features(args)
    proxy = rows.proxy.to_numpy(dtype=float)
    y = rows.target.to_numpy(dtype=float)
    held = rows.month.isin(months).to_numpy()
    valid = held & np.isfinite(y) & np.isfinite(proxy) & (proxy >= 0) & (proxy <= 7200)
    model_path = args.movement_dir / ("april_october_movement.txt" if
                                       name == "fresh_apr_oct" else f"{name}.txt")
    if name == "fresh_apr_oct":
        fresh = json.loads((args.movement_dir / "fresh_audit.json")
                           .read_text(encoding="utf-8"))
        if (fresh["heldout_months"] != list(months)
                or fresh["model_sha256"]["movement"] != sha256(model_path)
                or fresh["features_manifest_sha256"] !=
                   sha256(args.movement_dir / "features_manifest.json")):
            raise ValueError("Saved April/October model or heldout scope differs")
        rounds = int(fresh["movement_training"]["best_round"])
        fit_hash = sha256(args.movement_dir / "fresh_audit.json")
    else:
        fit_path = args.movement_dir / f"{name}_fit.json"
        fit = json.loads(fit_path.read_text(encoding="utf-8"))
        if (fit["heldout_months"] != list(months)
                or fit["features_manifest_sha256"] !=
                   sha256(args.movement_dir / "features_manifest.json")):
            raise ValueError("Saved movement fold or feature manifest differs")
        rounds = int(fit["best_round"])
        fit_hash = sha256(fit_path)
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
        "model_sha256": sha256(model_path), "fit_report_sha256": fit_hash,
        "features_manifest_sha256": sha256(args.movement_dir /
                                            "features_manifest.json"),
        "prepared_features_sha256": sha256(args.movement_dir /
                                            "features.parquet"),
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
    fit_path = (args.movement_dir / "fresh_audit.json" if name == "fresh_apr_oct"
                else args.movement_dir / f"{name}_fit.json")
    if (manifest["months"] != list(months)
            or manifest["output_sha256"] != sha256(path)
            or manifest["fit_report_sha256"] != sha256(fit_path)
            or manifest["features_manifest_sha256"] !=
               sha256(args.movement_dir / "features_manifest.json")
            or manifest["prepared_features_sha256"] !=
               sha256(args.movement_dir / "features.parquet")
            or manifest["model_sha256"] != sha256(args.movement_dir /
                ("april_october_movement.txt" if name == "fresh_apr_oct"
                 else f"{name}.txt"))):
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
    fresh = pd.read_parquet(args.v7_dir / "fresh_audit_predictions.parquet",
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


def audit(args: argparse.Namespace) -> None:
    fixed = protocol(args)
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
    fold_report, fold_rows, selected = fold_audit(args, v7)
    fresh_report, fresh_rows = fresh_audit(args, selected,
                                          v7_report["selected_weight"])
    fold_path = args.output_dir / "validation_predictions.parquet"
    fold_rows[["MVT_ID_mvt", "target", "fold", "month", "MVT_TIME_UTC_mvt",
               "a_valid", "candidate", "expert",
               "movement_valid_candidate"]].to_parquet(fold_path, index=False)
    fresh_path = args.output_dir / "fresh_audit_predictions.parquet"
    fresh_rows[["MVT_ID_mvt", "target", "MVT_TIME_UTC_mvt", "candidate",
                "expert", "movement_valid_candidate"]].to_parquet(
                    fresh_path, index=False)
    report = {"protocol": fixed, "selected_weight": selected,
              "folds": fold_report, "fresh_audit": fresh_report,
              "accepted": (all(part["passed"] for part in fold_report.values())
                           and fresh_report["passed"]),
              "source_sha256": {
                  "v7_validation": sha256(args.v7_dir / "validation.json"),
                  "v7_oof": sha256(v7_path),
                  "v7_fresh": sha256(args.v7_dir / "fresh_audit_predictions.parquet"),
                  "movement_seasonal": sha256(args.output_dir /
                                               "seasonal_jan_jul_valid_oof.parquet"),
                  "movement_forward": sha256(args.output_dir /
                                              "forward_nov_dec_valid_oof.parquet"),
                  "movement_fresh": sha256(args.output_dir /
                                            "fresh_apr_oct_valid_oof.parquet")},
              "output_sha256": {"fold_oof": sha256(fold_path),
                                "fresh_oof": sha256(fresh_path)}}
    write_json(args.output_dir / "audit.json", report)
    print(json.dumps({"selected_weight": selected, "folds": fold_report,
                      "fresh_audit": fresh_report,
                      "accepted": report["accepted"]}, indent=2))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("prepare", "predict-fold", "predict-fresh",
                                      "audit"), default="prepare")
    p.add_argument("--fold", choices=tuple(FOLDS))
    p.add_argument("--movement-dir", type=Path,
                   default=Path("artifacts/v6-movement-only"))
    p.add_argument("--v7-dir", type=Path,
                   default=Path("artifacts/v7-runway-traffic"))
    p.add_argument("--cache-dir", type=Path, default=Path("artifacts/baseline"))
    p.add_argument("--output-dir", type=Path,
                   default=Path("artifacts/v9-movement-valid"))
    args = p.parse_args()
    if args.mode == "prepare":
        value = protocol(args)
        print(json.dumps({"protocol_path": str(args.output_dir / "protocol.json"),
                          "source_sha256": value["source_sha256"]}, indent=2))
    elif args.mode == "predict-fold":
        if not args.fold:
            p.error("--predict-fold requires --fold")
        predict(args, args.fold, FOLDS[args.fold])
    elif args.mode == "predict-fresh":
        predict(args, "fresh_apr_oct", FRESH_MONTHS)
    else:
        audit(args)


if __name__ == "__main__":
    main()
