"""Evaluate and assemble the fixed v5 expert sequence from saved predictions.

No model is trained here. The legacy arrival correction is scored as a diagnostic
only: its forward stack fit depends on seasonal v4 OOF predictions whose base
models saw November/December labels. A clean complementary-fold expert needs
its own validation before this script can promote an arrival correction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


FOLDS = ("seasonal_jan_jul", "forward_nov_dec")
STAGES = ("v4", "deep", "deep_missing", "deep_missing_arrival_diagnostic")
EXPECTED_N = 672_428
SEED = 20261002


def rmse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(y - p))))


def read_unique(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    frame = pd.read_parquet(path, columns=columns)
    if frame.MVT_ID_mvt.isna().any() or frame.MVT_ID_mvt.duplicated().any():
        raise ValueError(f"Movement IDs are missing or duplicated: {path}")
    return frame


def align(base_ids: pd.Index, source: pd.DataFrame, *, name: str,
          full: bool) -> pd.DataFrame:
    index = pd.Index(source.MVT_ID_mvt)
    if not index.isin(base_ids).all():
        raise ValueError(f"{name} contains movement IDs outside the base")
    if full and (len(index) != len(base_ids) or not base_ids.isin(index).all()):
        raise ValueError(f"{name} does not cover every base movement ID")
    return source.set_index("MVT_ID_mvt").reindex(base_ids)


def fold_scores(frame: pd.DataFrame, predictions: dict[str, np.ndarray],
                mask: np.ndarray) -> dict:
    y = frame.target.to_numpy(dtype=float)[mask]
    return {"n": int(mask.sum()), **{name: rmse(y, p[mask])
                                    for name, p in predictions.items()}}


def day_bootstrap(frame: pd.DataFrame, predictions: dict[str, np.ndarray],
                  mask: np.ndarray, *, seed: int, repeats: int) -> dict:
    """Paired RMSE gains after resampling UTC calendar days, not individual rows."""
    dates = pd.to_datetime(frame.loc[mask, "MVT_TIME_UTC_mvt"], utc=True,
                           errors="coerce").dt.floor("D")
    if dates.isna().any():
        raise ValueError("Every OOF row needs a UTC movement date")
    groups, unique_days = pd.factorize(dates, sort=True)
    n_days = len(unique_days)
    n = np.bincount(groups, minlength=n_days).astype(np.float64)
    y = frame.target.to_numpy(dtype=float)[mask]
    sse = {name: np.bincount(groups, weights=np.square(y - p[mask]),
                            minlength=n_days) for name, p in predictions.items()}
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, n_days, size=(repeats, n_days), dtype=np.int32)
    sampled_n = n[sampled].sum(axis=1)
    sampled_rmse = {name: np.sqrt(values[sampled].sum(axis=1) / sampled_n)
                    for name, values in sse.items()}
    comparisons = [("v4", "deep"), ("deep", "deep_missing"),
                   ("deep_missing", "deep_missing_arrival_diagnostic"),
                   ("v4", "deep_missing")]
    if "deep_missing_clean_arrival" in predictions:
        comparisons.extend((("deep_missing", "deep_missing_clean_arrival"),
                            ("v4", "deep_missing_clean_arrival")))
    return {"days": n_days, "repeats": repeats,
            "comparisons": {f"{old}_to_{new}": {
                "observed_gain_sec": rmse(y, predictions[old][mask])
                                     - rmse(y, predictions[new][mask]),
                "gain_ci95_sec": [float(x) for x in np.quantile(
                    sampled_rmse[old] - sampled_rmse[new], [.025, .975])],
                "fraction_positive": float(np.mean(
                    sampled_rmse[old] > sampled_rmse[new])),
            } for old, new in comparisons}}


def score_table(frame: pd.DataFrame, predictions: dict[str, np.ndarray],
                group_columns: list[str]) -> list[dict]:
    rows = []
    for key, group in frame.groupby(group_columns, sort=True, observed=True):
        if not isinstance(key, tuple):
            key = (key,)
        index = group.index.to_numpy()
        y = group.target.to_numpy(dtype=float)
        row = {column: (int(value) if column == "month" else str(value))
               for column, value in zip(group_columns, key)}
        row["n"] = len(group)
        row.update({name + "_rmse_sec": rmse(y, pred[index])
                    for name, pred in predictions.items()})
        rows.append(row)
    return rows


def evaluate(args: argparse.Namespace) -> dict:
    base = read_unique(args.v4_oof,
                       ["MVT_ID_mvt", "target", "fold", "a_valid", "selected",
                        "airport", "month", "MVT_TIME_UTC_mvt"])
    if len(base) != EXPECTED_N or set(base.fold) != set(FOLDS):
        raise ValueError("The clipped frozen v4 validation set changed")
    base_ids = pd.Index(base.MVT_ID_mvt)
    y = base.target.to_numpy(dtype=float)
    valid = base.a_valid.to_numpy(dtype=bool)
    v4 = base.selected.to_numpy(dtype=float)
    if (not np.isfinite(y).all() or not np.isfinite(v4).all()
            or np.any(v4 < 0)):
        raise ValueError("The clipped v4 labels or output policy changed")

    deep_parts = []
    for fold in FOLDS:
        part = read_unique(args.deep_dir / f"{fold}_oof.parquet",
                           ["MVT_ID_mvt", "expert"])
        part["expert_fold"] = fold
        deep_parts.append(part)
    deep_source = pd.concat(deep_parts, ignore_index=True)
    if deep_source.MVT_ID_mvt.duplicated().any():
        raise ValueError("Deep OOF repeats a movement ID across folds")
    deep_aligned = align(base_ids, deep_source, name="deep OOF", full=False)
    deep_expert = deep_aligned.expert.to_numpy(dtype=float)
    has_deep = np.isfinite(deep_expert)
    if not np.array_equal(has_deep, valid):
        raise ValueError("Deep expert must cover exactly the valid-AOBT OOF rows")
    if not np.array_equal(deep_aligned.expert_fold.to_numpy()[valid],
                          base.fold.to_numpy()[valid]):
        raise ValueError("Deep expert OOF fold does not match the base fold")
    deep = v4.copy()
    deep[valid] = np.maximum(.5 * v4[valid] + .5 * deep_expert[valid], 0)

    missing_source = read_unique(args.missing_dir / "validation_predictions.parquet",
                                 ["MVT_ID_mvt", "target", "fold", "a_valid",
                                  "airport", "policy_v4", "policy_direct_half"])
    missing = align(base_ids, missing_source, name="missing OOF", full=True)
    if not np.array_equal(missing.target.to_numpy(dtype=float), y):
        raise ValueError("Missing expert validation labels differ from v4")
    if not np.array_equal(missing.fold.to_numpy(), base.fold.to_numpy()):
        raise ValueError("Missing expert validation fold differs from v4")
    if not np.array_equal(missing.a_valid.to_numpy(dtype=bool), valid):
        raise ValueError("Missing expert AOBT gate differs from v4")
    if not np.array_equal(missing.airport.to_numpy(), base.airport.to_numpy()):
        raise ValueError("Missing expert airports differ from v4")
    if not np.allclose(missing.policy_v4.to_numpy(dtype=float), v4,
                       rtol=0, atol=1e-9):
        raise ValueError("Missing expert's v4 base differs from clipped v4")
    missing_delta = (missing.policy_direct_half.to_numpy(dtype=float)
                     - missing.policy_v4.to_numpy(dtype=float))
    if (not np.isfinite(missing_delta).all()
            or np.any(missing_delta[valid] != 0)
            or np.any(missing_delta[base.airport.eq("LIRF").to_numpy()] != 0)):
        raise ValueError("Missing expert changed a valid-AOBT or LIRF row")
    deep_missing = np.maximum(deep + missing_delta, 0)
    if not np.array_equal(deep_missing[valid], deep[valid]):
        raise ValueError("Deep and missing gates were expected to be disjoint")

    arrival_source = read_unique(args.arrival_dir / "traffic_correction_oof.parquet",
                                 ["MVT_ID_mvt", "target", "fold", "a_valid",
                                  "arrival_correction"])
    arrival = align(base_ids, arrival_source, name="legacy ARR OOF", full=True)
    if (not np.array_equal(arrival.target.to_numpy(dtype=float), y)
            or not np.array_equal(arrival.fold.to_numpy(), base.fold.to_numpy())
            or not np.array_equal(arrival.a_valid.to_numpy(dtype=bool), valid)):
        raise ValueError("ARR validation labels, fold, or AOBT gate differ from v4")
    correction = arrival.arrival_correction.to_numpy(dtype=float)
    if not np.isfinite(correction[valid]).all():
        raise ValueError("ARR correction must cover every valid-AOBT OOF row")
    with_arrival = deep_missing.copy()
    with_arrival[valid] = np.maximum(
        deep_missing[valid] + .5 * correction[valid], 0)
    if not np.array_equal(with_arrival[~valid], deep_missing[~valid]):
        raise ValueError("ARR diagnostic changed an invalid-AOBT row")
    predictions = dict(zip(STAGES, (v4, deep, deep_missing, with_arrival)))
    clean_weight = 0.0
    clean_training_passes = False
    clean_available = all((args.clean_arrival_dir / f"{fold}_oof.parquet").exists()
                          for fold in FOLDS) and (args.clean_arrival_dir / "validation.json").exists()
    if clean_available:
        clean_report = json.loads((args.clean_arrival_dir / "validation.json").read_text(
            encoding="utf-8"))
        clean_weight = float(clean_report["selected_weight"])
        clean_training_passes = bool(clean_report["gain_both_folds"])
        if clean_weight not in (0.0, .1, .25, .5, 1.0):
            raise ValueError("Clean ARR weight was not one of the predeclared values")
        clean_parts = []
        for fold in FOLDS:
            part = read_unique(args.clean_arrival_dir / f"{fold}_oof.parquet",
                               ["MVT_ID_mvt", "target", "arrival_direct_expert"])
            part["expert_fold"] = fold
            clean_parts.append(part)
        clean_source = pd.concat(clean_parts, ignore_index=True)
        if clean_source.MVT_ID_mvt.duplicated().any():
            raise ValueError("Clean ARR OOF repeats a movement ID across folds")
        clean_aligned = align(base_ids, clean_source, name="clean ARR OOF", full=False)
        clean_expert = clean_aligned.arrival_direct_expert.to_numpy(dtype=float)
        clean_gate = np.isfinite(clean_expert)
        if not np.array_equal(clean_gate, valid):
            raise ValueError("Clean ARR expert must cover exactly valid-AOBT OOF rows")
        if not np.array_equal(clean_aligned.expert_fold.to_numpy()[valid],
                              base.fold.to_numpy()[valid]):
            raise ValueError("Clean ARR expert OOF fold differs from v4")
        if not np.array_equal(clean_aligned.target.to_numpy(dtype=float)[valid], y[valid]):
            raise ValueError("Clean ARR expert labels differ from v4")
        clean_pred = deep_missing.copy()
        clean_pred[valid] = np.maximum(
            deep_missing[valid] + clean_weight * (
                clean_expert[valid] - deep_missing[valid]), 0)
        predictions["deep_missing_clean_arrival"] = clean_pred
    by_fold = {}
    for k, fold in enumerate(FOLDS):
        mask = base.fold.eq(fold).to_numpy()
        by_fold[fold] = {"scores_all_finite": fold_scores(base, predictions, mask),
                         "valid_aobt_n": int(np.sum(mask & valid)),
                         "invalid_aobt_n": int(np.sum(mask & ~valid)),
                         "missing_changed_n": int(np.count_nonzero(
                             missing_delta[mask])),
                         "day_bootstrap": day_bootstrap(
                             base, predictions, mask, seed=args.seed + k,
                             repeats=args.repeats)}
    deep_passes = all(by_fold[f]["scores_all_finite"]["deep"] <
                      by_fold[f]["scores_all_finite"]["v4"] for f in FOLDS)
    missing_passes = all(by_fold[f]["scores_all_finite"]["deep_missing"] <
                         by_fold[f]["scores_all_finite"]["deep"] for f in FOLDS)
    arrival_passes = all(
        by_fold[f]["scores_all_finite"]["deep_missing_arrival_diagnostic"] <
        by_fold[f]["scores_all_finite"]["deep_missing"] for f in FOLDS)
    clean_passes = clean_available and clean_training_passes and clean_weight > 0 and all(
        by_fold[f]["scores_all_finite"]["deep_missing_clean_arrival"] <
        by_fold[f]["scores_all_finite"]["deep_missing"] for f in FOLDS)
    # This ARR stack is informative but not eligible for a ranking submission.
    selected_stage = "deep_missing" if deep_passes and missing_passes else (
        "deep" if deep_passes else "v4")
    if selected_stage == "deep_missing" and clean_passes:
        selected_stage = "deep_missing_clean_arrival"
    selected = predictions[selected_stage]
    if not np.isfinite(selected).all() or np.any(selected < 0):
        raise ValueError("Selected OOF output must be finite and nonnegative")
    report = {
        "policy": "clipped v4 -> .5 deep blend on valid AOBT -> .5 ordinary missing direct blend on its disjoint gate -> clean ARR at its fixed seasonal weight if it improves both folds; legacy ARR remains diagnostic only",
        "reference_file": str(args.v4_oof.resolve()),
        "validation_rows": len(base),
        "all_finite_scores": fold_scores(base, predictions,
                                         np.ones(len(base), dtype=bool)),
        "folds": by_fold,
        "monthly": score_table(base, predictions, ["fold", "month"]),
        "airport": score_table(base, predictions, ["fold", "airport"]),
        "gates": {"deep_valid_aobt_rows": int(valid.sum()),
                  "missing_delta_nonzero_rows": int(np.count_nonzero(missing_delta)),
                  "arrival_valid_aobt_rows": int(valid.sum()),
                  "clean_arrival_valid_aobt_rows": int(valid.sum()) if clean_available else 0},
        "selection": {"deep_improves_both_folds": deep_passes,
                      "missing_improves_both_folds_after_deep": missing_passes,
                      "legacy_arrival_improves_both_folds_diagnostic": arrival_passes,
                      "legacy_arrival_promoted": False,
                      "clean_arrival_oof_available": clean_available,
                      "clean_arrival_seasonal_weight": clean_weight,
                      "clean_arrival_original_gain_both_folds": clean_training_passes,
                      "clean_arrival_improves_deep_missing_both_folds": clean_passes,
                      "clean_arrival_promoted": selected_stage == "deep_missing_clean_arrival",
                      "selected_stage": selected_stage,
                      "reason_arrival_excluded": (
                          "The forward ARR stack was trained on Jan/Jul v4 OOF residuals; "
                          "the Jan/Jul v4 base models include Nov/Dec training labels, "
                          "so the ARR forward comparison is indirectly contaminated.")},
        "validation_limit": (
            "Both month pairs have been used for local model comparison. These are "
            "not untouched final estimates and no ranking outcomes were read."),
        "bootstrap": {"unit": "UTC calendar day within each fold",
                      "seed": args.seed, "repeats": args.repeats},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"MVT_ID_mvt": base.MVT_ID_mvt, "target": y,
                  "fold": base.fold, "airport": base.airport, "month": base.month,
                  "MVT_TIME_UTC_mvt": base.MVT_TIME_UTC_mvt,
                  "a_valid": valid, **predictions,
                  "selected": selected}).to_parquet(
                      args.output_dir / "validation_predictions.parquet", index=False)
    (args.output_dir / "validation.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    return report


def ranking(args: argparse.Namespace) -> dict:
    selection = json.loads((args.output_dir / "validation.json").read_text(
        encoding="utf-8"))["selection"]
    stage = selection["selected_stage"]
    if stage not in ("deep_missing_clean_arrival", "deep_missing", "deep", "v4"):
        raise ValueError("Only a validated, uncontaminated stage can be ranked")
    template = read_unique(args.data_dir / "submitting.parquet", ["MVT_ID_mvt"])
    ids = pd.Index(template.MVT_ID_mvt)
    v4_source = read_unique(args.v4_ranking,
                            ["MVT_ID_mvt", "TAXITIME_SEC_mvt"])
    v4_aligned = align(ids, v4_source, name="v4 ranking", full=True)
    v4 = v4_aligned.TAXITIME_SEC_mvt.to_numpy(dtype=float)
    if not np.isfinite(v4).all() or np.any(v4 < 0):
        raise ValueError("v4 ranking outputs must be finite and nonnegative")
    rank_rows = read_unique(args.ranking_rows, ["MVT_ID_mvt", "proxy", "airport"])
    if not np.array_equal(rank_rows.MVT_ID_mvt.to_numpy(), ids.to_numpy()):
        raise ValueError("Cached ranking covariate order differs from template")
    rank_features = pd.read_parquet(args.ranking_features,
                                    columns=["AOBT_3_flt_missing",
                                             "LOBT_flt_missing"])
    if len(rank_features) != len(ids):
        raise ValueError("Cached ranking feature count differs from template")
    a_valid = np.isfinite(rank_rows.proxy.to_numpy(dtype=float))
    expected_missing_gate = (
        ~a_valid & rank_features.AOBT_3_flt_missing.to_numpy(dtype=bool)
        & rank_features.LOBT_flt_missing.to_numpy(dtype=bool)
        & ~rank_rows.airport.eq("LIRF").to_numpy(dtype=bool))
    prediction = v4.copy()
    deep_gate = np.zeros(len(ids), dtype=bool)
    if stage in ("deep", "deep_missing", "deep_missing_clean_arrival"):
        deep_source = read_unique(args.deep_dir / "ranking_expert.parquet",
                                  ["MVT_ID_mvt", "expert"])
        deep_aligned = align(ids, deep_source, name="deep ranking expert", full=True)
        deep_expert = deep_aligned.expert.to_numpy(dtype=float)
        deep_gate = np.isfinite(deep_expert)
        if not np.array_equal(deep_gate, a_valid):
            raise ValueError("Deep ranking expert must cover exactly valid AOBT")
        prediction[deep_gate] = np.maximum(
            .5 * prediction[deep_gate] + .5 * deep_expert[deep_gate], 0)
    missing_gate = np.zeros(len(ids), dtype=bool)
    if stage in ("deep_missing", "deep_missing_clean_arrival"):
        missing_source = read_unique(args.missing_dir / "ranking_expert.parquet",
                                     ["MVT_ID_mvt", "ordinary_no_nm_gate",
                                      "catboost_direct_prediction"])
        missing_aligned = align(ids, missing_source,
                                name="missing ranking expert", full=True)
        missing_gate = missing_aligned.ordinary_no_nm_gate.to_numpy(dtype=bool)
        direct = missing_aligned.catboost_direct_prediction.to_numpy(dtype=float)
        if (not np.array_equal(missing_gate, expected_missing_gate)
                or not np.isfinite(direct[missing_gate]).all()):
            raise ValueError("Missing expert gate or finite output differs from covariates")
        if np.any(missing_gate & deep_gate):
            raise ValueError("Deep and missing ranking gates overlap")
        prediction[missing_gate] = np.maximum(
            .5 * prediction[missing_gate] + .5 * direct[missing_gate], 0)
    clean_gate = np.zeros(len(ids), dtype=bool)
    if stage == "deep_missing_clean_arrival":
        clean_source = read_unique(args.clean_arrival_dir / "ranking_raw_expert.parquet",
                                   ["MVT_ID_mvt", "arrival_direct_expert"])
        clean_aligned = align(ids, clean_source, name="clean ARR ranking expert", full=True)
        clean_expert = clean_aligned.arrival_direct_expert.to_numpy(dtype=float)
        clean_gate = np.isfinite(clean_expert)
        if not np.array_equal(clean_gate, a_valid):
            raise ValueError("Clean ARR ranking expert must cover exactly valid AOBT")
        weight = float(selection["clean_arrival_seasonal_weight"])
        prediction[clean_gate] = np.maximum(
            prediction[clean_gate] + weight * (
                clean_expert[clean_gate] - prediction[clean_gate]), 0)
    if (not np.isfinite(prediction).all() or np.any(prediction < 0)
            or not np.array_equal(prediction[~(deep_gate | missing_gate | clean_gate)],
                                  v4[~(deep_gate | missing_gate | clean_gate)])):
        raise ValueError("Ranking predictions are invalid or changed outside gates")
    result = pd.DataFrame({"MVT_ID_mvt": template.MVT_ID_mvt,
                           "TAXITIME_SEC_mvt": prediction})
    if not np.array_equal(result.MVT_ID_mvt.to_numpy(), template.MVT_ID_mvt.to_numpy()):
        raise ValueError("Ranking output differs from the submission template order")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result.to_parquet(args.output_dir / "predictions.parquet", index=False)
    manifest = {"selected_stage": stage, "ranking_rows": len(result),
                "deep_gate_rows": int(deep_gate.sum()),
                "missing_gate_rows": int(missing_gate.sum()),
                "clean_arrival_gate_rows": int(clean_gate.sum()),
                "clean_arrival_weight": float(selection["clean_arrival_seasonal_weight"]),
                "changed_rows": int(np.count_nonzero(prediction != v4)),
                "unchanged_rows": int(np.count_nonzero(prediction == v4)),
                "arrival_included": bool(clean_gate.any()),
                "legacy_arrival_included": False,
                "exact_template_order": True, "finite_nonnegative": True,
                "prediction_file": str((args.output_dir / "predictions.parquet").resolve()),
                "validation_file": str((args.output_dir / "validation.json").resolve()),
                "command": "python v5_ensemble.py --mode ranking",
                "no_upload_performed": True}
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("evaluate", "ranking"),
                        default="evaluate")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("artifacts/v5-ensemble"))
    parser.add_argument("--v4-oof", type=Path,
                        default=Path("artifacts/v4/validation_predictions.parquet"))
    parser.add_argument("--v4-ranking", type=Path,
                        default=Path("artifacts/catboost/source/sequential_predictions.parquet"))
    parser.add_argument("--deep-dir", type=Path,
                        default=Path("artifacts/v5-deep"))
    parser.add_argument("--missing-dir", type=Path,
                        default=Path("artifacts/v5-missing"))
    parser.add_argument("--arrival-dir", type=Path,
                        default=Path("artifacts/v5-arrival"))
    parser.add_argument("--clean-arrival-dir", type=Path,
                        default=Path("artifacts/v5-arrival-clean"))
    parser.add_argument("--ranking-rows", type=Path,
                        default=Path("artifacts/baseline/ranking_rows.parquet"))
    parser.add_argument("--ranking-features", type=Path,
                        default=Path("artifacts/baseline/ranking_features.parquet"))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--repeats", type=int, default=1000)
    args = parser.parse_args()
    result = evaluate(args) if args.mode == "evaluate" else ranking(args)
    if args.mode == "evaluate":
        print(json.dumps({"all_finite_scores": result["all_finite_scores"],
                          "by_fold": {fold: result["folds"][fold]["scores_all_finite"]
                                      for fold in FOLDS},
                          "selection": result["selection"]}, indent=2))
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
