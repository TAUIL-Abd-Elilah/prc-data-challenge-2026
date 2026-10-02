"""One shared all-positive RMSE expert for non-LIRF no-NM invalid AOBT.

Uses the ranking-safe 75-column baseline, weather, ARR, full-flight and
unclipped-schedule feature table from v6_missing_ensemble. The only modeling
change is a direct RMSE fit to every finite positive non-LIRF target, including
long labels. Fold training excludes the held-out months. A coarse blend is
selected on Jan/Jul and frozen for Nov/Dec; both day-level gain intervals must
be positive before a final ranking fit or fresh April/October audit.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

from v6_missing_ensemble import FOLDS, SEED, day_bootstrap, load_subset, rmse


WEIGHTS = (0.0, .1, .25, .5, 1.0)


def params(args: argparse.Namespace, iterations: int) -> dict:
    return dict(loss_function="RMSE", eval_metric="RMSE",
                iterations=iterations, learning_rate=.045, depth=6,
                l2_leaf_reg=20, random_strength=.5,
                bagging_temperature=.5, max_ctr_complexity=1,
                one_hot_max_size=20, border_count=128,
                thread_count=args.threads, used_ram_limit="4gb",
                random_seed=SEED, allow_writing_files=False, verbose=100)


def fit_model(args: argparse.Namespace, x: pd.DataFrame,
              y: np.ndarray, index: np.ndarray,
              iterations: int, internal_stop: bool):
    rng = np.random.default_rng(SEED)
    order = rng.permutation(index)
    if internal_stop:
        n_early = max(500, int(.1 * len(order)))
        early, fit = order[:n_early], order[n_early:]
    else:
        early = np.array([], dtype=np.int64)
        fit = order
    cats = x.select_dtypes(include="category").columns.tolist()
    pool = Pool(x.iloc[fit], label=y[fit], cat_features=cats)
    eval_pool = Pool(x.iloc[early], label=y[early],
                     cat_features=cats) if internal_stop else None
    model = CatBoostRegressor(**params(args, iterations))
    start = time.monotonic()
    if internal_stop:
        model.fit(pool, eval_set=eval_pool, use_best_model=True,
                  early_stopping_rounds=100)
    else:
        model.fit(pool)
    report = {"train_n": len(index), "fit_n": len(fit),
              "internal_early_n": len(early),
              "best_iteration": (int(model.get_best_iteration())
                                 if internal_stop else iterations - 1),
              "trees": int(model.tree_count_),
              "fit_seconds": time.monotonic() - start,
              "feature_names": list(x.columns),
              "categorical": cats}
    del pool, eval_pool
    gc.collect()
    return model, report


def fit_oof(args: argparse.Namespace) -> dict:
    rows, x = load_subset(args)
    non_lirf = ~rows.airport.eq("LIRF").to_numpy(dtype=bool)
    y = rows.target.to_numpy(dtype=float)
    reports = {}
    for fold in (FOLDS if args.fold == "both" else (args.fold,)):
        heldout = rows.month.isin(FOLDS[fold]).to_numpy(dtype=bool)
        train = np.flatnonzero(~heldout & non_lirf & np.isfinite(y) & (y > 0))
        test = np.flatnonzero(heldout & non_lirf & np.isfinite(y))
        model, report = fit_model(args, x, y, train,
                                  args.iterations, internal_stop=True)
        prediction = model.predict(x.iloc[test], thread_count=args.threads)
        if not np.isfinite(prediction).all():
            raise ValueError("RMSE expert produced nonfinite held-out output")
        out = rows.iloc[test][["MVT_ID_mvt", "target", "airport", "month", "time"]].copy()
        out["fold"] = fold
        out["expert"] = prediction
        args.output_dir.mkdir(parents=True, exist_ok=True)
        out.to_parquet(args.output_dir / f"{fold}_oof.parquet", index=False)
        model.save_model(str(args.output_dir / f"{fold}.cbm"))
        report["heldout_months"] = FOLDS[fold]
        report["test_n"] = len(test)
        reports[fold] = report
        (args.output_dir / f"{fold}_training.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8")
    return reports


def evaluate(args: argparse.Namespace) -> dict:
    base = pd.read_parquet(args.v5_oof,
                           columns=["MVT_ID_mvt", "target", "fold", "a_valid",
                                    "airport", "MVT_TIME_UTC_mvt", "selected"])
    if len(base) != 672428 or base.MVT_ID_mvt.duplicated().any():
        raise ValueError("Frozen v5 OOF is incomplete")
    specialist = pd.concat([pd.read_parquet(args.output_dir / f"{fold}_oof.parquet")
                            for fold in FOLDS], ignore_index=True)
    if specialist.MVT_ID_mvt.duplicated().any():
        raise ValueError("RMSE specialist repeated an ID")
    frame = base.merge(specialist[["MVT_ID_mvt", "target", "fold", "expert"]],
                       on="MVT_ID_mvt", how="left", validate="one_to_one",
                       suffixes=("", "_expert"))
    gate = frame.expert.notna().to_numpy(dtype=bool)
    if (gate.sum() != len(specialist)
            or frame.a_valid.to_numpy(dtype=bool)[gate].any()
            or frame.airport.eq("LIRF").to_numpy(dtype=bool)[gate].any()
            or not np.array_equal(frame.target.to_numpy(dtype=float)[gate],
                                  frame.target_expert.to_numpy(dtype=float)[gate])
            or not np.array_equal(frame.fold.to_numpy()[gate],
                                  frame.fold_expert.to_numpy()[gate])):
        raise ValueError("RMSE OOF gate, target or fold differs from v5")
    y = frame.target.to_numpy(dtype=float)
    old = frame.selected.to_numpy(dtype=float)
    new = frame.expert.to_numpy(dtype=float)
    predictions = {}
    for weight in WEIGHTS:
        pred = old.copy()
        pred[gate] = np.maximum(old[gate] + weight * (new[gate] - old[gate]), 0)
        predictions[weight] = pred
    fold_scores = {fold: {str(weight): rmse(y[mask], pred[mask])
                          for weight, pred in predictions.items()}
                   for fold in FOLDS
                   for mask in (frame.fold.eq(fold).to_numpy(dtype=bool),)}
    weight = min(WEIGHTS, key=lambda w:
                 (fold_scores["seasonal_jan_jul"][str(w)], w))
    selected = predictions[weight]
    day = {fold: day_bootstrap(frame, old, selected, fold) for fold in FOLDS}
    point_pass = weight > 0 and all(
        fold_scores[fold][str(weight)] < fold_scores[fold]["0.0"]
        for fold in FOLDS)
    robust_pass = point_pass and all(day[fold]["gain_ci95_sec"][0] > 0
                                     for fold in FOLDS)
    report = {"model": "Shared depth-6 RMSE CatBoost on all finite positive non-LIRF no-NM invalid-AOBT training labels",
              "weights_predeclared": WEIGHTS,
              "selected_weight_from_jan_jul": weight,
              "point_gain_both_folds": point_pass,
              "day_ci_lower_positive_both_folds": robust_pass,
              "promoted_for_audit": robust_pass,
              "all_finite_n": len(frame), "expert_gate_n": int(gate.sum()),
              "fold_scores_all_finite": fold_scores,
              "combined_scores_all_finite": {str(w): rmse(y, p)
                                             for w, p in predictions.items()},
              "day_bootstrap_selected_vs_v5": day,
              "validation_limit": "Both periods were used for local model comparison; no ranking outcomes were read."}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "validation.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame({"MVT_ID_mvt": frame.MVT_ID_mvt, "target": y,
                  "fold": frame.fold, "v5": old,
                  "candidate_gate": gate,
                  "selected": selected}).to_parquet(
                      args.output_dir / "validation_predictions.parquet", index=False)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("fit-oof", "evaluate"),
                        default="fit-oof")
    parser.add_argument("--fold", choices=(*FOLDS, "both"), default="both")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--cache-dir", type=Path,
                        default=Path("artifacts/baseline"))
    parser.add_argument("--arrival-dir", type=Path,
                        default=Path("artifacts/v5-arrival-clean"))
    parser.add_argument("--weather-file", type=Path,
                        default=Path("data/external/weather.parquet"))
    parser.add_argument("--v5-oof", type=Path,
                        default=Path("artifacts/v5-ensemble/validation_predictions.parquet"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("artifacts/v6-missing-rmse"))
    args = parser.parse_args()
    if not 1 <= args.threads <= 4:
        raise ValueError("At most four CPU threads are allowed")
    if args.mode == "fit-oof":
        reports = fit_oof(args)
        print(json.dumps({fold: {"train_n": v["train_n"],
                                 "test_n": v["test_n"],
                                 "trees": v["trees"]}
                          for fold, v in reports.items()}, indent=2))
    else:
        result = evaluate(args)
        print(json.dumps({k: result[k] for k in (
            "selected_weight_from_jan_jul", "promoted_for_audit",
            "fold_scores_all_finite", "day_bootstrap_selected_vs_v5")},
                         indent=2))


if __name__ == "__main__":
    main()
