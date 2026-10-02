"""Test schedule-clock features as a replacement source probability at LIRF/LTFM.

This is an isolated GPU experiment. The only labels are released 2025 taxi-out
targets; raw departure BLOCK/TAXITIME never enter the feature table. The
existing v4 source probability remains the comparison, and validation adds
only (p_new - p_old) times the schedule-versus-frozen-base difference.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import brier_score_loss, roc_auc_score

from catboost_source import candidate_mask
from deep_timestamp_expert import load_features


FOLDS = {"seasonal_jan_jul": (1, 7), "forward_nov_dec": (11, 12)}
SCALES = (0.0, .25, .5, 1.0)
SEED = 2026


def rmse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(y - p))))


def add_clock_features(x: pd.DataFrame) -> pd.DataFrame:
    """Use only published scheduled-departure clock time at prediction time."""
    x = x.copy()
    minute = x["time_minute_SCHED_TIME_UTC_mvt"].to_numpy(dtype=float)
    second = x["time_second_SCHED_TIME_UTC_mvt"].to_numpy(dtype=float)
    phase = np.mod(60 * minute + second, 900)
    distance = np.minimum(phase, 900 - phase)
    x["schedule_quarter_phase_sec"] = phase.astype("float32")
    x["schedule_quarter_distance_sec"] = distance.astype("float32")
    x["schedule_exact_quarter"] = np.where(
        np.isfinite(distance), distance <= 1, np.nan).astype("float32")
    x["schedule_exact_five_minute"] = np.where(
        np.isfinite(minute) & np.isfinite(second),
        (np.mod(minute, 5) == 0) & (second == 0), np.nan).astype("float32")
    return x


def params(args: argparse.Namespace, iterations: int) -> dict:
    return dict(task_type="GPU", devices="0", gpu_ram_part=.35,
                loss_function="Logloss", eval_metric="Logloss",
                iterations=iterations, depth=7, learning_rate=.05,
                l2_leaf_reg=8, random_strength=.5,
                bagging_temperature=.5, max_ctr_complexity=1,
                one_hot_max_size=20, border_count=128,
                thread_count=args.threads, random_seed=SEED,
                allow_writing_files=False, verbose=100)


def fit_fold(name: str, rows: pd.DataFrame, features: pd.DataFrame,
             args: argparse.Namespace) -> dict:
    months = FOLDS[name]
    candidate, schedule = candidate_mask(rows, features)
    y = rows.target.to_numpy(dtype=float)
    heldout = rows.month.isin(months).to_numpy(dtype=bool)
    train = np.flatnonzero(candidate & ~heldout & np.isfinite(y)
                           & (y >= 0) & (y <= 86400))
    test = np.flatnonzero(candidate & heldout & np.isfinite(y))
    label = (np.abs(y - schedule) <= 60).astype(np.int8)
    rng = np.random.default_rng(SEED)
    order = rng.permutation(train)
    n_early = max(3000, int(.08 * len(order)))
    early, fit = order[:n_early], order[n_early:]
    categories = features.select_dtypes(include="category").columns.tolist()
    model = CatBoostClassifier(**params(args, args.iterations))
    start = time.monotonic()
    model.fit(Pool(features.iloc[fit], label=label[fit],
                   cat_features=categories),
              eval_set=Pool(features.iloc[early], label=label[early],
                            cat_features=categories),
              early_stopping_rounds=100, use_best_model=True)
    seconds = time.monotonic() - start
    p_new = model.predict_proba(features.iloc[test],
                                thread_count=args.threads)[:, 1]
    source_path = args.old_source_dir / f"{name}_oof.parquet"
    old = pd.read_parquet(source_path,
                          columns=["MVT_ID_mvt", "target", "schedule_proxy_sec",
                                   "p_schedule_exact"])
    if old.MVT_ID_mvt.duplicated().any():
        raise ValueError("Existing source OOF has repeated IDs")
    out = rows.iloc[test][["MVT_ID_mvt", "target", "airport", "month"]].copy()
    out["fold"] = name
    out["schedule_proxy_sec"] = schedule[test]
    out["schedule_exact"] = label[test].astype(bool)
    out["p_new"] = p_new
    out = out.merge(old.rename(columns={"target": "old_target",
                                        "schedule_proxy_sec": "old_schedule",
                                        "p_schedule_exact": "p_old"}),
                    on="MVT_ID_mvt", how="left", validate="one_to_one")
    if (len(out) != len(old) or out.p_old.isna().any()
            or not np.array_equal(out.target.to_numpy(dtype=float),
                                  out.old_target.to_numpy(dtype=float))
            or not np.array_equal(out.schedule_proxy_sec.to_numpy(dtype=float),
                                  out.old_schedule.to_numpy(dtype=float))):
        raise ValueError("New and original source OOF gates, targets, or schedules differ")
    out = out.drop(columns=["old_target", "old_schedule"])
    if not np.isfinite(p_new).all() or np.any((p_new < 0) | (p_new > 1)):
        raise ValueError("New source probabilities are invalid")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output_dir / f"{name}_oof.parquet", index=False)
    model.save_model(str(args.output_dir / f"{name}.cbm"))
    report = {"fold": name, "heldout_months": months,
              "train_n": len(train), "test_n": len(test),
              "train_exact_rate": float(label[train].mean()),
              "test_exact_rate": float(label[test].mean()),
              "trees": model.tree_count_,
              "best_iteration": model.get_best_iteration(),
              "fit_seconds": seconds,
              "brier_old": float(brier_score_loss(label[test],
                                                    out.p_old.to_numpy(dtype=float))),
              "brier_new": float(brier_score_loss(label[test], p_new)),
              "auc_old": float(roc_auc_score(label[test],
                                              out.p_old.to_numpy(dtype=float))),
              "auc_new": float(roc_auc_score(label[test], p_new)),
              "feature_count": len(features.columns),
              "feature_names": list(features.columns)}
    (args.output_dir / f"{name}_training.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    return report


def fit_oof(args: argparse.Namespace) -> dict:
    rows, features = load_features(args)
    features = add_clock_features(features)
    report = {}
    for fold in (FOLDS if args.fold == "both" else (args.fold,)):
        report[fold] = fit_fold(fold, rows, features, args)
    return report


def bootstrap(frame: pd.DataFrame, base: np.ndarray, challenger: np.ndarray,
              fold: str, repeats: int = 1000) -> dict:
    use = frame.fold.eq(fold).to_numpy(dtype=bool)
    dates = pd.to_datetime(frame.loc[use, "MVT_TIME_UTC_mvt"], utc=True).dt.floor("D")
    group, unique_days = pd.factorize(dates, sort=True)
    y = frame.target.to_numpy(dtype=float)[use]
    b = base[use]
    c = challenger[use]
    n = np.bincount(group).astype(float)
    sse_b = np.bincount(group, weights=np.square(y - b))
    sse_c = np.bincount(group, weights=np.square(y - c))
    rng = np.random.default_rng(SEED + (0 if fold == "seasonal_jan_jul" else 1))
    picked = rng.integers(0, len(unique_days), size=(repeats, len(unique_days)))
    den = n[picked].sum(axis=1)
    gain = np.sqrt(sse_b[picked].sum(axis=1) / den) - np.sqrt(
        sse_c[picked].sum(axis=1) / den)
    return {"days": len(unique_days), "repeats": repeats,
            "gain_sec": rmse(y, b) - rmse(y, c),
            "gain_ci95_sec": [float(q) for q in np.quantile(gain, [.025, .975])],
            "fraction_positive": float(np.mean(gain > 0))}


def evaluate(args: argparse.Namespace) -> dict:
    base_frame = pd.read_parquet(args.base_oof,
                                 columns=["MVT_ID_mvt", "target", "fold",
                                          "a_valid", "MVT_TIME_UTC_mvt",
                                          "deep_missing"])
    if (len(base_frame) != 672428 or base_frame.MVT_ID_mvt.duplicated().any()
            or not np.isfinite(base_frame.deep_missing.to_numpy(dtype=float)).all()):
        raise ValueError("Frozen deep+missing OOF is incomplete")
    source = pd.concat([pd.read_parquet(args.output_dir / f"{fold}_oof.parquet")
                        for fold in FOLDS], ignore_index=True)
    if source.MVT_ID_mvt.duplicated().any():
        raise ValueError("New source OOF has repeated IDs")
    joined = base_frame.merge(source, on="MVT_ID_mvt", how="left",
                              validate="one_to_one", suffixes=("", "_source"))
    gate = joined.p_new.notna().to_numpy(dtype=bool)
    if (int(gate.sum()) != len(source)
            or not np.array_equal(joined.target.to_numpy(dtype=float)[gate],
                                  joined.target_source.to_numpy(dtype=float)[gate])
            or not np.array_equal(joined.fold.to_numpy()[gate],
                                  joined.fold_source.to_numpy()[gate])
            or not joined.a_valid.to_numpy(dtype=bool)[gate].all()):
        raise ValueError("New source OOF labels, folds, or valid-AOBT gates differ")
    y = joined.target.to_numpy(dtype=float)
    base = joined.deep_missing.to_numpy(dtype=float)
    schedule = joined.schedule_proxy_sec.to_numpy(dtype=float)
    p_old = joined.p_old.to_numpy(dtype=float)
    p_new = joined.p_new.to_numpy(dtype=float)
    if not (np.isfinite(schedule[gate]).all()
            and np.isfinite(p_old[gate]).all()
            and np.isfinite(p_new[gate]).all()):
        raise ValueError("New source OOF has nonfinite inputs")
    delta = np.zeros(len(joined), dtype=float)
    delta[gate] = (p_new[gate] - p_old[gate]) * (schedule[gate] - base[gate])
    predictions = {scale: np.maximum(base + scale * delta, 0) for scale in SCALES}
    mask = {fold: joined.fold.eq(fold).to_numpy(dtype=bool) for fold in FOLDS}
    fold_scores = {fold: {str(scale): rmse(y[use], pred[use])
                          for scale, pred in predictions.items()}
                   for fold, use in mask.items()}
    selected_scale = min(SCALES, key=lambda scale:
                         (fold_scores["seasonal_jan_jul"][str(scale)], scale))
    selected = predictions[selected_scale]
    point_gain_both = selected_scale > 0 and all(
        fold_scores[fold][str(selected_scale)] < fold_scores[fold]["0.0"]
        for fold in FOLDS)
    stability = {fold: bootstrap(joined, base, selected, fold) for fold in FOLDS}
    stable_both = point_gain_both and all(
        stability[fold]["gain_ci95_sec"][0] > 0 for fold in FOLDS)
    report = {"formula": "clip(deep_missing + scale*(p_new-p_old)*(schedule-deep_missing),0)",
              "scales_predeclared": SCALES, "selection_months": "January/July",
              "selected_scale": selected_scale,
              "point_gain_both_folds": point_gain_both,
              "day_block_stability_passes_both_folds": stable_both,
              "gain_both_folds": stable_both,
              "source_candidate_n": int(gate.sum()),
              "source_candidate_by_fold": {fold: int(np.sum(mask[fold] & gate))
                                           for fold in FOLDS},
              "fold_scores_all_finite": fold_scores,
              "combined_scores_all_finite": {str(scale): rmse(y, pred)
                                             for scale, pred in predictions.items()},
              "day_bootstrap_selected_vs_base": stability,
              "classification": {fold: json.loads(
                  (args.output_dir / f"{fold}_training.json").read_text(
                      encoding="utf-8")) for fold in FOLDS},
              "validation_limit": "Both held-out periods have been used for model comparison; no ranking outcomes were read."}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "validation.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame({"MVT_ID_mvt": joined.MVT_ID_mvt,
                  "target": y, "fold": joined.fold,
                  "deep_missing": base, "source_candidate": gate,
                  "selected": selected}).to_parquet(
                      args.output_dir / "validation_predictions.parquet", index=False)
    return report


def fit_final(args: argparse.Namespace) -> dict:
    validation = json.loads((args.output_dir / "validation.json").read_text(
        encoding="utf-8"))
    if not validation["gain_both_folds"]:
        raise ValueError("Source clock replacement did not improve both folds")
    trees = [json.loads((args.output_dir / f"{fold}_training.json").read_text(
        encoding="utf-8"))["trees"] for fold in FOLDS]
    iterations = int(np.median(trees))
    rows, features = load_features(args)
    features = add_clock_features(features)
    candidate, schedule = candidate_mask(rows, features)
    y = rows.target.to_numpy(dtype=float)
    train = np.flatnonzero(candidate & np.isfinite(y) & (y >= 0) & (y <= 86400))
    train_n = len(train)
    exact = (np.abs(y - schedule) <= 60).astype(np.int8)
    categories = features.select_dtypes(include="category").columns.tolist()
    model = CatBoostClassifier(**params(args, iterations))
    start = time.monotonic()
    model.fit(Pool(features.iloc[train], label=exact[train],
                   cat_features=categories))
    seconds = time.monotonic() - start
    model.save_model(str(args.output_dir / "final.cbm"))
    names = list(features.columns)
    del rows, features, candidate, schedule, y, train, exact
    gc.collect()
    rank_rows, rank_features = load_features(args, ranking=True)
    rank_features = add_clock_features(rank_features)
    if list(rank_features.columns) != names:
        raise ValueError("Training and ranking clock features differ")
    candidate, _ = candidate_mask(rank_rows, rank_features)
    indices = np.flatnonzero(candidate)
    probability = np.full(len(rank_rows), np.nan, dtype=float)
    probability[indices] = model.predict_proba(
        rank_features.iloc[indices], thread_count=args.threads)[:, 1]
    result = pd.DataFrame({"MVT_ID_mvt": rank_rows.MVT_ID_mvt,
                           "source_candidate": candidate,
                           "p_new": probability})
    result.to_parquet(args.output_dir / "ranking_probabilities.parquet", index=False)
    manifest = {"training_n": train_n,
                "iterations": iterations, "fold_trees": trees,
                "fit_seconds": seconds,
                "ranking_rows": len(result),
                "ranking_candidate_n": int(candidate.sum()),
                "selected_scale": validation["selected_scale"],
                "raw_file": str((args.output_dir / "ranking_probabilities.parquet").resolve()),
                "no_upload_performed": True}
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("fit-oof", "evaluate", "fit-final"),
                        default="fit-oof")
    parser.add_argument("--fold", choices=(*FOLDS, "both"), default="both")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--cache-dir", type=Path,
                        default=Path("artifacts/baseline"))
    parser.add_argument("--weather-file", type=Path,
                        default=Path("data/external/weather.parquet"))
    parser.add_argument("--old-source-dir", type=Path,
                        default=Path("artifacts/catboost/source"))
    parser.add_argument("--base-oof", type=Path,
                        default=Path("artifacts/v5-ensemble/validation_predictions.parquet"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("artifacts/v5-source-clock"))
    parser.add_argument("--iterations", type=int, default=600)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.threads > 4:
        raise ValueError("Limit this experiment to four host threads")
    if args.mode == "fit-oof":
        result = fit_oof(args)
        print(json.dumps({k: {p: v[p] for p in ("train_n", "test_n", "trees",
                                             "fit_seconds", "brier_old", "brier_new")}
                          for k, v in result.items()}, indent=2))
    elif args.mode == "evaluate":
        result = evaluate(args)
        print(json.dumps({k: result[k] for k in ("selected_scale", "gain_both_folds",
                                               "fold_scores_all_finite",
                                               "day_bootstrap_selected_vs_base")},
                         indent=2))
    else:
        print(json.dumps(fit_final(args), indent=2))


if __name__ == "__main__":
    main()
