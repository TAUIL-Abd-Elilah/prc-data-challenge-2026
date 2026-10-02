"""All-airport invalid-AOBT mixture using released NM, weather and ARR fields.

One predeclared candidate combines a robust ordinary taxi-time mean, a long
taxi-time probability, and a conditional long-time schedule ratio. Every 2025
finite positive no-NM target enters the ordinary and probability fits; the
tail ratio uses positive long targets with long schedule gaps. Departure BLOCK
and TAXITIME fields are never read as predictors. January/July selects one
coarse blend with the frozen v5 ensemble; November/December checks transfer.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from catboost import CatBoostClassifier, CatBoostRegressor, Pool

from missing_catboost import DIRECT_CATEGORICAL, NUMERIC
from solution import _training_files
from weather_model import add_weather


FOLDS = {"seasonal_jan_jul": (1, 7), "forward_nov_dec": (11, 12)}
WEIGHTS = (0.0, .1, .25, .5, 1.0)
SEED = 2026


def rmse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(y - p))))


def load_subset(args: argparse.Namespace, ranking: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Subset to no-NM invalid AOBT before joining weather and raw covariates."""
    prefix = "ranking" if ranking else "training"
    rows_all = pd.read_parquet(args.cache_dir / f"{prefix}_rows.parquet")
    x_all = pd.read_parquet(args.cache_dir / (
        "ranking_features.parquet" if ranking else "features.parquet"))
    if len(rows_all) != len(x_all):
        raise ValueError("Baseline row and feature caches disagree")
    mask = (~np.isfinite(rows_all.proxy.to_numpy(dtype=float))
            & x_all.AOBT_3_flt_missing.to_numpy(dtype=bool)
            & x_all.LOBT_flt_missing.to_numpy(dtype=bool))
    indices = np.flatnonzero(mask)
    rows = rows_all.iloc[indices].reset_index(drop=True)
    x = x_all.iloc[indices].reset_index(drop=True)
    arr = pd.read_parquet(args.arrival_dir / f"{prefix}_arrival_features.parquet")
    if (len(arr) != len(rows_all)
            or not np.array_equal(arr.MVT_ID_mvt.to_numpy(),
                                  rows_all.MVT_ID_mvt.to_numpy())):
        raise ValueError("ARR traffic cache does not align with baseline rows")
    arr = arr.iloc[indices].drop(columns="MVT_ID_mvt").reset_index(drop=True)
    x = pd.concat([x, arr], axis=1)
    del rows_all, x_all, arr
    gc.collect()

    paths = [args.data_dir / "ranking.parquet"] if ranking else _training_files(args.data_dir)
    ids = rows.MVT_ID_mvt.to_list()
    raw = (pl.scan_parquet([str(path) for path in paths])
           .filter((pl.col("PHASE_mvt") == "DEP")
                   & pl.col("MVT_ID_mvt").is_in(ids))
           .select(["MVT_ID_mvt", "MVT_TIME_UTC_mvt",
                    "SCHED_TIME_UTC_mvt", "FLIGHT_mvt"])
           .collect().to_pandas())
    if raw.MVT_ID_mvt.duplicated().any() or len(raw) != len(rows):
        raise ValueError("Raw departure covariates have incomplete movement IDs")
    raw = rows[["MVT_ID_mvt"]].merge(raw, on="MVT_ID_mvt", how="left",
                                      sort=False, validate="one_to_one")
    if raw.MVT_TIME_UTC_mvt.isna().any():
        raise ValueError("Departure movement timestamps are missing")
    movement = pd.to_datetime(raw.MVT_TIME_UTC_mvt, utc=True, errors="coerce")
    schedule = pd.to_datetime(raw.SCHED_TIME_UTC_mvt, utc=True, errors="coerce")
    gap = (movement - schedule).dt.total_seconds().to_numpy(dtype=float)
    x["schedule_unclipped_sec"] = gap.astype("float32")
    x["schedule_day_offset"] = np.floor(gap / 86400).astype("float32")
    x["schedule_hour"] = schedule.dt.hour.astype("float32")
    x["schedule_minute"] = schedule.dt.minute.astype("float32")
    x["schedule_second"] = schedule.dt.second.astype("float32")
    x["schedule_weekday"] = schedule.dt.dayofweek.astype("float32")
    phase = np.mod(60 * x.schedule_minute.to_numpy(dtype=float)
                   + x.schedule_second.to_numpy(dtype=float), 900)
    x["schedule_quarter_distance_sec"] = np.minimum(phase, 900 - phase).astype("float32")
    x["schedule_long_7200"] = (gap > 7200).astype("int8")
    x["schedule_long_20000"] = (gap > 20000).astype("int8")
    x["flight_name"] = raw.FLIGHT_mvt.astype("string").fillna("?").astype("category")
    x = add_weather(x, rows, args.weather_file)
    if any(col in x for col in ("BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt", "target")):
        raise ValueError("Departure label-derived field entered the feature table")
    clock_names = ("schedule_unclipped_sec", "schedule_day_offset",
                   "schedule_hour", "schedule_minute", "schedule_second",
                   "schedule_weekday", "schedule_quarter_distance_sec",
                   "schedule_long_7200", "schedule_long_20000")
    wanted = [col for col in NUMERIC if col in x]
    wanted += [col for col in x if col.startswith("wx_") or col.startswith("arr_")]
    wanted += [col for col in clock_names if col in x]
    wanted += [col for col in DIRECT_CATEGORICAL if col in x]
    x = x[list(dict.fromkeys(wanted))]
    return rows, x


def model_params(args: argparse.Namespace, kind: str,
                 iterations: int) -> dict:
    loss = {"ordinary": "Huber:delta=1200", "tail_probability": "Logloss",
            "tail_ratio": "RMSE"}[kind]
    return dict(loss_function=loss, eval_metric=loss,
                iterations=iterations, learning_rate=.045,
                depth=6 if kind == "ordinary" else (5 if kind == "tail_probability" else 4),
                l2_leaf_reg=15, random_strength=.5,
                bagging_temperature=.5, max_ctr_complexity=1,
                one_hot_max_size=20, border_count=128,
                thread_count=args.threads, used_ram_limit="4gb",
                random_seed=SEED, allow_writing_files=False, verbose=100)


def train_model(args: argparse.Namespace, name: str, x: pd.DataFrame,
                index: np.ndarray, label: np.ndarray,
                iterations: int, early_stop: bool,
                weights: np.ndarray | None = None):
    if len(index) < (100 if name == "tail_ratio" else 1000):
        raise ValueError(f"Too few complementary training rows for {name}")
    rng = np.random.default_rng(SEED + len(name))
    order = rng.permutation(index)
    if early_stop:
        n_early = max(60 if name == "tail_ratio" else 500,
                      int(.15 * len(order)))
        n_early = min(n_early, len(order) // 3)
        early, fit = order[:n_early], order[n_early:]
    else:
        early = np.array([], dtype=np.int64)
        fit = order
    cats = x.select_dtypes(include="category").columns.tolist()
    train_pool = Pool(x.iloc[fit], label=label[fit],
                      weight=weights[fit] if weights is not None else None,
                      cat_features=cats)
    eval_pool = (Pool(x.iloc[early], label=label[early],
                      weight=weights[early] if weights is not None else None,
                      cat_features=cats) if early_stop else None)
    cls = CatBoostClassifier if name == "tail_probability" else CatBoostRegressor
    model = cls(**model_params(args, name, iterations))
    start = time.monotonic()
    if early_stop:
        model.fit(train_pool, eval_set=eval_pool, use_best_model=True,
                  early_stopping_rounds=80)
    else:
        model.fit(train_pool)
    elapsed = time.monotonic() - start
    report = {"name": name, "eligible_train": int(len(index)),
              "fit_rows": int(len(fit)), "early_stop_rows": int(len(early)),
              "trees": int(model.tree_count_),
              "best_iteration": int(model.get_best_iteration()) if early_stop else iterations - 1,
              "fit_seconds": elapsed}
    del train_pool, eval_pool
    gc.collect()
    return model, report


def predict_mixture(models: dict, x: pd.DataFrame,
                    schedule: np.ndarray,
                    threads: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ordinary = np.clip(models["ordinary"].predict(x, thread_count=threads), 0, 14400)
    tail_p = models["tail_probability"].predict_proba(
        x, thread_count=threads)[:, 1]
    ratio = np.clip(models["tail_ratio"].predict(x, thread_count=threads),
                    .25, 2.0)
    long_gate = np.isfinite(schedule) & (schedule > 7200)
    prediction = ordinary.copy()
    prediction[long_gate] = (
        (1 - tail_p[long_gate]) * ordinary[long_gate]
        + tail_p[long_gate] * schedule[long_gate] * ratio[long_gate])
    if not np.isfinite(prediction).all():
        raise ValueError("Mixture produced a nonfinite prediction")
    return prediction, tail_p, ratio


def fit_fold(args: argparse.Namespace, fold: str, rows: pd.DataFrame,
             x: pd.DataFrame) -> dict:
    y = rows.target.to_numpy(dtype=float)
    schedule = x.schedule_unclipped_sec.to_numpy(dtype=float)
    heldout = rows.month.isin(FOLDS[fold]).to_numpy(dtype=bool)
    positive = np.isfinite(y) & (y > 0)
    ordinary_train = np.flatnonzero(positive & ~heldout)
    tail_train = np.flatnonzero(positive & ~heldout & (y > 7200)
                                & np.isfinite(schedule) & (schedule > 7200))
    tail_label = ((y > 7200) & np.isfinite(schedule)
                  & (schedule > 7200)).astype(np.int8)
    ratio_label = np.divide(y, schedule, out=np.zeros(len(y)),
                            where=np.isfinite(schedule) & (schedule != 0))
    ratio_label = np.clip(ratio_label, 0, 4)
    ratio_weights = np.clip(schedule / 7200, 1, 12) ** 2
    ratio_weights[~np.isfinite(ratio_weights)] = 1
    models = {}
    reports = {}
    for name, index, label, iters, weights in (
        ("ordinary", ordinary_train, y, args.ordinary_iterations, None),
        ("tail_probability", ordinary_train, tail_label,
         args.class_iterations, None),
        ("tail_ratio", tail_train, ratio_label,
         args.ratio_iterations, ratio_weights)):
        models[name], reports[name] = train_model(
            args, name, x, index, label, iters, True, weights)
    test = np.flatnonzero(heldout & np.isfinite(y))
    candidate, tail_p, ratio = predict_mixture(
        models, x.iloc[test], schedule[test], args.threads)
    out = rows.iloc[test][["MVT_ID_mvt", "target", "airport", "month", "time"]].copy()
    out["fold"] = fold
    out["schedule_unclipped_sec"] = schedule[test]
    out["candidate"] = candidate
    out["tail_probability"] = tail_p
    out["tail_ratio"] = ratio
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        model.save_model(str(args.output_dir / f"{fold}_{name}.cbm"))
    out.to_parquet(args.output_dir / f"{fold}_oof.parquet", index=False)
    report = {"fold": fold, "heldout_months": FOLDS[fold],
              "test_n": len(test), "positive_train": len(ordinary_train),
              "tail_train": len(tail_train), "models": reports,
              "candidate_range": [float(np.min(candidate)), float(np.max(candidate))]}
    (args.output_dir / f"{fold}_training.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    return report


def fit_oof(args: argparse.Namespace) -> dict:
    rows, x = load_subset(args)
    report = {}
    for fold in (FOLDS if args.fold == "both" else (args.fold,)):
        report[fold] = fit_fold(args, fold, rows, x)
    return report


def day_bootstrap(frame: pd.DataFrame, base: np.ndarray,
                  candidate: np.ndarray, fold: str) -> dict:
    use = frame.fold.eq(fold).to_numpy(dtype=bool)
    date = pd.to_datetime(frame.loc[use, "MVT_TIME_UTC_mvt"], utc=True).dt.floor("D")
    group, unique_days = pd.factorize(date, sort=True)
    y = frame.target.to_numpy(dtype=float)[use]
    b = base[use]
    c = candidate[use]
    n = np.bincount(group).astype(float)
    b_sse = np.bincount(group, weights=np.square(y - b))
    c_sse = np.bincount(group, weights=np.square(y - c))
    rng = np.random.default_rng(SEED + (0 if fold == "seasonal_jan_jul" else 1))
    draws = rng.integers(0, len(unique_days), size=(1000, len(unique_days)))
    count = n[draws].sum(axis=1)
    gain = np.sqrt(b_sse[draws].sum(axis=1) / count) - np.sqrt(
        c_sse[draws].sum(axis=1) / count)
    return {"days": len(unique_days), "repeats": 1000,
            "observed_gain_sec": rmse(y, b) - rmse(y, c),
            "gain_ci95_sec": [float(q) for q in np.quantile(gain, [.025, .975])],
            "fraction_positive": float(np.mean(gain > 0))}


def evaluate(args: argparse.Namespace) -> dict:
    base = pd.read_parquet(args.v5_oof,
                           columns=["MVT_ID_mvt", "target", "fold", "a_valid",
                                    "airport", "MVT_TIME_UTC_mvt", "selected"])
    if len(base) != 672428 or base.MVT_ID_mvt.duplicated().any():
        raise ValueError("Frozen v5 OOF is incomplete")
    specialist = pd.concat([pd.read_parquet(args.output_dir / f"{fold}_oof.parquet")
                            for fold in FOLDS], ignore_index=True)
    if specialist.MVT_ID_mvt.duplicated().any():
        raise ValueError("V6 missing OOF contains repeated movement IDs")
    frame = base.merge(specialist[["MVT_ID_mvt", "target", "fold", "candidate",
                                   "schedule_unclipped_sec"]],
                       on="MVT_ID_mvt", how="left", validate="one_to_one",
                       suffixes=("", "_expert"))
    gate = frame.candidate.notna().to_numpy(dtype=bool)
    if (gate.sum() != len(specialist)
            or frame.a_valid.to_numpy(dtype=bool)[gate].any()
            or not np.array_equal(frame.target.to_numpy(dtype=float)[gate],
                                  frame.target_expert.to_numpy(dtype=float)[gate])
            or not np.array_equal(frame.fold.to_numpy()[gate],
                                  frame.fold_expert.to_numpy()[gate])):
        raise ValueError("V6 expert labels, fold or invalid-AOBT gate differ")
    y = frame.target.to_numpy(dtype=float)
    old = frame.selected.to_numpy(dtype=float)
    proposed = frame.candidate.to_numpy(dtype=float)
    variants = {}
    for weight in WEIGHTS:
        pred = old.copy()
        pred[gate] = np.maximum(old[gate] + weight * (proposed[gate] - old[gate]), 0)
        variants[weight] = pred
    fold_scores = {fold: {str(weight): rmse(y[mask], pred[mask])
                          for weight, pred in variants.items()}
                   for fold in FOLDS
                   for mask in (frame.fold.eq(fold).to_numpy(dtype=bool),)}
    selected_weight = min(WEIGHTS, key=lambda w:
                          (fold_scores["seasonal_jan_jul"][str(w)], w))
    selected = variants[selected_weight]
    day = {fold: day_bootstrap(frame, old, selected, fold) for fold in FOLDS}
    point_pass = selected_weight > 0 and all(
        fold_scores[fold][str(selected_weight)] < fold_scores[fold]["0.0"]
        for fold in FOLDS)
    stable_pass = point_pass and all(day[fold]["gain_ci95_sec"][0] > 0
                                     for fold in FOLDS)
    report = {"candidate": "all-positive Huber ordinary + long-tail probability * conditional schedule ratio",
              "ranking_safe_covariates": "Baseline, ARR traffic, weather, full flight, unclipped schedule clock",
              "weights_predeclared": WEIGHTS,
              "selected_weight_from_jan_jul": selected_weight,
              "point_gain_both_folds": point_pass,
              "day_block_stability_both_folds": stable_pass,
              "promoted": stable_pass,
              "all_finite_n": len(frame), "expert_gate_n": int(gate.sum()),
              "expert_gate_by_fold": {fold: int(np.sum(gate & frame.fold.eq(fold).to_numpy()))
                                      for fold in FOLDS},
              "fold_scores_all_finite": fold_scores,
              "combined_scores_all_finite": {str(weight): rmse(y, pred)
                                             for weight, pred in variants.items()},
              "day_bootstrap_selected_vs_v5": day,
              "validation_limit": "Both held-out periods have been used for local model comparison; no ranking outcomes were read."}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "validation.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame({"MVT_ID_mvt": frame.MVT_ID_mvt, "target": y,
                  "fold": frame.fold, "v5": old,
                  "v6_candidate_gate": gate,
                  "selected": selected}).to_parquet(
                      args.output_dir / "validation_predictions.parquet", index=False)
    return report


def fit_final(args: argparse.Namespace) -> dict:
    validation = json.loads((args.output_dir / "validation.json").read_text(
        encoding="utf-8"))
    if not validation["promoted"]:
        raise ValueError("V6 missing specialist failed the fixed local promotion gate")
    rows, x = load_subset(args)
    y = rows.target.to_numpy(dtype=float)
    schedule = x.schedule_unclipped_sec.to_numpy(dtype=float)
    positive = np.isfinite(y) & (y > 0)
    tail = positive & (y > 7200) & np.isfinite(schedule) & (schedule > 7200)
    ordinary_index = np.flatnonzero(positive)
    tail_index = np.flatnonzero(tail)
    tail_label = tail.astype(np.int8)
    ratio_label = np.clip(np.divide(y, schedule, out=np.zeros(len(y)),
                                    where=np.isfinite(schedule) & (schedule != 0)), 0, 4)
    ratio_weights = np.clip(schedule / 7200, 1, 12) ** 2
    ratio_weights[~np.isfinite(ratio_weights)] = 1
    models = {}
    reports = {}
    for name, index, label, weights in (
        ("ordinary", ordinary_index, y, None),
        ("tail_probability", ordinary_index, tail_label, None),
        ("tail_ratio", tail_index, ratio_label, ratio_weights)):
        fold_trees = [json.loads((args.output_dir / f"{fold}_training.json").read_text(
            encoding="utf-8"))["models"][name]["trees"] for fold in FOLDS]
        iterations = int(np.median(fold_trees))
        models[name], reports[name] = train_model(
            args, name, x, index, label, iterations, False, weights)
        models[name].save_model(str(args.output_dir / f"final_{name}.cbm"))
        reports[name]["fold_trees"] = fold_trees
    del rows, x, y, schedule, positive, tail
    gc.collect()

    rank_rows, rank_x = load_subset(args, ranking=True)
    rank_schedule = rank_x.schedule_unclipped_sec.to_numpy(dtype=float)
    candidate, tail_p, ratio = predict_mixture(
        models, rank_x, rank_schedule, args.threads)
    raw = pd.DataFrame({"MVT_ID_mvt": rank_rows.MVT_ID_mvt,
                        "candidate": candidate,
                        "tail_probability": tail_p,
                        "tail_ratio": ratio})
    raw.to_parquet(args.output_dir / "ranking_expert.parquet", index=False)
    template = pd.read_parquet(args.data_dir / "submitting.parquet",
                               columns=["MVT_ID_mvt"])
    base = pd.read_parquet(args.v5_ranking,
                           columns=["MVT_ID_mvt", "TAXITIME_SEC_mvt"])
    if (len(base) != len(template)
            or not np.array_equal(base.MVT_ID_mvt.to_numpy(),
                                  template.MVT_ID_mvt.to_numpy())
            or base.MVT_ID_mvt.duplicated().any()):
        raise ValueError("Frozen v5 ranking file is not template aligned")
    aligned = base.merge(raw, on="MVT_ID_mvt", how="left", validate="one_to_one")
    gate = aligned.candidate.notna().to_numpy(dtype=bool)
    if gate.sum() != len(raw) or not np.isfinite(aligned.candidate.to_numpy()[gate]).all():
        raise ValueError("V6 ranking expert is incomplete")
    old = base.TAXITIME_SEC_mvt.to_numpy(dtype=float)
    output = old.copy()
    weight = float(validation["selected_weight_from_jan_jul"])
    output[gate] = np.maximum(old[gate] + weight * (
        aligned.candidate.to_numpy(dtype=float)[gate] - old[gate]), 0)
    if (not np.isfinite(output).all() or np.any(output < 0)
            or not np.array_equal(output[~gate], old[~gate])):
        raise ValueError("V6 ranking output changed a non-gate row or is invalid")
    pd.DataFrame({"MVT_ID_mvt": template.MVT_ID_mvt,
                  "TAXITIME_SEC_mvt": output}).to_parquet(
                      args.output_dir / "predictions.parquet", index=False)
    manifest = {"selected_weight": weight, "ranking_rows": len(template),
                "ranking_gate_n": int(gate.sum()),
                "changed_n": int(np.count_nonzero(output != old)),
                "models": reports, "finite_nonnegative": True,
                "exact_template_order": True, "no_upload_performed": True,
                "prediction_file": str((args.output_dir / "predictions.parquet").resolve())}
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("fit-oof", "evaluate", "fit-final"),
                        default="fit-oof")
    parser.add_argument("--fold", choices=(*FOLDS, "both"), default="both")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--ordinary-iterations", type=int, default=700)
    parser.add_argument("--class-iterations", type=int, default=650)
    parser.add_argument("--ratio-iterations", type=int, default=450)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--cache-dir", type=Path,
                        default=Path("artifacts/baseline"))
    parser.add_argument("--arrival-dir", type=Path,
                        default=Path("artifacts/v5-arrival-clean"))
    parser.add_argument("--weather-file", type=Path,
                        default=Path("data/external/weather.parquet"))
    parser.add_argument("--v5-oof", type=Path,
                        default=Path("artifacts/v5-ensemble/validation_predictions.parquet"))
    parser.add_argument("--v5-ranking", type=Path,
                        default=Path("artifacts/v5-ensemble/predictions.parquet"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("artifacts/v6-missing"))
    args = parser.parse_args()
    if not 1 <= args.threads <= 4:
        raise ValueError("The CPU fit must use at most four threads")
    if args.mode == "fit-oof":
        reports = fit_oof(args)
        print(json.dumps({fold: {"test_n": v["test_n"],
                                 "positive_train": v["positive_train"],
                                 "tail_train": v["tail_train"]}
                          for fold, v in reports.items()}, indent=2))
    elif args.mode == "evaluate":
        report = evaluate(args)
        print(json.dumps({k: report[k] for k in (
            "selected_weight_from_jan_jul", "promoted",
            "fold_scores_all_finite", "day_bootstrap_selected_vs_v5")}, indent=2))
    else:
        print(json.dumps(fit_final(args), indent=2))


if __name__ == "__main__":
    main()
