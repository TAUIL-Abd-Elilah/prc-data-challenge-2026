"""Held-out, ranking-safe classifier for competing off-block timestamps.

On a valid NM AOBT row, estimate whether LOBT, IOBT, or the airport schedule
is the source of the airport taxi-out label. The target is used only to make
2025 source-class labels and to score held-out predictions. No departure block
time, ranking label, or leaderboard response is read.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


FOLDS = {"seasonal_jan_jul": (1, 7), "forward_nov_dec": (11, 12)}
SCALES = (0.0, 0.25, 0.5, 1.0)
SOURCE_NAMES = ("LOBT", "IOBT", "SCHED")
SOURCE_COLS = ("takeoff_minus_LOBT_flt", "takeoff_minus_IOBT_flt",
               "takeoff_minus_SCHED_TIME_UTC_mvt")
FEATURE_COLS = (
    "dep_prev15", "dep_prev30", "dep_next15", "arr_prev15", "arr_prev30",
    "arr_next15", "same_runway_prev15", "taxi_backlog_at_aobt",
    "same_runway_backlog_at_aobt", "takeoff_minus_SCHED_TIME_UTC_mvt",
    "takeoff_minus_AOBT_3_flt", "takeoff_minus_LOBT_flt",
    "takeoff_minus_IOBT_flt", "takeoff_minus_EOBT_1_flt",
    "takeoff_minus_ARVT_1_flt", "takeoff_minus_ARVT_3_flt",
    "delta_AOBT_3_flt_LOBT_flt", "delta_AOBT_3_flt_IOBT_flt",
    "delta_AOBT_3_flt_EOBT_1_flt", "delta_LOBT_flt_EOBT_1_flt",
    "delta_SCHED_TIME_UTC_mvt_LOBT_flt", "proxy_seconds", "utc_hour",
    "utc_weekday", "utc_month", "local_hour", "flight_number",
    "flight_airport_agrees", "aircraft_type_agrees", "destination_agrees",
    "ADEP_mvt", "ADES_mvt", "ADES_FILED_flt", "RUNWAY_mvt", "STAND_mvt",
    "AIRCRAFT_TYPE_mvt", "AIRCRAFT_OPERATOR_flt", "MARKET_SEGMENT_flt",
    "WK_TBL_CAT_flt", "FLIGHT_TYPE_flt", "flight_prefix",
    "callsign_prefix", "route", "airport_stand_runway",
)
MAX_ALTERNATE_SEC = 7200.0
MIN_SEPARATION_SEC = 600.0
MIN_IOBT_LOBT_SEPARATION_SEC = 120.0
SOURCE_EXACT_SEC = 60.0
CORRECTION_CAP_SEC = 3600.0


def rmse(y: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - prediction) ** 2)))


def options(features: pd.DataFrame, proxy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    alternatives = features.loc[:, list(SOURCE_COLS)].to_numpy(dtype=np.float64)
    plausible = (np.isfinite(alternatives) & (alternatives >= 0) &
                 (alternatives <= MAX_ALTERNATE_SEC) &
                 (np.abs(alternatives - proxy[:, None]) > MIN_SEPARATION_SEC))
    # LOBT and IOBT often coincide; avoid duplicate source classes.
    plausible[:, 1] &= (~plausible[:, 0] |
                        (np.abs(alternatives[:, 1] - alternatives[:, 0]) >
                         MIN_IOBT_LOBT_SEPARATION_SEC))
    valid_proxy = np.isfinite(proxy) & (proxy >= 0) & (proxy <= 7200)
    candidate = valid_proxy & plausible.any(axis=1)
    return alternatives, plausible, candidate


def build_features(features: pd.DataFrame, plausible: np.ndarray) -> pd.DataFrame:
    x = features.loc[:, list(FEATURE_COLS)].copy()
    aobt = x["takeoff_minus_AOBT_3_flt"].to_numpy(dtype=np.float64)
    lobt = x["takeoff_minus_LOBT_flt"].to_numpy(dtype=np.float64)
    iobt = x["takeoff_minus_IOBT_flt"].to_numpy(dtype=np.float64)
    sched = x["takeoff_minus_SCHED_TIME_UTC_mvt"].to_numpy(dtype=np.float64)
    arv1 = x["takeoff_minus_ARVT_1_flt"].to_numpy(dtype=np.float64)
    arv3 = x["takeoff_minus_ARVT_3_flt"].to_numpy(dtype=np.float64)
    for name, values in (
        ("lobt_minus_iobt", lobt - iobt),
        ("sched_minus_aobt", sched - aobt),
        ("sched_minus_iobt", sched - iobt),
        ("arrival_m3_minus_m1", arv3 - arv1),
    ):
        x[name] = values.astype(np.float32)
    for j, source in enumerate(SOURCE_NAMES):
        x[f"{source.lower()}_candidate"] = plausible[:, j].astype(np.int8)
    x["alternate_count"] = plausible.sum(axis=1).astype(np.int8)
    forbidden = [name for name in x if ("TAXITIME" in name.upper() or
                                        "BLOCK_TIME" in name.upper() or
                                        name.lower() == "target")]
    if forbidden:
        raise ValueError(f"Forbidden label feature: {forbidden}")
    return x


def source_labels(y: np.ndarray, alternatives: np.ndarray,
                  plausible: np.ndarray) -> np.ndarray:
    distance = np.where(plausible, np.abs(alternatives - y[:, None]), np.inf)
    label = np.argmin(distance, axis=1).astype(np.int8) + 1
    label[distance.min(axis=1) > SOURCE_EXACT_SEC] = 0
    return label


def load_candidates(cache_dir: Path, ranking: bool) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    prefix = "ranking" if ranking else "training"
    rows = pd.read_parquet(cache_dir / f"{prefix}_rows.parquet")
    feature_path = cache_dir / ("ranking_features.parquet" if ranking else "features.parquet")
    features = pd.read_parquet(feature_path, columns=list(FEATURE_COLS))
    if len(rows) != len(features) or rows.MVT_ID_mvt.isna().any() or rows.MVT_ID_mvt.duplicated().any():
        raise ValueError("Cached row/feature alignment or movement IDs are invalid")
    alternatives, plausible, candidate = options(features, rows.proxy.to_numpy(dtype=np.float64))
    idx = np.flatnonzero(candidate)
    selected_rows = rows.iloc[idx].reset_index(drop=True)
    selected_features = build_features(features.iloc[idx].reset_index(drop=True), plausible[idx])
    return selected_rows, selected_features, alternatives[idx], plausible[idx]


def params(threads: int) -> dict:
    return dict(objective="multiclass", num_class=4, metric="multi_logloss",
                learning_rate=0.045, num_leaves=31, min_data_in_leaf=90,
                lambda_l2=20.0, feature_fraction=0.85, bagging_fraction=0.85,
                bagging_freq=1, cat_smooth=25, max_cat_threshold=64,
                verbosity=-1, num_threads=threads, seed=2026,
                feature_fraction_seed=2026, bagging_seed=2026,
                deterministic=True, force_col_wise=True)


def sample_training(labels: np.ndarray, eligible: np.ndarray, max_rows: int,
                    seed: int) -> tuple[np.ndarray, np.ndarray]:
    eligible_idx = np.flatnonzero(eligible)
    positive = eligible_idx[labels[eligible_idx] > 0]
    none = eligible_idx[labels[eligible_idx] == 0]
    if len(positive) >= max_rows:
        raise ValueError("Positive source labels exceed the training sample budget")
    rng = np.random.default_rng(seed)
    keep_none = (none if len(eligible_idx) <= max_rows else
                 rng.choice(none, size=max_rows - len(positive), replace=False))
    selected = np.concatenate([positive, keep_none])
    rng.shuffle(selected)
    weights = np.ones(len(selected), dtype=np.float64)
    weights[labels[selected] == 0] = len(none) / len(keep_none)
    return selected, weights


def fit_fold(name: str, months: tuple[int, int], rows: pd.DataFrame,
             x: pd.DataFrame, alternatives: np.ndarray, plausible: np.ndarray,
             args: argparse.Namespace) -> dict:
    y = rows.target.to_numpy(dtype=np.float64)
    month = rows.month.to_numpy(dtype=np.int16)
    labels = source_labels(y, alternatives, plausible)
    heldout = np.isin(month, months)
    core = np.isfinite(y) & (y >= 0) & (y <= 86400)
    sampled, weights = sample_training(labels, core & ~heldout,
                                       args.max_train_rows, args.seed)
    rng = np.random.default_rng(args.seed + (1 if name == "seasonal_jan_jul" else 2))
    early_parts, train_parts = [], []
    for klass in range(4):
        part = sampled[labels[sampled] == klass]
        part = rng.permutation(part)
        n_early = max(1, int(len(part) * 0.1))
        early_parts.append(part[:n_early])
        train_parts.append(part[n_early:])
    early_idx = np.concatenate(early_parts)
    train_idx = np.concatenate(train_parts)
    weight_by_index = pd.Series(weights, index=sampled)
    cats = x.select_dtypes(include="category").columns.tolist()
    train = lgb.Dataset(x.iloc[train_idx], label=labels[train_idx],
                        weight=weight_by_index.loc[train_idx].to_numpy(),
                        categorical_feature=cats, free_raw_data=True)
    early = lgb.Dataset(x.iloc[early_idx], label=labels[early_idx],
                        weight=weight_by_index.loc[early_idx].to_numpy(),
                        reference=train, categorical_feature=cats, free_raw_data=True)
    start = time.monotonic()
    model = lgb.train(params(args.threads), train, num_boost_round=args.rounds,
                      valid_sets=[early], callbacks=[
                          lgb.early_stopping(60, verbose=False),
                          lgb.log_evaluation(period=100)])
    elapsed = time.monotonic() - start
    query = np.flatnonzero(heldout & np.isfinite(y))
    prob = model.predict(x.iloc[query], num_threads=args.threads)
    if prob.shape != (len(query), 4) or not np.isfinite(prob).all():
        raise ValueError("Classifier probabilities are incomplete")
    output = rows.iloc[query][["MVT_ID_mvt", "target", "month", "airport", "time"]].copy()
    output["fold"] = name
    for j, source in enumerate(SOURCE_NAMES):
        output[f"{source.lower()}_proxy_sec"] = alternatives[query, j]
        output[f"{source.lower()}_plausible"] = plausible[query, j]
        output[f"p_{source.lower()}_exact"] = prob[:, j + 1]
    output.to_parquet(args.output_dir / f"{name}_oof.parquet", index=False)
    model.save_model(str(args.output_dir / f"{name}.txt"))
    return {"fold": name, "training_candidates": int((core & ~heldout).sum()),
            "sampled_rows": int(len(sampled)), "heldout_candidate_rows": int(len(query)),
            "sampled_class_counts": {str(c): int((labels[sampled] == c).sum()) for c in range(4)},
            "best_iteration": int(model.best_iteration), "fit_seconds": round(elapsed, 3),
            "internal_validation_logloss": float(model.best_score["valid_0"]["multi_logloss"])}


def correction(frame: pd.DataFrame) -> np.ndarray:
    base = frame.selected.to_numpy(dtype=np.float64)
    delta = np.zeros(len(frame), dtype=np.float64)
    for source in SOURCE_NAMES:
        lower = source.lower()
        plausible = frame[f"{lower}_plausible"].fillna(False).to_numpy(dtype=bool)
        probability = frame[f"p_{lower}_exact"].fillna(0).to_numpy(dtype=np.float64)
        alternative = frame[f"{lower}_proxy_sec"].to_numpy(dtype=np.float64)
        difference = np.zeros(len(frame), dtype=np.float64)
        difference[plausible] = np.clip(alternative[plausible] - base[plausible],
                                       -CORRECTION_CAP_SEC, CORRECTION_CAP_SEC)
        delta += probability * difference
    return delta


def day_bootstrap(frame: pd.DataFrame, baseline: np.ndarray,
                  candidate: np.ndarray, seed: int, repetitions: int) -> dict:
    work = pd.DataFrame({"month": frame.month.to_numpy(dtype=np.int16),
                         "day": pd.to_datetime(frame.MVT_TIME_UTC_mvt, utc=True).dt.date,
                         "base_sse": (frame.target.to_numpy(dtype=float) - baseline) ** 2,
                         "candidate_sse": (frame.target.to_numpy(dtype=float) - candidate) ** 2})
    days = work.groupby(["month", "day"], sort=True).agg(
        n=("base_sse", "size"), base_sse=("base_sse", "sum"),
        candidate_sse=("candidate_sse", "sum")).reset_index()
    rng = np.random.default_rng(seed)
    gains = np.empty(repetitions, dtype=np.float64)
    blocks = [group[["n", "base_sse", "candidate_sse"]].to_numpy(dtype=float)
              for _, group in days.groupby("month", sort=True)]
    for iteration in range(repetitions):
        selected = np.concatenate([block[rng.integers(0, len(block), size=len(block))]
                                   for block in blocks], axis=0)
        total_n = selected[:, 0].sum()
        gains[iteration] = (np.sqrt(selected[:, 1].sum() / total_n) -
                            np.sqrt(selected[:, 2].sum() / total_n))
    return {"repetitions": repetitions, "calendar_days": int(len(days)),
            "rmse_gain_ci95_sec": np.quantile(gains, [0.025, 0.975]).tolist(),
            "positive_fraction": float((gains > 0).mean()),
            "days_with_positive_sse_gain": int((days.base_sse > days.candidate_sse).sum())}


def evaluate(args: argparse.Namespace, fold_reports: dict) -> dict:
    oof = pd.concat([pd.read_parquet(args.output_dir / f"{name}_oof.parquet")
                     for name in FOLDS], ignore_index=True)
    if oof.MVT_ID_mvt.duplicated().any():
        raise ValueError("Duplicate multisource held-out IDs")
    base = pd.read_parquet(args.v4_oof)
    if base.MVT_ID_mvt.duplicated().any() or (base.selected < 0).any():
        raise ValueError("Frozen v4 reference IDs or nonnegative prediction are invalid")
    frame = base.merge(oof, on="MVT_ID_mvt", how="left", validate="one_to_one",
                       suffixes=("", "_candidate"))
    matched = frame.p_lobt_exact.notna().to_numpy()
    if (not np.array_equal(frame.loc[matched, "target"].to_numpy(dtype=float),
                           frame.loc[matched, "target_candidate"].to_numpy(dtype=float)) or
        not frame.loc[matched, "fold"].eq(frame.loc[matched, "fold_candidate"]).all()):
        raise ValueError("Multisource OOF IDs, targets, or held-out folds disagree with v4")
    if "MVT_TIME_UTC_mvt" in frame:
        time_col = "MVT_TIME_UTC_mvt"
    else:
        time_col = "time"
    if time_col not in frame:
        raise ValueError("Frozen v4 OOF must carry movement time for day stability")
    frame["MVT_TIME_UTC_mvt"] = frame[time_col]
    frame["delta"] = correction(frame)
    y = frame.target.to_numpy(dtype=float)
    baseline = frame.selected.to_numpy(dtype=float)
    delta = frame.delta.to_numpy(dtype=float)
    seasonal = frame.fold.eq("seasonal_jan_jul").to_numpy()
    forward = frame.fold.eq("forward_nov_dec").to_numpy()
    if not np.isfinite(y).all() or not (seasonal | forward).all():
        raise ValueError("v4 OOF must contain exactly all finite held-out labels")
    scale_scores = {str(scale): rmse(y[seasonal], np.maximum(baseline[seasonal] +
                              scale * delta[seasonal], 0)) for scale in SCALES}
    chosen = min(SCALES, key=lambda scale: scale_scores[str(scale)])
    hashes = pd.util.hash_pandas_object(frame.MVT_ID_mvt, index=False).to_numpy(dtype=np.uint64)
    outer = (hashes % 5).astype(np.int8)
    nested = baseline.copy()
    outer_choices = []
    for part in range(5):
        fit = seasonal & (outer != part)
        test = seasonal & (outer == part)
        fit_scores = {str(scale): rmse(y[fit], np.maximum(baseline[fit] +
                                    scale * delta[fit], 0)) for scale in SCALES}
        choice = min(SCALES, key=lambda scale: fit_scores[str(scale)])
        nested[test] = np.maximum(baseline[test] + choice * delta[test], 0)
        outer_choices.append({"outer_fold": part, "selected_scale": choice,
                              "test_n": int(test.sum())})
    nested[forward] = np.maximum(baseline[forward] + chosen * delta[forward], 0)
    fold_results = {}
    for name, mask in (("seasonal_jan_jul", seasonal), ("forward_nov_dec", forward)):
        sub = frame.loc[mask]
        b = baseline[mask]
        c = nested[mask]
        yy = y[mask]
        fold_results[name] = {
            "all_finite_n": int(mask.sum()), "candidate_n": int(matched[mask].sum()),
            "baseline_rmse_sec": rmse(yy, b), "corrected_rmse_sec": rmse(yy, c),
            "rmse_gain_sec": rmse(yy, b) - rmse(yy, c),
            "date_stability": day_bootstrap(sub, b, c,
                                              args.seed + (1 if name == "seasonal_jan_jul" else 2),
                                              args.bootstrap_repetitions),
            "by_airport": {str(a): {"n": int(len(group)),
                                    "baseline_rmse_sec": rmse(group.target.to_numpy(dtype=float),
                                                              baseline[group.index.to_numpy()]),
                                    "corrected_rmse_sec": rmse(group.target.to_numpy(dtype=float),
                                                               nested[group.index.to_numpy()])}
                           for a, group in sub.groupby("airport")},
        }
    report = {"rule": "valid AOBT; plausible alternate in [0,7200] separated by >600 s; distinct IOBT >120 s from LOBT; posterior mean correction clipped to +/-3600 s",
              "source_label": "nearest plausible LOBT/IOBT/schedule within 60 s of 2025 taxi-time label, else none",
              "selection": "January/July scale; nested five-way seasonal estimate; unchanged November/December forward check",
              "scale_scores_seasonal_all_finite": scale_scores,
              "selected_scale": chosen, "nested_outer_choices": outer_choices,
              "fold_models": fold_reports, "folds": fold_results,
              "pooled": {"all_finite_n": int(len(frame)),
                         "baseline_rmse_sec": rmse(y, baseline),
                         "corrected_rmse_sec": rmse(y, nested)},
              "validation_note": "Repeatedly examined 2025 labels; descriptive, not an untouched 2026 estimate. No leaderboard feedback used."}
    (args.output_dir / "validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame({"MVT_ID_mvt": frame.MVT_ID_mvt, "target": y,
                  "fold": frame.fold, "v4": baseline, "candidate": nested,
                  "source_candidate": matched}).to_parquet(
                      args.output_dir / "validation_predictions.parquet", index=False)
    return report


def run_fit(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, x, alternatives, plausible = load_candidates(args.cache_dir, ranking=False)
    print(json.dumps({"candidate_rows": len(rows), "feature_count": len(x.columns)}), flush=True)
    reports = {}
    for name, months in FOLDS.items():
        print(f"Fitting multisource classifier for {name}", flush=True)
        reports[name] = fit_fold(name, months, rows, x, alternatives, plausible, args)
        print(json.dumps(reports[name]), flush=True)
    report = evaluate(args, reports)
    print(json.dumps({"selected_scale": report["selected_scale"],
                      "folds": {name: {key: part[key] for key in
                                       ("baseline_rmse_sec", "corrected_rmse_sec", "rmse_gain_sec")}
                                for name, part in report["folds"].items()}}, indent=2), flush=True)


def run_rank(args: argparse.Namespace) -> None:
    report = json.loads((args.output_dir / "validation.json").read_text(encoding="utf-8"))
    scale = float(report["selected_scale"])
    if scale <= 0:
        raise ValueError("Seasonal validation selected zero correction; no ranking fit is warranted")
    rows, x, alternatives, plausible = load_candidates(args.cache_dir, ranking=False)
    y = rows.target.to_numpy(dtype=float)
    labels = source_labels(y, alternatives, plausible)
    eligible = np.isfinite(y) & (y >= 0) & (y <= 86400)
    sampled, weights = sample_training(labels, eligible, args.max_train_rows, args.seed)
    rounds = int(np.median([report["fold_models"][fold]["best_iteration"] for fold in FOLDS]))
    categories = x.select_dtypes(include="category").columns.tolist()
    dataset = lgb.Dataset(x.iloc[sampled], label=labels[sampled], weight=weights,
                          categorical_feature=categories, free_raw_data=True)
    model = lgb.train(params(args.threads), dataset, num_boost_round=rounds)
    model.save_model(str(args.output_dir / "final.txt"))
    feature_categories = {name: x[name].cat.categories for name in categories}
    rank_rows, rank_x, rank_alternatives, rank_plausible = load_candidates(args.cache_dir, ranking=True)
    if list(rank_x.columns) != list(x.columns):
        raise ValueError("Training and ranking feature columns differ")
    for name, values in feature_categories.items():
        rank_x[name] = rank_x[name].cat.set_categories(values)
    prob = model.predict(rank_x, num_threads=args.threads)
    candidate = rank_rows[["MVT_ID_mvt"]].copy()
    for j, source in enumerate(SOURCE_NAMES):
        lower = source.lower()
        candidate[f"{lower}_proxy_sec"] = rank_alternatives[:, j]
        candidate[f"{lower}_plausible"] = rank_plausible[:, j]
        candidate[f"p_{lower}_exact"] = prob[:, j + 1]
    base = pd.read_parquet(args.v4_ranking, columns=["MVT_ID_mvt", "TAXITIME_SEC_mvt"])
    if base.MVT_ID_mvt.duplicated().any() or (base.TAXITIME_SEC_mvt < 0).any():
        raise ValueError("Frozen v4 ranking predictions are invalid")
    combined = base.merge(candidate, on="MVT_ID_mvt", how="left", validate="one_to_one", sort=False)
    combined = combined.rename(columns={"TAXITIME_SEC_mvt": "selected"})
    delta = correction(combined)
    prediction = np.maximum(combined.selected.to_numpy(dtype=float) + scale * delta, 0)
    if not np.isfinite(prediction).all() or len(prediction) != len(base):
        raise ValueError("Final multisource prediction is incomplete")
    out = pd.DataFrame({"MVT_ID_mvt": base.MVT_ID_mvt,
                        "TAXITIME_SEC_mvt": prediction})
    out.to_parquet(args.output_dir / "ranking_predictions.parquet", index=False)
    manifest = {"selected_scale": scale, "final_rounds": rounds,
                "training_candidate_rows": int(eligible.sum()),
                "sampled_training_rows": int(len(sampled)),
                "ranking_candidate_rows": int(len(rank_rows)),
                "ranking_prediction_rows": int(len(out)),
                "changed_rows": int((prediction != base.TAXITIME_SEC_mvt.to_numpy(dtype=float)).sum())}
    (args.output_dir / "ranking_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("fit", "rank"), default="fit")
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/baseline"))
    parser.add_argument("--v4-oof", type=Path, default=Path("artifacts/v4/validation_predictions.parquet"))
    parser.add_argument("--v4-ranking", type=Path,
                        default=Path("artifacts/catboost/source/sequential_predictions.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/v5-source"))
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=400)
    parser.add_argument("--max-train-rows", type=int, default=300000)
    parser.add_argument("--bootstrap-repetitions", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.threads > 4:
        raise ValueError("This CPU expert is limited to four host threads")
    if args.mode == "rank":
        run_rank(args)
    else:
        run_fit(args)


if __name__ == "__main__":
    main()
