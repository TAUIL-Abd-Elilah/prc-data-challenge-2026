"""Local CatBoost specialists for invalid AOBT taxi-out predictions.

Three predeclared experts are tested: ordinary no-NM direct regression,
Rome long-schedule ratio regression, and Rome schedule-exact classification.
Only released prediction-time covariates enter the models. The frozen v4 OOF
predictions are the comparator, with January/July policy selection and
November/December transfer assessment.
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

from solution import _training_files
from weather_model import add_weather


FOLDS = {"seasonal_jan_jul": (1, 7), "forward_nov_dec": (11, 12)}
POLICIES = ("v4", "direct_half", "direct_full", "ratio_half", "mixture_half",
            "direct_half_ratio_half", "direct_half_mixture_half")
DIRECT_CATEGORICAL = ("ADEP_mvt", "ADES_mvt", "RUNWAY_mvt", "STAND_mvt",
                      "AIRCRAFT_TYPE_mvt", "MARKET_SEGMENT_flt", "flight_prefix",
                      "flight_name", "stand_zone", "airport_runway", "route")
LONG_CATEGORICAL = ("ADES_mvt", "RUNWAY_mvt", "AIRCRAFT_TYPE_mvt",
                    "MARKET_SEGMENT_flt", "flight_prefix", "flight_name",
                    "stand_zone")
NUMERIC = ("dep_prev5", "dep_prev15", "dep_prev30", "dep_next15",
           "arr_prev15", "arr_prev30", "arr_next15", "same_runway_prev15",
           "takeoff_minus_SCHED_TIME_UTC_mvt", "takeoff_minus_ARVT_1_flt",
           "takeoff_minus_ARVT_3_flt", "utc_hour", "utc_weekday", "utc_month",
           "utc_dayofyear", "local_hour", "local_weekday", "hour_sin",
           "hour_cos", "year_sin", "year_cos", "schedule_proxy_unclipped",
           "schedule_day_offset")


def rmse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - p) ** 2)))


def load_inputs(args: argparse.Namespace, ranking: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    stem = "ranking" if ranking else "training"
    rows = pd.read_parquet(args.cache_dir / f"{stem}_rows.parquet")
    features = pd.read_parquet(args.cache_dir / (
        "ranking_features.parquet" if ranking else "features.parquet"))
    if len(rows) != len(features):
        raise ValueError("Cached rows and features differ in length")
    if any("BLOCK_TIME" in col.upper() or "TAXITIME" in col.upper()
           for col in features.columns):
        raise ValueError("Forbidden label-derived feature in cache")
    features = add_weather(features, rows, args.weather_file)
    files = [args.data_dir / "ranking.parquet"] if ranking else _training_files(args.data_dir)
    raw = (pl.scan_parquet([str(path) for path in files])
           .filter(pl.col("PHASE_mvt") == "DEP")
           .select(["MVT_ID_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
                    "FLIGHT_mvt"]).collect().to_pandas())
    if not np.array_equal(raw.MVT_ID_mvt.to_numpy(), rows.MVT_ID_mvt.to_numpy()):
        raise ValueError("Raw schedule/flight rows are not aligned with cached IDs")
    raw_schedule = (pd.to_datetime(raw.MVT_TIME_UTC_mvt, utc=True) -
                    pd.to_datetime(raw.SCHED_TIME_UTC_mvt, utc=True)).dt.total_seconds()
    features["schedule_proxy_unclipped"] = raw_schedule.astype("float32")
    features["schedule_day_offset"] = np.floor(raw_schedule / 86400).astype("float32")
    features["flight_name"] = raw.FLIGHT_mvt.astype("string").fillna("?").astype("category")
    if not np.isfinite(features.schedule_proxy_unclipped.dropna()).all():
        raise ValueError("Non-finite raw schedule proxy")
    return rows, features


def columns(features: pd.DataFrame, long: bool) -> tuple[list[str], list[str]]:
    cats = LONG_CATEGORICAL if long else DIRECT_CATEGORICAL
    names = [col for col in NUMERIC if col in features]
    names += [col for col in features if col.startswith("wx_")]
    names += [col for col in cats if col in features]
    return names, [col for col in cats if col in features]


def masks(rows: pd.DataFrame, features: pd.DataFrame,
          months: tuple[int, int]) -> dict[str, np.ndarray]:
    y = rows.target.to_numpy(dtype=float)
    proxy = rows.proxy.to_numpy(dtype=float)
    schedule = features.schedule_proxy_unclipped.to_numpy(dtype=float)
    no_nm = (features.AOBT_3_flt_missing.to_numpy(dtype=bool)
             & features.LOBT_flt_missing.to_numpy(dtype=bool))
    invalid = ~np.isfinite(proxy)
    heldout = rows.month.isin(months).to_numpy(dtype=bool)
    ordinary_train = (no_nm & invalid & np.isfinite(y) & (y >= 0)
                      & (y <= 7200) & ~heldout)
    direct_test = (no_nm & invalid & ~rows.airport.eq("LIRF").to_numpy()
                   & np.isfinite(y) & heldout)
    lirf_long = (no_nm & invalid & rows.airport.eq("LIRF").to_numpy()
                 & np.isfinite(schedule) & (schedule > 7200))
    lirf_train = (no_nm & invalid & rows.airport.eq("LIRF").to_numpy()
                  & np.isfinite(schedule) & (schedule > 3600)
                  & np.isfinite(y) & (y >= 0) & ~heldout)
    lirf_test = lirf_long & np.isfinite(y) & heldout
    return {"ordinary_train": ordinary_train, "direct_test": direct_test,
            "lirf_train": lirf_train, "lirf_test": lirf_test,
            "no_nm": no_nm, "invalid": invalid, "heldout": heldout}


def params(args: argparse.Namespace, kind: str, iterations: int, depth: int) -> dict:
    return dict(loss_function="Logloss" if kind == "classifier" else "RMSE",
                eval_metric="Logloss" if kind == "classifier" else "RMSE",
                iterations=iterations, learning_rate=.045, depth=depth,
                l2_leaf_reg=20, random_strength=.5, bagging_temperature=.5,
                max_ctr_complexity=1, one_hot_max_size=20, border_count=128,
                thread_count=args.threads, used_ram_limit="6gb", random_seed=args.seed,
                allow_writing_files=False, verbose=100)


def train_model(args: argparse.Namespace, name: str, kind: str,
                features: pd.DataFrame, index: np.ndarray, target: np.ndarray,
                feature_names: list[str], cats: list[str],
                iterations: int, depth: int,
                sample_weight: np.ndarray | None = None) -> tuple[object, dict]:
    rng = np.random.default_rng(args.seed + len(name))
    shuffled = rng.permutation(index)
    n_early = max(50 if len(index) < 1000 else 500, int(.1 * len(index)))
    n_early = min(n_early, max(1, len(index) // 3))
    early_idx, fit_idx = shuffled[:n_early], shuffled[n_early:]
    x = features[feature_names]
    fit_weights = sample_weight[fit_idx] if sample_weight is not None else None
    early_weights = sample_weight[early_idx] if sample_weight is not None else None
    fit_pool = Pool(x.iloc[fit_idx], label=target[fit_idx],
                    weight=fit_weights, cat_features=cats)
    early_pool = Pool(x.iloc[early_idx], label=target[early_idx],
                      weight=early_weights, cat_features=cats)
    cls = CatBoostClassifier if kind == "classifier" else CatBoostRegressor
    model = cls(**params(args, kind, iterations, depth))
    start = time.monotonic()
    model.fit(fit_pool, eval_set=early_pool, use_best_model=True,
              early_stopping_rounds=70)
    elapsed = time.monotonic() - start
    report = {"model": name, "kind": kind, "train_rows": int(len(index)),
              "fit_rows": int(len(fit_idx)), "internal_early_rows": int(len(early_idx)),
              "best_iteration": int(model.get_best_iteration()),
              "fit_seconds": elapsed,
              "feature_names": feature_names, "categorical": cats}
    return model, report


def fit_fold(args: argparse.Namespace, fold: str, months: tuple[int, int],
             rows: pd.DataFrame, features: pd.DataFrame,
             v4: pd.DataFrame) -> dict:
    mask = masks(rows, features, months)
    y = rows.target.to_numpy(dtype=float)
    schedule = features.schedule_proxy_unclipped.to_numpy(dtype=float)
    direct_names, direct_cats = columns(features, long=False)
    long_names, long_cats = columns(features, long=True)
    ordinary = np.flatnonzero(mask["ordinary_train"])
    lirf = np.flatnonzero(mask["lirf_train"])
    if len(ordinary) < 1000 or len(lirf) < 100:
        raise ValueError("Too few complementary training rows")
    direct, direct_report = train_model(args, "ordinary_direct", "regression",
                                        features, ordinary, y,
                                        direct_names, direct_cats,
                                        args.direct_iterations, 6)
    ratio_y = np.divide(y, schedule, out=np.zeros_like(y),
                        where=np.isfinite(schedule) & (schedule != 0))
    ratio_y = np.clip(ratio_y, 0, 4)
    ratio_weight = np.clip(schedule / 3600, .5, 10) ** 2
    ratio, ratio_report = train_model(args, "lirf_ratio", "regression",
                                      features, lirf, ratio_y,
                                      long_names, long_cats,
                                      args.long_iterations, 4, ratio_weight)
    exact_y = (np.abs(y - schedule) <= 60).astype(np.int8)
    classifier, classifier_report = train_model(args, "lirf_exact", "classifier",
                                                 features, lirf, exact_y,
                                                 long_names, long_cats,
                                                 args.long_iterations, 4)

    heldout_invalid = (mask["heldout"] & mask["invalid"] & np.isfinite(y))
    test_idx = np.flatnonzero(heldout_invalid)
    output = rows.iloc[test_idx][["MVT_ID_mvt", "target", "airport", "month", "time"]].copy()
    output["direct_candidate"] = np.nan
    output["ratio_candidate"] = np.nan
    output["mixture_candidate"] = np.nan
    output["schedule_proxy_unclipped"] = schedule[test_idx]
    direct_idx = np.flatnonzero(mask["direct_test"])
    long_idx = np.flatnonzero(mask["lirf_test"])
    direct_pred = direct.predict(features.iloc[direct_idx][direct_names],
                                  thread_count=args.threads)
    ratio_pred = np.clip(ratio.predict(features.iloc[long_idx][long_names],
                                        thread_count=args.threads), 0, 4)
    ratio_pred = schedule[long_idx] * ratio_pred
    exact_p = classifier.predict_proba(features.iloc[long_idx][long_names],
                                       thread_count=args.threads)[:, 1]
    mixture_pred = exact_p * schedule[long_idx] + (1 - exact_p) * ratio_pred
    pos = pd.Index(test_idx)
    output.iloc[pos.get_indexer(direct_idx), output.columns.get_loc("direct_candidate")] = direct_pred
    output.iloc[pos.get_indexer(long_idx), output.columns.get_loc("ratio_candidate")] = ratio_pred
    output.iloc[pos.get_indexer(long_idx), output.columns.get_loc("mixture_candidate")] = mixture_pred
    output["p_schedule_exact"] = np.nan
    output.iloc[pos.get_indexer(long_idx), output.columns.get_loc("p_schedule_exact")] = exact_p
    output = output.merge(v4[["MVT_ID_mvt", "target", "selected"]].rename(
        columns={"selected": "v4_prediction"}),
        on="MVT_ID_mvt", how="left", validate="one_to_one", suffixes=("", "_v4"))
    if output.v4_prediction.isna().any() or not np.allclose(output.target, output.target_v4):
        raise ValueError("Frozen v4 OOF does not align with invalid-AOBT rows")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, model in (("ordinary_direct", direct), ("lirf_ratio", ratio),
                        ("lirf_exact", classifier)):
        model.save_model(str(args.output_dir / f"{fold}_{name}.cbm"))
    output.to_parquet(args.output_dir / f"{fold}_oof.parquet", index=False)
    report = {"fold": fold, "heldout_months": months,
              "heldout_invalid_n": int(len(output)),
              "direct_gate_n": int(len(direct_idx)), "lirf_long_gate_n": int(len(long_idx)),
              "models": [direct_report, ratio_report, classifier_report]}
    (args.output_dir / f"{fold}_training.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    return report


def predict_policy(frame: pd.DataFrame, policy: str) -> np.ndarray:
    pred = frame.v4_prediction.to_numpy(dtype=float).copy()
    direct = frame.direct_candidate.to_numpy(dtype=float)
    ratio = frame.ratio_candidate.to_numpy(dtype=float)
    mixture = frame.mixture_candidate.to_numpy(dtype=float)
    if "direct_half" in policy:
        gate = np.isfinite(direct)
        pred[gate] = .5 * pred[gate] + .5 * direct[gate]
    elif "direct_full" in policy:
        gate = np.isfinite(direct)
        pred[gate] = direct[gate]
    if "ratio_half" in policy:
        gate = np.isfinite(ratio)
        pred[gate] = .5 * pred[gate] + .5 * ratio[gate]
    elif "mixture_half" in policy:
        gate = np.isfinite(mixture)
        pred[gate] = .5 * pred[gate] + .5 * mixture[gate]
    return pred


def day_bootstrap(frame: pd.DataFrame, base: np.ndarray, candidate: np.ndarray,
                  mask: np.ndarray, seed: int) -> dict:
    subset = frame.loc[mask, ["MVT_TIME_UTC_mvt", "target"]].copy()
    yy = subset.target.to_numpy(dtype=float)
    base_error = (yy - base[mask]) ** 2
    new_error = (yy - candidate[mask]) ** 2
    day = pd.to_datetime(subset.MVT_TIME_UTC_mvt, utc=True).dt.floor("D")
    grouped = pd.DataFrame({"day": day, "base_sse": base_error,
                            "new_sse": new_error, "n": 1}).groupby("day").sum()
    values = grouped[["base_sse", "new_sse", "n"]].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(2000, len(values)))
    sampled = values[draws].sum(axis=1)
    delta = np.sqrt(sampled[:, 0] / sampled[:, 2]) - np.sqrt(sampled[:, 1] / sampled[:, 2])
    return {"days": int(len(values)),
            "observed_rmse_gain_sec": rmse(yy, base[mask]) - rmse(yy, candidate[mask]),
            "bootstrap_gain_95pct_interval_sec": np.quantile(delta, [.025, .975]).tolist(),
            "bootstrap_probability_improvement": float(np.mean(delta > 0)),
            "days_with_positive_sse_gain_fraction": float(np.mean(values[:, 0] > values[:, 1]))}


def evaluate_existing(args: argparse.Namespace) -> dict:
    paths = [args.output_dir / f"{fold}_oof.parquet" for fold in FOLDS]
    if not all(path.exists() for path in paths):
        raise FileNotFoundError("Both invalid-AOBT specialist OOF folds are required")
    specialist = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    if specialist.MVT_ID_mvt.duplicated().any():
        raise ValueError("Specialist OOF IDs are duplicated")
    # Always reload the current clipped v4 reference; older specialist OOF
    # files may contain an earlier diagnostic copy of v4 predictions.
    specialist = specialist.drop(columns=["target", "airport", "month", "time",
                                          "v4_prediction", "target_v4"], errors="ignore")
    v4 = pd.read_parquet(args.v4_dir / "validation_predictions.parquet",
                         columns=["MVT_ID_mvt", "target", "fold", "a_valid", "selected",
                                  "airport", "month", "MVT_TIME_UTC_mvt"])
    frame = v4.merge(specialist, on="MVT_ID_mvt", how="left", validate="one_to_one")
    frame = frame.rename(columns={"selected": "v4_prediction"})
    y = frame.target.to_numpy(dtype=float)
    base = frame.v4_prediction.to_numpy(dtype=float)
    valid = frame.a_valid.to_numpy(dtype=bool)
    seasonal = frame.fold.eq("seasonal_jan_jul").to_numpy(dtype=bool)
    forward = frame.fold.eq("forward_nov_dec").to_numpy(dtype=bool)
    if not (seasonal | forward).all() or not np.isfinite(y).all() or not np.isfinite(base).all():
        raise ValueError("Frozen v4 reference has incomplete labels or predictions")
    preds = {policy: np.maximum(predict_policy(frame, policy), 0) for policy in POLICIES}
    report = {"reference": "Clipped frozen v4 all-finite OOF",
              "policy_selection": "Nonoverlapping direct and LIRF long gates selected on Jan/Jul; require each gate to improve Nov/Dec before promotion",
              "policies": {}, "counts": {
                  "all_finite": int(len(frame)),
                  "invalid_aobt": int((~valid).sum()),
                  "direct_gate": int(frame.direct_candidate.notna().sum()),
                  "lirf_long_gate": int(frame.ratio_candidate.notna().sum())}}
    for policy, pred in preds.items():
        report["policies"][policy] = {}
        for fold, mask in (("seasonal_jan_jul", seasonal), ("forward_nov_dec", forward)):
            invalid_mask = mask & ~valid
            report["policies"][policy][fold] = {
                "all_finite_rmse_sec": rmse(y[mask], pred[mask]),
                "invalid_aobt_rmse_sec": rmse(y[invalid_mask], pred[invalid_mask]),
                "all_finite_n": int(mask.sum()),
                "invalid_aobt_n": int(invalid_mask.sum()),
                "day_bootstrap": day_bootstrap(frame, base, pred, mask, args.seed +
                                               (0 if fold == "seasonal_jan_jul" else 7)),
            }
    direct_options = ("v4", "direct_half", "direct_full")
    long_options = ("v4", "ratio_half", "mixture_half")
    direct_choice = min(direct_options, key=lambda name:
                        report["policies"][name]["seasonal_jan_jul"]["all_finite_rmse_sec"])
    long_choice = min(long_options, key=lambda name:
                      report["policies"][name]["seasonal_jan_jul"]["all_finite_rmse_sec"])
    promoted = []
    for choice in (direct_choice, long_choice):
        if choice != "v4" and all(
            report["policies"][choice][fold]["all_finite_rmse_sec"] <
            report["policies"]["v4"][fold]["all_finite_rmse_sec"]
            for fold in FOLDS):
            promoted.append(choice)
    report["seasonal_gate_choices"] = {"direct": direct_choice, "lirf_long": long_choice}
    report["promoted_after_forward_check"] = promoted
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "validation.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame({"MVT_ID_mvt": frame.MVT_ID_mvt, "target": y,
                  "fold": frame.fold, "airport": frame.airport,
                  "a_valid": valid, "v4": base,
                  **{f"policy_{name}": pred for name, pred in preds.items()}}).to_parquet(
                      args.output_dir / "validation_predictions.parquet", index=False)
    return report


def fit_final_direct(args: argparse.Namespace) -> dict:
    validation = json.loads((args.output_dir / "validation.json").read_text(
        encoding="utf-8"))
    if "direct_half" not in validation["promoted_after_forward_check"]:
        raise ValueError("The direct-half policy did not pass both local folds")
    fold_reports = [json.loads((args.output_dir / f"{fold}_training.json").read_text(
        encoding="utf-8")) for fold in FOLDS]
    iterations = int(np.median([part["models"][0]["best_iteration"] + 1
                                for part in fold_reports]))
    rows, features = load_inputs(args)
    y = rows.target.to_numpy(dtype=float)
    proxy = rows.proxy.to_numpy(dtype=float)
    no_nm = (features.AOBT_3_flt_missing.to_numpy(dtype=bool)
             & features.LOBT_flt_missing.to_numpy(dtype=bool))
    train = no_nm & ~np.isfinite(proxy) & np.isfinite(y) & (y >= 0) & (y <= 7200)
    idx = np.flatnonzero(train)
    direct_names, cats = columns(features, long=False)
    model = CatBoostRegressor(**params(args, "regression", iterations, 6))
    start = time.monotonic()
    model.fit(Pool(features.iloc[idx][direct_names], label=y[idx],
                   cat_features=cats))
    fit_seconds = time.monotonic() - start
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(str(args.output_dir / "final_ordinary_direct.cbm"))
    train_n = int(len(idx))
    del rows, features, y, proxy, no_nm, idx
    gc.collect()

    rank_rows, rank_features = load_inputs(args, ranking=True)
    rank_proxy = rank_rows.proxy.to_numpy(dtype=float)
    gate = (~np.isfinite(rank_proxy)
            & rank_features.AOBT_3_flt_missing.to_numpy(dtype=bool)
            & rank_features.LOBT_flt_missing.to_numpy(dtype=bool)
            & ~rank_rows.airport.eq("LIRF").to_numpy(dtype=bool))
    rank_idx = np.flatnonzero(gate)
    if columns(rank_features, long=False) != (direct_names, cats):
        raise ValueError("Ranking direct feature schema differs from training")
    direct = np.full(len(rank_rows), np.nan, dtype=float)
    direct[rank_idx] = model.predict(rank_features.iloc[rank_idx][direct_names],
                                     thread_count=args.threads)
    expert = pd.DataFrame({"MVT_ID_mvt": rank_rows.MVT_ID_mvt,
                           "ordinary_no_nm_gate": gate,
                           "catboost_direct_prediction": direct})
    expert.to_parquet(args.output_dir / "ranking_expert.parquet", index=False)

    base_file = args.v4_ranking
    base = pd.read_parquet(base_file,
                           columns=["MVT_ID_mvt", "TAXITIME_SEC_mvt"])
    combined = base.merge(expert, on="MVT_ID_mvt", how="left",
                          validate="one_to_one")
    combined = combined.merge(rank_rows[["MVT_ID_mvt", "proxy"]],
                              on="MVT_ID_mvt", how="left", validate="one_to_one")
    if len(combined) != len(base) or combined.ordinary_no_nm_gate.isna().any():
        raise ValueError("Ranking expert does not cover the frozen base IDs")
    old = combined.TAXITIME_SEC_mvt.to_numpy(dtype=float)
    aligned_proxy = combined.proxy.to_numpy(dtype=float)
    candidate_gate = combined.ordinary_no_nm_gate.to_numpy(dtype=bool)
    predicted = combined.catboost_direct_prediction.to_numpy(dtype=float)
    if not np.isfinite(predicted[candidate_gate]).all():
        raise ValueError("Non-finite direct prediction on a ranking gate row")
    revised = old.copy()
    revised[candidate_gate] = .5 * old[candidate_gate] + .5 * predicted[candidate_gate]
    revised = np.maximum(revised, 0)
    if not np.array_equal(revised[~candidate_gate], old[~candidate_gate]):
        raise ValueError("A non-gate ranking prediction changed")
    if not np.array_equal(revised[np.isfinite(aligned_proxy)], old[np.isfinite(aligned_proxy)]):
        raise ValueError("A valid-AOBT ranking prediction changed")
    template = pd.read_parquet(args.data_dir / "submitting.parquet",
                               columns=["MVT_ID_mvt"])
    output = pd.DataFrame({"MVT_ID_mvt": combined.MVT_ID_mvt,
                           "TAXITIME_SEC_mvt": revised})
    output = template.merge(output, on="MVT_ID_mvt", how="left",
                            validate="one_to_one", sort=False)
    if len(output) != len(template) or not np.isfinite(output.TAXITIME_SEC_mvt).all():
        raise ValueError("Ranking output is not template aligned or finite")
    output.to_parquet(args.output_dir / "predictions.parquet", index=False)
    manifest = {
        "purpose": "Local v5 invalid-AOBT candidate for review; no upload performed",
        "command": "python missing_catboost.py --fit-final-direct",
        "base_file": str(base_file.resolve()),
        "output_file": str((args.output_dir / "predictions.parquet").resolve()),
        "raw_expert_file": str((args.output_dir / "ranking_expert.parquet").resolve()),
        "model_file": str((args.output_dir / "final_ordinary_direct.cbm").resolve()),
        "iterations": iterations,
        "fold_best_iterations": {fold: part["models"][0]["best_iteration"]
                                 for fold, part in zip(FOLDS, fold_reports)},
        "training_rows": train_n,
        "fit_seconds": fit_seconds,
        "ranking_rows": int(len(output)),
        "ranking_gate_rows": int(candidate_gate.sum()),
        "ranking_changed_rows": int(np.count_nonzero(revised != old)),
        "blend": "0.5 * frozen base + 0.5 * CatBoost direct on no-NM invalid-AOBT non-LIRF rows",
        "non_gate_rows_unchanged": int((~candidate_gate).sum()),
        "valid_aobt_rows_unchanged": int(np.isfinite(aligned_proxy).sum()),
        "template_order_verified": True,
        "finite_nonnegative_verified": True,
        "validation": {fold: {
            "v4_all_finite_rmse_sec": validation["policies"]["v4"][fold]["all_finite_rmse_sec"],
            "v5_direct_half_all_finite_rmse_sec": validation["policies"]["direct_half"][fold]["all_finite_rmse_sec"]}
            for fold in FOLDS},
        "validation_note": "Both 2025 folds have supported local model comparisons; these are not untouched final estimates.",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/baseline"))
    parser.add_argument("--v4-dir", type=Path, default=Path("artifacts/v4"))
    parser.add_argument("--v4-ranking", type=Path,
                        default=Path("artifacts/catboost/source/sequential_predictions.parquet"))
    parser.add_argument("--weather-file", type=Path,
                        default=Path("data/external/weather.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/v5-missing"))
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--direct-iterations", type=int, default=700)
    parser.add_argument("--long-iterations", type=int, default=400)
    parser.add_argument("--fold", choices=(*FOLDS, "both"), default="both")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--fit-final-direct", action="store_true")
    args = parser.parse_args()
    if args.threads > 4:
        raise ValueError("Keep small CatBoost specialists at four CPU threads or fewer")
    if args.evaluate_only:
        report = evaluate_existing(args)
        print(json.dumps({"seasonal_gate_choices": report["seasonal_gate_choices"],
                          "promoted_after_forward_check": report["promoted_after_forward_check"],
                          "rmse": {policy: {fold: result[fold]["all_finite_rmse_sec"]
                                             for fold in FOLDS}
                                   for policy, result in report["policies"].items()}}, indent=2))
        return
    if args.fit_final_direct:
        manifest = fit_final_direct(args)
        print(json.dumps({key: manifest[key] for key in
                          ("iterations", "training_rows", "fit_seconds",
                           "ranking_rows", "ranking_gate_rows",
                           "ranking_changed_rows", "output_file")}, indent=2))
        return
    rows, features = load_inputs(args)
    v4 = pd.read_parquet(args.v4_dir / "validation_predictions.parquet",
                         columns=["MVT_ID_mvt", "target", "selected"])
    chosen = FOLDS if args.fold == "both" else {args.fold: FOLDS[args.fold]}
    reports = {}
    for fold, months in chosen.items():
        print(f"Training invalid-AOBT specialists: {fold}", flush=True)
        reports[fold] = fit_fold(args, fold, months, rows, features, v4)
        print(json.dumps({"fold": fold, "counts": [reports[fold]["heldout_invalid_n"],
                                                  reports[fold]["direct_gate_n"],
                                                  reports[fold]["lirf_long_gate_n"]],
                          "fit_seconds": [m["fit_seconds"] for m in reports[fold]["models"]]},
                         indent=2), flush=True)
    (args.output_dir / "training.json").write_text(
        json.dumps(reports, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
