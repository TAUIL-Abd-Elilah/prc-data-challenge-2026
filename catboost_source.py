"""Classify schedule-sourced labels for Rome and Istanbul departures.

The classifier uses only published prediction-time features. It is evaluated
on held-out 2025 months and leaves every other flight with the frozen v3 model.
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

from catboost_expert import add_flight_weather, load_cache, rmse


FOLDS = {"seasonal_jan_jul": (1, 7), "forward_nov_dec": (11, 12)}
SCALES = (0.0, .25, .5, 1.0)


def source_params(iterations: int, args: argparse.Namespace) -> dict:
    return dict(
        loss_function="Logloss", eval_metric="Logloss",
        iterations=iterations, learning_rate=.045, depth=args.depth,
        l2_leaf_reg=8, random_strength=.5, bagging_temperature=.5,
        max_ctr_complexity=1, one_hot_max_size=20, border_count=128,
        thread_count=args.threads, used_ram_limit="8gb", random_seed=args.seed,
        allow_writing_files=False, verbose=100,
    )


def all_metrics(y: np.ndarray, pred: np.ndarray, valid: np.ndarray) -> dict:
    return {"n": int(len(y)), "overall_rmse_sec": rmse(y, pred),
            "valid_n": int(valid.sum()),
            "valid_rmse_sec": rmse(y[valid], pred[valid]),
            "invalid_n": int((~valid).sum()),
            "invalid_rmse_sec": rmse(y[~valid], pred[~valid])}


def evaluate_existing(args: argparse.Namespace) -> dict:
    paths = [args.output_dir / f"{fold}_oof.parquet" for fold in FOLDS]
    if not all(path.exists() for path in paths):
        raise FileNotFoundError("Both source-classifier OOF folds are required")
    source = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    if source.MVT_ID_mvt.duplicated().any():
        raise ValueError("Source OOF IDs are duplicated")
    v3 = pd.read_parquet(args.v3_dir / "validation_predictions.parquet",
                         columns=["MVT_ID_mvt", "target", "fold", "a_valid", "selected"])
    frame = v3.merge(source[["MVT_ID_mvt", "target", "airport",
                             "schedule_proxy_sec", "p_schedule_exact"]],
                     on="MVT_ID_mvt", how="left", validate="one_to_one",
                     suffixes=("", "_source"))
    matched = frame.p_schedule_exact.notna().to_numpy()
    if not np.allclose(frame.loc[matched, "target"],
                       frame.loc[matched, "target_source"]):
        raise ValueError("Source OOF and v3 labels disagree")
    y = frame.target.to_numpy(dtype=float)
    base = frame.selected.to_numpy(dtype=float)
    valid = frame.a_valid.to_numpy(dtype=bool)
    p = frame.p_schedule_exact.fillna(0).to_numpy(dtype=float)
    schedule = frame.schedule_proxy_sec.fillna(0).to_numpy(dtype=float)
    delta = p * (schedule - base)
    seasonal = frame.fold.eq("seasonal_jan_jul").to_numpy()
    forward = frame.fold.eq("forward_nov_dec").to_numpy()
    if not (seasonal | forward).all():
        raise ValueError("Unknown v3 fold")
    seasonal_scores = {str(scale): rmse(y[seasonal], (base + scale * delta)[seasonal])
                       for scale in SCALES}
    selected_scale = min(SCALES, key=lambda scale: seasonal_scores[str(scale)])
    hashes = pd.util.hash_pandas_object(frame.MVT_ID_mvt, index=False).to_numpy(dtype=np.uint64)
    outer = (hashes % 5).astype(np.int8)
    nested = base.copy()
    outer_results = []
    for fold in range(5):
        train = seasonal & matched & (outer != fold)
        test = seasonal & matched & (outer == fold)
        scores = {str(scale): rmse(y[train], (base + scale * delta)[train])
                  for scale in SCALES}
        choice = min(SCALES, key=lambda scale: scores[str(scale)])
        nested[test] += choice * delta[test]
        outer_results.append({"outer_fold": fold, "selected_scale": choice,
                              "training_scores": scores,
                              "test_n": int(test.sum()),
                              "test_base_rmse_sec": rmse(y[test], base[test]),
                              "test_selected_rmse_sec": rmse(y[test], nested[test])})
    nested[forward] += selected_scale * delta[forward]
    selected = base + selected_scale * delta
    report = {
        "selection": "January/July only; unchanged scale applied to November/December",
        "selected_scale": selected_scale,
        "seasonal_candidate_scores_all_finite": seasonal_scores,
        "forward_candidate_scores_diagnostic": {
            str(scale): rmse(y[forward], (base + scale * delta)[forward])
            for scale in SCALES},
        "outer_folds": outer_results,
        "counts": {"all_finite_rows": int(len(frame)),
                   "source_candidate_rows": int(matched.sum()),
                   "seasonal_source_rows": int((matched & seasonal).sum()),
                   "forward_source_rows": int((matched & forward).sum())},
        "seasonal_nested": {
            "base": all_metrics(y[seasonal], base[seasonal], valid[seasonal]),
            "corrected": all_metrics(y[seasonal], nested[seasonal], valid[seasonal])},
        "forward_transfer": {
            "base": all_metrics(y[forward], base[forward], valid[forward]),
            "corrected": all_metrics(y[forward], nested[forward], valid[forward])},
        "combined_nested_plus_forward": {
            "base": all_metrics(y, base, valid),
            "corrected": all_metrics(y, nested, valid)},
        "source_subset": {},
        "source_by_airport_month": {},
    }
    for fold, month_mask in (("seasonal_jan_jul", seasonal),
                             ("forward_nov_dec", forward)):
        use = matched & month_mask
        report["source_subset"][fold] = {
            "n": int(use.sum()),
            "base_rmse_sec": rmse(y[use], base[use]),
            "corrected_rmse_sec": rmse(y[use], nested[use]),
        }
    source["corrected_at_frozen_scale"] = (
        source.v3_prediction + selected_scale * source.p_schedule_exact *
        (source.schedule_proxy_sec - source.v3_prediction))
    for (airport, month), group in source.groupby(["airport", "month"]):
        report["source_by_airport_month"][f"{airport}:{int(month)}"] = {
            "n": int(len(group)),
            "base_rmse_sec": rmse(group.target.to_numpy(dtype=float),
                                  group.v3_prediction.to_numpy(dtype=float)),
            "corrected_rmse_sec": rmse(group.target.to_numpy(dtype=float),
                                       group.corrected_at_frozen_scale.to_numpy(dtype=float)),
        }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "validation_combined.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame({"MVT_ID_mvt": frame.MVT_ID_mvt,
                  "target": y, "fold": frame.fold,
                  "a_valid": valid, "v3": base,
                  "p_schedule_exact": p,
                  "source_candidate": matched,
                  "corrected": nested}).to_parquet(
                      args.output_dir / "validation_predictions.parquet", index=False)
    return report


def evaluate_sequential_gpu(args: argparse.Namespace) -> dict:
    """Test fixed GPU .25 then schedule-source .5, with no retuning."""
    v3 = pd.read_parquet(args.v3_dir / "validation_predictions.parquet",
                         columns=["MVT_ID_mvt", "target", "fold", "a_valid", "selected"])
    gpu = pd.concat([pd.read_parquet(args.gpu_dir / f"{fold}_oof.parquet",
                                     columns=["MVT_ID_mvt", "target", "gpu_prediction"])
                     for fold in FOLDS], ignore_index=True)
    source = pd.concat([pd.read_parquet(args.output_dir / f"{fold}_oof.parquet",
                                        columns=["MVT_ID_mvt", "target",
                                                 "schedule_proxy_sec", "p_schedule_exact"])
                        for fold in FOLDS], ignore_index=True)
    if gpu.MVT_ID_mvt.duplicated().any() or source.MVT_ID_mvt.duplicated().any():
        raise ValueError("GPU or source OOF IDs are duplicated")
    frame = v3.merge(gpu, on="MVT_ID_mvt", how="left", validate="one_to_one",
                     suffixes=("", "_gpu"))
    frame = frame.merge(source, on="MVT_ID_mvt", how="left", validate="one_to_one",
                        suffixes=("", "_source"))
    gpu_available = frame.gpu_prediction.notna().to_numpy()
    source_available = frame.p_schedule_exact.notna().to_numpy()
    if not np.array_equal(gpu_available, frame.a_valid.to_numpy(dtype=bool)):
        raise ValueError("GPU OOF must cover exactly the valid AOBT rows")
    if not np.allclose(frame.loc[gpu_available, "target"],
                       frame.loc[gpu_available, "target_gpu"]):
        raise ValueError("GPU and v3 OOF targets disagree")
    if not np.allclose(frame.loc[source_available, "target"],
                       frame.loc[source_available, "target_source"]):
        raise ValueError("Source and v3 OOF targets disagree")
    y = frame.target.to_numpy(dtype=float)
    v3_pred = frame.selected.to_numpy(dtype=float)
    gpu_pred = frame.gpu_prediction.fillna(0).to_numpy(dtype=float)
    gpu_base = v3_pred.copy()
    gpu_base[gpu_available] += .25 * (gpu_pred[gpu_available] - gpu_base[gpu_available])
    final = gpu_base.copy()
    p = frame.p_schedule_exact.fillna(0).to_numpy(dtype=float)
    schedule = frame.schedule_proxy_sec.fillna(0).to_numpy(dtype=float)
    final[source_available] += .5 * p[source_available] * (
        schedule[source_available] - final[source_available])
    valid = frame.a_valid.to_numpy(dtype=bool)
    report = {"rule": "Fixed GPU .25 followed by source .5; no tuning on these combined OOF rows",
              "counts": {"all_finite": int(len(frame)),
                         "gpu_valid": int(gpu_available.sum()),
                         "source_candidate": int(source_available.sum())},
              "folds": {}}
    for fold in FOLDS:
        use = frame.fold.eq(fold).to_numpy(dtype=bool)
        source_use = use & source_available
        report["folds"][fold] = {
            "v3": all_metrics(y[use], v3_pred[use], valid[use]),
            "gpu_only": all_metrics(y[use], gpu_base[use], valid[use]),
            "gpu_plus_source": all_metrics(y[use], final[use], valid[use]),
            "source_subset": {"n": int(source_use.sum()),
                              "gpu_rmse_sec": rmse(y[source_use], gpu_base[source_use]),
                              "combined_rmse_sec": rmse(y[source_use], final[source_use])},
        }
    report["combined"] = {
        "v3": all_metrics(y, v3_pred, valid),
        "gpu_only": all_metrics(y, gpu_base, valid),
        "gpu_plus_source": all_metrics(y, final, valid),
    }
    (args.output_dir / "validation_sequential_gpu.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    return report


def fit_final_predict(args: argparse.Namespace) -> dict:
    evaluation = json.loads((args.output_dir / "validation_combined.json").read_text(
        encoding="utf-8"))
    if evaluation["selected_scale"] != .5:
        raise ValueError("The frozen January/July calibration is not the expected .5")
    for fold in ("seasonal_nested", "forward_transfer"):
        if evaluation[fold]["corrected"]["overall_rmse_sec"] >= evaluation[fold]["base"]["overall_rmse_sec"]:
            raise ValueError(f"Source correction does not improve {fold}")
    fold_reports = [json.loads((args.output_dir / f"{fold}_validation.json").read_text(
        encoding="utf-8")) for fold in FOLDS]
    iterations = int(np.median([part["best_iteration"] + 1 for part in fold_reports]))

    rows, features = load_cache(args.cache_dir, False)
    features = add_flight_weather(rows, features, args.data_dir, args.weather_file, False)
    feature_names = features.columns.tolist()
    candidate, schedule = candidate_mask(rows, features)
    y = rows.target.to_numpy(dtype=float)
    train = candidate & np.isfinite(y) & (y >= 0) & (y <= 86400)
    train_idx = np.flatnonzero(train)
    exact = (np.abs(y - schedule) <= 60).astype(np.int8)
    train_exact_rate = float(exact[train_idx].mean())
    cats = features.select_dtypes(include="category").columns.tolist()
    model = CatBoostClassifier(**source_params(iterations, args))
    start = time.monotonic()
    model.fit(Pool(features.iloc[train_idx], label=exact[train_idx],
                   cat_features=cats))
    fit_seconds = time.monotonic() - start
    model.save_model(str(args.output_dir / "final.cbm"))
    train_count = int(len(train_idx))
    del features, rows, schedule, y, candidate, train_idx, exact
    gc.collect()

    rank_rows, rank_features = load_cache(args.cache_dir, True)
    rank_features = add_flight_weather(rank_rows, rank_features,
                                       args.data_dir, args.weather_file, True)
    if rank_features.columns.tolist() != feature_names:
        raise ValueError("Ranking features do not match training features")
    rank_candidate, rank_schedule = candidate_mask(rank_rows, rank_features)
    rank_idx = np.flatnonzero(rank_candidate)
    prob = model.predict_proba(rank_features.iloc[rank_idx],
                               thread_count=args.threads)[:, 1]
    probability = np.full(len(rank_rows), np.nan, dtype=float)
    probability[rank_idx] = prob
    specialist = pd.DataFrame({
        "MVT_ID_mvt": rank_rows.MVT_ID_mvt,
        "source_candidate": rank_candidate,
        "schedule_proxy_sec": np.where(rank_candidate, rank_schedule, np.nan),
        "p_schedule_exact": probability,
    })
    specialist.to_parquet(args.output_dir / "ranking_source_probabilities.parquet",
                          index=False)
    v3 = pd.read_parquet(args.v3_dir / "predictions.parquet")
    ranking = v3.merge(specialist, on="MVT_ID_mvt", how="left",
                       validate="one_to_one")
    if len(ranking) != len(v3) or ranking.source_candidate.isna().any():
        raise ValueError("Ranking source probability IDs are incomplete")
    old = ranking.TAXITIME_SEC_mvt.to_numpy(dtype=float)
    updated = old.copy()
    selected = ranking.source_candidate.to_numpy(dtype=bool)
    updated[selected] += .5 * ranking.loc[selected, "p_schedule_exact"].to_numpy(dtype=float) * (
        ranking.loc[selected, "schedule_proxy_sec"].to_numpy(dtype=float) - updated[selected])
    output = pd.DataFrame({"MVT_ID_mvt": ranking.MVT_ID_mvt,
                           "TAXITIME_SEC_mvt": np.maximum(updated, 0)})
    template = pd.read_parquet(args.data_dir / "submitting.parquet",
                               columns=["MVT_ID_mvt"])
    output = template.merge(output, on="MVT_ID_mvt", how="left",
                            validate="one_to_one", sort=False)
    if len(output) != len(template) or not np.isfinite(output.TAXITIME_SEC_mvt).all():
        raise ValueError("Final ranking source candidate is incomplete or non-finite")
    output.to_parquet(args.output_dir / "predictions.parquet", index=False)
    manifest = {
        "purpose": "Local candidate for review; no upload performed",
        "source": "CatBoost schedule-source classifier on LIRF/LTFM valid AOBT disagreement",
        "base_v3_file": str((args.v3_dir / "predictions.parquet").resolve()),
        "output_file": str((args.output_dir / "predictions.parquet").resolve()),
        "model_file": str((args.output_dir / "final.cbm").resolve()),
        "iterations": iterations,
        "iteration_selection": {fold: report["best_iteration"] for fold, report in zip(FOLDS, fold_reports)},
        "training_rows": train_count,
        "training_schedule_exact_rate": train_exact_rate,
        "fit_seconds": fit_seconds,
        "ranking_rows": int(len(output)),
        "ranking_source_candidates": int(selected.sum()),
        "ranking_predictions_changed": int((updated != old).sum()),
        "scale": .5,
        "feature_names": feature_names,
        "categorical_features": cats,
        "validation_report": str((args.output_dir / "validation_combined.json").resolve()),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def combine_gpu_rank(args: argparse.Namespace) -> dict:
    """Apply the frozen source rule to the finalized .25 GPU/v3 ranking blend."""
    gpu_path = args.gpu_dir / "predictions.parquet"
    source_path = args.output_dir / "ranking_source_probabilities.parquet"
    if not gpu_path.exists() or not source_path.exists():
        raise FileNotFoundError("Final GPU blend and source probability files are required")
    gpu = pd.read_parquet(gpu_path, columns=["MVT_ID_mvt", "TAXITIME_SEC_mvt"])
    source = pd.read_parquet(source_path,
                              columns=["MVT_ID_mvt", "source_candidate",
                                       "schedule_proxy_sec", "p_schedule_exact"])
    if gpu.MVT_ID_mvt.duplicated().any() or source.MVT_ID_mvt.duplicated().any():
        raise ValueError("GPU or source ranking IDs are duplicated")
    combined = gpu.merge(source, on="MVT_ID_mvt", how="left", validate="one_to_one")
    if len(combined) != len(gpu) or combined.source_candidate.isna().any():
        raise ValueError("Source probabilities do not cover every GPU ranking ID")
    rank_rows = pd.read_parquet(args.cache_dir / "ranking_rows.parquet",
                                columns=["MVT_ID_mvt", "proxy"])
    combined = combined.merge(rank_rows, on="MVT_ID_mvt", how="left",
                              validate="one_to_one")
    gate = combined.source_candidate.to_numpy(dtype=bool)
    proxy = combined.proxy.to_numpy(dtype=float)
    proxy_valid = np.isfinite(proxy) & (proxy >= 0) & (proxy <= 7200)
    if not proxy_valid[gate].all():
        raise ValueError("Source gate includes a row with invalid AOBT proxy")
    base = combined.TAXITIME_SEC_mvt.to_numpy(dtype=float)
    probability = combined.p_schedule_exact.to_numpy(dtype=float)
    schedule = combined.schedule_proxy_sec.to_numpy(dtype=float)
    if not np.isfinite(base).all() or not np.isfinite(probability[gate]).all() or not np.isfinite(schedule[gate]).all():
        raise ValueError("Final combination has a non-finite input")
    prediction = base.copy()
    prediction[gate] += .5 * probability[gate] * (schedule[gate] - prediction[gate])
    prediction = np.maximum(prediction, 0)
    if not np.array_equal(prediction[~gate], base[~gate]):
        raise ValueError("A non-source row changed in the final combination")
    if not np.array_equal(prediction[~proxy_valid], base[~proxy_valid]):
        raise ValueError("An invalid-AOBT row changed in the final combination")
    output = pd.DataFrame({"MVT_ID_mvt": combined.MVT_ID_mvt,
                           "TAXITIME_SEC_mvt": prediction})
    template = pd.read_parquet(args.data_dir / "submitting.parquet",
                               columns=["MVT_ID_mvt"])
    if not gpu.MVT_ID_mvt.equals(template.MVT_ID_mvt):
        raise ValueError("GPU blend is not in submission-template order")
    if not output.MVT_ID_mvt.equals(template.MVT_ID_mvt) or not np.isfinite(prediction).all():
        raise ValueError("Sequential output does not match the submission template")
    output_path = args.output_dir / "sequential_predictions.parquet"
    output.to_parquet(output_path, index=False)
    validation = json.loads((args.output_dir / "validation_sequential_gpu.json").read_text(
        encoding="utf-8"))
    manifest = {
        "purpose": "Local sequential GPU + source candidate for review; no upload performed",
        "command": "python catboost_source.py --combine-gpu",
        "gpu_base_file": str(gpu_path.resolve()),
        "source_probability_file": str(source_path.resolve()),
        "output_file": str(output_path.resolve()),
        "gpu_weight_in_base": .25,
        "source_probability_scale": .5,
        "rule": "base=v3+.25*(gpu-v3) on valid AOBT; final=base+.5*p_source*(schedule-base) on source gate",
        "rows": int(len(output)),
        "source_gate_rows": int(gate.sum()),
        "changed_rows": int(np.count_nonzero(prediction != base)),
        "invalid_aobt_rows_unchanged": int((~proxy_valid).sum()),
        "non_source_rows_unchanged": int((~gate).sum()),
        "template_order_verified": True,
        "finite_nonnegative_verified": True,
        "validation": {
            fold: {"gpu_only_rmse_sec": validation["folds"][fold]["gpu_only"]["overall_rmse_sec"],
                   "gpu_plus_source_rmse_sec": validation["folds"][fold]["gpu_plus_source"]["overall_rmse_sec"]}
            for fold in FOLDS},
        "validation_note": "Both 2025 folds were used across local experiments; these are model-comparison estimates, not an untouched final forecast.",
    }
    (args.output_dir / "sequential_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def candidate_mask(rows: pd.DataFrame, features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    schedule = features["takeoff_minus_SCHED_TIME_UTC_mvt"].to_numpy(dtype=float)
    proxy = rows.proxy.to_numpy(dtype=float)
    airports = rows.airport.isin(("LIRF", "LTFM")).to_numpy()
    candidate = (airports & np.isfinite(proxy) & (proxy >= 0) & (proxy <= 7200)
                 & np.isfinite(schedule) & (schedule >= 0) & (schedule <= 86400)
                 & (np.abs(schedule - proxy) > 600))
    return candidate, schedule


def fit_fold(name: str, months: tuple[int, int], rows: pd.DataFrame,
             features: pd.DataFrame, comparison: pd.DataFrame,
             args: argparse.Namespace) -> dict:
    candidate, schedule = candidate_mask(rows, features)
    target = rows.target.to_numpy(dtype=float)
    heldout = rows.month.isin(months).to_numpy()
    train_mask = (candidate & ~heldout & np.isfinite(target)
                  & (target >= 0) & (target <= 86400))
    test_mask = candidate & heldout & np.isfinite(target)
    train_idx = np.flatnonzero(train_mask)
    test_idx = np.flatnonzero(test_mask)
    exact = np.abs(target - schedule) <= 60
    rng = np.random.default_rng(args.seed)
    shuffled = rng.permutation(train_idx)
    n_early = max(3000, int(.08 * len(shuffled)))
    early_idx = shuffled[:n_early]
    fit_idx = shuffled[n_early:]
    cats = features.select_dtypes(include="category").columns.tolist()
    model = CatBoostClassifier(**source_params(args.iterations, args))
    start = time.monotonic()
    model.fit(Pool(features.iloc[fit_idx], label=exact[fit_idx].astype(np.int8),
                   cat_features=cats),
              eval_set=Pool(features.iloc[early_idx],
                            label=exact[early_idx].astype(np.int8),
                            cat_features=cats),
              use_best_model=True, early_stopping_rounds=100)
    fit_seconds = time.monotonic() - start
    prob = model.predict_proba(features.iloc[test_idx],
                               thread_count=args.threads)[:, 1]
    output = rows.iloc[test_idx][["MVT_ID_mvt", "target", "proxy", "month", "airport"]].copy()
    output["schedule_proxy_sec"] = schedule[test_idx]
    output["schedule_exact"] = exact[test_idx]
    output["p_schedule_exact"] = prob
    output = output.merge(comparison[["MVT_ID_mvt", "selected"]].rename(
        columns={"selected": "v3_prediction"}), on="MVT_ID_mvt",
        how="left", validate="one_to_one")
    if output.v3_prediction.isna().any() or len(output) != len(test_idx):
        raise ValueError("Frozen v3 OOF does not cover source-classifier test rows")
    y = output.target.to_numpy(dtype=float)
    v3 = output.v3_prediction.to_numpy(dtype=float)
    alternative = output.schedule_proxy_sec.to_numpy(dtype=float)
    p = output.p_schedule_exact.to_numpy(dtype=float)
    scores = {str(scale): rmse(y, v3 + scale * p * (alternative - v3))
              for scale in SCALES}
    by_airport = {}
    for airport, group in output.groupby("airport"):
        q_y = group.target.to_numpy(dtype=float)
        q_v3 = group.v3_prediction.to_numpy(dtype=float)
        q_p = group.p_schedule_exact.to_numpy(dtype=float)
        q_schedule = group.schedule_proxy_sec.to_numpy(dtype=float)
        by_airport[str(airport)] = {
            "n": int(len(group)),
            "schedule_exact_rate": float(group.schedule_exact.mean()),
            "v3_rmse_sec": rmse(q_y, q_v3),
            "half_probability_blend_rmse_sec": rmse(
                q_y, q_v3 + .5 * q_p * (q_schedule - q_v3)),
        }
    report = {
        "fold": name, "months": months,
        "training_n": int(train_mask.sum()),
        "test_n": int(test_mask.sum()),
        "training_exact_rate": float(exact[train_mask].mean()),
        "test_exact_rate": float(exact[test_mask].mean()),
        "best_iteration": int(model.get_best_iteration()),
        "fit_seconds": fit_seconds,
        "roc_auc": float(roc_auc_score(exact[test_mask], prob)),
        "brier": float(brier_score_loss(exact[test_mask], prob)),
        "scores": scores,
        "by_airport": by_airport,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(str(args.output_dir / f"{name}.cbm"))
    output.to_parquet(args.output_dir / f"{name}_oof.parquet", index=False)
    (args.output_dir / f"{name}_validation.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/baseline"))
    parser.add_argument("--v3-dir", type=Path, default=Path("artifacts/lobt_ensemble"))
    parser.add_argument("--gpu-dir", type=Path, default=Path("artifacts/catboost/gpu"))
    parser.add_argument("--weather-file", type=Path,
                        default=Path("data/external/weather.parquet"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("artifacts/catboost/source"))
    parser.add_argument("--fold", choices=(*FOLDS, "both"),
                        default="seasonal_jan_jul")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--evaluate-sequential-gpu", action="store_true")
    parser.add_argument("--fit-final-rank", action="store_true")
    parser.add_argument("--combine-gpu", action="store_true")
    parser.add_argument("--iterations", type=int, default=600)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--depth", type=int, default=7)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.threads > 6:
        raise ValueError("Limit CatBoost to at most six host threads")
    if args.evaluate_only:
        report = evaluate_existing(args)
        print(json.dumps({"selected_scale": report["selected_scale"],
                          "seasonal_nested": report["seasonal_nested"],
                          "forward_transfer": report["forward_transfer"],
                          "combined_nested_plus_forward": report["combined_nested_plus_forward"]},
                         indent=2))
        return
    if args.evaluate_sequential_gpu:
        report = evaluate_sequential_gpu(args)
        print(json.dumps(report["folds"], indent=2))
        return
    if args.fit_final_rank:
        manifest = fit_final_predict(args)
        print(json.dumps({key: manifest[key] for key in
                          ("iterations", "training_rows", "fit_seconds",
                           "ranking_rows", "ranking_source_candidates",
                           "ranking_predictions_changed", "output_file")}, indent=2))
        return
    if args.combine_gpu:
        manifest = combine_gpu_rank(args)
        print(json.dumps({key: manifest[key] for key in
                          ("rows", "source_gate_rows", "changed_rows",
                           "invalid_aobt_rows_unchanged", "output_file")}, indent=2))
        return
    rows, features = load_cache(args.cache_dir, False)
    features = add_flight_weather(rows, features, args.data_dir,
                                  args.weather_file, False)
    comparison = pd.read_parquet(args.v3_dir / "validation_predictions.parquet",
                                  columns=["MVT_ID_mvt", "selected"])
    selected = FOLDS if args.fold == "both" else {args.fold: FOLDS[args.fold]}
    reports = {}
    for fold, months in selected.items():
        print(f"Training source classifier: {fold}", flush=True)
        reports[fold] = fit_fold(fold, months, rows, features, comparison, args)
        print(json.dumps({"fold": fold, "scores": reports[fold]["scores"],
                          "roc_auc": reports[fold]["roc_auc"],
                          "fit_seconds": reports[fold]["fit_seconds"]}, indent=2),
              flush=True)
    (args.output_dir / "validation.json").write_text(
        json.dumps(reports, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
