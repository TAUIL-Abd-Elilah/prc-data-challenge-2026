"""Calibrate taxi-out experts using 2025 out-of-fold predictions only.

January and July rows provide nested calibration. The selected calibration is
then applied unchanged to November and December for temporal transfer checks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize


GAP_EDGES = (300, 600, 900, 1200, 1500, 1800, 2400, 3600)
DISAGREEMENT_EDGES = (60, 120, 300, 600, 1200, 3600)
SEGMENTS = {
    "global": (),
    "airport": ("airport",),
    "airport_gap": ("airport", "gap_bin"),
    "airport_disagreement": ("airport", "disagreement_bin"),
    "airport_gap_disagreement": ("airport", "gap_bin", "disagreement_bin"),
}
STRENGTHS = (50, 250, 1000)
MISSING_EXPERTS = ("baseline_direct", "missing_direct", "missing_schedule", "grouped",
                   "tail_gate_12000", "tail_gate_20000", "tail_gate_30000",
                   "tail_schedule_12000", "tail_schedule_20000", "tail_schedule_30000",
                   "lirf_ratio_missing", "lirf_ratio_baseline", "lirf_ratio_analog",
                   "lirf_ratio_high_schedule")
MISSING_CONFIGS = (("baseline", 0), ("global", 0),
                   ("airport", 0), ("airport", 20), ("airport", 50),
                   ("airport", 200), ("airport", 1000))


def _rmse(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - pred) ** 2)))


def _metrics(y: np.ndarray, pred: np.ndarray, valid: np.ndarray) -> dict:
    return {"overall_rmse_sec": _rmse(y, pred),
            "valid_proxy_rmse_sec": _rmse(y[valid], pred[valid]),
            "invalid_proxy_rmse_sec": _rmse(y[~valid], pred[~valid]),
            "n": int(len(y)), "valid_proxy_n": int(valid.sum()),
            "invalid_proxy_n": int((~valid).sum())}


def _covariates(path: Path) -> pd.DataFrame:
    cols = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt",
            "AOBT_3_flt", "LOBT_flt"]
    frame = pd.read_parquet(path, columns=cols)
    frame = frame.loc[frame.PHASE_mvt.eq("DEP")].copy()
    gap = (frame.MVT_TIME_UTC_mvt - frame.AOBT_3_flt).dt.total_seconds().to_numpy(dtype=float)
    disagreement = (frame.AOBT_3_flt - frame.LOBT_flt).dt.total_seconds().to_numpy(dtype=float)
    frame["airport"] = frame.ADEP_mvt.fillna("__MISSING__")
    frame["a_valid"] = np.isfinite(gap) & (gap >= 0) & (gap <= 7200)
    frame["gap_bin"] = np.searchsorted(GAP_EDGES, np.nan_to_num(gap, nan=-1), side="left").astype(np.int8)
    frame["disagreement_bin"] = np.where(
        np.isfinite(disagreement),
        np.searchsorted(DISAGREEMENT_EDGES, np.nan_to_num(np.abs(disagreement), nan=-1), side="left"),
        -1,
    ).astype(np.int8)
    return frame[["MVT_ID_mvt", "airport", "a_valid", "gap_bin", "disagreement_bin"]]


def _codes(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    result = {}
    for name, keys in SEGMENTS.items():
        if not keys:
            result[name] = np.zeros(len(frame), dtype=np.int32)
        elif len(keys) == 1:
            result[name] = pd.factorize(frame[keys[0]], sort=True)[0].astype(np.int32)
        else:
            result[name] = pd.factorize(pd.MultiIndex.from_frame(frame[list(keys)]), sort=True)[0].astype(np.int32)
    return result


def _new_codes(train: pd.DataFrame, new: pd.DataFrame, keys: tuple[str, ...]) -> np.ndarray:
    """Reuse training group IDs; return -1 for a previously unseen group."""
    if not keys:
        return np.zeros(len(new), dtype=np.int32)
    if len(keys) == 1:
        categories = pd.Index(pd.unique(train[keys[0]])).sort_values()
        return categories.get_indexer(new[keys[0]]).astype(np.int32)
    categories = pd.MultiIndex.from_frame(train[list(keys)]).unique().sort_values()
    lookup = pd.MultiIndex.from_frame(new[list(keys)])
    return categories.get_indexer(lookup).astype(np.int32)


def _fit_weights(y: np.ndarray, baseline: np.ndarray, grouped: np.ndarray,
                 valid: np.ndarray, train: np.ndarray, codes: np.ndarray,
                 strength: int) -> dict:
    use = train & valid
    d = grouped[use] - baseline[use]
    r = y[use] - baseline[use]
    if not len(d):
        raise ValueError("No valid-proxy rows in blend training fold")
    numerator = np.bincount(codes[use], weights=d * r, minlength=int(codes.max()) + 1)
    denominator = np.bincount(codes[use], weights=d * d, minlength=int(codes.max()) + 1)
    global_weight = float(np.clip(numerator.sum() / max(denominator.sum(), 1e-12), 0, 1))
    shrink = strength * max(float(np.mean(d * d)), 1e-9)
    weights = np.clip((numerator + shrink * global_weight) / (denominator + shrink), 0, 1)
    return {"global_weight": global_weight, "weights": weights}


def _predict(baseline: np.ndarray, grouped: np.ndarray, valid: np.ndarray,
             codes: np.ndarray, model: dict) -> np.ndarray:
    pred = baseline.copy()
    pred[valid] += model["weights"][codes[valid]] * (grouped[valid] - baseline[valid])
    return pred


def _hash_folds(ids: pd.Series, count: int, shift: int = 0) -> np.ndarray:
    hashes = pd.util.hash_pandas_object(ids, index=False).to_numpy(dtype=np.uint64)
    return ((hashes >> shift) % count).astype(np.int8)


def _candidate_names() -> list[tuple[str, int]]:
    return [("convex_mix", 0), ("global", 250), ("airport", 250),
            ("airport_gap", 1000), ("airport_disagreement", 1000)]


def _fit_simplex(y: np.ndarray, experts: np.ndarray, use: np.ndarray) -> np.ndarray:
    """Least-squares convex weights with hybrid as the reference expert."""
    x = experts[use]
    target = y[use]
    if len(x) == 0:
        raise ValueError("No valid rows for convex mixture")
    delta = x[:, 1:] - x[:, :1]
    residual = target - x[:, 0]
    scale = max(float(np.mean(np.sum(delta * delta, axis=1))), 1.0)
    gram = (delta.T @ delta) / (len(x) * scale)
    cross = (delta.T @ residual) / (len(x) * scale)
    m = delta.shape[1]
    if m == 0:
        return np.array([1.0])
    def objective(z):
        return float(.5 * z @ gram @ z - cross @ z)
    def gradient(z):
        return gram @ z - cross
    result = minimize(objective, np.zeros(m), jac=gradient, method="SLSQP",
                      bounds=[(0, 1)] * m,
                      constraints=[{"type": "ineq", "fun": lambda z: 1 - z.sum(),
                                    "jac": lambda z: -np.ones(m)}],
                      options={"ftol": 1e-11, "maxiter": 200})
    candidates = [np.zeros(m), *[np.eye(m)[i] for i in range(m)]]
    if result.success and np.isfinite(result.x).all():
        candidates.append(np.clip(result.x, 0, 1))
    z = min(candidates, key=objective)
    return np.r_[max(0., 1 - z.sum()), z]


def _fit_valid(y: np.ndarray, experts: np.ndarray, valid: np.ndarray, train: np.ndarray,
               codes: dict[str, np.ndarray], name: str, strength: int) -> dict:
    if name == "convex_mix":
        return {"kind": "convex_mix", "weights": _fit_simplex(y, experts, train & valid)}
    model = _fit_weights(y, experts[:, 0], experts[:, 2], valid, train,
                         codes[name], strength)
    return {"kind": "grouped_gate", **model}


def _predict_valid(experts: np.ndarray, valid: np.ndarray, codes: dict[str, np.ndarray],
                   name: str, model: dict) -> np.ndarray:
    pred = experts[:, 0].copy()
    if model["kind"] == "convex_mix":
        pred[valid] = experts[valid] @ model["weights"]
    else:
        pred = _predict(pred, experts[:, 2], valid, codes[name], model)
    return pred


def _choose_candidate(y: np.ndarray, experts: np.ndarray, valid: np.ndarray,
                      eligible: np.ndarray, codes: dict[str, np.ndarray],
                      inner_folds: np.ndarray) -> tuple[str, int, dict]:
    scores = {}
    for name, strength in _candidate_names():
        squared_error = 0.0
        n = 0
        for fold in range(3):
            train = eligible & (inner_folds != fold)
            test = eligible & (inner_folds == fold) & valid
            model = _fit_valid(y, experts, valid, train, codes, name, strength)
            pred = _predict_valid(experts, valid, codes, name, model)
            squared_error += float(np.sum((y[test] - pred[test]) ** 2))
            n += int(test.sum())
        scores[f"{name}:{strength}"] = float(np.sqrt(squared_error / n))
    selected = min(scores, key=scores.get)
    name, strength = selected.rsplit(":", 1)
    return name, int(strength), scores


def _fit_missing(y: np.ndarray, experts: np.ndarray, valid: np.ndarray,
                 train: np.ndarray, airport_codes: np.ndarray,
                 name: str, strength: int) -> dict:
    use = train & ~valid
    if not use.any():
        raise ValueError("No missing-proxy rows in blend training fold")
    squared = (experts[use] - y[use, None]) ** 2
    global_risk = squared.mean(axis=0)
    global_choice = int(np.argmin(global_risk))
    if name == "baseline":
        return {"global_choice": 0, "choices": np.zeros(int(airport_codes.max()) + 1, dtype=np.int8)}
    if name == "global":
        return {"global_choice": global_choice,
                "choices": np.full(int(airport_codes.max()) + 1, global_choice, dtype=np.int8)}
    group = airport_codes[use]
    ngroup = int(airport_codes.max()) + 1
    n = np.bincount(group, minlength=ngroup)
    sums = np.column_stack([np.bincount(group, weights=squared[:, i], minlength=ngroup)
                            for i in range(experts.shape[1])])
    risk = (sums + strength * global_risk[None, :]) / (n[:, None] + strength)
    return {"global_choice": global_choice, "choices": np.argmin(risk, axis=1).astype(np.int8)}


def _predict_missing(experts: np.ndarray, valid: np.ndarray,
                     airport_codes: np.ndarray, model: dict) -> np.ndarray:
    pred = np.full(len(valid), np.nan, dtype=np.float64)
    rows = np.flatnonzero(~valid)
    if len(rows):
        codes = airport_codes[rows]
        choices = np.where(codes >= 0, model["choices"][np.maximum(codes, 0)],
                           model["global_choice"])
        pred[rows] = experts[rows, choices]
    return pred


def _choose_missing(y: np.ndarray, experts: np.ndarray, valid: np.ndarray,
                    eligible: np.ndarray, airport_codes: np.ndarray,
                    inner_folds: np.ndarray) -> tuple[str, int, dict]:
    scores = {}
    for name, strength in MISSING_CONFIGS:
        squared_error = 0.0
        n = 0
        for fold in range(3):
            train = eligible & (inner_folds != fold)
            test = eligible & (inner_folds == fold) & ~valid
            model = _fit_missing(y, experts, valid, train, airport_codes, name, strength)
            pred = _predict_missing(experts, valid, airport_codes, model)
            squared_error += float(np.sum((y[test] - pred[test]) ** 2))
            n += int(test.sum())
        scores[f"{name}:{strength}"] = float(np.sqrt(squared_error / n))
    selected = min(scores, key=scores.get)
    name, strength = selected.rsplit(":", 1)
    return name, int(strength), scores


def _find_oof(directory: Path) -> list[Path]:
    files = sorted(directory.glob("*oof.parquet"))
    if not files:
        files = sorted(directory.glob("*oof*.parquet"))
    if not files:
        raise FileNotFoundError(f"No OOF Parquet files in {directory}")
    return files


def _load_oof(directory: Path) -> pd.DataFrame:
    files = _find_oof(directory)
    frames = []
    for path in files:
        frame = pd.read_parquet(path)
        required = {"MVT_ID_mvt", "direct", "hybrid"}
        if not required <= set(frame.columns):
            raise ValueError(f"OOF file {path} lacks {sorted(required - set(frame.columns))}")
        frames.append(frame)
    oof = pd.concat(frames, ignore_index=True)
    if oof.MVT_ID_mvt.isna().any() or oof.MVT_ID_mvt.duplicated().any():
        raise ValueError("Baseline OOF movement IDs must be unique and non-null")
    return oof


def _load_missing_oof(directory: Path) -> pd.DataFrame:
    frames = []
    for path in _find_oof(directory):
        frame = pd.read_parquet(path, columns=["MVT_ID_mvt", "direct", "schedule_residual"])
        frames.append(frame.rename(columns={"direct": "missing_direct",
                                             "schedule_residual": "missing_schedule"}))
    result = pd.concat(frames, ignore_index=True)
    if result.MVT_ID_mvt.isna().any() or result.MVT_ID_mvt.duplicated().any():
        raise ValueError("Missing-expert OOF IDs must be unique and non-null")
    return result


def _load_grouped_oof(directory: Path) -> pd.DataFrame:
    files = [("seasonal_jan_jul", directory / "validation_predictions.parquet"),
             ("forward_nov_dec", directory / "forward_validation_predictions.parquet")]
    frames = []
    for fold, path in files:
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_parquet(path)
        frame["fold"] = fold
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    if result.MVT_ID_mvt.duplicated().any():
        raise ValueError("Grouped OOF movement IDs are duplicated")
    return result


def _tail_candidates(frame: pd.DataFrame, fallback: np.ndarray) -> list[np.ndarray]:
    selected = frame.selected_candidate_sec.to_numpy(dtype=np.float64)
    selected = np.where(np.isfinite(selected), selected, fallback)
    return [np.where(frame[f"gate_{threshold}"].fillna(False).to_numpy(dtype=bool),
                     selected, fallback)
            for threshold in (12000, 20000, 30000)]


def _tail_schedule_candidates(frame: pd.DataFrame, baseline: np.ndarray,
                              schedule: np.ndarray) -> list[np.ndarray]:
    selected = frame.selected_candidate_sec.to_numpy(dtype=np.float64)
    selected = np.where(np.isfinite(selected), selected, baseline)
    airport = frame.airport.eq("LIRF").to_numpy(dtype=bool)
    return [np.where(airport & frame[f"gate_{threshold}"].fillna(False).to_numpy(dtype=bool),
                     selected, np.where(airport, schedule, baseline))
            for threshold in (12000, 20000, 30000)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--baseline-dir", type=Path, default=Path("artifacts/baseline"))
    parser.add_argument("--missing-dir", type=Path, default=Path("artifacts/missing"))
    parser.add_argument("--tail-dir", type=Path, default=Path("artifacts/tail"))
    parser.add_argument("--lirf-dir", type=Path, default=Path("artifacts/lirf"))
    parser.add_argument("--weather-dir", type=Path, default=Path("artifacts/weather"))
    parser.add_argument("--airports-dir", type=Path, default=Path("artifacts/airports"))
    parser.add_argument("--grouped-dir", type=Path, default=Path("artifacts/grouped"))
    parser.add_argument("--baseline-ranking", type=Path,
                        default=Path("artifacts/baseline/ranking_predictions.parquet"),
                        help="Full-year baseline ranking experts; omit file to evaluate OOF only")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/ensemble"))
    args = parser.parse_args()
    grouped = _load_grouped_oof(args.grouped_dir)
    oof = _load_oof(args.baseline_dir)
    missing_oof = _load_missing_oof(args.missing_dir)
    tail_oof = pd.read_parquet(args.tail_dir / "oof.parquet")
    tail_cols = ["MVT_ID_mvt", "target_sec", "selected_candidate_sec",
                 "gate_12000", "gate_20000", "gate_30000",
                 "schedule_proxy_sec", "analog_count", "analog_exact_rate"]
    tail_oof = tail_oof[tail_cols]
    lirf_oof = pd.read_parquet(args.lirf_dir / "oof.parquet",
                               columns=["MVT_ID_mvt", "ratio_model", "schedule_proxy_sec"])
    lirf_oof = lirf_oof.rename(columns={"schedule_proxy_sec": "lirf_schedule_proxy_sec"})
    cov_paths = [args.data_dir / f"training_2025-{mo:02d}-01_2025-{mo + 1:02d}-01.parquet"
                 for mo in (1, 7, 11)]
    cov_paths.append(args.data_dir / "training_2025-12-01_2026-01-01.parquet")
    cov = pd.concat([_covariates(path) for path in cov_paths], ignore_index=True)
    frame = grouped.merge(oof, on="MVT_ID_mvt", how="left", validate="one_to_one",
                          suffixes=("", "_baseline"))
    frame = frame.merge(cov, on="MVT_ID_mvt", how="left", validate="one_to_one",
                        suffixes=("", "_cov"))
    frame = frame.merge(missing_oof, on="MVT_ID_mvt", how="left", validate="one_to_one")
    frame = frame.merge(tail_oof, on="MVT_ID_mvt", how="left", validate="one_to_one")
    frame = frame.merge(lirf_oof, on="MVT_ID_mvt", how="left", validate="one_to_one")
    expert_names = ["hybrid", "direct", "grouped"]
    airport_files = [args.airports_dir / f"{fold}_oof.parquet"
                     for fold in ("seasonal_jan_jul", "forward_nov_dec")]
    if all(path.exists() for path in airport_files):
        airport_oof = pd.concat([pd.read_parquet(path,
                                 columns=["MVT_ID_mvt", "airport_prediction"])
                                for path in airport_files], ignore_index=True)
        frame = frame.merge(airport_oof, on="MVT_ID_mvt", how="left", validate="one_to_one")
        if frame.loc[frame.a_valid, "airport_prediction"].notna().all():
            expert_names.append("airport_prediction")
    weather_files = [args.weather_dir / f"{fold}_oof.parquet"
                     for fold in ("seasonal_jan_jul", "forward_nov_dec")]
    if all(path.exists() for path in weather_files):
        weather_oof = pd.concat([pd.read_parquet(path,
                                 columns=["MVT_ID_mvt", "weather_prediction"])
                                for path in weather_files], ignore_index=True)
        frame = frame.merge(weather_oof, on="MVT_ID_mvt", how="left", validate="one_to_one")
        if frame.loc[frame.a_valid, "weather_prediction"].notna().all():
            expert_names.append("weather_prediction")
    if len(frame) != len(grouped) or frame.airport_cov.isna().any():
        raise ValueError("Covariate coverage does not match grouped validation rows")
    excluded = frame.loc[frame.direct.isna()].copy()
    frame = frame.loc[frame.direct.notna()].copy()
    if not frame.airport.eq(frame.airport_cov).all() or not frame.a_valid.eq(frame.a_valid_cov).all():
        raise ValueError("Grouped validation and source covariates disagree")
    if "target" in frame and not np.allclose(frame.TAXITIME_SEC_mvt, frame.target):
        raise ValueError("Baseline and grouped validation targets disagree")
    tail_rows = frame.target_sec.notna()
    if not np.allclose(frame.loc[tail_rows, "TAXITIME_SEC_mvt"], frame.loc[tail_rows, "target_sec"]):
        raise ValueError("Tail and grouped validation targets disagree")
    need_tail = frame.airport.eq("LIRF") & ~frame.a_valid
    tail_uncovered = int((need_tail & ~tail_rows).sum())
    frame = frame.sort_values("MVT_ID_mvt").reset_index(drop=True)
    y = frame.TAXITIME_SEC_mvt.to_numpy(dtype=np.float64)
    valid = frame.a_valid.to_numpy(dtype=bool)
    direct = frame.direct.to_numpy(dtype=np.float64)
    hybrid = frame.hybrid.to_numpy(dtype=np.float64)
    baseline = np.where(valid, hybrid, direct)
    grouped_pred = frame.prediction_sec.to_numpy(dtype=np.float64)
    valid_column = {"hybrid": "hybrid", "direct": "direct", "grouped": "prediction_sec",
                    "airport_prediction": "airport_prediction",
                    "weather_prediction": "weather_prediction"}
    valid_experts = np.column_stack([
        frame[valid_column[name]].to_numpy(dtype=np.float64) for name in expert_names])
    tail_fallback = direct
    schedule_fallback = frame.missing_schedule.to_numpy(dtype=np.float64)
    ratio = frame.ratio_model.to_numpy(dtype=np.float64)
    lirf_mask = frame.airport.eq("LIRF").to_numpy(dtype=bool)
    lirf_ratio_missing = np.where(lirf_mask, ratio,
                                   frame.missing_direct.to_numpy(dtype=np.float64))
    lirf_ratio_baseline = np.where(lirf_mask, ratio, direct)
    analog_gate = (lirf_mask & (frame.schedule_proxy_sec.fillna(0).to_numpy(dtype=float) > 7200)
                   & (frame.analog_count.fillna(0).to_numpy(dtype=float) >= 3)
                   & (frame.analog_exact_rate.fillna(0).to_numpy(dtype=float) >= .8)
                   & frame.selected_candidate_sec.notna().to_numpy())
    lirf_ratio_analog = np.where(analog_gate,
                                  frame.selected_candidate_sec.to_numpy(dtype=float),
                                  lirf_ratio_baseline)
    schedule_proxy = frame.schedule_proxy_sec.fillna(
        frame.lirf_schedule_proxy_sec).to_numpy(dtype=float)
    high_schedule_gate = lirf_mask & (schedule_proxy >= 80000) & (schedule_proxy <= 100000)
    lirf_ratio_high_schedule = np.where(
        high_schedule_gate, .5 * lirf_ratio_baseline + .5 * schedule_proxy,
        lirf_ratio_baseline)
    missing_experts = np.column_stack([
        direct,
        frame.missing_direct.to_numpy(dtype=np.float64),
        schedule_fallback,
        grouped_pred,
        *_tail_candidates(frame, tail_fallback),
        *_tail_schedule_candidates(frame, tail_fallback, schedule_fallback),
        lirf_ratio_missing,
        lirf_ratio_baseline,
        lirf_ratio_analog,
        lirf_ratio_high_schedule,
    ])
    if not (np.isfinite(y).all() and np.isfinite(baseline).all() and np.isfinite(grouped_pred).all()):
        raise ValueError("OOF target or predictions contain non-finite values")
    if not np.isfinite(missing_experts[~valid]).all():
        raise ValueError("Missing-expert OOF does not cover every invalid-proxy row")
    if not np.isfinite(valid_experts[valid]).all():
        raise ValueError("Valid-expert OOF does not cover every valid-proxy row")
    codes = _codes(frame)
    airport_codes = codes["airport"]
    outer = _hash_folds(frame.MVT_ID_mvt, 5)
    inner = _hash_folds(frame.MVT_ID_mvt, 3, shift=8)
    seasonal = frame.fold.eq("seasonal_jan_jul").to_numpy(dtype=bool)
    forward = frame.fold.eq("forward_nov_dec").to_numpy(dtype=bool)
    if not (seasonal | forward).all():
        raise ValueError("Unknown validation fold")
    nested = baseline.copy()
    choices = []
    for fold in range(5):
        train = seasonal & (outer != fold)
        test = seasonal & (outer == fold)
        name, strength, scores = _choose_candidate(y, valid_experts, valid,
                                                    train, codes, inner)
        missing_name, missing_strength, missing_scores = _choose_missing(
            y, missing_experts, valid, train, airport_codes, inner)
        model = _fit_valid(y, valid_experts, valid, train, codes, name, strength)
        pred = _predict_valid(valid_experts, valid, codes, name, model)
        missing_model = _fit_missing(y, missing_experts, valid, train, airport_codes,
                                     missing_name, missing_strength)
        missing_pred = _predict_missing(missing_experts, valid, airport_codes, missing_model)
        pred[~valid] = missing_pred[~valid]
        nested[test] = pred[test]
        choices.append({"outer_fold": fold, "selected": name,
                        "strength": strength,
                        "valid_weights": (model["weights"].tolist()
                                          if name == "convex_mix" else None),
                        "global_grouped_weight": model.get("global_weight"),
                        "missing_selected": missing_name,
                        "missing_strength": missing_strength,
                        "missing_global_choice": MISSING_EXPERTS[missing_model["global_choice"]],
                        "outer_rmse_sec": _rmse(y[test], pred[test]),
                        "inner_valid_proxy_scores": scores,
                        "inner_invalid_proxy_scores": missing_scores})
    seasonal_name, seasonal_strength, seasonal_scores = _choose_candidate(
        y, valid_experts, valid, seasonal, codes, inner
    )
    seasonal_model = _fit_valid(y, valid_experts, valid, seasonal, codes,
                                seasonal_name, seasonal_strength)
    seasonal_missing_name, seasonal_missing_strength, seasonal_missing_scores = _choose_missing(
        y, missing_experts, valid, seasonal, airport_codes, inner)
    seasonal_missing_model = _fit_missing(y, missing_experts, valid, seasonal,
                                          airport_codes, seasonal_missing_name,
                                          seasonal_missing_strength)
    transfer = _predict_valid(valid_experts, valid, codes, seasonal_name, seasonal_model)
    transfer_missing = _predict_missing(missing_experts, valid, airport_codes,
                                         seasonal_missing_model)
    transfer[~valid] = transfer_missing[~valid]
    nested[forward] = transfer[forward]
    all_rows = np.ones(len(frame), dtype=bool)
    final_name, final_strength, final_scores = _choose_candidate(
        y, valid_experts, valid, all_rows, codes, inner)
    final_model = _fit_valid(y, valid_experts, valid, all_rows, codes,
                             final_name, final_strength)
    final_missing_name, final_missing_strength, final_missing_scores = _choose_missing(
        y, missing_experts, valid, all_rows, airport_codes, inner)
    final_missing_model = _fit_missing(y, missing_experts, valid, all_rows,
                                       airport_codes, final_missing_name,
                                       final_missing_strength)
    summary = {
        "valid_experts": expert_names,
        "baseline": _metrics(y, baseline, valid),
        "grouped": _metrics(y, grouped_pred, valid),
        "nested_seasonal_plus_forward_transfer": _metrics(y, nested, valid),
        "seasonal_nested_baseline": _metrics(y[seasonal], baseline[seasonal], valid[seasonal]),
        "seasonal_nested_ensemble": _metrics(y[seasonal], nested[seasonal], valid[seasonal]),
        "forward_transfer_baseline": _metrics(y[forward], baseline[forward], valid[forward]),
        "forward_transfer_ensemble": _metrics(y[forward], transfer[forward], valid[forward]),
        "excluded_from_baseline_oof": {
            "n": int(len(excluded)),
            "grouped_only_rmse_sec": (_rmse(excluded.TAXITIME_SEC_mvt.to_numpy(dtype=float),
                                             excluded.prediction_sec.to_numpy(dtype=float))
                                      if len(excluded) else None),
        },
        "LIRF_invalid_proxy_without_tail_expert": tail_uncovered,
        "seasonal_selection": {"method": seasonal_name, "strength": seasonal_strength,
                               "valid_weights": (seasonal_model["weights"].tolist()
                                                 if seasonal_name == "convex_mix" else None),
                               "missing_method": seasonal_missing_name,
                               "missing_strength": seasonal_missing_strength,
                               "missing_global_choice": MISSING_EXPERTS[seasonal_missing_model["global_choice"]],
                               "inner_valid_proxy_scores": seasonal_scores,
                               "inner_invalid_proxy_scores": seasonal_missing_scores},
        "final_selection": {"segment": final_name, "strength": final_strength,
                            "valid_weights": (final_model["weights"].tolist()
                                              if final_name == "convex_mix" else None),
                            "global_grouped_weight": final_model.get("global_weight"),
                            "inner_valid_proxy_scores": final_scores,
                            "missing_method": final_missing_name,
                            "missing_strength": final_missing_strength,
                            "missing_global_choice": MISSING_EXPERTS[final_missing_model["global_choice"]],
                            "missing_airport_choices": {
                                str(a): MISSING_EXPERTS[final_missing_model["choices"][i]]
                                for i, a in enumerate(pd.Index(pd.unique(frame.airport)).sort_values())},
                            "inner_invalid_proxy_scores": final_missing_scores},
        "missing_experts": {
            name: _rmse(y[~valid], missing_experts[~valid, i])
            for i, name in enumerate(MISSING_EXPERTS)},
        "by_airport": {},
        "outer_folds": choices,
        "n_by_month": frame.groupby("month").size().to_dict(),
    }
    for airport in sorted(frame.airport.unique()):
        mask = frame.airport.eq(airport).to_numpy()
        missing_mask = mask & ~valid
        summary["by_airport"][str(airport)] = {
            "baseline": _metrics(y[mask], baseline[mask], valid[mask]),
            "nested_seasonal_plus_forward_transfer": _metrics(y[mask], nested[mask], valid[mask]),
            "missing_experts": {
                name: _rmse(y[missing_mask], missing_experts[missing_mask, i])
                for i, name in enumerate(MISSING_EXPERTS)},
        }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "validation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    keys = SEGMENTS.get(final_name, ())
    if keys:
        groups = frame[list(keys)].drop_duplicates().sort_values(list(keys)).to_dict(orient="records")
    else:
        groups = [{}]
    model_export = {"segment": final_name, "strength": final_strength,
                    "valid_experts": expert_names,
                    "convex_weights": (final_model["weights"].tolist()
                                       if final_name == "convex_mix" else None),
                    "global_grouped_weight": final_model.get("global_weight"),
                    "segment_groups": groups,
                    "segment_weights": (final_model["weights"].tolist()
                                        if final_name != "convex_mix" else None),
                    "missing_method": final_missing_name,
                    "missing_strength": final_missing_strength,
                    "missing_global_choice": MISSING_EXPERTS[final_missing_model["global_choice"]],
                    "missing_airport_choices": summary["final_selection"]["missing_airport_choices"]}
    (args.output_dir / "model.json").write_text(json.dumps(model_export, indent=2), encoding="utf-8")
    pd.DataFrame({"MVT_ID_mvt": frame.MVT_ID_mvt,
                  "target": y, "baseline": baseline, "grouped": grouped_pred,
                  "nested_ensemble": nested, "a_valid": valid,
                  "fold": frame.fold,
                  "missing_direct": missing_experts[:, 1],
                  "missing_schedule": missing_experts[:, 2]}).to_parquet(
                      args.output_dir / "validation_predictions.parquet", index=False)
    invalid_diagnostics = pd.DataFrame({
        "MVT_ID_mvt": frame.loc[~valid, "MVT_ID_mvt"],
        "airport": frame.loc[~valid, "airport"],
        "fold": frame.loc[~valid, "fold"],
        "target": y[~valid], "nested_ensemble": nested[~valid],
        **{name: missing_experts[~valid, i] for i, name in enumerate(MISSING_EXPERTS)},
    })
    invalid_diagnostics.to_parquet(args.output_dir / "invalid_experts.parquet", index=False)
    ranking_expert_files = [args.baseline_ranking, args.missing_dir / "predictions.parquet"]
    if "airport_prediction" in expert_names:
        ranking_expert_files.append(args.airports_dir / "ranking_predictions.parquet")
    if "weather_prediction" in expert_names:
        ranking_expert_files.append(args.weather_dir / "ranking_predictions.parquet")
    if all(path.exists() for path in ranking_expert_files):
        baseline_rank = pd.read_parquet(args.baseline_ranking)
        if not {"MVT_ID_mvt", "direct", "hybrid"} <= set(baseline_rank):
            raise ValueError("Baseline ranking file lacks direct/hybrid experts")
        missing_rank = pd.read_parquet(args.missing_dir / "predictions.parquet")
        tail_rank = pd.read_parquet(args.tail_dir / "ranking.parquet")
        grouped_rank = pd.read_parquet(args.grouped_dir / "predictions.parquet")
        rank_cov = _covariates(args.data_dir / "ranking.parquet")
        rank = grouped_rank.rename(columns={"TAXITIME_SEC_mvt": "grouped_sec"}).merge(
            baseline_rank[["MVT_ID_mvt", "direct", "hybrid"]],
            on="MVT_ID_mvt", validate="one_to_one")
        rank = rank.merge(missing_rank[["MVT_ID_mvt", "direct", "schedule_residual"]].rename(
            columns={"direct": "missing_direct", "schedule_residual": "missing_schedule"}),
            on="MVT_ID_mvt", how="left", validate="one_to_one")
        rank = rank.merge(tail_rank[["MVT_ID_mvt", "selected_candidate_sec",
                                     "gate_12000", "gate_20000", "gate_30000",
                                     "schedule_proxy_sec", "analog_count", "analog_exact_rate"]],
                          on="MVT_ID_mvt", how="left", validate="one_to_one")
        lirf_rank = pd.read_parquet(args.lirf_dir / "ranking.parquet",
                                    columns=["MVT_ID_mvt", "ratio_model", "schedule_proxy_sec"])
        lirf_rank = lirf_rank.rename(columns={
            "schedule_proxy_sec": "lirf_schedule_proxy_sec"})
        rank = rank.merge(lirf_rank, on="MVT_ID_mvt", how="left", validate="one_to_one")
        rank = rank.merge(rank_cov, on="MVT_ID_mvt", validate="one_to_one")
        if "airport_prediction" in expert_names:
            airport_rank = pd.read_parquet(args.airports_dir / "ranking_predictions.parquet",
                                           columns=["MVT_ID_mvt", "airport_prediction"])
            rank = rank.merge(airport_rank, on="MVT_ID_mvt", how="left", validate="one_to_one")
        if "weather_prediction" in expert_names:
            weather_rank = pd.read_parquet(args.weather_dir / "ranking_predictions.parquet",
                                           columns=["MVT_ID_mvt", "weather_prediction"])
            rank = rank.merge(weather_rank, on="MVT_ID_mvt", how="left", validate="one_to_one")
        if len(rank) != len(grouped_rank):
            raise ValueError("Ranking ensemble input IDs differ")
        m = rank.a_valid.to_numpy(dtype=bool)
        base = np.where(m, rank.hybrid.to_numpy(dtype=np.float64),
                        rank.direct.to_numpy(dtype=np.float64))
        grp = rank.grouped_sec.to_numpy(dtype=np.float64)
        pred = base.copy()
        if final_name == "convex_mix":
            rank_column = {"hybrid": "hybrid", "direct": "direct", "grouped": "grouped_sec",
                           "airport_prediction": "airport_prediction",
                           "weather_prediction": "weather_prediction"}
            rank_valid_experts = np.column_stack([
                rank[rank_column[name]].to_numpy(dtype=np.float64) for name in expert_names])
            if not np.isfinite(rank_valid_experts[m]).all():
                raise ValueError("Ranking valid experts do not cover valid proxy rows")
            pred[m] = rank_valid_experts[m] @ final_model["weights"]
        else:
            rank_codes = _new_codes(frame, rank, SEGMENTS[final_name])
            weights = final_model["weights"]
            # Unseen groups use the overall validated grouped weight.
            weight = np.where(rank_codes >= 0, weights[np.maximum(rank_codes, 0)],
                              final_model["global_weight"])
            pred[m] += weight[m] * (grp[m] - base[m])
        rank_lirf = rank.airport.eq("LIRF").to_numpy(dtype=bool)
        rank_ratio = np.where(rank_lirf,
                              rank.ratio_model.to_numpy(dtype=np.float64),
                              rank.direct.to_numpy(dtype=np.float64))
        rank_analog_gate = (rank_lirf
                            & (rank.schedule_proxy_sec.fillna(0).to_numpy(dtype=float) > 7200)
                            & (rank.analog_count.fillna(0).to_numpy(dtype=float) >= 3)
                            & (rank.analog_exact_rate.fillna(0).to_numpy(dtype=float) >= .8)
                            & rank.selected_candidate_sec.notna().to_numpy())
        rank_schedule_proxy = rank.schedule_proxy_sec.fillna(
            rank.lirf_schedule_proxy_sec).to_numpy(dtype=float)
        rank_high_schedule_gate = (rank_lirf & (rank_schedule_proxy >= 80000)
                                   & (rank_schedule_proxy <= 100000))
        rank_missing = np.column_stack([rank.direct.to_numpy(dtype=np.float64),
                                        rank.missing_direct.to_numpy(dtype=np.float64),
                                        rank.missing_schedule.to_numpy(dtype=np.float64), grp,
                                        *_tail_candidates(rank,
                                            rank.direct.to_numpy(dtype=np.float64)),
                                        *_tail_schedule_candidates(rank,
                                            rank.direct.to_numpy(dtype=np.float64),
                                            rank.missing_schedule.to_numpy(dtype=np.float64)),
                                        np.where(rank.airport.eq("LIRF").to_numpy(dtype=bool),
                                                 rank.ratio_model.to_numpy(dtype=np.float64),
                                                 rank.missing_direct.to_numpy(dtype=np.float64)),
                                        np.where(rank.airport.eq("LIRF").to_numpy(dtype=bool),
                                                 rank.ratio_model.to_numpy(dtype=np.float64),
                                                 rank.direct.to_numpy(dtype=np.float64)),
                                        np.where(rank_analog_gate,
                                                 rank.selected_candidate_sec.to_numpy(dtype=np.float64),
                                                 rank_ratio),
                                        np.where(rank_high_schedule_gate,
                                                 .5 * rank_ratio + .5 * rank_schedule_proxy,
                                                 rank_ratio)])
        if not np.isfinite(rank_missing[~m]).all():
            raise ValueError("Ranking missing experts do not cover invalid-proxy rows")
        rank_airport_codes = _new_codes(frame, rank, ("airport",))
        missing_pred = _predict_missing(rank_missing, m, rank_airport_codes,
                                         final_missing_model)
        pred[~m] = missing_pred[~m]
        output = pd.DataFrame({"MVT_ID_mvt": rank.MVT_ID_mvt,
                               "TAXITIME_SEC_mvt": np.maximum(pred, 0)})
        template = pd.read_parquet(args.data_dir / "submitting.parquet", columns=["MVT_ID_mvt"])
        if set(output.MVT_ID_mvt) != set(template.MVT_ID_mvt):
            raise ValueError("Ensemble ranking IDs do not match submission template")
        output = template.merge(output, on="MVT_ID_mvt", how="left", validate="one_to_one", sort=False)
        output.to_parquet(args.output_dir / "predictions.parquet", index=False)
    print(json.dumps({"baseline": summary["baseline"],
                      "seasonal_nested_ensemble": summary["seasonal_nested_ensemble"],
                      "forward_transfer_ensemble": summary["forward_transfer_ensemble"],
                      "final_selection": {"segment": final_name,
                                          "strength": final_strength,
                                          "valid_weights": summary["final_selection"]["valid_weights"],
                                          "missing_method": final_missing_name,
                                          "missing_strength": final_missing_strength},
                      "output_dir": str(args.output_dir.resolve())}, indent=2))


if __name__ == "__main__":
    main()
