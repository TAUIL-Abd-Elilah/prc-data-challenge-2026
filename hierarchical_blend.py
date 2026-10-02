"""Evaluate airport-specific convex expert weights on labeled 2025 OOF rows.

Global weights provide a shrinkage prior. January/July selects the shrinkage
with nested calibration, and November/December checks it unchanged. Frozen v3
missing-AOBT and LOBT overrides are preserved. No ranking scores are read.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ensemble import _fit_simplex, _hash_folds

FOLDS = ("seasonal_jan_jul", "forward_nov_dec")
EXPERTS = ("hybrid", "direct", "grouped", "airport_prediction", "weather_prediction")
STRENGTHS = (0, 1000, 5000)


def rmse(y, prediction):
    return float(np.sqrt(np.mean((y - prediction) ** 2)))


def fit(y, x, airport, use, strength):
    prior = _fit_simplex(y, x, use)
    result = {"global": prior, "airports": {}}
    for name in np.unique(airport[use]):
        mask = use & (airport == name)
        delta = x[mask, 1:] - x[mask, :1]
        residual = y[mask] - x[mask, 0]
        scale = max(float(np.mean(np.sum(delta * delta, axis=1))), 1.)
        gram = delta.T @ delta / (len(delta) * scale)
        cross = delta.T @ residual / (len(delta) * scale)
        penalty = strength / len(delta)
        # The reference weight is 1-sum(z), so include it in the shrinkage.
        def objective(z):
            w = np.r_[1. - z.sum(), z]
            return float(.5 * z @ gram @ z - cross @ z + .5 * penalty * np.sum((w - prior) ** 2))
        def gradient(z):
            w = np.r_[1. - z.sum(), z]
            return gram @ z - cross + penalty * ((w - prior)[1:] - (w - prior)[0])
        fitted = minimize(objective, prior[1:], jac=gradient, method="SLSQP",
                          bounds=[(0., 1.)] * (x.shape[1] - 1),
                          constraints=[{"type": "ineq", "fun": lambda z: 1. - z.sum(),
                                        "jac": lambda z: -np.ones(len(z))}],
                          options={"ftol": 1e-10, "maxiter": 200})
        z = fitted.x if fitted.success and np.isfinite(fitted.x).all() else prior[1:]
        w = np.maximum(np.r_[1. - z.sum(), z], 0.)
        result["airports"][str(name)] = w / w.sum()
    return result


def predict(x, airport, use, base, model):
    result = base.copy()
    result[use] = x[use] @ model["global"]
    for name, weights in model["airports"].items():
        mask = use & (airport == name)
        result[mask] = x[mask] @ weights
    return result


def load():
    base = pd.read_parquet("artifacts/lobt_ensemble/validation_predictions.parquet")
    parts = []
    for fold in FOLDS:
        frame = pd.read_parquet(f"artifacts/baseline/{fold}_oof.parquet",
                                columns=["MVT_ID_mvt", "airport", "direct", "hybrid"])
        for directory, column in (("airports", "airport_prediction"), ("weather", "weather_prediction")):
            extra = pd.read_parquet(f"artifacts/{directory}/{fold}_oof.parquet",
                                    columns=["MVT_ID_mvt", column])
            frame = frame.merge(extra, on="MVT_ID_mvt", how="left", validate="one_to_one")
        parts.append(frame)
    experts = pd.concat(parts, ignore_index=True)
    grouped = pd.read_parquet("artifacts/ensemble/validation_predictions.parquet",
                              columns=["MVT_ID_mvt", "grouped"])
    result = base.merge(experts, on="MVT_ID_mvt", how="left", validate="one_to_one")
    result = result.merge(grouped, on="MVT_ID_mvt", how="left", validate="one_to_one")
    if len(result) != len(base) or result.airport.isna().any():
        raise ValueError("Incomplete expert coverage")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/hierarchical"))
    args = parser.parse_args()
    frame = load()
    y = frame.target.to_numpy(dtype=float)
    x = frame[list(EXPERTS)].to_numpy(dtype=float)
    base = frame.selected.to_numpy(dtype=float)
    airport = frame.airport.astype(str).to_numpy()
    use = frame.a_valid.to_numpy(dtype=bool) & ~frame.valid_disagreement.to_numpy(dtype=bool)
    if not np.isfinite(x[use]).all():
        raise ValueError("Non-finite eligible experts")
    seasonal = frame.fold.eq(FOLDS[0]).to_numpy(dtype=bool)
    forward = frame.fold.eq(FOLDS[1]).to_numpy(dtype=bool)
    outer, inner = _hash_folds(frame.MVT_ID_mvt, 5), _hash_folds(frame.MVT_ID_mvt, 3, 8)
    def choose(eligible):
        scores = {}
        for strength in STRENGTHS:
            prediction = base.copy()
            for k in range(3):
                model = fit(y, x, airport, eligible & use & (inner != k), strength)
                test = eligible & use & (inner == k)
                prediction[test] = predict(x, airport, test, base, model)[test]
            scores[str(strength)] = rmse(y[eligible & use], prediction[eligible & use])
        return min(STRENGTHS, key=lambda strength: scores[str(strength)]), scores
    nested = base.copy()
    selections = []
    for k in range(5):
        train = seasonal & (outer != k)
        strength, scores = choose(train)
        model = fit(y, x, airport, train & use, strength)
        test = seasonal & (outer == k) & use
        nested[test] = predict(x, airport, test, base, model)[test]
        selections.append({"fold": k, "strength": strength, "inner_scores": scores})
    strength, scores = choose(seasonal)
    transfer_model = fit(y, x, airport, seasonal & use, strength)
    nested[forward & use] = predict(x, airport, forward & use, base, transfer_model)[forward & use]
    final_model = fit(y, x, airport, use, strength)
    report = {"selection": "Nested January/July; fixed November/December transfer",
              "selected_strength": strength, "seasonal_inner_scores": scores,
              "outer_selections": selections, "experts": list(EXPERTS), "n": len(frame),
              "eligible_n": int(use.sum()), "metrics": {}, "by_airport": {}}
    for name, mask in (("seasonal_nested", seasonal), ("forward_transfer", forward),
                       ("combined", np.ones(len(y), dtype=bool))):
        report["metrics"][name] = {"baseline_rmse_sec": rmse(y[mask], base[mask]),
                                  "hierarchical_rmse_sec": rmse(y[mask], nested[mask]),
                                  "baseline_eligible_rmse_sec": rmse(y[mask & use], base[mask & use]),
                                  "hierarchical_eligible_rmse_sec": rmse(y[mask & use], nested[mask & use])}
    for name in np.unique(airport):
        mask = airport == name
        report["by_airport"][str(name)] = {"baseline_rmse_sec": rmse(y[mask], base[mask]),
                                          "hierarchical_rmse_sec": rmse(y[mask], nested[mask])}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    exported = {"experts": list(EXPERTS), "strength": strength,
                "global": final_model["global"].tolist(),
                "airports": {name: w.tolist() for name, w in final_model["airports"].items()}}
    (args.output_dir / "model.json").write_text(json.dumps(exported, indent=2), encoding="utf-8")
    frame[["MVT_ID_mvt", "target", "fold", "airport"]].assign(
        baseline=base, prediction=nested, eligible=use).to_parquet(
            args.output_dir / "validation_predictions.parquet", index=False)
    print(json.dumps(report["metrics"], indent=2), flush=True)


if __name__ == "__main__":
    main()
