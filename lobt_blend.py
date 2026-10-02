"""Blend the validated LOBT expert into the frozen ensemble v2 prediction.

Only 2025 held-out labels are used to select the conditional valid-AOBT blend.
The final ranking output preserves the order of the supplied submission template.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


FOLDS = ("seasonal_jan_jul", "forward_nov_dec")
VALID_WEIGHTS = (.25, .5, 1.0)


def rmse(y: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - prediction) ** 2)))


def metrics(y: np.ndarray, prediction: np.ndarray, valid: np.ndarray) -> dict:
    return {
        "n": int(len(y)),
        "overall_rmse_sec": rmse(y, prediction),
        "valid_proxy_n": int(valid.sum()),
        "valid_proxy_rmse_sec": rmse(y[valid], prediction[valid]),
        "invalid_proxy_n": int((~valid).sum()),
        "invalid_proxy_rmse_sec": rmse(y[~valid], prediction[~valid]),
    }


def expert_masks(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    available = np.isfinite(frame.lobt_prediction.to_numpy(dtype=float))
    valid = frame.a_valid.to_numpy(dtype=bool)
    lobt_valid = frame.aobt_valid.fillna(False).to_numpy(dtype=bool)
    if not np.array_equal(valid[available], lobt_valid[available]):
        raise ValueError("AOBT validity disagrees between frozen and LOBT experts")
    gap = frame.aobt_lobt_abs_gap.to_numpy(dtype=float)
    invalid_fallback = available & ~valid
    valid_disagreement = available & valid & np.isfinite(gap) & (gap > 3600)
    return invalid_fallback, valid_disagreement


def blend(base: np.ndarray, lobt: np.ndarray, invalid_fallback: np.ndarray,
          valid_disagreement: np.ndarray, valid_weight: float) -> np.ndarray:
    result = base.copy()
    result[invalid_fallback] = lobt[invalid_fallback]
    result[valid_disagreement] += valid_weight * (
        lobt[valid_disagreement] - result[valid_disagreement])
    return result


def hash_folds(ids: pd.Series, count: int) -> np.ndarray:
    hashes = pd.util.hash_pandas_object(ids, index=False).to_numpy(dtype=np.uint64)
    return (hashes % count).astype(np.int8)


def load_validation(ensemble_dir: Path, lobt_dir: Path) -> pd.DataFrame:
    base = pd.read_parquet(ensemble_dir / "validation_predictions.parquet",
                           columns=["MVT_ID_mvt", "target", "nested_ensemble",
                                    "a_valid", "fold"])
    if base.MVT_ID_mvt.isna().any() or base.MVT_ID_mvt.duplicated().any():
        raise ValueError("Frozen ensemble OOF IDs must be unique and non-null")
    parts = []
    for fold in FOLDS:
        part = pd.read_parquet(lobt_dir / f"{fold}_oof.parquet",
                               columns=["MVT_ID_mvt", "target", "aobt_valid",
                                        "aobt_lobt_abs_gap", "lobt_prediction"])
        part["fold_lobt"] = fold
        parts.append(part)
    lobt = pd.concat(parts, ignore_index=True)
    if lobt.MVT_ID_mvt.isna().any() or lobt.MVT_ID_mvt.duplicated().any():
        raise ValueError("LOBT OOF IDs must be unique and non-null")
    merged = base.merge(lobt, on="MVT_ID_mvt", how="left", validate="one_to_one",
                        suffixes=("", "_lobt"))
    matched = merged.fold_lobt.notna()
    if not merged.loc[matched, "fold"].eq(merged.loc[matched, "fold_lobt"]).all():
        raise ValueError("LOBT expert and frozen ensemble folds disagree")
    if not np.allclose(merged.loc[matched, "target"],
                       merged.loc[matched, "target_lobt"]):
        raise ValueError("LOBT and frozen ensemble labels disagree")
    if not np.isfinite(merged.target).all() or not np.isfinite(merged.nested_ensemble).all():
        raise ValueError("Frozen validation targets or predictions are not finite")
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ensemble-dir", type=Path, default=Path("artifacts/ensemble"))
    parser.add_argument("--lobt-dir", type=Path, default=Path("artifacts/lobt"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/lobt_ensemble"))
    args = parser.parse_args()

    frame = load_validation(args.ensemble_dir, args.lobt_dir)
    y = frame.target.to_numpy(dtype=float)
    base = frame.nested_ensemble.to_numpy(dtype=float)
    lobt = frame.lobt_prediction.to_numpy(dtype=float)
    valid = frame.a_valid.to_numpy(dtype=bool)
    seasonal = frame.fold.eq(FOLDS[0]).to_numpy(dtype=bool)
    forward = frame.fold.eq(FOLDS[1]).to_numpy(dtype=bool)
    if not (seasonal | forward).all():
        raise ValueError("Unknown fold in frozen ensemble OOF")
    invalid_fallback, valid_disagreement = expert_masks(frame)
    if not np.isfinite(lobt[invalid_fallback | valid_disagreement]).all():
        raise ValueError("LOBT prediction is non-finite on a selected row")

    candidate_predictions = {
        weight: blend(base, lobt, invalid_fallback, valid_disagreement, weight)
        for weight in VALID_WEIGHTS
    }
    outer = hash_folds(frame.MVT_ID_mvt, 5)
    seasonal_nested = base.copy()
    selections = []
    for fold in range(5):
        train = seasonal & (outer != fold)
        test = seasonal & (outer == fold)
        scores = {str(weight): rmse(y[train], pred[train])
                  for weight, pred in candidate_predictions.items()}
        selected = min(VALID_WEIGHTS, key=lambda weight: scores[str(weight)])
        seasonal_nested[test] = candidate_predictions[selected][test]
        selections.append({"outer_fold": fold, "selected_valid_weight": selected,
                           "training_scores": scores,
                           "test_rmse_sec": rmse(y[test], seasonal_nested[test])})

    seasonal_scores = {str(weight): rmse(y[seasonal], pred[seasonal])
                       for weight, pred in candidate_predictions.items()}
    selected_weight = min(VALID_WEIGHTS, key=lambda weight: seasonal_scores[str(weight)])
    selected = candidate_predictions[selected_weight]
    fixed_quarter = candidate_predictions[.25]
    evaluation = base.copy()
    evaluation[seasonal] = seasonal_nested[seasonal]
    evaluation[forward] = selected[forward]

    report = {
        "source_ensemble": str(args.ensemble_dir.resolve()),
        "source_lobt": str(args.lobt_dir.resolve()),
        "selection": "January/July 2025 only; forward fold receives unchanged rule",
        "selected_valid_disagreement_weight": selected_weight,
        "invalid_aobt_lobt_fallback_weight": 1.0,
        "valid_disagreement_threshold_sec": 3600,
        "candidate_weights": list(VALID_WEIGHTS),
        "candidate_seasonal_scores": seasonal_scores,
        "candidate_forward_scores": {
            str(weight): rmse(y[forward], pred[forward])
            for weight, pred in candidate_predictions.items()},
        "candidate_combined_scores": {
            str(weight): rmse(y, pred)
            for weight, pred in candidate_predictions.items()},
        "seasonal_nested_selections": selections,
        "counts": {
            "all_finite_rows": int(len(frame)),
            "lobt_eligible": int(np.isfinite(lobt).sum()),
            "invalid_aobt_lobt_fallback": int(invalid_fallback.sum()),
            "valid_aobt_high_disagreement": int(valid_disagreement.sum()),
            "uncovered_kept_from_frozen_ensemble": int((~np.isfinite(lobt)).sum()),
        },
        "seasonal_nested": {
            "baseline": metrics(y[seasonal], base[seasonal], valid[seasonal]),
            "selected": metrics(y[seasonal], seasonal_nested[seasonal], valid[seasonal]),
            "fixed_quarter": metrics(y[seasonal], fixed_quarter[seasonal], valid[seasonal]),
        },
        "forward_transfer": {
            "baseline": metrics(y[forward], base[forward], valid[forward]),
            "selected": metrics(y[forward], selected[forward], valid[forward]),
            "fixed_quarter": metrics(y[forward], fixed_quarter[forward], valid[forward]),
        },
        "combined_nested_and_forward": {
            "baseline": metrics(y, base, valid),
            "selected": metrics(y, evaluation, valid),
            "fixed_quarter": metrics(y, fixed_quarter, valid),
        },
        "selected_segments": {},
    }
    for label, mask in (("invalid_aobt_lobt_valid", invalid_fallback),
                        ("valid_aobt_gap_over_3600", valid_disagreement)):
        report["selected_segments"][label] = {}
        for fold, fold_mask in ((FOLDS[0], seasonal), (FOLDS[1], forward)):
            use = mask & fold_mask
            report["selected_segments"][label][fold] = {
                "n": int(use.sum()),
                "base_rmse_sec": rmse(y[use], base[use]) if use.any() else None,
                "selected_rmse_sec": rmse(y[use], selected[use]) if use.any() else None,
                "fixed_quarter_rmse_sec": rmse(y[use], fixed_quarter[use]) if use.any() else None,
            }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "validation.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    (args.output_dir / "model.json").write_text(json.dumps({
        "selected_valid_disagreement_weight": selected_weight,
        "invalid_aobt_lobt_fallback_weight": 1.0,
        "valid_disagreement_threshold_sec": 3600,
        "selection_fold": FOLDS[0],
        "forward_check_fold": FOLDS[1],
    }, indent=2), encoding="utf-8")
    pd.DataFrame({"MVT_ID_mvt": frame.MVT_ID_mvt, "target": y, "fold": frame.fold,
                  "a_valid": valid, "base": base, "lobt_prediction": lobt,
                  "selected": evaluation, "fixed_quarter": fixed_quarter,
                  "invalid_fallback": invalid_fallback,
                  "valid_disagreement": valid_disagreement}).to_parquet(
                      args.output_dir / "validation_predictions.parquet", index=False)

    ranking_path = args.lobt_dir / "ranking_predictions.parquet"
    if ranking_path.exists():
        rank_base = pd.read_parquet(args.ensemble_dir / "predictions.parquet")
        rank_lobt = pd.read_parquet(ranking_path,
                                    columns=["MVT_ID_mvt", "lobt_prediction",
                                             "aobt_valid", "aobt_lobt_abs_gap"])
        rank = rank_base.merge(rank_lobt, on="MVT_ID_mvt", how="left",
                               validate="one_to_one")
        if len(rank) != len(rank_base):
            raise ValueError("Ranking merge changed row count")
        available = np.isfinite(rank.lobt_prediction.to_numpy(dtype=float))
        rank_valid = rank.aobt_valid.fillna(False).to_numpy(dtype=bool)
        gap = rank.aobt_lobt_abs_gap.to_numpy(dtype=float)
        rank_invalid_fallback = available & ~rank_valid
        rank_valid_disagreement = available & rank_valid & np.isfinite(gap) & (gap > 3600)
        rank_prediction = blend(rank.TAXITIME_SEC_mvt.to_numpy(dtype=float),
                                rank.lobt_prediction.to_numpy(dtype=float),
                                rank_invalid_fallback, rank_valid_disagreement,
                                selected_weight)
        output = pd.DataFrame({"MVT_ID_mvt": rank.MVT_ID_mvt,
                               "TAXITIME_SEC_mvt": np.maximum(rank_prediction, 0)})
        template = pd.read_parquet(args.data_dir / "submitting.parquet",
                                   columns=["MVT_ID_mvt"])
        output = template.merge(output, on="MVT_ID_mvt", how="left",
                                validate="one_to_one", sort=False)
        if len(output) != len(template) or not np.isfinite(output.TAXITIME_SEC_mvt).all():
            raise ValueError("Ranking output is incomplete or non-finite")
        output.to_parquet(args.output_dir / "predictions.parquet", index=False)
        report["ranking_counts"] = {
            "n": int(len(output)),
            "lobt_eligible": int(available.sum()),
            "invalid_aobt_lobt_fallback": int(rank_invalid_fallback.sum()),
            "valid_aobt_high_disagreement": int(rank_valid_disagreement.sum()),
        }
        (args.output_dir / "validation.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "selected_weight": selected_weight,
        "seasonal_nested": report["seasonal_nested"],
        "forward_transfer": report["forward_transfer"],
        "counts": report["counts"],
        "ranking_ready": ranking_path.exists(),
    }, indent=2))


if __name__ == "__main__":
    main()
