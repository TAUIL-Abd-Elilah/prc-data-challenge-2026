"""Predeclared v7 + v8 fixed-weight comparison on 2025 local labels.

This evaluates saved out-of-fold predictions only. It never trains a model,
uses ranking outcomes or selects a new combination weight.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import deep_arrival_expert as arrival
import deep_timestamp_expert as deep


FOLDS = {"seasonal_jan_jul": (1, 7), "forward_nov_dec": (11, 12)}
EXPECTED_OOF_ROWS = 672428


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def assert_ids(actual: pd.Series, expected: pd.Series, label: str) -> None:
    a, e = pd.Index(actual), pd.Index(expected)
    if (a.has_duplicates or e.has_duplicates or a.isna().any() or e.isna().any()
            or len(a) != len(e) or not a.isin(e).all() or not e.isin(a).all()):
        raise ValueError(f"{label}: exact unique ID set differs")


def protocol(args: argparse.Namespace) -> dict:
    value = {
        "purpose": "Fixed v7+v8 local OOF comparison, no training or leaderboard feedback",
        "prerequisite": "Both standalone v7 and v8 reports promoted under their own frozen gates",
        "reference": "same frozen v6 all-finite 2025 OOF movement IDs and labels",
        "source_protocol_sha256": {
            "v7": sha256(args.v7_dir / "protocol.json"),
            "v8": sha256(args.v8_dir / "protocol.json"),
        },
        "existing_fold_policy": {
            "months": {name: list(months) for name, months in FOLDS.items()},
            "base": "v7 clipped candidate on every finite row",
            "expert": "v8 raw OOF expert on exact valid-AOBT IDs",
            "weight": "v8 standalone January/July-selected weight, unchanged",
            "formula": "clip(v7_candidate + weight*(v8_expert-v7_candidate), lower=0) only on valid AOBT",
            "gate": "RMSE improvement and UTC-day bootstrap 95% lower gain >0 separately in both folds",
        },
        "fresh_audit_policy": {
            "months": [4, 10],
            "base": "v7 saved fixed fresh blend: depth10 ARR prior + v7 selected weight*(v7 fresh expert-prior)",
            "expert": "v8 separately refitted April/October raw fresh expert",
            "weight": "same v8 standalone January/July-selected weight",
            "formula": "clip(v7_fresh_base + weight*(v8_fresh_expert-v7_fresh_base), lower=0)",
            "gate": "exact valid-AOBT IDs, aligned labels/time, positive paired UTC-day bootstrap 95% lower gain",
        },
        "decision": "accept fixed combination only if both existing folds and fresh paired audit pass; otherwise reject combination without retuning",
        "limitations": "Repeated 2025 comparison decisions; no untouched 2026 accuracy estimate.",
    }
    path = args.output_dir / "protocol.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise ValueError("Frozen v8 combination protocol changed")
    else:
        write_json(path, value)
    return value


def aligned_oof(args: argparse.Namespace) -> pd.DataFrame:
    reference = pd.read_parquet(args.v8_dir / "frozen_v6_oof_reference.parquet",
                                columns=["MVT_ID_mvt", "target", "fold", "month",
                                         "MVT_TIME_UTC_mvt", "a_valid"])
    v7 = pd.read_parquet(args.v7_dir / "validation_predictions.parquet",
                         columns=["MVT_ID_mvt", "target", "fold", "month",
                                  "MVT_TIME_UTC_mvt", "a_valid", "candidate"])
    v8 = pd.read_parquet(args.v8_dir / "validation_predictions.parquet",
                         columns=["MVT_ID_mvt", "target", "fold", "month",
                                  "MVT_TIME_UTC_mvt", "a_valid", "expert"])
    if len(reference) != EXPECTED_OOF_ROWS:
        raise ValueError("Frozen reference row count differs")
    for name, source in (("v7", v7), ("v8", v8)):
        assert_ids(source.MVT_ID_mvt, reference.MVT_ID_mvt, name)
        if len(source) != EXPECTED_OOF_ROWS:
            raise ValueError(f"{name} OOF row count differs")
    frame = reference.merge(v7, on="MVT_ID_mvt", how="left", sort=False,
                            validate="one_to_one", suffixes=("", "_v7"))
    frame = frame.merge(v8, on="MVT_ID_mvt", how="left", sort=False,
                        validate="one_to_one", suffixes=("", "_v8"))
    if len(frame) != len(reference) or not frame.MVT_ID_mvt.equals(reference.MVT_ID_mvt):
        raise ValueError("Existing-fold OOF join moved rows")
    for suffix in ("_v7", "_v8"):
        if (not np.array_equal(frame.fold.to_numpy(), frame[f"fold{suffix}"].to_numpy())
                or not np.array_equal(frame.month.to_numpy(),
                                      frame[f"month{suffix}"].to_numpy())
                or not np.array_equal(frame.a_valid.to_numpy(dtype=bool),
                                      frame[f"a_valid{suffix}"].to_numpy(dtype=bool))
                or not np.allclose(frame.target.to_numpy(dtype=float),
                                   frame[f"target{suffix}"].to_numpy(dtype=float),
                                   rtol=0, atol=1e-6)
                or not np.array_equal(
                    pd.to_datetime(frame.MVT_TIME_UTC_mvt, utc=True).to_numpy(),
                    pd.to_datetime(frame[f"MVT_TIME_UTC_mvt{suffix}"],
                                   utc=True).to_numpy())):
            raise ValueError(f"Existing-fold {suffix} labels, gate or time differ")
    expected_valid = frame.a_valid.to_numpy(dtype=bool)
    if (not np.array_equal(frame.expert.notna().to_numpy(), expected_valid)
            or not np.isfinite(frame.candidate.to_numpy(dtype=float)).all()
            or (frame.candidate.to_numpy(dtype=float) < 0).any()
            or not np.isfinite(frame.loc[expected_valid, "expert"]
                               .to_numpy(dtype=float)).all()):
        raise ValueError("Existing-fold v7 base or v8 expert coverage invalid")
    return frame


def fresh_audit(args: argparse.Namespace, weight: float,
                v7_weight: float) -> tuple[dict, pd.DataFrame]:
    prior = pd.read_parquet(args.prior_fresh_oof,
                            columns=["MVT_ID_mvt", "expert"])
    v7 = pd.read_parquet(args.v7_dir / "fresh_audit_predictions.parquet",
                         columns=["MVT_ID_mvt", "target", "MVT_TIME_UTC_mvt",
                                  "prior", "new", "candidate"])
    v8 = pd.read_parquet(args.v8_dir / "fresh_audit_predictions.parquet",
                         columns=["MVT_ID_mvt", "target", "MVT_TIME_UTC_mvt",
                                  "prior", "new"])
    assert_ids(v7.MVT_ID_mvt, prior.MVT_ID_mvt, "v7 fresh")
    assert_ids(v8.MVT_ID_mvt, prior.MVT_ID_mvt, "v8 fresh")
    expected = pd.read_parquet(args.cache_dir / "training_rows.parquet",
                               columns=["MVT_ID_mvt", "target", "proxy", "month"])
    proxy = expected.proxy.to_numpy(dtype=float)
    target = expected.target.to_numpy(dtype=float)
    valid = (expected.month.isin((4, 10)).to_numpy()
             & np.isfinite(target) & np.isfinite(proxy)
             & (proxy >= 0) & (proxy <= 7200))
    assert_ids(v7.MVT_ID_mvt, expected.loc[valid, "MVT_ID_mvt"],
               "fresh all-finite valid-AOBT April/October")
    frame = v7.merge(v8, on="MVT_ID_mvt", how="left", sort=False,
                     validate="one_to_one", suffixes=("_v7", "_v8"))
    frame = frame.merge(prior.rename(columns={"expert": "prior_reference"}),
                        on="MVT_ID_mvt", how="left", sort=False,
                        validate="one_to_one")
    if (len(frame) != len(v7) or not frame.MVT_ID_mvt.equals(v7.MVT_ID_mvt)
            or not np.isfinite(frame[["target_v7", "prior_v7", "new_v7",
                                      "candidate", "target_v8", "prior_v8",
                                      "new_v8", "prior_reference"]]
                               .to_numpy(dtype=float)).all()
            or not np.allclose(frame.target_v7, frame.target_v8, rtol=0,
                               atol=1e-6)
            or not np.allclose(frame.prior_v7, frame.prior_v8, rtol=0,
                               atol=1e-6)
            or not np.allclose(frame.prior_v7, frame.prior_reference, rtol=0,
                               atol=1e-6)
            or not np.array_equal(
                pd.to_datetime(frame.MVT_TIME_UTC_mvt_v7, utc=True).to_numpy(),
                pd.to_datetime(frame.MVT_TIME_UTC_mvt_v8, utc=True).to_numpy())):
        raise ValueError("Fresh paired labels, prior expert or times misaligned")
    clipped_prior = np.maximum(frame.prior_v7.to_numpy(dtype=float), 0)
    expected_v7_base = np.maximum(
        clipped_prior + v7_weight *
        (frame.new_v7.to_numpy(dtype=float) - clipped_prior), 0)
    if not np.allclose(frame.candidate.to_numpy(dtype=float), expected_v7_base,
                       rtol=0, atol=1e-6):
        raise ValueError("Saved v7 fresh comparator differs from its fixed blend")
    base = frame.candidate.to_numpy(dtype=float)
    alternative = frame.new_v8.to_numpy(dtype=float)
    combined = np.maximum(base + weight * (alternative-base), 0)
    frame["combined"] = combined
    frame.rename(columns={"target_v7": "target",
                          "MVT_TIME_UTC_mvt_v7": "MVT_TIME_UTC_mvt"},
                 inplace=True)
    bootstrap = arrival.bootstrap(frame, base, combined, seed=20261009)
    report = {"n": len(frame), "v7_rmse": deep.rmse(frame.target.to_numpy(), base),
              "combined_rmse": deep.rmse(frame.target.to_numpy(), combined),
              "bootstrap": bootstrap,
              "passed": (deep.rmse(frame.target.to_numpy(), combined)
                         < deep.rmse(frame.target.to_numpy(), base)
                         and bootstrap["gain_ci95_sec"][0] > 0)}
    return report, frame


def audit(args: argparse.Namespace) -> None:
    fixed = protocol(args)
    v7_report = json.loads((args.v7_dir / "validation.json").read_text())
    v8_report = json.loads((args.v8_dir / "validation.json").read_text())
    if not v7_report.get("promoted") or not v8_report.get("promoted"):
        raise ValueError("Standalone v7 and v8 must both pass local gates")
    for name, source, report in (("v7", args.v7_dir, v7_report),
                                 ("v8", args.v8_dir, v8_report)):
        if report.get("validation_predictions_sha256") != sha256(
                source / "validation_predictions.parquet"):
            raise ValueError(f"{name} saved OOF differs from its validation report")
    weight = v8_report["selected_weight"]
    if weight not in (0.1, 0.25, 0.5, 1.0):
        raise ValueError("v8 seasonal-selected weight invalid for combination")
    frame = aligned_oof(args)
    base = frame.candidate.to_numpy(dtype=float)
    alternative = frame.expert.fillna(frame.candidate).to_numpy(dtype=float)
    frame["combined"] = np.maximum(base + weight*(alternative-base), 0)
    fold_report = {}
    for name, months in FOLDS.items():
        part = frame.loc[frame.fold.eq(name)]
        if not part.month.isin(months).all():
            raise ValueError(f"{name} month coverage differs")
        y = part.target.to_numpy(dtype=float)
        old = part.candidate.to_numpy(dtype=float)
        new = part.combined.to_numpy(dtype=float)
        bootstrap = arrival.bootstrap(part, old, new, seed=20261009)
        fold_report[name] = {
            "n": len(part), "v7_rmse": deep.rmse(y, old),
            "combined_rmse": deep.rmse(y, new),
            "bootstrap": bootstrap,
            "passed": (deep.rmse(y, new) < deep.rmse(y, old)
                       and bootstrap["gain_ci95_sec"][0] > 0),
        }
    fresh, fresh_rows = fresh_audit(args, weight,
                                    v7_report["selected_weight"])
    path = args.output_dir / "validation_predictions.parquet"
    frame[["MVT_ID_mvt", "target", "fold", "month", "MVT_TIME_UTC_mvt",
           "a_valid", "candidate", "expert", "combined"]].to_parquet(
               path, index=False)
    fresh_path = args.output_dir / "fresh_audit_predictions.parquet"
    fresh_rows[["MVT_ID_mvt", "target", "MVT_TIME_UTC_mvt", "prior_v7",
                "candidate", "new_v8", "combined"]].to_parquet(
                    fresh_path, index=False)
    report = {
        "protocol": fixed,
        "weight": weight,
        "folds": fold_report,
        "fresh_audit": fresh,
        "accepted": (all(item["passed"] for item in fold_report.values())
                     and fresh["passed"]),
        "source_sha256": {
            "v7_validation": sha256(args.v7_dir / "validation.json"),
            "v8_validation": sha256(args.v8_dir / "validation.json"),
            "v7_oof": sha256(args.v7_dir / "validation_predictions.parquet"),
            "v8_oof": sha256(args.v8_dir / "validation_predictions.parquet"),
            "v7_fresh": sha256(args.v7_dir / "fresh_audit_predictions.parquet"),
            "v8_fresh": sha256(args.v8_dir / "fresh_audit_predictions.parquet"),
            "prior_fresh": sha256(args.prior_fresh_oof),
            "v6_reference": sha256(args.v8_dir / "frozen_v6_oof_reference.parquet"),
        },
        "output_sha256": {"existing_oof": sha256(path),
                          "fresh_oof": sha256(fresh_path)},
    }
    write_json(args.output_dir / "audit.json", report)
    print(json.dumps({"weight": weight, "folds": fold_report,
                      "fresh_audit": fresh, "accepted": report["accepted"]},
                     indent=2))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("prepare", "audit"), default="prepare")
    p.add_argument("--v7-dir", type=Path, default=Path("artifacts/v7-runway-traffic"))
    p.add_argument("--v8-dir", type=Path, default=Path("artifacts/v8-lightgbm"))
    p.add_argument("--cache-dir", type=Path, default=Path("artifacts/baseline"))
    p.add_argument("--prior-fresh-oof", type=Path,
                   default=Path("artifacts/v6-deep-arrival/fresh_new/fresh_apr_oct_oof.parquet"))
    p.add_argument("--output-dir", type=Path, default=Path("artifacts/v8-combo"))
    args = p.parse_args()
    if args.mode == "prepare":
        value = protocol(args)
        print(json.dumps({"protocol_path": str(args.output_dir / "protocol.json"),
                          "source_protocol_sha256": value["source_protocol_sha256"]},
                         indent=2))
    else:
        audit(args)


if __name__ == "__main__":
    main()
