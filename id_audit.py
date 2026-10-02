"""Assess movement-ID neighborhood as a covariate-only anomaly signal.

Neighbor features use only supplied ID/time/phase/airport/type/flight fields.
Validation targets are joined only after neighbor features have been built.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "artifacts/id-audit"
COLS = ["MVT_ID_mvt", "MVT_TIME_UTC_mvt", "PHASE_mvt", "ADEP_mvt",
        "AIRCRAFT_TYPE_mvt", "FLIGHT_mvt"]


def load_movements() -> pd.DataFrame:
    parts = []
    for m in range(1, 13):
        start = f"2025-{m:02d}-01"
        end = f"2025-{m + 1:02d}-01" if m < 12 else "2026-01-01"
        path = ROOT / "data" / f"training_{start}_{end}.parquet"
        parts.append(pd.read_parquet(path, columns=COLS))
    x = pd.concat(parts, ignore_index=True)
    x = x[x.MVT_ID_mvt.notna() & x.MVT_TIME_UTC_mvt.notna()].copy()
    if x.MVT_ID_mvt.duplicated().any():
        raise ValueError("Duplicate movement IDs in canonical training files")
    return x


def add_neighbors(x: pd.DataFrame, airport_only: bool) -> pd.DataFrame:
    x = x.sort_values("MVT_ID_mvt").copy()
    prefix = "airport" if airport_only else "global"
    groups = x.groupby("ADEP_mvt", dropna=False, sort=False) if airport_only else None
    for direction, shift in (("prev", 1), ("next", -1)):
        for col in COLS:
            neighbor = groups[col].shift(shift) if airport_only else x[col].shift(shift)
            if col == "MVT_ID_mvt":
                x[f"{prefix}_{direction}_id_gap"] = (x[col] - neighbor).abs()
            elif col == "MVT_TIME_UTC_mvt":
                signed = (x[col] - neighbor).dt.total_seconds()
                if direction == "next":
                    signed = -signed
                x[f"{prefix}_{direction}_time_step_sec"] = signed
            elif col in ("ADEP_mvt", "PHASE_mvt", "AIRCRAFT_TYPE_mvt", "FLIGHT_mvt"):
                label = {"ADEP_mvt": "same_airport", "PHASE_mvt": "same_phase",
                         "AIRCRAFT_TYPE_mvt": "same_type", "FLIGHT_mvt": "same_flight"}[col]
                x[f"{prefix}_{direction}_{label}"] = x[col].eq(neighbor).fillna(False)
    cols = ["MVT_ID_mvt"] + [c for c in x.columns if c.startswith(prefix + "_")]
    return x[cols]


def nearby_flight_matches(x: pd.DataFrame) -> pd.DataFrame:
    ordered = x.sort_values("MVT_ID_mvt")
    keys = ordered.FLIGHT_mvt.fillna("").to_numpy()
    airports = ordered.ADEP_mvt.fillna("").to_numpy()
    ids = ordered.MVT_ID_mvt.to_numpy(dtype=float)
    n = len(ordered)
    prev_match = np.zeros(n, dtype=bool)
    next_match = np.zeros(n, dtype=bool)
    for offset in range(1, 6):
        prev_match[offset:] |= (
            (keys[offset:] == keys[:-offset]) & (keys[offset:] != "")
            & (airports[offset:] == airports[:-offset])
            & (ids[offset:] - ids[:-offset] <= 10))
        next_match[:-offset] |= (
            (keys[:-offset] == keys[offset:]) & (keys[:-offset] != "")
            & (airports[:-offset] == airports[offset:])
            & (ids[offset:] - ids[:-offset] <= 10))
    return pd.DataFrame({"MVT_ID_mvt": ids,
                         "global_prev5_same_flight": prev_match,
                         "global_next5_same_flight": next_match})


def summarize(x: pd.DataFrame) -> dict:
    if len(x) == 0:
        return {"n": 0}
    err = x.target.to_numpy(dtype=float) - x.selected.to_numpy(dtype=float)
    return {
        "n": len(x),
        "rmse_sec": round(float(np.sqrt(np.mean(err ** 2))), 3),
        "mean_abs_error_sec": round(float(np.mean(np.abs(err))), 3),
        "large_error_gt3000": int((np.abs(err) > 3000).sum()),
        "target_gt7200": int((x.target > 7200).sum()),
        "sse": round(float(np.sum(err ** 2)), 3),
    }


def evaluate_mask(x: pd.DataFrame, mask: pd.Series) -> dict:
    return {"flagged": summarize(x[mask]), "unflagged": summarize(x[~mask])}


def main() -> None:
    movements = load_movements()
    global_neighbors = add_neighbors(movements, airport_only=False)
    airport_neighbors = add_neighbors(movements, airport_only=True)
    flight_neighbors = nearby_flight_matches(movements)
    oof = pd.read_parquet(ROOT / "artifacts/diagnostic-v3/row_diagnostics.parquet",
                          columns=["MVT_ID_mvt", "target", "selected", "fold",
                                   "ADEP_mvt", "source_class", "FLIGHT_mvt"])
    x = oof.merge(global_neighbors, on="MVT_ID_mvt", how="left", validate="one_to_one")
    x = x.merge(airport_neighbors, on="MVT_ID_mvt", how="left", validate="one_to_one")
    x = x.merge(flight_neighbors, on="MVT_ID_mvt", how="left", validate="one_to_one")
    if x.global_prev_time_step_sec.isna().all():
        raise ValueError("Movement ID join failed")
    report = {
        "scope": "Neighbor features from supplied local training covariates only; targets used for OOF assessment",
        "movement_count": len(movements),
        "overall": summarize(x),
        "folds": {},
        "LFPG_no_record": {},
    }
    for fold, z in x.groupby("fold"):
        report["folds"][fold] = summarize(z)
    groups = {
        "all": x,
        "LFPG_no_record": x[(x.ADEP_mvt == "LFPG") & (x.source_class == "invalid_AOBT_no_flight_record")],
        "LIRF_no_record": x[(x.ADEP_mvt == "LIRF") & (x.source_class == "invalid_AOBT_no_flight_record")],
    }
    for group_name, z in groups.items():
        result = {"all": summarize(z), "folds": {}}
        for fold, f in z.groupby("fold"):
            result["folds"][fold] = {}
            for feature in (
                "global_prev_time_step_sec", "global_next_time_step_sec",
                "airport_prev_time_step_sec", "airport_next_time_step_sec",
            ):
                step = f[feature]
                for threshold in (0, -3600, -21600, -86400):
                    key = f"{feature}_lt_{threshold}"
                    result["folds"][fold][key] = evaluate_mask(f, step < threshold)
            for feature in (
                "global_prev_same_flight", "global_next_same_flight",
                "airport_prev_same_flight", "airport_next_same_flight",
                "global_prev5_same_flight", "global_next5_same_flight",
            ):
                result["folds"][fold][feature] = evaluate_mask(f, f[feature])
        report[group_name] = result

    top = x.assign(sq=(x.target - x.selected) ** 2).nlargest(30, "sq")
    top_cols = (["MVT_ID_mvt", "ADEP_mvt", "FLIGHT_mvt", "fold", "target", "selected", "sq"]
                + [c for c in x.columns if c.startswith("global_") or c.startswith("airport_")])
    report["top_30"] = top[top_cols].replace({np.nan: None}).to_dict(orient="records")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    x.to_parquet(OUT / "oof_neighbor_features.parquet", index=False)
    print(json.dumps({"movement_count": len(movements), "overall": report["overall"],
                      "output": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
