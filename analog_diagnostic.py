"""Test exact-flight historical analogs on local missing-flight-record OOF rows.

Each fold's lookup excludes its held-out months. Reads training labels only.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "artifacts/diagnostic-v3"
MONTHS = [(i, f"2025-{i:02d}-01", f"2025-{i+1:02d}-01" if i < 12 else "2026-01-01")
          for i in range(1, 13)]
COLS = ["FLIGHT_ID_mvt", "FLIGHT_mvt", "ADEP_mvt", "PHASE_mvt",
        "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "TAXITIME_SEC_mvt"]


def load_history() -> pd.DataFrame:
    parts = []
    for month, start, end in MONTHS:
        raw = pd.read_parquet(ROOT / "data" / f"training_{start}_{end}.parquet", columns=COLS)
        use = raw.PHASE_mvt.eq("DEP") & raw.FLIGHT_ID_mvt.isna() & raw.TAXITIME_SEC_mvt.notna()
        raw = raw.loc[use].copy()
        raw["month"] = month
        raw["schedule_proxy"] = (raw.MVT_TIME_UTC_mvt - raw.SCHED_TIME_UTC_mvt).dt.total_seconds()
        raw["residual"] = raw.TAXITIME_SEC_mvt - raw.schedule_proxy
        parts.append(raw[["FLIGHT_mvt", "ADEP_mvt", "month", "TAXITIME_SEC_mvt",
                          "schedule_proxy", "residual"]])
    return pd.concat(parts, ignore_index=True)


def lookup(history: pd.DataFrame, months: tuple[int, ...]) -> pd.DataFrame:
    train = history[~history.month.isin(months)]
    return (train.groupby(["ADEP_mvt", "FLIGHT_mvt"], dropna=False, observed=True)
            .agg(n=("TAXITIME_SEC_mvt", "size"), mean_target=("TAXITIME_SEC_mvt", "mean"),
                 mean_residual=("residual", "mean"), mean_schedule=("schedule_proxy", "mean"),
                 long_rate=("TAXITIME_SEC_mvt", lambda x: (x > 7200).mean()))
            .reset_index())


def score(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - p) ** 2)))


def main() -> None:
    history = load_history()
    oof = pd.read_parquet(OUT / "row_diagnostics.parquet")
    oof = oof[oof.source_class.eq("invalid_AOBT_no_flight_record")].copy()
    oof["month"] = oof.source_month.astype(int)
    folds = {
        "seasonal_jan_jul": (1, 7),
        "forward_nov_dec": (11, 12),
    }
    details = []
    for fold, months in folds.items():
        x = oof[oof.fold.eq(fold)].merge(lookup(history, months),
            on=["ADEP_mvt", "FLIGHT_mvt"], how="left", validate="many_to_one")
        x["analog_direct"] = x.mean_target
        x["analog_residual"] = x.schedule_proxy + x.mean_residual
        x["baseline_se"] = (x.target - x.selected) ** 2
        for strategy in ("analog_direct", "analog_residual"):
            for min_n in (1, 2, 3, 5):
                mask = x.n.ge(min_n).fillna(False)
                for weight in (.1, .25, .5, 1.):
                    pred = x.selected.copy()
                    pred.loc[mask] = np.clip(
                        pred.loc[mask] * (1 - weight) + x.loc[mask, strategy] * weight,
                        0, 120000)
                    details.append({
                        "fold": fold, "strategy": strategy, "min_n": min_n,
                        "weight": weight, "n_total": len(x), "n_matched": int(mask.sum()),
                        "rmse": round(score(x.target.to_numpy(), pred.to_numpy()), 3),
                        "base_rmse": round(score(x.target.to_numpy(), x.selected.to_numpy()), 3),
                        "matched_rmse": round(score(x.loc[mask, "target"].to_numpy(),
                                                     pred.loc[mask].to_numpy()), 3) if mask.any() else None,
                    })
        top = x.nlargest(20, "baseline_se")[["ADEP_mvt", "FLIGHT_mvt", "target",
            "selected", "schedule_proxy", "n", "mean_target", "mean_residual", "mean_schedule"]]
        (OUT / f"analog_{fold}_top20.json").write_text(
            top.to_json(orient="records", indent=2), encoding="utf-8")
    report = {
        "scope": "Local training labels; fold lookup excludes held-out months; no ranking data",
        "history_no_flight_record_n": len(history),
        "candidate_scores": details,
    }
    (OUT / "analog_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "history_no_flight_record_n": len(history),
        "best_seasonal": min((x for x in details if x["fold"] == "seasonal_jan_jul"),
                             key=lambda d: d["rmse"]),
        "best_forward": min((x for x in details if x["fold"] == "forward_nov_dec"),
                            key=lambda d: d["rmse"]),
    }, indent=2))


if __name__ == "__main__":
    main()
