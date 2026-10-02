"""Run the full pipeline on artificial data to check I/O and row alignment."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from solution import _unix_seconds, audit, predict, train  # noqa: E402


def make_rows(year: int, n: int, seed: int, first_id: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    months = rng.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] if year == 2025 else [1, 7], n)
    days = rng.integers(1, 27, n)
    hours = rng.integers(5, 23, n)
    minute = rng.integers(0, 60, n)
    takeoff = pd.to_datetime({"year": year, "month": months, "day": days,
                              "hour": hours, "minute": minute}, utc=True)
    phase = np.where(rng.random(n) < 0.75, "DEP", "ARR")
    airports = rng.choice(["EDDF", "EGLL", "LFPG"], n)
    taxi = 500 + 100 * (airports == "EGLL") + rng.normal(0, 90, n)
    taxi = np.maximum(taxi, 60)
    offblock = takeoff - pd.to_timedelta(taxi, unit="s")
    nm_offset = 25 * (airports == "EGLL") + rng.normal(0, 45, n)
    aobt = offblock + pd.to_timedelta(nm_offset, unit="s")
    missing = rng.random(n) < 0.12
    aobt = aobt.mask(missing)
    stand = rng.choice(["A12", "A13", "B4", "C22"], n)
    runway = rng.choice(["09L", "18", "27R"], n)
    dest = rng.choice(["LEMD", "LIRF", "LSZH"], n)
    result = pd.DataFrame({
        "MVT_ID_mvt": np.arange(first_id, first_id + n),
        "PHASE_mvt": phase,
        "ADEP_mvt": airports,
        "ADES_mvt": dest,
        "ADEP_flt": airports,
        "ADES_flt": dest,
        "ADES_FILED_flt": dest,
        "MVT_TIME_UTC_mvt": takeoff,
        "BLOCK_TIME_UTC_mvt": offblock,
        "SCHED_TIME_UTC_mvt": takeoff - pd.Timedelta(minutes=20),
        "AOBT_3_flt": aobt,
        "LOBT_flt": offblock + pd.Timedelta(minutes=1),
        "IOBT_flt": offblock - pd.Timedelta(minutes=3),
        "EOBT_1_flt": offblock + pd.Timedelta(minutes=2),
        "TAXITIME_SEC_mvt": taxi,
        "RUNWAY_mvt": runway,
        "STAND_mvt": stand,
        "AIRCRAFT_TYPE_mvt": rng.choice(["A320", "B738"], n),
        "AIRCRAFT_TYPE_flt": rng.choice(["A320", "B738", "E190"], n),
        "AIRCRAFT_OPERATOR_flt": rng.choice(["AAA", "BBB"], n),
        "MARKET_SEGMENT_flt": "Mainline",
        "WK_TBL_CAT_flt": "M",
        "FLIGHT_RULE_mvt": "I",
        "FLIGHT_RULE_flt": "I",
        "FLIGHT_TYPE_flt": "S",
        "FLIGHT_mvt": "AB123",
        "CALLSIGN_flt": "ABC123",
    })
    return result


def main() -> None:
    timestamps = pd.Series(pd.to_datetime(["2025-01-01T00:00:00Z"])).dt.as_unit("us")
    assert _unix_seconds(timestamps).tolist() == [1735689600]
    with tempfile.TemporaryDirectory(prefix="prc_smoke_") as tmp:
        root = Path(tmp)
        data = root / "data"
        data.mkdir()
        training = make_rows(2025, 5000, 1, 1)
        for month, frame in training.groupby(training["MVT_TIME_UTC_mvt"].dt.month):
            start = pd.Timestamp(2025, month, 1)
            end = start + pd.offsets.MonthBegin(1)
            frame.to_parquet(data / f"training_{start.date()}_{end.date()}.parquet", index=False)
        ranking = make_rows(2026, 350, 2, 100_000)
        departures = ranking["PHASE_mvt"] == "DEP"
        template = ranking.loc[departures, ["MVT_ID_mvt"]].copy()
        template["TAXITIME_SEC_mvt"] = np.nan
        template = template.sample(frac=1, random_state=3).reset_index(drop=True)
        ranking.loc[departures, ["BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt"]] = None
        ranking.to_parquet(data / "ranking.parquet", index=False)
        template.to_parquet(data / "submitting.parquet", index=False)
        audit(data)
        train(data, root / "artifacts", threads=2, rounds=25)
        output = root / "submissions" / "smoke_v1.parquet"
        predict(data, root / "artifacts", output, threads=2)
        result = pd.read_parquet(output)
        assert result["MVT_ID_mvt"].equals(template["MVT_ID_mvt"])
        assert np.isfinite(result["TAXITIME_SEC_mvt"]).all()
        print("Synthetic smoke test passed")


if __name__ == "__main__":
    main()
