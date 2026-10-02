"""Analyze unusual training labels using only test-available covariates.

The ranking file is read only for covariate distributions and matching checks.
No ranking labels or ranking off-block targets are loaded or reconstructed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")

import numpy as np
import polars as pl


TRAIN_PATTERN = re.compile(r"training_2025-\d\d-01_202[56]-\d\d-01\.parquet$")
SOURCES = ("AOBT_3_flt", "LOBT_flt", "IOBT_flt", "EOBT_1_flt", "SCHED_TIME_UTC_mvt")


def safe(value: float | None) -> float | None:
    return round(float(value), 3) if value is not None and np.isfinite(value) else None


def source_errors(frame: pl.DataFrame) -> dict:
    target = frame["TAXITIME_SEC_mvt"].cast(pl.Float64).to_numpy()
    result = {}
    for col in SOURCES:
        proxy = frame[f"proxy_{col}"].cast(pl.Float64).to_numpy()
        mask = np.isfinite(proxy) & np.isfinite(target)
        errors = proxy[mask] - target[mask]
        result[col] = {
            "available": int(mask.sum()),
            "median_absolute_error_sec": safe(np.median(np.abs(errors))) if errors.size else None,
            "rmse_sec": safe(np.sqrt(np.mean(errors ** 2))) if errors.size else None,
            "within_60_sec": int((np.abs(errors) <= 60).sum()),
            "within_300_sec": int((np.abs(errors) <= 300).sum()),
        }
    return result


def summarize_anomalies(frame: pl.DataFrame) -> dict:
    categories = {
        "long_over_7200": frame.filter(pl.col("TAXITIME_SEC_mvt") > 7200),
        "negative": frame.filter(pl.col("TAXITIME_SEC_mvt") < 0),
        "long_nm_present": frame.filter((pl.col("TAXITIME_SEC_mvt") > 7200) &
                                        pl.col("AOBT_3_flt").is_not_null()),
        "long_nm_missing": frame.filter((pl.col("TAXITIME_SEC_mvt") > 7200) &
                                        pl.col("AOBT_3_flt").is_null()),
        "lirf_long_nm_missing": frame.filter((pl.col("ADEP_mvt") == "LIRF") &
                                             (pl.col("TAXITIME_SEC_mvt") > 7200) &
                                             pl.col("AOBT_3_flt").is_null()),
    }
    return {
        name: {
            "rows": part.height,
            "by_airport": {str(k): int(n) for k, n in part.group_by("ADEP_mvt").len().iter_rows()},
            "nm_missing": part["AOBT_3_flt"].null_count(),
            "flight_id_missing": part["FLIGHT_ID_mvt"].null_count(),
            "target_min": int(part["TAXITIME_SEC_mvt"].min()) if part.height else None,
            "target_max": int(part["TAXITIME_SEC_mvt"].max()) if part.height else None,
            "source_error": source_errors(part),
        }
        for name, part in categories.items()
    }


def schedule_bands(frame: pl.DataFrame, training: bool) -> list[dict]:
    bounds = ((-1000000, 0), (0, 1800), (1800, 3600), (3600, 5400),
              (5400, 7200), (7200, 12000), (12000, 30000), (30000, 200000))
    output = []
    for lo, hi in bounds:
        part = frame.filter(pl.col("proxy_SCHED_TIME_UTC_mvt").is_between(lo, hi, closed="left"))
        row = {"range_sec": [lo, hi], "rows": part.height}
        if training:
            target = part["TAXITIME_SEC_mvt"].cast(pl.Float64).to_numpy()
            proxy = part["proxy_SCHED_TIME_UTC_mvt"].cast(pl.Float64).to_numpy()
            diff = np.abs(proxy - target)
            row.update({"target_over_7200": int((target > 7200).sum()),
                        "schedule_within_60_sec": int((diff <= 60).sum()),
                        "median_absolute_error_sec": safe(np.median(diff)) if diff.size else None})
        output.append(row)
    return output


def departure_frame(paths: list[Path], training: bool) -> pl.DataFrame:
    selected = ["MVT_ID_mvt", "FLIGHT_ID_mvt", "FLIGHT_mvt", "PHASE_mvt",
                "ADEP_mvt", "ADES_mvt", "ADEP_flt", "ADES_flt",
                "MVT_TIME_UTC_mvt", *SOURCES]
    if training:
        selected.append("TAXITIME_SEC_mvt")
    scan = pl.scan_parquet([str(p) for p in paths]).filter(pl.col("PHASE_mvt") == "DEP")
    result = scan.select(list(dict.fromkeys(selected)))
    result = result.with_columns([
        (pl.col("MVT_TIME_UTC_mvt") - pl.col(col)).dt.total_seconds()
        .cast(pl.Float64).alias(f"proxy_{col}") for col in SOURCES
    ])
    # Only a small subset is materialized for expensive row-level anomaly work.
    return result.filter(pl.col("AOBT_3_flt").is_null() |
                         (pl.col("TAXITIME_SEC_mvt") > 7200) |
                         (pl.col("TAXITIME_SEC_mvt") < 0)).collect() if training else (
                             result.filter(pl.col("AOBT_3_flt").is_null()).collect())


def pair_recovery(paths: list[Path], missing: pl.DataFrame) -> dict:
    candidates = missing.filter(pl.col("FLIGHT_ID_mvt").is_not_null())
    ids = candidates["FLIGHT_ID_mvt"].to_list()
    arrivals = (pl.scan_parquet([str(p) for p in paths])
                .filter((pl.col("PHASE_mvt") == "ARR") & pl.col("FLIGHT_ID_mvt").is_in(ids))
                .select("FLIGHT_ID_mvt", "AOBT_3_flt", "LOBT_flt")
                .collect())
    return {"missing_aobt_with_flight_id": candidates.height,
            "paired_arrival_rows": arrivals.height,
            "recovered_aobt": int(arrivals["AOBT_3_flt"].is_not_null().sum()),
            "recovered_lobt": int(arrivals["LOBT_flt"].is_not_null().sum())}


def airport_agreement(paths: list[Path], training: bool) -> dict:
    scan = (pl.scan_parquet([str(p) for p in paths])
            .filter((pl.col("PHASE_mvt") == "DEP") & pl.col("AOBT_3_flt").is_not_null()))
    result = scan.select(
        pl.len().alias("nm_matched_departures"),
        (pl.col("ADEP_mvt") != pl.col("ADEP_flt")).sum().alias("adep_disagree"),
        (pl.col("ADES_mvt") != pl.col("ADES_flt")).sum().alias("ades_disagree"),
        pl.col("ADES_flt").null_count().alias("ades_flt_missing"),
    ).collect().to_dicts()[0]
    return {k: int(v) for k, v in result.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/anomaly_report.json"))
    args = parser.parse_args()
    files = sorted(p for p in args.data_dir.glob("training_2025-*.parquet")
                   if TRAIN_PATTERN.fullmatch(p.name))
    if len(files) != 12:
        raise ValueError(f"Expected 12 canonical training files, found {len(files)}")
    rank = args.data_dir / "ranking.parquet"
    if not rank.exists():
        raise FileNotFoundError(rank)

    training = departure_frame(files, True)
    ranking = departure_frame([rank], False)
    report = {
        "training_files": [p.name for p in files],
        "notes": "Training target used for diagnostics only. Ranking target and block time were not read.",
        "training_anomalies": summarize_anomalies(training),
        "all_missing_nm": {
            "training": {"rows": int(training.filter(pl.col("AOBT_3_flt").is_null()).height),
                         "flight_id_missing": int(training.filter(pl.col("AOBT_3_flt").is_null())["FLIGHT_ID_mvt"].null_count())},
            "ranking": {"rows": int(ranking.height),
                        "flight_id_missing": int(ranking["FLIGHT_ID_mvt"].null_count())},
        },
        "lirf_missing_nm_schedule_bands": {
            "training": schedule_bands(training.filter((pl.col("ADEP_mvt") == "LIRF") &
                                                     pl.col("AOBT_3_flt").is_null()), True),
            "ranking": schedule_bands(ranking.filter(pl.col("ADEP_mvt") == "LIRF"), False),
        },
        "flight_id_arrival_recovery": {
            "training": pair_recovery(files, training.filter(pl.col("AOBT_3_flt").is_null())),
            "ranking": pair_recovery([rank], ranking),
        },
        "movement_vs_nm_airport_agreement": {
            "training": airport_agreement(files, True),
            "ranking": airport_agreement([rank], False),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Saved {args.output.resolve()}")


if __name__ == "__main__":
    main()
