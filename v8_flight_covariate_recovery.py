"""Audit whether a shared flight ID can recover released NM clock covariates.

This reads only movement IDs, flight join keys, phases, routes and NM timestamps.
It never reads taxi labels or departure block time and fits no model.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from solution import _training_files


CLOCKS = ("AOBT_3_flt", "LOBT_flt", "IOBT_flt", "EOBT_1_flt",
          "ARVT_1_flt", "ARVT_3_flt")
OUT = Path("artifacts/v8-flight-recovery")


def audit(paths: list[Path]) -> dict:
    scan = pl.scan_parquet([str(p) for p in paths]).select(
        ["MVT_ID_mvt", "FLIGHT_ID_mvt", "PHASE_mvt", "ADEP_mvt",
         "ADES_mvt", *CLOCKS])
    dep = scan.filter(pl.col("PHASE_mvt") == "DEP")
    arr = scan.filter(pl.col("PHASE_mvt") == "ARR")
    coverage = dep.select(
        pl.len().alias("departures"),
        pl.col("FLIGHT_ID_mvt").is_null().sum().alias("missing_flight_id"),
        *[pl.col(c).is_null().sum().alias(f"missing_{c}") for c in CLOCKS],
        (pl.col("FLIGHT_ID_mvt").is_not_null()
         & pl.col("AOBT_3_flt").is_null()).sum().alias("aobt_missing_with_id"),
    ).collect().to_dicts()[0]

    # This narrow subset bounds any possible recovery before touching labels.
    missing = dep.filter(pl.col("FLIGHT_ID_mvt").is_not_null()
                         & pl.col("AOBT_3_flt").is_null()).select(
        ["MVT_ID_mvt", "FLIGHT_ID_mvt", "ADEP_mvt", "ADES_mvt", *CLOCKS]
    ).collect()
    ids = missing.get_column("FLIGHT_ID_mvt").unique().to_list()
    if ids:
        counterpart = arr.filter(pl.col("FLIGHT_ID_mvt").is_in(ids)).select(
            ["FLIGHT_ID_mvt", "ADEP_mvt", "ADES_mvt", *CLOCKS]).collect()
        joined = missing.join(counterpart, on="FLIGHT_ID_mvt", how="left",
                              suffix="_arr")
        report_recovery = {
            "missing_aobt_with_id": missing.height,
            "unique_missing_departure_ids": missing.get_column("MVT_ID_mvt").n_unique(),
            "with_arrival_counterpart": joined.filter(
                pl.col("ADEP_mvt_arr").is_not_null()).get_column("MVT_ID_mvt").n_unique(),
            "with_route_consistent_counterpart": joined.filter(
                (pl.col("ADEP_mvt") == pl.col("ADEP_mvt_arr"))
                & (pl.col("ADES_mvt") == pl.col("ADES_mvt_arr"))
            ).get_column("MVT_ID_mvt").n_unique(),
            "recoverable_aobt": joined.filter(pl.col("AOBT_3_flt_arr").is_not_null())
            .get_column("MVT_ID_mvt").n_unique(),
            "recoverable_by_clock": {
                c: joined.filter(pl.col(f"{c}_arr").is_not_null())
                .get_column("MVT_ID_mvt").n_unique() for c in CLOCKS},
        }
    else:
        report_recovery = {"missing_aobt_with_id": 0,
                           "unique_missing_departure_ids": 0,
                           "with_arrival_counterpart": 0,
                           "with_route_consistent_counterpart": 0,
                           "recoverable_aobt": 0,
                           "recoverable_by_clock": {c: 0 for c in CLOCKS}}

    # One consistency check over all shared IDs: an ARR record for the same
    # flight should carry the same NM AOBT as its departure record.
    dep_aobt = dep.filter(pl.col("FLIGHT_ID_mvt").is_not_null()).select(
        ["FLIGHT_ID_mvt", "ADEP_mvt", "ADES_mvt", "AOBT_3_flt"])
    arr_aobt = arr.filter(pl.col("FLIGHT_ID_mvt").is_not_null()).select(
        ["FLIGHT_ID_mvt", "ADEP_mvt", "ADES_mvt", "AOBT_3_flt"])
    paired = dep_aobt.join(arr_aobt, on="FLIGHT_ID_mvt", how="inner",
                           suffix="_arr")
    consistency = paired.select(
        pl.len().alias("shared_id_pairs"),
        ((pl.col("ADEP_mvt") == pl.col("ADEP_mvt_arr"))
         & (pl.col("ADES_mvt") == pl.col("ADES_mvt_arr")))
        .sum().alias("route_consistent_pairs"),
        (pl.col("AOBT_3_flt").is_not_null()
         & pl.col("AOBT_3_flt_arr").is_not_null())
        .sum().alias("both_aobt_present"),
        ((pl.col("AOBT_3_flt") == pl.col("AOBT_3_flt_arr"))
         & pl.col("AOBT_3_flt").is_not_null())
        .sum().alias("aobt_exact_equal_pairs"),
    ).collect().to_dicts()[0]
    return {"coverage": coverage, "recovery": report_recovery,
            "shared_id_consistency": consistency}


def main() -> None:
    result = {
        "training_2025": audit(_training_files(Path("data"))),
        "ranking_2026": audit([Path("data/ranking.parquet")]),
        "source_columns": ["MVT_ID_mvt", "FLIGHT_ID_mvt", "PHASE_mvt",
                           "ADEP_mvt", "ADES_mvt", *CLOCKS],
        "forbidden_source_columns": ["BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt"],
        "label_reads": False,
        "model_fit": False,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "recovery_audit.json").write_text(json.dumps(result, indent=2),
                                              encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
