"""Read-only diagnostics for the PRC 2026 taxi-out data.

This profiles labels only in the 2025 training files. The 2026 ranking file is
used solely for covariate coverage and drift; its departure labels and block
times are never read.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")

import numpy as np
import polars as pl


MONTH_FILE = re.compile(r"training_2025-(\d{2})-01_.*\.parquet$")
SOURCE_TIMES = ("AOBT_3_flt", "LOBT_flt", "IOBT_flt", "EOBT_1_flt")
CATEGORIES = ("RUNWAY_mvt", "STAND_mvt", "AIRCRAFT_TYPE_mvt",
              "AIRCRAFT_OPERATOR_flt", "ADES_mvt")
TRAIN_COLUMNS = ("MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "ADES_mvt",
                 "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt",
                 *SOURCE_TIMES, *CATEGORIES)
RANK_COLUMNS = ("MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "ADES_mvt",
                "MVT_TIME_UTC_mvt", *SOURCE_TIMES, *CATEGORIES)
QUANTILES = (0.01, 0.05, 0.5, 0.95, 0.99, 0.999)


def number(value: float) -> float | None:
    return round(float(value), 4) if np.isfinite(value) else None


def values(df: pl.DataFrame, column: str) -> np.ndarray:
    return df.get_column(column).cast(pl.Float64).to_numpy()


def summary(arr: np.ndarray) -> dict:
    valid = arr[np.isfinite(arr)]
    result = {"n": int(valid.size), "coverage": number(valid.size / arr.size) if arr.size else None}
    if valid.size:
        result.update({"mean": number(valid.mean()), "min": number(valid.min()),
                       "max": number(valid.max()),
                       "quantiles": {str(q): number(np.quantile(valid, q)) for q in QUANTILES}})
    return result


def timestamp_coverage(df: pl.DataFrame) -> dict:
    return {col: number(1 - df.get_column(col).null_count() / df.height)
            for col in SOURCE_TIMES if col in df.columns}


def enrich(df: pl.DataFrame, training: bool) -> pl.DataFrame:
    expressions = [
        (pl.col("MVT_TIME_UTC_mvt") - pl.col(col)).dt.total_seconds()
        .cast(pl.Float64).alias(f"takeoff_minus_{col}") for col in SOURCE_TIMES
    ]
    expressions += [
        (pl.col("AOBT_3_flt") - pl.col("LOBT_flt")).dt.total_seconds()
        .cast(pl.Float64).alias("aobt_minus_lobt"),
        (pl.col("AOBT_3_flt") - pl.col("IOBT_flt")).dt.total_seconds()
        .cast(pl.Float64).alias("aobt_minus_iobt"),
        (pl.col("AOBT_3_flt") - pl.col("EOBT_1_flt")).dt.total_seconds()
        .cast(pl.Float64).alias("aobt_minus_eobt"),
        pl.col("MVT_TIME_UTC_mvt").dt.month().alias("month"),
    ]
    if training:
        expressions += [
            (pl.col("BLOCK_TIME_UTC_mvt") - pl.col(col)).dt.total_seconds()
            .cast(pl.Float64).alias(f"block_minus_{col}") for col in SOURCE_TIMES
        ]
        expressions.append(
            ((pl.col("MVT_TIME_UTC_mvt") - pl.col("BLOCK_TIME_UTC_mvt"))
             .dt.total_seconds() - pl.col("TAXITIME_SEC_mvt"))
            .cast(pl.Float64).alias("target_identity_error")
        )
    return df.with_columns(expressions)


def profile_group(df: pl.DataFrame, training: bool) -> dict:
    result = {"rows": df.height, "source_time_coverage": timestamp_coverage(df)}
    for col in CATEGORIES:
        result[f"{col}_coverage"] = number(1 - df.get_column(col).null_count() / df.height)
    for col in SOURCE_TIMES:
        proxy = values(df, f"takeoff_minus_{col}")
        plausible = np.isfinite(proxy) & (proxy >= 0) & (proxy <= 7200)
        result[f"{col}_proxy"] = {
            "plausible_coverage": number(plausible.mean()),
            "negative": int((np.isfinite(proxy) & (proxy < 0)).sum()),
            "over_7200": int((np.isfinite(proxy) & (proxy > 7200)).sum()),
            "plausible_seconds": summary(proxy[plausible]),
        }
        if training:
            target = values(df, "TAXITIME_SEC_mvt")
            use = plausible & np.isfinite(target)
            error = proxy[use] - target[use]
            result[f"{col}_proxy"]["against_target"] = {
                "n": int(use.sum()), "rmse": number(np.sqrt(np.mean(error ** 2))) if use.any() else None,
                "mae": number(np.mean(np.abs(error))) if use.any() else None,
                "bias": number(np.mean(error)) if use.any() else None,
                "error_seconds": summary(error),
            }
            gap = values(df, f"block_minus_{col}")
            finite_gap = gap[np.isfinite(gap)]
            result[f"block_minus_{col}"] = {
                "seconds": summary(gap),
                "abs_over_60": int((np.abs(finite_gap) > 60).sum()),
                "abs_over_300": int((np.abs(finite_gap) > 300).sum()),
                "abs_over_900": int((np.abs(finite_gap) > 900).sum()),
            }
    for col in ("aobt_minus_lobt", "aobt_minus_iobt", "aobt_minus_eobt"):
        result[col] = summary(values(df, col))
    if training:
        target = values(df, "TAXITIME_SEC_mvt")
        finite = target[np.isfinite(target)]
        result["target"] = {"seconds": summary(target),
                            "negative": int((finite < 0).sum()),
                            "over_3600": int((finite > 3600).sum()),
                            "over_7200": int((finite > 7200).sum()),
                            "over_14400": int((finite > 14400).sum())}
        identity = values(df, "target_identity_error")
        result["target_identity_error"] = {"seconds": summary(identity),
                                            "abs_over_1": int((np.isfinite(identity) & (np.abs(identity) > 1)).sum())}
    return result


def path_month(path: Path) -> int:
    match = MONTH_FILE.fullmatch(path.name)
    if not match:
        raise ValueError(f"Unexpected training filename: {path.name}")
    return int(match.group(1))


def scan_file(path: Path, columns: tuple[str, ...], training: bool) -> tuple[dict, pl.DataFrame]:
    schema = pl.scan_parquet(str(path)).collect_schema()
    missing = set(columns) - set(schema.names())
    if missing:
        raise ValueError(f"{path.name} is missing {sorted(missing)}")
    phases = (pl.scan_parquet(str(path)).group_by("PHASE_mvt").len().collect()
              .to_dicts())
    deps = (pl.scan_parquet(str(path)).filter(pl.col("PHASE_mvt") == "DEP")
            .select(list(dict.fromkeys(columns))).collect())
    ids = deps.get_column("MVT_ID_mvt")
    finite_ids = ids.drop_nulls().cast(pl.Float64).to_numpy()
    return {"schema": {k: str(v) for k, v in schema.items()},
            "phases": {str(x["PHASE_mvt"]): int(x["len"]) for x in phases},
            "departure_rows": deps.height,
            "id_null_count": ids.null_count(),
            "id_duplicate_count": deps.height - ids.n_unique(),
            "id_fractional_count": int(np.sum(finite_ids != np.floor(finite_ids))),
            "id_over_exact_float_limit_count": int(np.sum(np.abs(finite_ids) > 2 ** 53)),
            }, enrich(deps, training)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/diagnostics.json"))
    args = parser.parse_args()
    files = sorted(args.data_dir.glob("training_2025-*.parquet"))
    months = [path_month(p) for p in files]
    if sorted(months) != list(range(1, 13)):
        raise ValueError(f"Need exactly one complete file for each 2025 month; found {sorted(months)}")
    ranking = args.data_dir / "ranking.parquet"
    if not ranking.exists():
        raise FileNotFoundError(ranking)

    report: dict = {"generated_utc": datetime.now(timezone.utc).isoformat(),
                    "training": {}, "ranking": {}, "drift": {}}
    vocab: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    train_comparison: dict[str, dict[str, dict]] = defaultdict(dict)
    reference_schema = None
    for path in files:
        month = path_month(path)
        info, frame = scan_file(path, TRAIN_COLUMNS, True)
        if reference_schema is None:
            reference_schema = info["schema"]
        elif info["schema"] != reference_schema:
            report.setdefault("schema_differences", {})[path.name] = {
                k: v for k, v in info["schema"].items() if reference_schema.get(k) != v}
        per_airport = {}
        for airport_frame in frame.partition_by("ADEP_mvt", maintain_order=False):
            airport = str(airport_frame["ADEP_mvt"][0])
            profile = profile_group(airport_frame, True)
            per_airport[airport] = profile
            if month in (1, 7):
                train_comparison[str(month)][airport] = profile
            for col in CATEGORIES:
                vocab[airport][col].update(str(v) for v in airport_frame[col].drop_nulls().unique())
        report["training"][path.name] = {"month": month, "phases": info["phases"],
                                         "departure_rows": info["departure_rows"],
                                         "ids": {k: info[k] for k in info if k.startswith("id_")},
                                         "by_airport": per_airport}
        print(f"profiled {path.name}: {frame.height:,} departures", flush=True)
        del frame

    rank_info, rank_frame = scan_file(ranking, RANK_COLUMNS, False)
    report["training_schema"] = reference_schema
    report["ranking"]["schema"] = rank_info["schema"]
    report["ranking"]["phases"] = rank_info["phases"]
    report["ranking"]["departure_rows"] = rank_info["departure_rows"]
    report["ranking"]["ids"] = {k: rank_info[k] for k in rank_info if k.startswith("id_")}
    report["ranking"]["by_month_airport"] = {}
    for month_frame in rank_frame.partition_by("month", maintain_order=False):
        month = str(month_frame["month"][0])
        report["ranking"]["by_month_airport"][month] = {}
        for airport_frame in month_frame.partition_by("ADEP_mvt", maintain_order=False):
            airport = str(airport_frame["ADEP_mvt"][0])
            profile = profile_group(airport_frame, False)
            report["ranking"]["by_month_airport"][month][airport] = profile
            base = train_comparison.get(month, {}).get(airport)
            if base:
                drift = {}
                for col in SOURCE_TIMES:
                    key = f"{col}_proxy"
                    drift[f"{col}_plausible_coverage_delta"] = number(
                        profile[key]["plausible_coverage"] - base[key]["plausible_coverage"])
                    rank_median = profile[key]["plausible_seconds"].get("quantiles", {}).get("0.5")
                    train_median = base[key]["plausible_seconds"].get("quantiles", {}).get("0.5")
                    drift[f"{col}_median_proxy_delta_sec"] = (
                        number(rank_median - train_median)
                        if rank_median is not None and train_median is not None else None)
                for col in CATEGORIES:
                    nonnull = airport_frame[col].drop_nulls()
                    total = len(nonnull)
                    unseen = sum(str(v) not in vocab[airport][col] for v in nonnull)
                    drift[f"{col}_novel_rate"] = number(unseen / total) if total else None
                    drift[f"{col}_coverage_delta"] = number(
                        profile[f"{col}_coverage"] - base[f"{col}_coverage"])
                report["drift"].setdefault(month, {})[airport] = drift

    training_groups = [profile for file in report["training"].values()
                       for profile in file["by_airport"].values()]
    ranking_groups = [profile for month in report["ranking"]["by_month_airport"].values()
                      for profile in month.values()]
    train_rows = sum(profile["rows"] for profile in training_groups)
    rank_rows = sum(profile["rows"] for profile in ranking_groups)
    report["summary"] = {
        "training_departures": train_rows,
        "ranking_departures": rank_rows,
        "training_target_negative": sum(profile["target"]["negative"] for profile in training_groups),
        "training_target_over_3600": sum(profile["target"]["over_3600"] for profile in training_groups),
        "training_target_over_7200": sum(profile["target"]["over_7200"] for profile in training_groups),
        "training_target_over_14400": sum(profile["target"]["over_14400"] for profile in training_groups),
        "training_target_identity_errors_over_1s": sum(
            profile["target_identity_error"]["abs_over_1"] for profile in training_groups),
        "sources": {},
    }
    for col in SOURCE_TIMES:
        key = f"{col}_proxy"
        counts = [profile[key]["against_target"]["n"] for profile in training_groups]
        squared_error = sum(n * profile[key]["against_target"]["rmse"] ** 2
                            for n, profile in zip(counts, training_groups)
                            if n and profile[key]["against_target"]["rmse"] is not None)
        report["summary"]["sources"][col] = {
            "training_plausible_coverage": number(sum(
                profile["rows"] * profile[key]["plausible_coverage"]
                for profile in training_groups) / train_rows),
            "ranking_plausible_coverage": number(sum(
                profile["rows"] * profile[key]["plausible_coverage"]
                for profile in ranking_groups) / rank_rows),
            "training_proxy_rmse_sec": number(np.sqrt(squared_error / sum(counts)))
            if sum(counts) else None,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(f"saved {args.output.resolve()}")


if __name__ == "__main__":
    main()
