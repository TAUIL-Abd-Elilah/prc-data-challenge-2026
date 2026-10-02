"""Audit and derive prediction-time features from released ARR movement rows.

Never reads a departure BLOCK or TAXITIME column from raw files. Local OOF
departure labels are read only after covariate matching for validation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from lightgbm import LGBMRegressor

from solution import _training_files


ARR_COLS = ["MVT_ID_mvt", "FLIGHT_ID_mvt", "FLIGHT_mvt", "ADEP_mvt",
            "ADES_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt",
            "TAXITIME_SEC_mvt", "AIRCRAFT_TYPE_mvt", "RUNWAY_mvt",
            "STAND_mvt", "AOBT_3_flt", "LOBT_flt", "IOBT_flt", "EOBT_1_flt"]
DEP_COLS = ["MVT_ID_mvt", "FLIGHT_ID_mvt", "FLIGHT_mvt", "ADEP_mvt",
            "ADES_mvt", "MVT_TIME_UTC_mvt", "AIRCRAFT_TYPE_mvt", "RUNWAY_mvt",
            "STAND_mvt", "AOBT_3_flt", "LOBT_flt"]
KEYS = ["FLIGHT_mvt", "ADEP_mvt", "ADES_mvt"]
SOURCES = ["AOBT_3_flt", "LOBT_flt", "IOBT_flt", "EOBT_1_flt"]


def load_arrivals(files: list[Path]) -> pd.DataFrame:
    """The PHASE filter executes before selecting taxi and block columns."""
    paths = [str(x) for x in files]
    arr = (pl.scan_parquet(paths)
           .filter(pl.col("PHASE_mvt") == "ARR")
           .select(ARR_COLS).collect().to_pandas())
    if arr.MVT_ID_mvt.isna().any() or arr.MVT_ID_mvt.duplicated().any():
        raise ValueError("Arrival movement IDs must be unique and non-null")
    return arr


def load_validation_departures(files: list[Path], oof: pd.DataFrame) -> pd.DataFrame:
    ids = pd.Index(oof.MVT_ID_mvt)
    parts = []
    for path in files:
        raw = pd.read_parquet(path, columns=DEP_COLS)
        parts.append(raw.loc[raw.MVT_ID_mvt.isin(ids)])
    deps = pd.concat(parts, ignore_index=True)
    if deps.MVT_ID_mvt.isna().any() or deps.MVT_ID_mvt.duplicated().any():
        raise ValueError("Validation departure movement IDs must be unique and non-null")
    deps = deps.merge(oof, on="MVT_ID_mvt", how="inner", validate="one_to_one")
    if len(deps) != len(oof):
        raise ValueError("Validation OOF and raw departure covariates differ")
    return deps


def exact_id_match(deps: pd.DataFrame, arr: pd.DataFrame) -> pd.DataFrame:
    unique = arr[arr.FLIGHT_ID_mvt.notna()].copy()
    unique = unique[~unique.FLIGHT_ID_mvt.duplicated(keep=False)]
    rename = {c: f"{c}_arr" for c in unique.columns if c != "FLIGHT_ID_mvt"}
    unique = unique.rename(columns=rename)
    out = deps.merge(unique, on="FLIGHT_ID_mvt", how="left", validate="many_to_one")
    dt = (out.MVT_TIME_UTC_mvt_arr - out.MVT_TIME_UTC_mvt).dt.total_seconds()
    route = (out.ADEP_mvt.eq(out.ADEP_mvt_arr)
             & out.ADES_mvt.eq(out.ADES_mvt_arr))
    valid = route & dt.between(900, 64800)
    out.loc[~valid, "MVT_ID_mvt_arr"] = np.nan
    return out


def strict_name_match(deps: pd.DataFrame, arr: pd.DataFrame) -> pd.DataFrame:
    left = deps.dropna(subset=KEYS + ["MVT_TIME_UTC_mvt"]).copy()
    right = arr.dropna(subset=KEYS + ["MVT_TIME_UTC_mvt"]).copy()
    left = left[left.FLIGHT_mvt.astype("string").str.len().gt(0)]
    right = right[right.FLIGHT_mvt.astype("string").str.len().gt(0)]
    right = right.sort_values("MVT_TIME_UTC_mvt")
    right["next_same_key_arrival"] = (
        right.groupby(KEYS, sort=False)["MVT_TIME_UTC_mvt"].shift(-1))
    rename = {c: f"{c}_arr" for c in right.columns if c not in KEYS}
    right = right.rename(columns=rename)
    left = left.sort_values("MVT_TIME_UTC_mvt")
    out = pd.merge_asof(
        left, right, left_on="MVT_TIME_UTC_mvt", right_on="MVT_TIME_UTC_mvt_arr",
        by=KEYS, direction="forward", tolerance=pd.Timedelta(hours=18))
    dt = (out.MVT_TIME_UTC_mvt_arr - out.MVT_TIME_UTC_mvt).dt.total_seconds()
    next_dt = (out.next_same_key_arrival_arr - out.MVT_TIME_UTC_mvt).dt.total_seconds()
    same_type = (out.AIRCRAFT_TYPE_mvt.notna()
                 & out.AIRCRAFT_TYPE_mvt.eq(out.AIRCRAFT_TYPE_mvt_arr))
    valid = dt.between(900, 64800) & ~next_dt.between(0, 64800) & same_type
    out.loc[~valid, "MVT_ID_mvt_arr"] = np.nan
    return out


def audit_one(deps: pd.DataFrame, match: pd.DataFrame, label: str) -> dict:
    joined = deps[["MVT_ID_mvt", "target", "selected", "fold", "a_valid",
                   "ADEP_mvt", "MVT_TIME_UTC_mvt"]].merge(
        match[["MVT_ID_mvt", "MVT_ID_mvt_arr", "MVT_TIME_UTC_mvt_arr"]
              + [f"{c}_arr" for c in SOURCES]],
        on="MVT_ID_mvt", how="left", validate="one_to_one")
    use = joined[~joined.a_valid].copy()
    matched = use.MVT_ID_mvt_arr.notna()
    report: dict = {"label": label, "invalid_rows": len(use),
                    "matched_arrival_rows": int(matched.sum()), "sources": {},
                    "by_fold": {}}
    for source in SOURCES:
        proxy = (use.MVT_TIME_UTC_mvt - use[f"{source}_arr"]).dt.total_seconds()
        available = matched & np.isfinite(proxy) & proxy.between(0, 172800)
        if available.any():
            y = use.loc[available, "target"].to_numpy(dtype=float)
            raw = proxy[available].to_numpy(dtype=float)
            baseline = use.loc[available, "selected"].to_numpy(dtype=float)
            record = {"n": int(available.sum()),
                      "raw_proxy_rmse_sec": float(np.sqrt(np.mean((y - raw) ** 2))),
                      "v4_rmse_sec": float(np.sqrt(np.mean((y - baseline) ** 2))),
                      "raw_proxy_median_abs_error_sec": float(np.median(np.abs(y - raw))),
                      "within_60_sec": int((np.abs(y - raw) <= 60).sum())}
        else:
            record = {"n": 0}
        report["sources"][source] = record
        for fold, group in use.groupby("fold"):
            avail_f = available.loc[group.index]
            if not avail_f.any():
                report["by_fold"].setdefault(fold, {})[source] = {"n": 0}
                continue
            yy = group.loc[avail_f, "target"].to_numpy(dtype=float)
            pp = proxy.loc[group.index][avail_f].to_numpy(dtype=float)
            bb = group.loc[avail_f, "selected"].to_numpy(dtype=float)
            report["by_fold"].setdefault(fold, {})[source] = {
                "n": int(avail_f.sum()),
                "raw_proxy_rmse_sec": float(np.sqrt(np.mean((yy - pp) ** 2))),
                "v4_rmse_sec": float(np.sqrt(np.mean((yy - bb) ** 2))),
                "within_60_sec": int((np.abs(yy - pp) <= 60).sum())}
    return report


def recovery_audit(args: argparse.Namespace) -> dict:
    files = _training_files(args.data_dir)
    oof = pd.read_parquet(args.v4_dir / "validation_predictions.parquet",
                          columns=["MVT_ID_mvt", "target", "selected", "fold", "a_valid"])
    deps = load_validation_departures(files, oof)
    arr = load_arrivals(files)
    exact = exact_id_match(deps, arr)
    strict = strict_name_match(deps, arr)
    report = {"scope": "2025 local OOF labels only; raw covariate matching excludes departure BLOCK/TAXITIME",
              "arrival_rows": len(arr), "validation_departures": len(deps),
              "invalid_departures": int((~deps.a_valid).sum()),
              "exact_flight_id": audit_one(deps, exact, "exact_flight_id"),
              "strict_name_route_time_type": audit_one(deps, strict,
                                                          "strict_name_route_time_type")}
    # On rows with a known flight ID, check how often the strict name match
    # agrees with the exact flight ID. This is a precision check, not a feature.
    agreement = strict[["MVT_ID_mvt", "MVT_ID_mvt_arr"]].merge(
        exact[["MVT_ID_mvt", "MVT_ID_mvt_arr"]], on="MVT_ID_mvt",
        how="left", suffixes=("_name", "_id"), validate="one_to_one")
    both = agreement.MVT_ID_mvt_arr_name.notna() & agreement.MVT_ID_mvt_arr_id.notna()
    report["known_id_crosscheck"] = {
        "both_match_n": int(both.sum()),
        "same_arrival_n": int(agreement.loc[both, "MVT_ID_mvt_arr_name"].eq(
            agreement.loc[both, "MVT_ID_mvt_arr_id"]).sum()),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "recovery_audit.json").write_text(json.dumps(report, indent=2),
                                                           encoding="utf-8")
    return report


def _seconds(values: pd.Series) -> np.ndarray:
    """Convert pandas 3 datetime[us, UTC] safely to Unix seconds."""
    dt = pd.to_datetime(values, utc=True, errors="coerce")
    return dt.dt.as_unit("ns").astype("int64").to_numpy() // 1_000_000_000


def build_arrival_traffic(deps: pd.DataFrame, arr: pd.DataFrame) -> pd.DataFrame:
    """Past observed ARR taxi-in/stand occupancy at each DEP takeoff time."""
    n = len(deps)
    out = pd.DataFrame(index=deps.index)
    windows = (900, 3600, 10800)
    for window in windows:
        name = window // 60
        out[f"arr_taxi_count_{name}m"] = np.zeros(n, dtype=np.float32)
        out[f"arr_taxi_mean_{name}m"] = np.full(n, np.nan, dtype=np.float32)
        out[f"arr_taxi_highshare_{name}m"] = np.full(n, np.nan, dtype=np.float32)
    out["arr_taxiing_now"] = np.zeros(n, dtype=np.float32)
    out["arr_block_prev_60m"] = np.zeros(n, dtype=np.float32)
    out["arr_same_stand_last_block_gap"] = np.full(n, np.nan, dtype=np.float32)
    out["arr_same_stand_last_taxi"] = np.full(n, np.nan, dtype=np.float32)
    out["arr_same_stand_blocks_4h"] = np.zeros(n, dtype=np.float32)

    d_airport = deps.ADEP_mvt.astype("string").to_numpy()
    a_airport = arr.ADES_mvt.astype("string").to_numpy()
    d_time = _seconds(deps.MVT_TIME_UTC_mvt)
    a_time = _seconds(arr.MVT_TIME_UTC_mvt)
    a_block = _seconds(arr.BLOCK_TIME_UTC_mvt)
    taxi = pd.to_numeric(arr.TAXITIME_SEC_mvt, errors="coerce").to_numpy(dtype=float)
    good_taxi = np.isfinite(taxi) & (taxi >= 0) & (taxi <= 3600)
    for airport in pd.unique(d_airport):
        if pd.isna(airport):
            continue
        q = np.flatnonzero(d_airport == airport)
        ai = np.flatnonzero((a_airport == airport) & (a_time > 0))
        if not len(q) or not len(ai):
            continue
        order = np.argsort(a_time[ai], kind="stable")
        ai = ai[order]
        times = a_time[ai]
        tt = taxi[ai]
        valid = good_taxi[ai]
        count = np.r_[0, np.cumsum(valid, dtype=np.int64)]
        total = np.r_[0, np.cumsum(np.where(valid, tt, 0), dtype=np.float64)]
        high = np.r_[0, np.cumsum(valid & (tt >= 1200), dtype=np.int64)]
        end = np.searchsorted(times, d_time[q], side="left")
        for window in windows:
            name = window // 60
            start = np.searchsorted(times, d_time[q] - window, side="left")
            number = count[end] - count[start]
            out.iloc[q, out.columns.get_loc(f"arr_taxi_count_{name}m")] = number
            mean = np.divide(total[end] - total[start], number,
                             out=np.full(len(q), np.nan), where=number > 0)
            highshare = np.divide(high[end] - high[start], number,
                                  out=np.full(len(q), np.nan), where=number > 0)
            out.iloc[q, out.columns.get_loc(f"arr_taxi_mean_{name}m")] = mean.astype(np.float32)
            out.iloc[q, out.columns.get_loc(f"arr_taxi_highshare_{name}m")] = highshare.astype(np.float32)
        valid_block = (a_block[ai] >= times) & (a_block[ai] <= times + 7200)
        landings_with_block = times[valid_block]
        block = np.sort(a_block[ai][valid_block])
        inblock_end = np.searchsorted(block, d_time[q], side="left")
        inblock_start = np.searchsorted(block, d_time[q] - 3600, side="left")
        out.iloc[q, out.columns.get_loc("arr_block_prev_60m")] = (inblock_end - inblock_start)
        # Number of arrivals whose observed taxi-in overlaps the query time.
        taxiing = (np.searchsorted(landings_with_block, d_time[q], side="left")
                   - np.searchsorted(block, d_time[q], side="right"))
        out.iloc[q, out.columns.get_loc("arr_taxiing_now")] = np.maximum(taxiing, 0)

    d_stand = deps.STAND_mvt.astype("string").fillna("").to_numpy()
    a_stand = arr.STAND_mvt.astype("string").fillna("").to_numpy()
    dep_keys = pd.Series(d_airport, dtype="string") + "|" + pd.Series(d_stand, dtype="string")
    arr_keys = pd.Series(a_airport, dtype="string") + "|" + pd.Series(a_stand, dtype="string")
    arr_stand_groups = arr_keys.groupby(arr_keys, sort=False).indices
    for key, q in dep_keys.groupby(dep_keys, sort=False).indices.items():
        if key.endswith("|") or key not in arr_stand_groups:
            continue
        ai = np.asarray(arr_stand_groups[key], dtype=np.int64)
        ai = ai[(a_block[ai] > 0) & (a_block[ai] >= a_time[ai]) &
                (a_block[ai] <= a_time[ai] + 7200)]
        if not len(ai):
            continue
        ai = ai[np.argsort(a_block[ai], kind="stable")]
        blocks = a_block[ai]
        where = np.searchsorted(blocks, d_time[q], side="left")
        has_previous = where > 0
        last = np.maximum(where - 1, 0)
        gap = (d_time[q] - blocks[last]).astype(float)
        gap[~has_previous] = np.nan
        gap[gap > 86400] = np.nan
        recent = np.searchsorted(blocks, d_time[q] - 14400, side="left")
        out.iloc[q, out.columns.get_loc("arr_same_stand_last_block_gap")] = gap.astype(np.float32)
        last_taxi = taxi[ai[last]].astype(float)
        last_taxi[~has_previous] = np.nan
        last_taxi[~np.isfinite(gap)] = np.nan
        out.iloc[q, out.columns.get_loc("arr_same_stand_last_taxi")] = last_taxi.astype(np.float32)
        out.iloc[q, out.columns.get_loc("arr_same_stand_blocks_4h")] = where - recent
    return out


def _rmse(target: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean((target - prediction) ** 2)))


def _correction_features(deps: pd.DataFrame, arrivals: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    stamp = pd.to_datetime(deps.MVT_TIME_UTC_mvt, utc=True)
    proxy = (stamp - pd.to_datetime(deps.AOBT_3_flt, utc=True)).dt.total_seconds()
    control = pd.DataFrame(index=deps.index)
    control["airport"] = deps.ADEP_mvt.astype("string").fillna("?").astype("category")
    control["runway"] = deps.RUNWAY_mvt.astype("string").fillna("?").astype("category")
    control["stand"] = deps.STAND_mvt.astype("string").fillna("?").astype("category")
    control["hour"] = (stamp.dt.hour + stamp.dt.minute / 60).astype("float32")
    control["dayofweek"] = stamp.dt.dayofweek.astype("float32")
    control["proxy"] = proxy.astype("float32")
    control["v4_selected"] = deps.selected.astype("float32")
    augmented = pd.concat([control, arrivals], axis=1)
    return control, augmented


def _new_correction_model() -> LGBMRegressor:
    return LGBMRegressor(n_estimators=240, learning_rate=.04, num_leaves=15,
                         min_child_samples=500, reg_lambda=30,
                         colsample_bytree=.85, verbosity=-1, n_jobs=2,
                         random_state=123)


def _fit_correction(features: pd.DataFrame, resid: np.ndarray,
                    train: np.ndarray, test: np.ndarray) -> np.ndarray:
    model = _new_correction_model()
    model.fit(features.iloc[train], resid[train])
    return model.predict(features.iloc[test], num_threads=2)


def traffic_eval(args: argparse.Namespace) -> dict:
    files = _training_files(args.data_dir)
    oof = pd.read_parquet(args.v4_dir / "validation_predictions.parquet",
                          columns=["MVT_ID_mvt", "target", "selected", "fold", "a_valid"])
    deps = load_validation_departures(files, oof).reset_index(drop=True)
    arrivals = load_arrivals(files)
    arr_features = build_arrival_traffic(deps, arrivals)
    arr_features.insert(0, "MVT_ID_mvt", deps.MVT_ID_mvt.to_numpy())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    arr_features.to_parquet(args.output_dir / "oof_arrival_features.parquet", index=False)
    control, augmented = _correction_features(deps, arr_features.drop(columns="MVT_ID_mvt"))
    base = deps.selected.to_numpy(dtype=float)
    target = deps.target.to_numpy(dtype=float)
    residual = target - base
    valid = deps.a_valid.to_numpy(dtype=bool)
    month = pd.to_datetime(deps.MVT_TIME_UTC_mvt, utc=True).dt.month.to_numpy()
    fold = deps.fold.to_numpy()
    train_safe = valid & np.isfinite(target) & (target >= 0) & (target <= 86400)
    weights = (0, .1, .25, .5, 1.0)
    output = deps[["MVT_ID_mvt", "target", "selected", "fold", "a_valid"]].copy()
    report: dict = {"scope": "2025 local OOF; arrival rows only; no ranking labels",
                    "arrival_feature_names": arr_features.columns[1:].tolist(),
                    "features_nonnull_fraction": {
                        c: float(arr_features[c].notna().mean())
                        for c in arr_features.columns[1:]}, "models": {}}
    for label, features in (("control", control), ("arrival", augmented)):
        correction = np.full(len(deps), np.nan, dtype=float)
        for test_month, train_month in ((1, 7), (7, 1)):
            train_idx = np.flatnonzero(train_safe & (fold == "seasonal_jan_jul") & (month == train_month))
            test_idx = np.flatnonzero(valid & (fold == "seasonal_jan_jul") & (month == test_month))
            correction[test_idx] = _fit_correction(features, residual, train_idx, test_idx)
        train_idx = np.flatnonzero(train_safe & (fold == "seasonal_jan_jul"))
        test_idx = np.flatnonzero(valid & (fold == "forward_nov_dec"))
        correction[test_idx] = _fit_correction(features, residual, train_idx, test_idx)
        seasonal = valid & (fold == "seasonal_jan_jul")
        forward = valid & (fold == "forward_nov_dec")
        if not np.isfinite(correction[valid]).all():
            raise ValueError("Correction does not cover valid-AOBT OOF rows")
        seasonal_scores = {str(w): _rmse(target[seasonal], base[seasonal] + w * correction[seasonal])
                           for w in weights}
        chosen = min(weights, key=lambda w: (seasonal_scores[str(w)], w))
        forward_scores = {str(w): _rmse(target[forward], base[forward] + w * correction[forward])
                          for w in weights}
        all_scores = {}
        for name in ("seasonal_jan_jul", "forward_nov_dec"):
            mask = fold == name
            modified = base[mask].copy()
            use = valid[mask]
            modified[use] += chosen * correction[mask][use]
            all_scores[name] = {"n": int(mask.sum()),
                                "v4_rmse_sec": _rmse(target[mask], base[mask]),
                                "chosen_blend_rmse_sec": _rmse(target[mask], modified)}
        report["models"][label] = {"seasonal_valid_scores": seasonal_scores,
                                   "chosen_weight": chosen,
                                   "forward_valid_scores": forward_scores,
                                   "all_row_scores": all_scores}
        output[f"{label}_correction"] = correction
    output.to_parquet(args.output_dir / "traffic_correction_oof.parquet", index=False)
    (args.output_dir / "traffic_eval.json").write_text(json.dumps(report, indent=2),
                                                       encoding="utf-8")
    return report


def traffic_ablation(args: argparse.Namespace) -> dict:
    """Check which released ARR fields explain the held-out improvement."""
    files = _training_files(args.data_dir)
    oof = pd.read_parquet(args.v4_dir / "validation_predictions.parquet",
                          columns=["MVT_ID_mvt", "target", "selected", "fold", "a_valid"])
    deps = load_validation_departures(files, oof).reset_index(drop=True)
    saved = pd.read_parquet(args.output_dir / "oof_arrival_features.parquet")
    features = deps[["MVT_ID_mvt"]].merge(saved, on="MVT_ID_mvt", how="left",
                                           validate="one_to_one")
    if features["arr_taxi_count_15m"].isna().any() or len(features) != len(deps):
        raise ValueError("Saved arrival features do not align with OOF IDs")
    control, _ = _correction_features(deps, features.drop(columns="MVT_ID_mvt"))
    all_arrival = features.drop(columns="MVT_ID_mvt")
    taxi_columns = [c for c in all_arrival if ("mean_" in c or "highshare_" in c
                                                  or c == "arr_same_stand_last_taxi")]
    count_columns = [c for c in all_arrival if c not in taxi_columns]
    no_stand_columns = [c for c in all_arrival if "same_stand" not in c]
    variants = {"control": [], "counts_and_stand": count_columns,
                "taxi_values": taxi_columns, "no_stand": no_stand_columns,
                "all_arrival": all_arrival.columns.tolist()}
    base = deps.selected.to_numpy(dtype=float)
    target = deps.target.to_numpy(dtype=float)
    residual = target - base
    valid = deps.a_valid.to_numpy(dtype=bool)
    month = pd.to_datetime(deps.MVT_TIME_UTC_mvt, utc=True).dt.month.to_numpy()
    fold = deps.fold.to_numpy()
    train_safe = valid & np.isfinite(target) & (target >= 0) & (target <= 86400)
    report = {"scope": "Local 2025 OOF only; fixed model hyperparameters",
              "variants": {}}
    for name, columns in variants.items():
        frame = pd.concat([control, all_arrival[columns]], axis=1)
        correction = np.full(len(deps), np.nan, dtype=float)
        for test_month, train_month in ((1, 7), (7, 1)):
            tr = np.flatnonzero(train_safe & (fold == "seasonal_jan_jul") & (month == train_month))
            te = np.flatnonzero(valid & (fold == "seasonal_jan_jul") & (month == test_month))
            correction[te] = _fit_correction(frame, residual, tr, te)
        tr = np.flatnonzero(train_safe & (fold == "seasonal_jan_jul"))
        te = np.flatnonzero(valid & (fold == "forward_nov_dec"))
        correction[te] = _fit_correction(frame, residual, tr, te)
        result = {"columns": columns, "fixed_half_blend_allrow": {}}
        for split in ("seasonal_jan_jul", "forward_nov_dec"):
            mask = fold == split
            pred = base[mask].copy()
            use = valid[mask]
            pred[use] += .5 * correction[mask][use]
            pred = np.maximum(pred, 0)
            result["fixed_half_blend_allrow"][split] = _rmse(target[mask], pred)
        report["variants"][name] = result
    (args.output_dir / "traffic_ablation.json").write_text(json.dumps(report, indent=2),
                                                           encoding="utf-8")
    return report


def final_predict(args: argparse.Namespace) -> dict:
    """Fit the frozen ARR correction on local OOF residuals and predict ranking."""
    files = _training_files(args.data_dir)
    oof = pd.read_parquet(args.v4_dir / "validation_predictions.parquet",
                          columns=["MVT_ID_mvt", "target", "selected", "fold", "a_valid"])
    deps = load_validation_departures(files, oof).reset_index(drop=True)
    saved = pd.read_parquet(args.output_dir / "oof_arrival_features.parquet")
    train_arr = deps[["MVT_ID_mvt"]].merge(saved, on="MVT_ID_mvt", how="left",
                                           validate="one_to_one")
    if (len(train_arr) != len(deps) or train_arr["arr_taxi_count_15m"].isna().any()
            or saved.MVT_ID_mvt.duplicated().any()):
        raise ValueError("Saved OOF ARR features are missing or misaligned")
    _, train_x = _correction_features(deps, train_arr.drop(columns="MVT_ID_mvt"))
    target = deps.target.to_numpy(dtype=float)
    base = deps.selected.to_numpy(dtype=float)
    eligible = (deps.a_valid.to_numpy(dtype=bool) & np.isfinite(target)
                & (target >= 0) & (target <= 86400))
    model = _new_correction_model()
    model.fit(train_x.loc[eligible], (target - base)[eligible])

    ranking_file = args.data_dir / "ranking.parquet"
    rank_dep = (pl.scan_parquet(str(ranking_file))
                .filter(pl.col("PHASE_mvt") == "DEP")
                .select(DEP_COLS).collect().to_pandas())
    frozen = pd.read_parquet(args.v4_ranking,
                             columns=["MVT_ID_mvt", "TAXITIME_SEC_mvt"])
    if frozen.MVT_ID_mvt.duplicated().any() or rank_dep.MVT_ID_mvt.duplicated().any():
        raise ValueError("Ranking ID uniqueness failed")
    rank = frozen.merge(rank_dep, on="MVT_ID_mvt", how="left", sort=False,
                        validate="one_to_one")
    if len(rank) != len(frozen) or rank.MVT_TIME_UTC_mvt.isna().any():
        raise ValueError("Ranking DEP covariates do not align with frozen v4")
    rank["selected"] = rank.TAXITIME_SEC_mvt
    rank_arrivals = load_arrivals([ranking_file])
    rank_arr = build_arrival_traffic(rank, rank_arrivals)
    _, rank_x = _correction_features(rank, rank_arr)
    if list(train_x.columns) != list(rank_x.columns):
        raise ValueError("Training and ranking ARR feature schema differ")
    proxy = (pd.to_datetime(rank.MVT_TIME_UTC_mvt, utc=True)
             - pd.to_datetime(rank.AOBT_3_flt, utc=True)).dt.total_seconds().to_numpy()
    valid = np.isfinite(proxy) & (proxy >= 0) & (proxy <= 7200)
    correction = np.full(len(rank), np.nan, dtype=float)
    correction[valid] = model.predict(rank_x.iloc[np.flatnonzero(valid)], num_threads=2)
    prediction = rank.TAXITIME_SEC_mvt.to_numpy(dtype=float).copy()
    prediction[valid] += .5 * correction[valid]
    prediction = np.maximum(prediction, 0)
    if not np.isfinite(prediction).all():
        raise ValueError("ARR ranking prediction has nonfinite values")
    template = pd.read_parquet(args.data_dir / "submitting.parquet",
                               columns=["MVT_ID_mvt"])
    if not np.array_equal(template.MVT_ID_mvt.to_numpy(), rank.MVT_ID_mvt.to_numpy()):
        raise ValueError("ARR ranking IDs differ from submission template order")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(args.output_dir / "final_correction_lgbm.txt"))
    pd.DataFrame({"MVT_ID_mvt": rank.MVT_ID_mvt,
                  "arrival_correction": correction}).to_parquet(
                      args.output_dir / "ranking_raw_correction.parquet", index=False)
    pd.DataFrame({"MVT_ID_mvt": rank.MVT_ID_mvt,
                  "TAXITIME_SEC_mvt": prediction}).to_parquet(
                      args.output_dir / "ranking_proposal.parquet", index=False)
    rank_arr.insert(0, "MVT_ID_mvt", rank.MVT_ID_mvt.to_numpy())
    rank_arr.to_parquet(args.output_dir / "ranking_arrival_features.parquet", index=False)
    report = {"training_oof_rows": len(deps), "training_eligible_rows": int(eligible.sum()),
              "ranking_rows": len(rank), "ranking_valid_aobt_rows": int(valid.sum()),
              "selected_blend": .5, "frozen_v4_ranking": str(args.v4_ranking),
              "model_features": train_x.columns.tolist(),
              "meta_validation_limit": (
                  "The Jan/Jul frozen v4 OOF base experts trained on other months, "
                  "including Nov/Dec. A stack fit to Jan/Jul OOF residuals and checked "
                  "on Nov/Dec is a model-comparison estimate, not a fully independent "
                  "forward end-to-end validation."),
              "uses_only_arrival_ground_truth": True,
              "reads_ranking_departure_labels": False}
    (args.output_dir / "final_report.json").write_text(json.dumps(report, indent=2),
                                                         encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--v4-dir", type=Path, default=Path("artifacts/v4"))
    parser.add_argument("--v4-ranking", type=Path,
                        default=Path("artifacts/catboost/source/sequential_predictions.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/v5-arrival"))
    parser.add_argument("--mode", choices=("recovery-audit", "traffic-eval",
                                          "traffic-ablation", "final-predict"),
                        default="recovery-audit")
    args = parser.parse_args()
    action = {"recovery-audit": recovery_audit, "traffic-eval": traffic_eval,
              "traffic-ablation": traffic_ablation,
              "final-predict": final_predict}[args.mode]
    print(json.dumps(action(args), indent=2))


if __name__ == "__main__":
    main()
