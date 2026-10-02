"""Original PRC Data Challenge 2026 taxi-out modeling pipeline.

The ranking file exposes actual takeoff and NM actual off-block times. This
pipeline evaluates that proxy on held-out months, learns its discrepancy from
airport off-block records, and falls back to a direct model when unavailable.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import re
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl


TIME_COLS = (
    "MVT_TIME_UTC_mvt",
    "SCHED_TIME_UTC_mvt",
    "AOBT_3_flt",
    "LOBT_flt",
    "IOBT_flt",
    "EOBT_1_flt",
    "ARVT_1_flt",
    "ARVT_3_flt",
)
CATEGORY_COLS = (
    "ADEP_mvt",
    "ADES_mvt",
    "ADEP_flt",
    "ADES_flt",
    "ADES_FILED_flt",
    "RUNWAY_mvt",
    "STAND_mvt",
    "AIRCRAFT_TYPE_mvt",
    "AIRCRAFT_TYPE_flt",
    "AIRCRAFT_OPERATOR_flt",
    "MARKET_SEGMENT_flt",
    "WK_TBL_CAT_flt",
    "FLIGHT_RULE_mvt",
    "FLIGHT_RULE_flt",
    "FLIGHT_TYPE_flt",
)
RAW_COLS = (
    "MVT_ID_mvt",
    "FLIGHT_ID_mvt",
    "PHASE_mvt",
    "FLIGHT_mvt",
    "CALLSIGN_flt",
    "TAXITIME_SEC_mvt",
    *TIME_COLS,
    *CATEGORY_COLS,
)
REQUIRED = {"MVT_ID_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt", "ADEP_mvt", "ADES_mvt"}
AIRPORT_TZ = {
    "EDDF": "Europe/Berlin",
    "EDDM": "Europe/Berlin",
    "EGLL": "Europe/London",
    "EHAM": "Europe/Amsterdam",
    "LEBL": "Europe/Madrid",
    "LEMD": "Europe/Madrid",
    "LFPG": "Europe/Paris",
    "LIRF": "Europe/Rome",
    "LTFM": "Europe/Istanbul",
    "LSZH": "Europe/Zurich",
}
MAX_PROXY_SEC = 7200.0


def _schema_columns(paths: list[Path]) -> set[str]:
    return set(pl.scan_parquet([str(p) for p in paths]).collect_schema().names())


def load_data(paths: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read departure covariates and a narrow all-movements traffic table."""
    if not paths:
        raise FileNotFoundError("No Parquet input files were supplied")
    columns = _schema_columns(paths)
    missing = REQUIRED - columns
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    scan = pl.scan_parquet([str(p) for p in paths])
    dep_cols = [c for c in RAW_COLS if c in columns]
    deps = scan.filter(pl.col("PHASE_mvt") == "DEP").select(dep_cols).collect().to_pandas()
    traffic_cols = [c for c in ("PHASE_mvt", "ADEP_mvt", "ADES_mvt", "RUNWAY_mvt", "MVT_TIME_UTC_mvt", "AOBT_3_flt") if c in columns]
    traffic = scan.select(traffic_cols).collect().to_pandas()
    for frame in (deps, traffic):
        frame["MVT_TIME_UTC_mvt"] = pd.to_datetime(frame["MVT_TIME_UTC_mvt"], utc=True, errors="coerce")
    for col in TIME_COLS:
        if col in deps and col != "MVT_TIME_UTC_mvt":
            deps[col] = pd.to_datetime(deps[col], utc=True, errors="coerce")
    if "AOBT_3_flt" in traffic:
        traffic["AOBT_3_flt"] = pd.to_datetime(traffic["AOBT_3_flt"], utc=True, errors="coerce")
    # Avoid retaining millions of repeated Python strings while fitting.
    for col in (*CATEGORY_COLS, "FLIGHT_mvt", "CALLSIGN_flt"):
        if col in deps:
            deps[col] = deps[col].astype("category")
    for col in ("PHASE_mvt", "ADEP_mvt", "ADES_mvt", "RUNWAY_mvt"):
        if col in traffic:
            traffic[col] = traffic[col].astype("category")
    if deps["MVT_ID_mvt"].isna().any() or deps["MVT_ID_mvt"].duplicated().any():
        raise ValueError("Departure MVT_ID_mvt values must be unique and non-null")
    return deps, traffic


def _unix_seconds(ts: pd.Series) -> np.ndarray:
    return ts.dt.as_unit("ns").astype("int64").to_numpy() // 1_000_000_000


def _count_windows(query: np.ndarray, events: np.ndarray, before: int, after: int) -> np.ndarray:
    if not len(events):
        return np.zeros(len(query), dtype=np.int32)
    events = np.sort(events)
    return (np.searchsorted(events, query + after, side="right") -
            np.searchsorted(events, query - before, side="left")).astype(np.int32)


def add_traffic_features(deps: pd.DataFrame, traffic: pd.DataFrame) -> pd.DataFrame:
    """Counts use only fields retained in ranking, including observed AOBT."""
    n = len(deps)
    out = pd.DataFrame(index=deps.index)
    specs = [("dep_prev5", "DEP", 300, 0),
             ("dep_prev15", "DEP", 900, 0),
             ("dep_prev30", "DEP", 1800, 0),
             ("dep_next15", "DEP", 0, 900),
             ("arr_prev15", "ARR", 900, 0),
             ("arr_prev30", "ARR", 1800, 0),
             ("arr_next15", "ARR", 0, 900)]
    for name, _, _, _ in specs:
        out[name] = np.zeros(n, dtype=np.float32)
    out["same_runway_prev15"] = np.zeros(n, dtype=np.float32)
    out["taxi_backlog_at_aobt"] = np.full(n, np.nan, dtype=np.float32)
    out["same_runway_backlog_at_aobt"] = np.full(n, np.nan, dtype=np.float32)

    traffic_airport = np.where(traffic["PHASE_mvt"].eq("DEP"),
                               traffic["ADEP_mvt"], traffic["ADES_mvt"])
    traffic_t = _unix_seconds(traffic["MVT_TIME_UTC_mvt"])
    traffic_start = (_unix_seconds(traffic["AOBT_3_flt"])
                     if "AOBT_3_flt" in traffic else np.full(len(traffic), np.iinfo(np.int64).min))
    dep_t = _unix_seconds(deps["MVT_TIME_UTC_mvt"])
    dep_start = (_unix_seconds(deps["AOBT_3_flt"])
                 if "AOBT_3_flt" in deps else np.full(n, np.iinfo(np.int64).min))
    dep_airport = deps["ADEP_mvt"].astype("string").fillna("__MISSING__").to_numpy()
    phases = traffic["PHASE_mvt"].to_numpy()
    t_runway = traffic["RUNWAY_mvt"].astype("string").fillna("__MISSING__").to_numpy() if "RUNWAY_mvt" in traffic else None
    d_runway = deps["RUNWAY_mvt"].astype("string").fillna("__MISSING__").to_numpy() if "RUNWAY_mvt" in deps else None
    valid_t = traffic["MVT_TIME_UTC_mvt"].notna().to_numpy()

    for airport in pd.unique(dep_airport):
        qidx = np.flatnonzero(dep_airport == airport)
        q = dep_t[qidx]
        for phase in ("DEP", "ARR"):
            emask = (traffic_airport == airport) & (phases == phase) & valid_t
            events = traffic_t[emask]
            for name, want_phase, before, after in specs:
                if want_phase == phase:
                    counts = _count_windows(q, events, before, after)
                    if phase == "DEP":
                        counts -= ((q + after >= q) & (q - before <= q)).astype(np.int32)
                    out.loc[qidx, name] = np.maximum(counts, 0)

        if d_runway is not None and t_runway is not None:
            for runway in pd.unique(d_runway[qidx]):
                sub = qidx[d_runway[qidx] == runway]
                emask = (traffic_airport == airport) & (phases == "DEP") & valid_t & (t_runway == runway)
                counts = _count_windows(dep_t[sub], traffic_t[emask], 900, 0) - 1
                out.loc[sub, "same_runway_prev15"] = np.maximum(counts, 0)

        # Interval count at each candidate AOBT: starts <= query < takeoff.
        # Exclude the querying flight itself, which lies in its own interval.
        valid_interval = ((traffic_airport == airport) & (phases == "DEP") & valid_t &
                          (traffic_start > 0) & (traffic_start < traffic_t) &
                          ((traffic_t - traffic_start) <= MAX_PROXY_SEC))
        starts = np.sort(traffic_start[valid_interval])
        ends = np.sort(traffic_t[valid_interval])
        valid_query = ((dep_start[qidx] > 0) & (dep_start[qidx] < dep_t[qidx]) &
                       ((dep_t[qidx] - dep_start[qidx]) <= MAX_PROXY_SEC))
        sub = qidx[valid_query]
        if len(sub):
            counts = (np.searchsorted(starts, dep_start[sub], side="right") -
                      np.searchsorted(ends, dep_start[sub], side="right") - 1)
            out.loc[sub, "taxi_backlog_at_aobt"] = np.maximum(counts, 0)
        if d_runway is not None and t_runway is not None:
            for runway in pd.unique(d_runway[qidx]):
                sub = qidx[(d_runway[qidx] == runway) & valid_query]
                if not len(sub):
                    continue
                rmask = valid_interval & (t_runway == runway)
                starts = np.sort(traffic_start[rmask])
                ends = np.sort(traffic_t[rmask])
                counts = (np.searchsorted(starts, dep_start[sub], side="right") -
                          np.searchsorted(ends, dep_start[sub], side="right") - 1)
                out.loc[sub, "same_runway_backlog_at_aobt"] = np.maximum(counts, 0)
    return out


def _seconds(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a - b).dt.total_seconds().astype("float32").clip(-86_400, 86_400)


def build_features(deps: pd.DataFrame, traffic: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    x = add_traffic_features(deps, traffic)
    n = len(deps)
    t = deps["MVT_TIME_UTC_mvt"]
    missing_time = pd.Series(pd.NaT, index=deps.index, dtype="datetime64[ns, UTC]")
    for col in TIME_COLS:
        if col == "MVT_TIME_UTC_mvt":
            continue
        x[f"takeoff_minus_{col}"] = _seconds(t, deps[col] if col in deps else missing_time)
    for a, b in (("AOBT_3_flt", "LOBT_flt"),
                 ("AOBT_3_flt", "IOBT_flt"),
                 ("AOBT_3_flt", "EOBT_1_flt"),
                 ("LOBT_flt", "EOBT_1_flt"),
                 ("SCHED_TIME_UTC_mvt", "LOBT_flt")):
        x[f"delta_{a}_{b}"] = _seconds(deps[a] if a in deps else missing_time,
                                           deps[b] if b in deps else missing_time)
    raw_proxy = _seconds(t, deps["AOBT_3_flt"] if "AOBT_3_flt" in deps else missing_time).to_numpy()
    proxy = np.where(np.isfinite(raw_proxy) & (raw_proxy >= 0) &
                     (raw_proxy <= MAX_PROXY_SEC), raw_proxy, np.nan).astype(np.float32)
    x["proxy_valid"] = np.isfinite(proxy).astype(np.int8)
    x["proxy_seconds"] = proxy
    x["aobt_lobt_abs_difference"] = x["delta_AOBT_3_flt_LOBT_flt"].abs()
    for col in ("AOBT_3_flt", "LOBT_flt", "EOBT_1_flt", "IOBT_flt"):
        x[f"{col}_missing"] = (deps[col].isna() if col in deps else pd.Series(True, index=deps.index)).astype(np.int8)
    def agrees(left: str, right: str) -> pd.Series:
        a = deps[left].astype("string") if left in deps else pd.Series(pd.NA, index=deps.index, dtype="string")
        b = deps[right].astype("string") if right in deps else pd.Series(pd.NA, index=deps.index, dtype="string")
        return a.eq(b).fillna(False).astype(np.int8)

    x["flight_airport_agrees"] = agrees("ADEP_flt", "ADEP_mvt")
    x["aircraft_type_agrees"] = agrees("AIRCRAFT_TYPE_flt", "AIRCRAFT_TYPE_mvt")
    x["destination_agrees"] = agrees("ADES_flt", "ADES_mvt")

    x["utc_hour"] = (t.dt.hour + t.dt.minute / 60).astype("float32")
    x["utc_weekday"] = t.dt.dayofweek.astype("float32")
    x["utc_month"] = t.dt.month.astype("float32")
    x["utc_dayofyear"] = t.dt.dayofyear.astype("float32")
    x["is_weekend"] = t.dt.dayofweek.isin((5, 6)).astype(np.int8)
    local_hour = np.full(n, np.nan, dtype=np.float32)
    local_weekday = np.full(n, np.nan, dtype=np.float32)
    for airport, tz in AIRPORT_TZ.items():
        idx = np.flatnonzero(deps["ADEP_mvt"].eq(airport).to_numpy())
        if len(idx):
            local = t.iloc[idx].dt.tz_convert(tz)
            local_hour[idx] = local.dt.hour.to_numpy() + local.dt.minute.to_numpy() / 60
            local_weekday[idx] = local.dt.dayofweek.to_numpy()
    x["local_hour"] = local_hour
    x["local_weekday"] = local_weekday
    x["hour_sin"] = np.sin(2 * np.pi * local_hour / 24).astype(np.float32)
    x["hour_cos"] = np.cos(2 * np.pi * local_hour / 24).astype(np.float32)
    x["year_sin"] = np.sin(2 * np.pi * x["utc_dayofyear"] / 365.25).astype(np.float32)
    x["year_cos"] = np.cos(2 * np.pi * x["utc_dayofyear"] / 365.25).astype(np.float32)

    for col in CATEGORY_COLS:
        if col in deps:
            x[col] = deps[col].astype("string").fillna("__MISSING__").astype("category")
    flight = deps["FLIGHT_mvt"].astype("string") if "FLIGHT_mvt" in deps else pd.Series("", index=deps.index, dtype="string")
    callsign = deps["CALLSIGN_flt"].astype("string") if "CALLSIGN_flt" in deps else pd.Series("", index=deps.index, dtype="string")
    x["flight_prefix"] = flight.str.extract(r"^([A-Za-z]{1,3})", expand=False).fillna("__MISSING__").astype("category")
    x["callsign_prefix"] = callsign.str.extract(r"^([A-Za-z]{1,3})", expand=False).fillna("__MISSING__").astype("category")
    x["flight_number"] = pd.to_numeric(flight.str.extract(r"(\d{1,4})", expand=False), errors="coerce").astype("float32")
    stand = deps["STAND_mvt"].astype("string") if "STAND_mvt" in deps else pd.Series("", index=deps.index, dtype="string")
    x["stand_zone"] = stand.str.extract(r"^([A-Za-z]{1,3})", expand=False).fillna("__MISSING__").astype("category")
    x["stand_number"] = pd.to_numeric(stand.str.extract(r"(\d{1,4})", expand=False), errors="coerce").astype("float32")
    x["route"] = (deps["ADEP_mvt"].astype("string").fillna("?") + "_" +
                  deps.get("ADES_mvt", pd.Series("?", index=deps.index)).astype("string").fillna("?")).astype("category")
    airport = deps["ADEP_mvt"].astype("string").fillna("?")
    runway = deps["RUNWAY_mvt"].astype("string").fillna("?") if "RUNWAY_mvt" in deps else pd.Series("?", index=deps.index)
    stand_text = stand.fillna("?")
    x["airport_runway"] = (airport + "_" + runway).astype("category")
    x["airport_stand"] = (airport + "_" + stand_text).astype("category")
    x["airport_stand_runway"] = (airport + "_" + stand_text + "_" + runway).astype("category")
    for col in x.columns:
        if pd.api.types.is_numeric_dtype(x[col]) and x[col].dtype != np.int8:
            x[col] = x[col].astype(np.float32)
    return x, proxy


def _params(threads: int) -> dict:
    return dict(objective="regression", metric="rmse", learning_rate=0.045,
                num_leaves=63, min_data_in_leaf=90, feature_fraction=0.85,
                bagging_fraction=0.85, bagging_freq=1, lambda_l2=8.0,
                max_cat_threshold=64, cat_smooth=20, verbosity=-1,
                num_threads=threads, seed=2026, feature_fraction_seed=2026,
                bagging_seed=2026, deterministic=True, force_col_wise=True)


def _fit(x_train: pd.DataFrame, y_train: np.ndarray, x_valid: pd.DataFrame | None,
         y_valid: np.ndarray | None, threads: int, rounds: int) -> lgb.Booster:
    categorical = [c for c in x_train if isinstance(x_train[c].dtype, pd.CategoricalDtype)]
    train_set = lgb.Dataset(x_train, label=y_train, categorical_feature=categorical, free_raw_data=True)
    if x_valid is None:
        return lgb.train(_params(threads), train_set, num_boost_round=rounds)
    valid_set = lgb.Dataset(x_valid, label=y_valid, reference=train_set, free_raw_data=True)
    return lgb.train(_params(threads), train_set, num_boost_round=rounds,
                     valid_sets=[valid_set], callbacks=[lgb.early_stopping(100, verbose=False)])


def _rmse(y: np.ndarray, pred: np.ndarray) -> float:
    mask = np.isfinite(y) & np.isfinite(pred)
    return float(np.sqrt(np.mean((y[mask] - pred[mask]) ** 2))) if mask.any() else math.nan


def _scores(y: np.ndarray, pred: np.ndarray, airport: np.ndarray, valid_proxy: np.ndarray) -> dict:
    result = {"overall": _rmse(y, pred), "n": int(len(y)),
              "proxy_available": _rmse(y[valid_proxy], pred[valid_proxy]),
              "proxy_missing": _rmse(y[~valid_proxy], pred[~valid_proxy])}
    result["by_airport"] = {str(a): _rmse(y[airport == a], pred[airport == a])
                             for a in sorted(pd.unique(airport))}
    return result


def _training_files(data_dir: Path) -> list[Path]:
    paths = sorted(p for p in data_dir.glob("training_2025-*.parquet")
                   if re.fullmatch(r"training_2025-\d{2}-01_20\d{2}-\d{2}-01\.parquet", p.name))
    if not paths:
        raise FileNotFoundError(f"No training_2025-*.parquet files in {data_dir}")
    return paths


def audit(data_dir: Path) -> None:
    deps, traffic = load_data(_training_files(data_dir))
    y = pd.to_numeric(deps["TAXITIME_SEC_mvt"], errors="coerce").to_numpy(dtype=np.float64)
    if "AOBT_3_flt" in deps:
        proxy = _seconds(deps["MVT_TIME_UTC_mvt"], deps["AOBT_3_flt"]).to_numpy()
        mask = np.isfinite(proxy) & (proxy >= 0) & (proxy <= MAX_PROXY_SEC) & np.isfinite(y)
    else:
        proxy = np.full(len(deps), np.nan)
        mask = np.zeros(len(deps), dtype=bool)
    result = {"training_files": len(_training_files(data_dir)), "departures": len(deps),
              "all_movements": len(traffic), "labeled_departures": int(np.isfinite(y).sum()),
              "proxy_coverage": float(mask.mean()), "proxy_rmse_sec": _rmse(y[mask], proxy[mask]),
              "proxy_bias_sec": float(np.mean(proxy[mask] - y[mask])) if mask.any() else math.nan,
              "target_quantiles_sec": {str(q): float(np.nanquantile(y, q)) for q in (0.01, 0.5, 0.99)}}
    print(json.dumps(result, indent=2))


def train(data_dir: Path, output_dir: Path, threads: int, rounds: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = _training_files(data_dir)
    if len(paths) != 12:
        raise ValueError(f"Expected all 12 monthly training files, found {len(paths)}")
    deps, traffic = load_data(paths)
    if "TAXITIME_SEC_mvt" not in deps:
        raise ValueError("Training data has no TAXITIME_SEC_mvt")
    x, proxy = build_features(deps, traffic)
    del traffic
    gc.collect()
    y = pd.to_numeric(deps["TAXITIME_SEC_mvt"], errors="coerce").to_numpy(dtype=np.float32)
    month = deps["MVT_TIME_UTC_mvt"].dt.month.to_numpy()
    airport = deps["ADEP_mvt"].astype("string").fillna("__MISSING__").to_numpy()
    rows = pd.DataFrame({"MVT_ID_mvt": deps["MVT_ID_mvt"], "target": y, "proxy": proxy,
                         "month": month, "airport": airport, "time": deps["MVT_TIME_UTC_mvt"]})
    x.to_parquet(output_dir / "features.parquet", index=False)
    rows.to_parquet(output_dir / "training_rows.parquet", index=False)
    del deps
    gc.collect()
    labeled = np.isfinite(y)
    # Core trees model ordinary operations. Separate specialists retain and
    # predict the exceptional overnight labels; every finite label is scored.
    core_labeled = labeled & (y >= 0) & (y <= 86_400)
    folds = {"seasonal_jan_jul": np.isin(month, (1, 7)),
             "forward_nov_dec": np.isin(month, (11, 12))}
    fold_results = {}
    pooled = []
    direct_rounds = []
    residual_rounds = []
    for name, valid_month in folds.items():
        train_idx = core_labeled & ~valid_month
        valid_idx = labeled & valid_month
        early_idx = core_labeled & valid_month
        if train_idx.sum() < 1000 or valid_idx.sum() < 100:
            print(f"Skipping {name}: too few labeled records")
            continue
        direct = _fit(x.loc[train_idx], y[train_idx], x.loc[early_idx], y[early_idx], threads, rounds)
        direct.save_model(str(output_dir / f"{name}_direct.txt"))
        d = direct.predict(x.loc[valid_idx], num_threads=threads).astype(np.float32)
        direct_rounds.append(direct.best_iteration or rounds)
        proxy_train = train_idx & np.isfinite(proxy)
        proxy_valid = valid_idx & np.isfinite(proxy)
        proxy_early = early_idx & np.isfinite(proxy)
        hybrid = d.copy()
        residual = None
        if proxy_train.sum() >= 1000 and proxy_valid.sum() >= 100:
            residual = _fit(x.loc[proxy_train], y[proxy_train] - proxy[proxy_train],
                            x.loc[proxy_early], y[proxy_early] - proxy[proxy_early], threads, rounds)
            residual.save_model(str(output_dir / f"{name}_residual.txt"))
            valid_positions = np.flatnonzero(np.isfinite(proxy[valid_idx]))
            hybrid[valid_positions] = (proxy[proxy_valid] + residual.predict(
                x.loc[proxy_valid], num_threads=threads)).astype(np.float32)
            residual_rounds.append(residual.best_iteration or rounds)
        yi = y[valid_idx]
        pi = proxy[valid_idx]
        baseline = np.where(np.isfinite(pi), pi, d)
        oof = rows.loc[valid_idx].copy()
        oof["row_index"] = np.flatnonzero(valid_idx)
        oof["direct"] = d
        oof["raw_proxy_fallback"] = baseline
        oof["hybrid"] = hybrid
        oof.to_parquet(output_dir / f"{name}_oof.parquet", index=False)
        fold_results[name] = {"direct": _scores(yi, d, airport[valid_idx], np.isfinite(pi)),
                              "proxy_with_direct_fallback": _scores(yi, baseline, airport[valid_idx], np.isfinite(pi)),
                              "residual_with_direct_fallback": _scores(yi, hybrid, airport[valid_idx], np.isfinite(pi)),
                              "direct_best_round": int(direct.best_iteration or rounds),
                              "residual_best_round": int(residual.best_iteration or rounds) if residual else None}
        pooled.append((yi, d, baseline, hybrid, airport[valid_idx], np.isfinite(pi)))
        print(json.dumps({"fold": name, **fold_results[name]}, indent=2))
        del direct, residual, d, hybrid
        gc.collect()
    if not pooled:
        raise ValueError("No validation fold had sufficient data")
    yy = np.concatenate([p[0] for p in pooled])
    dd = np.concatenate([p[1] for p in pooled])
    bb = np.concatenate([p[2] for p in pooled])
    hh = np.concatenate([p[3] for p in pooled])
    aa = np.concatenate([p[4] for p in pooled])
    pp = np.concatenate([p[5] for p in pooled])
    # Select the proxy expert between uncorrected and modeled on held-out rows,
    # then shrink it toward the direct prediction. This includes the raw proxy.
    best = (float("inf"), 0.0, 0.0)
    for residual_share in (0.0, 0.25, 0.5, 0.75, 1.0):
        expert = (1 - residual_share) * bb + residual_share * hh
        diff = expert - dd
        denom = float(np.dot(diff.astype(np.float64), diff.astype(np.float64)))
        alpha = float(np.clip(np.dot((yy - dd).astype(np.float64), diff.astype(np.float64)) /
                              denom, 0, 1)) if denom > 0 else 0.0
        candidate = (1 - alpha) * dd + alpha * expert
        score = _rmse(yy, candidate)
        if score < best[0]:
            best = (score, alpha, residual_share)
    _, alpha, residual_share = best
    expert = (1 - residual_share) * bb + residual_share * hh
    blended = (1 - alpha) * dd + alpha * expert
    report = {"folds": fold_results, "blend_alpha": alpha,
              "residual_share": residual_share,
              "pooled": _scores(yy, blended, aa, pp),
              "pooled_direct_rmse": _rmse(yy, dd),
              "pooled_proxy_rmse": _rmse(yy, bb),
              "pooled_hybrid_rmse": _rmse(yy, hh),
              "proxy_coverage": float(np.isfinite(proxy[labeled]).mean())}
    (output_dir / "validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"selected_alpha": alpha, "residual_share": residual_share,
                      "pooled_rmse": report["pooled"]["overall"],
                      "direct_rmse": report["pooled_direct_rmse"],
                      "proxy_rmse": report["pooled_proxy_rmse"],
                      "hybrid_rmse": report["pooled_hybrid_rmse"]}, indent=2))

    final_direct_rounds = int(np.median(direct_rounds))
    direct = _fit(x.loc[core_labeled], y[core_labeled], None, None, threads, final_direct_rounds)
    direct.save_model(str(output_dir / "direct.txt"))
    proxy_labeled = core_labeled & np.isfinite(proxy)
    final_residual_rounds = int(np.median(residual_rounds)) if residual_rounds else None
    if final_residual_rounds and proxy_labeled.sum() >= 1000 and residual_share > 0:
        residual = _fit(x.loc[proxy_labeled], y[proxy_labeled] - proxy[proxy_labeled],
                        None, None, threads, final_residual_rounds)
        residual.save_model(str(output_dir / "residual.txt"))
    else:
        residual_share = 0.0
    metadata = {"features": list(x.columns), "blend_alpha": alpha,
                "residual_share": residual_share,
                "direct_rounds": final_direct_rounds, "residual_rounds": final_residual_rounds,
                "training_rows": int(core_labeled.sum()), "proxy_training_rows": int(proxy_labeled.sum()),
                "core_label_range_sec": [0, 86400], "validation_includes_all_finite_labels": True,
                "max_proxy_sec": MAX_PROXY_SEC}
    (output_dir / "model.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def predict(data_dir: Path, model_dir: Path, out: Path, threads: int) -> None:
    ranking = data_dir / "ranking.parquet"
    template = data_dir / "submitting.parquet"
    if not ranking.exists() or not template.exists():
        raise FileNotFoundError("ranking.parquet and submitting.parquet must be in the data directory")
    metadata = json.loads((model_dir / "model.json").read_text(encoding="utf-8"))
    deps, traffic = load_data([ranking])
    x, proxy = build_features(deps, traffic)
    dep_ids = deps["MVT_ID_mvt"].to_numpy(copy=True)
    x.to_parquet(model_dir / "ranking_features.parquet", index=False)
    ranking_rows = pd.DataFrame({"MVT_ID_mvt": dep_ids, "proxy": proxy,
                                 "airport": deps["ADEP_mvt"].astype("string"),
                                 "month": deps["MVT_TIME_UTC_mvt"].dt.month,
                                 "time": deps["MVT_TIME_UTC_mvt"]})
    ranking_rows.to_parquet(model_dir / "ranking_rows.parquet", index=False)
    del deps, traffic
    gc.collect()
    if set(metadata["features"]) != set(x.columns):
        raise ValueError("Ranking feature schema differs from training. Check input files and columns.")
    x = x[metadata["features"]]
    direct = lgb.Booster(model_file=str(model_dir / "direct.txt"))
    pred_direct = direct.predict(x, num_threads=threads)
    pred = pred_direct.copy()
    hybrid = pred_direct.copy()
    valid = np.isfinite(proxy)
    if metadata["blend_alpha"] > 0 and valid.any():
        proxy_expert = proxy[valid].astype(np.float64)
        if metadata["residual_share"] > 0:
            residual_file = model_dir / "residual.txt"
            if not residual_file.exists():
                raise FileNotFoundError(f"Missing residual model: {residual_file}")
            residual = lgb.Booster(model_file=str(residual_file))
            corrected = proxy_expert + residual.predict(x.loc[valid], num_threads=threads)
            hybrid[valid] = corrected
            proxy_expert = ((1 - metadata["residual_share"]) * proxy_expert +
                            metadata["residual_share"] * corrected)
        pred[valid] = ((1 - metadata["blend_alpha"]) * pred_direct[valid] +
                       metadata["blend_alpha"] * proxy_expert)
    pred = np.maximum(pred, 0)
    expert_predictions = ranking_rows.copy()
    expert_predictions["direct"] = pred_direct
    expert_predictions["raw_proxy_fallback"] = np.where(valid, proxy, pred_direct)
    expert_predictions["hybrid"] = hybrid
    expert_predictions["selected"] = pred
    expert_predictions.to_parquet(model_dir / "ranking_predictions.parquet", index=False)
    if not np.isfinite(pred).all():
        raise ValueError("Predictions contain non-finite values")
    prediction = pd.Series(pred, index=dep_ids)
    submission = pd.read_parquet(template)
    if list(submission.columns) != ["MVT_ID_mvt", "TAXITIME_SEC_mvt"]:
        raise ValueError("Unexpected submitting.parquet columns")
    ids = submission["MVT_ID_mvt"]
    if ids.isna().any() or ids.duplicated().any():
        raise ValueError("Template IDs must be unique and non-null")
    if set(ids) != set(prediction.index):
        raise ValueError("Template departure IDs do not exactly match ranking IDs")
    submission["TAXITIME_SEC_mvt"] = ids.map(prediction).to_numpy(dtype=np.float64)
    if submission["TAXITIME_SEC_mvt"].isna().any():
        raise ValueError("Submission has missing predictions")
    out.parent.mkdir(parents=True, exist_ok=True)
    submission.to_parquet(out, index=False)
    check = pd.read_parquet(out)
    if not check["MVT_ID_mvt"].equals(ids) or not np.isfinite(check["TAXITIME_SEC_mvt"]).all():
        raise ValueError("Saved submission failed round-trip verification")
    print(json.dumps({"submission": str(out.resolve()), "rows": len(check),
                      "proxy_coverage": float(valid.mean()),
                      "prediction_mean_sec": float(np.mean(pred)),
                      "prediction_min_sec": float(np.min(pred)),
                      "prediction_max_sec": float(np.max(pred))}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    a = sub.add_parser("audit", help="Measure supplied off-block proxy before modeling")
    a.add_argument("--data-dir", type=Path, default=Path("data"))
    t = sub.add_parser("train", help="Validate and fit final models")
    t.add_argument("--data-dir", type=Path, default=Path("data"))
    t.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    t.add_argument("--threads", type=int, default=8)
    t.add_argument("--rounds", type=int, default=1800)
    p = sub.add_parser("predict", help="Write and verify a valid submission Parquet")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--model-dir", type=Path, default=Path("artifacts"))
    p.add_argument("--team", required=True)
    p.add_argument("--version", type=int, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("submissions"))
    p.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if args.command == "audit":
        audit(args.data_dir)
    elif args.command == "train":
        train(args.data_dir, args.output_dir, args.threads, args.rounds)
    else:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.team) or args.version < 1:
            parser.error("--team must be a simple team name and --version must be positive")
        out = args.output_dir / f"{args.team}_v{args.version}.parquet"
        if out.exists():
            parser.error(f"Refusing to overwrite existing submission: {out}")
        predict(args.data_dir, args.model_dir, out, args.threads)


if __name__ == "__main__":
    main()
