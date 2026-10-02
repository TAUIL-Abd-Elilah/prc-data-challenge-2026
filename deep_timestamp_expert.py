"""Higher-capacity GPU residual expert with released timestamp precision features.

Training and ranking departure BLOCK/TAXITIME fields never enter the feature
loader. The only training labels come from the existing separate label cache.
"""
from __future__ import annotations
import argparse
import gc
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
from catboost import CatBoostRegressor, Pool
from catboost_expert import add_flight_weather, load_cache, rmse
from solution import TIME_COLS, _training_files
from v4_reference import load_v4

FOLDS = {'seasonal_jan_jul': (1, 7), 'forward_nov_dec': (11, 12)}
WEIGHTS = (0., .1, .25, .5, 1.)


def augment_timestamps(rows, x, data_dir, ranking=False):
    paths = [data_dir / 'ranking.parquet'] if ranking else _training_files(data_dir)
    scan = pl.scan_parquet([str(p) for p in paths])
    names = scan.collect_schema().names()
    cols = [c for c in ('MVT_ID_mvt', 'FLIGHT_ID_mvt', 'CALLSIGN_flt', 'FLIGHT_mvt', *TIME_COLS) if c in names]
    raw = scan.filter(pl.col('PHASE_mvt') == 'DEP').select(cols).collect().to_pandas()
    if not np.array_equal(raw.MVT_ID_mvt.to_numpy(), rows.MVT_ID_mvt.to_numpy()):
        raise ValueError('Departure timestamp feature alignment failed')
    x = x.copy()
    for col in TIME_COLS:
        if col not in raw:
            continue
        ts = pd.to_datetime(raw[col], utc=True, errors='coerce')
        raw[col] = ts
        x[f'time_second_{col}'] = ts.dt.second.astype('float32')
        x[f'time_minute_{col}'] = ts.dt.minute.astype('float32')
        x[f'time_hour_{col}'] = ts.dt.hour.astype('float32')
        x[f'time_weekday_{col}'] = ts.dt.dayofweek.astype('float32')
    for left, right, name in (
        ('ARVT_3_flt', 'AOBT_3_flt', 'nm_flown_duration'),
        ('ARVT_1_flt', 'EOBT_1_flt', 'nm_planned_duration'),
        ('ARVT_3_flt', 'ARVT_1_flt', 'nm_arrival_delay'),
        ('EOBT_1_flt', 'IOBT_flt', 'nm_planning_revision'),
        ('MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'schedule_gap_unclipped'),
    ):
        if left in raw and right in raw:
            x[name] = (raw[left] - raw[right]).dt.total_seconds().astype('float32')
    if 'FLIGHT_ID_mvt' in raw:
        x['nm_id_missing'] = raw.FLIGHT_ID_mvt.isna().astype('int8')
    if 'CALLSIGN_flt' in raw:
        call = raw.CALLSIGN_flt.astype('string').fillna('__MISSING__')
        flight = raw.FLIGHT_mvt.astype('string').fillna('__MISSING__')
        x['full_nm_callsign'] = call.astype('category')
        x['flight_callsign_equal'] = call.eq(flight).astype('int8')
    return x


def params(args):
    return dict(task_type='GPU', devices='0', gpu_ram_part=.45,
        loss_function='RMSE', eval_metric='RMSE', iterations=args.iterations,
        depth=args.depth, learning_rate=.04, l2_leaf_reg=12,
        random_strength=.5, bagging_temperature=.5, max_ctr_complexity=1,
        one_hot_max_size=20, border_count=128, thread_count=args.threads,
        random_seed=2026, allow_writing_files=False, verbose=500)


def load_features(args, ranking=False):
    rows, x = load_cache(args.cache_dir, ranking)
    x = add_flight_weather(rows, x, args.data_dir, args.weather_file, ranking)
    x = augment_timestamps(rows, x, args.data_dir, ranking)
    return rows, x


def fit_fold(name, months, rows, x, reference, args):
    y = rows.target.to_numpy(dtype=float)
    proxy = rows.proxy.to_numpy(dtype=float)
    proxy_valid = np.isfinite(proxy) & (proxy >= 0) & (proxy <= 7200)
    heldout = rows.month.isin(months).to_numpy()
    train = np.flatnonzero(~heldout & proxy_valid & np.isfinite(y) & (y >= 0) & (y <= 86400))
    test = np.flatnonzero(heldout & proxy_valid & np.isfinite(y))
    rng = np.random.default_rng(2026)
    order = rng.permutation(train)
    n_early = max(20000, int(.06 * len(order)))
    early, fit_idx = order[:n_early], order[n_early:]
    cats = x.select_dtypes(include='category').columns.tolist()
    labels = y - proxy
    model = CatBoostRegressor(**params(args))
    pool = Pool(x.iloc[fit_idx], label=labels[fit_idx], cat_features=cats)
    ev = Pool(x.iloc[early], label=labels[early], cat_features=cats)
    start = time.monotonic()
    model.fit(pool, eval_set=ev, early_stopping_rounds=200, use_best_model=True)
    elapsed = time.monotonic() - start
    del pool, ev
    gc.collect()
    pred = proxy[test] + model.predict(x.iloc[test], thread_count=args.threads)
    out = pd.DataFrame({'MVT_ID_mvt': rows.MVT_ID_mvt.iloc[test], 'expert': pred})
    full = reference[reference.fold.eq(name)].merge(out, on='MVT_ID_mvt', how='left', validate='one_to_one')
    target = full.target.to_numpy(dtype=float)
    base = full.selected.to_numpy(dtype=float)
    present = full.expert.notna().to_numpy()
    alternative = full.expert.fillna(full.selected).to_numpy(dtype=float)
    scores = {str(w): rmse(target, np.maximum(base + w * (alternative - base), 0)) for w in WEIGHTS}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output_dir / f'{name}_oof.parquet', index=False)
    model.save_model(str(args.output_dir / f'{name}.cbm'))
    report = {'fold': name, 'n_all_finite': len(full), 'n_eligible': int(present.sum()),
        'training_eligible': len(train), 'features': list(x), 'best_iteration': model.get_best_iteration(),
        'trees': model.tree_count_, 'fit_seconds': elapsed, 'scores_all_finite': scores}
    (args.output_dir / f'{name}_validation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('fold','trees','fit_seconds','scores_all_finite')}, indent=2), flush=True)
    return report


def evaluate_saved(args):
    """Apply the submitted nonnegative policy without repeating model fits."""
    reference = load_v4()
    report = {'method': 'Released timestamp precision/duration + deeper GPU residual; fixed v4 comparison',
              'selection': 'January/July only', 'weights': WEIGHTS,
              'output_policy': 'Finite predictions, clipped at zero'}
    for name in FOLDS:
        existing = json.loads((args.output_dir / f'{name}_validation.json').read_text())
        expert = pd.read_parquet(args.output_dir / f'{name}_oof.parquet')
        full = reference[reference.fold.eq(name)].merge(expert, on='MVT_ID_mvt',
            how='left', validate='one_to_one')
        base = full.selected.to_numpy(dtype=float)
        alternate = full.expert.fillna(full.selected).to_numpy(dtype=float)
        existing['scores_all_finite'] = {str(w): rmse(full.target.to_numpy(),
            np.maximum(base + w * (alternate - base), 0)) for w in WEIGHTS}
        (args.output_dir / f'{name}_validation.json').write_text(json.dumps(existing, indent=2), encoding='utf-8')
        report[name] = existing
    selected = min(WEIGHTS, key=lambda w: report['seasonal_jan_jul']['scores_all_finite'][str(w)])
    report['selected_weight'] = selected
    report['forward_selected_rmse'] = report['forward_nov_dec']['scores_all_finite'][str(selected)]
    report['promoted'] = selected > 0 and report['forward_selected_rmse'] < report['forward_nov_dec']['scores_all_finite']['0.0']
    (args.output_dir / 'validation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('selected_weight', 'forward_selected_rmse', 'promoted')}, indent=2), flush=True)
    return report


def fit_final(args):
    validation = evaluate_saved(args)
    if not validation['promoted']:
        raise ValueError('The deep expert did not pass local selection and forward checking')
    iterations = int(np.median([validation[name]['trees'] for name in FOLDS]))
    rows, x = load_features(args)
    y = rows.target.to_numpy(dtype=float)
    proxy = rows.proxy.to_numpy(dtype=float)
    eligible = np.isfinite(y) & (y >= 0) & (y <= 86400) & np.isfinite(proxy) & (proxy >= 0) & (proxy <= 7200)
    train = np.flatnonzero(eligible)
    cats = x.select_dtypes(include='category').columns.tolist()
    model_params = params(args)
    model_params['iterations'] = iterations
    model = CatBoostRegressor(**model_params)
    pool = Pool(x.iloc[train], label=(y - proxy)[train], cat_features=cats)
    start = time.monotonic()
    model.fit(pool)
    elapsed = time.monotonic() - start
    model.save_model(str(args.output_dir / 'full_2025.cbm'))
    del pool, rows, x
    gc.collect()
    rank_rows, rank_x = load_features(args, ranking=True)
    rank_proxy = rank_rows.proxy.to_numpy(dtype=float)
    valid = np.isfinite(rank_proxy) & (rank_proxy >= 0) & (rank_proxy <= 7200)
    pred = np.full(len(rank_rows), np.nan)
    pred[valid] = rank_proxy[valid] + model.predict(rank_x.loc[valid], thread_count=args.threads)
    pd.DataFrame({'MVT_ID_mvt': rank_rows.MVT_ID_mvt, 'expert': pred}).to_parquet(
        args.output_dir / 'ranking_expert.parquet', index=False)
    report = {'training_eligible': int(eligible.sum()), 'ranking_eligible': int(valid.sum()),
              'ranking_rows': len(rank_rows), 'iterations': iterations, 'fit_seconds': elapsed,
              'blend_weight': validation['selected_weight'], 'features': list(rank_x),
              'labels': '2025 only; core labels 0..86400 seconds',
              'command': 'python deep_timestamp_expert.py --mode final-predict --iterations 4500 --depth 9 --threads 2'}
    (args.output_dir / 'manifest.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('training_eligible', 'ranking_eligible', 'iterations', 'fit_seconds')}, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, default=Path('data'))
    p.add_argument('--cache-dir', type=Path, default=Path('artifacts/baseline'))
    p.add_argument('--weather-file', type=Path, default=Path('data/external/weather.parquet'))
    p.add_argument('--output-dir', type=Path, default=Path('artifacts/v5-deep'))
    p.add_argument('--iterations', type=int, default=4500)
    p.add_argument('--depth', type=int, default=9)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--mode', choices=('fit', 'evaluate', 'final-predict'), default='fit')
    args = p.parse_args()
    if args.mode == 'evaluate':
        evaluate_saved(args)
        return
    if args.mode == 'final-predict':
        fit_final(args)
        return
    rows, x = load_features(args)
    reference = load_v4()
    report = {'method': 'Released timestamp precision/duration + deeper GPU residual; fixed v4 comparison',
              'selection': 'January/July only', 'weights': WEIGHTS}
    name = 'seasonal_jan_jul'
    report[name] = fit_fold(name, FOLDS[name], rows, x, reference, args)
    weight = min(WEIGHTS, key=lambda w: report[name]['scores_all_finite'][str(w)])
    report['selected_weight'] = weight
    if weight > 0:
        name = 'forward_nov_dec'
        report[name] = fit_fold(name, FOLDS[name], rows, x, reference, args)
        report['forward_selected_rmse'] = report[name]['scores_all_finite'][str(weight)]
    (args.output_dir / 'validation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
