"""Predeclared larger CatBoost residual using released clocks and ARR traffic.

Select one blend on January/July 2025, then check it unchanged on November/
December. A paired April/October audit refits both the prior deep architecture
and this candidate without those months. Ranking feedback never selects models.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from catboost import CatBoostRegressor, Pool
import deep_timestamp_expert as deep

FOLDS = deep.FOLDS
WEIGHTS = deep.WEIGHTS


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_features(args, ranking=False, arrival=True):
    rows, features = deep.load_features(args, ranking)
    if arrival:
        prefix = 'ranking' if ranking else 'training'
        arr = pd.read_parquet(args.arrival_dir / f'{prefix}_arrival_features.parquet')
        if not np.array_equal(arr.MVT_ID_mvt.to_numpy(), rows.MVT_ID_mvt.to_numpy()):
            raise ValueError('ARR cache does not align with departure IDs')
        features = pd.concat([features.reset_index(drop=True),
                              arr.drop(columns='MVT_ID_mvt').reset_index(drop=True)], axis=1)
    if features.columns.duplicated().any():
        raise ValueError('Duplicate predictor names')
    return rows, features


def bootstrap(frame, old, new, seed=20261002):
    dates = pd.to_datetime(frame.MVT_TIME_UTC_mvt, utc=True).dt.floor('D')
    groups, days = pd.factorize(dates, sort=True)
    n = np.bincount(groups).astype(float)
    y = frame.target.to_numpy(dtype=float)
    a = np.bincount(groups, weights=(y-old)**2)
    b = np.bincount(groups, weights=(y-new)**2)
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(days), size=(1000, len(days)))
    sampled_n = n[samples].sum(axis=1)
    gains = np.sqrt(a[samples].sum(axis=1)/sampled_n) - np.sqrt(b[samples].sum(axis=1)/sampled_n)
    return {'days': len(days), 'repeats': 1000,
            'gain_ci95_sec': np.quantile(gains, [.025, .975]).tolist(),
            'fraction_positive': float((gains > 0).mean()),
            'observed_gain_sec': deep.rmse(y, old)-deep.rmse(y, new)}


def protocol(args):
    value = {'reference_sha256': digest(args.reference),
             'selection_months': [1, 7], 'corroboration_months': [11, 12],
             'fresh_architecture_audit_months': [4, 10],
             'weights': list(WEIGHTS), 'iterations': args.iterations,
             'depth': args.depth, 'seed': 2026,
             'features': 'Prior deep timestamp features plus the fixed 14 ARR traffic predictors',
             'training': 'Residual to AOBT, labels 0..86400 seconds, valid proxy 0..7200',
             'promotion': 'Seasonal selected nonzero blend; both existing folds have positive day CI lower bound; fresh paired architecture audit improves at same weight',
             'leaderboard_use': 'Progress only; no ranking values enter training, selection or promotion'}
    path = args.output_dir / 'protocol.json'
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError('Frozen experiment protocol changed')
    else:
        write_json(path, value)


def evaluate(args, include_audit=True):
    protocol(args)
    ref = pd.read_parquet(args.reference)
    if len(ref) != 672428 or ref.MVT_ID_mvt.duplicated().any():
        raise ValueError('Frozen v5 reference row universe changed')
    report = {'protocol': json.loads((args.output_dir/'protocol.json').read_text()),
              'folds': {}, 'selected_weight': None, 'promoted': False}
    parts = []
    for name in FOLDS:
        expert = pd.read_parquet(args.output_dir/f'{name}_oof.parquet')
        part = ref.loc[ref.fold.eq(name)].merge(expert, on='MVT_ID_mvt',
                                               how='left', validate='one_to_one')
        present = part.expert.notna().to_numpy()
        if not np.array_equal(present, part.a_valid.to_numpy(dtype=bool)):
            raise ValueError('Expert coverage must equal the valid-AOBT gate')
        base = part.selected.to_numpy(dtype=float)
        alt = part.expert.fillna(part.selected).to_numpy(dtype=float)
        y = part.target.to_numpy(dtype=float)
        scores = {str(w): deep.rmse(y, np.maximum(base+w*(alt-base), 0)) for w in WEIGHTS}
        if name == 'seasonal_jan_jul':
            report['selected_weight'] = min(WEIGHTS, key=lambda w: (scores[str(w)], w))
        weight = report['selected_weight']
        pred = np.maximum(base+weight*(alt-base), 0)
        info = json.loads((args.output_dir/f'{name}_validation.json').read_text())
        report['folds'][name] = {'scores': scores, 'n': len(part), 'n_changed_gate': int(present.sum()),
                                 'trees': info['trees'], 'fit_seconds': info['fit_seconds'],
                                 'bootstrap': bootstrap(part, base, pred)}
        part['candidate'] = pred
        parts.append(part)
    report['existing_folds_passed'] = report['selected_weight'] > 0 and all(
        row['bootstrap']['gain_ci95_sec'][0] > 0 for row in report['folds'].values())
    audit_path = args.output_dir/'fresh_audit.json'
    if include_audit and audit_path.exists():
        report['fresh_audit'] = json.loads(audit_path.read_text())
        audit = report['fresh_audit']
        if not audit.get('coverage_verified'):
            raise ValueError('Fresh audit needs the complete-ID coverage verification')
        for key, path in (
            ('training_rows_sha256', args.cache_dir/'training_rows.parquet'),
            ('prior_oof_sha256', args.output_dir/'fresh_old/fresh_apr_oct_oof.parquet'),
            ('new_oof_sha256', args.output_dir/'fresh_new/fresh_apr_oct_oof.parquet'),
            ('paired_predictions_sha256', args.output_dir/'fresh_audit_predictions.parquet'),
        ):
            if audit.get(key) != digest(path):
                raise ValueError(f'Fresh audit fingerprint changed: {key}')
        if report['fresh_audit']['weight'] != report['selected_weight']:
            raise ValueError('Fresh audit blend was changed after seasonal selection')
        report['promoted'] = report['existing_folds_passed'] and report['fresh_audit']['passed']
    pd.concat(parts, ignore_index=True).to_parquet(args.output_dir/'validation_predictions.parquet', index=False)
    write_json(args.output_dir/'validation.json', report)
    print(json.dumps(report, indent=2), flush=True)
    return report


def fit(args):
    protocol(args)
    rows, features = load_features(args)
    ref = pd.read_parquet(args.reference)
    for name, months in FOLDS.items():
        if not (args.output_dir/f'{name}_oof.parquet').exists():
            deep.fit_fold(name, months, rows, features, ref, args)
    del rows, features, ref
    gc.collect()
    evaluate(args)


def fresh_audit(args):
    """Paired models never trained/early-stopped on April or October targets."""
    report = evaluate(args, include_audit=False)
    if not report['existing_folds_passed']:
        raise ValueError('Candidate failed before fresh architecture audit')
    rows, features = load_features(args)
    proxy = rows.proxy.to_numpy(dtype=float)
    y = rows.target.to_numpy(dtype=float)
    held = rows.month.isin((4, 10)).to_numpy()
    valid = np.isfinite(proxy) & (proxy >= 0) & (proxy <= 7200) & np.isfinite(y)
    ref = rows.loc[held & valid, ['MVT_ID_mvt', 'target', 'time']].copy()
    ref.rename(columns={'time':'MVT_TIME_UTC_mvt'}, inplace=True)
    ref['selected'] = proxy[held & valid]
    ref['fold'] = 'fresh_apr_oct'
    old_args = argparse.Namespace(**vars(args))
    old_args.output_dir = args.output_dir/'fresh_old'
    old_args.iterations = 4500
    old_args.depth = 9
    if not (old_args.output_dir/'fresh_apr_oct_oof.parquet').exists():
        # The ARR cache is the authoritative feature list, including same-stand fields.
        arr_names = [c for c in pq.read_schema(args.arrival_dir/'training_arrival_features.parquet').names
                     if c != 'MVT_ID_mvt']
        prior_features = features.drop(columns=arr_names)
        deep.fit_fold('fresh_apr_oct', (4, 10), rows, prior_features, ref, old_args)
        del prior_features
        gc.collect()
    new_args = argparse.Namespace(**vars(args))
    new_args.output_dir = args.output_dir/'fresh_new'
    if not (new_args.output_dir/'fresh_apr_oct_oof.parquet').exists():
        deep.fit_fold('fresh_apr_oct', (4, 10), rows, features, ref, new_args)
    old = pd.read_parquet(old_args.output_dir/'fresh_apr_oct_oof.parquet').rename(columns={'expert':'old'})
    new = pd.read_parquet(new_args.output_dir/'fresh_apr_oct_oof.parquet').rename(columns={'expert':'new'})
    expected_ids = pd.Index(ref.MVT_ID_mvt)
    if expected_ids.has_duplicates or expected_ids.isna().any():
        raise ValueError('Fresh held-out reference has invalid IDs')
    for name, source in (('old', old), ('new', new)):
        ids = pd.Index(source.MVT_ID_mvt)
        if (ids.has_duplicates or ids.isna().any() or len(ids) != len(expected_ids)
                or not ids.isin(expected_ids).all() or not expected_ids.isin(ids).all()
                or not np.isfinite(source[name].to_numpy(dtype=float)).all()):
            raise ValueError(f'Fresh {name} OOF must cover every held-out valid-AOBT ID')
    paired = ref.merge(old, on='MVT_ID_mvt', how='left', sort=False, validate='one_to_one').merge(
        new, on='MVT_ID_mvt', how='left', sort=False, validate='one_to_one')
    if (len(paired) != len(ref) or not np.array_equal(paired.MVT_ID_mvt.to_numpy(), ref.MVT_ID_mvt.to_numpy())
            or not np.array_equal(paired.target.to_numpy(), ref.target.to_numpy())
            or not np.isfinite(paired[['old','new','target']].to_numpy(dtype=float)).all()):
        raise ValueError('Fresh paired predictions or targets are incomplete/misaligned')
    base = np.maximum(paired.old.to_numpy(dtype=float), 0)
    alt = paired.new.to_numpy(dtype=float)
    weight = report['selected_weight']
    pred = np.maximum(base+weight*(alt-base), 0)
    audit = {'months': [4,10], 'weight': weight, 'n_valid_aobt': len(paired),
             'prior_architecture_rmse': deep.rmse(paired.target.to_numpy(), base),
             'blended_new_architecture_rmse': deep.rmse(paired.target.to_numpy(), pred),
             'bootstrap': bootstrap(paired, base, pred, seed=20261004),
             'scope': 'Independent architecture comparison; not a complete v5 ensemble forecast'}
    audit['passed'] = audit['bootstrap']['gain_ci95_sec'][0] > 0
    paired['candidate'] = pred
    paired.to_parquet(args.output_dir/'fresh_audit_predictions.parquet', index=False)
    audit.update({'coverage_verified': True,
                  'training_rows_sha256': digest(args.cache_dir/'training_rows.parquet'),
                  'prior_oof_sha256': digest(old_args.output_dir/'fresh_apr_oct_oof.parquet'),
                  'new_oof_sha256': digest(new_args.output_dir/'fresh_apr_oct_oof.parquet'),
                  'paired_predictions_sha256': digest(args.output_dir/'fresh_audit_predictions.parquet')})
    write_json(args.output_dir/'fresh_audit.json', audit)
    del rows, features
    gc.collect()
    evaluate(args)


def final_predict(args):
    report = evaluate(args)
    if not report['promoted']:
        raise ValueError('Candidate failed a predeclared local validation gate')
    params = deep.params(args)
    params['iterations'] = int(np.median([r['trees'] for r in report['folds'].values()]))
    model = CatBoostRegressor(**params)
    model_path = args.output_dir/'full_2025.cbm'
    if not model_path.exists():
        rows, features = load_features(args)
        proxy = rows.proxy.to_numpy(dtype=float)
        y = rows.target.to_numpy(dtype=float)
        core = np.isfinite(y)&(y>=0)&(y<=86400)&np.isfinite(proxy)&(proxy>=0)&(proxy<=7200)
        cats = features.select_dtypes(include='category').columns.tolist()
        pool = Pool(features.loc[core], label=(y-proxy)[core], cat_features=cats)
        model.fit(pool)
        model.save_model(str(model_path))
        del pool, rows, features
        gc.collect()
    else:
        model.load_model(str(model_path))
    rows, features = load_features(args, ranking=True)
    proxy = rows.proxy.to_numpy(dtype=float)
    valid = np.isfinite(proxy)&(proxy>=0)&(proxy<=7200)
    pred = np.full(len(rows), np.nan)
    pred[valid] = proxy[valid]+model.predict(features.loc[valid], thread_count=args.threads)
    raw = pd.DataFrame({'MVT_ID_mvt': rows.MVT_ID_mvt, 'expert': pred})
    raw.to_parquet(args.output_dir/'ranking_expert.parquet', index=False)
    frozen = pd.read_parquet(args.ranking_reference)
    if not np.array_equal(frozen.MVT_ID_mvt.to_numpy(), rows.MVT_ID_mvt.to_numpy()):
        raise ValueError('Ranking reference ID order differs from the feature cache')
    base = frozen.merge(raw, on='MVT_ID_mvt', validate='one_to_one', sort=False)
    present = base.expert.notna().to_numpy()
    if len(base)!=344841 or not np.array_equal(present, valid):
        raise ValueError('Ranking reference coverage/order differs from cached features')
    values = base.TAXITIME_SEC_mvt.to_numpy(dtype=float, copy=True)
    weight = report['selected_weight']
    values[present] = np.maximum(values[present]+weight*(base.expert.to_numpy()[present]-values[present]),0)
    base['TAXITIME_SEC_mvt'] = values
    base[['MVT_ID_mvt','TAXITIME_SEC_mvt']].to_parquet(args.output_dir/'predictions.parquet', index=False)
    write_json(args.output_dir/'manifest.json', {'trees': model.tree_count_, 'weight': weight,
        'n_rank': len(base), 'n_eligible': int(valid.sum()), 'features': list(features),
        'selection': 'Labeled 2025 validation only; all predeclared gates passed',
        'ranking_reference_sha256': digest(args.ranking_reference)})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('fit','evaluate','fresh-audit','final-predict'), default='fit')
    p.add_argument('--data-dir',type=Path,default=Path('data'))
    p.add_argument('--cache-dir',type=Path,default=Path('artifacts/baseline'))
    p.add_argument('--weather-file',type=Path,default=Path('data/external/weather.parquet'))
    p.add_argument('--arrival-dir',type=Path,default=Path('artifacts/v5-arrival-clean'))
    p.add_argument('--reference',type=Path,default=Path('artifacts/v5-ensemble/validation_predictions.parquet'))
    p.add_argument('--ranking-reference',type=Path,default=Path('submissions/merry-mushroom_v5.parquet'))
    p.add_argument('--output-dir',type=Path,default=Path('artifacts/v6-deep-arrival'))
    p.add_argument('--iterations',type=int,default=10000)
    p.add_argument('--depth',type=int,default=10)
    p.add_argument('--threads',type=int,default=2)
    args=p.parse_args()
    {'fit':fit,'evaluate':evaluate,'fresh-audit':fresh_audit,'final-predict':final_predict}[args.mode](args)


if __name__=='__main__':
    main()
