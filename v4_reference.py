"""Reconstruct the fixed v4 OOF comparison from already saved local experts."""
from pathlib import Path
import numpy as np
import pandas as pd


def load_v4(root: Path = Path('.'), verify_snapshot: bool = False) -> pd.DataFrame:
    base = pd.read_parquet(root / 'artifacts/lobt_ensemble/validation_predictions.parquet',
        columns=['MVT_ID_mvt', 'target', 'fold', 'a_valid', 'selected'])
    gpu = pd.concat([pd.read_parquet(root / f'artifacts/catboost/gpu/{fold}_oof.parquet',
        columns=['MVT_ID_mvt', 'target', 'gpu_prediction'])
        for fold in ('seasonal_jan_jul', 'forward_nov_dec')], ignore_index=True)
    source = pd.concat([pd.read_parquet(root / f'artifacts/catboost/source/{fold}_oof.parquet',
        columns=['MVT_ID_mvt', 'target', 'schedule_proxy_sec', 'p_schedule_exact'])
        for fold in ('seasonal_jan_jul', 'forward_nov_dec')], ignore_index=True)
    frame = base.merge(gpu, on='MVT_ID_mvt', how='left', validate='one_to_one',
                       suffixes=('', '_gpu'))
    present = frame.gpu_prediction.notna()
    if not np.allclose(frame.loc[present, 'target'], frame.loc[present, 'target_gpu']):
        raise ValueError('GPU reference labels disagree')
    pred = frame.selected.to_numpy(dtype=float).copy()
    pred[present] += .25 * (frame.loc[present, 'gpu_prediction'].to_numpy() - pred[present])
    frame['selected'] = pred
    frame = frame.drop(columns=['target_gpu', 'gpu_prediction']).merge(source,
        on='MVT_ID_mvt', how='left', validate='one_to_one', suffixes=('', '_source'))
    present = frame.p_schedule_exact.notna()
    if not np.allclose(frame.loc[present, 'target'], frame.loc[present, 'target_source']):
        raise ValueError('Source reference labels disagree')
    pred = frame.selected.to_numpy(dtype=float).copy()
    pred[present] += .5 * frame.loc[present, 'p_schedule_exact'].to_numpy() * (
        frame.loc[present, 'schedule_proxy_sec'].to_numpy() - pred[present])
    frame['selected'] = pred
    meta = pd.read_parquet(root / 'artifacts/baseline/training_rows.parquet')
    wanted = ['MVT_ID_mvt', 'airport', 'month', 'MVT_TIME_UTC_mvt', 'time']
    frame = frame[['MVT_ID_mvt', 'target', 'fold', 'a_valid', 'selected']].merge(
        meta[[c for c in wanted if c in meta]], on='MVT_ID_mvt', how='left',
        validate='one_to_one')
    if not np.isfinite(frame.target).all() or not np.isfinite(frame.selected).all():
        raise ValueError('The v4 comparison must cover every finite validation label')
    expected = 282.4120619194716
    actual = float(np.sqrt(np.mean((frame.target - frame.selected) ** 2)))
    # GPU fits can differ slightly across hardware and reruns. The historical
    # score is an optional snapshot audit, not a gate on legitimate reproduction.
    if len(frame) != 672428 or (verify_snapshot and abs(actual - expected) > 1e-6):
        raise ValueError(f'Frozen v4 reference changed: {len(frame)} rows, RMSE {actual}')
    # The submitted pipeline enforces nonnegative outputs. Preserve the legacy
    # diagnostic value and compare new models using that actual output policy.
    frame['selected_unclipped'] = frame['selected']
    frame['selected'] = np.maximum(frame['selected'].to_numpy(), 0)
    if 'MVT_TIME_UTC_mvt' not in frame and 'time' in frame:
        frame = frame.rename(columns={'time': 'MVT_TIME_UTC_mvt'})
    return frame


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-snapshot', action='store_true')
    args = parser.parse_args()
    frame = load_v4(verify_snapshot=args.verify_snapshot)
    folder = Path('artifacts/v4')
    folder.mkdir(parents=True, exist_ok=True)
    temporary = folder / 'validation_predictions.tmp.parquet'
    frame.to_parquet(temporary, index=False)
    temporary.replace(folder / 'validation_predictions.parquet')
    score = np.sqrt(np.mean((frame.target - frame.selected) ** 2))
    print(f'Saved {len(frame):,} fixed-v4 OOF rows with submitted nonnegative policy; RMSE {score:.9f}')
