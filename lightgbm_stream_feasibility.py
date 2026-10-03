"""Synthetic-only audit of exact LightGBM pandas versus streamed-memmap fits.

This script never reads competition rows, saved models, or ranking data. It
tests the installed LightGBM's pandas encoding against bounded conversion to
a disk-backed NumPy matrix, then repeats the same native training call.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

import movement_only_expert as movement


def synthetic_frame() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20261003)
    n = 6600
    cats = pd.CategoricalDtype(categories=["LIRF", "EGLL", "LFPG", "LSZH"],
                               ordered=False)
    stand = pd.CategoricalDtype(categories=["C", "A", "D", "B"], ordered=False)
    frame = pd.DataFrame({
        "numeric_f32": rng.normal(size=n).astype(np.float32),
        "airport": pd.Series(rng.choice(cats.categories, n), dtype=cats),
        "numeric_f64": rng.uniform(-2, 4, size=n),
        "stand": pd.Series(rng.choice(stand.categories, n), dtype=stand),
        "integer_i16": rng.integers(0, 60, size=n, dtype=np.int16),
    })
    frame.loc[::23, "airport"] = None
    frame.loc[::17, "stand"] = None
    frame.loc[::19, "numeric_f32"] = np.nan
    y = (0.8 * np.nan_to_num(frame.numeric_f32.to_numpy(dtype=float))
         + 1.5 * frame.airport.cat.codes.to_numpy(dtype=float)
         + 0.35 * frame.stand.cat.codes.to_numpy(dtype=float)
         + np.sin(frame.integer_i16.to_numpy(dtype=float) / 9)
         + rng.normal(scale=.08, size=n)).astype(np.float32)
    day_number = np.arange(n) // 55
    month = np.where(day_number % 12 == 0, 1,
                     np.where(day_number % 12 == 6, 7, 3))
    ordinary = (y >= -10) & (y <= 7200) & ~np.isin(month, (1, 7))
    early = ordinary & (day_number % 11 == 0)
    fit = ordinary & ~early
    if fit.sum() < 1000 or early.sum() < 100:
        raise ValueError("Synthetic complement/early masks are too small")
    return frame, y, fit, early


def encoded_memmap(frame: pd.DataFrame, mask: np.ndarray, path: Path,
                   categories: list[list]) -> np.memmap:
    """Use LightGBM's own pandas conversion in bounded, global-category chunks."""
    columns = list(frame)
    cats = [name for name in columns
            if isinstance(frame[name].dtype, pd.CategoricalDtype)]
    first, _, _, _ = lgb.basic._data_from_pandas(
        frame.iloc[:2], columns, cats, categories)
    out = np.memmap(path, mode="w+", dtype=first.dtype,
                    shape=(int(mask.sum()), len(columns)))
    cursor = 0
    for start in range(0, len(frame), 37):
        end = min(start + 37, len(frame))
        selected = mask[start:end]
        if not selected.any():
            continue
        part = frame.iloc[start:end].loc[selected]
        values, names, categorical, observed = lgb.basic._data_from_pandas(
            part, columns, cats, categories)
        if names != columns or categorical != cats or observed != categories:
            raise ValueError("LightGBM streamed pandas category conversion changed")
        out[cursor:cursor + len(values)] = values
        cursor += len(values)
    out.flush()
    if cursor != int(mask.sum()):
        raise ValueError("Streamed fit/early mask coverage differs")
    return out


def fit(dataset: lgb.Dataset, early: lgb.Dataset,
        bin_sample_count: int | None) -> lgb.Booster:
    params = movement.model_params(3)
    if bin_sample_count is not None:
        # Synthetic stress test of native bin sampling only; the frozen
        # competition configuration leaves LightGBM's default unchanged.
        params["bin_construct_sample_cnt"] = bin_sample_count
    return lgb.train(params, dataset, num_boost_round=90,
                     valid_sets=[early], callbacks=[
                         lgb.early_stopping(20, verbose=False)])


def run(bin_sample_count: int | None = None) -> dict:
    frame, y, fit_mask, early_mask = synthetic_frame()
    columns = list(frame)
    cat_names = [name for name in columns
                 if isinstance(frame[name].dtype, pd.CategoricalDtype)]
    categories = [list(frame[name].cat.categories) for name in cat_names]
    full_array, _, _, _ = lgb.basic._data_from_pandas(
        frame.loc[fit_mask], columns, cat_names, None)
    full_early_array, _, _, _ = lgb.basic._data_from_pandas(
        frame.loc[early_mask], columns, cat_names, categories)
    pandas_train = lgb.Dataset(frame.loc[fit_mask], label=y[fit_mask],
                               categorical_feature=cat_names, free_raw_data=True)
    pandas_early = lgb.Dataset(frame.loc[early_mask], label=y[early_mask],
                               reference=pandas_train,
                               categorical_feature=cat_names, free_raw_data=True)
    model_pandas = fit(pandas_train, pandas_early, bin_sample_count)
    with tempfile.TemporaryDirectory(prefix="lgb-stream-feasibility-") as tmp:
        root = Path(tmp)
        train_values = encoded_memmap(frame, fit_mask, root / "train.dat", categories)
        early_values = encoded_memmap(frame, early_mask, root / "early.dat", categories)
        if (train_values.dtype != full_array.dtype
                or early_values.dtype != full_early_array.dtype
                or not np.array_equal(train_values, full_array, equal_nan=True)
                or not np.array_equal(early_values, full_early_array,
                                      equal_nan=True)):
            raise ValueError("Bounded conversion differs from full pandas encoding")
        memmap_train = lgb.Dataset(train_values, label=y[fit_mask],
                                   feature_name=columns,
                                   categorical_feature=cat_names,
                                   free_raw_data=True)
        memmap_train.pandas_categorical = categories
        memmap_early = lgb.Dataset(early_values, label=y[early_mask],
                                   reference=memmap_train,
                                   feature_name=columns,
                                   categorical_feature=cat_names,
                                   free_raw_data=True)
        model_memmap = fit(memmap_train, memmap_early, bin_sample_count)
        pred_pandas = model_pandas.predict(frame, num_threads=3)
        pred_memmap = model_memmap.predict(frame, num_threads=3)
        result = {
            "lightgbm_version": lgb.__version__,
            "synthetic_rows": len(frame),
            "fit_rows": int(fit_mask.sum()),
            "early_rows": int(early_mask.sum()),
            "matrix_dtype": str(train_values.dtype),
            "synthetic_bin_sample_count": (bin_sample_count if
                                           bin_sample_count is not None else
                                           "LightGBM default"),
            "encoded_values_exact": True,
            "best_iteration_pandas": int(model_pandas.best_iteration),
            "best_iteration_memmap": int(model_memmap.best_iteration),
            "tree_dump_equal": (model_pandas.dump_model()["tree_info"] ==
                                model_memmap.dump_model()["tree_info"]),
            "model_text_equal": (model_pandas.model_to_string() ==
                                 model_memmap.model_to_string()),
            "prediction_bitwise_equal": bool(np.array_equal(
                pred_pandas, pred_memmap)),
            "prediction_max_abs_diff": float(np.max(np.abs(
                pred_pandas - pred_memmap))),
        }
        del model_memmap, memmap_train, memmap_early, train_values, early_values
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-bin-sample-count", type=int)
    options = parser.parse_args()
    print(json.dumps(run(options.synthetic_bin_sample_count), indent=2),
          flush=True)
