"""Synthetic-only LightGBM Sequence versus pandas Dataset parity audit.

This reads no competition rows or saved models. The default sampling case uses
all 6,600 synthetic rows for bin construction. The 1,000-row sampling stress
represents the full-scale regime, where rows exceed the bin sample cap.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path
import tempfile

import lightgbm as lgb
from lightgbm.basic import Sequence
import numpy as np
import pandas as pd

import lightgbm_stream_feasibility as synthetic
import movement_only_expert as movement


class DiskRows(Sequence):
    """Bounded reads of a fixed, pandas-encoded float64 matrix on disk."""

    batch_size = 113

    def __init__(self, path: Path, shape: tuple[int, int], dtype: np.dtype):
        self.array = np.memmap(path, mode="r", dtype=dtype, shape=shape,
                               order="C")
        self.max_batch_rows = 0

    def __len__(self) -> int:
        return self.array.shape[0]

    def __getitem__(self, index):
        result = np.asarray(self.array[index])
        self.max_batch_rows = max(
            self.max_batch_rows, len(result) if result.ndim == 2 else 1)
        return result

    def close(self) -> None:
        self.array._mmap.close()


def write_matrix(path: Path, values: np.ndarray) -> None:
    matrix = np.memmap(path, mode="w+", dtype=values.dtype,
                       shape=values.shape, order="C")
    matrix[:] = values
    matrix.flush()
    matrix._mmap.close()


def train(dataset: lgb.Dataset, early: lgb.Dataset,
          sample_count: int | None) -> lgb.Booster:
    params = movement.model_params(3)
    if sample_count is not None:
        params["bin_construct_sample_cnt"] = sample_count
    return lgb.train(params, dataset, num_boost_round=90,
                     valid_sets=[early], callbacks=[
                         lgb.early_stopping(20, verbose=False)])


def audit_one(frame: pd.DataFrame, y: np.ndarray, fit: np.ndarray,
              early: np.ndarray, names: list[str], cats: list[str],
              categories: list[list], root: Path,
              sample_count: int | None) -> dict:
    fit_frame, early_frame = frame.loc[fit], frame.loc[early]
    fit_encoded, fit_names, fit_cats, fit_levels = lgb.basic._data_from_pandas(
        fit_frame, names, cats, categories)
    early_encoded, early_names, early_cats, early_levels = lgb.basic._data_from_pandas(
        early_frame, names, cats, categories)
    if (fit_names != names or early_names != names or fit_cats != cats
            or early_cats != cats or fit_levels != categories
            or early_levels != categories):
        raise ValueError("Synthetic pandas category schema changed")
    fit_path, early_path = root / "fit.dat", root / "early.dat"
    write_matrix(fit_path, fit_encoded)
    write_matrix(early_path, early_encoded)
    fit_seq = DiskRows(fit_path, fit_encoded.shape, fit_encoded.dtype)
    early_seq = DiskRows(early_path, early_encoded.shape, early_encoded.dtype)
    try:
        encoded_equal = (
            np.array_equal(np.asarray(fit_seq.array), fit_encoded, equal_nan=True)
            and np.array_equal(np.asarray(early_seq.array), early_encoded,
                               equal_nan=True))
        if not encoded_equal:
            raise ValueError("Disk Sequence changed pandas-encoded values")
        pandas_train = lgb.Dataset(fit_frame, label=y[fit],
                                   categorical_feature=cats,
                                   free_raw_data=True)
        pandas_early = lgb.Dataset(early_frame, label=y[early],
                                   reference=pandas_train,
                                   categorical_feature=cats,
                                   free_raw_data=True)
        pandas_model = train(pandas_train, pandas_early, sample_count)
        sequence_train = lgb.Dataset(fit_seq, label=y[fit],
                                     feature_name=names,
                                     categorical_feature=cats,
                                     free_raw_data=True)
        sequence_train.pandas_categorical = categories
        sequence_early = lgb.Dataset(early_seq, label=y[early],
                                     reference=sequence_train,
                                     feature_name=names,
                                     categorical_feature=cats,
                                     free_raw_data=True)
        sequence_model = train(sequence_train, sequence_early, sample_count)
        pandas_pred = pandas_model.predict(frame, num_threads=3)
        sequence_pred = sequence_model.predict(frame, num_threads=3)
        result = {
            "bin_construct_sample_cnt": (sample_count if sample_count is not None
                                         else "LightGBM default"),
            "fit_rows": int(fit.sum()),
            "internal_early_rows": int(early.sum()),
            "encoded_values_exact": encoded_equal,
            "categorical_names_and_levels_exact": (
                pandas_model.feature_name() == sequence_model.feature_name() == names
                and pandas_model.pandas_categorical ==
                sequence_model.pandas_categorical == categories),
            "best_iteration_pandas": int(pandas_model.best_iteration),
            "best_iteration_sequence": int(sequence_model.best_iteration),
            "complete_model_text_equal": (pandas_model.model_to_string()
                                          == sequence_model.model_to_string()),
            "tree_dump_equal": (pandas_model.dump_model()["tree_info"]
                                == sequence_model.dump_model()["tree_info"]),
            "predictions_bitwise_equal": bool(np.array_equal(
                pandas_pred, sequence_pred)),
            "max_absolute_prediction_difference": float(np.max(np.abs(
                pandas_pred - sequence_pred))),
            "maximum_sequence_batch_rows": max(fit_seq.max_batch_rows,
                                                early_seq.max_batch_rows),
        }
        del (pandas_model, sequence_model, pandas_train, pandas_early,
             sequence_train, sequence_early)
        gc.collect()
        return result
    finally:
        fit_seq.close()
        early_seq.close()


def main() -> None:
    frame, y, fit, early = synthetic.synthetic_frame()
    names = list(frame)
    cats = [name for name in names
            if isinstance(frame[name].dtype, pd.CategoricalDtype)]
    categories = [list(frame[name].cat.categories) for name in cats]
    results = []
    for sample_count in (None, 1000):
        with tempfile.TemporaryDirectory(prefix="lgb-sequence-synthetic-") as temp:
            results.append(audit_one(frame, y, fit, early, names, cats,
                                     categories, Path(temp), sample_count))
    print(json.dumps({
        "synthetic_rows": len(frame),
        "lightgbm_version": lgb.__version__,
        "frozen_model_params": movement.model_params(3),
        "results": results,
        "competition_data_or_models_read": False,
        "competition_fit_performed": False,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
