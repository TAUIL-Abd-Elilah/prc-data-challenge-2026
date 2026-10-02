"""Create an exactly aligned, checked and hashed PRC competition submission."""
from __future__ import annotations
import argparse
import hashlib
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", type=Path, default=Path("artifacts/ensemble/predictions.parquet"))
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--output-dir", type=Path, default=Path("submissions"))
    p.add_argument("--team", default="merry-mushroom")
    p.add_argument("--version", type=int, required=True)
    args = p.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.team) or args.version < 1:
        p.error("Invalid team name or version")
    template = pd.read_parquet(args.data_dir / "submitting.parquet")
    prediction = pd.read_parquet(args.predictions)
    required = ["MVT_ID_mvt", "TAXITIME_SEC_mvt"]
    if list(template) != required or not set(required).issubset(prediction):
        raise ValueError("Submission columns differ from organizer specification")
    ids = template.MVT_ID_mvt
    if ids.isna().any() or ids.duplicated().any() or prediction.MVT_ID_mvt.isna().any() or prediction.MVT_ID_mvt.duplicated().any():
        raise ValueError("IDs must be unique and non-null")
    if set(ids) != set(prediction.MVT_ID_mvt):
        raise ValueError("Predictions do not match every template ID exactly")
    series = prediction.set_index("MVT_ID_mvt").TAXITIME_SEC_mvt
    values = ids.map(series).to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Predictions must be finite and nonnegative")
    output = template.copy()
    output.TAXITIME_SEC_mvt = values
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / f"{args.team}_v{args.version}.parquet"
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    output.to_parquet(path, index=False)
    check = pd.read_parquet(path)
    assert list(check) == required and check.MVT_ID_mvt.equals(ids)
    assert np.array_equal(check.TAXITIME_SEC_mvt.to_numpy(), values)
    result = {"file": path.name, "rows": len(output), "columns": required,
              "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size,
              "all_template_ids_in_original_order": True, "finite_predictions": True,
              "prediction_min_sec": float(values.min()), "prediction_max_sec": float(values.max()),
              "prediction_mean_sec": float(values.mean()), "predictions_above_24h": int((values > 86400).sum())}
    (path.with_suffix(".manifest.json")).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
