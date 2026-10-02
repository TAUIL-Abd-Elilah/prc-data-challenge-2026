"""Expand an earlier baseline validation export to every finite training label."""
from pathlib import Path
import json
import lightgbm as lgb
import numpy as np
import pandas as pd
from solution import _scores


def main():
    folder = Path("artifacts/baseline")
    rows = pd.read_parquet(folder / "training_rows.parquet")
    x = pd.read_parquet(folder / "features.parquet")
    metrics = {}
    for name, months in {"seasonal_jan_jul": [1, 7], "forward_nov_dec": [11, 12]}.items():
        path = folder / f"{name}_oof.parquet"
        if not path.exists():
            continue
        oof = pd.read_parquet(path)
        valid = rows.month.isin(months) & np.isfinite(rows.target)
        missing = valid & ~rows.MVT_ID_mvt.isin(oof.MVT_ID_mvt)
        if missing.any():
            d = lgb.Booster(model_file=str(folder / f"{name}_direct.txt")).predict(x.loc[missing], num_threads=2)
            proxy = rows.loc[missing, "proxy"].to_numpy()
            hybrid = d.copy()
            usable = np.isfinite(proxy)
            if usable.any():
                residual = lgb.Booster(model_file=str(folder / f"{name}_residual.txt"))
                hybrid[usable] = proxy[usable] + residual.predict(x.loc[missing].loc[usable], num_threads=2)
            extra = rows.loc[missing].copy()
            extra["row_index"] = np.flatnonzero(missing)
            extra["direct"] = d
            extra["raw_proxy_fallback"] = np.where(usable, proxy, d)
            extra["hybrid"] = hybrid
            oof = pd.concat([oof, extra], ignore_index=True).sort_values("row_index")
            oof.to_parquet(path, index=False)
        metrics[name] = {c: _scores(oof.target.to_numpy(), oof[c].to_numpy(),
                                     oof.airport.to_numpy(), np.isfinite(oof.proxy.to_numpy()))
                         for c in ["direct", "raw_proxy_fallback", "hybrid"]}
    (folder / "validation_all_rows.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
