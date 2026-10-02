# Post-v3 local research

These experiments use labeled 2025 training data and the frozen v3 out-of-fold predictions. They do not use ranking labels or leaderboard feedback to choose model settings. Outputs go under `artifacts/`, which is excluded from Git. Four exploratory candidates were rejected; the fixed GPU residual followed by source correction was selected locally for v4.

The January/July and November/December 2025 labels have been examined repeatedly across experiments. The forward period is therefore no longer an untouched final estimate. The frozen v3 experts also used held-out month labels for some early stopping. Treat all local scores as diagnostic estimates, not guarantees for 2026.

## Reproduce completed experiments

First run the v3 pipeline in [README.md](README.md) to create its feature caches and out-of-fold predictions. Install the optional research dependencies with `python -m pip install -r requirements-research.txt`. From the repository root, run:

```powershell
python diagnostic_v3.py
python analog_diagnostic.py
python id_audit.py
python hierarchical_blend.py
python catboost_expert.py --fold seasonal_jan_jul --sample-max 300000 --iterations 300 --threads 4 --depth 7
python catboost_expert.py --fold seasonal_jan_jul --sample-max 400000 --iterations 500 --threads 4 --depth 7 --with-flight-weather --output-dir artifacts/catboost/full_feature
```

| Experiment | Aggregate local result, RMSE in seconds | Decision |
|---|---|---|
| Airport-specific convex weights (`hierarchical_blend.py`) | All-row January/July nested: v3 329.403, candidate 329.284. November/December with the selected setting unchanged: v3 225.625, candidate 225.668. | Rejected: the small seasonal gain reverses forward. |
| Historical flight analog (`analog_diagnostic.py`) | On 5,366 January/July missing-flight-record departures, the seasonally selected direct analog (at least 2 historical matches, weight 0.25) changes RMSE 1874.390 to 1862.791. The same rule on 2,992 November/December rows changes 1014.492 to 1052.468. | Rejected: forward error increases. |
| Movement-ID neighborhood (`id_audit.py`) | Audited 4,167,797 local movements against 672,428 frozen v3 OOF departures. No validated candidate prediction or RMSE gain was established. | Rejected as a model feature: opaque ID order may reflect ingestion and may not transfer to 2026. |
| Bounded CPU CatBoost residual (`catboost_expert.py`) | On 338,802 January/July valid-proxy rows, v3 is 231.570. A 25% blend is 233.564 with cached features, or 232.719 with flight/weather features. | Rejected: both blends are worse than v3. |

`diagnostic_v3.py` produced aggregate error breakdowns that motivated these experiments. Its target bins and largest-error rows are post-prediction diagnostics and are not model inputs.

## Selected local v4 candidate

The GPU residual expert changes only rows with a valid departure proxy. Its blend weight, 0.25, was selected on January/July. The source classifier estimates the probability of a scheduled-time match for Rome/Istanbul departures whose valid AOBT and schedule proxies disagree by more than 600 seconds. Its correction weight, 0.5, was also selected on January/July. The sequential rule applies the fixed GPU blend first, then the fixed source correction; its combined weights were not optimized on the forward months. Every score below uses **all finite-label rows**, including rows where neither expert changes v3.

| Fixed prediction rule | January/July RMSE, 344,419 rows | November/December RMSE, 328,009 rows | Pooled RMSE, 672,428 rows |
|---|---:|---:|---:|
| Frozen v3 | 329.403 | 225.625 | 283.566 |
| Source correction alone | 329.022 | 225.299 | 283.212 |
| GPU residual alone | 328.664 | 224.837 | 282.820 |
| GPU then source (selected locally for v4) | **328.221** | **224.465** | **282.412** |

Reproduce the expert OOF predictions and final ranking predictions after the v3 pipeline has built its caches and `data/external/weather.parquet`:

```powershell
python catboost_gpu.py --mode fit --iterations 1500 --depth 8 --threads 2
python catboost_gpu.py --mode final-predict --iterations 1500 --depth 8 --threads 2
python catboost_source.py --fold both --iterations 600 --depth 7 --threads 4
python catboost_source.py --evaluate-only
python catboost_source.py --evaluate-sequential-gpu
python catboost_source.py --fit-final-rank
python catboost_source.py --combine-gpu
```

The `source_stability.py` diagnostic reads only IDs, event times, airports, labels, and saved OOF predictions. It does not fit or choose a prediction rule. It compares paired squared errors and resamples UTC calendar days within each held-out month:

```powershell
python source_stability.py --candidate source --repetitions 1000 --seed 2026
python source_stability.py --candidate gpu --repetitions 1000 --seed 2026
python source_stability.py --candidate gpu_then_source --repetitions 1000 --seed 2026
```

| Paired comparison | January/July RMSE gain, 95% day-block interval | November/December RMSE gain, 95% day-block interval | Pooled RMSE gain, 95% day-block interval |
|---|---:|---:|---:|
| Source over v3 | 0.382 [0.050, 0.711] | 0.326 [0.020, 0.670] | 0.354 [0.135, 0.600] |
| GPU over v3 | 0.739 [0.535, 0.988] | 0.789 [0.642, 0.932] | 0.746 [0.595, 0.915] |
| Sequential over GPU | 0.443 [0.095, 0.791] | 0.371 [0.065, 0.718] | 0.408 [0.178, 0.668] |

The sequential-over-GPU gain is positive in 99.6% of the seasonal day resamples and 99.2% of the forward resamples. These intervals describe day-to-day variation in the repeatedly examined 2025 folds, not an untouched estimate of 2026 accuracy. Source gains are concentrated on several dates, and its standalone forward Rome/Istanbul result includes a slight Istanbul regression. The GPU expert improves pooled forward error but worsens Frankfurt and Amsterdam airport-specific forward error. The selected sequential rule has not been validated by a new, untouched year.
