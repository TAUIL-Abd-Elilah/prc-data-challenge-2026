# Post-v3 local research

These experiments use labeled 2025 training data and the frozen v3 out-of-fold predictions. They do not use ranking labels or leaderboard feedback to choose model settings. Outputs go under `artifacts/`, which is excluded from Git. The completed experiments in the table below were rejected; the source-classifier result is provisional.

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

## Pending

`catboost_source.py` has provisional candidate-segment results for a 0.5 probability blend selected on January/July. On 33,591 Rome/Istanbul January/July rows, v3 RMSE is 402.728 and the blend is 399.517 seconds. Applied unchanged to 26,490 November/December rows, v3 is 327.08 and the blend is 324.28 seconds. Full finite-label scoring and final-model training are still running, so no submission decision follows yet. A GPU CatBoost experiment also remains pending. Document the complete prediction path and all-row evaluation before considering another submission.
