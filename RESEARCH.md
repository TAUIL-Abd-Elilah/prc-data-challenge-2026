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

## v5 research

The comparator was corrected to match the submission's nonnegative output policy. Clipping 57 legacy negative v4 OOF values changes pooled RMSE from 282.412062 to 282.408734; it does not change the already submitted v4 file. New experts use internal training splits for early stopping. January/July select coarse blend weights, which are then fixed for November/December assessment.

| Fixed rule | January/July RMSE | November/December RMSE | Pooled RMSE |
|---|---:|---:|---:|
| Clipped frozen v4 | 328.216 | 224.464 | 282.409 |
| 50% larger timestamp residual expert on valid AOBT | 327.366 | 223.227 | 281.423 |
| Then 50% missing-NM direct expert outside Rome | **326.531** | **221.652** | **280.317** |
| Then 25% independently trained ARR-ground residual expert | **325.822** | **221.142** | **279.697** |

Paired UTC-day bootstrap intervals for the combination's RMSE gain over v4 are +0.962 to +2.750 seconds on January/July and +1.883 to +4.086 seconds on November/December (1,000 resamples, seed 20261002). The missing expert improves each held-out month separately; some airport subsets still regress. These repeatedly examined periods remain model comparisons, not untouched estimates.

The independent arrival expert trains a direct taxi-time-minus-AOBT-proxy residual using released flight/weather fields plus 14 ARR-ground features. Every fold excludes its held-out departure labels from fitting and early stopping. Its weight is fixed at the seasonally selected 0.25. The gain after the timestamp/missing combination is +0.709 seconds on January/July (95% paired day-block interval +0.453 to +0.997) and +0.510 on November/December (+0.295 to +0.728). The entire ensemble still inherits the older baseline's validation limitations. This differs from the excluded arrival stack below.

Additional candidates were rejected or withheld:

- A Rome long-schedule ratio expert improves seasonal error but worsens forward error, so it is excluded. The related classifier mixture also fails the forward check.
- A four-class alternate-timestamp classifier (`multisource_expert.py`) compares LOBT, IOBT and schedule candidates. Its best seasonal correction scale is zero; no ranking model is fit.
- An arrival correction stack (`arrival_features.py`) improves both apparent periods, including when added to deep/missing predictions. Its seasonal v4 base experts were trained using November/December labels, however, so fitting the forward correction on seasonal residuals creates an indirect dependence on forward labels. This stack is excluded from submission. The separate direct residual expert in `arrival_residual_expert.py` uses complementary-month training to check arrival features without this path.
- Adding scheduled clock time to the Rome missing-NM classifier/ratio mixture (`rome_clock_expert.py`) selects a 0.25 blend: seasonal RMSE 327.863 and forward RMSE 223.982 against clipped v4. Its day-block gain intervals include zero on both periods (seasonal -1.227 to +1.847 seconds; forward -0.946 to +1.630), so it fails its predeclared stability gate and is excluded. Reproduce with `python rome_clock_expert.py --mode fit --threads 4`.
- A revised valid-AOBT Rome/Istanbul source classifier (`source_clock_expert.py`) adds timestamp precision and schedule-clock features, then tests a correction proportional to the difference from the old source probability. Its seasonally selected scale of 1.0 improves deep/missing seasonal RMSE only from 326.531 to 326.461 and worsens forward RMSE from 221.652 to 221.896. Day-block gain intervals are -0.062 to +0.221 seconds seasonal and -0.668 to -0.017 forward. It is excluded, with no final ranking model. Reproduce with `python source_clock_expert.py --mode fit-oof --fold both --threads 2`, then `python source_clock_expert.py --mode evaluate`.

All arrival ground-truth inputs are restricted to PHASE=ARR. Departure block and taxi labels are never predictors. Arrival taxi/in-block fields are available in the complete supplied ranking batch. No new choice uses official ranking scores or ranking labels.

## Accepted v5 outcome

The fixed deep/missing/clean-arrival ensemble was uploaded once after local policy selection. It was accepted over all 344,841 pairs at official RMSE **283.0418 seconds**, improving v4 by 5.4818 seconds. The team was 74th by distinct lower-best-score teams at 2026-10-02 19:34:16 UTC. These observations were recorded after the policy and file were frozen; they were not used to adjust the submitted weights. The downloaded object matched SHA-256 `4c3f97e3686378609a19a11ee97d5ee4b967c85b2dd0a26083d7c4aba7dcb9f3`. First place has not been achieved.

## Further research under the first-place goal

First place and prize eligibility remain the objective. Ranking responses are progress records only: they do not select features, hyperparameters, gates or blend weights. Every new component is selected on labeled 2025 data. Uploads use fresh version numbers within the official five-per-day and 1 GB bucket limits. The final policy, source and reproducible aggregate validation are frozen before upload.

The larger clock/arrival residual candidate (`deep_arrival_expert.py`) predeclares depth 10, at most 10,000 GPU trees and blends 0, 0.1, 0.25, 0.5, 1. January/July select the blend; November/December check that same choice. Positive paired UTC-day confidence bounds are required in both periods. A further April/October architecture audit refits both the prior depth-9 timestamp expert and the new expert with those months excluded from fitting and early stopping. This additional audit compares the two architectures, rather than estimating the complete ensemble's 2026 performance.

The all-airport missing-NM mixture (`v6_missing_ensemble.py`) combines an all-positive Huber ordinary prediction with long-target probability and a conditional schedule ratio. The seasonally selected 0.5 blend changes January/July all-finite RMSE from 325.822 to 323.817, but worsens November/December from 221.142 to 226.292. The paired day intervals include zero in both periods. It is rejected; no final ranking model is fitted.

A shared non-Rome RMSE expert (`v6_missing_rmse.py`) trains on every finite positive no-NM target, including longer targets omitted by the accepted ordinary expert. Its selected 0.5 blend improves January/July by 0.169 seconds and November/December by 0.089, but both day intervals cross zero. It is also rejected without a ranking fit. Aggregate reports for these two candidates are in `reports/missing_mixture_validation_v6.json` and `reports/missing_rmse_validation_v6.json`.

The aggregate v5 diagnostic (`v6_diagnostic.py`) finds invalid AOBT on 8,764 of 672,428 validation rows, contributing 41.7% of squared error. Same-flight ARR landing minus NM arrival time has negligible association with the departure proxy residual across 102,378 matched valid rows. A Rome same-stand arrival-gap pattern does not repeat across both folds. Neither audit supplies a new submitted correction, and rare individual extreme rows are not fitted with hand-written exceptions.

The neighbor-proxy experiment (`proxy_neighbour_expert.py`) builds 24 congestion features from other departures' released AOBT estimates: count, mean and dispersion, by airport or runway, in strict past/future 15/60 minute windows. The query flight and same-second peers are excluded. These are covariates from the supplied batch; other departure BLOCK/TAXITIME labels never enter them.

Its initial 31-leaf, 800-round LightGBM expert worsens the stronger v5 comparator for every nonzero blend in both periods. Seasonal selection chooses zero; no ranking model is fitted. This rejects that model/blend, rather than isolating the value of the added features. A separately predeclared matched architecture ablation tests the latter. The aggregate comparison is in `reports/neighbour_validation_v6.json`.

The matched January/July ablation trains the same architecture, seed, fit indices and internal early-stop indices with and without the 24 added features. Expert-only valid-AOBT RMSE changes from 239.053 to 237.782 on 338,802 rows, a gain of 1.272 seconds (95% paired UTC-day interval 0.927 to 1.661). This supports testing those covariates in a stronger model; it does not promote the rejected LightGBM blend. The report is `reports/neighbour_ablation_v6.json`.

`runway_arrival_features.py` builds eight additional released-field features: same-runway arrival counts and nearest arrival/departure headways around the supplied movement timestamp. Missing/placeholder runway queries receive missing features. An association audit conditions on airport, runway, month, UTC hour, airport departure density and airport arrival density. Same-runway arrivals retain a positive association with v5 residuals in all four OOF months, with positive day-bootstrap intervals. Effects differ across airports; this is evidence for a predictor, not a causal claim. The report is `reports/runway_arrival_association_v6.json`.

The combined next experiment (`traffic_deep_expert.py`) keeps the depth-10/10,000-tree architecture fixed and adds those 24 neighbor plus eight runway features. Its protocol and v6 OOF reference are frozen before any v6 official result is observed. It has its own seasonal selection, fixed forward check and matched April/October feature audit against the saved depth-10 ARR model. The prepared protocol is `reports/traffic_protocol_v7.json`.

Reproduce the added feature caches and the frozen traffic experiment after completing v6. Run fitting stages serially on the reference machine. The fresh audit rejects a candidate whose two-fold gates fail; final prediction rejects a candidate without every required local gate.

```powershell
python proxy_neighbour_expert.py --mode build-features
python runway_arrival_features.py --mode build
python traffic_deep_expert.py --mode prepare
python traffic_deep_expert.py --mode fit
python traffic_deep_expert.py --mode fresh-audit
python traffic_deep_expert.py --mode final-predict
```

## Accepted v6 outcome

The validated larger clock/arrival blend was finalized and its source published before upload. It was accepted over all 344,841 pairs at official RMSE **280.48 seconds**, with team rank **72nd** at 2026-10-02 22:28:39 UTC. Remote readback matched SHA-256 `3fa41955c243f2821626203b9c33f83b40f71271993f7204acdc6f0294dc161d`. The traffic experiment's architecture, features and selection protocol were already frozen before this outcome. No setting was changed from the official result. First place remains unachieved.

The locally validated v7 traffic blend was finalized and published before its upload. It was accepted over all 344,841 pairs at official RMSE **278.7888 seconds**, with team rank **69th** at 2026-10-02 23:01:39 UTC. Remote readback matched SHA-256 `314e5dcc537b46954744ddc02018c7e2a65cd1dfe10be8e13da313c69eb0b0eb`. The CPU alternative and fixed combination protocols were frozen before this outcome. The team has five submissions in the conservative preceding-24-hour window, so further uploads wait for a slot; local model validation continues.

The separate `movement_only_expert.py` protocol tests transfer from all ordinary 2025 departures to the sparse invalid-AOBT/no-NM/non-Rome gate. Its predictor allowlist removes all NM fields and off-block proxies, while retaining common movement, schedule, traffic, NOAA and released ARR fields. It is a direct taxi-time model rather than a residual-to-NM expert. Preparation does not promote or upload it; the same independent local validation gates are required.

Its first two complementary-fold fits are complete. January/July select full replacement on the predeclared missing-clock gate: all-finite RMSE changes from 325.822 to 324.910 seconds, with gain interval 0.556 to 1.346. November/December at the unchanged weight changes from 221.142 to 219.434, with gain interval 0.488 to 3.314. Aggregate original-fold evidence is in `reports/movement_only_validation_v6.json`. The later audit results are recorded below; no ranking prediction has been created yet.

The same saved ordinary movement models can be assessed on valid-AOBT rows independently. The original `v9_movement_valid_audit.py` protocol relies entirely on already saved models and is preserved. A separate prospective `v9b_movement_valid_audit.py` protocol adds a gated April/October refit if the original missing-clock route cannot produce one. It reuses the unchanged training architecture and requires valid-route improvement with positive day intervals in both original folds, followed by both April/October point gains and a positive pooled day interval. Any final full-data model and ranking prediction require all those gates. Invalid-AOBT predictions remain at the v7 reference. No valid-route selection result has been examined yet.

```powershell
python movement_only_expert.py --mode verify-prepared
python v9b_movement_valid_audit.py --mode prepare
python v9b_movement_valid_audit.py --mode predict-fold --fold seasonal_jan_jul
python v9b_movement_valid_audit.py --mode predict-fold --fold forward_nov_dec
python v9b_movement_valid_audit.py --mode evaluate-folds
```

Use the guarded fresh/final modes only if the preceding gates pass; `python v9b_movement_valid_audit.py --help` lists them. Full source, prospective protocols and input hashes are public before this audit runs.

The shared-flight covariate recovery audit (`v8_flight_covariate_recovery.py`) reads no block/taxi labels and uses flight IDs only as join keys. Missing departure NM clocks have no matching ARR counterpart in either the 2025 training set or the supplied 2026 ranking set. It recovers zero clocks, so this route is rejected. Shared records with available AOBT agree exactly in all 315,234 training pairs and all 50,923 ranking pairs. Reproduce with `python v8_flight_covariate_recovery.py`; the aggregate report is `reports/flight_covariate_recovery_audit_v8.json`.

The fixed CPU alternative (`v8_lightgbm_residual.py`) uses the timestamp, ARR, neighbor and runway covariates with 255 LightGBM leaves, minimum leaf size 80, L2 penalty 30, learning rate 0.03 and at most 2,000 rounds. Early stopping uses complete calendar days inside complementary months. Its frozen comparator is v6; January/July select one coarse weight, November/December apply it unchanged, and a separate April/October paired architecture audit is mandatory. All raw files, caches, weather, references and the template are fingerprinted. Run its CPU stages serially with other training jobs:

```powershell
python v8_lightgbm_residual.py --mode prepare
python v8_lightgbm_residual.py --mode fit
python v8_lightgbm_residual.py --mode fresh-audit
```

The two CPU folds have completed at 912 and 929 trees. January/July select weight 0.25: all-finite RMSE changes from 323.723 to 323.056 seconds, with paired-day gain interval 0.430 to 0.941. November/December at that fixed weight changes from 217.903 to 217.267, with interval 0.439 to 0.854. Both original-fold gates pass. The April/October audit and v7 combination check are pending, so this is not promoted or submitted. The aggregate snapshot is `reports/lightgbm_validation_v8.json`.

The separate `v8_combo_audit.py` protocol is frozen before v8 training. If both standalone candidates pass, it applies the same v8 seasonally selected weight toward the v8 expert from the v7 candidate, without choosing another weight. Promotion requires improvement over v7 and positive day confidence bounds in both OOF folds and the paired April/October audit. A failed combination is rejected without retuning. Reproduce with `python v8_combo_audit.py --mode prepare`, then `python v8_combo_audit.py --mode audit` once both standalone validation reports are promoted.

ARR summaries likewise use the complete released batch and can include an arrival whose in-block time follows the queried departure. This is retrospective challenge prediction, not a claim that every field would be available in a real-time departure forecast.

## Reserved model-change guard

A source/protocol audit found no scored held-out model comparison on February/August or May/September. Those records have already entered complementary-month training, internal early stopping and aggregate data-quality checks, so scoring an existing fitted model on them would be in-sample. February/August are now reserved for one additional fixed model-change guard after the current candidate set completes its existing gates. Both comparator and replacement architectures must be refitted with those months excluded from fitting and early stopping. Features, weights and routes are locked before that guard; a failed replacement is excluded without adjusting it to the guard results. May/September remain reserved. The full prospective rules are in `reports/reserved_guard_protocol.json`.

This additional check evaluates fixed component changes. It does not estimate the complete legacy ensemble independently, and it cannot guarantee a ranking result.

The current candidate portfolio is also fixed before any v8/v9b scored OOF outcome. The valid-AOBT route compares unchanged v7 with the already specified v8 combo and v9b blend using only common-universe January/July RMSE among candidates passing their existing gates. That choice is locked before the reserved guard; failure retains v7 without retuning. The missing-clock movement route is disjoint and requires its own complete gates. Any composition must pass exact-coverage and paired-day checks on both original folds, with every other prediction unchanged. Details are in `reports/current_candidate_policy_protocol.json`.

`reserved_valid_guard.py --mode select-policy` refuses to freeze a choice while either candidate's original folds or necessary fresh audit are pending. It writes the fixed choice to `artifacts/reserved-valid-guard/selected_policy.json`. If that choice is a replacement, `reserved_v7_comparator.py --mode prepare` and `--mode fit` bind the choice and refit the v7 comparator. Prepare the matching route with `reserved_valid_guard.py --mode prepare --route v8_combo` or `--route v9b`, fit only its replacement with `--mode fit-v8` or `--mode fit-movement`, then run `--mode score --route ...`. These resource-heavy stages run serially. A failed guard retains v7.

The missing-clock movement branch independently runs `movement_only_expert.py --mode fresh-audit`, then `--mode reserved-audit` only if the preceding audit passes. Its final modes require both audits and the original two-fold gates. Existing prepared movement caches must first pass `--mode verify-prepared`, which rebuilds the unchanged allowed covariates in a separate directory and proves exact schema, categorical levels, values and ID order before adding an immutable sidecar. The original cache, manifest and model-fold evidence remain intact. Future preparations create that sidecar directly. Ranking inputs are separately sealed before feature reads and rechecked before output.

The older v6-based standalone v8 ranking path is excluded from this portfolio. Its `--mode final-predict` now refuses that path. Only the selected and guarded v7 combination can use `reserved_valid_guard.py --mode final-v8-combo`; v9b's final modes likewise require its selected-route guard. The missing-clock guard has since completed as recorded below. The valid-route guard and new ranking predictions remain pending.

`compose_current_candidate.py` implements the final fixed policy after those routes are terminal. Its prospective specification is `reports/current_composition_spec.json`. `validate` checks the exact common 672,428-row finite-label universe, labels, fold/month/time metadata and routing masks. It verifies that v5 and v7 agree on the missing-clock gate, applies each permitted correction once, and requires positive RMSE gains and paired UTC-day confidence bounds in both original folds. A failed composition is rejected without retuning; an unchanged v7 composition cannot pass a positive-gain gate.

Only after that check passes, `assemble` reads fully guarded component outputs and verifies their model, source, input and validation manifests. It snapshots hashes before ranking reads and rechecks the same hashes before output and the final manifest. It requires all 344,841 template IDs in exact order, finite nonnegative values, and unchanged predictions outside the disjoint gates. Existing outputs are never overwritten. These modes create internal candidate artifacts; finalization, publication and a fresh CLI quota check still precede any upload. Static compilation and peer review passed. Calls with the current pending prerequisites correctly refused without creating artifacts; no candidate OOF evaluation or ranking assembly has run.

Before its first execution, the assembler's provenance check was strengthened to bind both v8 combo OOF outputs, including its April/October paired predictions and the two upstream fresh OOF files, to the accepted combo audit. These hashes remain in the before/after ranking snapshots even when a failed reserved guard retains v7. A synthetic temporary-file check accepted matching hashes and rejected modified combo bytes, modified upstream bytes and a failed audit. Scientific formulas, selection and gates are unchanged.

```powershell
python compose_current_candidate.py --mode contract
python compose_current_candidate.py --mode validate
python compose_current_candidate.py --mode assemble
```

The latter modes refuse pending prerequisites and exclude failed routes according to the fixed policy. They must not be used to bypass the selected-route or reserved-audit sequence.

## Prospective same-runway ARR taxi context

`runway_arrival_taxi_features.py` defines ten fixed fields from released ARR covariates: counts, means and population standard deviations of completed taxi-in intervals in prior 15/60/180-minute windows, plus the number still taxiing on the same airport/runway. Completion windows are anchored to ARR in-block time, with strict upper bounds. Both summaries require finite taxi times in 0..7200 seconds and agreement with in-block minus landing time within one second. Zero taxi intervals can enter completed summaries but cannot count as active. Missing runway queries remain missing.

The raw DEP allowlist contains only movement ID, airport, runway and movement time; DEP block/taxi fields are absent. The deterministic synthetic check verifies phase-specific provenance, window boundaries, active intervals, zero durations and missing runways. The source/raw/cache fingerprint protocol is frozen in `reports/runway_arrival_taxi_protocol_v10.json`. This is a later research family, outside the current v8/v9b portfolio. No February/August labels have been scored for it.

```powershell
python runway_arrival_taxi_features.py --mode synthetic
python runway_arrival_taxi_features.py --mode prepare
python runway_arrival_taxi_features.py --mode build
```

The full build requires at least 4 GiB free memory and should be scheduled after current fitting jobs. The builder itself cannot promote a component or generate a submission.

`v10_runway_taxi_expert.py` now freezes the model specification in `reports/runway_taxi_model_spec_v10.json` before full feature construction. It keeps the v7 depth-10, at most 10,000-tree CatBoost residual trainer, seed and other parameters unchanged, adding exactly the ten declared ARR taxi fields. January/July select one of 0, 0.1, 0.25, 0.5 or 1 against the accepted v7 candidate. November/December use that weight unchanged, with positive paired UTC-day confidence bounds required in both folds. A complementary April/October refit then compares the same fixed blend with the saved, independently refitted v7 candidate; both months must improve and the pooled day interval must be positive.

Preparation binds all twelve training files, source, feature caches, weather, exact reference IDs/labels/times and saved comparator models by hash. Loading the ten-field cache requires exact baseline ID order and the ARR-only provenance manifest. Static compilation and peer review passed. The fixed builder has now completed 2,085,047 training and 344,841 ranking departure rows with all ten fields and exact baseline ID order. Output schemas, counts and SHA-256 values were independently checked against `reports/runway_arrival_taxi_build_v10.json`. The training-cache hash is `7a205a67989874b1496020cc30942854dca44b01b7f16110466dc1a8958eca98`; the ranking-cache hash is `81eefe1a470b561d650baa98813a19e4089508fd5d21801ac5b0d4b30960eb6a`. No departure labels entered construction. Model preparation, fitting and scored comparison remain pending. Final fitting and ranking modes reject until a later guard is frozen; February/August do not select this family, and May/September remain reserved.

```powershell
python v10_runway_taxi_expert.py --mode show-spec
python v10_runway_taxi_expert.py --mode prepare
python v10_runway_taxi_expert.py --mode fit-folds
python v10_runway_taxi_expert.py --mode fresh-audit
```

### Saved movement fold provenance

`record_movement_fold_provenance.py` records the existing movement models, fit reports, original OOF files, feature cache, raw/source inputs and published original validation before any new valid-AOBT inference. It reads model metadata and hashes, without prediction or scoring. The current immutable receipt is `reports/movement_fold_model_provenance_v2.json`; both model hashes and cache hashes match the prior audit. V2 stores the recorder source path relative to the repository for portability. The earlier receipt remains historical evidence. Reproduction can generate a new receipt from its own fitted models using the same recorder and a fresh `--receipt` path. The receipt does not promote either movement route or replace the required feature-cache proof and later validation gates.

```powershell
python record_movement_fold_provenance.py
```

The optional operational helpers preserve the frozen feature and model policy while avoiding full pandas matrices. `movement_prepared_stream_verify.py` independently derives the global category vocabulary, rebuilds all 79 columns in bounded batches, and proves exact values, category levels/order, Arrow schema metadata, IDs and manifest before writing the original cache's integrity sidecar. It does not rewrite the original cache. `v9b_stream_predict.py` then uses that sidecar and the published fold receipt, the same held-out valid-AOBT mask, saved rounds and three prediction threads. Its output must pass the existing `v9b.verify_prediction` checks before the unchanged evaluator can select a weight. The synthetic categorical test checks bitwise equality of full and uneven-batch predictions. Neither helper trains competition models or reads ranking data. The original training memory gates remain 10 GiB.

```powershell
python movement_prepared_stream_verify.py --mode preview
python movement_prepared_stream_verify.py --mode verify
python v9b_stream_predict.py --mode parity-test
python v9b_stream_predict.py --mode predict-fold --fold seasonal_jan_jul
python v9b_stream_predict.py --mode predict-fold --fold forward_nov_dec
python v9b_movement_valid_audit.py --mode evaluate-folds
```

Streaming verification defaults to 3.5 GiB available initially, and saved inference to 4 GiB. Each has a runtime memory floor and refuses to overwrite output. For independent reproduction, freeze a new receipt with the recorder's `--receipt` option and pass its SHA-256 through `v9b_stream_predict.py --receipt ... --receipt-sha256 ...`. The current run uses the published v2 receipt. Complete monitored cache verification passed as documented below; valid-AOBT prediction outcomes remain pending.

`lightgbm_stream_feasibility.py` tests a separate disk-backed training input on synthetic data only. Bounded conversion through the installed LightGBM pandas encoder retains global categorical coding and writes a C-contiguous float64 NumPy memmap. The default-parameter test and a synthetic bin-sampling stress test both match full pandas encoded values, complete model text and predictions exactly; see `reports/lightgbm_stream_feasibility.json`. This establishes an operational hypothesis, not a competition fit or score. Native binning memory and full-scale equivalence remain untested, so the production 10 GiB gate is unchanged.

```powershell
python lightgbm_stream_feasibility.py
python lightgbm_stream_feasibility.py --synthetic-bin-sample-count 1000
```

Further hypotheses from primary research are deferred until prospective local tests can be frozen. The [Simaiakis–Balakrishnan airport congestion paper](https://web.mit.edu/hamsa/www/pubs/SimaiakisBalakrishnan_TS2014.pdf) motivates counting other proxy pushbacks during a target's supplied taxi interval and estimating low-traffic taxi priors. The [MIT runway queue study](https://dspace.mit.edu/entities/publication/f0891972-627a-409e-9fbc-2e12fe4ba28b) motivates recent airport-wide runway configuration summaries. These methods could be implemented originally from released timestamps and categories; a target-based prior would require fold-only fitting and date cross-fitting. No feature cache, model, selection or ranking prediction uses these ideas yet.

The verifier's operational memory audit found no intended allocation above 1 GiB: categorical vocabularies are read one column at a time, the raw flight-name vocabulary is streamed, weather is 0.8 MiB on disk, and full ID uniqueness uses a 16 MiB vector. A monitored verification may therefore use the configurable 2.5 GiB initial / 1.5 GiB runtime gate with 32k batches. `run_bounded_worker.py` separately watches only its directly spawned PRC process and terminates it if a 0.5-second sample finds available RAM below 1.5 GiB or its resident working set above 2 GiB. It records observed memory and exit status; unsuccessful or incomplete workers are not accepted. The sampled peak does not capture every brief allocation. The scientific cache equivalence checks and all original full-matrix model-fit gates remain unchanged. Metadata-preview and deliberate low-RSS abort checks passed, including verifying the unrelated job remained alive; nonfinite limit arguments are rejected before launch.

```powershell
python run_bounded_worker.py --worker movement_prepared_stream_verify.py --report artifacts/v6-movement-only/stream_worker_memory.json -- --mode verify --min-free-gib 2.5 --abort-free-gib 1.5
```

The first monitored full-cache attempt failed before sealing at the initial batch. Its sampled peak RSS was 0.579 GiB and minimum available RAM was 2.449 GiB. Diagnosis found no logical flight-category differences: pre-write Pandas `string` versus Parquet-readback `str` category representation caused strict Series equality to fail. The independent temporary writer now derives categorical UTF-8 storage from the baseline source schema and all twelve canonical raw flight-name schemas, rather than from the original prepared cache. Category labels, order and codes must match before writing; the original exact physical Arrow schema/metadata, reread values and categorical checks remain mandatory. A 1,024-row physical round-trip check passed; the complete rerun is still required.

The repaired complete rerun passed all 2,085,047 departures and 79 fields. Exact schema/metadata, categorical levels/order, physical reread values, IDs and manifest match. The original prepared cache retains SHA-256 `454c9c8c9220564bd9b208c666143056725541683e05da774a479f55e6083a5f`. The immutable seal is copied to `reports/movement_prepared_integrity.json`; the successful watchdog report is copied to `reports/movement_stream_memory.json`. Sampled peak RSS was 0.934 GiB and minimum available memory 2.144 GiB over 51.6 seconds. The worker exited successfully and both source hashes were checked independently.

The saved-model streaming producer also accepts `--min-free-gib 2.5` for monitored runs while retaining its 4 GiB default. It enforces a 1.5 GiB runtime floor. This changes resource handling only; it preserves the same source receipts, model, features, masks, rounds, predictions and downstream validation checks.

`movement_memmap_fit.py` implements the provisional disk-backed native fit for an operational equivalence test only. It requires the independent prepared seal, a 3 GiB initial gate and a successful externally monitored run. Target dtype, 0..7200 ordinary range, UTC-day modulo-11 internal split, feature/category order, original 63-leaf parameters, 1200-round cap and 100-round stopping rule are unchanged. Only seasonal Jan/Jul fitting is enabled; all other month pairs refuse until separate scientific integration. Its model remains provisional until `verify-run` binds the watchdog, source, masks, matrices and model, and `verify-equivalence` proves byte-identical complete model output against the published original-model receipt. Its batch-encoder and native synthetic model parity checks passed. No full-scale memmap fit has run yet.

```powershell
python movement_memmap_fit.py --mode synthetic-parity
python run_bounded_worker.py --worker movement_memmap_fit.py --report artifacts/movement-memmap/seasonal-watchdog.json -- --mode fit-fold --fold seasonal_jan_jul --run-dir artifacts/movement-memmap/seasonal --watchdog-report artifacts/movement-memmap/seasonal-watchdog.json
python movement_memmap_fit.py --mode verify-run --fold seasonal_jan_jul --run-dir artifacts/movement-memmap/seasonal --watchdog-report artifacts/movement-memmap/seasonal-watchdog.json
python movement_memmap_fit.py --mode verify-equivalence --fold seasonal_jan_jul --run-dir artifacts/movement-memmap/seasonal --watchdog-report artifacts/movement-memmap/seasonal-watchdog.json
```

The frozen v9b saved-model audit has completed. Exact downstream coverage checks passed on 338,802 valid-AOBT January/July rows and 324,862 November/December rows, with all 672,428 finite-label rows retained for evaluation. January/July selected weight **0**: v7 RMSE 322.310529 seconds versus 322.843436 at the smallest nonzero weight 0.1. The unchanged November/December comparison was 215.198602 versus 216.169931 at 0.1; larger weights were also worse in both periods. Both required gates fail, so this valid-AOBT route is excluded without fresh fitting, retuning or ranking prediction. The aggregate is `reports/movement_valid_validation_v9b.json`; saved-model input, output and memory receipts are published alongside it. This does not alter the separately frozen missing-clock movement route, which passed its original gates on a disjoint mask.

A synthetic alternative using `LightGBM.Sequence` was also rejected as an equivalent operational fit. Its encoded data and categories matched, but when bin sampling was active the model text, trees and predictions differed from the native pandas route (maximum prediction difference 0.216811 on the synthetic fixture). The unsampled case matched exactly. Source and negative results are `lightgbm_sequence_feasibility.py` and `reports/lightgbm_sequence_feasibility.json`. No competition model uses this route. Available RAM has now cleared the original 10 GiB gate; the existing full-matrix April/October missing-clock audit has started, with a reserved audit conditional on success. Larger fits remain serial.

The latter three stages require the completed feature cache and at least 10 GiB free memory. Run them serially after current fitting and audit jobs.

OpenStreetMap geometry was considered only for a read-only extraction audit. No maps were obtained, matching coverage is unknown, and no OSM data enter a model or submission. ODbL is an open-data license; the challenge's external-data wording says open-source license without naming accepted data licenses. Geometry is excluded while that interpretation remains unresolved. No OurAirports or OpenAP data are used.

### Missing-clock audits completed, October 3

The original April/October and February/August missing-clock architecture audits both completed successfully, with the previously selected weight **1.0** unchanged. Each refits the movement expert and the ordinary no-NM CatBoost comparator while excluding its audit months from fitting and internal early stopping. Exact finite-label gate coverage, prepared-cache provenance, prediction hashes and saved-model hashes passed the original verifiers.

| Audit | Finite gate rows | First month RMSE, reference → candidate | Second month RMSE, reference → candidate | Pooled UTC-day gain, 95% interval |
|---|---:|---:|---:|---:|
| April/October | 2,813 | 371.391 → 331.916 s | 438.152 → 418.021 s | 26.147 s [7.556, 44.846] |
| February/August | 3,185 | 1720.756 → 1654.592 s | 374.906 → 364.871 s | 38.745 s [22.792, 55.360] |

The exact aggregate copies are `reports/movement_only_fresh_audit.json` (SHA-256 `083fbc90551c46ec7084c9767b7710bb5764a0b478cf3743f9977d8d36ec0be0`) and `reports/movement_only_reserved_audit.json` (`5839fbdbd11f19e9d4d4c5ce349052caa1eb0289568eab1575c07d141d384132`). These are component-change checks and do not estimate the complete ensemble's unseen score. May/September remain reserved for later research. The missing-clock route is eligible for final fitting under the fixed current portfolio. The final round count remains **931**, the integer median of the original 876/987 complementary-fold rounds; the later audit rounds do not change it.

The v8 April/October audit is running next with its original 0.25 weight. Its v7 combination and the portfolio's valid-route choice remain pending. Full fitting, ranking output, combined OOF verification and submission are still required; no new file has been uploaded.

```powershell
python movement_only_expert.py --mode fresh-audit
python movement_only_expert.py --mode reserved-audit
```

### Terminal v8 combination decision

The original LightGBM April/October audit passed at fixed weight 0.25 on all 357,813 eligible rows: paired RMSE 198.491222 → 196.892042 seconds, with UTC-day gain interval [1.296352, 1.883122]. The original complementary-fold gates remain positive. Exact aggregate copies are `reports/lightgbm_fresh_audit_v8.json` and `reports/lightgbm_validation_v8_complete.json`.

The separately frozen v7 combination was then evaluated without choosing another weight. January/July improves 322.310529 → 321.858278 seconds with gain interval [0.218506, 0.730483]. April/October improves 195.888222 → 195.264389 with interval [0.366230, 0.890854]. November/December improves only 215.198602 → 215.174044, and its interval [-0.152834, 0.211483] crosses zero. The mandatory combination gate therefore fails; the LightGBM replacement is excluded from this portfolio without retuning. The full aggregate is `reports/combination_validation_v8.json`. Together with v9b's failed original folds, this leaves v7 as the eligible valid-AOBT policy.

The selector initially refused because its v7 master was aligned to chronological baseline order while the saved expert OOFs were concatenated by fold. All 672,428 unique IDs, targets, folds, months, gates and UTC timestamps were exactly equal after alignment by ID. Its operational fix now proves the complete unique ID set, aligns the expert to the master with a one-to-one join, and then performs every existing metadata and unchanged-row check. It does not rewrite OOF files or change scores, weights or eligibility. Independent synthetic checks reject duplicate, missing, extra and null IDs. The repair is published before the selection receipt is created.

The repaired selector has frozen **v7** for the valid-AOBT route; its exact receipt is `reports/current_selected_policy.json`. No valid-route February/August fit is needed because no replacement passed its required gates. The independent missing-clock correction retains its previously selected weight 1.0.

The final composed 2025 OOF passed on the exact 672,428-row common universe before ranking prediction. January/July RMSE changes 322.310529 → 321.389265 seconds with UTC-day gain interval [0.510029, 1.328221]; November/December changes 215.198602 → 213.443318 with interval [0.502394, 3.396335]. All 7,841 missing-clock OOF rows are covered, the two masks are disjoint, v5 and v7 match exactly on that gate, and all other predictions remain unchanged. The exact aggregate is `reports/current_composition_validation.json`. The original full-data movement fit is now running at 931 rounds. Ranking generation, assembly, finalization and quota clearance remain pending.

```powershell
python reserved_valid_guard.py --mode select-policy
python compose_current_candidate.py --mode validate
python movement_only_expert.py --mode fit-final
```

### Prospective departure interval-flow experiment

`taxi_interval_flow_features.py` defines ten original numeric features inspired by the [MIT taxi queue study](https://dspace.mit.edu/entities/publication/f0891972-627a-409e-9fbc-2e12fe4ba28b). For both airport and airport/runway scopes, it counts other takeoffs and quality-valid proxy starts strictly between the query's released NM AOBT and takeoff; valid proxy intervals active at takeoff; intervals starting later but finishing earlier; and intervals starting earlier but finishing later. These are retrospective proxy relationships, not measured runway queues. No paper code or data are reused.

The DEP phase filter precedes the strict ID, airport, runway, MVT and AOBT projection. Departure block/taxi labels are never read. IDs only align rows and prove uniqueness. Each proxy interval must be nonnegative and at most 7,200 seconds. Strict inequalities exclude the query and define timestamp ties. Missing groups and invalid queries produce NaNs. The builder freezes all raw/baseline/source hashes, stages output, rechecks inputs and publishes exclusively. The full build has a 4 GiB launch floor and O(n log n) sweeps. Synthetic brute-force checks cover 18 hand-built cases, 160 random rows and six extreme timestamp cases; peer review found and repaired integer overflow and out-of-range date conversion before any data construction. The fixed specification is `reports/taxi_interval_flow_spec_v11.json`.

`v11_taxi_interval_flow_expert.py` preserves the v7 residual trainer, adding exactly those ten fields. It uses the same coarse January/July weight choice, unchanged November/December check and independently refitted April/October gate as v10, with its own fixed bootstrap seed. Its prospective specification is `reports/taxi_flow_model_spec_v11.json`. Both v10 and v11 now verify fixed trainer parameters and all transitive feature sources, recheck source hashes around each fit, and require main/fresh fold provenance receipts covering exact feature schema, categories, model settings, tree counts and output hashes. Post-fit large frames are released before evaluation. The existing v10 scientific specification is unchanged. Neither new family has been fitted or scored; final and ranking modes still refuse.

The later choice between v10, v11 and the unchanged current policy is frozen before their model comparisons in `reports/later_feature_portfolio_protocol.json`. Original and April/October gates plus exact fixed composition checks precede selection; the selected route is then locked before one May/September paired refit guard. Guard failure retains the current policy without alternative switching or retuning. These are component checks after repeated 2025 exploration, and no leaderboard outcome selects a model.

```powershell
python taxi_interval_flow_features.py synthetic
python taxi_interval_flow_features.py prepare
python taxi_interval_flow_features.py build
python v11_taxi_interval_flow_expert.py --mode prepare
python v11_taxi_interval_flow_expert.py --mode fit-folds
python v11_taxi_interval_flow_expert.py --mode fresh-audit
```

Run full builds and fitting stages serially. The later portfolio's guard and final implementation remain required before a later candidate can be submitted.


### Version 8 finalized, quota pending

The 931-round movement-only final fit completed on 2,084,094 eligible ordinary 2025 rows. Its model SHA-256 is `a05e1e062d2598953ef22404524cff482c76ab413536054861658ce81065bc32`; `reports/movement_only_final_model.json` is the exact fit receipt. The historical `ranking_prediction_created: false` field describes the moment that fit receipt was written. Later ranking generation and assembly completed and are separately recorded in `reports/movement_only_ranking_manifest.json`, `reports/movement_only_ranking_inputs.json`, `reports/current_candidate_ranking_manifest.json` and `reports/current_candidate_ranking_sources.json`.

`merry-mushroom_v8.parquet` contains all 344,841 template rows in original order, with finite nonnegative predictions. Exactly 4,907 missing-clock rows change from accepted v7; every other prediction is bitwise unchanged. The finalized SHA-256 is `8fc6519610a573dd77a4b5ca18f49ab816a564754db13d26d91f26b53d04b9e5` and size is 4,707,529 bytes; the exact seal is `reports/submission_v8_finalized_manifest.json`. The failed LightGBM combination is absent from this file.

No version 8 upload or official score exists yet. At 2026-10-03 02:24 UTC the conservative preceding-24-hour check found five submissions and refused upload (`reports/submission_quota_v8.json`). Its next count slot is after 13:46:52.428 UTC; a fresh quota and bucket check is required before upload. Best accepted v7 remains RMSE 278.7888 seconds, last verified at rank 69 on October 2 at 23:01 UTC. Ranking feedback has not selected any model or weight.


The prospective v11 departure-flow cache is now built without reading departure block/taxi labels. Independent readback confirms exact baseline/raw departure IDs and original order for all 2,085,047 training and 344,841 ranking rows; ID plus ten float32 fields; no infinite or negative counts; and exact all-NaN invalid-query masks (23,501 training and 5,464 ranking). Frozen source and input hashes matched before and after verification. Exact aggregates are `reports/taxi_interval_flow_protocol_v11.json` and `reports/taxi_interval_flow_build_v11.json`. Training output SHA-256 is `093cdd432177de02ce798e30bdc071536ed6c735dbd783c5638dc717c05d4ab4`; ranking output is `c17f204f581b6a9718eeac1b798ff6dea422b3e09f02bc2624385d14d04b89c4`. No v11 model has been fitted or scored.

The fixed v10 model inputs and complementary-fold reference are sealed in `reports/runway_taxi_model_protocol_v10.json` (SHA-256 `b83783bf2c58403403de9c7edc65ab91d47d68874a08fbda796212c2abf34cfd`). Its original complementary validation fitting has started. All scientific settings remain those of the prospective v10 model specification; the new cache build did not select them. Later portfolio selection binds the complete current baseline and both candidate specifications and sources before their model outcomes.


### V10 original gates passed

The same-runway arrival-taxi candidate passed both original all-finite gates at January/July's selected weight **0.25**. January/July RMSE is 322.310529 → 322.128076 seconds, with paired UTC-day gain interval [0.123041, 0.264010]. November/December retains 0.25 and improves 215.198602 → 214.700432, interval [0.388600, 0.621016]. The original tree counts are 9,999 and 9,997. Exact original-stage evidence is `reports/runway_taxi_original_validation_v10.json` and the two `runway_taxi_*_provenance_v10.json` receipts. Their pending fresh flag describes that snapshot. The April/October audit is running next, and this candidate remains unauthorized for ranking.

A repository audit found Git's automatic line-ending conversion made some published JSON bytes differ from the original recorded hashes. `.gitattributes` now preserves Python/JSON bytes, and all affected tracked files were republished with the exact original bytes. An independent normalized-content check proved no source algorithm, specification, record value or working file changed. The existing frozen hashes remain authentic; see README's independent-reproduction guidance.


### Executable later selection and reserved guard

`later_feature_portfolio.py` implements the already published v10/v11/current compatibility policy. It verifies original/fresh source and model receipts and recomputes their fixed scores from saved OOF before comparing the two eligible compositions with the current policy. All 25 published baseline/spec/source bindings and every original candidate's transitive input/source/raw-file hash remain sealed through the decision, including rejected routes. It preserves the missing-clock correction and selects only after both families have terminal evidence. Its source SHA-256 is `f0dde94e95479b0e0dae00d9d4b37b3437bcc7c8804f4dfd912cba2f78568a0d`.

`later_reserved_guard.py` freezes the selected family/weight before May/September scoring and binds feature-cache inputs back to the selected family's original protocol. It runs serial matched v7 and selected-family refits, excluding May/September from both training and internal early stopping, then checks each reserved month and the pooled paired UTC-day interval for the single fixed blend. Guard failure retains the current policy. Its source SHA-256 is `3e655cafb47286a4dd5004f7d678d5f197c3336ab5734ec445617f18e29247d8`. Compile, synthetic ID/mask/formula/source-tamper checks and independent static review passed. Neither executable has yet compared real candidates or scored May/September.

After both original/fresh candidate decisions are terminal, and after publishing the selection receipt before the reserved fits:

```powershell
python later_feature_portfolio.py --mode evaluate-select --expected-source-sha256 f0dde94e95479b0e0dae00d9d4b37b3437bcc7c8804f4dfd912cba2f78568a0d
python later_reserved_guard.py --mode freeze
python later_reserved_guard.py --mode fit-comparator
python later_reserved_guard.py --mode fit-replacement
python later_reserved_guard.py --mode evaluate
```

The three fit/evaluate commands are conditional on selecting a replacement; unchanged current selection creates its terminal receipt without a reserved fit or score. Full jobs remain serial with a 10 GiB initial memory gate. Final fitting/ranking additionally requires the separately implemented, reviewed and published final-stage source and a passing selected replacement guard.


### V10 matched audit passed; v11 fitting

The unchanged 0.25 same-runway arrival-taxi blend passed April/October on all 357,813 finite-label valid-AOBT rows. April RMSE improves 186.344151 → 185.882733 seconds and October 204.527379 → 203.983989; the pooled UTC-day gain interval is [0.428200, 0.590764]. Its independently refitted model used 10,000 trees, which does not alter the original-fold median rule. Independent read-only verification recomputed all original weight scores, selected-weight day intervals, fresh formulas and scores, and checked actual model metadata plus 69 source/model/cache/report fingerprints. Exact snapshots are `reports/runway_taxi_fresh_audit_v10.json`, `reports/runway_taxi_complete_validation_v10.json` and `reports/runway_taxi_fresh_provenance_v10.json`. This remains a component architecture comparison, with no ranking authorization.

The v11 model input protocol is now sealed in `reports/taxi_flow_model_protocol_v11.json`, and its original complementary-fold fits have started with the published settings. The later compatibility/choice waits for both families to be terminal. No May/September score, replacement final fit or new submission has occurred.


`later_feature_final.py` completes the guarded final extension; source SHA-256 `b5d0c9d49e293b16f1c42cb7dce5d12a3429e27e29713330e6aa96b630852fad`. It verifies the selected family's original inputs and the reserved model/OOF/fit receipts, then replays the same single fixed reserved formula and score as an integrity check. Only a passing selected replacement may train all eligible 2025 rows, with exactly 184 fields and the original fold median tree count. It seals ranking sources before reading features, predicts the exact valid-proxy mask, uses the frozen v7 blend formula, preserves current v8 on every other row, and checks complete template order, finite nonnegative output and readback. Current-policy selection verifies the existing file and creates no new fit. Source, syntax, API, synthetic mask/formula/ID/receipt-tamper checks and independent static review passed; real final execution has not occurred.

Only after the selected reserved guard passes, publish its aggregate receipt before final preparation:

```powershell
python later_feature_final.py --mode prepare
python later_feature_final.py --mode fit-final
python later_feature_final.py --mode freeze-ranking-inputs
python later_feature_final.py --mode final-predict
```

The extension produces an internal artifact only. Finalization, public reproduction records, a fresh increasing-version/day/size quota check, MinIO CLI upload and remote readback/acceptance checks remain separate mandatory steps.

### V11 original gates passed

The departure interval-flow candidate passed both original all-finite gates at January/July's selected weight **0.5**. January/July RMSE is 322.310529 → 321.916357 seconds, with paired UTC-day gain interval [0.248558, 0.600218]. November/December retains 0.5 and improves 215.198602 → 214.459611, interval [0.544528, 0.963144]. Both original models used 10,000 trees. Exact original-stage evidence is `reports/taxi_flow_original_validation_v11.json`, `reports/taxi_flow_seasonal_provenance_v11.json` and `reports/taxi_flow_forward_provenance_v11.json`. Their pending fresh flag describes this snapshot. The independently refitted April/October audit is running next; no compatibility selection or May/September score has occurred, and the candidate remains unauthorized for ranking.

The unchanged 0.5 departure-flow blend subsequently passed the matched April/October audit on all 357,813 finite-label valid-AOBT rows. April improves 186.344151 → 185.676904 seconds and October 204.527379 → 203.561518. Pooled RMSE improves 195.888222 → 195.061692, with paired UTC-day gain interval [0.592227, 1.082443]. The separate audit fit used 10,000 trees. Exact complete evidence is `reports/taxi_flow_fresh_audit_v11.json`, `reports/taxi_flow_complete_validation_v11.json` and `reports/taxi_flow_fresh_provenance_v11.json`. Both candidate families now have terminal original/fresh evidence; their fixed current-policy compatibility comparison and the selected route's May/September guard remain outstanding.

### Later portfolio locked before May/September

The published selector verified both families' terminal source/model/OOF evidence, then executed the previously frozen compatibility formula without choosing new weights. Both alternatives passed against the complete current v8 policy over the exact 672,428 finite-label OOF rows. V10 at 0.25 gives January/July RMSE 321.206289 and November/December 212.941042. V11 at 0.5 gives 320.993962 and 212.698229, respectively, versus current 321.389265 and 213.443318. V11's paired UTC-day gain intervals are [0.241627, 0.586704] and [0.550572, 0.968144]. The frozen minimum January/July rule therefore locks **v11, weight 0.5**. All invalid-proxy and missing-clock predictions remain exactly current. Independent v11 original/fresh verification passed across 67 tracked inputs and receipts.

Exact decision evidence is `reports/later_feature_portfolio_evaluation.json` (SHA-256 `cc476112287ddc93a2fd85a2574e5beb19a48bd2be9108f54f61c39a4f09f3f5`) and `reports/later_feature_portfolio_selection.json`; the sealed OOF prediction hash is `2626c43410bc6c03c8cc3a91855bd9c8dbe3b117bb181903bf4afd6378a466d6`. May/September have not been scored. That single selected route must now pass its separately frozen matched component guard; failure retains current without switching to v10. Ranking feedback remains outside selection and ranking generation remains unauthorized.

Independent read-only selection verification passed all decision hashes and confirmed exact unique coverage plus equality of all 8,764 nonvalid current predictions. The selected v11/0.5 May/September protocol is now sealed in `reports/later_reserved_guard_protocol.json` before either reserved fit or score. It binds the selected original family input protocol, source and raw/cache hashes, 174-field comparator and 184-field replacement schemas, fixed parameters and identical month exclusion policy. Full reserved jobs run serially; ranking generation remains unauthorized until this fixed component guard passes.

### Locked May/September component guard passed

The matched v7 comparator and selected v11 replacement completed with identical fit, early-stop and 364,846 held-out IDs and categorical vocabularies. May/September were excluded from fitting and internal early stopping. The single frozen 0.5 blend improves May RMSE 192.182053 → 191.760338 seconds (183,518 rows), September 205.207898 → 204.994186 (181,328), and pooled 198.762612 → 198.447886. Its 61-day paired bootstrap gain interval is [0.170071, 0.439567] using the prospectively fixed seed and 1,000 repeats. The guard passes and retains v11 without alternative switching or parameter changes.

Exact evidence is `reports/later_reserved_guard_terminal.json`, `reports/later_reserved_comparator_fit.json`, `reports/later_reserved_comparator_receipt.json`, `reports/later_reserved_replacement_fit.json` and `reports/later_reserved_replacement_receipt.json`. The paired predictions SHA-256 is `3184ee279b57ef50531cd3bf3afb64d201540fe1d25c04d460d81a92c547e713`. The reserved trees (9,991 and 9,999) do not select the full-fit count: the final v11 architecture uses exactly the original two-fold median, **10,000 trees**. This is a matched component guard after repeated 2025 exploration, not a claim of an unbiased complete-ensemble estimate. Final preparation, fitting and ranking generation are still pending; no new upload or official score occurred.

The selected full fit subsequently completed on 2,061,428 eligible departures using exactly 184 fields and 10,000 trees. Its model SHA-256 is `dccda016515f0af4b8941e2dcce1b2915ee1f446487a176feea0edaacf9b87bd`. The immutable fit-input protocol, full-model receipt and pre-feature-read ranking seal are published as `reports/later_feature_final_protocol.json`, `reports/later_feature_final_model.json` and `reports/later_feature_ranking_inputs.json`. Ranking generation is running from those seals; v9 remains unfinalized and unsubmitted at this stage. V8 will be submitted before v9, with a fresh quota check for each, preserving consecutive upload versions.

Ranking generation and distinct v9 finalization subsequently completed. `merry-mushroom_v9.parquet` contains all 344,841 template IDs in original order, finite nonnegative predictions and unchanged current values outside the 339,377 valid-AOBT mask. Its SHA-256 is `bc465ae7ff48deac5f93cd449a3799fee1a361a8021ec2ca03ff70baccc0f417`, size 4,707,529 bytes. The two over-24-hour predictions are retained from current v8; no ranking-value-driven exception was introduced. Exact prediction/input/model evidence is `reports/later_feature_ranking_manifest.json` and `reports/submission_v9_finalized_manifest.json`. Neither v8 nor v9 has been uploaded or officially scored.

A fresh MinIO CLI quota check at 2026-10-03 04:10:06 UTC again found five uploads in the preceding 24 hours and refused v8 upload. `reports/submission_quota_pending_v8_20261003_0410.json` records the 23,540,059-byte bucket and earliest count slot after 13:46:52.428 UTC. The queued order remains v8 then v9, with a fresh count/size/version check immediately before each upload, source publication and complete remote readback/acceptance verification. Best accepted v7 remains the last verified official RMSE 278.7888 seconds, rank 69 at the October 2 23:01 UTC snapshot. First place and a prize have not been achieved.

### Prospective historical runway geometry

A separate, original 16-field family is now specified and implemented in `runway_geometry_features.py` (SHA-256 `2e4429306cfabad8c6b4ee7990718e4350fb952daf08c325367a48ff381c1059`), with exact specification `reports/runway_geometry_feature_spec.json`. No real feature cache or model fit has run. This family is absent from v8 and the current v9 pipeline.

[OurAirports](https://ourairports.com/data/) releases its data to the public domain, and pinned commit `59574226b6df9417f5a7a8d03eeb6c7158a76168` includes the Unlicense. That 2024-12-30 snapshot predates both competition periods. Exact download URLs, license and file hashes are in `reports/runway_geometry_source_receipt.json`. Independent upstream file/license verification passed. A movement-column-only audit maps complete endpoint geometry for 2,084,589/2,085,047 2025 departures and 344,747/344,841 ranking departures. Unmatched helicopter/other codes remain unmapped; current closed status is excluded. Later runway changes, incomplete metadata and line-segment geometry remain limitations. These features do not establish actual taxiway crossings.

The fixed family contains true directional heading sine/cosine, strip length, true headwind/positive tailwind/absolute crosswind, counts of other intersecting/nearby parallel strips, and eight past 15/60-minute ARR/DEP counts on those strip groups. Nearby parallel means nonintersecting, unoriented bearing difference at most 15 degrees and segment distance at most 2,000 m. Exact reciprocal ends share one physical strip and are excluded from their own traffic groups. All thresholds are prospective. Unknown geometry gives all-NaN features; unknown movement time gives missing traffic/wind, preserving static geometry. NOAA wind uses exact airport/UTC-hour keys and documented m/s and true wind-from degrees. The earlier wind model already has runway-number-based wind components; this proposes a physical refinement and other-runway traffic grouping, with no measured gain claimed.

The builder selects only phase, alignment ID, airport, runway and MVT, phase first, and only four weather covariates. It never selects departure BLOCK/TAXI values or ranking labels. A minimum 4 GiB free-memory floor, immutable source/raw/cache/runtime protocol, exclusive output publication and exact ID/order/float32 readback are required. Compilation, independent static review and synthetic endpoint/intersection/parallel, weather sign, missingness, extreme/nanosecond timestamps and 90-row brute-force traffic parity passed. No paper code or data was reused; [MIT taxi research](https://web.mit.edu/hamsa/www/pubs/SimaiakisBalakrishnanTRR.pdf) is methodological background only.

To rebuild this prospective cache from the existing baseline/weather inputs in a clean output directory, download the three pinned files to `artifacts/prospective-runway-geometry` using the exact source-receipt URLs and verify their hashes. Copy the public source receipt there as `source_receipt.json` and the public specification as `feature_spec.json`. Then run serially:

```powershell
python runway_geometry_features.py --mode self-test
python runway_geometry_features.py --mode prepare
python runway_geometry_features.py --mode build
```

Publish the prepared input protocol before a real build. Any future model comparison requires its own prospective architecture, frozen baseline/weight policy and matched component audits. Repeatedly inspected 2025 labels do not provide an untouched generalization estimate; public leaderboard feedback cannot choose this family's features, settings or promotion.

The label-free geometry build subsequently completed from the previously published input protocol (`reports/runway_geometry_feature_protocol.json`, SHA-256 `a901a9d442ff01b0f8e1c5931cd1ea56c4246f89bfc12d4b8f35ed966089d502`). Exact aggregate evidence is `reports/runway_geometry_feature_build.json`. Training has 2,085,047 rows, SHA-256 `8cab44bb942976d12c2c75407c1aee4200a0087a3fd19da5a1924916bdc47005`; ranking has 344,841, SHA-256 `b790d2afbc9e0964bc0c8df71d3c41d18edfa447e6f77e7c691a90677281315c`. Independent full streaming readback confirmed exact unique baseline IDs/order, ID plus 16 float32 fields, no infinities or negative/fractional counts, 15-minute counts no greater than 60-minute counts, and exact all-NaN geometry masks (458 training/94 ranking). Wind is available for 1,990,657/337,411 rows. All 22 source/input fingerprints remained unchanged before and after verification. An independent covariate-only check of 18 genuine departures reproduced 144 traffic counts and their static geometry and exact-hour weather values. No BLOCK/TAXI labels or model scores were used. Counts describe the released batches; missing history at the January/July ranking period boundaries remains a limitation.

Independent v9 final verification also passed: exact template order and fixed 0.5 blend on all 339,377 valid-proxy rows; all 5,464 other predictions, including 4,907 missing-clock rows, retain v8 exactly. The finalized file and candidate bytes both match SHA-256 `bc465ae7ff48deac5f93cd449a3799fee1a361a8021ec2ca03ff70baccc0f417`. Source/model/input fingerprints were unchanged before and after reads. These integrity checks did not access leaderboard feedback or change any prediction.
