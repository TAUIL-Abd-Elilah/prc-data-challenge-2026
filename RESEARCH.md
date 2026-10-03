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
