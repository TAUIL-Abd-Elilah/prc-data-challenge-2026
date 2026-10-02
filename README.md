# PRC Data Challenge 2026 — merry-mushroom

This repository predicts departure taxi-out time for the [PRC Data Challenge 2026](https://prc-data-challenge-2026.netlify.app/). The organizer scores root mean square error (RMSE) on January and July 2026 departures. The code is original and licensed under GNU GPLv3.

## Why this model

The [data specification](https://prc-data-challenge-2026.netlify.app/data.html) defines taxi-out time as the time from off-block to takeoff. It removes the movement record's off-block time from the ranking file but retains Network Manager's `AOBT_3_flt` actual off-block time. Their difference is a strong candidate predictor, though the organizer warns that the two source systems can disagree. `solution.py` first measures this proxy on 2025 data, then learns a correction model. A separate direct model handles records with missing or implausible NM off-block times.

Both models use only fields available in `ranking.parquet`. Features include airport, route, runway, stand, aircraft, operator, calendar, disagreements between supplied time estimates, nearby takeoffs and landings, and the number of other aircraft still taxiing at the estimated off-block time. Neither movement `BLOCK_TIME_UTC_mvt` nor other departures' `TAXITIME_SEC_mvt` is used as a feature.

## Data and access

The approved team is `merry-mushroom`. Download the twelve canonical 2025 training files, `ranking.parquet`, and `submitting.parquet` from `prc-2026-datasets` into `data/`, using the [OpenSky MinIO console](https://s3-console.opensky-network.org/) and **Other Authentication Methods → Login with SSO**. The submission bucket is `prc-2026-merry-mushroom`. The [official ranking instructions](https://prc-data-challenge-2026.netlify.app/ranking.html) require the MinIO Client CLI to upload a submission. Keep credentials and raw competition data out of this repository. Browser duplicate copies such as `... (1).parquet` are ignored by the loader.

Expected layout:

```text
data/
  training_2025-01-01_2025-02-01.parquet
  ...
  training_2025-12-01_2026-01-01.parquet
  ranking.parquet
  submitting.parquet
```

## Run

Python 3.11 or later is recommended. On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-lock.txt
python solution.py audit --data-dir data
python solution.py train --data-dir data --output-dir artifacts/baseline --threads 8 --rounds 900
python solution.py predict --data-dir data --model-dir artifacts/baseline --team merry-mushroom --version 1 --output-dir artifacts/baseline-proposal --threads 8
python missing_expert.py --threads 2 --rounds 1400
python grouped_proxy.py
python download_noaa_weather.py
python weather_model.py train --threads 8 --rounds 1500
python weather_model.py complete-oof --threads 2
python weather_model.py predict-only --threads 8
python airport_models.py --threads 6 --rounds 1000
python schedule_tail.py
python lirf_expert.py --threads 2 --rounds 220
python ensemble.py
python lobt_expert.py --threads 8 --rounds 900
python lobt_blend.py
python finalize_submission.py --predictions artifacts/lobt_ensemble/predictions.parquet --team merry-mushroom --version 3
```

`audit` reports the direct proxy's coverage, RMSE and bias. Base models validate on whole January and July 2025 months, and separately on November and December 2025. The first split imitates the scoring months; the second tests a forward time shift. Ranking prediction also writes feature and expert caches for the specialist stages. Core shared trees use labels between 0 and 86,400 seconds; separate specialists retain the exceptional overnight labels. Every finite label, including negative and above-24-hour records, is included in the final ensemble validation. `complete_oof.py` can expand an earlier baseline export to include those records.

The weather and airport models predict corrections to the supplied off-block proxy. A Rome specialist models taxi time divided by the supplied schedule difference, allowing extrapolation on rare long records. `ensemble.py` learns nonnegative expert weights using nested calibration folds within the seasonal holdout, and evaluates those weights unchanged on the forward holdout. Final models train on the complete 2025 data. The authoritative report is `artifacts/ensemble/validation.json`.

The experts use held-out month labels for early stopping, so these validation numbers are indicative and are not strictly unbiased estimates. The candidate's aggregate report, model weights, and file manifest are saved in `reports/`. `lobt_expert.py` predicts a correction to the last off-block estimate; `lobt_blend.py` applies it only when the actual off-block estimate is invalid or differs by more than one hour. This conditional policy was selected on seasonal calibration and checked unchanged on the forward period.

`finalize_submission.py` checks all 344,841 template IDs in their original order, verifies finite nonnegative predictions through a Parquet round trip, and writes the submission plus a SHA-256 manifest. It refuses to overwrite a finalized submission. Use a fresh output directory or version when repeating prediction. The reference runtime is Python 3.11 on Windows with 32 GB RAM; exact package versions are in `requirements-lock.txt`.

## Reproduce v4

After running the v3 pipeline above, install the additional pinned dependencies and run:

```powershell
python -m pip install -r requirements-research.txt
python catboost_gpu.py --mode fit --iterations 1500 --depth 8 --threads 2
python catboost_source.py --fold both --iterations 600 --depth 7 --threads 4
python catboost_source.py --evaluate-only
python catboost_source.py --evaluate-sequential-gpu
python catboost_gpu.py --mode final-predict --iterations 1500 --depth 8 --threads 2
python catboost_source.py --fit-final-rank --threads 4
python catboost_source.py --combine-gpu
python finalize_submission.py --predictions artifacts/catboost/source/sequential_predictions.parquet --team merry-mushroom --version 4
```

The additional residual CatBoost model uses all eligible 2025 labels, recurring flight categories and the same NOAA weather features. It requires a CUDA-capable NVIDIA GPU; the reference run uses an RTX 3090. A 25% blend with v3 was selected on January/July. A separate classifier estimates whether Rome and Istanbul records use the scheduled timestamp; its fixed half-probability correction follows the GPU blend. Invalid AOBT rows retain the v3 prediction. Both stages use only ranking-available covariates.

The fixed sequential policy reduced local all-finite RMSE from 329.403 to 328.221 seconds on January/July and from 225.625 to 224.465 on November/December, over 672,428 rows in total. Combined RMSE was 283.566 to 282.412. These periods have been examined across multiple experiments, so this is a model comparison rather than an untouched forecast. The aggregate report is [reports/validation_v4.json](reports/validation_v4.json). [RESEARCH.md](RESEARCH.md) records rejected experiments and validation limitations.

## Reproduce v5 research and candidate

After the v4 pipeline, run the following commands. The larger timestamp model uses CUDA; the missing-record specialist uses at most four CPU threads.

```powershell
python v4_reference.py
python deep_timestamp_expert.py --mode fit --iterations 4500 --depth 9 --threads 2
python deep_timestamp_expert.py --mode evaluate
python deep_timestamp_expert.py --mode final-predict --iterations 4500 --depth 9 --threads 2
python missing_catboost.py --fold both --threads 4
python missing_catboost.py --evaluate-only
python missing_catboost.py --fit-final-direct
python arrival_features.py --mode traffic-eval
python arrival_residual_expert.py --mode build-features --threads 4
python arrival_residual_expert.py --mode validate --threads 4
python arrival_residual_expert.py --mode final-predict --threads 4
python v5_ensemble.py --mode evaluate
python v5_ensemble.py --mode ranking
python finalize_submission.py --predictions artifacts/v5-ensemble/predictions.parquet --team merry-mushroom --version 5
```

The timestamp expert adds second/minute precision, flight duration, planning revisions and NM callsign features from released fields. It uses internal training splits for early stopping and a fixed 50% blend on valid AOBT rows. Its final model trains on 2,061,428 eligible 2025 departures. The second 50% blend applies a direct CatBoost expert only to departures with missing NM off-block records outside Rome. The two gates are disjoint. Every prediction is clipped at zero.

The fixed timestamp/missing combination improves local all-finite RMSE from 328.216 to 326.531 seconds on January/July and from 224.464 to 221.652 on November/December; pooled RMSE improves from 282.409 to 280.317 over 672,428 rows. The v4 comparator now uses the same nonnegative policy as the actual submission; older saved diagnostics included 57 negative predictions. GPU fits can vary slightly by hardware, so historical score equality is an optional `v4_reference.py --verify-snapshot` audit.

Arrival ground features use the ARR block/taxi fields retained in the supplied ranking file. The initial correction stack is diagnostic only: its seasonal base models included November/December labels, which creates an indirect dependence in its forward check. `v5_ensemble.py` excludes that stack from submissions. The separate `arrival_residual_expert.py` trains directly on complementary months. Its seasonally selected 25% blend improves the stronger timestamp/missing combination in both periods: January/July RMSE becomes 325.822, November/December 221.142, and pooled 279.697. The incremental gain has positive day-bootstrap intervals in both folds. This direct arrival expert is included in v5.

Local folds have been used repeatedly for model comparison. Improvements and bootstrap intervals do not guarantee a 2026 score or a prize. The public aggregate reports and final submission receipt record the accepted policy and observed outcome.

## Reproduce v6

After reproducing v5, run the larger clock-and-arrival expert. Run GPU stages serially on the reference 32 GB machine.

```powershell
python deep_arrival_expert.py --mode fit
python deep_arrival_expert.py --mode fresh-audit
python deep_arrival_expert.py --mode final-predict
python finalize_submission.py --predictions artifacts/v6-deep-arrival/predictions.parquet --team merry-mushroom --version 6
python submission_quota.py --submission submissions/merry-mushroom_v6.parquet --report reports/submission_quota_v6.json
```

This expert uses the released timestamp features plus the 14 released ARR traffic features, with depth 10 and at most 10,000 CatBoost trees. January/July select a 0.5 blend with v5 on valid AOBT rows. The fixed blend changes all-finite January/July RMSE from 325.822 to 323.723 seconds and November/December from 221.142 to 217.903. Both paired day-bootstrap intervals are positive. Invalid AOBT rows retain the v5 prediction.

The additional April/October audit retrains both the previous depth-9 timestamp architecture and the new architecture while excluding those months from fitting and early stopping. On all 357,813 eligible held-out flights, the same 0.5 blend changes RMSE from 204.521 to 200.307, with a 95% paired day interval for the gain of 3.872 to 4.599 seconds. This checks the architecture change; it is not an estimate of the complete ensemble's 2026 score. Complete ID coverage, finite predictions and input fingerprints are required before promotion.

The read-only quota check counts uploads in the preceding 24 hours because the official five-per-day rule does not publish a reset timezone. It also checks a conservative 1,000,000,000-byte bucket limit, the team destination and a fresh higher version number. No submissions are deleted. Upload the frozen v6 file with the same MinIO command below, then download it and its result and run `python record_submission.py --version 6`. Only aggregate receipts are published. Ranking results never select model settings.

## Reproduce v7

After completing v6, build the released neighbor/runway features and run the fixed traffic architecture. See the exact commands in [RESEARCH.md](RESEARCH.md). January/July select a 0.5 blend with v6; the unchanged blend improves local all-finite RMSE from 323.723 to 322.311 seconds on January/July and from 217.903 to 215.199 on November/December, with positive paired UTC-day confidence intervals. The matched April/October architecture audit improves from 198.491 to 195.888 over all 357,813 eligible flights, with a gain interval of 2.282 to 2.946 seconds. The architecture and protocol were frozen before the v6 official result. Aggregate validation is in [reports/traffic_validation_v7.json](reports/traffic_validation_v7.json).

The final model uses 9,999 trees fitted on all eligible 2025 departures. Its 344,841 ranking predictions preserve the v6 policy on all 5,464 invalid-AOBT rows. The prediction file and input/model hashes are recorded in [reports/model_v7.json](reports/model_v7.json). Finalize and check quota with:

```powershell
python finalize_submission.py --predictions artifacts/v7-runway-traffic/predictions.parquet --team merry-mushroom --version 7
python submission_quota.py --submission submissions/merry-mushroom_v7.parquet --report reports/submission_quota_v7.json
```

## Submit with MinIO Client

The organizer accepts uploads through the MinIO Client (`mc`), using the team's own bucket. On Windows PowerShell, use the [official community Windows release](https://github.com/minio/mc/releases/download/RELEASE.2025-08-13T08-35-41Z/mc.windows-amd64.RELEASE.2025-08-13T08-35-41Z.exe) (AGPL-3.0), generate an OpenSky access key and secret for your account, replace the placeholders below, and upload the finalized file for the selected version. The [current AIStor Windows client](https://dl.min.io/aistor/mc/release/windows-amd64/mc.exe) is also available from MinIO.

```powershell
$mc = Join-Path $env:TEMP 'mc-community.exe'
Invoke-WebRequest 'https://github.com/minio/mc/releases/download/RELEASE.2025-08-13T08-35-41Z/mc.windows-amd64.RELEASE.2025-08-13T08-35-41Z.exe' -OutFile $mc
if ((Get-FileHash $mc -Algorithm SHA256).Hash.ToLowerInvariant() -ne 'c8db13ebeda31497f354c0e950809db0ae9b2a2a69b8afee68c128c37300c157') { throw 'MinIO Client checksum mismatch' }
& $mc alias set opensky 'https://s3.opensky-network.org/' '<ACCESS_KEY>' '<SECRET_KEY>'
$version = 7
$filename = "merry-mushroom_v$version.parquet"
python submission_quota.py --mc $mc --submission "submissions/$filename"
if ($LASTEXITCODE -ne 0) { throw 'Submission quota check failed' }
& $mc cp --disable-multipart ".\submissions\$filename" "opensky/prc-2026-merry-mushroom/$filename"
```

Keep access keys and secrets private; never commit them. Check the [official leaderboard](https://prc-data-challenge-2026.netlify.app/ranking.html) for the scored entry after upload. The repository does not perform the upload.

To reproduce the verification, download the stored submission and result file with `mc`, then run the receipt command. It checks the remote bytes against the finalized SHA-256, requires acceptance of every pair, compares the result with the public API, and counts distinct teams with lower best scores. Raw result files stay under ignored `artifacts/`; only aggregate receipt fields are published.

```powershell
& $mc cp "opensky/prc-2026-merry-mushroom/$filename" ".\artifacts\submission-receipt\verified_$filename"
& $mc cp "opensky/prc-2026-merry-mushroom/${filename}_result.json" ".\artifacts\submission-receipt\${filename}_result.json"
python record_submission.py --version $version
```

## Official result

The accepted v7 submission improved the team's official score by 1.6912 seconds RMSE over v6. All five submissions were scored over every one of the 344,841 pairs. The v7 remote readback exactly matches the finalized SHA-256.

| Submission | Official RMSE | Best-score team rank at snapshot | Public receipt |
|---|---:|---:|---|
| `merry-mushroom_v3.parquet` | 289.2078 seconds | 88th at 2026-10-02 13:48:21 UTC | [v3 receipt](reports/submission_v3.json) |
| `merry-mushroom_v4.parquet` | 288.5236 seconds | 87th at 2026-10-02 14:28:43 UTC | [v4 receipt](reports/submission_v4.json) |
| `merry-mushroom_v5.parquet` | 283.0418 seconds | 74th at 2026-10-02 21:35:58 UTC | [v5 receipt](reports/submission_v5.json) |
| `merry-mushroom_v6.parquet` | 280.48 seconds | 72nd at 2026-10-02 22:28:39 UTC | [v6 receipt](reports/submission_v6.json) |
| `merry-mushroom_v7.parquet` | **278.7888 seconds** | **69th** at 2026-10-02 23:01:39 UTC | [v7 receipt](reports/submission_v7.json) |

Ranks count distinct teams with a lower best score, plus one. They are leaderboard snapshots, and first place has not been achieved. The scores are available from the [official team results API](https://datacomp.opensky-network.org/api/competitions/bb3693e1-26bc-4a9e-8619-4fe78b4eab0c/leaderboard?teamName=merry-mushroom&limit=200).

After v7, the conservative preceding-24-hour upload count is five. The next count slot opens at **2026-10-03 13:46:52.428 UTC**. Each future file still needs a fresh quota/version/size check before upload. The post-upload audit is [reports/submission_quota_after_v7.json](reports/submission_quota_after_v7.json); it intentionally rejects re-uploading the already stored v7 file. Local research continues while quota is closed.

The synthetic pipeline check is `python tests/smoke.py`. It tests execution and Parquet alignment; its generated RMSE has no competition meaning.

## Improving a submission

Compare `validation.json` overall RMSE and per-airport RMSE with the `audit` proxy. Investigate missing AOBT and flight/movement mismatches before tuning. A validation gain is evidence for a candidate submission; there is no guarantee it transfers to January and July 2026. The [live ranking](https://prc-data-challenge-2026.netlify.app/ranking.html) accepts at most five submissions per team per day. Do not tune to individual leaderboard responses; the organizer prohibits exploiting the ranking process.

For prize eligibility, the [rules](https://prc-data-challenge-2026.netlify.app/eligibility.html) require public GPLv3 source code, sufficient reproduction instructions, and open licensing for any external data. Weather uses NOAA's CC0 GHCNh dataset; source citation, station substitutions, units and caveats are documented in [WEATHER_SOURCES.md](WEATHER_SOURCES.md). The exact download URLs and SHA-256 digests are recorded in [weather_sources.json](weather_sources.json). NOAA source files can be revised, so a later download may differ from this snapshot. Raw data, fitted artifacts, and submissions are excluded from Git.
