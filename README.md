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

## Submit with MinIO Client

The organizer accepts uploads through the MinIO Client (`mc`), using the team's own bucket. On Windows PowerShell, download the [current Windows `mc.exe`](https://dl.min.io/aistor/mc/release/windows-amd64/mc.exe), generate an OpenSky access key and secret for your account, replace the placeholders below, and upload the finalized v3 file:

```powershell
$mc = Join-Path $env:TEMP 'mc.exe'
Invoke-WebRequest 'https://dl.min.io/aistor/mc/release/windows-amd64/mc.exe' -OutFile $mc
& $mc alias set opensky 'https://s3.opensky-network.org/' '<ACCESS_KEY>' '<SECRET_KEY>'
& $mc cp '.\submissions\merry-mushroom_v3.parquet' 'opensky/prc-2026-merry-mushroom/merry-mushroom_v3.parquet'
```

Keep access keys and secrets private; never commit them. Check the [official leaderboard](https://prc-data-challenge-2026.netlify.app/ranking.html) for the scored entry after upload. The repository does not perform the upload.

The synthetic pipeline check is `python tests/smoke.py`. It tests execution and Parquet alignment; its generated RMSE has no competition meaning.

## Improving a submission

Compare `validation.json` overall RMSE and per-airport RMSE with the `audit` proxy. Investigate missing AOBT and flight/movement mismatches before tuning. A validation gain is evidence for a candidate submission; there is no guarantee it transfers to January and July 2026. The [live ranking](https://prc-data-challenge-2026.netlify.app/ranking.html) accepts at most five submissions per team per day. Do not tune to individual leaderboard responses; the organizer prohibits exploiting the ranking process.

For prize eligibility, the [rules](https://prc-data-challenge-2026.netlify.app/eligibility.html) require public GPLv3 source code, sufficient reproduction instructions, and open licensing for any external data. Weather uses NOAA's CC0 GHCNh dataset; source citation, station substitutions, units and caveats are documented in [WEATHER_SOURCES.md](WEATHER_SOURCES.md). The exact download URLs and SHA-256 digests are recorded in [weather_sources.json](weather_sources.json). NOAA source files can be revised, so a later download may differ from this snapshot. Raw data, fitted artifacts, and submissions are excluded from Git.
