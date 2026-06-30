# HockeyShotMap

HockeyShotMap is a hockey data collection project for building shot and goal datasets that can later be used for heatmaps, shooting percentage analysis, or visualization work.

The project now also includes a separate read-only visualization app in [visualization/app.py](visualization/app.py) that reads the SQLite database directly and serves an interactive rink heatmap dashboard.

## Script roles

- [Main.py](Main.py) is the active NHL Web API + Stats REST ETL pipeline.
- [Scraper.py](Scraper.py) is a separate prototype track for future player-level and cross-league data collection (not integrated into the active pipeline).
- [DataCleaning.py](DataCleaning.py) contains a basic CSV loader utility for follow-on analysis work.

## What the active scraper collects

For each shot or goal event, [Main.py](Main.py) records:

- shot result (`Goal` or `ngshot` for non-goal shots)
- x/y rink coordinates
- shot type
- shooter name
- shooter and goalie IDs when available
- period time and remaining time
- shot distance and angle
- empty-net flag
- strength state, score differential, zone, and event id
- team tri-code
- home/away flag
- period
- season year
- game id
- API source (`web` or `stats`)

In addition, when enabled, the scraper stores NHL Edge payload snapshots from the Web API in a separate SQLite table.

The Edge data families are useful if you want to enrich the shot feed later:

- Team landing/comparison: shot attempts over 90, bursts over 22 mph, distance per 60, high-danger shot totals, and zone-time summaries.
- Skater detail: top shot speed, skating speed, distance skated, shot-on-goal summaries/details, and zone-start/zone-time breakdowns.
- Goalie detail: goals-against average, games above .900, goal differential per 60, goal support, point percentage, and shot-location summaries/details.

## How [Main.py](Main.py) works

Pipeline flow:

1. Build game URLs for one or both NHL API families.
2. Fetch JSON from selected source(s):
	- NHL Web API / Gamecenter (`api-web.nhle.com`)
	- NHL Stats REST API (`api.nhle.com/stats/rest`)
3. Parse `Goal` and `Shot` events into normalized rows.
4. Persist rows into SQLite with duplicate-safe inserts.
5. Optionally capture NHL Edge summary payloads.
6. Optionally export CSV format for visualization workflows.

Web API play-by-play URL format:

`https://api-web.nhle.com/v1/gamecenter/{full_game_id}/play-by-play`

Stats REST shiftcharts URL format:

`https://api.nhle.com/stats/rest/en/shiftcharts?cayenneExp=gameId={full_game_id}&limit=-1`

By default, the script uses:

- season `2013`
- regular season game type (`02`)
- game ids `0001` through `1271`
- output database `hockey_shots.db`

## Setup

1. Create and activate a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the scraper

Default run (regular season, full game range, SQLite output):

```bash
python Main.py
```

By default this runs from 2005 (first season with coordinate shot data) through the current NHL season.

When a specific game, team, or player endpoint does not exist for a season (for example expansion-era teams in earlier years), 404 responses are treated as optional and skipped so the scrape can continue.

Use only one source explicitly:

```bash
python Main.py --api-source web
python Main.py --api-source stats
```

Use both sources (recommended for capture completeness):

```bash
python Main.py --api-source both
```

Capture NHL Edge snapshots for each scraped season:

```bash
python Main.py --api-source both --capture-edge
```

Capture the deeper Edge crawl too:

```bash
python Main.py --api-source both --capture-edge --capture-edge-deep
```

Run a bounded season range:

```bash
python Main.py --start-season 2013 --end-season 2018
```

Run a single season only:

```bash
python Main.py --season 2017
```

Run a small bounded range:

```bash
python Main.py --season 2013 --game-type 02 --start-game 1 --end-game 10
```

Run and export Tableau-compatible CSV:

```bash
python Main.py --export-csv 2018NHLShotInfoV2.csv
```

## Run advanced metrics (post-scrape)

Advanced metrics are computed in a separate standalone script so the scrape path in `Main.py` stays unchanged.

`Metrics.py` now trains and scores with a calibrated XGBoost xG model. It supports two model modes:

- **Base situation model** — uses only shot-location and game-state features (league-average shooter vs. league-average goalie).
- **Player-adaptive model** (default) — adds `shooter_id` and `goalie_id` as features so the model learns individual shooter finishing skill and goalie saving ability. This produces per-shot xG that reflects who took the shot and who was in net.

It also writes validation and calibration-monitoring diagnostics into SQLite so you can track model quality over time, and computes career trajectory tables that track player and goalie performance across multiple seasons.

After scraping data into SQLite, run:

```bash
python Metrics.py --db-path hockey_shots.db --season 2024
```

Refresh behavior is season-aware and incremental:

- metrics are stored season-by-season in derived tables
- rerunning with unchanged source rows for a season skips recalculating that season
- when training data or model settings change, affected seasons are recalculated
- this keeps season-level outputs stable for later multi-season rollups in visualization queries

Useful options:

- `--end-season`
- `--train-start-season`
- `--train-end-season`
- `--min-shots` (default `50` for league comparison thresholds)
- `--learning-rate` (XGBoost learning rate)
- `--epochs` (XGBoost boosting rounds / `n_estimators`)
- `--l2` (XGBoost `reg_lambda`)
- `--validation-split` (default `0.2` holdout for monitoring and calibration)
- `--calibration-method` (currently `sigmoid`)
- `--calibration-bins` (default `10` bins for reliability tracking)
- `--random-seed`
- `--xgb-max-depth`
- `--xgb-subsample`
- `--xgb-colsample-bytree`
- `--xgb-min-child-weight`
- `--no-player-effects` (disables shooter_id/goalie_id features, falling back to situation-only model)
- `--career-lookback` (default `3` trailing seasons for trajectory tracking)

Example with explicit model controls:

```bash
python Metrics.py --db-path hockey_shots.db --season 2024 --train-start-season 2022 --validation-split 0.2 --epochs 500 --xgb-max-depth 5
```

Run a situation-only (no player identity) model:

```bash
python Metrics.py --db-path hockey_shots.db --season 2024 --no-player-effects
```

### Player-adaptive model

When enabled (default for seasons 2010+), `shooter_id` and `goalie_id` are added as categorical features to the XGBoost model. This allows the model to learn:

- **Shooter skill** — some players are consistently better finishers from the same locations
- **Goalie skill** — some goalies save shots at a higher rate from the same situations

The model version will show `xgboost_player_adaptive_v1` as the model type. The xG values in `shot_xg` then reflect "what is the probability this specific shooter scores on this specific goalie from this situation."

### Career trajectory tables

Two new tables track multi-year performance to answer "is this player getting better or worse?":

- `player_career_trajectory` — per-shooter, per-season:
  - Career totals (shots, goals, xG, GAx, shooting%)
  - Trailing 3-year rolling averages
  - Current season totals
- `goalie_career_trajectory` — per-goalie, per-season:
  - Career totals (shots against, goals against, xGA, saves above avg, save%)
  - Trailing 3-year rolling averages
  - Current season totals

You can query these tables directly:

```sql
-- Find shooters whose 3-year xG/shot is above their career average (improving)
SELECT shooter, season,
       career_xg_per_shot,
       trailing_3yr_xg / trailing_3yr_shots AS trailing_xg_per_shot
FROM player_career_trajectory
WHERE trailing_3yr_shots >= 100
  AND (trailing_3yr_xg / trailing_3yr_shots) > career_xg_per_shot
ORDER BY (trailing_3yr_xg / trailing_3yr_shots) - career_xg_per_shot DESC;

-- Goalies trending down
SELECT goalie, season, career_saves_above_avg, trailing_3yr_saves_above_avg
FROM goalie_career_trajectory
WHERE trailing_3yr_saves_above_avg < career_saves_above_avg;
```

### Metrics model monitoring outputs

Each run stores:

- model metadata and validation metrics in `metrics_model_runs`
- reliability-bin diagnostics in `metrics_model_validation_bins`
- per-shot xG in `shot_xg`

The run summary logged by `Metrics.py` includes:

- validation metrics (`auc`, `log_loss`, `brier`, `ece`) when a holdout split is available
- `top_features` from XGBoost feature importance
- `feature_pruning_candidates` for near-zero-importance features

Calibration note: sigmoid calibration (Platt scaling) learns a 1D correction from raw model probabilities to better align predicted probabilities with observed goal rates.

## Run the visualization app

The dashboard reads the database directly, so you can open it later without rerunning the scraper.

```bash
python -m visualization.app --db-path hockey_shots.db
```

Then open the local URL printed in the terminal.

Use a custom database path:

```bash
python Main.py --db-path nhl_shots_2013.db
```

Useful CLI arguments:

- `--season`
- `--start-season`
- `--end-season`
- `--game-type` (`01`, `02`, `03`, `04`)
- `--api-source` (`web`, `stats`, `both`)
- `--capture-edge`
- `--capture-edge-deep`
- `--start-game`
- `--end-game`
- `--timeout`
- `--db-path`
- `--export-csv`
- `--log-level`

## Storage model

Primary storage is SQLite in the `shots` table.

Additional storage when `--capture-edge` is used:

- `edge_payloads` table stores raw JSON snapshots from endpoints such as:
	- `/v1/edge/team-landing/{season}/{game-type}`
	- `/v1/edge/skater-landing/{season}/{game-type}`
	- `/v1/edge/goalie-landing/{season}/{game-type}`
	- `/v1/edge/by-the-numbers`

Additional storage when `--capture-edge-deep` is used:

- `edge_detail_payloads` table stores raw JSON snapshots for:
	- `team-detail`
	- `roster`
	- `skater-detail`
	- `goalie-detail`

Idempotency behavior:

- Rows are keyed by a deterministic event hash.
- Re-running the same season/game range does not insert duplicate events.

Export fields are preserved in this order:

- `Shot`
- `X`
- `Y`
- `Shot_Type`
- `Shooter`
- `Shooter_ID`
- `Team`
- `Home_Away`
- `Period`
- `Period_Time`
- `Period_Time_Remaining`
- `Year`
- `GameID`
- `API_Source`
- `Goalie`
- `Goalie_ID`
- `Shot_Distance`
- `Shot_Angle`
- `Is_Empty_Net`
- `Strength_State`
- `Score_Differential`
- `Zone`
- `Event_ID`

## Tests

Run tests with:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

Current test coverage includes:

- URL builder correctness
- parser behavior with complete and partial event payloads
- web play-by-play parser behavior
- duplicate-safe persistence behavior in SQLite
- metrics pipeline with player-adaptive xG model, validation metrics, calibration monitoring, and career trajectory computation

## What [Scraper.py](Scraper.py) is for

[Scraper.py](Scraper.py) is now kept as an explicit prototype scaffold for future player-level research and scouting datasets.

Intended long-term uses include:

- ranking NHL players for Hall of Fame style analysis projects
- collecting junior/college/international player histories
- building draft-prospect style ranking models for non-NHL players

It is intentionally not wired into the active shot-map ingestion path in [Main.py](Main.py).

Because this is a strategic but separate idea, there are two valid paths:

- keep it in this repository as an incubation module while requirements evolve
- split it into a dedicated repository later once scope and schema stabilize

Current state:

- API shape and persistence schema are not finalized
- functions are documented placeholders and raise `NotImplementedError`
- no tests are currently attached to this prototype module

## Requirements

- Python 3.x
- Dependencies listed in [requirements.txt](requirements.txt)

## Data files

- [2018NHLShotInfo.csv](2018NHLShotInfo.csv) - sample or earlier export data
- [2018NHLShotInfoV2.csv](2018NHLShotInfoV2.csv) - optional Tableau-compatible export target

## Current limitations

- [Scraper.py](Scraper.py) is intentionally a non-production prototype scaffold for future player-level/multi-league work.
- The active scraper relies on NHL Web API and NHL Stats REST endpoint formats used in the code. If those payloads change, parsing may need updates.
