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

1. Build game URLs using season-based source rules.
2. Fetch JSON from NHL API families based on season:
	- Pre-2009 seasons: NHL Stats REST API only
	- 2009+ seasons: NHL Web API / Gamecenter first, with NHL Stats REST fallback
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
- output database `hockey_data.db`

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

By default this runs from 2009 (first season with reliable coordinate shot data) through the current NHL season.

**Note on pre-2009 data**: The NHL Web API only has sparse/manual shot coordinates before the 2009-10 season. A few games in 2008 may have coordinates but most do not. To attempt scraping older seasons:

```bash
python Main.py --start-season 2008 --end-season 2008 --empty-game-streak 500
```

When a specific game, team, or player endpoint does not exist for a season (for example expansion-era teams in earlier years), 404 responses are treated as optional and skipped so the scrape can continue.

Shot source behavior is hardcoded for simplicity:

- Pre-2009 seasons use Stats REST shot parsing.
- 2009+ seasons use Web API shot parsing with Stats REST fallback when a game returns no web shot rows.

NHL Edge summary capture is enabled by default for 2021+ seasons.

Capture NHL Edge snapshots explicitly for each scraped season:

```bash
python Main.py --capture-edge
```

Disable default EDGE summary capture for a run:

```bash
python Main.py --no-capture-edge
```

EDGE capture is hardcoded to start at 2021+ seasons when enabled.

## Player seasonal stats

Starting with the 2025-26 season, `Main.py` automatically captures per-player seasonal statistics from the NHL Stats REST API (`/stats/rest/en/skater/summary` and `/stats/rest/en/goalie/summary`) during every scrape run. This includes skater and goalie stats such as:

- **Skater**: games played, TOI, goals, assists, points, shots on goal, plus/minus, PIM, power-play goals, short-handed goals, game-winning goals, blocked shots, hits, faceoffs, takeaways, giveaways
- **Goalie**: save percentage, goals-against average, shutouts
- **Position**: player position (C/LW/RW/D/G) for position-based analysis

The data is stored in the `player_seasonal_stats` table, keyed by `(player_id, season, game_type)`. This table is indexed by `season`, `team`, and `position` for fast queries.

Player stats are captured by default for all non-EDGE-only scrape runs. To disable:

```bash
python Main.py --season 2024 --no-player-stats
```

To backfill player stats for a season you've already scraped:

```bash
python -c "
import Main
Main.fetch_and_store_player_season_stats('hockey_data.db', '2024', '02', 10)
"
```

To backfill player stats for **all seasons** in your database (all game types):

```bash
python Main.py --player-stats-backfill --db-path hockey_data.db
```

This will automatically:
- Detect all seasons from your existing shot data
- Detect all game types for each season
- Fetch and store player seasonal stats with rate limiting
- Skip seasons that already have stats (idempotent)

The Stats REST API provides data going back much further than the NHL Edge API, so this works for all seasons where you have shot data (2009 onward — though the Stats REST skater/goalie summary endpoints reach back much earlier).

Run summary EDGE capture for a single season:

```bash
python Main.py --capture-edge --season 2024
```

Run deep EDGE capture for a single season:

```bash
python Main.py --capture-edge-deep --season 2024
```

Run summary and deep EDGE capture in one pass:

```bash
python Main.py --capture-edge --capture-edge-deep --season 2024
```

Capture the deeper Edge crawl too:

```bash
python Main.py --capture-edge --capture-edge-deep
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

## Database reset and data population order

### What `reset_db.py` does

The `reset_db.py` script drops all tables in your SQLite database, effectively resetting it to a clean state. It:

1. Checks if the database file exists
2. Lists all tables that will be dropped
3. Drops each table (excluding internal SQLite tables like `sqlite_sequence`)
4. Closes the connection

This is useful when you want to start fresh with a new scrape or when you need to rebuild derived tables after schema changes.

**Note:** This script does not delete the database file itself, only the tables within it.

### Recommended order for populating tables

The scripts in this project have dependencies on each other. Here's the recommended order to run them to populate all tables:

1. **Reset the database (optional)** - Start fresh if needed:
   ```bash
   python reset_db.py
   ```

2. **Run the main scraper** - Populates the core `shots` table and optionally `edge_payloads`/`edge_detail_payloads`:
   ```bash
   python Main.py --season 2024
   ```
   This creates the `shots` table with all shot/goal event data.

3. **Fetch player biographical data** - Populates the `players` table (optional but recommended for enriched analysis):
   ```bash
   python player_bio.py --db-path hockey_data.db
   ```
   This fetches player info (height, weight, position, draft info, etc.) for shooters and goalies found in your shot data.

4. **Run metrics computation** - Populates `shot_xg`, `metrics_model_runs`, `player_career_trajectory`, `goalie_career_trajectory`, and other derived tables:
   ```bash
   python Metrics.py --db-path hockey_data.db --season 2024
   ```
   This requires the `shots` table to exist and optionally the `players` table for player-adaptive models.

5. **Run age-adjusted metrics (optional)** - Populates age-adjusted tables:
   ```bash
   python age_model.py --db-path hockey_data.db --season 2024
   ```

### Quick reference: Table dependencies

```
shots (Main.py) ───────────────┐
                               ├──► shot_xg (Metrics.py)
players (player_bio.py) ───────┘
                               ├──► player_career_trajectory (Metrics.py)
                               ├──► goalie_career_trajectory (Metrics.py)
                               └──► metrics_model_runs (Metrics.py)
```

### Alternative: Backfill player stats

If you already have shot data and want to add player seasonal stats (separate from biographical data), use:

```bash
# Backfill for a specific season
python -c "import Main; Main.fetch_and_store_player_season_stats('hockey_data.db', '2024', '02', 10)"
```

## Run advanced metrics (post-scrape)

Advanced metrics are computed in a separate standalone script so the scrape path in `Main.py` stays unchanged.

`Metrics.py` trains and scores with a calibrated XGBoost xG model. It supports two model modes:

- **Base situation model** — uses only shot-location and game-state features (league-average shooter vs. league-average goalie).
- **Player-adaptive model** (default) — adds `shooter_id` and `goalie_id` as features so the model learns individual shooter finishing skill and goalie saving ability. This produces per-shot xG that reflects who took the shot and who was in net.

It also writes validation and calibration-monitoring diagnostics into SQLite so you can track model quality over time, and computes career trajectory tables that track player and goalie performance across multiple seasons.

### Age curve model

The age-adjusted metrics (age-adjusted GAx) are computed by a separate module in [`age_model.py`](age_model.py). This module fits a smooth league-wide xG-per-game age curve using a 4th-degree polynomial with ridge regularization, blended with a prior lifecycle curve. It also computes player-specific trend multipliers from recent performance to adjust projections.

The age model is independent from the core xG model and can be evolved or replaced separately.

### Future: opponent strength features

The `team_strength_metrics` table is created but currently stores placeholder data. A future enhancement will incorporate opponent team and goalie strength as features in the xG model. This would allow the model to account for:

- **Team defense quality** — shots against stronger defensive teams are harder to score on
- **Goalie quality** — facing a top-tier goalie reduces scoring probability beyond what the goalie_id feature captures

To implement this, the feature spec in `_build_feature_spec()` would need additional numeric features for opponent metrics, and the `_vectorize_rows()` function would need to join against `team_strength_metrics` during vectorization.

After scraping data into SQLite, run:

```bash
python Metrics.py --db-path hockey_data.db --season 2024
```

To force a re-run that replaces all output tables for a season (ignoring the staleness check):

```bash
python Metrics.py --db-path hockey_data.db --season 2024 --force
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
- `--validation-split-strategy` (default `random`, options: `random` or `temporal` — temporal splits by game date to avoid leakage)
- `--rolling-window` (default `3` seasons for rolling benchmark metrics)
- `--no-rate-metrics` (disables per-60 rate metrics)
- `--no-age-adjusted` (disables age-adjusted metrics)

Example with explicit model controls:

```bash
python Metrics.py --db-path hockey_data.db --season 2024 --train-start-season 2022 --validation-split 0.2 --epochs 500 --xgb-max-depth 5
```

Run a situation-only (no player identity) model:

```bash
python Metrics.py --db-path hockey_data.db --season 2024 --no-player-effects
```

Use temporal validation split (recommended for time-series data):

```bash
python Metrics.py --db-path hockey_data.db --season 2024 --validation-split-strategy temporal
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
python -m visualization.app --db-path hockey_data.db
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
- `--empty-game-streak` (default `200`, increase for older/sparse seasons)
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
- `Period_Time_Remaining`
- `Year`
- `GameID`
- `API_Source`
- `Goalie`
- `Goalie_ID`
- `Shot_Distance`
- `Shot_Angle`
- `Is_Empty_Net`
- `Score_Differential`
- `Zone`
- `Event_ID`

## Player biographical data (player_bio.py)

A separate module fetches and stores NHL player biographical data from the NHL Web API (`/v1/player/{id}/landing`). This is useful for enriching shot data with player attributes like height, weight, handedness, birth date, and draft info.

```bash
# Default: collect player IDs from the shots table and fetch their bios
python player_bio.py --db-path hockey_data.db

# Override: fetch a specific range of player IDs
python player_bio.py --start 8440000 --end 8486000 --db-path hockey_data.db

# Force re-fetch all (overwrites existing)
python player_bio.py --force-refresh --db-path hockey_data.db

# Check current fetch status
python player_bio.py --status --db-path hockey_data.db
```

Key features:
- **Shot-driven**: By default, collects player IDs (shooters + goalies) from your existing shot data — no wasteful full-range scan
- **Range override**: Use `--start` and `--end` to manually specify an ID range if needed
- **Idempotent**: Skips players already stored; use `--force-refresh` to overwrite
- **Rate limited**: 0.1s default delay between requests (configurable via `--request-delay`)
- **Parallel**: 2 workers by default (`--max-workers`)
- **Idempotent**: `INSERT OR REPLACE` on `player_id` primary key; skips existing unless `--force-refresh`
- **Not found handling**: Player IDs that return 404 are counted but don't stop the run

Stored in `players` table:
- `player_id` (PK), `first_name`, `last_name`, `full_name`
- `height_inches`, `height_cm`, `weight_lbs`, `weight_kg`
- `birth_date`, `birth_city`, `birth_state_province`, `birth_country`
- `shoots_catches` (L/R), `position` (C/LW/RW/D/G)
- `draft_year`, `draft_team`, `draft_round`, `draft_pick_in_round`, `draft_overall_pick`
- `fetched_at` timestamp

Join with `shots` table on `shooter_id` or `goalie_id` for enriched analysis.

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

## Future Work / Ideas for Next Steps

### Age regression curves for player development
A natural next step is modeling how player skill evolves with age. Instead of treating each player as a static effect, we could:

- **General age curve** — fit a league-wide aging curve (e.g., peak at ~25-27, decline after 30) for shooters and goalies separately
- **Feature-screened nonlinear age model** — start with a broad feature set available in this project (age, xG/game progression, GAX trend, TOI/game, shot quality mix, role/position, and selected Edge pace metrics), train a simple nonlinear model (e.g., gradient boosting), remove weak features using permutation importance/SHAP, then retrain and compare holdout performance to keep only meaningful predictors
- **Archetype-specific curves** — cluster players by style (sniper, playmaker, power forward, butterfly goalie, hybrid) and fit separate curves
- **Per-player estimation** — with enough seasons, estimate individual aging trajectories (hierarchical Bayesian or mixed-effects model)
- **Application to non-NHL evaluation** — project junior/college/European players forward using age curves to identify undervalued prospects

This would live in a separate analysis project but could feed back into the xG model as a prior on player effects.

### Other ideas
- Shift-chart lineups (on-ice teammates/opponents per shot)
- Pre-shot context (rebound flag, rush vs cycle, zone entry type)
- Goalie depth/angle from Edge data
- Model comparison CLI (`--compare-model`)
- Visualization integration for trajectory tables

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
