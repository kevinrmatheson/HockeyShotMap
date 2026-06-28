# HockeyShotMap

HockeyShotMap is a hockey data collection project for building shot and goal datasets that can later be used for heatmaps, shooting percentage analysis, or visualization work.

## Script roles

- [Main.py](Main.py) is the active NHL Stats API ETL pipeline.
- [Scraper.py](Scraper.py) is a separate HockeyDB prototype and is not integrated into the active pipeline.
- [DataCleaning.py](DataCleaning.py) contains a basic CSV loader utility for follow-on analysis work.

## What the active scraper collects

For each shot or goal event, [Main.py](Main.py) records:

- shot result (`Goal` or `ngshot` for non-goal shots)
- x/y rink coordinates
- shot type
- shooter name
- team tri-code
- home/away flag
- period
- season year
- game id

## How [Main.py](Main.py) works

Pipeline flow:

1. Build NHL game feed URL.
2. Fetch game JSON from the NHL API.
3. Parse `Goal` and `Shot` events into normalized rows.
4. Persist rows into SQLite with duplicate-safe inserts.
5. Optionally export legacy CSV format for visualization workflows.

Game feed URL format:

`https://statsapi.web.nhl.com/api/v1/game/{season}{game_type}{game_id}/feed/live`

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

Default run (regular season 2013, full game range, SQLite output):

```bash
python Main.py
```

By default this runs from the earliest full season detected from NHL API endpoints through the current NHL season.

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

Run and export legacy CSV for Tableau compatibility:

```bash
python Main.py --export-csv 2018NHLShotInfoV2.csv
```

Use a custom database path:

```bash
python Main.py --db-path nhl_shots_2013.db
```

Useful CLI arguments:

- `--season`
- `--start-season`
- `--end-season`
- `--game-type` (`01`, `02`, `03`, `04`)
- `--start-game`
- `--end-game`
- `--timeout`
- `--db-path`
- `--export-csv`
- `--log-level`

## Storage model

Primary storage is SQLite in the `shots` table.

Idempotency behavior:

- Rows are keyed by a deterministic event hash.
- Re-running the same season/game range does not insert duplicate events.

Legacy export fields are preserved in this order:

- `Shot`
- `X`
- `Y`
- `Shot_Type`
- `Shooter`
- `Team`
- `Home_Away`
- `Period`
- `Year`
- `GameID`

## Tests

Run tests with:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

Current test coverage includes:

- URL builder correctness
- parser behavior with complete and partial event payloads
- duplicate-safe persistence behavior in SQLite

## What [Scraper.py](Scraper.py) appears to do

[Scraper.py](Scraper.py) imports `BeautifulSoup`, `requests`, and `numpy`, then defines functions for scraping league, team, and player pages from HockeyDB.

It is intentionally kept out of the active NHL pipeline for now and still needs repair work before production use.

Known issues include:

- `BeautifulSoup` is being given a URL string directly instead of fetched HTML.
- `League_Dict` is treated like a function in one place even though it is a dictionary.
- Several loops iterate over empty lists instead of the parsed results.
- `__main__` calls `league_year_scrape(OHL, 2019)` without quoting `OHL`.

## Requirements

- Python 3.x
- Dependencies listed in [requirements.txt](requirements.txt)

## Data files

- [2018NHLShotInfo.csv](2018NHLShotInfo.csv) - sample or earlier export data
- [2018NHLShotInfoV2.csv](2018NHLShotInfoV2.csv) - optional legacy-format export target

## Current limitations

- [Scraper.py](Scraper.py) remains an unfinished prototype and will need fixes before it can be used.
- The active scraper relies on the NHL Stats API endpoint used in the code. If the API or its game feed format changes, the script may need updates.

## Inspiration

This project was inspired by https://github.com/tomljr2/ShootingPercentageByDistance/blob/master/shootingpercentage.py.
