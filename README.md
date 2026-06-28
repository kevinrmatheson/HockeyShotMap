# HockeyShotMap

HockeyShotMap is a hockey data collection project for building shot and goal datasets that can later be used for heatmaps, shooting percentage analysis, or visualization work.

## Script roles

- [Main.py](Main.py) is the active NHL Stats API scraper. It loops through regular-season games, extracts shot and goal events, and appends them to [2018NHLShotInfoV2.csv](2018NHLShotInfoV2.csv).
- [Scraper.py](Scraper.py) is a separate, experimental scraper aimed at HockeyDB. It looks like a different project path from [DataCleaning.py](DataCleaning.py), not a replacement for it.
- [DataCleaning.py](DataCleaning.py) is currently incomplete and does not run as-is.

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

The scraper builds NHL game URLs in this format:

`https://statsapi.web.nhl.com/api/v1/game/{season}{game_type}{game_id}/feed/live`

By default, the script uses:

- `YEAR = "2013"`
- regular season games only (`02`)
- game ids from `0001` through `1271`

It loops through the API response, looks for `Goal` and `Shot` plays, and writes each row directly to the output CSV.

## What [Scraper.py](Scraper.py) appears to do

[Scraper.py](Scraper.py) imports `BeautifulSoup`, `requests`, and `numpy`, then defines functions for scraping league, team, and player pages from HockeyDB.

It does not appear to be wired into the rest of the project yet, and it has several issues that would need to be fixed before it can run:

- `BeautifulSoup` is being given a URL string directly instead of fetched HTML.
- `League_Dict` is treated like a function in one place even though it is a dictionary.
- Several loops iterate over empty lists instead of the parsed results.
- `__main__` calls `league_year_scrape(OHL, 2019)` without quoting `OHL`.

## Requirements

- Python 3.x
- `requests`
- `pandas` for the cleaning script
- `beautifulsoup4` and `numpy` for the HockeyDB scraper prototype

## Running the active scraper

1. Install dependencies.
2. Open [Main.py](Main.py) and set `YEAR` to the season you want to collect.
3. Run the script.

The scraper opens [2018NHLShotInfoV2.csv](2018NHLShotInfoV2.csv) in append mode, so delete or rename the file first if you want a fresh export.

## Data files

- [2018NHLShotInfo.csv](2018NHLShotInfo.csv) - sample or earlier export data
- [2018NHLShotInfoV2.csv](2018NHLShotInfoV2.csv) - current scraper output target

## Current limitations

- [DataCleaning.py](DataCleaning.py) is not ready to run as-is; it contains an incomplete data-loading stub and a typo in the pandas call (`pd.read.csv` should be `pd.read_csv`).
- [Scraper.py](Scraper.py) looks like an unfinished prototype and will need fixes before it can be used.
- The active scraper relies on the NHL Stats API endpoint used in the code. If the API or its game feed format changes, the script may need updates.

## Inspiration

This project was inspired by https://github.com/tomljr2/ShootingPercentageByDistance/blob/master/shootingpercentage.py.
