import argparse
import csv
import hashlib
import logging
import sqlite3
from datetime import datetime
from dataclasses import dataclass

import requests

NHL_API_BASE_URL = "https://statsapi.web.nhl.com/api/v1"
DEFAULT_START_SEASON_FALLBACK = 2010

# Supported NHL game type codes from the stats API.
PRESEASON = "01"
REGULAR_SEASON = "02"
PLAYOFFS = "03"
ALLSTAR = "04"

# Legacy CSV column order preserved for Tableau compatibility exports.
CSV_FIELDNAMES = [
   "Shot",
   "X",
   "Y",
   "Shot_Type",
   "Shooter",
   "Team",
   "Home_Away",
   "Period",
   "Year",
   "GameID",
]


@dataclass
class ScrapeConfig:
   # Runtime settings for one scrape run.
   season: str = "2013"
   game_type: str = REGULAR_SEASON
   start_game: int = 1
   end_game: int = 1271
   timeout_seconds: int = 10
   db_path: str = "hockey_shots.db"
   export_csv: str | None = None


def build_game_url(season: str, game_type: str, game_id: int) -> str:
   # NHL API expects game id zero-padded to 4 digits.
   return f"{NHL_API_BASE_URL}/game/{season}{game_type}{game_id:04d}/feed/live"


def _fetch_json(url: str, timeout_seconds: int, params: dict | None = None) -> dict | None:
   try:
      response = requests.get(url, timeout=timeout_seconds, params=params)
      response.raise_for_status()
      return response.json()
   except (requests.RequestException, ValueError) as exc:
      logging.warning("Failed to fetch %s: %s", url, exc)
      return None


def fetch_game_feed(url: str, timeout_seconds: int) -> dict | None:
   # Network and JSON errors are logged and treated as a skipped game.
   return _fetch_json(url, timeout_seconds)


def _extract_home_away(game_json: dict) -> tuple[str | None, str | None]:
   # Pull team tri-codes once and reuse during event normalization.
   teams = game_json.get("gameData", {}).get("teams", {})
   home = teams.get("home", {}).get("triCode")
   away = teams.get("away", {}).get("triCode")
   return home, away


def _extract_shooter(play: dict) -> str:
   # The first named player is used as the shooter in this dataset.
   players = play.get("players", [])
   for player in players:
      full_name = player.get("player", {}).get("fullName")
      if full_name:
         return full_name
   return "Unknown"


def _home_away_value(team_code: str | None, home_code: str | None, away_code: str | None) -> int | None:
   # Keep existing convention: 1 for home, 0 for away, None if unknown.
   if team_code and home_code and team_code == home_code:
      return 1
   if team_code and away_code and team_code == away_code:
      return 0
   return None


def normalize_event_to_row(play: dict, season: str, game_id: int, home_code: str | None, away_code: str | None) -> dict | None:
   # Only include shot attempts that became a Shot or Goal event.
   result = play.get("result", {})
   event_name = result.get("event")
   if event_name not in {"Goal", "Shot"}:
      return None

   # Skip events missing rink coordinates because location is required downstream.
   coordinates = play.get("coordinates", {})
   if "x" not in coordinates or "y" not in coordinates:
      return None

   team_code = play.get("team", {}).get("triCode")
   period = play.get("about", {}).get("period")
   shot_result = "Goal" if event_name == "Goal" else "ngshot"
   shot_type = result.get("secondaryType", "Unknown")
   shooter = _extract_shooter(play)

   return {
      "Shot": shot_result,
      "X": float(coordinates["x"]),
      "Y": float(coordinates["y"]),
      "Shot_Type": shot_type,
      "Shooter": shooter,
      "Team": team_code,
      "Home_Away": _home_away_value(team_code, home_code, away_code),
      "Period": period,
      "Year": season,
      "GameID": game_id,
   }


def parse_shot_events(game_json: dict, season: str, game_id: int) -> list[dict]:
   # Convert one game's play stream into normalized shot rows.
   home_code, away_code = _extract_home_away(game_json)
   plays = game_json.get("liveData", {}).get("plays", {}).get("allPlays", [])
   rows = []
   for play in plays:
      row = normalize_event_to_row(play, season, game_id, home_code, away_code)
      if row is None:
         continue
      rows.append(row)
   return rows


def _event_hash(row: dict) -> str:
   # Deterministic signature used to make reruns idempotent.
   signature = (
      f"{row['Year']}|{row['GameID']}|{row['Period']}|{row['Team']}|{row['Shooter']}|"
      f"{row['X']}|{row['Y']}|{row['Shot_Type']}|{row['Shot']}"
   )
   return hashlib.sha256(signature.encode("utf-8")).hexdigest()


def initialize_database(db_path: str) -> None:
   # Create storage table once; safe to call on every run.
   with sqlite3.connect(db_path) as connection:
      cursor = connection.cursor()
      cursor.execute(
         """
         CREATE TABLE IF NOT EXISTS shots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_hash TEXT NOT NULL UNIQUE,
            shot_result TEXT NOT NULL,
            x REAL NOT NULL,
            y REAL NOT NULL,
            shot_type TEXT,
            shooter TEXT,
            team TEXT,
            home_away INTEGER,
            period INTEGER,
            season TEXT NOT NULL,
            game_id INTEGER NOT NULL
         )
         """
      )
      connection.commit()


def persist_rows(db_path: str, rows: list[dict]) -> int:
   # INSERT OR IGNORE prevents duplicate rows when scraping the same games again.
   if not rows:
      return 0

   inserted = 0
   with sqlite3.connect(db_path) as connection:
      cursor = connection.cursor()
      for row in rows:
         cursor.execute(
            """
            INSERT OR IGNORE INTO shots (
               event_hash, shot_result, x, y, shot_type, shooter, team,
               home_away, period, season, game_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
               _event_hash(row),
               row["Shot"],
               row["X"],
               row["Y"],
               row["Shot_Type"],
               row["Shooter"],
               row["Team"],
               row["Home_Away"],
               row["Period"],
               row["Year"],
               row["GameID"],
            ),
         )
         if cursor.rowcount == 1:
            inserted += 1
      connection.commit()
   return inserted


def export_to_csv(db_path: str, csv_path: str) -> int:
   # Optional legacy export so existing Tableau workflows still work.
   with sqlite3.connect(db_path) as connection:
      cursor = connection.cursor()
      cursor.execute(
         """
         SELECT shot_result, x, y, shot_type, shooter, team, home_away, period, season, game_id
         FROM shots
         ORDER BY season, game_id, id
         """
      )
      rows = cursor.fetchall()

   with open(csv_path, "w", newline="", encoding="utf-8") as output_file:
      writer = csv.DictWriter(output_file, delimiter=",", fieldnames=CSV_FIELDNAMES)
      writer.writeheader()
      for row in rows:
         writer.writerow(
            {
               "Shot": row[0],
               "X": row[1],
               "Y": row[2],
               "Shot_Type": row[3],
               "Shooter": row[4],
               "Team": row[5],
               "Home_Away": row[6],
               "Period": row[7],
               "Year": row[8],
               "GameID": row[9],
            }
         )
   return len(rows)


def run_season_scrape(config: ScrapeConfig) -> tuple[int, int]:
   # Orchestrates fetch -> parse -> persist for a game range.
   initialize_database(config.db_path)

   games_processed = 0
   rows_inserted = 0

   for game_id in range(config.start_game, config.end_game + 1):
      url = build_game_url(config.season, config.game_type, game_id)
      game_json = fetch_game_feed(url, config.timeout_seconds)
      if game_json is None:
         continue

      rows = parse_shot_events(game_json, config.season, game_id)
      inserted = persist_rows(config.db_path, rows)
      games_processed += 1
      rows_inserted += inserted

      logging.info(
         "Game %s parsed rows=%s inserted=%s",
         game_id,
         len(rows),
         inserted,
      )

   return games_processed, rows_inserted


def current_nhl_season_start_year(today: datetime | None = None) -> int:
   # NHL seasons roll over in the fall, so Jan-Aug belongs to previous start year.
   today = today or datetime.now()
   return today.year if today.month >= 9 else today.year - 1


def discover_earliest_full_season(game_type: str, timeout_seconds: int) -> int:
   # Detect earliest season where schedule and game-feed endpoints both respond.
   seasons_payload = _fetch_json(f"{NHL_API_BASE_URL}/seasons", timeout_seconds)
   if not seasons_payload:
      return DEFAULT_START_SEASON_FALLBACK

   season_ids = []
   for season in seasons_payload.get("seasons", []):
      season_id = str(season.get("seasonId", ""))
      if len(season_id) == 8 and season_id.isdigit():
         season_ids.append(season_id)

   for season_id in sorted(season_ids):
      start_year = int(season_id[:4])
      schedule = _fetch_json(
         f"{NHL_API_BASE_URL}/schedule",
         timeout_seconds,
         params={"season": season_id, "gameType": game_type},
      )
      if not schedule or schedule.get("totalItems", 0) == 0:
         continue

      dates = schedule.get("dates", [])
      if not dates or not dates[0].get("games") or not dates[-1].get("games"):
         continue

      first_game_pk = dates[0]["games"][0].get("gamePk")
      last_game_pk = dates[-1]["games"][-1].get("gamePk")
      if not first_game_pk or not last_game_pk:
         continue

      first_feed = _fetch_json(f"{NHL_API_BASE_URL}/game/{first_game_pk}/feed/live", timeout_seconds)
      last_feed = _fetch_json(f"{NHL_API_BASE_URL}/game/{last_game_pk}/feed/live", timeout_seconds)
      if first_feed and last_feed:
         return start_year

   return DEFAULT_START_SEASON_FALLBACK


def season_range(start_season: int, end_season: int | None) -> list[str]:
   # Build inclusive season start-year labels, for example 2013, 2014, ... 2025.
   resolved_end = end_season if end_season is not None else current_nhl_season_start_year()
   if resolved_end < start_season:
      raise ValueError("end season must be greater than or equal to start season")
   return [str(year) for year in range(start_season, resolved_end + 1)]


def parse_args() -> argparse.Namespace:
   # CLI keeps scrape settings out of the source code.
   parser = argparse.ArgumentParser(description="Scrape NHL shot and goal events into SQLite.")
   parser.add_argument("--season", default=None, help="Single season start year (for example 2013 for 2013-2014).")
   parser.add_argument("--start-season", type=int, default=None, help="Start season for multi-season run. Defaults to earliest full API season.")
   parser.add_argument("--end-season", type=int, default=None, help="Optional end season for multi-season run. Defaults to current NHL season.")
   parser.add_argument("--game-type", default=REGULAR_SEASON, choices=[PRESEASON, REGULAR_SEASON, PLAYOFFS, ALLSTAR])
   parser.add_argument("--start-game", type=int, default=1)
   parser.add_argument("--end-game", type=int, default=1271)
   parser.add_argument("--timeout", type=int, default=10, help="HTTP timeout in seconds.")
   parser.add_argument("--db-path", default="hockey_shots.db", help="SQLite database path.")
   parser.add_argument("--export-csv", default=None, help="Optional legacy CSV export path.")
   parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
   return parser.parse_args()


def main() -> None:
   # Entrypoint: read args, run scrape, optionally export CSV.
   args = parse_args()
   logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")

   if args.season is not None and (args.start_season is not None or args.end_season is not None):
      raise ValueError("Use either --season or --start-season/--end-season, not both.")

   if args.season is not None:
      seasons_to_run = [str(args.season)]
   else:
      resolved_start = args.start_season if args.start_season is not None else discover_earliest_full_season(args.game_type, args.timeout)
      logging.info("Using start season %s", resolved_start)
      seasons_to_run = season_range(resolved_start, args.end_season)

   total_games_processed = 0
   total_rows_inserted = 0

   for season in seasons_to_run:
      logging.info("Starting season %s", season)
      config = ScrapeConfig(
         season=season,
         game_type=args.game_type,
         start_game=args.start_game,
         end_game=args.end_game,
         timeout_seconds=args.timeout,
         db_path=args.db_path,
         export_csv=None,
      )
      games_processed, rows_inserted = run_season_scrape(config)
      total_games_processed += games_processed
      total_rows_inserted += rows_inserted
      logging.info(
         "Finished season %s. games_processed=%s rows_inserted=%s",
         season,
         games_processed,
         rows_inserted,
      )

   logging.info(
      "Finished scrape range. seasons=%s games_processed=%s rows_inserted=%s",
      len(seasons_to_run),
      total_games_processed,
      total_rows_inserted,
   )

   if args.export_csv:
      exported = export_to_csv(args.db_path, args.export_csv)
      logging.info("Exported %s rows to %s", exported, args.export_csv)


if __name__ == "__main__":
   main()

