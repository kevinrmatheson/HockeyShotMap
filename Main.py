import argparse
import csv
import hashlib
import json
import logging
import sqlite3
from datetime import datetime
from dataclasses import dataclass

import requests

LEGACY_API_BASE_URL = "https://statsapi.web.nhl.com/api/v1"
WEB_API_BASE_URL = "https://api-web.nhle.com/v1"
STATS_REST_BASE_URL = "https://api.nhle.com/stats/rest"
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
   "API_Source",
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
   # Legacy Stats API expects game id zero-padded to 4 digits.
   return f"{LEGACY_API_BASE_URL}/game/{season}{game_type}{game_id:04d}/feed/live"


def build_web_play_by_play_url(season: str, game_type: str, game_id: int) -> str:
   # Web API gamecenter uses full NHL game ID (season + game type + game number).
   full_game_id = int(f"{season}{game_type}{game_id:04d}")
   return f"{WEB_API_BASE_URL}/gamecenter/{full_game_id}/play-by-play"


def _fetch_json(url: str, timeout_seconds: int, params: dict | None = None) -> dict | None:
   try:
      response = requests.get(url, timeout=timeout_seconds, params=params)
      response.raise_for_status()
      return response.json()
   except (requests.RequestException, ValueError) as exc:
      logging.warning("Failed to fetch %s: %s", url, exc)
      return None


def preflight_api_check(timeout_seconds: int) -> None:
   # Fail fast with a clear message if the NHL API host is unavailable.
   payload = _fetch_json(f"{LEGACY_API_BASE_URL}/seasons", timeout_seconds)
   if payload is None:
      raise RuntimeError(
         "NHL API is unreachable. Check network/DNS access to statsapi.web.nhl.com and retry."
      )


def detect_available_sources(api_source: str, timeout_seconds: int) -> list[str]:
   requested = ["legacy", "web"] if api_source == "both" else [api_source]
   available = []

   if "legacy" in requested:
      if _fetch_json(f"{LEGACY_API_BASE_URL}/seasons", timeout_seconds) is not None:
         available.append("legacy")
      else:
         logging.warning("Legacy Stats API unavailable: %s", LEGACY_API_BASE_URL)

   if "web" in requested:
      if _fetch_json(f"{WEB_API_BASE_URL}/season", timeout_seconds) is not None:
         available.append("web")
      else:
         logging.warning("Web API unavailable: %s", WEB_API_BASE_URL)

   if not available:
      raise RuntimeError("No requested NHL API sources are reachable. Check DNS/network access.")

   return available


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


def _season_id_yyyyyyyy(start_year: str) -> str:
   year = int(start_year)
   return f"{year}{year + 1}"


def _player_position_code(player: dict) -> str | None:
   # Roster payloads use a few different shapes, so we check the common ones.
   position = player.get("position") or player.get("positionCode") or player.get("position_code") or {}
   if isinstance(position, dict):
      return position.get("code") or position.get("abbreviation") or position.get("positionCode")
   if isinstance(position, str):
      return position
   return player.get("positionCode") or player.get("position")


def _extract_team_catalog_entries(payload: dict | list) -> list[dict]:
   if isinstance(payload, list):
      return [entry for entry in payload if isinstance(entry, dict)]

   for key in ("data", "teams", "items", "response", "records"):
      value = payload.get(key)
      if isinstance(value, list):
         return [entry for entry in value if isinstance(entry, dict)]

   return [payload] if isinstance(payload, dict) else []


def _team_catalog_key(candidate: dict) -> str | None:
   for key in ("triCode", "abbrev", "abbreviation", "teamAbbrev", "teamCode", "code"):
      value = candidate.get(key)
      if value:
         return str(value)
   return None


def fetch_team_catalog(timeout_seconds: int) -> dict[str, int]:
   # Build a team-code-to-team-id map so the deep crawl can query team detail endpoints.
   payload = _fetch_json(f"{STATS_REST_BASE_URL}/en/team", timeout_seconds)
   if not payload:
      return {}

   catalog: dict[str, int] = {}
   for team in _extract_team_catalog_entries(payload):
      team_id = team.get("id") or team.get("teamId") or team.get("team_id")
      if team_id is None:
         continue

      try:
         team_id_int = int(team_id)
      except (TypeError, ValueError):
         continue

      for key in (
         _team_catalog_key(team),
         str(team.get("triCode") or "").upper(),
         str(team.get("abbrev") or "").upper(),
      ):
         if key:
            catalog[key.upper()] = team_id_int

   return catalog


def _player_id_from_roster_entry(player: dict) -> int | None:
   candidate_values = [
      player.get("playerId"),
      player.get("id"),
      player.get("person", {}).get("id") if isinstance(player.get("person"), dict) else None,
   ]
   for candidate in candidate_values:
      try:
         if candidate is not None:
            return int(candidate)
      except (TypeError, ValueError):
         continue
   return None


def _extract_roster_players(payload: dict | list) -> list[dict]:
   # The roster payload shape can vary, so this helper looks for player-like dictionaries recursively.
   players: list[dict] = []

   def walk(value: object) -> None:
      if isinstance(value, dict):
         if isinstance(value.get("person"), dict) and value["person"].get("id"):
            players.append(value)
         elif value.get("playerId") or value.get("id"):
            name = value.get("name") or value.get("fullName") or value.get("playerName")
            if name or value.get("playerId") or value.get("id"):
               players.append(value)
         for nested_value in value.values():
            walk(nested_value)
      elif isinstance(value, list):
         for item in value:
            walk(item)

   walk(payload)

   deduped: list[dict] = []
   seen_ids: set[int] = set()
   for player in players:
      player_id = _player_id_from_roster_entry(player)
      if player_id is None or player_id in seen_ids:
         continue
      seen_ids.add(player_id)
      deduped.append(player)

   return deduped


def build_web_roster_url(team_code: str, season: str) -> str:
   # Team rosters are exposed by the Web API using the three-letter team code.
   season_id = _season_id_yyyyyyyy(season)
   return f"{WEB_API_BASE_URL}/roster/{team_code}/{season_id}"


def build_web_edge_team_detail_url(team_id: int, season: str, game_type: str) -> str:
   # Team detail includes shot attempts over 90, bursts over 22, distance per 60, and zone-time summaries.
   season_id = _season_id_yyyyyyyy(season)
   return f"{WEB_API_BASE_URL}/edge/team-detail/{team_id}/{season_id}/{int(game_type)}"


def build_web_edge_skater_detail_url(player_id: int, season: str, game_type: str) -> str:
   # Skater detail includes top shot speed, skating speed, distance skated, shot-on-goal summaries, and zone starts.
   season_id = _season_id_yyyyyyyy(season)
   return f"{WEB_API_BASE_URL}/edge/skater-detail/{player_id}/{season_id}/{int(game_type)}"


def build_web_edge_goalie_detail_url(player_id: int, season: str, game_type: str) -> str:
   # Goalie detail includes GAA, games above .900, goal differential per 60, goal support, and shot-location summaries.
   season_id = _season_id_yyyyyyyy(season)
   return f"{WEB_API_BASE_URL}/edge/goalie-detail/{player_id}/{season_id}/{int(game_type)}"


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
      "API_Source": "legacy",
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


def parse_web_shot_events(game_json: dict, season: str, game_id: int) -> list[dict]:
   # Parse shot and goal events from the newer gamecenter play-by-play payload.
   home_code = (
      game_json.get("homeTeam", {}).get("abbrev")
      or game_json.get("gameData", {}).get("teams", {}).get("home", {}).get("triCode")
   )
   away_code = (
      game_json.get("awayTeam", {}).get("abbrev")
      or game_json.get("gameData", {}).get("teams", {}).get("away", {}).get("triCode")
   )

   plays = game_json.get("plays", [])
   rows = []
   for play in plays:
      event_key = str(play.get("typeDescKey", "")).lower()
      if event_key not in {"goal", "shot-on-goal", "shot"}:
         continue

      details = play.get("details", {})
      x_coord = details.get("xCoord")
      y_coord = details.get("yCoord")
      if x_coord is None or y_coord is None:
         continue

      team_code = details.get("eventOwnerTeamTricode") or play.get("team", {}).get("triCode")
      shot_type = details.get("shotType") or details.get("secondaryType") or "Unknown"
      shooter = details.get("shootingPlayerName") or details.get("scoringPlayerName") or "Unknown"
      period = (
         play.get("periodDescriptor", {}).get("number")
         or play.get("about", {}).get("period")
      )

      rows.append(
         {
            "Shot": "Goal" if event_key == "goal" else "ngshot",
            "X": float(x_coord),
            "Y": float(y_coord),
            "Shot_Type": shot_type,
            "Shooter": shooter,
            "Team": team_code,
            "Home_Away": _home_away_value(team_code, home_code, away_code),
            "Period": period,
            "Year": season,
            "GameID": game_id,
            "API_Source": "web",
         }
      )

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
            game_id INTEGER NOT NULL,
            api_source TEXT NOT NULL DEFAULT 'legacy'
         )
         """
      )
      # Backward-compatible migration for existing databases.
      cursor.execute("PRAGMA table_info(shots)")
      columns = {row[1] for row in cursor.fetchall()}
      if "api_source" not in columns:
         cursor.execute("ALTER TABLE shots ADD COLUMN api_source TEXT NOT NULL DEFAULT 'legacy'")

      cursor.execute(
         """
         CREATE TABLE IF NOT EXISTS edge_payloads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            season TEXT NOT NULL,
            game_type TEXT NOT NULL,
            endpoint TEXT NOT NULL,
            source TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            fetched_at TEXT NOT NULL,
            UNIQUE(season, game_type, endpoint, source)
         )
         """
      )

      cursor.execute(
         """
         CREATE TABLE IF NOT EXISTS edge_detail_payloads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            season TEXT NOT NULL,
            game_type TEXT NOT NULL,
            source TEXT NOT NULL,
            snapshot_type TEXT NOT NULL,
            entity_type TEXT NOT NULL,
            entity_id TEXT NOT NULL,
            endpoint TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            fetched_at TEXT NOT NULL,
            UNIQUE(season, game_type, source, snapshot_type, entity_type, entity_id, endpoint)
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
               home_away, period, season, game_id, api_source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
               row.get("API_Source", "legacy"),
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
         SELECT shot_result, x, y, shot_type, shooter, team, home_away, period, season, game_id, api_source
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
               "API_Source": row[10],
            }
         )
   return len(rows)


def capture_edge_summary_snapshots(db_path: str, season: str, game_type: str, timeout_seconds: int) -> int:
   # Store raw NHL Edge summary snapshots so downstream analysis can evolve without re-scraping.
   # These landing pages summarize team and player context such as top shot speed,
   # skating speed, high-danger SOG, distance skated, and zone-time totals.
   season_id = _season_id_yyyyyyyy(season)
   edge_game_type = str(int(game_type))
   endpoints = [
      f"{WEB_API_BASE_URL}/edge/team-landing/{season_id}/{edge_game_type}",
      f"{WEB_API_BASE_URL}/edge/skater-landing/{season_id}/{edge_game_type}",
      f"{WEB_API_BASE_URL}/edge/goalie-landing/{season_id}/{edge_game_type}",
      f"{WEB_API_BASE_URL}/edge/by-the-numbers",
   ]

   inserted = 0
   fetched_at = datetime.utcnow().isoformat(timespec="seconds")
   with sqlite3.connect(db_path) as connection:
      cursor = connection.cursor()
      for endpoint in endpoints:
         payload = _fetch_json(endpoint, timeout_seconds)
         if payload is None:
            continue

         cursor.execute(
            """
            INSERT OR REPLACE INTO edge_payloads (season, game_type, endpoint, source, payload_json, fetched_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
               season,
               game_type,
               endpoint,
               "web",
               json.dumps(payload),
               fetched_at,
            ),
         )
         inserted += 1
      connection.commit()

   return inserted


def _store_edge_detail_payload(
   db_path: str,
   season: str,
   game_type: str,
   source: str,
   snapshot_type: str,
   entity_type: str,
   entity_id: str,
   endpoint: str,
   payload: dict,
) -> None:
   fetched_at = datetime.utcnow().isoformat(timespec="seconds")
   with sqlite3.connect(db_path) as connection:
      cursor = connection.cursor()
      cursor.execute(
         """
         INSERT OR REPLACE INTO edge_detail_payloads
         (season, game_type, source, snapshot_type, entity_type, entity_id, endpoint, payload_json, fetched_at)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
         """,
         (
            season,
            game_type,
            source,
            snapshot_type,
            entity_type,
            str(entity_id),
            endpoint,
            json.dumps(payload),
            fetched_at,
         ),
      )
      connection.commit()


def capture_edge_deep_snapshots(db_path: str, season: str, game_type: str, timeout_seconds: int, team_codes: set[str]) -> int:
   # Deep crawl captures team detail, roster, and player detail snapshots.
   # Team detail captures team-level Edge stats such as shot attempts over 90, bursts over 22,
   # distance per 60, and zone-time summaries.
   # Skater detail captures player shot speed, skating speed, distance skated, shot-on-goal
   # summaries/details, and zone-start/zone-time breakdowns.
   # Goalie detail captures GAA, games above .900, goal differential per 60, goal support,
   # point percentage, and goalie shot-location summaries/details.
   if not team_codes:
      logging.warning("No team codes were collected for season %s, skipping Edge deep crawl.", season)
      return 0

   team_catalog = fetch_team_catalog(timeout_seconds)
   if not team_catalog:
      logging.warning("Could not load team catalog from stats REST; skipping Edge deep crawl for season %s.", season)
      return 0

   total_snapshots = 0
   edge_game_type = str(int(game_type))

   for team_code in sorted(team_codes):
      team_id = team_catalog.get(team_code.upper())
      if team_id is None:
         logging.warning("Could not map team code %s to a team id; skipping team detail.", team_code)
         continue

      team_detail_endpoint = build_web_edge_team_detail_url(team_id, season, game_type)
      team_detail_payload = _fetch_json(team_detail_endpoint, timeout_seconds)
      if team_detail_payload is not None:
         _store_edge_detail_payload(
            db_path,
            season,
            edge_game_type,
            "web",
            "team-detail",
            "team",
            str(team_id),
            team_detail_endpoint,
            team_detail_payload,
         )
         total_snapshots += 1

      roster_endpoint = build_web_roster_url(team_code, season)
      roster_payload = _fetch_json(roster_endpoint, timeout_seconds)
      if roster_payload is None:
         continue

      _store_edge_detail_payload(
         db_path,
         season,
         edge_game_type,
         "web",
         "roster",
         "team",
         team_code,
         roster_endpoint,
         roster_payload,
      )
      total_snapshots += 1

      for player in _extract_roster_players(roster_payload):
         player_id = _player_id_from_roster_entry(player)
         if player_id is None:
            continue

         position_code = _player_position_code(player)
         is_goalie = str(position_code).upper() == "G"

         if is_goalie:
            player_endpoint = build_web_edge_goalie_detail_url(player_id, season, game_type)
            snapshot_type = "goalie-detail"
         else:
            player_endpoint = build_web_edge_skater_detail_url(player_id, season, game_type)
            snapshot_type = "skater-detail"

         player_payload = _fetch_json(player_endpoint, timeout_seconds)
         if player_payload is None:
            continue

         _store_edge_detail_payload(
            db_path,
            season,
            edge_game_type,
            "web",
            snapshot_type,
            "player",
            str(player_id),
            player_endpoint,
            player_payload,
         )
         total_snapshots += 1

   return total_snapshots


def run_season_scrape(config: ScrapeConfig, active_sources: list[str], capture_edge: bool, capture_edge_deep: bool) -> tuple[int, int, int, int]:
   # Orchestrates fetch -> parse -> persist for a game range.
   initialize_database(config.db_path)

   games_processed = 0
   rows_inserted = 0
   edge_payloads_inserted = 0
   edge_detail_payloads_inserted = 0
   team_codes_seen: set[str] = set()

   for game_id in range(config.start_game, config.end_game + 1):
      rows = []

      if "legacy" in active_sources:
         legacy_url = build_game_url(config.season, config.game_type, game_id)
         legacy_json = fetch_game_feed(legacy_url, config.timeout_seconds)
         if legacy_json is not None:
            rows.extend(parse_shot_events(legacy_json, config.season, game_id))

      if "web" in active_sources:
         web_url = build_web_play_by_play_url(config.season, config.game_type, game_id)
         web_json = _fetch_json(web_url, config.timeout_seconds)
         if web_json is not None:
            rows.extend(parse_web_shot_events(web_json, config.season, game_id))

      if not rows:
         continue

      inserted = persist_rows(config.db_path, rows)
      games_processed += 1
      rows_inserted += inserted
      for row in rows:
         team_code = row.get("Team")
         if team_code:
            team_codes_seen.add(str(team_code).upper())

      logging.info(
         "Game %s parsed rows=%s inserted=%s",
         game_id,
         len(rows),
         inserted,
      )

   if capture_edge and "web" in active_sources:
      edge_payloads_inserted = capture_edge_summary_snapshots(
         config.db_path,
         config.season,
         config.game_type,
         config.timeout_seconds,
      )
      logging.info(
         "Season %s captured edge_payloads=%s",
         config.season,
         edge_payloads_inserted,
      )

   if capture_edge_deep and "web" in active_sources:
      edge_detail_payloads_inserted = capture_edge_deep_snapshots(
         config.db_path,
         config.season,
         config.game_type,
         config.timeout_seconds,
         team_codes_seen,
      )
      logging.info(
         "Season %s captured edge_detail_payloads=%s",
         config.season,
         edge_detail_payloads_inserted,
      )

   return games_processed, rows_inserted, edge_payloads_inserted, edge_detail_payloads_inserted


def current_nhl_season_start_year(today: datetime | None = None) -> int:
   # NHL seasons roll over in the fall, so Jan-Aug belongs to previous start year.
   today = today or datetime.now()
   return today.year if today.month >= 9 else today.year - 1


def discover_earliest_full_season(game_type: str, timeout_seconds: int, preferred_source: str = "legacy") -> int:
   # Detect earliest season using the preferred source, with safe fallback.
   if preferred_source == "web":
      seasons_payload = _fetch_json(f"{WEB_API_BASE_URL}/season", timeout_seconds)
      if not seasons_payload:
         return DEFAULT_START_SEASON_FALLBACK

      season_ids = []
      if isinstance(seasons_payload, list):
         candidate_list = seasons_payload
      else:
         candidate_list = seasons_payload.get("seasons", [])

      for season in candidate_list:
         season_id = str(season)
         if isinstance(season, dict):
            season_id = str(season.get("id", season.get("seasonId", "")))
         if len(season_id) == 8 and season_id.isdigit():
            season_ids.append(season_id)

      if season_ids:
         return int(sorted(season_ids)[0][:4])
      return DEFAULT_START_SEASON_FALLBACK

   # Legacy mode: verify schedule and game-feed availability.
   seasons_payload = _fetch_json(f"{LEGACY_API_BASE_URL}/seasons", timeout_seconds)
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
         f"{LEGACY_API_BASE_URL}/schedule",
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

      first_feed = _fetch_json(f"{LEGACY_API_BASE_URL}/game/{first_game_pk}/feed/live", timeout_seconds)
      last_feed = _fetch_json(f"{LEGACY_API_BASE_URL}/game/{last_game_pk}/feed/live", timeout_seconds)
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
   parser.add_argument("--api-source", default="both", choices=["legacy", "web", "both"], help="Choose legacy API, newer web API, or both.")
   parser.add_argument("--capture-edge", action="store_true", help="Capture NHL Edge summary payloads when web API is enabled.")
   parser.add_argument("--capture-edge-deep", action="store_true", help="Capture NHL Edge team and player detail payloads after each season.")
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

   # Detect source reachability once up front to avoid noisy per-game failures.
   active_sources = detect_available_sources(args.api_source, args.timeout)
   logging.info("Using API sources: %s", ", ".join(active_sources))

   if args.season is not None and (args.start_season is not None or args.end_season is not None):
      raise ValueError("Use either --season or --start-season/--end-season, not both.")

   if args.season is not None:
      seasons_to_run = [str(args.season)]
   else:
      discovery_source = "legacy" if "legacy" in active_sources else "web"
      resolved_start = args.start_season if args.start_season is not None else discover_earliest_full_season(
         args.game_type,
         args.timeout,
         preferred_source=discovery_source,
      )
      logging.info("Using start season %s", resolved_start)
      seasons_to_run = season_range(resolved_start, args.end_season)

   total_games_processed = 0
   total_rows_inserted = 0
   total_edge_payloads = 0
   total_edge_detail_payloads = 0

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
      games_processed, rows_inserted, edge_payloads, edge_detail_payloads = run_season_scrape(
         config,
         active_sources,
         args.capture_edge,
         args.capture_edge_deep,
      )
      total_games_processed += games_processed
      total_rows_inserted += rows_inserted
      total_edge_payloads += edge_payloads
      total_edge_detail_payloads += edge_detail_payloads
      logging.info(
         "Finished season %s. games_processed=%s rows_inserted=%s edge_payloads=%s edge_detail_payloads=%s",
         season,
         games_processed,
         rows_inserted,
         edge_payloads,
         edge_detail_payloads,
      )

   logging.info(
      "Finished scrape range. seasons=%s games_processed=%s rows_inserted=%s edge_payloads=%s edge_detail_payloads=%s",
      len(seasons_to_run),
      total_games_processed,
      total_rows_inserted,
      total_edge_payloads,
      total_edge_detail_payloads,
   )

   if args.export_csv:
      exported = export_to_csv(args.db_path, args.export_csv)
      logging.info("Exported %s rows to %s", exported, args.export_csv)


if __name__ == "__main__":
   main()

