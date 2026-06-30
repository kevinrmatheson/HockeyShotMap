import argparse
import csv
import hashlib
import json
import logging
import sqlite3
import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from dataclasses import dataclass

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

WEB_API_BASE_URL = "https://api-web.nhle.com/v1"
STATS_REST_BASE_URL = "https://api.nhle.com/stats/rest"
# First season observed to include reliable x/y shot coordinates.
COORDINATE_DATA_START_SEASON = 2005

# Performance tuning defaults for long historical runs.
DB_COMMIT_INTERVAL_GAMES = 50
SEASON_END_PROBE_START_GAME = 50
SEASON_END_EMPTY_STREAK_GAMES = 50
DEFAULT_MAX_WORKERS = 4
DEFAULT_REQUEST_DELAY_SECONDS = 0.05
HEARTBEAT_LOG_INTERVAL_SECONDS = 20
HTTP_RETRY_TOTAL = 5
HTTP_RETRY_BACKOFF_FACTOR = 0.6
HTTP_RATE_LIMIT_STATUSES = {429, 500, 502, 503, 504}

# Supported NHL game type codes from the stats API.
PRESEASON = "01"
REGULAR_SEASON = "02"
PLAYOFFS = "03"
ALLSTAR = "04"

# Stable CSV column order preserved for Tableau compatibility exports.
CSV_FIELDNAMES = [
   "Shot",
   "X",
   "Y",
   "Shot_Type",
   "Shooter",
   "Shooter_ID",
   "Team",
   "Home_Away",
   "Period",
   "Period_Time",
   "Period_Time_Remaining",
   "Year",
   "GameID",
   "API_Source",
   "Goalie",
   "Goalie_ID",
   "Shot_Distance",
   "Shot_Angle",
   "Is_Empty_Net",
   "Strength_State",
   "Score_Differential",
   "Zone",
   "Event_ID",
]


_HTTP_SESSION = requests.Session()
_THREAD_LOCAL = threading.local()


def _build_retry_adapter() -> HTTPAdapter:
   retry = Retry(
      total=HTTP_RETRY_TOTAL,
      connect=HTTP_RETRY_TOTAL,
      read=HTTP_RETRY_TOTAL,
      status=HTTP_RETRY_TOTAL,
      backoff_factor=HTTP_RETRY_BACKOFF_FACTOR,
      status_forcelist=HTTP_RATE_LIMIT_STATUSES,
      allowed_methods=frozenset({"GET"}),
      raise_on_status=False,
   )
   return HTTPAdapter(max_retries=retry, pool_connections=DEFAULT_MAX_WORKERS * 2, pool_maxsize=DEFAULT_MAX_WORKERS * 2)


_HTTP_SESSION.mount("http://", _build_retry_adapter())
_HTTP_SESSION.mount("https://", _build_retry_adapter())


class _RequestRateLimiter:
   def __init__(self, minimum_interval_seconds: float) -> None:
      self.minimum_interval_seconds = max(0.0, minimum_interval_seconds)
      self._lock = threading.Lock()
      self._next_allowed_time = 0.0

   def acquire(self) -> None:
      if self.minimum_interval_seconds <= 0:
         return

      with self._lock:
         now = time.monotonic()
         if now < self._next_allowed_time:
            sleep_seconds = self._next_allowed_time - now
            self._next_allowed_time += self.minimum_interval_seconds
         else:
            sleep_seconds = 0.0
            self._next_allowed_time = now + self.minimum_interval_seconds

      if sleep_seconds > 0:
         time.sleep(sleep_seconds)


_REQUEST_RATE_LIMITER = _RequestRateLimiter(DEFAULT_REQUEST_DELAY_SECONDS)


def configure_request_rate_limit(minimum_interval_seconds: float) -> None:
   _REQUEST_RATE_LIMITER.minimum_interval_seconds = max(0.0, minimum_interval_seconds)


def _get_http_session() -> requests.Session:
   if threading.current_thread() is threading.main_thread():
      return _HTTP_SESSION

   session = getattr(_THREAD_LOCAL, "session", None)
   if session is None:
      session = requests.Session()
      adapter = _build_retry_adapter()
      session.mount("http://", adapter)
      session.mount("https://", adapter)
      _THREAD_LOCAL.session = session
   return session


@dataclass
class ScrapeConfig:
   # Runtime settings for one scrape run.
   season: str = "2013"
   game_type: str = REGULAR_SEASON
   start_game: int = 1
   end_game: int = 1271
   timeout_seconds: int = 10
   max_workers: int = DEFAULT_MAX_WORKERS
   request_delay_seconds: float = DEFAULT_REQUEST_DELAY_SECONDS
   db_path: str = "hockey_shots.db"
   export_csv: str | None = None


def build_stats_shiftcharts_url() -> str:
   return f"{STATS_REST_BASE_URL}/en/shiftcharts"


def build_stats_games_url() -> str:
   return f"{STATS_REST_BASE_URL}/en/game"


def build_web_play_by_play_url(season: str, game_type: str, game_id: int) -> str:
   # Web API gamecenter uses full NHL game ID (season + game type + game number).
   full_game_id = int(f"{season}{game_type}{game_id:04d}")
   return f"{WEB_API_BASE_URL}/gamecenter/{full_game_id}/play-by-play"


def _fetch_json(url: str, timeout_seconds: int, params: dict | None = None) -> dict | None:
   try:
      _REQUEST_RATE_LIMITER.acquire()
      response = _get_http_session().get(url, timeout=timeout_seconds, params=params)
      response.raise_for_status()
      return response.json()
   except (requests.RequestException, ValueError) as exc:
      logging.warning("Failed to fetch %s: %s", url, exc)
      return None


def _fetch_json_allow_404(url: str, timeout_seconds: int, params: dict | None = None) -> dict | None:
   # Some Edge endpoints are intermittently unavailable for certain seasons; treat 404 as optional.
   try:
      _REQUEST_RATE_LIMITER.acquire()
      response = _get_http_session().get(url, timeout=timeout_seconds, params=params)
      if response.status_code == 404:
         logging.info("Optional endpoint unavailable (404): %s", url)
         return None
      response.raise_for_status()
      return response.json()
   except (requests.RequestException, ValueError) as exc:
      logging.warning("Failed to fetch %s: %s", url, exc)
      return None


def edge_is_supported_for_season(season: str, game_type: str, timeout_seconds: int) -> bool:
   # Edge payloads include an availability index; use it to skip unsupported seasons.
   season_id = _season_id_yyyyyyyy(season)
   probe = _fetch_json_allow_404(f"{WEB_API_BASE_URL}/edge/team-landing/{season_id}/{int(game_type)}", timeout_seconds)
   if not isinstance(probe, dict):
      return False

   availability = probe.get("seasonsWithEdgeStats", [])
   if not isinstance(availability, list):
      return False

   target_season_id = int(season_id)
   target_game_type = int(game_type)
   for entry in availability:
      if not isinstance(entry, dict):
         continue
      if int(entry.get("id", 0)) != target_season_id:
         continue
      game_types = entry.get("gameTypes", [])
      if isinstance(game_types, list) and target_game_type in game_types:
         return True
   return False


def detect_available_sources(api_source: str, timeout_seconds: int) -> list[str]:
   requested = ["web", "stats"] if api_source == "both" else [api_source]
   verified = []

   if "web" in requested:
      if _fetch_json(f"{WEB_API_BASE_URL}/season", timeout_seconds) is not None:
         verified.append("web")
      else:
         logging.warning(
            "Web API preflight check failed (%s). Keeping web enabled and attempting at runtime.",
            WEB_API_BASE_URL,
         )

   if "stats" in requested:
      if _fetch_json(f"{STATS_REST_BASE_URL}/ping", timeout_seconds) is not None:
         verified.append("stats")
      else:
         logging.warning(
            "Stats REST API preflight check failed (%s). Keeping stats enabled and attempting at runtime.",
            STATS_REST_BASE_URL,
         )

   if not verified:
      logging.warning(
         "No API sources passed preflight checks. Proceeding with requested sources anyway: %s",
         ", ".join(requested),
      )

   return requested


def _home_away_value(team_code: str | None, home_code: str | None, away_code: str | None) -> int | None:
   # Keep existing convention: 1 for home, 0 for away, None if unknown.
   if team_code and home_code and team_code == home_code:
      return 1
   if team_code and away_code and team_code == away_code:
      return 0
   return None


def _first_non_none(*values: object) -> object | None:
   for value in values:
      if value is not None:
         return value
   return None


def _normalize_text(value: object | None) -> str | None:
   if value is None:
      return None
   text = str(value).strip()
   return text or None


def _coerce_int(value: object | None) -> int | None:
   if value is None:
      return None
   if isinstance(value, bool):
      return int(value)
   if isinstance(value, int):
      return value
   if isinstance(value, float):
      return int(value)
   text = str(value).strip()
   if not text:
      return None
   try:
      return int(float(text))
   except (TypeError, ValueError):
      return None


def _coerce_float(value: object | None) -> float | None:
   if value is None:
      return None
   if isinstance(value, bool):
      return float(int(value))
   if isinstance(value, (int, float)):
      return float(value)
   text = str(value).strip()
   if not text:
      return None
   try:
      return float(text)
   except (TypeError, ValueError):
      return None


def _coerce_bool_int(value: object | None) -> int | None:
   if value is None:
      return None
   if isinstance(value, bool):
      return int(value)
   if isinstance(value, (int, float)):
      return 1 if float(value) != 0 else 0
   text = str(value).strip().lower()
   if not text:
      return None
   if text in {"true", "t", "yes", "y", "1"}:
      return 1
   if text in {"false", "f", "no", "n", "0"}:
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


def fetch_season_game_numbers_from_stats(season: str, game_type: str, timeout_seconds: int) -> list[int]:
   # Fetch one season's game manifest in a single call to avoid probing non-existent game IDs.
   season_id = _season_id_yyyyyyyy(season)
   payload = _fetch_json(
      build_stats_games_url(),
      timeout_seconds,
      params={"cayenneExp": f"season={season_id} and gameType={int(game_type)}", "limit": -1},
   )
   if not isinstance(payload, dict):
      return []

   rows = payload.get("data", [])
   if not isinstance(rows, list):
      return []

   # Prefer final/live games so current-season runs skip future schedule placeholders.
   acceptable_states = {6, 7}
   game_numbers: set[int] = set()
   for row in rows:
      if not isinstance(row, dict):
         continue

      state = row.get("gameStateId")
      if state is not None:
         try:
            if int(state) not in acceptable_states:
               continue
         except (TypeError, ValueError):
            continue

      number = row.get("gameNumber")
      if number is None:
         game_id = row.get("id")
         try:
            number = int(game_id) % 10000 if game_id is not None else None
         except (TypeError, ValueError):
            number = None

      try:
         if number is not None:
            game_numbers.add(int(number))
      except (TypeError, ValueError):
         continue

   return sorted(game_numbers)


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


def parse_web_shot_events(game_json: dict, season: str, game_id: int) -> list[dict]:
   # Parse shot and goal events from the newer gamecenter play-by-play payload.
   home_team = game_json.get("homeTeam", {})
   away_team = game_json.get("awayTeam", {})
   home_code = home_team.get("abbrev") or game_json.get("gameData", {}).get("teams", {}).get("home", {}).get("triCode")
   away_code = away_team.get("abbrev") or game_json.get("gameData", {}).get("teams", {}).get("away", {}).get("triCode")
   team_id_to_code = {}
   if home_team.get("id") and home_code:
      team_id_to_code[int(home_team.get("id"))] = home_code
   if away_team.get("id") and away_code:
      team_id_to_code[int(away_team.get("id"))] = away_code

   plays = game_json.get("plays", [])
   rows = []
   for play in plays:
      event_key = str(play.get("typeDescKey", "")).lower()
      if event_key not in {"goal", "shot-on-goal", "shot"}:
         continue

      details = play.get("details", {})
      about = play.get("about", {})
      period_descriptor = play.get("periodDescriptor", {})
      x_coord = details.get("xCoord")
      y_coord = details.get("yCoord")
      if x_coord is None or y_coord is None:
         continue

      owner_team_id = details.get("eventOwnerTeamId")
      try:
         owner_team_id = int(owner_team_id) if owner_team_id is not None else None
      except (TypeError, ValueError):
         owner_team_id = None

      team_code = (
         details.get("eventOwnerTeamTricode")
         or details.get("eventOwnerTeamAbbrev")
         or (team_id_to_code.get(owner_team_id) if owner_team_id is not None else None)
         or play.get("team", {}).get("triCode")
      )
      shot_type = _normalize_text(_first_non_none(details.get("shotType"), details.get("secondaryType"))) or "Unknown"
      shooter = _normalize_text(_first_non_none(details.get("shootingPlayerName"), details.get("scoringPlayerName"))) or "Unknown"
      shooter_id = _coerce_int(_first_non_none(details.get("shootingPlayerId"), details.get("scoringPlayerId"), details.get("playerId")))
      goalie = _normalize_text(_first_non_none(details.get("goalieInNetName"), details.get("goalieName")))
      goalie_id = _coerce_int(_first_non_none(details.get("goalieInNetId"), details.get("goalieId")))
      period = _coerce_int(_first_non_none(period_descriptor.get("number"), about.get("period")))
      period_time = _normalize_text(_first_non_none(about.get("periodTime"), play.get("periodTime")))
      period_time_remaining = _normalize_text(_first_non_none(about.get("periodTimeRemaining"), play.get("periodTimeRemaining")))
      shot_distance = _coerce_float(_first_non_none(details.get("shotDistance"), details.get("distanceFromNet"), details.get("distance")))
      shot_angle = _coerce_float(_first_non_none(details.get("shotAngle"), details.get("angleFromNet"), details.get("angle")))
      is_empty_net = _coerce_bool_int(_first_non_none(details.get("emptyNet"), details.get("isEmptyNet"), details.get("empty_net")))
      strength_state = _normalize_text(_first_non_none(details.get("strength"), details.get("situationCode"), play.get("situationCode")))
      score_differential = _coerce_int(_first_non_none(details.get("scoreDifferential"), details.get("goalDifferential"), play.get("scoreDifferential")))
      zone = _normalize_text(_first_non_none(details.get("zone"), details.get("zoneCode"), play.get("zone"), play.get("zoneCode")))
      event_id = _coerce_int(_first_non_none(play.get("eventId"), details.get("eventId")))

      rows.append(
         {
            "Shot": "Goal" if event_key == "goal" else "ngshot",
            "X": float(x_coord),
            "Y": float(y_coord),
            "Shot_Type": shot_type,
            "Shooter": shooter,
            "Shooter_ID": shooter_id,
            "Team": team_code,
            "Home_Away": _home_away_value(team_code, home_code, away_code),
            "Period": period,
            "Period_Time": period_time,
            "Period_Time_Remaining": period_time_remaining,
            "Year": season,
            "GameID": game_id,
            "API_Source": "web",
            "Goalie": goalie,
            "Goalie_ID": goalie_id,
            "Shot_Distance": shot_distance,
            "Shot_Angle": shot_angle,
            "Is_Empty_Net": is_empty_net,
            "Strength_State": strength_state,
            "Score_Differential": score_differential,
            "Zone": zone,
            "Event_ID": event_id,
         }
      )

   return rows


def _extract_record_list(payload: dict | list) -> list[dict]:
   if isinstance(payload, list):
      return [entry for entry in payload if isinstance(entry, dict)]
   for key in ("data", "records", "items", "response"):
      value = payload.get(key)
      if isinstance(value, list):
         return [entry for entry in value if isinstance(entry, dict)]
   return []


def parse_stats_shift_events(payload: dict | list, season: str, game_id: int) -> list[dict]:
   # Stats REST may include event/coordinate records for some games; keep parsing defensive.
   rows = []
   for record in _extract_record_list(payload):
      event_type = str(
         record.get("eventType")
         or record.get("typeCode")
         or record.get("eventDescription")
         or ""
      ).lower()
      if "shot" not in event_type and "goal" not in event_type:
         continue

      x_coord = record.get("xCoord") if record.get("xCoord") is not None else record.get("xcoord")
      y_coord = record.get("yCoord") if record.get("yCoord") is not None else record.get("ycoord")
      if x_coord is None or y_coord is None:
         continue

      rows.append(
         {
            "Shot": "Goal" if "goal" in event_type else "ngshot",
            "X": float(x_coord),
            "Y": float(y_coord),
            "Shot_Type": record.get("shotType") or "Unknown",
            "Shooter": record.get("playerName") or record.get("lastName") or "Unknown",
            "Team": record.get("teamAbbrev") or record.get("teamTriCode"),
            "Home_Away": None,
            "Period": record.get("period"),
            "Year": season,
            "GameID": game_id,
            "API_Source": "stats",
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
            shooter_id INTEGER,
            team TEXT,
            home_away INTEGER,
            period INTEGER,
            period_time TEXT,
            period_time_remaining TEXT,
            season TEXT NOT NULL,
            game_id INTEGER NOT NULL,
            api_source TEXT NOT NULL DEFAULT 'web',
            goalie TEXT,
            goalie_id INTEGER,
            shot_distance REAL,
            shot_angle REAL,
            is_empty_net INTEGER,
            strength_state TEXT,
            score_differential INTEGER,
            zone TEXT,
            event_id INTEGER
         )
         """
      )
      # Backward-compatible migration for existing databases.
      cursor.execute("PRAGMA table_info(shots)")
      columns = {row[1] for row in cursor.fetchall()}
      if "api_source" not in columns:
         cursor.execute("ALTER TABLE shots ADD COLUMN api_source TEXT NOT NULL DEFAULT 'web'")
      if "shooter_id" not in columns:
         cursor.execute("ALTER TABLE shots ADD COLUMN shooter_id INTEGER")
      if "period_time" not in columns:
         cursor.execute("ALTER TABLE shots ADD COLUMN period_time TEXT")
      if "period_time_remaining" not in columns:
         cursor.execute("ALTER TABLE shots ADD COLUMN period_time_remaining TEXT")
      if "goalie" not in columns:
         cursor.execute("ALTER TABLE shots ADD COLUMN goalie TEXT")
      if "goalie_id" not in columns:
         cursor.execute("ALTER TABLE shots ADD COLUMN goalie_id INTEGER")
      if "shot_distance" not in columns:
         cursor.execute("ALTER TABLE shots ADD COLUMN shot_distance REAL")
      if "shot_angle" not in columns:
         cursor.execute("ALTER TABLE shots ADD COLUMN shot_angle REAL")
      if "is_empty_net" not in columns:
         cursor.execute("ALTER TABLE shots ADD COLUMN is_empty_net INTEGER")
      if "strength_state" not in columns:
         cursor.execute("ALTER TABLE shots ADD COLUMN strength_state TEXT")
      if "score_differential" not in columns:
         cursor.execute("ALTER TABLE shots ADD COLUMN score_differential INTEGER")
      if "zone" not in columns:
         cursor.execute("ALTER TABLE shots ADD COLUMN zone TEXT")
      if "event_id" not in columns:
         cursor.execute("ALTER TABLE shots ADD COLUMN event_id INTEGER")

      cursor.execute("CREATE INDEX IF NOT EXISTS idx_shots_season ON shots(season)")
      cursor.execute("CREATE INDEX IF NOT EXISTS idx_shots_team ON shots(team)")
      cursor.execute("CREATE INDEX IF NOT EXISTS idx_shots_shooter ON shots(shooter)")
      cursor.execute("CREATE INDEX IF NOT EXISTS idx_shots_strength_state ON shots(strength_state)")
      cursor.execute("CREATE INDEX IF NOT EXISTS idx_shots_shot_result ON shots(shot_result)")
      cursor.execute("CREATE INDEX IF NOT EXISTS idx_shots_period ON shots(period)")
      cursor.execute("CREATE INDEX IF NOT EXISTS idx_shots_home_away ON shots(home_away)")

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

      cursor.execute(
         """
         CREATE TABLE IF NOT EXISTS edge_metric_values (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            season TEXT NOT NULL,
            game_type TEXT NOT NULL,
            source TEXT NOT NULL,
            snapshot_type TEXT NOT NULL,
            entity_type TEXT NOT NULL,
            entity_id TEXT NOT NULL,
            endpoint TEXT NOT NULL,
            metric_path TEXT NOT NULL,
            metric_value_text TEXT,
            metric_value_numeric REAL,
            fetched_at TEXT NOT NULL,
            UNIQUE(season, game_type, source, snapshot_type, entity_type, entity_id, endpoint, metric_path)
         )
         """
      )

      cursor.execute(
         """
         CREATE TABLE IF NOT EXISTS season_scrape_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            season TEXT NOT NULL,
            game_type TEXT NOT NULL,
            start_game INTEGER NOT NULL,
            end_game INTEGER NOT NULL,
            completed_through_game INTEGER NOT NULL DEFAULT 0,
            games_processed INTEGER NOT NULL DEFAULT 0,
            rows_inserted INTEGER NOT NULL DEFAULT 0,
            status TEXT NOT NULL DEFAULT 'running',
            updated_at TEXT NOT NULL,
            UNIQUE(season, game_type, start_game, end_game)
         )
         """
      )
      connection.commit()


def _flatten_json_scalars(value: object, prefix: str = "") -> list[tuple[str, object]]:
   flattened: list[tuple[str, object]] = []

   if isinstance(value, dict):
      for key, nested_value in value.items():
         path = f"{prefix}.{key}" if prefix else str(key)
         flattened.extend(_flatten_json_scalars(nested_value, path))
      return flattened

   if isinstance(value, list):
      for index, nested_value in enumerate(value):
         path = f"{prefix}[{index}]" if prefix else f"[{index}]"
         flattened.extend(_flatten_json_scalars(nested_value, path))
      return flattened

   if prefix:
      flattened.append((prefix, value))
   return flattened


def _metric_value_parts(value: object) -> tuple[str | None, float | None]:
   if value is None:
      return None, None
   if isinstance(value, bool):
      return str(int(value)), float(int(value))
   if isinstance(value, (int, float)):
      return str(value), float(value)

   text = str(value)
   try:
      numeric = float(text)
   except (TypeError, ValueError):
      numeric = None
   return text, numeric


def _store_edge_metric_values(
   cursor: sqlite3.Cursor,
   season: str,
   game_type: str,
   source: str,
   snapshot_type: str,
   entity_type: str,
   entity_id: str,
   endpoint: str,
   payload: dict | list,
) -> int:
   fetched_at = datetime.now(UTC).isoformat(timespec="seconds")
   metric_rows = []
   for metric_path, metric_value in _flatten_json_scalars(payload):
      value_text, value_numeric = _metric_value_parts(metric_value)
      metric_rows.append(
         (
            season,
            game_type,
            source,
            snapshot_type,
            entity_type,
            str(entity_id),
            endpoint,
            metric_path,
            value_text,
            value_numeric,
            fetched_at,
         )
      )

   if not metric_rows:
      return 0

   cursor.executemany(
      """
      INSERT OR REPLACE INTO edge_metric_values
      (season, game_type, source, snapshot_type, entity_type, entity_id, endpoint, metric_path,
       metric_value_text, metric_value_numeric, fetched_at)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
      """,
      metric_rows,
   )
   return len(metric_rows)


def _run_state_key(season: str, game_type: str, start_game: int, end_game: int) -> tuple[str, str, int, int]:
   return season, str(game_type), start_game, end_game


def _load_completed_run(db_path: str, season: str, game_type: str, start_game: int, end_game: int) -> dict | None:
   try:
      with sqlite3.connect(db_path) as connection:
         connection.row_factory = sqlite3.Row
         row = connection.execute(
            """
            SELECT *
            FROM season_scrape_runs
            WHERE season = ? AND game_type = ? AND start_game = ? AND end_game = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            _run_state_key(season, game_type, start_game, end_game),
         ).fetchone()
   except sqlite3.OperationalError:
      return None
   return dict(row) if row is not None else None


def _upsert_run_state(
   cursor: sqlite3.Cursor,
   season: str,
   game_type: str,
   start_game: int,
   end_game: int,
   completed_through_game: int,
   games_processed: int,
   rows_inserted: int,
   status: str,
) -> None:
   updated_at = datetime.now(UTC).isoformat(timespec="seconds")
   cursor.execute(
      """
      INSERT INTO season_scrape_runs
      (season, game_type, start_game, end_game, completed_through_game, games_processed, rows_inserted, status, updated_at)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
      ON CONFLICT(season, game_type, start_game, end_game)
      DO UPDATE SET
         completed_through_game = excluded.completed_through_game,
         games_processed = excluded.games_processed,
         rows_inserted = excluded.rows_inserted,
         status = excluded.status,
         updated_at = excluded.updated_at
      """,
      (season, game_type, start_game, end_game, completed_through_game, games_processed, rows_inserted, status, updated_at),
   )


def _season_run_is_complete(db_path: str, season: str, game_type: str, start_game: int, end_game: int) -> bool:
   run_state = _load_completed_run(db_path, season, game_type, start_game, end_game)
   if not run_state:
      return False
   return run_state.get("status") == "complete" and int(run_state.get("completed_through_game") or 0) >= end_game


def _resume_start_game(db_path: str, season: str, game_type: str, start_game: int, end_game: int) -> int:
   run_state = _load_completed_run(db_path, season, game_type, start_game, end_game)
   if not run_state:
      return start_game

   completed_through = int(run_state.get("completed_through_game") or 0)
   if run_state.get("status") == "complete" and completed_through >= end_game:
      return end_game + 1

   return max(start_game, completed_through + 1)


def _fetch_and_parse_game_rows(season: str, game_type: str, game_id: int, timeout_seconds: int, api_sources: list[str]) -> list[dict]:
   rows: list[dict] = []
   if "web" in api_sources:
      web_url = build_web_play_by_play_url(season, game_type, game_id)
      web_json = _fetch_json_allow_404(web_url, timeout_seconds)
      if web_json is not None:
         rows.extend(parse_web_shot_events(web_json, season, game_id))

   if "stats" in api_sources and not rows:
      full_game_id = int(f"{season}{game_type}{game_id:04d}")
      stats_payload = _fetch_json(
         build_stats_shiftcharts_url(),
         timeout_seconds,
         params={"cayenneExp": f"gameId={full_game_id}", "limit": -1},
      )
      if stats_payload is not None:
         rows.extend(parse_stats_shift_events(stats_payload, season, game_id))
   return rows


def _iter_game_rows_parallel(
   season: str,
   game_type: str,
   game_numbers: list[int],
   timeout_seconds: int,
   active_sources: list[str],
   max_workers: int,
) -> Iterator[tuple[int, list[dict]]]:
   if max_workers <= 1 or len(game_numbers) <= 1:
      for game_id in game_numbers:
         yield game_id, _fetch_and_parse_game_rows(season, game_type, game_id, timeout_seconds, active_sources)
      return

   total_games = len(game_numbers)
   logging.info(
      "Season %s: fetching %s games with %s workers",
      season,
      total_games,
      max_workers,
   )
   with ThreadPoolExecutor(max_workers=max_workers) as executor:
      futures = {
         executor.submit(_fetch_and_parse_game_rows, season, game_type, game_id, timeout_seconds, active_sources): game_id
         for game_id in game_numbers
      }
      completed_rows: dict[int, list[dict]] = {}
      next_index = 0

      for future in as_completed(futures):
         game_id = futures[future]
         completed_rows[game_id] = future.result()
         while next_index < total_games:
            next_game_id = game_numbers[next_index]
            if next_game_id not in completed_rows:
               break
            yield next_game_id, completed_rows.pop(next_game_id)
            next_index += 1


def _persist_rows_with_cursor(cursor: sqlite3.Cursor, rows: list[dict]) -> int:
   if not rows:
      return 0

   params = [
      (
         _event_hash(row),
         row["Shot"],
         row["X"],
         row["Y"],
         row["Shot_Type"],
         row["Shooter"],
         row.get("Shooter_ID"),
         row["Team"],
         row["Home_Away"],
         row["Period"],
         row.get("Period_Time"),
         row.get("Period_Time_Remaining"),
         row["Year"],
         row["GameID"],
         row.get("API_Source", "web"),
         row.get("Goalie"),
         row.get("Goalie_ID"),
         row.get("Shot_Distance"),
         row.get("Shot_Angle"),
         row.get("Is_Empty_Net"),
         row.get("Strength_State"),
         row.get("Score_Differential"),
         row.get("Zone"),
         row.get("Event_ID"),
      )
      for row in rows
   ]

   before = cursor.connection.total_changes
   cursor.executemany(
      """
      INSERT OR IGNORE INTO shots (
         event_hash, shot_result, x, y, shot_type, shooter, shooter_id, team,
         home_away, period, period_time, period_time_remaining, season, game_id, api_source,
         goalie, goalie_id, shot_distance, shot_angle, is_empty_net, strength_state,
         score_differential, zone, event_id
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
      """,
      params,
   )
   return cursor.connection.total_changes - before


def persist_rows(db_path: str, rows: list[dict]) -> int:
   # INSERT OR IGNORE prevents duplicate rows when scraping the same games again.
   if not rows:
      return 0

   with sqlite3.connect(db_path) as connection:
      cursor = connection.cursor()
      inserted = _persist_rows_with_cursor(cursor, rows)
      connection.commit()
   return inserted


def export_to_csv(db_path: str, csv_path: str) -> int:
   # Optional Tableau-compatible export with stable column order.
   with sqlite3.connect(db_path) as connection:
      cursor = connection.cursor()
      cursor.execute(
         """
         SELECT shot_result, x, y, shot_type, shooter, shooter_id, team, home_away, period, period_time,
                period_time_remaining, season, game_id, api_source, goalie, goalie_id, shot_distance,
                shot_angle, is_empty_net, strength_state, score_differential, zone, event_id
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
               "Shooter_ID": row[5],
               "Team": row[6],
               "Home_Away": row[7],
               "Period": row[8],
               "Period_Time": row[9],
               "Period_Time_Remaining": row[10],
               "Year": row[11],
               "GameID": row[12],
               "API_Source": row[13],
               "Goalie": row[14],
               "Goalie_ID": row[15],
               "Shot_Distance": row[16],
               "Shot_Angle": row[17],
               "Is_Empty_Net": row[18],
               "Strength_State": row[19],
               "Score_Differential": row[20],
               "Zone": row[21],
               "Event_ID": row[22],
            }
         )
   return len(rows)


def capture_edge_summary_snapshots(db_path: str, season: str, game_type: str, timeout_seconds: int) -> int:
   # Store raw NHL Edge summary snapshots so downstream analysis can evolve without re-scraping.
   # These landing pages summarize team and player context such as top shot speed,
   # skating speed, high-danger SOG, distance skated, and zone-time totals.
   season_id = _season_id_yyyyyyyy(season)
   edge_game_type = str(int(game_type))
   required_endpoints = [
      f"{WEB_API_BASE_URL}/edge/team-landing/{season_id}/{edge_game_type}",
      f"{WEB_API_BASE_URL}/edge/skater-landing/{season_id}/{edge_game_type}",
      f"{WEB_API_BASE_URL}/edge/goalie-landing/{season_id}/{edge_game_type}",
   ]
   optional_endpoints = [
      f"{WEB_API_BASE_URL}/edge/by-the-numbers",
   ]

   inserted = 0
   fetched_at = datetime.now(UTC).isoformat(timespec="seconds")
   with sqlite3.connect(db_path) as connection:
      cursor = connection.cursor()
      for endpoint in required_endpoints:
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
         _store_edge_metric_values(
            cursor,
            season,
            edge_game_type,
            "web",
            "summary",
            "season",
            season_id,
            endpoint,
            payload,
         )
         inserted += 1

      for endpoint in optional_endpoints:
         payload = _fetch_json_allow_404(endpoint, timeout_seconds)
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
         _store_edge_metric_values(
            cursor,
            season,
            edge_game_type,
            "web",
            "summary",
            "season",
            season_id,
            endpoint,
            payload,
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
   cursor: sqlite3.Cursor | None = None,
) -> None:
   fetched_at = datetime.now(UTC).isoformat(timespec="seconds")
   if cursor is not None:
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
      return

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
      logging.warning(
         "No team codes were collected for season %s, skipping Edge deep crawl to avoid noisy 404 fan-out.",
         season,
      )
      return 0

   team_catalog = fetch_team_catalog(timeout_seconds)
   if not team_catalog:
      logging.warning("Could not load team catalog from stats REST; skipping Edge deep crawl for season %s.", season)
      return 0

   total_snapshots = 0
   edge_game_type = str(int(game_type))

   with sqlite3.connect(db_path) as connection:
      cursor = connection.cursor()

      for team_code in sorted(team_codes):
         team_id = team_catalog.get(team_code.upper())
         if team_id is None:
            logging.warning("Could not map team code %s to a team id; skipping team detail.", team_code)
            continue

         team_detail_endpoint = build_web_edge_team_detail_url(team_id, season, game_type)
         team_detail_payload = _fetch_json_allow_404(team_detail_endpoint, timeout_seconds)
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
               cursor,
            )
            _store_edge_metric_values(
               cursor,
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
         roster_payload = _fetch_json_allow_404(roster_endpoint, timeout_seconds)
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
            cursor,
         )
         _store_edge_metric_values(
            cursor,
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

            player_payload = _fetch_json_allow_404(player_endpoint, timeout_seconds)
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
               cursor,
            )
            _store_edge_metric_values(
               cursor,
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

      connection.commit()

   return total_snapshots


def run_season_scrape(config: ScrapeConfig, active_sources: list[str], capture_edge: bool, capture_edge_deep: bool) -> tuple[int, int, int, int]:
   # Orchestrates fetch -> parse -> persist for a game range.
   initialize_database(config.db_path)
   configure_request_rate_limit(config.request_delay_seconds)

   games_processed = 0
   rows_inserted = 0
   edge_payloads_inserted = 0
   edge_detail_payloads_inserted = 0
   team_codes_seen: set[str] = set()
   if _season_run_is_complete(config.db_path, config.season, config.game_type, config.start_game, config.end_game):
      logging.info(
         "Season %s game_type %s range %s-%s already completed in the database; skipping scrape.",
         config.season,
         config.game_type,
         config.start_game,
         config.end_game,
      )
      return 0, 0, 0, 0

   resume_start_game = _resume_start_game(config.db_path, config.season, config.game_type, config.start_game, config.end_game)
   if resume_start_game > config.end_game:
      logging.info(
         "Season %s game_type %s range %s-%s already present; nothing to do.",
         config.season,
         config.game_type,
         config.start_game,
         config.end_game,
      )
      return 0, 0, 0, 0

   game_numbers = list(range(resume_start_game, config.end_game + 1))
   if "stats" in active_sources:
      manifest_numbers = fetch_season_game_numbers_from_stats(config.season, config.game_type, config.timeout_seconds)
      if manifest_numbers:
         game_numbers = [n for n in manifest_numbers if resume_start_game <= n <= config.end_game]
         logging.info(
            "Season %s: loaded %s game numbers from stats manifest (range-filtered).",
            config.season,
            len(game_numbers),
         )
      else:
         logging.info(
            "Season %s: stats game manifest unavailable, falling back to numeric game sweep.",
            config.season,
         )

   total_games_to_check = len(game_numbers)

   empty_game_streak = 0

   edge_supported = False
   if "web" in active_sources and (capture_edge or capture_edge_deep):
      edge_supported = edge_is_supported_for_season(config.season, config.game_type, config.timeout_seconds)
      if not edge_supported:
         logging.info("Season %s: skipping EDGE capture because season is not listed in seasonsWithEdgeStats.", config.season)

   with sqlite3.connect(config.db_path) as connection:
      cursor = connection.cursor()
      completed_through_game = resume_start_game - 1
      completed_normally = True
      last_heartbeat = time.monotonic()

      def log_heartbeat(index: int, game_id: int) -> None:
         nonlocal last_heartbeat
         now = time.monotonic()
         if now - last_heartbeat >= HEARTBEAT_LOG_INTERVAL_SECONDS:
            logging.info(
               "Season %s heartbeat: checked %s/%s games, games_with_rows=%s, rows_inserted=%s, last_game=%s",
               config.season,
               index,
               total_games_to_check,
               games_processed,
               rows_inserted,
               game_id,
            )
            last_heartbeat = now

      try:
         for index, (game_id, rows) in enumerate(
            _iter_game_rows_parallel(
               config.season,
               config.game_type,
               game_numbers,
               config.timeout_seconds,
               active_sources,
               config.max_workers,
            ),
            start=1,
         ):
            if not rows:
               empty_game_streak += 1
               if index % 50 == 0 or index == total_games_to_check:
                  logging.info(
                     "Season %s progress: checked %s/%s games, games_with_rows=%s, rows_inserted=%s",
                     config.season,
                     index,
                     total_games_to_check,
                     games_processed,
                     rows_inserted,
                  )

               if game_id >= SEASON_END_PROBE_START_GAME and empty_game_streak >= SEASON_END_EMPTY_STREAK_GAMES:
                  logging.info(
                     "Season %s: stopping early at game %s after %s consecutive empty games.",
                     config.season,
                     game_id,
                     empty_game_streak,
                  )
                  completed_normally = False
                  _upsert_run_state(
                     cursor,
                     config.season,
                     config.game_type,
                     config.start_game,
                     config.end_game,
                     completed_through_game,
                     games_processed,
                     rows_inserted,
                     "partial",
                  )
                  connection.commit()
                  break
               if index % DB_COMMIT_INTERVAL_GAMES == 0:
                  _upsert_run_state(
                     cursor,
                     config.season,
                     config.game_type,
                     config.start_game,
                     config.end_game,
                     game_id,
                     games_processed,
                     rows_inserted,
                     "running",
                  )
                  connection.commit()
               log_heartbeat(index, game_id)
               continue

            empty_game_streak = 0
            completed_through_game = game_id
            inserted = _persist_rows_with_cursor(cursor, rows)
            games_processed += 1
            rows_inserted += inserted
            if capture_edge_deep:
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
            log_heartbeat(index, game_id)

            if index % DB_COMMIT_INTERVAL_GAMES == 0:
               _upsert_run_state(
                  cursor,
                  config.season,
                  config.game_type,
                  config.start_game,
                  config.end_game,
                  game_id,
                  games_processed,
                  rows_inserted,
                  "running",
               )
               connection.commit()

            if index % 50 == 0 or index == total_games_to_check:
               logging.info(
                  "Season %s progress: checked %s/%s games, games_with_rows=%s, rows_inserted=%s",
                  config.season,
                  index,
                  total_games_to_check,
                  games_processed,
                  rows_inserted,
               )

         final_status = "complete" if completed_normally else "partial"
         if completed_normally and total_games_to_check > 0 and rows_inserted == 0:
            logging.warning(
               "Season %s produced zero inserted rows after checking %s games; marking run as partial for safe retry.",
               config.season,
               total_games_to_check,
            )
            final_status = "partial"

         _upsert_run_state(
            cursor,
            config.season,
            config.game_type,
            config.start_game,
            config.end_game,
            completed_through_game,
            games_processed,
            rows_inserted,
            final_status,
         )
         connection.commit()
      except Exception:
         _upsert_run_state(
            cursor,
            config.season,
            config.game_type,
            config.start_game,
            config.end_game,
            completed_through_game,
            games_processed,
            rows_inserted,
            "failed",
         )
         connection.commit()
         raise

   if capture_edge and "web" in active_sources and edge_supported:
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

   if capture_edge_deep and "web" in active_sources and edge_supported:
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
   parser.add_argument(
      "--start-season",
      type=int,
      default=COORDINATE_DATA_START_SEASON,
      help=f"Start season for multi-season run. Defaults to {COORDINATE_DATA_START_SEASON} (first coordinate-data season).",
   )
   parser.add_argument("--end-season", type=int, default=None, help="Optional end season for multi-season run. Defaults to current NHL season.")
   parser.add_argument("--api-source", default="both", choices=["web", "stats", "both"], help="Choose Web API, Stats REST API, or both.")
   parser.add_argument("--capture-edge", action="store_true", help="Capture NHL Edge summary payloads when web API is enabled.")
   parser.add_argument("--capture-edge-deep", action="store_true", help="Capture NHL Edge team and player detail payloads after each season.")
   parser.add_argument("--game-type", default=REGULAR_SEASON, choices=[PRESEASON, REGULAR_SEASON, PLAYOFFS, ALLSTAR])
   parser.add_argument("--start-game", type=int, default=1)
   parser.add_argument("--end-game", type=int, default=1271)
   parser.add_argument("--timeout", type=int, default=10, help="HTTP timeout in seconds.")
   parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS, help="Maximum number of concurrent game fetch workers.")
   parser.add_argument("--request-delay", type=float, default=DEFAULT_REQUEST_DELAY_SECONDS, help="Minimum delay in seconds between API requests.")
   parser.add_argument("--db-path", default="hockey_shots.db", help="SQLite database path.")
   parser.add_argument("--export-csv", default=None, help="Optional Tableau-compatible CSV export path.")
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
      resolved_start = args.start_season
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
         max_workers=args.max_workers,
         request_delay_seconds=args.request_delay,
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

