#!/usr/bin/env python3
"""
NHL Game Shifts Scraper.

Fetches game shift data from the NHL Stats REST API and stores it in the game_shifts table.

API Endpoint: https://api.nhle.com/stats/rest/v1/en/shiftcharts
"""

import hashlib
import logging
import sqlite3
from datetime import UTC, datetime

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Reuse constants from shots_scraper
from shots_scraper import (
    build_stats_shiftcharts_url,
    build_web_play_by_play_url,
    _fetch_json,
    _fetch_json_allow_404,
    _extract_record_list,
    _coerce_int,
    _coerce_float,
    _normalize_text,
    _season_id_yyyyyyyy,
    _first_non_none,
    REGULAR_SEASON,
    DEFAULT_REQUEST_DELAY_SECONDS,
    HTTP_RETRY_TOTAL,
    HTTP_RETRY_BACKOFF_FACTOR,
    HTTP_RATE_LIMIT_STATUSES,
)

# HTTP session with retry adapter
_GAME_SHIFTS_HTTP_SESSION = requests.Session()


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
    return HTTPAdapter(max_retries=retry, pool_connections=4, pool_maxsize=4)


_GAME_SHIFTS_HTTP_SESSION.mount("http://", _build_retry_adapter())
_GAME_SHIFTS_HTTP_SESSION.mount("https://", _build_retry_adapter())


def _get_http_session() -> requests.Session:
    return _GAME_SHIFTS_HTTP_SESSION


def _parse_shift_record(record: dict, season: str, game_id: int) -> dict | None:
    """Parse a single shift record into a normalized row."""
    player_id = record.get("playerId")
    if player_id is None:
        return None

    team_id = record.get("teamId")
    if team_id is None:
        return None

    # Parse shift times
    shift_start = record.get("shiftStart") or record.get("startTime")
    shift_end = record.get("shiftEnd") or record.get("endTime")

    # Calculate duration if not provided
    duration_seconds = record.get("duration")
    if duration_seconds is None and shift_start and shift_end:
        try:
            start_sec = _parse_period_time_to_seconds(shift_start)
            end_sec = _parse_period_time_to_seconds(shift_end)
            if start_sec is not None and end_sec is not None:
                duration_seconds = end_sec - start_sec
        except (ValueError, TypeError):
            pass

    player_name = _extract_player_name(record) or "Unknown"

    return {
        "player_id": int(player_id),
        "player_name": player_name,
        "team_id": int(team_id),
        "team_abbrev": str(record.get("teamAbbrev", record.get("teamCode", "Unknown"))),
        "season": season,
        "game_id": game_id,
        "period": _coerce_int(record.get("period")),
        "shift_start": _normalize_text(shift_start),
        "shift_end": _normalize_text(shift_end),
        "duration_seconds": _coerce_float(duration_seconds),
        "shift_type": _normalize_text(record.get("shiftType", record.get("type", "normal"))),
        "is_shifting": _coerce_int(record.get("isShifting", record.get("shifting", 0))),
        "jersey_number": _coerce_int(record.get("jerseyNumber")),
        "position": _normalize_text(record.get("position", record.get("positionCode"))),
        "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def _name_piece_to_text(value: object) -> str | None:
    """Normalize first/last name payload shapes into text."""
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, dict):
        # Web API often uses locale maps like {"default": "Name"}
        for key in ("default", "full", "name", "value"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
    return None


def _extract_player_name(record: dict) -> str | None:
    """Extract player name across known NHL payload field variants."""
    candidates = [
        record.get("playerName"),
        record.get("fullName"),
        record.get("skaterFullName"),
        record.get("goalieFullName"),
        record.get("name"),
    ]

    for candidate in candidates:
        text = _name_piece_to_text(candidate)
        if text:
            return text

    # Some payloads split first/last names.
    first = _name_piece_to_text(record.get("firstName"))
    last = _name_piece_to_text(record.get("lastName"))
    full = " ".join(part for part in (first, last) if part).strip()
    return full or None


def _parse_period_time_to_seconds(period_time: str) -> int | None:
    """Parse period time string (e.g., '10:30') to seconds."""
    if not period_time or ":" not in period_time:
        return None
    try:
        parts = period_time.split(":")
        if len(parts) != 2:
            return None
        minutes = int(parts[0])
        seconds = int(parts[1])
        return minutes * 60 + seconds
    except (ValueError, TypeError):
        return None


def _shift_hash(shift: dict) -> str:
    """Generate a deterministic hash for a shift record."""
    signature = f"{shift['season']}|{shift['game_id']}|{shift['player_id']}|{shift['period']}|{shift['shift_start']}|{shift['shift_end']}"
    return hashlib.sha256(signature.encode("utf-8")).hexdigest()


def initialize_game_shifts_database(db_path: str) -> None:
    """Create game_shifts table if it doesn't exist."""
    with sqlite3.connect(db_path) as connection:
        cursor = connection.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS game_shifts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                shift_hash TEXT NOT NULL,
                player_id INTEGER NOT NULL,
                player_name TEXT,
                team_id INTEGER NOT NULL,
                team_abbrev TEXT,
                season TEXT NOT NULL,
                game_id INTEGER NOT NULL,
                period INTEGER,
                shift_start TEXT,
                shift_end TEXT,
                duration_seconds REAL,
                shift_type TEXT,
                is_shifting INTEGER,
                jersey_number INTEGER,
                position TEXT,
                fetched_at TEXT NOT NULL,
                UNIQUE(shift_hash)
            )
            """
        )
        cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_game_shifts_hash ON game_shifts(shift_hash)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_game_shifts_season ON game_shifts(season)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_game_shifts_game_id ON game_shifts(game_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_game_shifts_player_id ON game_shifts(player_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_game_shifts_team_id ON game_shifts(team_id)")
        connection.commit()


def _fetch_game_shifts_from_stats(
    season: str,
    game_type: str,
    game_id: int,
    timeout_seconds: int,
) -> list[dict]:
    """Fetch shift data from the Stats REST shiftcharts endpoint."""
    full_game_id = int(f"{season}{game_type}{game_id:04d}")
    url = build_stats_shiftcharts_url()
    params = {
        "cayenneExp": f"gameId={full_game_id}",
        "limit": -1,
    }

    payload = _fetch_json(url, timeout_seconds, params=params)
    if payload is None:
        return []

    records = _extract_record_list(payload)
    shifts = []
    for record in records:
        shift = _parse_shift_record(record, season, game_id)
        if shift is not None:
            shift["shift_hash"] = _shift_hash(shift)
            shifts.append(shift)

    return shifts


def _fetch_game_shifts_from_web(
    season: str,
    game_type: str,
    game_id: int,
    timeout_seconds: int,
) -> list[dict]:
    """Fetch shift data from the Web API gamecenter endpoint (fallback)."""
    url = build_web_play_by_play_url(season, game_type, game_id)
    payload = _fetch_json_allow_404(url, timeout_seconds)
    if payload is None:
        return []

    # Extract shifts from play-by-play if available
    plays = payload.get("plays", [])
    shifts = []

    for play in plays:
        if not isinstance(play, dict):
            continue

        # Look for shift-related events
        event_type = str(play.get("typeDescKey", "")).lower()
        if "shift" in event_type:
            # Parse shift info from play
            details = play.get("details", {})
            about = play.get("about", {})

            shift = {
                "season": season,
                "game_id": game_id,
                "period": _coerce_int(about.get("period")),
                "shift_start": _normalize_text(about.get("periodTimeRemaining")),
                "shift_end": None,
                "duration_seconds": None,
                "shift_type": event_type,
                "player_id": _coerce_int(details.get("playerId")),
                "player_name": _normalize_text(details.get("playerName")),
                "team_id": None,
                "team_abbrev": _normalize_text(details.get("teamAbbrev")),
                "is_shifting": 1,
                "jersey_number": _coerce_int(details.get("jerseyNumber")),
                "position": _normalize_text(details.get("positionCode")),
            }
            if shift["player_id"] is not None:
                shift["shift_hash"] = _shift_hash(shift)
                shifts.append(shift)

    return shifts


def persist_game_shifts(db_path: str, shifts: list[dict]) -> int:
    """Persist shift records to the database."""
    if not shifts:
        return 0

    params = [
        (
            shift["shift_hash"],
            shift["player_id"],
            shift.get("player_name"),
            shift["team_id"],
            shift.get("team_abbrev"),
            shift["season"],
            shift["game_id"],
            shift.get("period"),
            shift.get("shift_start"),
            shift.get("shift_end"),
            shift.get("duration_seconds"),
            shift.get("shift_type"),
            shift.get("is_shifting"),
            shift.get("jersey_number"),
            shift.get("position"),
            shift["fetched_at"],
        )
        for shift in shifts
    ]

    with sqlite3.connect(db_path) as connection:
        cursor = connection.cursor()
        before = cursor.connection.total_changes
        cursor.executemany(
            """
            INSERT INTO game_shifts (
                shift_hash, player_id, player_name, team_id, team_abbrev,
                season, game_id, period, shift_start, shift_end,
                duration_seconds, shift_type, is_shifting, jersey_number, position, fetched_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(shift_hash) DO UPDATE SET
                player_name = CASE
                    WHEN excluded.player_name IS NOT NULL
                         AND TRIM(excluded.player_name) <> ''
                         AND excluded.player_name <> 'Unknown'
                    THEN excluded.player_name
                    ELSE game_shifts.player_name
                END,
                team_id = COALESCE(excluded.team_id, game_shifts.team_id),
                team_abbrev = COALESCE(excluded.team_abbrev, game_shifts.team_abbrev),
                period = COALESCE(excluded.period, game_shifts.period),
                shift_start = COALESCE(excluded.shift_start, game_shifts.shift_start),
                shift_end = COALESCE(excluded.shift_end, game_shifts.shift_end),
                duration_seconds = COALESCE(excluded.duration_seconds, game_shifts.duration_seconds),
                shift_type = COALESCE(excluded.shift_type, game_shifts.shift_type),
                is_shifting = COALESCE(excluded.is_shifting, game_shifts.is_shifting),
                jersey_number = COALESCE(excluded.jersey_number, game_shifts.jersey_number),
                position = COALESCE(excluded.position, game_shifts.position),
                fetched_at = excluded.fetched_at
            """,
            params,
        )
        connection.commit()
        return cursor.connection.total_changes - before


def backfill_game_shift_player_names(db_path: str) -> int:
        """Backfill Unknown/blank game_shifts.player_name from players.full_name via player_id."""
        with sqlite3.connect(db_path) as connection:
                cursor = connection.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='players'")
                if cursor.fetchone() is None:
                        return 0

                cursor.execute(
                        """
                        UPDATE game_shifts
                        SET player_name = (
                                SELECT full_name
                                FROM players
                                WHERE players.player_id = game_shifts.player_id
                        )
                        WHERE (player_name IS NULL OR TRIM(player_name) = '' OR player_name = 'Unknown')
                            AND EXISTS (
                                SELECT 1
                                FROM players
                                WHERE players.player_id = game_shifts.player_id
                            )
                        """
                )
                filled = cursor.rowcount
                connection.commit()
                return int(filled or 0)


def fetch_and_store_game_shifts(
    db_path: str,
    season: str,
    game_type: str = REGULAR_SEASON,
    timeout_seconds: int = 10,
    game_numbers: list[int] | None = None,
) -> int:
    """
    Fetch game shifts from the Stats REST API and store in game_shifts table.
    Returns the number of rows inserted.
    """
    initialize_game_shifts_database(db_path)

    # Get game numbers if not provided
    if game_numbers is None:
        from shots_scraper import fetch_season_game_numbers_from_stats
        game_numbers = fetch_season_game_numbers_from_stats(season, game_type, timeout_seconds)

    if not game_numbers:
        logging.warning("No game numbers found for season %s (game_type=%s)", season, game_type)
        return 0

    total_inserted = 0
    for game_id in game_numbers:
        shifts = _fetch_game_shifts_from_stats(season, game_type, game_id, timeout_seconds)

        # Fallback to web API if stats returns empty
        if not shifts:
            shifts = _fetch_game_shifts_from_web(season, game_type, game_id, timeout_seconds)

        if shifts:
            inserted = persist_game_shifts(db_path, shifts)
            total_inserted += inserted
            logging.info("Game %s: fetched %d shifts, inserted %d rows", game_id, len(shifts), inserted)

    logging.info(
        "Season %s: total game shifts inserted = %d",
        season,
        total_inserted,
    )

    backfilled = backfill_game_shift_player_names(db_path)
    if backfilled > 0:
        logging.info("Season %s: backfilled %d shift player names from players table", season, backfilled)

    return total_inserted


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch game shifts from NHL API")
    parser.add_argument("--season", required=True, help="Season start year (e.g., 2023)")
    parser.add_argument("--game-type", default=REGULAR_SEASON, choices=["01", "02", "03", "04"])
    parser.add_argument("--db-path", default="hockey_data.db")
    parser.add_argument("--timeout", type=int, default=10)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    fetch_and_store_game_shifts(args.db_path, args.season, args.game_type, args.timeout)