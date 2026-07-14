#!/usr/bin/env python3
"""
NHL Player Season Stats Scraper.

Fetches player season statistics from the NHL Stats REST API and stores them
in the player_seasonal_stats table.

API Endpoint: https://api.nhle.com/stats/rest/v1/en/skater/summary
              https://api.nhle.com/stats/rest/v1/en/goalie/summary
"""

import logging
import sqlite3
from datetime import UTC, datetime

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Reuse constants from shots_scraper
from shots_scraper import (
    build_stats_skater_summary_url,
    build_stats_goalie_summary_url,
    _fetch_json,
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
_PLAYER_SEASON_HTTP_SESSION = requests.Session()


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


_PLAYER_SEASON_HTTP_SESSION.mount("http://", _build_retry_adapter())
_PLAYER_SEASON_HTTP_SESSION.mount("https://", _build_retry_adapter())


def _get_http_session() -> requests.Session:
    return _PLAYER_SEASON_HTTP_SESSION


def _parse_player_season_stats_row(record: dict, season: str, game_type: str) -> dict | None:
    """Parse a single player-season-stats record into a normalized row."""
    player_id = record.get("playerId")
    if player_id is None:
        return None

    games_played = int(record.get("gamesPlayed", 0) or 0)
    toi_seconds = _coerce_int(record.get("timeOnIce", record.get("toi")))
    if toi_seconds is None:
        toi_per_game = _coerce_float(record.get("timeOnIcePerGame"))
        toi_seconds = int((toi_per_game or 0.0) * games_played)

    position = _normalize_text(record.get("positionCode") or record.get("position"))
    if not position and record.get("goalieFullName"):
        position = "G"

    return {
        "player_id": int(player_id),
        "player_name": str(
            record.get("playerName")
            or record.get("fullName")
            or record.get("skaterFullName")
            or record.get("goalieFullName")
            or "Unknown"
        ),
        "season": season,
        "game_type": game_type,
        "team": str(record.get("teamAbbrev", record.get("teamAbbrevs", record.get("team", "")))),
        "position": str(position or ""),
        "games_played": games_played,
        "toi_seconds": int(toi_seconds or 0),
        "goals": int(record.get("goals", 0) or 0),
        "assists": int(record.get("assists", 0) or 0),
        "points": int(record.get("points", 0) or 0),
        "shots_on_goal": int(record.get("shotsOnGoal", record.get("shots", 0)) or 0),
        "plus_minus": int(record.get("plusMinus", record.get("plusminus", 0)) or 0),
        "penalty_minutes": int(record.get("penaltyMinutes", record.get("pim", 0)) or 0),
        "power_play_goals": int(record.get("powerPlayGoals", record.get("ppGoals", 0)) or 0),
        "short_handed_goals": int(record.get("shortHandedGoals", record.get("shGoals", 0)) or 0),
        "game_winning_goals": int(record.get("gameWinningGoals", 0) or 0),
        "blocked_shots": int(record.get("blockedShots", 0) or 0),
        "hits": int(record.get("hits", 0) or 0),
        "faceoffs_won": int(record.get("faceoffsWon", 0) or 0),
        "faceoffs_lost": int(record.get("faceoffsLost", 0) or 0),
        "takeaways": int(record.get("takeaways", 0) or 0),
        "giveaways": int(record.get("giveaways", 0) or 0),
        "save_pct": float(record.get("savePct", 0.0) or 0.0),
        "goals_against_avg": float(record.get("goalsAgainstAverage", record.get("gaa", 0.0)) or 0.0),
        "shutouts": int(record.get("shutouts", 0) or 0),
    }


def initialize_player_season_database(db_path: str) -> None:
    """Create player_seasonal_stats table if it doesn't exist."""
    with sqlite3.connect(db_path) as connection:
        cursor = connection.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS player_seasonal_stats (
                player_id INTEGER NOT NULL,
                player_name TEXT NOT NULL,
                season TEXT NOT NULL,
                game_type TEXT NOT NULL,
                team TEXT,
                position TEXT,
                games_played INTEGER NOT NULL,
                toi_seconds INTEGER NOT NULL,
                goals INTEGER NOT NULL,
                assists INTEGER NOT NULL,
                points INTEGER NOT NULL,
                shots_on_goal INTEGER NOT NULL,
                plus_minus INTEGER NOT NULL,
                penalty_minutes INTEGER NOT NULL,
                power_play_goals INTEGER NOT NULL,
                short_handed_goals INTEGER NOT NULL,
                game_winning_goals INTEGER NOT NULL,
                blocked_shots INTEGER NOT NULL,
                hits INTEGER NOT NULL,
                faceoffs_won INTEGER NOT NULL,
                faceoffs_lost INTEGER NOT NULL,
                takeaways INTEGER NOT NULL,
                giveaways INTEGER NOT NULL,
                save_pct REAL NOT NULL,
                goals_against_avg REAL NOT NULL,
                shutouts INTEGER NOT NULL,
                fetched_at TEXT NOT NULL DEFAULT (datetime('now')),
                PRIMARY KEY (player_id, season, game_type)
            )
            """
        )
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_player_seasonal_stats_season ON player_seasonal_stats(season)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_player_seasonal_stats_team ON player_seasonal_stats(team)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_player_seasonal_stats_position ON player_seasonal_stats(position)")
        connection.commit()


def fetch_and_store_player_season_stats(
    db_path: str,
    season: str,
    game_type: str = REGULAR_SEASON,
    timeout_seconds: int = 10,
) -> int:
    """
    Fetch player season stats from the Stats REST API and store in player_seasonal_stats table.
    Returns the number of rows inserted.
    """
    initialize_player_season_database(db_path)

    season_id = int(_season_id_yyyyyyyy(season))
    skater_url = build_stats_skater_summary_url()
    goalie_url = build_stats_goalie_summary_url()
    inserted = 0

    # Fetch skater stats
    skater_params = {
        "cayenneExp": f"seasonId={season_id} and gameTypeId={int(game_type)}",
        "limit": -1,
    }
    skater_payload = _fetch_json(skater_url, timeout_seconds, params=skater_params)
    if skater_payload is None:
        logging.warning("Failed to fetch skater season stats for season %s (game_type=%s)", season, game_type)

    skater_rows = _extract_record_list(skater_payload)

    # Fetch goalie stats separately
    goalie_params = {
        "cayenneExp": f"seasonId={season_id} and gameTypeId={int(game_type)}",
        "limit": -1,
    }
    goalie_payload = _fetch_json(goalie_url, timeout_seconds, params=goalie_params)
    goalie_rows = []
    if goalie_payload is not None:
        goalie_rows = _extract_record_list(goalie_payload)

    if not skater_rows and not goalie_rows:
        logging.warning("No player season rows returned for season %s (game_type=%s).", season, game_type)
        return 0

    # Merge and deduplicate by player_id
    all_records: dict[int, dict] = {}
    for record in skater_rows + goalie_rows:
        pid = record.get("playerId")
        if pid is not None:
            pid = int(pid)
            # Goalie payload takes precedence for goalie-specific fields
            if pid not in all_records or record.get("savePct") is not None:
                all_records[pid] = record

    parsed_rows = []
    for record in all_records.values():
        row = _parse_player_season_stats_row(record, season, game_type)
        if row is not None:
            parsed_rows.append(row)

    if not parsed_rows:
        logging.info("No player season stats found for season %s", season)
        return 0

    # Persist to database
    with sqlite3.connect(db_path) as connection:
        cursor = connection.cursor()
        for row in parsed_rows:
            try:
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO player_seasonal_stats (
                        player_id, player_name, season, game_type, team, position,
                        games_played, toi_seconds, goals, assists, points, shots_on_goal,
                        plus_minus, penalty_minutes, power_play_goals, short_handed_goals,
                        game_winning_goals, blocked_shots, hits, faceoffs_won, faceoffs_lost,
                        takeaways, giveaways, save_pct, goals_against_avg, shutouts
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row["player_id"], row["player_name"], row["season"], row["game_type"],
                        row["team"], row["position"],
                        row["games_played"], row["toi_seconds"], row["goals"], row["assists"],
                        row["points"], row["shots_on_goal"], row["plus_minus"],
                        row["penalty_minutes"], row["power_play_goals"], row["short_handed_goals"],
                        row["game_winning_goals"], row["blocked_shots"], row["hits"],
                        row["faceoffs_won"], row["faceoffs_lost"], row["takeaways"],
                        row["giveaways"], row["save_pct"], row["goals_against_avg"],
                        row["shutouts"],
                    ),
                )
                inserted += 1
            except sqlite3.Error as exc:
                logging.debug("Skipping player %s: %s", row["player_id"], exc)

        connection.commit()

    logging.info(
        "Season %s: stored %d player season stats rows from %d records",
        season,
        inserted,
        len(parsed_rows),
    )
    return inserted


def fetch_player_season_stats_range(
    db_path: str,
    start_season: str,
    end_season: str,
    game_type: str = REGULAR_SEASON,
    timeout_seconds: int = 10,
) -> int:
    """
    Fetch player season stats for a range of seasons and store in player_seasonal_stats table.
    Returns the total number of rows inserted across all seasons.
    """
    start_year = int(start_season)
    end_year = int(end_season)

    if start_year > end_year:
        raise ValueError(f"start_season ({start_season}) must be <= end_season ({end_season})")

    total_inserted = 0
    for year in range(start_year, end_year + 1):
        season = str(year)
        inserted = fetch_and_store_player_season_stats(db_path, season, game_type, timeout_seconds)
        total_inserted += inserted

    logging.info(
        "Season range %s-%s: total %d player season stats rows inserted",
        start_season,
        end_season,
        total_inserted,
    )
    return total_inserted


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch player season stats from NHL API")
    parser.add_argument("--start-season", required=True, help="Start season year (e.g., 2023)")
    parser.add_argument("--end-season", required=True, help="End season year (e.g., 2024)")
    parser.add_argument("--game-type", default=REGULAR_SEASON, choices=["01", "02", "03", "04"])
    parser.add_argument("--db-path", default="hockey_data.db")
    parser.add_argument("--timeout", type=int, default=10)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    fetch_player_season_stats_range(args.db_path, args.start_season, args.end_season, args.game_type, args.timeout)