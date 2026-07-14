#!/usr/bin/env python3
"""
NHL Team Season Stats Scraper.

Fetches team season statistics from the NHL Stats REST API and stores them
in the team_seasonal_stats table.

API Endpoint: https://api.nhle.com/stats/rest/v1/en/team/summary
"""

import logging
import sqlite3
from datetime import UTC, datetime

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Reuse constants from shots_scraper
from shots_scraper import (
    build_stats_games_url,
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
_TEAM_SEASON_HTTP_SESSION = requests.Session()


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


_TEAM_SEASON_HTTP_SESSION.mount("http://", _build_retry_adapter())
_TEAM_SEASON_HTTP_SESSION.mount("https://", _build_retry_adapter())


def _get_http_session() -> requests.Session:
    return _TEAM_SEASON_HTTP_SESSION


def build_stats_team_summary_url() -> str:
    """Build URL for the Stats REST team summary endpoint."""
    return "https://api.nhle.com/stats/rest/v1/en/team/summary"


def _parse_team_season_stats_row(record: dict, season: str, game_type: str) -> dict | None:
    """Parse a single team-season-stats record into a normalized row."""
    team_id = record.get("teamId")
    if team_id is None:
        return None

    return {
        "team_id": int(team_id),
        "team_name": str(record.get("teamName", record.get("team", {}).get("name", "Unknown"))),
        "team_abbrev": str(record.get("teamAbbrev", record.get("team", {}).get("abbrev", "Unknown"))),
        "team_city": str(record.get("teamCity", record.get("team", {}).get("city", "Unknown"))),
        "season": season,
        "game_type": game_type,
        "games_played": int(record.get("gamesPlayed", 0) or 0),
        "wins": int(record.get("wins", 0) or 0),
        "losses": int(record.get("losses", 0) or 0),
        "ot_losses": int(record.get("otLosses", 0) or 0),
        "shootouts_wins": int(record.get("shootoutWins", 0) or 0),
        "shootouts_losses": int(record.get("shootoutLosses", 0) or 0),
        "goals_for": int(record.get("goalsFor", 0) or 0),
        "goals_against": int(record.get("goalsAgainst", 0) or 0),
        "goal_differential": int(record.get("goalDifferential", 0) or 0),
        "power_play_opportunities": int(record.get("powerPlayOpportunities", 0) or 0),
        "power_play_percentage": float(record.get("powerPlayPercentage", 0.0) or 0.0),
        "penalty_killing_opportunities": int(record.get("penaltyKillOpportunities", 0) or 0),
        "penalty_kill_percentage": float(record.get("penaltyKillPercentage", 0.0) or 0.0),
        "shots_for": int(record.get("shotsFor", 0) or 0),
        "shots_against": int(record.get("shotsAgainst", 0) or 0),
        "save_percentage": float(record.get("savePercentage", 0.0) or 0.0),
        "faceoff_wins": int(record.get("faceOffWins", 0) or 0),
        "faceoff_losses": int(record.get("faceOffLosses", 0) or 0),
        "takeaways": int(record.get("takeaways", 0) or 0),
        "giveaways": int(record.get("giveaways", 0) or 0),
        "blocked_shots": int(record.get("blockedShots", 0) or 0),
        "hits": int(record.get("hits", 0) or 0),
        "pim": int(record.get("pim", 0) or 0),
        "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def initialize_team_season_database(db_path: str) -> None:
    """Create team_seasonal_stats table if it doesn't exist."""
    with sqlite3.connect(db_path) as connection:
        cursor = connection.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS team_seasonal_stats (
                team_id INTEGER NOT NULL,
                team_name TEXT NOT NULL,
                team_abbrev TEXT NOT NULL,
                team_city TEXT NOT NULL,
                season TEXT NOT NULL,
                game_type TEXT NOT NULL,
                games_played INTEGER NOT NULL,
                wins INTEGER NOT NULL,
                losses INTEGER NOT NULL,
                ot_losses INTEGER NOT NULL,
                shootouts_wins INTEGER NOT NULL,
                shootouts_losses INTEGER NOT NULL,
                goals_for INTEGER NOT NULL,
                goals_against INTEGER NOT NULL,
                goal_differential INTEGER NOT NULL,
                power_play_opportunities INTEGER NOT NULL,
                power_play_percentage REAL NOT NULL,
                penalty_killing_opportunities INTEGER NOT NULL,
                penalty_kill_percentage REAL NOT NULL,
                shots_for INTEGER NOT NULL,
                shots_against INTEGER NOT NULL,
                save_percentage REAL NOT NULL,
                faceoff_wins INTEGER NOT NULL,
                faceoff_losses INTEGER NOT NULL,
                takeaways INTEGER NOT NULL,
                giveaways INTEGER NOT NULL,
                blocked_shots INTEGER NOT NULL,
                hits INTEGER NOT NULL,
                pim INTEGER NOT NULL,
                fetched_at TEXT NOT NULL,
                PRIMARY KEY (team_id, season, game_type)
            )
            """
        )
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_team_seasonal_stats_season ON team_seasonal_stats(season)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_team_seasonal_stats_team_abbrev ON team_seasonal_stats(team_abbrev)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_team_seasonal_stats_team_city ON team_seasonal_stats(team_city)")
        connection.commit()


def fetch_and_store_team_season_stats(
    db_path: str,
    season: str,
    game_type: str = REGULAR_SEASON,
    timeout_seconds: int = 10,
) -> int:
    """
    Fetch team season stats from the Stats REST API and store in team_seasonal_stats table.
    Returns the number of rows inserted.
    """
    initialize_team_season_database(db_path)

    season_id = int(_season_id_yyyyyyyy(season))
    team_url = build_stats_team_summary_url()
    inserted = 0

    # Fetch team stats
    params = {
        "cayenneExp": f"seasonId={season_id} and gameTypeId={int(game_type)}",
        "limit": -1,
    }
    payload = _fetch_json(team_url, timeout_seconds, params=params)
    if payload is None:
        logging.warning("Failed to fetch team season stats for season %s (game_type=%s)", season, game_type)
        return 0

    rows = _extract_record_list(payload)
    if not rows:
        logging.warning("No team season rows returned for season %s (game_type=%s).", season, game_type)
        return 0

    # Parse and persist to database
    with sqlite3.connect(db_path) as connection:
        cursor = connection.cursor()
        for record in rows:
            row = _parse_team_season_stats_row(record, season, game_type)
            if row is None:
                continue

            try:
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO team_seasonal_stats (
                        team_id, team_name, team_abbrev, team_city, season, game_type,
                        games_played, wins, losses, ot_losses, shootouts_wins, shootouts_losses,
                        goals_for, goals_against, goal_differential,
                        power_play_opportunities, power_play_percentage,
                        penalty_killing_opportunities, penalty_kill_percentage,
                        shots_for, shots_against, save_percentage,
                        faceoff_wins, faceoff_losses, takeaways, giveaways,
                        blocked_shots, hits, pim, fetched_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row["team_id"], row["team_name"], row["team_abbrev"], row["team_city"],
                        row["season"], row["game_type"],
                        row["games_played"], row["wins"], row["losses"], row["ot_losses"],
                        row["shootouts_wins"], row["shootouts_losses"],
                        row["goals_for"], row["goals_against"], row["goal_differential"],
                        row["power_play_opportunities"], row["power_play_percentage"],
                        row["penalty_killing_opportunities"], row["penalty_kill_percentage"],
                        row["shots_for"], row["shots_against"], row["save_percentage"],
                        row["faceoff_wins"], row["faceoff_losses"], row["takeaways"],
                        row["giveaways"], row["blocked_shots"], row["hits"], row["pim"],
                        row["fetched_at"],
                    ),
                )
                inserted += 1
            except sqlite3.Error as exc:
                logging.debug("Skipping team %s: %s", row.get("team_id"), exc)

        connection.commit()

    logging.info(
        "Season %s: stored %d team season stats rows from %d records",
        season,
        inserted,
        len(rows),
    )
    return inserted


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch team season stats from NHL API")
    parser.add_argument("--season", required=True, help="Season start year (e.g., 2023)")
    parser.add_argument("--game-type", default=REGULAR_SEASON, choices=["01", "02", "03", "04"])
    parser.add_argument("--db-path", default="hockey_data.db")
    parser.add_argument("--timeout", type=int, default=10)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    fetch_and_store_team_season_stats(args.db_path, args.season, args.game_type, args.timeout)