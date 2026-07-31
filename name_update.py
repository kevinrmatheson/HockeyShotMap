#!/usr/bin/env python3
"""
Quick backfill script for game_shifts.player_name.

Updates missing/Unknown names in game_shifts using player_seasonal_stats:
1) First pass: match by (player_id, season)
2) Second pass: fallback match by player_id only
"""

from __future__ import annotations

import argparse
import logging
import sqlite3


def _table_exists(connection: sqlite3.Connection, table_name: str) -> bool:
    row = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    return row is not None


def _row_count(connection: sqlite3.Connection, query: str, params: tuple = ()) -> int:
    row = connection.execute(query, params).fetchone()
    return int(row[0]) if row else 0


def backfill_shift_names_from_player_season(db_path: str) -> dict[str, int]:
    with sqlite3.connect(db_path) as connection:
        if not _table_exists(connection, "game_shifts"):
            raise ValueError("game_shifts table not found")
        if not _table_exists(connection, "player_seasonal_stats"):
            raise ValueError("player_seasonal_stats table not found")

        before_missing = _row_count(
            connection,
            """
            SELECT COUNT(*)
            FROM game_shifts
            WHERE player_name IS NULL OR TRIM(player_name) = '' OR player_name = 'Unknown'
            """,
        )

        cursor = connection.cursor()

        # Pass 1: season-aware mapping (most specific)
        cursor.execute(
            """
            UPDATE game_shifts
            SET player_name = (
                SELECT p.player_name
                FROM player_seasonal_stats p
                WHERE p.player_id = game_shifts.player_id
                  AND p.season = game_shifts.season
                  AND p.player_name IS NOT NULL
                  AND TRIM(p.player_name) <> ''
                  AND p.player_name <> 'Unknown'
                ORDER BY p.games_played DESC, p.toi_seconds DESC
                LIMIT 1
            )
            WHERE (player_name IS NULL OR TRIM(player_name) = '' OR player_name = 'Unknown')
              AND EXISTS (
                SELECT 1
                FROM player_seasonal_stats p
                WHERE p.player_id = game_shifts.player_id
                  AND p.season = game_shifts.season
                  AND p.player_name IS NOT NULL
                  AND TRIM(p.player_name) <> ''
                  AND p.player_name <> 'Unknown'
              )
            """
        )
        season_pass_updated = int(cursor.rowcount or 0)

        # Pass 2: player_id-only fallback
        cursor.execute(
            """
            UPDATE game_shifts
            SET player_name = (
                SELECT p.player_name
                FROM player_seasonal_stats p
                WHERE p.player_id = game_shifts.player_id
                  AND p.player_name IS NOT NULL
                  AND TRIM(p.player_name) <> ''
                  AND p.player_name <> 'Unknown'
                ORDER BY p.games_played DESC, p.toi_seconds DESC
                LIMIT 1
            )
            WHERE (player_name IS NULL OR TRIM(player_name) = '' OR player_name = 'Unknown')
              AND EXISTS (
                SELECT 1
                FROM player_seasonal_stats p
                WHERE p.player_id = game_shifts.player_id
                  AND p.player_name IS NOT NULL
                  AND TRIM(p.player_name) <> ''
                  AND p.player_name <> 'Unknown'
              )
            """
        )
        fallback_pass_updated = int(cursor.rowcount or 0)

        connection.commit()

        after_missing = _row_count(
            connection,
            """
            SELECT COUNT(*)
            FROM game_shifts
            WHERE player_name IS NULL OR TRIM(player_name) = '' OR player_name = 'Unknown'
            """,
        )

    return {
        "before_missing": before_missing,
        "season_pass_updated": season_pass_updated,
        "fallback_pass_updated": fallback_pass_updated,
        "after_missing": after_missing,
        "total_updated": season_pass_updated + fallback_pass_updated,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill game_shifts.player_name from player_seasonal_stats",
    )
    parser.add_argument("--db-path", default="hockey_data.db", help="SQLite database path")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")

    try:
        summary = backfill_shift_names_from_player_season(args.db_path)
    except Exception as exc:
        logging.error("Name backfill failed: %s", exc)
        return 1

    logging.info("Name backfill complete: %s", summary)
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
