from __future__ import annotations

from dataclasses import dataclass
import sqlite3
from typing import Any


DEFAULT_BIN_SIZE = 5.0


@dataclass(frozen=True)
class HeatmapFilters:
   season: str | None = None
   team: str | None = None
   player: str | None = None
   strength_state: str | None = None
   shot_result: str | None = None
   home_away: int | None = None
   period: int | None = None


def _connect(db_path: str) -> sqlite3.Connection:
   connection = sqlite3.connect(db_path)
   connection.row_factory = sqlite3.Row
   return connection


def _clean_text(value: str | None) -> str | None:
   if value is None:
      return None
   text = value.strip()
   return text or None


def _build_where(filters: HeatmapFilters) -> tuple[str, list[Any]]:
   clauses: list[str] = []
   params: list[Any] = []

   if filters.season:
      clauses.append("season = ?")
      params.append(filters.season)
   if filters.team:
      clauses.append("team = ?")
      params.append(filters.team)
   if filters.player:
      clauses.append("shooter = ?")
      params.append(filters.player)
   if filters.strength_state:
      clauses.append("strength_state = ?")
      params.append(filters.strength_state)
   if filters.home_away is not None:
      clauses.append("home_away = ?")
      params.append(filters.home_away)
   if filters.period is not None:
      clauses.append("period = ?")
      params.append(filters.period)
   if filters.shot_result and filters.shot_result != "all":
      clauses.append("shot_result = ?")
      params.append(filters.shot_result)

   if not clauses:
      return "1 = 1", params
   return " AND ".join(clauses), params


def latest_season(db_path: str) -> str | None:
   with _connect(db_path) as connection:
      row = connection.execute("SELECT MAX(CAST(season AS INTEGER)) AS season FROM shots").fetchone()
   if row is None or row["season"] is None:
      return None
   return str(row["season"])


def get_filter_options(db_path: str) -> dict[str, list[str]]:
   with _connect(db_path) as connection:
      season_rows = connection.execute(
         "SELECT DISTINCT season FROM shots WHERE season IS NOT NULL ORDER BY CAST(season AS INTEGER) DESC"
      ).fetchall()
      team_rows = connection.execute(
         "SELECT DISTINCT team FROM shots WHERE team IS NOT NULL AND team != '' ORDER BY team"
      ).fetchall()
      player_rows = connection.execute(
         "SELECT DISTINCT shooter FROM shots WHERE shooter IS NOT NULL AND shooter != '' ORDER BY shooter"
      ).fetchall()
      strength_rows = connection.execute(
         "SELECT DISTINCT strength_state FROM shots WHERE strength_state IS NOT NULL AND strength_state != '' ORDER BY strength_state"
      ).fetchall()
      period_rows = connection.execute(
         "SELECT DISTINCT period FROM shots WHERE period IS NOT NULL ORDER BY period"
      ).fetchall()

   return {
      "seasons": [row["season"] for row in season_rows if row["season"]],
      "teams": [row["team"] for row in team_rows if row["team"]],
      "players": [row["shooter"] for row in player_rows if row["shooter"]],
      "strength_states": [row["strength_state"] for row in strength_rows if row["strength_state"]],
      "periods": [str(row["period"]) for row in period_rows if row["period"] is not None],
      "shot_results": ["all", "Goal", "ngshot"],
      "home_away": ["all", "home", "away"],
   }


def get_summary(db_path: str, filters: HeatmapFilters) -> dict[str, Any]:
   where_clause, params = _build_where(filters)
   query = f"""
      SELECT
         COUNT(*) AS shot_count,
         SUM(CASE WHEN shot_result = 'Goal' THEN 1 ELSE 0 END) AS goal_count
      FROM shots
      WHERE {where_clause}
   """

   with _connect(db_path) as connection:
      row = connection.execute(query, params).fetchone()

   shot_count = int(row["shot_count"] or 0) if row else 0
   goal_count = int(row["goal_count"] or 0) if row and row["goal_count"] is not None else 0
   goal_pct = (goal_count / shot_count * 100.0) if shot_count else 0.0
   return {
      "shot_count": shot_count,
      "goal_count": goal_count,
      "goal_pct": round(goal_pct, 2),
   }


def get_heatmap_bins(db_path: str, filters: HeatmapFilters, bin_size: float = DEFAULT_BIN_SIZE) -> list[dict[str, Any]]:
   where_clause, params = _build_where(filters)
   query = f"""
      SELECT
         CAST((x + 100.0) / ? AS INTEGER) AS bin_x,
         CAST((y + 42.5) / ? AS INTEGER) AS bin_y,
         COUNT(*) AS shot_count,
         SUM(CASE WHEN shot_result = 'Goal' THEN 1 ELSE 0 END) AS goal_count
      FROM shots
      WHERE {where_clause}
      GROUP BY bin_x, bin_y
      ORDER BY shot_count DESC
   """

   results: list[dict[str, Any]] = []
   with _connect(db_path) as connection:
      rows = connection.execute(query, [bin_size, bin_size, *params]).fetchall()

   for row in rows:
      shot_count = int(row["shot_count"] or 0)
      goal_count = int(row["goal_count"] or 0) if row["goal_count"] is not None else 0
      x_bin = int(row["bin_x"])
      y_bin = int(row["bin_y"])
      center_x = (x_bin * bin_size) - 100.0 + (bin_size / 2.0)
      center_y = (y_bin * bin_size) - 42.5 + (bin_size / 2.0)
      goal_pct = (goal_count / shot_count * 100.0) if shot_count else 0.0

      results.append(
         {
            "x": round(center_x, 2),
            "y": round(center_y, 2),
            "bin_x": x_bin,
            "bin_y": y_bin,
            "shot_count": shot_count,
            "goal_count": goal_count,
            "goal_pct": round(goal_pct, 2),
         }
      )

   return results


def build_filters_from_args(args: dict[str, str]) -> HeatmapFilters:
   home_away_value: int | None = None
   raw_home_away = _clean_text(args.get("home_away"))
   if raw_home_away == "home":
      home_away_value = 1
   elif raw_home_away == "away":
      home_away_value = 0

   period_value: int | None = None
   raw_period = _clean_text(args.get("period"))
   if raw_period:
      try:
         period_value = int(raw_period)
      except ValueError:
         period_value = None

   return HeatmapFilters(
      season=_clean_text(args.get("season")),
      team=_clean_text(args.get("team")),
      player=_clean_text(args.get("player")),
      strength_state=_clean_text(args.get("strength_state")),
      shot_result=_clean_text(args.get("shot_result")),
      home_away=home_away_value,
      period=period_value,
   )
