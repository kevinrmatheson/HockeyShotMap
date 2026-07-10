import argparse
import base64
import hashlib
import json
import logging
import math
import pickle
import sqlite3
from dataclasses import dataclass
from datetime import UTC, date, datetime

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from age_model import (
   _build_age_xg_curve,
   _player_trend_multiplier,
   compute_age_adjusted_gax,
   compute_age_adjusted_gax_per_60,
)


@dataclass
class MetricsConfig:
   db_path: str
   score_start_season: int
   score_end_season: int | None = None
   train_start_season: int | None = None
   train_end_season: int | None = None
   min_shots_for_comparison: int = 50
   learning_rate: float = 0.05
   epochs: int = 400
   l2_regularization: float = 0.0005
   validation_split: float = 0.2
   validation_split_strategy: str = "random"  # "random" or "temporal"
   calibration_method: str = "sigmoid"
   calibration_bins: int = 10
   random_seed: int = 42
   xgb_max_depth: int = 4
   xgb_subsample: float = 0.9
   xgb_colsample_bytree: float = 0.9
   xgb_min_child_weight: float = 1.0
   use_player_effects: bool = True
   career_lookback_seasons: int = 3
   # Options for enhanced metrics
   compute_rate_metrics: bool = True  # Compute per-60 rate metrics
   rolling_window_seasons: int = 3  # Number of seasons for rolling benchmarks
   include_age_adjusted: bool = True  # Age-adjusted career trajectory comparisons
   force_refresh: bool = False  # Ignore staleness check; re-score and replace all target seasons


FEATURE_SPEC_VERSION = "v6_xgb_prior_event_groups_expanded"


def classify_shot_strength(situation_code: int, event_team_side: str) -> str:
   """
   Classify shot strength from the shooter's perspective using NHL API situationCode.
   
   The situationCode is a 4-digit number where each digit represents:
   - Digit 0 (AwayGoalie): 0 = pulled, 1 = in net
   - Digit 1 (AwaySkaters): number of away team skaters on ice
   - Digit 2 (HomeSkaters): number of home team skaters on ice
   - Digit 3 (HomeGoalie): 0 = pulled, 1 = in net
   
   Parameters
   ----------
   situation_code : int
       4-digit NHL API situation code (e.g., 1551 for 5v5, 1560 for 6v5 EN)
   event_team_side : str
       'Home' or 'Away' - which team took the shot
   
   Returns
   -------
   str
       One of: '5v5', 'Even_Opened', 'PP_1', 'PP_2', 'PK_1', 'PK_2', 'EN_For', 'EN_Against', 'Other'
   
   Notes
   -----
   Classification is from the shooter's perspective:
   - EN_For: Shooter's goalie is pulled (extra attacker for shooter's team)
   - EN_Against: Shooting at an open net (opponent's goalie is pulled)
   """
   # Validate inputs
   if event_team_side not in ('Home', 'Away'):
      raise ValueError(f"event_team_side must be 'Home' or 'Away', got: {event_team_side}")
   
   # Convert to string and pad with leading zeros if needed
   code_str = f"{situation_code:04d}"
   
   # Extract digits: AwayGoalie, AwaySkaters, HomeSkaters, HomeGoalie
   away_goalie = int(code_str[0])
   away_skaters = int(code_str[1])
   home_skaters = int(code_str[2])
   home_goalie = int(code_str[3])
   
   # Determine shooter's and opponent's values based on team side
   if event_team_side == 'Home':
      shooter_goalie = home_goalie
      shooter_skaters = home_skaters
      opp_goalie = away_goalie
      opp_skaters = away_skaters
   else:  # Away
      shooter_goalie = away_goalie
      shooter_skaters = away_skaters
      opp_goalie = home_goalie
      opp_skaters = home_skaters
   
   # Calculate skater advantage from shooter's perspective
   skater_diff = shooter_skaters - opp_skaters
   
   # Check for empty net situations first (goalie pulled = 0)
   if shooter_goalie == 0 and opp_goalie == 1:
      # Shooter's goalie pulled, opponent's goalie in net
      return 'EN_For'
   elif shooter_goalie == 1 and opp_goalie == 0:
      # Shooter's goalie in net, opponent's goalie pulled
      return 'EN_Against'
   elif shooter_goalie == 0 and opp_goalie == 0:
      # Both goalies pulled - unusual but possible in late-game situations
      return 'Other'
   
   # Both goalies in net - classify by skater advantage
   if shooter_skaters == opp_skaters:
      if shooter_skaters == 5:
         return '5v5'
      else:
         # Equal but less than 5 (4v4, 3v3, etc.)
         return 'Even_Opened'
   elif skater_diff == 1:
      return 'PP_1'
   elif skater_diff == 2:
      return 'PP_2'
   elif skater_diff == -1:
      return 'PK_1'
   elif skater_diff == -2:
      return 'PK_2'
   else:
      # Any other mismatch (3v5, 4v6, etc.)
      return 'Other'


# Test cases for the classify_shot_strength function
def _test_classify_shot_strength():
   """Test the classify_shot_strength function with various edge cases."""
   # 5v5 - both goalies in net, 5 skaters each
   # 1551: AwayGoalie=1, AwaySkaters=5, HomeSkaters=5, HomeGoalie=1
   assert classify_shot_strength(1551, 'Home') == '5v5'
   assert classify_shot_strength(1551, 'Away') == '5v5'
   
   # 4v4 - both goalies in net, equal but less than 5
   # 1441: AwayGoalie=1, AwaySkaters=4, HomeSkaters=4, HomeGoalie=1
   assert classify_shot_strength(1441, 'Home') == 'Even_Opened'
   assert classify_shot_strength(1441, 'Away') == 'Even_Opened'
   
   # PP_1 - shooter has +1 advantage, both goalies in net
   # 1551 is 5v5, need 5v4: AwayGoalie=1, AwaySkaters=4, HomeSkaters=5, HomeGoalie=1
   # Code: 1451 (Away 4v5 Home)
   # When Home shoots: Home=5, Away=4, diff=+1 -> PP_1
   assert classify_shot_strength(1451, 'Home') == 'PP_1'  # Home 5v4
   # When Away shoots: Away=4, Home=5, diff=-1 -> PK_1
   assert classify_shot_strength(1451, 'Away') == 'PK_1'  # Away 4v5
   
   # PP_2 - shooter has +2 advantage
   # 6v4: AwayGoalie=1, AwaySkaters=4, HomeSkaters=6, HomeGoalie=1 -> Code: 1461
   assert classify_shot_strength(1461, 'Home') == 'PP_2'  # Home 6v4
   assert classify_shot_strength(1461, 'Away') == 'PK_2'  # Away 4v6
   
   # PK_1 - shooter has -1 disadvantage (already covered above)
   # 4v5: AwayGoalie=1, AwaySkaters=5, HomeSkaters=4, HomeGoalie=1 -> Code: 1541
   assert classify_shot_strength(1541, 'Home') == 'PK_1'  # Home 4v5
   assert classify_shot_strength(1541, 'Away') == 'PP_1'  # Away 5v4
   
   # PK_2 - shooter has -2 disadvantage
   # 3v5: AwayGoalie=1, AwaySkaters=5, HomeSkaters=3, HomeGoalie=1 -> Code: 1531
   assert classify_shot_strength(1531, 'Home') == 'PK_2'  # Home 3v5
   assert classify_shot_strength(1531, 'Away') == 'PP_2'  # Away 5v3
   
   # EN_For - shooter's goalie pulled (extra attacker)
   # 1560: AwayGoalie=1, AwaySkaters=5, HomeSkaters=6, HomeGoalie=0
   # When Home shoots: Home goalie=0 (pulled), Away goalie=1 -> EN_For
   # When Away shoots: Away goalie=1, Home goalie=0 -> EN_Against
   assert classify_shot_strength(1560, 'Home') == 'EN_For'  # Home has extra attacker
   assert classify_shot_strength(1560, 'Away') == 'EN_Against'  # Away shoots at open net
   
   # EN_Against - opponent's goalie pulled (shooting at open net)
   # 0651: AwayGoalie=0, AwaySkaters=6, HomeSkaters=5, HomeGoalie=1
   # When Home shoots: Home goalie=1, Away goalie=0 -> EN_Against
   # When Away shoots: Away goalie=0, Home goalie=1 -> EN_For
   assert classify_shot_strength(651, 'Home') == 'EN_Against'  # Home shoots at open net
   assert classify_shot_strength(651, 'Away') == 'EN_For'  # Away has extra attacker
   
   # Other - both goalies pulled
   assert classify_shot_strength(0, 'Home') == 'Other'
   assert classify_shot_strength(0, 'Away') == 'Other'
   
   # Other - unusual mismatches (7v5, etc.)
   # 1751: AwayGoalie=1, AwaySkaters=5, HomeSkaters=7, HomeGoalie=1 -> 5v7
   # Home has -2 advantage, Away has +2
   assert classify_shot_strength(1751, 'Home') == 'PK_2'  # Home 5v7
   assert classify_shot_strength(1751, 'Away') == 'PP_2'  # Away 7v5
   
   # Test a truly unusual case: 8v5 (3+ advantage)
   # 1851: AwayGoalie=1, AwaySkaters=5, HomeSkaters=8, HomeGoalie=1 -> 5v8
   assert classify_shot_strength(1851, 'Home') == 'Other'  # 5v8 is very unusual
   
   print("All test cases passed!")


# Run tests when module is executed directly
if __name__ == "__main__":
   _test_classify_shot_strength()


def _backfill_player_names(connection: sqlite3.Connection) -> dict[str, int]:
   """Fill missing shooter/goalie names in shots table from players table."""
   cursor = connection.cursor()

   # Check if the players table exists first
   cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='players'")
   if cursor.fetchone() is None:
      return {"shooter_filled": 0, "goalie_filled": 0}

   # Backfill shooter names
   cursor.execute("""
      UPDATE shots
      SET shooter = (
         SELECT full_name FROM players WHERE players.player_id = shots.shooter_id
      )
      WHERE (shooter IS NULL OR shooter = '')
        AND shooter_id IS NOT NULL
        AND EXISTS (SELECT 1 FROM players WHERE players.player_id = shots.shooter_id)
   """)
   shooter_filled = cursor.rowcount

   # Backfill goalie names
   cursor.execute("""
      UPDATE shots
      SET goalie = (
         SELECT full_name FROM players WHERE players.player_id = shots.goalie_id
      )
      WHERE (goalie IS NULL OR goalie = '')
        AND goalie_id IS NOT NULL
        AND EXISTS (SELECT 1 FROM players WHERE players.player_id = shots.goalie_id)
   """)
   goalie_filled = cursor.rowcount

   return {"shooter_filled": shooter_filled, "goalie_filled": goalie_filled}


def _season_range(start_season: int, end_season: int | None) -> list[str]:
   resolved_end = start_season if end_season is None else end_season
   if resolved_end < start_season:
      raise ValueError("end season must be greater than or equal to start season")
   return [str(year) for year in range(start_season, resolved_end + 1)]


def _table_exists(executor: sqlite3.Connection | sqlite3.Cursor, table_name: str) -> bool:
   return (
      executor.execute(
         "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
         (table_name,),
      ).fetchone()
      is not None
   )


def group_prior_event_types(df: "pd.DataFrame", column: str = "prev_event_type") -> "pd.DataFrame":
   """Group raw prior_event_type strings into tactical categories for XGBoost."""
   # Define the mapping from raw event types to tactical categories
   # NHL API typeDescKey values mapped to tactical situations
   event_type_mapping = {
      # Rebound situations - shot attempts that didn't score
      "shot-on-goal": "rebound",
      "missed-shot": "rebound",
      "blocked-shot": "rebound",
      "failed-shot-attempt": "rebound",
      # Rush situations - transition plays, puck movement, chaotic changes
      "takeaway": "rush",
      "giveaway": "rush",
      "puck-bounce": "rush",  # Puck bouncing off boards/rim creates chaotic transition
      # Set play situations - structured, stationary plays
      "faceoff": "set_play",
      "penalty": "set_play",
      "stoppage": "set_play",
      "play-stoppage": "set_play",  # Explicit play stoppage
      "period-start": "set_play",  # Period start/reset
      "period-end": "set_play",  # Period end/reset
      "coach-challenge": "set_play",  # Review stoppage
      "referee-emergency": "set_play",  # Official emergency
      "line-umpire": "set_play",  # Official line change
      "official-throw": "set_play",  # Official throw of puck
      # Locational clash situations - physical contact
      "hit": "locational_clash",
   }

   # Create a copy to avoid modifying the original
   result = df.copy()

   normalized_event_type = result[column].fillna("").astype(str).str.strip().str.lower()
   is_goal_prior_event = normalized_event_type == "goal"

   # Map raw event types case-insensitively. Any unmapped value defaults to set_play,
   # which is a conservative baseline where the goalie is presumed set.
   result["prior_event_grouped"] = normalized_event_type.map(event_type_mapping).fillna("set_play")
   result.loc[is_goal_prior_event, "prior_event_grouped"] = "set_play"

   # Explicit goal override: reset continuous tracking metrics for those rows.
   if "seconds_since_last_event" in result.columns:
      result.loc[is_goal_prior_event, "seconds_since_last_event"] = 0
   if "puck_velocity" in result.columns:
      result.loc[is_goal_prior_event, "puck_velocity"] = 0
   if "crossed_royal_road" in result.columns:
      if pd.api.types.is_bool_dtype(result["crossed_royal_road"]):
         result.loc[is_goal_prior_event, "crossed_royal_road"] = False
      else:
         result.loc[is_goal_prior_event, "crossed_royal_road"] = 0

   # Convert to categorical dtype for XGBoost optimization
   result["prior_event_grouped"] = result["prior_event_grouped"].astype("category")

   return result


def _load_player_seasonal_stats(
   connection: sqlite3.Connection,
   seasons: list[str],
) -> dict[tuple[int, str], dict[str, object]]:
   if not seasons or not _table_exists(connection, "player_seasonal_stats"):
      return {}

   placeholders = ",".join("?" for _ in seasons)
   rows = connection.execute(
      f"""
      SELECT player_id,
             season,
             MAX(player_name) AS player_name,
             SUM(games_played) AS games_played,
             SUM(toi_seconds) AS toi_seconds
      FROM player_seasonal_stats
      WHERE season IN ({placeholders})
      GROUP BY player_id, season
      """,
      seasons,
   ).fetchall()

   stats: dict[tuple[int, str], dict[str, object]] = {}
   for row in rows:
      stats[(int(row[0]), str(row[1]))] = {
         "player_name": str(row[2] or "Unknown"),
         "games_played": int(row[3] or 0),
         "toi_seconds": int(row[4] or 0),
      }
   return stats


def engineer_prior_event_features(df: "pd.DataFrame") -> "pd.DataFrame":
   """Engineer prior event features: seconds_since_last_event, prior_event_type, puck_distance_delta, puck_velocity, crossed_royal_road."""

   # Create a copy to avoid modifying the original
   result = df.copy()

   # Initialize new columns with default values
   result['seconds_since_last_event'] = 0.0
   result['prior_event_type'] = None
   result['puck_distance_delta'] = 0.0
   result['puck_velocity'] = 0.0
   result['crossed_royal_road'] = False

   # Get prior event data from Main.py's already-computed columns
   # These columns are populated by Main.py's parse_web_shot_events function
   # Each row already contains the previous event's info directly
   prior_event_type = result['prev_event_type']
   prior_x = pd.to_numeric(result['prev_event_x'], errors='coerce')
   prior_y = pd.to_numeric(result['prev_event_y'], errors='coerce')
   prior_seconds_ago = pd.to_numeric(result['prev_event_seconds_ago'], errors='coerce')

   # Check if prior event data exists (not null) for edge case safety
   # This prevents using invalid data when there's no prior event
   has_prior_event = prior_x.notna() & prior_y.notna() & (prior_seconds_ago > 0)

   curr_x = pd.to_numeric(result['x'], errors='coerce')
   curr_y = pd.to_numeric(result['y'], errors='coerce')

   # Calculate Euclidean distance between puck positions
   distance_delta = np.sqrt((curr_x - prior_x) ** 2 + (curr_y - prior_y) ** 2)

   # Royal road: crossed center line (y coordinates have different signs)
   crossed_royal_road = (prior_y * curr_y) < 0

   # Apply edge case safety: only use computed values when prior event data exists
   # For rows without prior event data, keep default values (0/null)
   result['seconds_since_last_event'] = np.where(
       has_prior_event,
       prior_seconds_ago,
       0.0
   )

   result['prior_event_type'] = np.where(
       has_prior_event,
       prior_event_type,
       None
   )

   result['puck_distance_delta'] = np.where(
       has_prior_event,
       distance_delta,
       0.0
   )

   # Calculate velocity (avoid division by zero)
   result['puck_velocity'] = np.where(
       has_prior_event,
       distance_delta / prior_seconds_ago,
       0.0
   )

   result['crossed_royal_road'] = has_prior_event & crossed_royal_road

   return result


def _temporal_split_indices(
   game_ids: np.ndarray,
   test_size: float,
) -> tuple[np.ndarray, np.ndarray]:
   """Split indices temporally by game_id ordering (higher game_id = later in time)."""
   if len(game_ids) == 0:
      return np.array([], dtype=int), np.array([], dtype=int)
   sorted_indices = np.argsort(game_ids)
   split_point = max(1, min(int(len(sorted_indices) * (1.0 - test_size)), len(sorted_indices) - 1))
   return sorted_indices[:split_point], sorted_indices[split_point:]


def initialize_metrics_tables(db_path: str) -> None:
   with sqlite3.connect(db_path) as connection:
      cursor = connection.cursor()
      cursor.execute(
         """
         CREATE TABLE IF NOT EXISTS metrics_model_runs (
            model_version TEXT PRIMARY KEY,
            model_type TEXT NOT NULL,
            created_at TEXT NOT NULL,
            train_start_season TEXT NOT NULL,
            train_end_season TEXT NOT NULL,
            score_start_season TEXT NOT NULL,
            score_end_season TEXT NOT NULL,
            row_count INTEGER NOT NULL,
            feature_spec_json TEXT NOT NULL,
            weights_json TEXT NOT NULL,
            bias REAL NOT NULL,
            config_signature TEXT,
            train_signature TEXT,
            validation_auc REAL,
            validation_log_loss REAL,
            validation_brier REAL,
            validation_ece REAL,
            calibration_method TEXT,
            calibration_payload_json TEXT,
            feature_importance_json TEXT
         )
         """
      )
      cursor.execute("PRAGMA table_info(metrics_model_runs)")
      model_run_columns = {row[1] for row in cursor.fetchall()}
      if "config_signature" not in model_run_columns:
         cursor.execute("ALTER TABLE metrics_model_runs ADD COLUMN config_signature TEXT")
      if "train_signature" not in model_run_columns:
         cursor.execute("ALTER TABLE metrics_model_runs ADD COLUMN train_signature TEXT")
      if "validation_auc" not in model_run_columns:
         cursor.execute("ALTER TABLE metrics_model_runs ADD COLUMN validation_auc REAL")
      if "validation_log_loss" not in model_run_columns:
         cursor.execute("ALTER TABLE metrics_model_runs ADD COLUMN validation_log_loss REAL")
      if "validation_brier" not in model_run_columns:
         cursor.execute("ALTER TABLE metrics_model_runs ADD COLUMN validation_brier REAL")
      if "validation_ece" not in model_run_columns:
         cursor.execute("ALTER TABLE metrics_model_runs ADD COLUMN validation_ece REAL")
      if "calibration_method" not in model_run_columns:
         cursor.execute("ALTER TABLE metrics_model_runs ADD COLUMN calibration_method TEXT")
      if "calibration_payload_json" not in model_run_columns:
         cursor.execute("ALTER TABLE metrics_model_runs ADD COLUMN calibration_payload_json TEXT")
      if "feature_importance_json" not in model_run_columns:
         cursor.execute("ALTER TABLE metrics_model_runs ADD COLUMN feature_importance_json TEXT")
      cursor.execute(
         """
         CREATE TABLE IF NOT EXISTS shot_xg (
            event_hash TEXT NOT NULL,
            season TEXT NOT NULL,
            game_id INTEGER NOT NULL,
            model_version TEXT NOT NULL,
            xg REAL NOT NULL,
            PRIMARY KEY (event_hash, model_version)
         )
         """
      )
      cursor.execute("CREATE INDEX IF NOT EXISTS idx_shot_xg_model_season ON shot_xg(model_version, season)")

      cursor.execute(
         """
         CREATE TABLE IF NOT EXISTS player_season_metrics (
            season TEXT NOT NULL,
            model_version TEXT NOT NULL,
            shooter TEXT NOT NULL,
            shooter_id INTEGER NOT NULL,
            team TEXT NOT NULL,
            shots INTEGER NOT NULL,
            goals INTEGER NOT NULL,
            xg REAL NOT NULL,
            goals_above_expected REAL NOT NULL,
            shooting_pct REAL NOT NULL,
            xg_per_shot REAL NOT NULL,
            league_avg_xg_per_shot REAL,
            league_avg_shooting_pct REAL,
            delta_xg_per_shot REAL,
            delta_shooting_pct REAL,
            qualified INTEGER NOT NULL,
            rank_gax INTEGER,
            PRIMARY KEY (season, model_version, shooter, shooter_id, team)
         )
         """
      )

      cursor.execute(
         """
         CREATE TABLE IF NOT EXISTS team_season_metrics (
            season TEXT NOT NULL,
            model_version TEXT NOT NULL,
            team TEXT NOT NULL,
            shots INTEGER NOT NULL,
            goals INTEGER NOT NULL,
            xg REAL NOT NULL,
            goals_above_expected REAL NOT NULL,
            shooting_pct REAL NOT NULL,
            xg_per_shot REAL NOT NULL,
            PRIMARY KEY (season, model_version, team)
         )
         """
      )

      cursor.execute(
         """
         CREATE TABLE IF NOT EXISTS goalie_season_metrics (
            season TEXT NOT NULL,
            model_version TEXT NOT NULL,
            goalie TEXT NOT NULL,
            goalie_id INTEGER NOT NULL,
            shots_against INTEGER NOT NULL,
            goals_against INTEGER NOT NULL,
            xga REAL NOT NULL,
            expected_save_pct REAL NOT NULL,
            save_pct REAL NOT NULL,
            saves_above_average REAL NOT NULL,
            PRIMARY KEY (season, model_version, goalie, goalie_id)
         )
         """
      )

      cursor.execute(
         """
         CREATE TABLE IF NOT EXISTS metrics_season_refresh_state (
            season TEXT PRIMARY KEY,
            config_signature TEXT NOT NULL,
            train_signature TEXT NOT NULL,
            source_row_count INTEGER NOT NULL,
            source_goal_count INTEGER NOT NULL,
            source_max_shot_id INTEGER NOT NULL,
            model_version TEXT NOT NULL,
            updated_at TEXT NOT NULL
         )
         """
      )

      cursor.execute(
         """
         CREATE TABLE IF NOT EXISTS metrics_model_validation_bins (
            model_version TEXT NOT NULL,
            dataset TEXT NOT NULL,
            bin_index INTEGER NOT NULL,
            bin_start REAL NOT NULL,
            bin_end REAL NOT NULL,
            shot_count INTEGER NOT NULL,
            avg_pred REAL NOT NULL,
            goal_rate REAL NOT NULL,
            PRIMARY KEY (model_version, dataset, bin_index)
         )
         """
      )

      cursor.execute(
         """
         CREATE TABLE IF NOT EXISTS player_career_trajectory (
            model_version TEXT NOT NULL,
            shooter TEXT NOT NULL,
            shooter_id INTEGER NOT NULL,
            season TEXT NOT NULL,
            career_shots INTEGER NOT NULL,
            career_goals INTEGER NOT NULL,
            career_xg REAL NOT NULL,
            career_gax REAL NOT NULL,
            career_shooting_pct REAL NOT NULL,
            career_xg_per_shot REAL NOT NULL,
            trailing_3yr_shots INTEGER NOT NULL,
            trailing_3yr_goals INTEGER NOT NULL,
            trailing_3yr_xg REAL NOT NULL,
            trailing_3yr_gax REAL NOT NULL,
            trailing_3yr_shooting_pct REAL NOT NULL,
            trailing_3yr_xg_per_shot REAL NOT NULL,
            this_season_shots INTEGER NOT NULL,
            this_season_goals INTEGER NOT NULL,
            this_season_xg REAL NOT NULL,
            this_season_gax REAL NOT NULL,
            PRIMARY KEY (model_version, shooter_id, season)
         )
         """
      )

      cursor.execute(
         """
         CREATE TABLE IF NOT EXISTS goalie_career_trajectory (
            model_version TEXT NOT NULL,
            goalie TEXT NOT NULL,
            goalie_id INTEGER NOT NULL,
            season TEXT NOT NULL,
            career_shots_against INTEGER NOT NULL,
            career_goals_against INTEGER NOT NULL,
            career_xga REAL NOT NULL,
            career_saves_above_avg REAL NOT NULL,
            career_save_pct REAL NOT NULL,
            trailing_3yr_shots_against INTEGER NOT NULL,
            trailing_3yr_goals_against INTEGER NOT NULL,
            trailing_3yr_xga REAL NOT NULL,
            trailing_3yr_saves_above_avg REAL NOT NULL,
            trailing_3yr_save_pct REAL NOT NULL,
            this_season_shots_against INTEGER NOT NULL,
            this_season_goals_against INTEGER NOT NULL,
            this_season_xga REAL NOT NULL,
            this_season_saves_above_avg REAL NOT NULL,
            PRIMARY KEY (model_version, goalie_id, season)
         )
         """
      )

      # New tables for enhanced metrics
      cursor.execute(
         """
         CREATE TABLE IF NOT EXISTS team_strength_metrics (
            season TEXT NOT NULL,
            team TEXT NOT NULL,
            goals_for INTEGER NOT NULL,
            goals_against INTEGER NOT NULL,
            xg_for REAL NOT NULL,
            xga REAL NOT NULL,
            goalie_save_pct_avg REAL NOT NULL,
            PRIMARY KEY (season, team)
         )
         """
      )

      cursor.execute(
         """
         CREATE TABLE IF NOT EXISTS player_season_advanced_metrics (
            season TEXT NOT NULL,
            model_version TEXT NOT NULL,
            shooter TEXT NOT NULL,
            shooter_id INTEGER NOT NULL,
            team TEXT NOT NULL,
            games INTEGER NOT NULL,
            toi REAL NOT NULL,
            shots INTEGER NOT NULL,
            goals INTEGER NOT NULL,
            xg REAL NOT NULL,
            gax REAL NOT NULL,
            shooting_pct REAL NOT NULL,
            xg_per_shot REAL NOT NULL,
            gax_per_60 REAL NOT NULL,
            xg_per_60 REAL NOT NULL,
            shooting_pct_per_60 REAL NOT NULL,
            league_avg_xg_per_60 REAL,
            league_avg_shooting_pct_per_60 REAL,
            delta_gax_per_60 REAL,
            delta_shooting_pct_per_60 REAL,
            rolling_3yr_gax_per_60 REAL,
            rolling_3yr_xg_per_60 REAL,
            age INTEGER,
            age_adjusted_gax_per_60 REAL,
            age_adjusted_gax REAL,
            PRIMARY KEY (season, model_version, shooter, shooter_id, team)
         )
         """
      )

      cursor.execute(
         """
         CREATE TABLE IF NOT EXISTS player_career_advanced (
            model_version TEXT NOT NULL,
            shooter TEXT NOT NULL,
            shooter_id INTEGER NOT NULL,
            career_age INTEGER NOT NULL,
            career_gax_per_60 REAL NOT NULL,
            career_xg_per_60 REAL NOT NULL,
            career_shooting_pct_per_60 REAL NOT NULL,
            peak_gax_per_60_season TEXT,
            peak_gax_per_60_value REAL,
            trajectory_slope REAL,
            trajectory_r_squared REAL,
            PRIMARY KEY (model_version, shooter_id, career_age)
         )
         """
      )

      # Backward-compatible migration: ensure player_season_advanced_metrics has age column
      cursor.execute("PRAGMA table_info(player_season_advanced_metrics)")
      adv_columns = {row[1] for row in cursor.fetchall()}
      if "age" not in adv_columns:
         cursor.execute("ALTER TABLE player_season_advanced_metrics ADD COLUMN age INTEGER")
      if "age_adjusted_gax" not in adv_columns:
         cursor.execute("ALTER TABLE player_season_advanced_metrics ADD COLUMN age_adjusted_gax REAL")

      connection.commit()


def _config_signature(config: MetricsConfig, train_seasons: list[str]) -> str:
   payload = {
      "feature_spec_version": FEATURE_SPEC_VERSION,
      "train_seasons": train_seasons,
      "learning_rate": config.learning_rate,
      "epochs": config.epochs,
      "l2_regularization": config.l2_regularization,
      "min_shots_for_comparison": config.min_shots_for_comparison,
      "validation_split": config.validation_split,
      "validation_split_strategy": config.validation_split_strategy,
      "calibration_method": config.calibration_method,
      "calibration_bins": config.calibration_bins,
      "random_seed": config.random_seed,
      "xgb_max_depth": config.xgb_max_depth,
      "xgb_subsample": config.xgb_subsample,
      "xgb_colsample_bytree": config.xgb_colsample_bytree,
      "xgb_min_child_weight": config.xgb_min_child_weight,
      "use_player_effects": config.use_player_effects,
      "career_lookback_seasons": config.career_lookback_seasons,
      "compute_rate_metrics": config.compute_rate_metrics,
      "rolling_window_seasons": config.rolling_window_seasons,
      "include_age_adjusted": config.include_age_adjusted,
   }
   encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
   return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _collect_source_fingerprint(connection: sqlite3.Connection, seasons: list[str]) -> dict[str, dict[str, int]]:
   placeholders = ",".join("?" for _ in seasons)
   rows = connection.execute(
      f"""
      SELECT season,
             COUNT(*) AS row_count,
             SUM(CASE WHEN shot_result = 'Goal' THEN 1 ELSE 0 END) AS goal_count,
             COALESCE(MAX(id), 0) AS max_shot_id
      FROM shots
      WHERE season IN ({placeholders})
        AND shot_result IN ('Goal', 'ngshot')
        AND x IS NOT NULL
        AND y IS NOT NULL
        AND COALESCE(is_empty_net, 0) = 0
      GROUP BY season
      """,
      seasons,
   ).fetchall()

   by_season = {
      str(row[0]): {
         "row_count": int(row[1] or 0),
         "goal_count": int(row[2] or 0),
         "max_shot_id": int(row[3] or 0),
      }
      for row in rows
   }
   for season in seasons:
      by_season.setdefault(str(season), {"row_count": 0, "goal_count": 0, "max_shot_id": 0})
   return by_season


def _train_signature(train_fingerprint: dict[str, dict[str, int]]) -> str:
   encoded = json.dumps(train_fingerprint, sort_keys=True, separators=(",", ":"))
   return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _lookup_cached_model(
   connection: sqlite3.Connection,
   config_signature: str,
   train_signature: str,
) -> tuple[str, dict, dict] | None:
   connection.row_factory = sqlite3.Row
   row = connection.execute(
      """
      SELECT model_version, feature_spec_json, weights_json
      FROM metrics_model_runs
      WHERE config_signature = ? AND train_signature = ?
      ORDER BY created_at DESC
      LIMIT 1
      """,
      (config_signature, train_signature),
   ).fetchone()
   if row is None:
      return None

   feature_spec_payload = json.loads(row["feature_spec_json"])
   model_payload = json.loads(row["weights_json"])
   if not isinstance(model_payload, dict) or "model_blob_b64" not in model_payload:
      return None

   # Check feature spec version to detect incompatible cached models
   cached_version = feature_spec_payload.get("feature_spec_version")
   if cached_version != FEATURE_SPEC_VERSION:
      return None

   return str(row["model_version"]), feature_spec_payload, model_payload


def _season_state_row(connection: sqlite3.Connection, season: str) -> sqlite3.Row | None:
   connection.row_factory = sqlite3.Row
   return connection.execute(
      """
      SELECT season, config_signature, train_signature, source_row_count, source_goal_count,
             source_max_shot_id, model_version
      FROM metrics_season_refresh_state
      WHERE season = ?
      """,
      (season,),
   ).fetchone()


def _upsert_season_state(
   cursor: sqlite3.Cursor,
   season: str,
   config_signature: str,
   train_signature: str,
   source_row_count: int,
   source_goal_count: int,
   source_max_shot_id: int,
   model_version: str,
) -> None:
   cursor.execute(
      """
      INSERT INTO metrics_season_refresh_state (
         season, config_signature, train_signature, source_row_count,
         source_goal_count, source_max_shot_id, model_version, updated_at
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
      ON CONFLICT(season) DO UPDATE SET
         config_signature = excluded.config_signature,
         train_signature = excluded.train_signature,
         source_row_count = excluded.source_row_count,
         source_goal_count = excluded.source_goal_count,
         source_max_shot_id = excluded.source_max_shot_id,
         model_version = excluded.model_version,
         updated_at = excluded.updated_at
      """,
      (
         season,
         config_signature,
         train_signature,
         source_row_count,
         source_goal_count,
         source_max_shot_id,
         model_version,
         datetime.now(UTC).isoformat(timespec="seconds"),
      ),
   )


def _derived_distance_and_angle(x_coord: float, y_coord: float) -> tuple[float, float]:
   # Approximate distance/angle to the attacking net using nearest net at +/-89 feet.
   net_x = 89.0
   dx = abs(net_x - abs(x_coord))
   distance = math.hypot(dx, y_coord)
   angle = math.degrees(math.atan2(abs(y_coord), max(dx, 1e-6)))
   return distance, angle


def _calculate_angle_from_coords(x_coord: float, y_coord: float) -> float:
   """Calculate the angle from a point on the ice to the goal at (89, 0).
   
   The angle is measured from the center of the goal (89, 0) to the shot location.
   Uses the standard hockey rink coordinate system where x is width (-100 to 100)
   and y is length (-200 to 200), with the goal at x=89.
   
   Parameters
   ----------
   x_coord : float
       X-coordinate on the ice (width axis, -100 to 100)
   y_coord : float
       Y-coordinate on the ice (length axis, -200 to 200)
   
   Returns
   -------
   float
       Angle in degrees from the goal center to the shot location
   """
   # Goal position is at (89, 0) - center of the goal
   goal_x = 89.0
   goal_y = 0.0
   
   # Calculate angle using atan2 relative to goal position
   dx = x_coord - goal_x
   dy = y_coord - goal_y
   
   # Angle in degrees, using absolute value for symmetry
   angle = math.degrees(math.atan2(abs(dy), abs(dx)))
   return angle


def _calculate_change_in_angle(current_angle: float, prev_angle: float) -> float:
   """Calculate the change in angle between current and previous event."""
   return abs(current_angle - prev_angle)


def _calculate_change_in_distance(current_distance: float, prev_distance: float) -> float:
   """Calculate the change in distance between current and previous event."""
   return current_distance - prev_distance


def _calculate_net_openness(distance: float, angle: float) -> float:
   """Calculate the visible size of the net from a given shot position.

   Uses the law of cosines to calculate the angle subtended by the goal posts
   from the shot location.

   Parameters
   ----------
   distance : float
       Distance from shot location to goal center (feet)
   angle : float
       Angle from goal center to shot location (degrees)

   Returns
   -------
   float
       Visible net angle in degrees (how much of the net is visible)
   """
   # NHL goal width is 6 feet
   goal_width = 6.0

   # Goal posts are at (89, -3) and (89, 3) - 3 feet from center
   post_offset = 3.0

   # Convert angle to radians for calculations
   angle_rad = math.radians(angle)

   if distance < 1.0:
      # Too close to calculate meaningful net openness
      return 90.0  # Maximum visibility

   cos_a = math.cos(angle_rad)
   sin_a = math.sin(angle_rad)

   # Distance to left post (y = -3)
   dist_to_left = math.sqrt((distance * cos_a) ** 2 + (distance * sin_a + post_offset) ** 2)
   # Distance to right post (y = 3)
   dist_to_right = math.sqrt((distance * cos_a) ** 2 + (distance * sin_a - post_offset) ** 2)

   # Law of cosines: θ = arccos((a² + b² - c²) / (2ab))
   a = dist_to_left
   b = dist_to_right
   c = goal_width

   # Clamp the argument to avoid numerical issues
   cos_theta = (a ** 2 + b ** 2 - c ** 2) / (2 * a * b)
   cos_theta = max(-1.0, min(1.0, cos_theta))

   net_openness_rad = math.acos(cos_theta)
   net_openness_deg = math.degrees(net_openness_rad)

   return net_openness_deg


def _safe_float(value: object, default: float = 0.0) -> float:
   try:
      if value is None:
         return default
      return float(value)
   except (TypeError, ValueError):
      return default


def _load_shot_rows_for_features(connection: sqlite3.Connection, seasons: list[str]) -> list[sqlite3.Row]:
   placeholders = ",".join("?" for _ in seasons)
   query = f"""
   SELECT event_hash, season, game_id, shot_result, x, y, shot_type, strength_state,
             score_differential, period, home_away, is_empty_net, shot_distance, shot_angle,
             prev_event_type, prev_event_x, prev_event_y, prev_event_seconds_ago
      FROM shots
      WHERE season IN ({placeholders})
        AND shot_result IN ('Goal', 'ngshot')
        AND x IS NOT NULL
        AND y IS NOT NULL
        AND COALESCE(is_empty_net, 0) = 0
   """
   connection.row_factory = sqlite3.Row
   return list(connection.execute(query, seasons).fetchall())


def _build_feature_spec(rows: list[sqlite3.Row]) -> dict:
   shot_types = sorted({(row["shot_type"] or "Unknown").strip() for row in rows})
   strength_states = sorted({(row["strength_state"] or "Unknown").strip() for row in rows})

   if rows:
      prior_event_df = group_prior_event_types(pd.DataFrame([dict(row) for row in rows]))
      prior_event_groups = sorted(set(prior_event_df["prior_event_grouped"].astype(str).tolist()))
   else:
      prior_event_groups = []

   return {
      "numeric": [
         "shot_distance",
         "shot_angle",
         "score_differential",
         "period",
         "home_away",
         "angle_squared",
         "distance_times_angle",
         "log_shot_distance",
         "trailing_by_two_plus",
         "leading_by_two_plus",
         "seconds_since_last_event",
         "puck_distance_delta",
         "puck_velocity",
         "crossed_royal_road",
         "change_in_angle",
         "change_in_distance",
         "net_openness",
      ],
      "shot_type": shot_types,
      "strength_state": strength_states,
      "prior_event_grouped": prior_event_groups,
   }


def _vectorize_rows(
   rows: list[sqlite3.Row],
   feature_spec: dict,
   fit_scaler: bool,
   scaler: dict | None = None,
   shooter_encoder: LabelEncoder | None = None,
   goalie_encoder: LabelEncoder | None = None,
) -> tuple:
   """Vectorize shot rows into model features and targets."""
   use_entities = shooter_encoder is not None or goalie_encoder is not None
   numeric_feature_names = list(feature_spec["numeric"])
   categorical_feature_names = [
      *(f"shot_type::{value}" for value in feature_spec["shot_type"]),
      *(f"strength_state::{value}" for value in feature_spec["strength_state"]),
      *(f"prior_event_grouped::{value}" for value in feature_spec.get("prior_event_grouped", [])),
   ]
   feature_names = [*numeric_feature_names, *categorical_feature_names]
   if use_entities:
      feature_names.extend(["shooter_id_encoded", "goalie_id_encoded"])

   numeric_width = len(numeric_feature_names)
   categorical_width = len(categorical_feature_names)
   feature_width = numeric_width + categorical_width + (2 if use_entities else 0)

   if not rows:
      empty_features = np.empty((0, feature_width), dtype=np.float64)
      empty_targets = np.empty((0,), dtype=np.float64)
      if fit_scaler:
         scaler = {"means": [0.0] * numeric_width, "stds": [1.0] * numeric_width}
      elif scaler is None:
         raise ValueError("scaler must be provided when fit_scaler is False")
      if use_entities:
         empty_entities = np.empty((0,), dtype=np.int64)
         return empty_features, empty_targets, scaler, feature_names, empty_entities, empty_entities, shooter_encoder, goalie_encoder
      return empty_features, empty_targets, scaler, feature_names

   frame = pd.DataFrame([dict(row) for row in rows])
   frame = engineer_prior_event_features(frame)
   frame = group_prior_event_types(frame, "prior_event_type")

   def _home_away_value(value: object) -> float:
      text = str(value).strip().lower()
      if text in {"home", "h", "1", "true"}:
         return 1.0
      if text in {"away", "a", "0", "false"}:
         return 0.0
      return _safe_float(value, 0.0)

   numeric_rows: list[list[float]] = []
   categorical_rows: list[list[float]] = []
   labels: list[float] = []
   shooter_ids_raw: list[str] = []
   goalie_ids_raw: list[str] = []

   shot_type_to_idx = {value: idx for idx, value in enumerate(feature_spec["shot_type"])}
   strength_to_idx = {value: idx for idx, value in enumerate(feature_spec["strength_state"])}
   prior_event_grouped_to_idx = {value: idx for idx, value in enumerate(feature_spec.get("prior_event_grouped", []))}

   for row in frame.itertuples(index=False):
      dist = _safe_float(getattr(row, "shot_distance", None), default=float("nan"))
      angle = _safe_float(getattr(row, "shot_angle", None), default=float("nan"))
      curr_x = _safe_float(getattr(row, "x", None))
      curr_y = _safe_float(getattr(row, "y", None))
      if math.isnan(dist) or math.isnan(angle):
         dist_fallback, angle_fallback = _derived_distance_and_angle(curr_x, curr_y)
         if math.isnan(dist):
            dist = dist_fallback
         if math.isnan(angle):
            angle = angle_fallback

      prev_x = _safe_float(getattr(row, "prev_event_x", None), None)
      prev_y = _safe_float(getattr(row, "prev_event_y", None), None)
      if prev_x is not None and prev_y is not None:
         prev_angle = _calculate_angle_from_coords(prev_x, prev_y)
         prev_distance = math.hypot(89.0 - abs(prev_x), abs(prev_y))
      else:
         prev_angle = 0.0
         prev_distance = dist

      seconds_since_last_event = _safe_float(getattr(row, "seconds_since_last_event", 0.0), 0.0)
      puck_distance_delta = _safe_float(getattr(row, "puck_distance_delta", 0.0), 0.0)
      puck_velocity = _safe_float(getattr(row, "puck_velocity", 0.0), 0.0)
      crossed_royal_road = 1.0 if bool(getattr(row, "crossed_royal_road", False)) else 0.0
      change_in_angle = _calculate_change_in_angle(angle, prev_angle)
      change_in_distance = _calculate_change_in_distance(dist, prev_distance)
      net_openness = _calculate_net_openness(dist, angle)
      score_differential = _safe_float(getattr(row, "score_differential", 0.0), 0.0)

      numeric_rows.append([
         dist,
         angle,
         score_differential,
         _safe_float(getattr(row, "period", 0.0), 0.0),
         _home_away_value(getattr(row, "home_away", 0.0)),
         angle * angle,
         dist * angle,
         math.log(max(dist, 1e-6)),
         1.0 if score_differential <= -2.0 else 0.0,
         1.0 if score_differential >= 2.0 else 0.0,
         seconds_since_last_event,
         puck_distance_delta,
         puck_velocity,
         crossed_royal_road,
         change_in_angle,
         change_in_distance,
         net_openness,
      ])

      categorical_rows.append([
         *[1.0 if str(getattr(row, "shot_type", "Unknown") or "Unknown").strip() == value else 0.0 for value in feature_spec["shot_type"]],
         *[1.0 if str(getattr(row, "strength_state", "Unknown") or "Unknown").strip() == value else 0.0 for value in feature_spec["strength_state"]],
         *[1.0 if str(getattr(row, "prior_event_grouped", "other") or "other").strip() == value else 0.0 for value in feature_spec.get("prior_event_grouped", [])],
      ])
      labels.append(1.0 if str(getattr(row, "shot_result", "")) == "Goal" else 0.0)
      shooter_ids_raw.append(str(int(getattr(row, "shooter_id"))) if getattr(row, "shooter_id", None) is not None else "UNKNOWN")
      goalie_ids_raw.append(str(int(getattr(row, "goalie_id"))) if getattr(row, "goalie_id", None) is not None else "UNKNOWN")

   numeric_array = np.array(numeric_rows, dtype=np.float64)
   categorical_array = np.array(categorical_rows, dtype=np.float64)

   if fit_scaler:
      means = numeric_array.mean(axis=0)
      stds = numeric_array.std(axis=0)
      stds = np.where(stds < 1e-9, 1.0, stds)
      scaler = {"means": means.tolist(), "stds": stds.tolist()}
   if scaler is None:
      raise ValueError("scaler must be provided when fit_scaler is False")

   means_np = np.array(scaler["means"], dtype=np.float64)
   stds_np = np.array(scaler["stds"], dtype=np.float64)
   standardized_numeric = (numeric_array - means_np) / stds_np

   features = np.concatenate([standardized_numeric, categorical_array], axis=1)
   targets = np.array(labels, dtype=np.float64)

   if use_entities:
      if fit_scaler or shooter_encoder is None or goalie_encoder is None:
         shooter_encoder, goalie_encoder, shooter_encoded, goalie_encoded = _build_entity_maps(rows)
      else:
         try:
            shooter_encoded = shooter_encoder.transform(shooter_ids_raw)
         except ValueError:
            shooter_encoder, _, shooter_encoded, _ = _build_entity_maps(rows)
         try:
            goalie_encoded = goalie_encoder.transform(goalie_ids_raw)
         except ValueError:
            _, goalie_encoder, _, goalie_encoded = _build_entity_maps(rows)

      entity_column = np.column_stack([shooter_encoded.astype(np.float64), goalie_encoded.astype(np.float64)])
      features = np.concatenate([features, entity_column], axis=1)
      return features, targets, scaler, feature_names, shooter_encoded, goalie_encoded, shooter_encoder, goalie_encoder

   return features, targets, scaler, feature_names


def _build_entity_maps(rows: list[sqlite3.Row]) -> tuple[LabelEncoder, LabelEncoder, np.ndarray, np.ndarray]:
   """Build label encoders for shooter_id and goalie_id from training rows."""
   shooter_ids_raw = []
   goalie_ids_raw = []

   for row in rows:
      sid = row["shooter_id"]
      gid = row["goalie_id"]
      shooter_ids_raw.append(str(int(sid)) if sid is not None else "UNKNOWN")
      goalie_ids_raw.append(str(int(gid)) if gid is not None else "UNKNOWN")

   shooter_encoder = LabelEncoder()
   goalie_encoder = LabelEncoder()

   shooter_encoded = shooter_encoder.fit_transform(shooter_ids_raw)
   goalie_encoded = goalie_encoder.fit_transform(goalie_ids_raw)

   return shooter_encoder, goalie_encoder, shooter_encoded, goalie_encoded


def _load_shot_rows_with_entities(connection: sqlite3.Connection, seasons: list[str]) -> list[sqlite3.Row]:
   """Load shot rows including shooter_id and goalie_id for player-adaptive modeling."""
   placeholders = ",".join("?" for _ in seasons)
   query = f"""
   SELECT event_hash, season, game_id, shot_result, x, y, shot_type, strength_state,
             score_differential, period, home_away, is_empty_net, shot_distance, shot_angle,
             shooter_id, goalie_id, prev_event_type, prev_event_x, prev_event_y, prev_event_seconds_ago
      FROM shots
      WHERE season IN ({placeholders})
        AND shot_result IN ('Goal', 'ngshot')
        AND x IS NOT NULL
        AND y IS NOT NULL
        AND COALESCE(is_empty_net, 0) = 0
        AND zone = 'O'
        AND shooter_id IS NOT NULL
        AND goalie_id IS NOT NULL
   """
   connection.row_factory = sqlite3.Row
   return list(connection.execute(query, seasons).fetchall())


def _fit_xgboost_classifier(features: np.ndarray, targets: np.ndarray, config: MetricsConfig) -> XGBClassifier:
   if features.shape[0] == 0:
      raise ValueError("cannot fit model with zero rows")

   model = XGBClassifier(
      objective="binary:logistic",
      eval_metric="logloss",
      n_estimators=config.epochs,
      learning_rate=config.learning_rate,
      max_depth=config.xgb_max_depth,
      min_child_weight=config.xgb_min_child_weight,
      subsample=config.xgb_subsample,
      colsample_bytree=config.xgb_colsample_bytree,
      reg_lambda=config.l2_regularization,
      random_state=config.random_seed,
      n_jobs=1,
   )
   model.fit(features, targets)
   return model


def _fit_sigmoid_calibrator(raw_probs: np.ndarray, targets: np.ndarray) -> LogisticRegression | None:
   if raw_probs.shape[0] < 50:
      return None

   unique_targets = np.unique(targets)
   if unique_targets.shape[0] < 2:
      return None

   clipped = np.clip(raw_probs, 1e-6, 1.0 - 1e-6)
   logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
   calibrator = LogisticRegression(solver="lbfgs")
   calibrator.fit(logits, targets)
   return calibrator


def _apply_sigmoid_calibration(raw_probs: np.ndarray, calibrator: LogisticRegression | None) -> np.ndarray:
   if calibrator is None:
      return np.clip(raw_probs, 0.0, 1.0)

   clipped = np.clip(raw_probs, 1e-6, 1.0 - 1e-6)
   logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
   calibrated = calibrator.predict_proba(logits)[:, 1]
   return np.clip(calibrated, 0.0, 1.0)


def _calibration_bins(predictions: np.ndarray, targets: np.ndarray, bin_count: int) -> tuple[list[dict], float]:
   if predictions.size == 0:
      return [], 0.0

   bins = max(2, int(bin_count))
   buckets: list[list[tuple[float, float]]] = [[] for _ in range(bins)]
   for pred, target in zip(predictions, targets, strict=True):
      idx = min(bins - 1, int(pred * bins))
      buckets[idx].append((float(pred), float(target)))

   total = float(predictions.size)
   rows: list[dict] = []
   ece = 0.0
   for idx, bucket in enumerate(buckets):
      start = idx / bins
      end = (idx + 1) / bins
      if not bucket:
         rows.append(
            {
               "bin_index": idx,
               "bin_start": start,
               "bin_end": end,
               "shot_count": 0,
               "avg_pred": 0.0,
               "goal_rate": 0.0,
            }
         )
         continue

      preds = np.array([pair[0] for pair in bucket], dtype=np.float64)
      goals = np.array([pair[1] for pair in bucket], dtype=np.float64)
      avg_pred = float(preds.mean())
      goal_rate = float(goals.mean())
      count = int(goals.size)
      ece += (count / total) * abs(avg_pred - goal_rate)
      rows.append(
         {
            "bin_index": idx,
            "bin_start": start,
            "bin_end": end,
            "shot_count": count,
            "avg_pred": avg_pred,
            "goal_rate": goal_rate,
         }
      )

   return rows, float(ece)


def _validation_summary(targets: np.ndarray, predictions: np.ndarray, ece: float) -> dict[str, float] | None:
   if targets.size == 0:
      return None
   if np.unique(targets).shape[0] < 2:
      return None

   return {
      "auc": float(roc_auc_score(targets, predictions)),
      "log_loss": float(log_loss(targets, np.clip(predictions, 1e-6, 1.0 - 1e-6))),
      "brier": float(brier_score_loss(targets, predictions)),
      "ece": float(ece),
   }


def _serialize_model_payload(model: XGBClassifier, calibrator: LogisticRegression | None) -> dict:
   payload: dict[str, object] = {
      "model_blob_b64": base64.b64encode(pickle.dumps(model)).decode("ascii"),
      "framework": "xgboost_sklearn",
   }
   if calibrator is not None:
      payload["calibrator_blob_b64"] = base64.b64encode(pickle.dumps(calibrator)).decode("ascii")
   return payload


def _deserialize_model_payload(model_payload: dict) -> tuple[XGBClassifier, LogisticRegression | None]:
   model_blob = base64.b64decode(str(model_payload["model_blob_b64"]))
   model = pickle.loads(model_blob)

   calibrator_blob = model_payload.get("calibrator_blob_b64")
   calibrator: LogisticRegression | None = None
   if calibrator_blob:
      calibrator = pickle.loads(base64.b64decode(str(calibrator_blob)))
   return model, calibrator


def _feature_importance_summary(feature_names: list[str], model: XGBClassifier) -> dict:
   importances = np.array(model.feature_importances_, dtype=np.float64)
   if importances.size != len(feature_names):
      return {"ranked": [], "near_zero": []}

   ranked = [
      {"feature": feature_names[idx], "importance": float(score)}
      for idx, score in enumerate(importances)
   ]
   ranked.sort(key=lambda row: row["importance"], reverse=True)
   near_zero = [row["feature"] for row in ranked if row["importance"] <= 1e-8]
   return {"ranked": ranked, "near_zero": near_zero}


def _persist_validation_bins(cursor: sqlite3.Cursor, model_version: str, dataset: str, bin_rows: list[dict]) -> None:
   cursor.executemany(
      """
      INSERT OR REPLACE INTO metrics_model_validation_bins (
         model_version, dataset, bin_index, bin_start, bin_end,
         shot_count, avg_pred, goal_rate
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
      """,
      [
         (
            model_version,
            dataset,
            int(row["bin_index"]),
            float(row["bin_start"]),
            float(row["bin_end"]),
            int(row["shot_count"]),
            float(row["avg_pred"]),
            float(row["goal_rate"]),
         )
         for row in bin_rows
      ],
   )


def _delete_existing_model_outputs(cursor: sqlite3.Cursor, seasons: list[str]) -> None:
   placeholders = ",".join("?" for _ in seasons)
   cursor.execute(f"DELETE FROM shot_xg WHERE season IN ({placeholders})", seasons)
   cursor.execute(f"DELETE FROM player_season_metrics WHERE season IN ({placeholders})", seasons)
   cursor.execute(f"DELETE FROM team_season_metrics WHERE season IN ({placeholders})", seasons)
   cursor.execute(f"DELETE FROM goalie_season_metrics WHERE season IN ({placeholders})", seasons)
   cursor.execute(f"DELETE FROM player_season_advanced_metrics WHERE season IN ({placeholders})", seasons)
   cursor.execute(f"DELETE FROM team_strength_metrics WHERE season IN ({placeholders})", seasons)
   cursor.execute(f"DELETE FROM player_career_trajectory WHERE season IN ({placeholders})", seasons)
   cursor.execute(f"DELETE FROM goalie_career_trajectory WHERE season IN ({placeholders})", seasons)


def _persist_player_season_metrics(cursor: sqlite3.Cursor, model_version: str, seasons: list[str], min_shots: int) -> int:
   placeholders = ",".join("?" for _ in seasons)
   if _table_exists(cursor, "player_seasonal_stats"):
      query = f"""
         WITH player_stats AS (
            SELECT player_id,
                   season,
                   MAX(player_name) AS player_name,
                   SUM(games_played) AS games_played,
                   SUM(toi_seconds) AS toi_seconds
            FROM player_seasonal_stats
            WHERE season IN ({placeholders})
            GROUP BY player_id, season
         )
         SELECT s.season,
                COALESCE(ps.player_name, s.shooter, 'Unknown') AS shooter,
                s.shooter_id,
                s.team,
                COUNT(*) AS shots,
                SUM(CASE WHEN s.shot_result = 'Goal' THEN 1 ELSE 0 END) AS goals,
                SUM(x.xg) AS xg
         FROM shots s
         JOIN shot_xg x ON x.event_hash = s.event_hash
         LEFT JOIN player_stats ps ON ps.player_id = s.shooter_id AND ps.season = s.season
         WHERE x.model_version = ?
           AND s.season IN ({placeholders})
         GROUP BY s.season, s.shooter_id, s.team
      """
      rows = [dict(row) for row in cursor.execute(query, [*seasons, model_version, *seasons]).fetchall()]
   else:
      query = f"""
         SELECT s.season,
                COALESCE(s.shooter, 'Unknown') AS shooter,
                s.shooter_id,
                s.team,
                COUNT(*) AS shots,
                SUM(CASE WHEN s.shot_result = 'Goal' THEN 1 ELSE 0 END) AS goals,
                SUM(x.xg) AS xg
         FROM shots s
         JOIN shot_xg x ON x.event_hash = s.event_hash
         WHERE x.model_version = ?
           AND s.season IN ({placeholders})
         GROUP BY s.season, s.shooter_id, s.team
      """
      rows = [dict(row) for row in cursor.execute(query, [model_version, *seasons]).fetchall()]
   if not rows:
      return 0

   league_by_season: dict[str, dict[str, float]] = {}
   for season in seasons:
      qualified = [row for row in rows if row["season"] == season and int(row["shots"] or 0) >= min_shots]
      if not qualified:
         league_by_season[season] = {"xg_per_shot": 0.0, "shooting_pct": 0.0}
         continue
      total_xg = sum(float(row["xg"] or 0.0) for row in qualified)
      total_goals = sum(float(row["goals"] or 0.0) for row in qualified)
      total_shots = sum(float(row["shots"] or 0.0) for row in qualified)
      league_by_season[season] = {
         "xg_per_shot": total_xg / total_shots if total_shots else 0.0,
         "shooting_pct": total_goals / total_shots if total_shots else 0.0,
      }

   rank_map: dict[tuple[str, str, object, object], int] = {}
   for season in seasons:
      ranked_rows = [row for row in rows if row["season"] == season and int(row["shots"] or 0) >= min_shots]
      ranked_rows.sort(key=lambda row: float(row["goals"] or 0.0) - float(row["xg"] or 0.0), reverse=True)
      for index, row in enumerate(ranked_rows, start=1):
         rank_map[(row["season"], row["shooter"], row["shooter_id"], row["team"])] = index

   insert_rows = []
   for row in rows:
      shots = int(row["shots"] or 0)
      goals = int(row["goals"] or 0)
      xg = float(row["xg"] or 0.0)
      gax = goals - xg
      shooting_pct = goals / shots if shots else 0.0
      xg_per_shot = xg / shots if shots else 0.0
      season = row["season"]
      qualified = 1 if shots >= min_shots else 0
      league_avg = league_by_season.get(season, {"xg_per_shot": 0.0, "shooting_pct": 0.0})
      rank = rank_map.get((row["season"], row["shooter"], row["shooter_id"], row["team"]))
      insert_rows.append(
         (
            season,
            model_version,
            row["shooter"],
            -1 if row["shooter_id"] is None else int(row["shooter_id"]),
            "" if row["team"] is None else str(row["team"]),
            shots,
            goals,
            xg,
            gax,
            shooting_pct,
            xg_per_shot,
            league_avg["xg_per_shot"],
            league_avg["shooting_pct"],
            xg_per_shot - league_avg["xg_per_shot"],
            shooting_pct - league_avg["shooting_pct"],
            qualified,
            rank,
         )
      )

   cursor.executemany(
      """
      INSERT INTO player_season_metrics (
         season, model_version, shooter, shooter_id, team, shots, goals, xg,
         goals_above_expected, shooting_pct, xg_per_shot, league_avg_xg_per_shot,
         league_avg_shooting_pct, delta_xg_per_shot, delta_shooting_pct, qualified, rank_gax
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
      """,
      insert_rows,
   )
   return len(insert_rows)


def _persist_team_season_metrics(cursor: sqlite3.Cursor, model_version: str, seasons: list[str]) -> int:
   placeholders = ",".join("?" for _ in seasons)
   rows = cursor.execute(
      f"""
      SELECT s.season,
             COALESCE(s.team, 'Unknown') AS team,
             COUNT(*) AS shots,
             SUM(CASE WHEN s.shot_result = 'Goal' THEN 1 ELSE 0 END) AS goals,
             SUM(x.xg) AS xg
      FROM shots s
      JOIN shot_xg x ON x.event_hash = s.event_hash
      WHERE x.model_version = ?
        AND s.season IN ({placeholders})
      GROUP BY s.season, team
      """,
      [model_version, *seasons],
   ).fetchall()

   insert_rows = []
   for row in rows:
      shots = int(row[2] or 0)
      goals = int(row[3] or 0)
      xg = float(row[4] or 0.0)
      insert_rows.append(
         (
            row[0],
            model_version,
            row[1],
            shots,
            goals,
            xg,
            goals - xg,
            goals / shots if shots else 0.0,
            xg / shots if shots else 0.0,
         )
      )

   if insert_rows:
      cursor.executemany(
         """
         INSERT INTO team_season_metrics (
            season, model_version, team, shots, goals, xg, goals_above_expected,
            shooting_pct, xg_per_shot
         ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
         """,
         insert_rows,
      )
   return len(insert_rows)


def _persist_goalie_season_metrics(cursor: sqlite3.Cursor, model_version: str, seasons: list[str]) -> int:
   placeholders = ",".join("?" for _ in seasons)
   if _table_exists(cursor, "player_seasonal_stats"):
      rows = cursor.execute(
         f"""
         WITH player_stats AS (
            SELECT player_id,
                   season,
                   MAX(player_name) AS player_name
            FROM player_seasonal_stats
            WHERE season IN ({placeholders})
            GROUP BY player_id, season
         )
         SELECT s.season,
                COALESCE(ps.player_name, s.goalie, 'Unknown') AS goalie,
                s.goalie_id,
                COUNT(*) AS shots_against,
                SUM(CASE WHEN s.shot_result = 'Goal' THEN 1 ELSE 0 END) AS goals_against,
                SUM(x.xg) AS xga
         FROM shots s
         JOIN shot_xg x ON x.event_hash = s.event_hash
         LEFT JOIN player_stats ps ON ps.player_id = s.goalie_id AND ps.season = s.season
         WHERE x.model_version = ?
           AND s.season IN ({placeholders})
           AND s.goalie_id IS NOT NULL
         GROUP BY s.season, s.goalie_id
         """,
         [*seasons, model_version, *seasons],
      ).fetchall()
   else:
      rows = cursor.execute(
         f"""
         SELECT s.season,
                COALESCE(s.goalie, 'Unknown') AS goalie,
                s.goalie_id,
                COUNT(*) AS shots_against,
                SUM(CASE WHEN s.shot_result = 'Goal' THEN 1 ELSE 0 END) AS goals_against,
                SUM(x.xg) AS xga
         FROM shots s
         JOIN shot_xg x ON x.event_hash = s.event_hash
         WHERE x.model_version = ?
           AND s.season IN ({placeholders})
           AND s.goalie_id IS NOT NULL
         GROUP BY s.season, s.goalie_id
         """,
         [model_version, *seasons],
      ).fetchall()

   insert_rows = []
   for row in rows:
      shots_against = int(row[3] or 0)
      goals_against = int(row[4] or 0)
      xga = float(row[5] or 0.0)
      saves = shots_against - goals_against
      expected_saves = shots_against - xga
      save_pct = saves / shots_against if shots_against else 0.0
      expected_save_pct = expected_saves / shots_against if shots_against else 0.0
      saves_above_average = saves - expected_saves
      insert_rows.append(
         (
            row[0],
            model_version,
            row[1],
            -1 if row[2] is None else int(row[2]),
            shots_against,
            goals_against,
            xga,
            expected_save_pct,
            save_pct,
            saves_above_average,
         )
      )

   if insert_rows:
      cursor.executemany(
         """
         INSERT INTO goalie_season_metrics (
            season, model_version, goalie, goalie_id, shots_against, goals_against,
            xga, expected_save_pct, save_pct, saves_above_average
         ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
         """,
         insert_rows,
      )
   return len(insert_rows)


def _compute_player_career_trajectory(
   connection: sqlite3.Connection,
   cursor: sqlite3.Cursor,
   model_version: str,
   seasons: list[str],
   lookback: int,
) -> int:
   """Compute career and trailing multi-year stats for each shooter."""
   rows = cursor.execute(f"""
      SELECT COALESCE(s.shooter, 'Unknown') AS shooter,
             COALESCE(s.shooter_id, -1) AS shooter_id,
             s.season,
             COUNT(*) AS shots,
             SUM(CASE WHEN s.shot_result = 'Goal' THEN 1 ELSE 0 END) AS goals,
             SUM(x.xg) AS xg
      FROM shots s
      JOIN shot_xg x ON x.event_hash = s.event_hash
      WHERE x.model_version = ?
        AND s.shooter_id IS NOT NULL
        AND s.zone = 'O'
      GROUP BY s.shooter_id, s.season
   """, [model_version]).fetchall()

   # Build a dict: shooter_id -> {season: {shots, goals, xg}}
   player_data: dict[int, dict[str, dict[str, float]]] = {}
   for row in rows:
      sid = int(row[1] or -1)
      if sid == -1:
         continue
      season = str(row[2])
      player_data.setdefault(sid, {})
      player_data[sid][season] = {
         "name": str(row[0] or "Unknown"),
         "shots": float(row[3] or 0),
         "goals": float(row[4] or 0),
         "xg": float(row[5] or 0.0),
      }

   # Get sorted unique seasons across all players
   all_seasons = sorted({str(s) for s_data in player_data.values() for s in s_data})
   season_set = set(seasons)

   insert_rows = []
   for sid, season_map in player_data.items():
      name = next(iter(season_map.values()))["name"]

      for season in all_seasons:
         if season not in season_set:
            continue

         this_season = season_map.get(season, {"shots": 0, "goals": 0, "xg": 0.0})

         # Career totals: all seasons up to and including current
         career_shots = 0.0
         career_goals = 0.0
         career_xg = 0.0
         for s in all_seasons:
            if s <= season:
               entry = season_map.get(s, {"shots": 0, "goals": 0, "xg": 0.0})
               career_shots += entry["shots"]
               career_goals += entry["goals"]
               career_xg += entry["xg"]

         # Trailing lookback: last N seasons including current
         trailing_seasons = [s for s in all_seasons if s <= season][-lookback:]
         trail_shots = 0.0
         trail_goals = 0.0
         trail_xg = 0.0
         for s in trailing_seasons:
            entry = season_map.get(s, {"shots": 0, "goals": 0, "xg": 0.0})
            trail_shots += entry["shots"]
            trail_goals += entry["goals"]
            trail_xg += entry["xg"]

         insert_rows.append((
            model_version,
            name,
            sid,
            season,
            int(career_shots),
            int(career_goals),
            career_xg,
            career_goals - career_xg,
            career_goals / career_shots if career_shots else 0.0,
            career_xg / career_shots if career_shots else 0.0,
            int(trail_shots),
            int(trail_goals),
            trail_xg,
            trail_goals - trail_xg,
            trail_goals / trail_shots if trail_shots else 0.0,
            trail_xg / trail_shots if trail_shots else 0.0,
            int(this_season["shots"]),
            int(this_season["goals"]),
            this_season["xg"],
            this_season["goals"] - this_season["xg"],
         ))

   if insert_rows:
      cursor.executemany(
         """
         INSERT OR REPLACE INTO player_career_trajectory (
            model_version, shooter, shooter_id, season,
            career_shots, career_goals, career_xg, career_gax,
            career_shooting_pct, career_xg_per_shot,
            trailing_3yr_shots, trailing_3yr_goals, trailing_3yr_xg, trailing_3yr_gax,
            trailing_3yr_shooting_pct, trailing_3yr_xg_per_shot,
            this_season_shots, this_season_goals, this_season_xg, this_season_gax
         ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
         """,
         insert_rows,
      )

   return len(insert_rows)


def _compute_goalie_career_trajectory(
   connection: sqlite3.Connection,
   cursor: sqlite3.Cursor,
   model_version: str,
   seasons: list[str],
   lookback: int,
) -> int:
   """Compute career and trailing multi-year stats for each goalie."""
   rows = cursor.execute(f"""
      SELECT COALESCE(s.goalie, 'Unknown') AS goalie,
             COALESCE(s.goalie_id, -1) AS goalie_id,
             s.season,
             COUNT(*) AS shots_against,
             SUM(CASE WHEN s.shot_result = 'Goal' THEN 1 ELSE 0 END) AS goals_against,
             SUM(x.xg) AS xga
      FROM shots s
      JOIN shot_xg x ON x.event_hash = s.event_hash
      WHERE x.model_version = ?
        AND s.goalie_id IS NOT NULL
      GROUP BY s.goalie_id, s.season
   """, [model_version]).fetchall()

   goalie_data: dict[int, dict[str, dict[str, float]]] = {}
   for row in rows:
      gid = int(row[1] or -1)
      if gid == -1:
         continue
      season = str(row[2])
      goalie_data.setdefault(gid, {})
      goalie_data[gid][season] = {
         "name": str(row[0] or "Unknown"),
         "shots_against": float(row[3] or 0),
         "goals_against": float(row[4] or 0),
         "xga": float(row[5] or 0.0),
      }

   all_seasons = sorted({str(s) for g_data in goalie_data.values() for s in g_data})
   season_set = set(seasons)

   insert_rows = []
   for gid, season_map in goalie_data.items():
      name = next(iter(season_map.values()))["name"]

      for season in all_seasons:
         if season not in season_set:
            continue

         this = season_map.get(season, {"shots_against": 0, "goals_against": 0, "xga": 0.0})

         # Career
         career_sa = 0.0
         career_ga = 0.0
         career_xga = 0.0
         for s in all_seasons:
            if s <= season:
               entry = season_map.get(s, {"shots_against": 0, "goals_against": 0, "xga": 0.0})
               career_sa += entry["shots_against"]
               career_ga += entry["goals_against"]
               career_xga += entry["xga"]

         # Trailing lookback
         trailing_seasons = [s for s in all_seasons if s <= season][-lookback:]
         trail_sa = 0.0
         trail_ga = 0.0
         trail_xga = 0.0
         for s in trailing_seasons:
            entry = season_map.get(s, {"shots_against": 0, "goals_against": 0, "xga": 0.0})
            trail_sa += entry["shots_against"]
            trail_ga += entry["goals_against"]
            trail_xga += entry["xga"]

         career_saves = career_sa - career_ga
         career_exp_saves = career_sa - career_xga
         trail_saves = trail_sa - trail_ga
         trail_exp_saves = trail_sa - trail_xga

         insert_rows.append((
            model_version,
            name,
            gid,
            season,
            int(career_sa),
            int(career_ga),
            career_xga,
            career_saves - career_exp_saves,
            career_saves / career_sa if career_sa else 0.0,
            int(trail_sa),
            int(trail_ga),
            trail_xga,
            trail_saves - trail_exp_saves,
            trail_saves / trail_sa if trail_sa else 0.0,
            int(this["shots_against"]),
            int(this["goals_against"]),
            this["xga"],
            (this["shots_against"] - this["goals_against"]) - (this["shots_against"] - this["xga"]),
         ))

   if insert_rows:
      cursor.executemany(
         """
         INSERT OR REPLACE INTO goalie_career_trajectory (
            model_version, goalie, goalie_id, season,
            career_shots_against, career_goals_against, career_xga,
            career_saves_above_avg, career_save_pct,
            trailing_3yr_shots_against, trailing_3yr_goals_against, trailing_3yr_xga,
            trailing_3yr_saves_above_avg, trailing_3yr_save_pct,
            this_season_shots_against, this_season_goals_against, this_season_xga,
            this_season_saves_above_avg
         ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
         """,
         insert_rows,
      )

   return len(insert_rows)


def _compute_team_strength_metrics(
   connection: sqlite3.Connection,
   cursor: sqlite3.Cursor,
   model_version: str,
   seasons: list[str],
) -> int:
   """Compute team-level strength metrics for opponent adjustment."""
   placeholders = ",".join("?" for _ in seasons)
   rows = cursor.execute(f"""
      SELECT s.team,
             s.season,
             COUNT(*) AS shots,
             SUM(CASE WHEN s.shot_result = 'Goal' THEN 1 ELSE 0 END) AS goals,
             SUM(x.xg) AS xg,
             AVG(s.shot_distance) AS avg_shot_distance,
             AVG(s.shot_angle) AS avg_shot_angle
      FROM shots s
      JOIN shot_xg x ON x.event_hash = s.event_hash
      WHERE x.model_version = ?
        AND s.season IN ({placeholders})
        AND s.shot_result IN ('Goal', 'ngshot')
      GROUP BY s.team, s.season
   """, [model_version, *seasons]).fetchall()

   # Get goalie save percentages by team for the season
   goalie_rows = cursor.execute(f"""
      SELECT s.team,
             COUNT(*) AS total_shots_against,
             SUM(CASE WHEN s.shot_result = 'Goal' THEN 1 ELSE 0 END) AS total_goals_against
      FROM shots s
      WHERE s.season IN ({placeholders})
        AND s.goalie_id IS NOT NULL
        AND s.shot_result IN ('Goal', 'ngshot')
      GROUP BY s.team
   """, seasons).fetchall()

   team_goalie_save_pct = {}
   for row in goalie_rows:
      sa = int(row[1] or 0)
      g = int(row[2] or 0)
      team_goalie_save_pct[str(row[0])] = (sa - g) / sa if sa else 0.0

   insert_rows = []
   for row in rows:
      team = str(row[0])
      season = str(row[1])
      shots = int(row[2] or 0)
      goals = int(row[3] or 0)
      xg = float(row[4] or 0.0)
      insert_rows.append((
         season,
         team,
         goals,
         goals,  # goals_against - using same as goals_for for now (would need opponent data)
         xg,
         xg,  # xga - placeholder
         team_goalie_save_pct.get(team, 0.0),
      ))

   if insert_rows:
      cursor.executemany(
         """
         INSERT OR REPLACE INTO team_strength_metrics (
            season, team, goals_for, goals_against, xg_for, xga, goalie_save_pct_avg
         ) VALUES (?, ?, ?, ?, ?, ?, ?)
         """,
         insert_rows,
      )

   return len(insert_rows)


def _compute_player_season_advanced_metrics(
   connection: sqlite3.Connection,
   cursor: sqlite3.Cursor,
   model_version: str,
   seasons: list[str],
   min_shots: int,
   rolling_window: int,
   include_age_adjusted: bool,
) -> int:
   """Compute advanced rate-based metrics for players."""

   saved_factory = connection.row_factory
   connection.row_factory = sqlite3.Row

   player_rows = cursor.execute(f"""
      SELECT season, shooter, shooter_id, team, shots, goals, xg,
             goals_above_expected, shooting_pct, xg_per_shot
      FROM player_season_metrics
      WHERE model_version = ?
        AND season IN ({','.join('?' for _ in seasons)})
   """, [model_version, *seasons]).fetchall()
   season_stats = _load_player_seasonal_stats(connection, seasons)

   game_count_rows = cursor.execute(f"""
      SELECT shooter_id, season, COUNT(DISTINCT game_id) AS games_played
      FROM shots
      WHERE season IN ({','.join('?' for _ in seasons)})
        AND shot_result IN ('Goal', 'ngshot')
        AND shooter_id IS NOT NULL
        AND COALESCE(is_empty_net, 0) = 0
        AND zone = 'O'
      GROUP BY shooter_id, season
   """, seasons).fetchall()

   games_by_player: dict[int, dict[str, int]] = {}
   for row in game_count_rows:
      sid = int(row["shooter_id"])
      games_by_player.setdefault(sid, {})[str(row["season"])] = int(row["games_played"])

   player_birth: dict[int, date] = {}
   age_table_exists = cursor.execute(
      "SELECT name FROM sqlite_master WHERE type='table' AND name='players'"
   ).fetchone() is not None
   if age_table_exists:
      age_rows = cursor.execute("""
         SELECT player_id, birth_date
         FROM players
         WHERE birth_date IS NOT NULL
      """).fetchall()
      for row in age_rows:
         pid = int(row["player_id"])
         bd = str(row["birth_date"]).strip()
         try:
            dt = datetime.strptime(bd, "%Y-%m-%d").date()
         except (ValueError, TypeError):
            continue
         player_birth[pid] = dt

   def _age_at_season_start(birth: date, season: str) -> int:
      """Age on October 1st of the given season year."""
      try:
         season_start = date(int(season), 10, 1)
      except (ValueError, TypeError):
         return 0
      return season_start.year - birth.year - (
         (season_start.month, season_start.day) < (birth.month, birth.day)
      )

   player_season_data: dict[int, dict[str, dict]] = {}
   for row in player_rows:
      sid = int(row["shooter_id"] or -1)
      if sid == -1:
         continue
      season = str(row["season"])
      goals = int(row["goals"] or 0)
      xg = float(row["xg"] or 0.0)
      gax = goals - xg
      stats = season_stats.get((sid, season))
      games = int(stats["games_played"]) if stats else games_by_player.get(sid, {}).get(season, 0)
      shooting_pct = float(row["shooting_pct"] or 0.0)
      xg_per_shot = float(row["xg_per_shot"] or 0.0)
      birth = player_birth.get(sid)
      age = _age_at_season_start(birth, season) if birth else 0
      player_season_data.setdefault(sid, {})[season] = {
         "name": str(stats["player_name"] if stats else row["shooter"]),
         "team": str(row["team"]),
         "shots": int(row["shots"] or 0),
         "goals": goals,
         "xg": xg,
         "gax": gax,
         "shooting_pct": shooting_pct,
         "xg_per_shot": xg_per_shot,
         "games": games,
         "age": age,
         "toi_minutes": (float(stats["toi_seconds"]) / 60.0) if stats else float(games * 60.0),
      }

   all_seasons = sorted({s for sd in player_season_data.values() for s in sd})
   age_xg_curve = _build_age_xg_curve(player_season_data)

   league_avg: dict[str, dict[str, float]] = {}
   for season in seasons:
      season_players = [
         d for sd in player_season_data.values()
         if season in sd for d in [sd[season]]
      ]
      qualified = [d for d in season_players if d["shots"] >= min_shots and d["games"] > 0]
      if not qualified:
         league_avg[season] = {"gax_per_game": 0.0, "xg_per_game": 0.0, "shooting_pct": 0.0}
         continue
      league_avg[season] = {
         "gax_per_game": sum(d["gax"] for d in qualified) / sum(d["games"] for d in qualified),
         "xg_per_game": sum(d["xg"] for d in qualified) / sum(d["games"] for d in qualified),
         "shooting_pct": sum(d["goals"] for d in qualified) / sum(d["shots"] for d in qualified),
      }

   insert_rows: list[tuple] = []

   for sid, season_map in player_season_data.items():
      name = next(iter(season_map.values()))["name"]

      for season in all_seasons:
         if season not in season_map:
            continue

         data = season_map[season]
         shots = data["shots"]
         goals = data["goals"]
         xg = data["xg"]
         gax = data["gax"]
         shooting_pct = data["shooting_pct"]
         xg_per_shot = data["xg_per_shot"]
         games = data["games"]
         age = data["age"]

         toi = float(data["toi_minutes"])

         gax_per_60 = (gax / toi * 60.0) if toi > 0 else 0.0
         xg_per_60 = (xg / toi * 60.0) if toi > 0 else 0.0
         shooting_pct_per_60 = shooting_pct  # already a proportion

         lavg = league_avg.get(season, {"gax_per_game": 0.0, "xg_per_game": 0.0, "shooting_pct": 0.0})
         league_avg_gax_per_60 = lavg["gax_per_game"] * 60.0 / 60.0  # = gax/game; same scale
         league_avg_xg_per_60 = lavg["xg_per_game"] * 60.0 / 60.0
         league_avg_shooting_pct = lavg["shooting_pct"]

         delta_gax_per_60 = gax_per_60 - league_avg_gax_per_60
         delta_shooting_pct = shooting_pct - league_avg_shooting_pct

         season_idx = all_seasons.index(season)
         rolling_seasons_list = all_seasons[max(0, season_idx - rolling_window + 1):season_idx + 1]

         rolling_gax = 0.0
         rolling_xg = 0.0
         rolling_games = 0
         for rs in rolling_seasons_list:
            if rs in season_map:
               rolling_gax += season_map[rs]["gax"]
               rolling_xg += season_map[rs]["xg"]
               rolling_games += season_map[rs]["games"]

         rolling_gax_per_60 = (rolling_gax / (rolling_games * 60.0) * 60.0) if rolling_games > 0 else 0.0
         rolling_xg_per_60 = (rolling_xg / (rolling_games * 60.0) * 60.0) if rolling_games > 0 else 0.0

         if include_age_adjusted and age > 0:
            trend_factor = _player_trend_multiplier(season_map, season)
            age_adjusted_gax = compute_age_adjusted_gax(gax, float(age), age_xg_curve, trend_factor)
            age_adjusted_gax_per_60 = compute_age_adjusted_gax_per_60(gax_per_60, float(age), age_xg_curve, trend_factor)
         else:
            age_adjusted_gax_per_60 = gax_per_60
            age_adjusted_gax = gax

         insert_rows.append((
            season,
            model_version,
            name,
            sid,
            data["team"],
            games,
            toi,
            shots,
            goals,
            xg,
            gax,
            shooting_pct,
            xg_per_shot,
            gax_per_60,
            xg_per_60,
            shooting_pct_per_60,
            league_avg_xg_per_60,
            league_avg_shooting_pct,
            delta_gax_per_60,
            delta_shooting_pct,
            rolling_gax_per_60,
            rolling_xg_per_60,
            age,
            age_adjusted_gax_per_60,
            age_adjusted_gax,
         ))

   if insert_rows:
      cursor.executemany(
         """
         INSERT OR REPLACE INTO player_season_advanced_metrics (
            season, model_version, shooter, shooter_id, team, games, toi,
            shots, goals, xg, gax, shooting_pct, xg_per_shot,
            gax_per_60, xg_per_60, shooting_pct_per_60,
            league_avg_xg_per_60, league_avg_shooting_pct_per_60,
            delta_gax_per_60, delta_shooting_pct_per_60,
            rolling_3yr_gax_per_60, rolling_3yr_xg_per_60,
            age, age_adjusted_gax_per_60, age_adjusted_gax
         ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
         """,
         insert_rows,
      )

   connection.row_factory = saved_factory
   return len(insert_rows)


def _compute_player_career_advanced(
   connection: sqlite3.Connection,
   cursor: sqlite3.Cursor,
   model_version: str,
   seasons: list[str],
) -> int:
   """Compute career-level advanced metrics with trajectory analysis."""

   saved_factory = connection.row_factory
   connection.row_factory = sqlite3.Row

   # Get all player season data (using named columns to avoid index bugs)
   player_rows = cursor.execute(f"""
      SELECT shooter, shooter_id, season, shots, goals, xg, goals_above_expected, shooting_pct, xg_per_shot
      FROM player_season_metrics
      WHERE model_version = ?
        AND season IN ({','.join('?' for _ in seasons)})
   """, [model_version, *seasons]).fetchall()
   season_stats = _load_player_seasonal_stats(connection, seasons)

   # Get game counts per shooter per season
   game_count_rows = cursor.execute(f"""
      SELECT shooter_id, season, COUNT(DISTINCT game_id) AS games_played
      FROM shots
      WHERE season IN ({','.join('?' for _ in seasons)})
        AND shot_result IN ('Goal', 'ngshot')
        AND shooter_id IS NOT NULL
        AND COALESCE(is_empty_net, 0) = 0
      GROUP BY shooter_id, season
   """, seasons).fetchall()

   games_by_player: dict[int, dict[str, int]] = {}
   for row in game_count_rows:
      sid = int(row["shooter_id"])
      games_by_player.setdefault(sid, {})[str(row["season"])] = int(row["games_played"])

   # Get player ages
   player_birth: dict[int, date] = {}
   age_table_exists = cursor.execute(
      "SELECT name FROM sqlite_master WHERE type='table' AND name='players'"
   ).fetchone() is not None
   if age_table_exists:
      age_rows = cursor.execute("""
         SELECT player_id, birth_date
         FROM players
         WHERE birth_date IS NOT NULL
      """).fetchall()
      for row in age_rows:
         pid = int(row["player_id"])
         bd = str(row["birth_date"]).strip()
         try:
            dt = datetime.strptime(bd, "%Y-%m-%d").date()
         except (ValueError, TypeError):
            continue
         player_birth[pid] = dt

   def _age_at_season_start(birth: date, season: str) -> int:
      try:
         season_start = date(int(season), 10, 1)
      except (ValueError, TypeError):
         return 0
      return season_start.year - birth.year - (
         (season_start.month, season_start.day) < (birth.month, birth.day)
      )

   # Build player data using named columns
   player_data: dict[int, dict[str, dict]] = {}
   for row in player_rows:
      sid = int(row["shooter_id"] or -1)
      if sid == -1:
         continue
      season = str(row["season"])
      goals = int(row["goals"] or 0)
      xg = float(row["xg"] or 0.0)
      gax = goals - xg
      games = games_by_player.get(sid, {}).get(season, 0)
      stats = season_stats.get((sid, season))
      games = int(stats["games_played"]) if stats else games
      birth = player_birth.get(sid)
      age = _age_at_season_start(birth, season) if birth else 0
      player_data.setdefault(sid, {})[season] = {
         "name": str(stats["player_name"] if stats else row["shooter"]),
         "shots": int(row["shots"] or 0),
         "goals": goals,
         "xg": xg,
         "gax": gax,
         "shooting_pct": float(row["shooting_pct"] or 0.0),
         "xg_per_shot": float(row["xg_per_shot"] or 0.0),
         "games": games,
         "age": age,
         "toi_minutes": (float(stats["toi_seconds"]) / 60.0) if stats else float(games * 60.0),
      }

   all_seasons = sorted({s for sd in player_data.values() for s in sd})

   insert_rows = []
   for sid, season_map in player_data.items():
      name = next(iter(season_map.values()))["name"]

      # Career age: use the most recent season's age
      latest_season = max(season_map.keys())
      career_age = season_map[latest_season]["age"]

      # Calculate career totals
      career_shots = sum(s["shots"] for s in season_map.values())
      career_gax = sum(s["gax"] for s in season_map.values())
      career_xg = sum(s["xg"] for s in season_map.values())
      career_goals = sum(s["goals"] for s in season_map.values())
      career_games = sum(s["games"] for s in season_map.values())

      # Rate metrics using actual game counts
      toi = sum(float(s["toi_minutes"]) for s in season_map.values())
      career_gax_per_60 = (career_gax / toi * 60.0) if toi > 0 else 0.0
      career_xg_per_60 = (career_xg / toi * 60.0) if toi > 0 else 0.0
      career_shooting_pct = career_goals / career_shots if career_shots > 0 else 0.0

      # Find peak season
      peak_season = None
      peak_gax = float('-inf')
      for season, data in season_map.items():
         if data["gax"] > peak_gax:
            peak_gax = data["gax"]
            peak_season = season

      # Calculate trajectory slope (linear regression on GAX over time)
      if len(season_map) >= 3:
         seasons_list = sorted(season_map.keys())
         gax_values = [season_map[s]["gax"] for s in seasons_list]
         x = np.arange(len(seasons_list))
         if len(x) > 1 and np.std(gax_values) > 0:
            slope = np.polyfit(x, gax_values, 1)[0]
            y_pred = np.polyval([slope, np.mean(gax_values)], x)
            ss_res = np.sum((np.array(gax_values) - y_pred) ** 2)
            ss_tot = np.sum((np.array(gax_values) - np.mean(gax_values)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
         else:
            slope = 0.0
            r_squared = 0.0
      else:
         slope = 0.0
         r_squared = 0.0

      insert_rows.append((
         model_version,
         name,
         sid,
         career_age,
         career_gax_per_60,
         career_xg_per_60,
         career_shooting_pct,
         peak_season,
         peak_gax,
         slope,
         r_squared,
      ))

   if insert_rows:
      cursor.executemany(
         """
         INSERT OR REPLACE INTO player_career_advanced (
            model_version, shooter, shooter_id, career_age,
            career_gax_per_60, career_xg_per_60, career_shooting_pct_per_60,
            peak_gax_per_60_season, peak_gax_per_60_value,
            trajectory_slope, trajectory_r_squared
         ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
         """,
         insert_rows,
      )

   connection.row_factory = saved_factory
   return len(insert_rows)


def run_metrics_refresh(config: MetricsConfig) -> dict:
   initialize_metrics_tables(config.db_path)

   if config.calibration_method != "sigmoid":
      raise ValueError("only sigmoid calibration is currently supported")

   score_seasons = _season_range(config.score_start_season, config.score_end_season)
   train_start = config.score_start_season if config.train_start_season is None else config.train_start_season
   train_end = (config.score_end_season if config.score_end_season is not None else config.score_start_season) if config.train_end_season is None else config.train_end_season
   train_seasons = _season_range(train_start, train_end)

   with sqlite3.connect(config.db_path) as connection:
      connection.row_factory = sqlite3.Row

      # Backfill missing shooter/goalie names from the players table before
      # computing any metrics, so downstream tables get proper player names.
      backfill_result = _backfill_player_names(connection)
      if backfill_result["shooter_filled"] > 0 or backfill_result["goalie_filled"] > 0:
         logging.info(
            "Backfilled player names: %d shooters, %d goalies",
            backfill_result["shooter_filled"],
            backfill_result["goalie_filled"],
         )
      connection.commit()
      score_fingerprint = _collect_source_fingerprint(connection, score_seasons)
      train_fingerprint = _collect_source_fingerprint(connection, train_seasons)

      config_signature = _config_signature(config, train_seasons)
      train_signature = _train_signature(train_fingerprint)

      cached_model = None if config.force_refresh else _lookup_cached_model(connection, config_signature, train_signature)
      trained_new_model = False
      feature_importance: dict[str, object] = {"ranked": [], "near_zero": []}
      validation_metrics: dict[str, float] | None = None

      # Player-adaptive model uses shooter_id and goalie_id as features.
      # Falls back to base (situation-only) model when entity IDs are sparse.
      use_player_features = config.use_player_effects and config.score_start_season >= 2010

      if cached_model is None:
         # Try player-adaptive data first; fall back to situation-only data
         if use_player_features:
            training_rows = _load_shot_rows_with_entities(connection, train_seasons)
            if len(training_rows) < 200:
               use_player_features = False
               training_rows = _load_shot_rows_for_features(connection, train_seasons)
         else:
            training_rows = _load_shot_rows_for_features(connection, train_seasons)

         if len(training_rows) < 200:
            raise ValueError("not enough training rows to fit xG model; scrape more games/seasons first")

         feature_spec = _build_feature_spec(training_rows)

         if use_player_features:
            training_features, training_targets, scaler, feature_names, _, _, shooter_encoder, goalie_encoder = _vectorize_rows(
               training_rows, feature_spec, fit_scaler=True,
               shooter_encoder=LabelEncoder(), goalie_encoder=LabelEncoder(),
            )
         else:
            training_features, training_targets, scaler, feature_names = _vectorize_rows(training_rows, feature_spec, fit_scaler=True)

         validation_split = min(max(float(config.validation_split), 0.0), 0.4)
         can_split = (
            validation_split > 0.0
            and training_features.shape[0] >= 200
            and np.unique(training_targets).shape[0] >= 2
         )
         if can_split:
            if config.validation_split_strategy == "temporal":
               # Extract game_ids from training rows for temporal split
               game_ids = np.array([row["game_id"] for row in training_rows], dtype=np.int64)
               train_idx, valid_idx = _temporal_split_indices(game_ids, validation_split)
               x_train, x_valid = training_features[train_idx], training_features[valid_idx]
               y_train, y_valid = training_targets[train_idx], training_targets[valid_idx]
            else:
               x_train, x_valid, y_train, y_valid = train_test_split(
                  training_features,
                  training_targets,
                  test_size=validation_split,
                  random_state=config.random_seed,
                  stratify=training_targets,
               )
         else:
            x_train, y_train = training_features, training_targets
            x_valid = np.empty((0, training_features.shape[1]), dtype=np.float64)
            y_valid = np.empty((0,), dtype=np.float64)

         model = _fit_xgboost_classifier(x_train, y_train, config)
         calibrator: LogisticRegression | None = None
         validation_bin_rows: list[dict] = []

         if x_valid.shape[0] > 0:
            valid_raw_probs = model.predict_proba(x_valid)[:, 1]
            calibrator = _fit_sigmoid_calibrator(valid_raw_probs, y_valid)
            valid_probs = _apply_sigmoid_calibration(valid_raw_probs, calibrator)
            validation_bin_rows, ece = _calibration_bins(valid_probs, y_valid, config.calibration_bins)
            validation_metrics = _validation_summary(y_valid, valid_probs, ece)

         feature_importance = _feature_importance_summary(feature_names, model)

         model_slug = "xgb_player" if use_player_features else "xgb_base"
         model_type_name = "xgboost_player_adaptive_v1" if use_player_features else "xgboost_calibrated_v1"
         model_version = f"xg_{model_slug}_v1_{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}"
         model_payload = _serialize_model_payload(model, calibrator)

         feature_spec_payload = {
            "feature_spec_version": FEATURE_SPEC_VERSION,
            "feature_spec": feature_spec,
            "scaler": scaler,
            "feature_names": feature_names,
            "use_player_features": use_player_features,
         }
         if use_player_features:
            feature_spec_payload["shooter_classes"] = shooter_encoder.classes_.tolist()
            feature_spec_payload["goalie_classes"] = goalie_encoder.classes_.tolist()

         cursor = connection.cursor()
         cursor.execute(
            """
            INSERT INTO metrics_model_runs (
               model_version, model_type, created_at, train_start_season, train_end_season,
               score_start_season, score_end_season, row_count, feature_spec_json,
               weights_json, bias, config_signature, train_signature,
               validation_auc, validation_log_loss, validation_brier, validation_ece,
               calibration_method, calibration_payload_json, feature_importance_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
               model_version,
               model_type_name,
               datetime.now(UTC).isoformat(timespec="seconds"),
               train_seasons[0],
               train_seasons[-1],
               score_seasons[0],
               score_seasons[-1],
               len(training_rows),
               json.dumps(feature_spec_payload),
               json.dumps(model_payload),
               0.0,
               config_signature,
               train_signature,
               None if validation_metrics is None else validation_metrics["auc"],
               None if validation_metrics is None else validation_metrics["log_loss"],
               None if validation_metrics is None else validation_metrics["brier"],
               None if validation_metrics is None else validation_metrics["ece"],
               config.calibration_method,
               json.dumps({"method": config.calibration_method}),
               json.dumps(feature_importance),
            ),
         )
         if validation_bin_rows:
            _persist_validation_bins(cursor, model_version, "validation", validation_bin_rows)
         connection.commit()
         trained_new_model = True
      else:
         model_version, feature_spec_payload, model_payload = cached_model
         model, calibrator = _deserialize_model_payload(model_payload)
         feature_spec_payload.setdefault("use_player_features", False)
         use_player_features = bool(feature_spec_payload["use_player_features"])
         if use_player_features:
            training_rows = _load_shot_rows_with_entities(connection, train_seasons)
         else:
            training_rows = _load_shot_rows_for_features(connection, train_seasons)
         row = connection.execute(
            """
            SELECT validation_auc, validation_log_loss, validation_brier, validation_ece, feature_importance_json
            FROM metrics_model_runs
            WHERE model_version = ?
            """,
            (model_version,),
         ).fetchone()
         if row is not None and row[0] is not None:
            validation_metrics = {
               "auc": float(row[0]),
               "log_loss": float(row[1]),
               "brier": float(row[2]),
               "ece": float(row[3]),
            }
         if row is not None and row[4]:
            feature_importance = json.loads(row[4])

      feature_spec = feature_spec_payload["feature_spec"]
      scaler = feature_spec_payload["scaler"]
      if cached_model is None:
         model, calibrator = _deserialize_model_payload(model_payload)

      dirty_seasons: list[str] = []
      skipped_seasons: list[str] = []
      for season in score_seasons:
         if config.force_refresh:
            dirty_seasons.append(season)
            continue
         state = _season_state_row(connection, season)
         fingerprint = score_fingerprint[season]
         if state is None:
            dirty_seasons.append(season)
            continue

         unchanged_source = (
            int(state["source_row_count"]) == int(fingerprint["row_count"])
            and int(state["source_goal_count"]) == int(fingerprint["goal_count"])
            and int(state["source_max_shot_id"]) == int(fingerprint["max_shot_id"])
         )
         same_signatures = (
            str(state["config_signature"]) == config_signature
            and str(state["train_signature"]) == train_signature
            and str(state["model_version"]) == model_version
         )
         if unchanged_source and same_signatures:
            skipped_seasons.append(season)
         else:
            dirty_seasons.append(season)

      if not dirty_seasons:
         return {
            "model_version": model_version,
            "model_type": "player_adaptive" if use_player_features else "base_situation",
            "training_rows": len(training_rows),
            "scored_shots": 0,
            "player_season_rows": 0,
            "team_season_rows": 0,
            "goalie_season_rows": 0,
            "player_career_rows": 0,
            "goalie_career_rows": 0,
            "team_strength_rows": 0,
            "advanced_metrics_rows": 0,
            "career_advanced_rows": 0,
            "scored_seasons": [],
            "trained_seasons": train_seasons,
            "skipped_seasons": skipped_seasons,
            "trained_new_model": trained_new_model,
            "validation": validation_metrics,
            "calibration_method": config.calibration_method,
            "feature_pruning_candidates": feature_importance.get("near_zero", []),
         }

      # Score dirty seasons with appropriate vectorizer
      if use_player_features:
         scoring_rows = _load_shot_rows_with_entities(connection, dirty_seasons)
         shooter_encoder = LabelEncoder()
         goalie_encoder = LabelEncoder()
         shooter_classes = feature_spec_payload.get("shooter_classes", [])
         goalie_classes = feature_spec_payload.get("goalie_classes", [])
         if shooter_classes:
            shooter_encoder.classes_ = np.array(shooter_classes)
         if goalie_classes:
            goalie_encoder.classes_ = np.array(goalie_classes)
         scoring_features, _, _, _, _, _, _, _ = _vectorize_rows(
            scoring_rows, feature_spec, fit_scaler=False, scaler=scaler,
            shooter_encoder=shooter_encoder, goalie_encoder=goalie_encoder,
         )
      else:
         scoring_rows = _load_shot_rows_for_features(connection, dirty_seasons)
         scoring_features, _, _, _ = _vectorize_rows(scoring_rows, feature_spec, fit_scaler=False, scaler=scaler)

      raw_probs = model.predict_proba(scoring_features)[:, 1]
      xg_values = _apply_sigmoid_calibration(raw_probs, calibrator)

      cursor = connection.cursor()
      _delete_existing_model_outputs(cursor, dirty_seasons)

      shot_rows = [
         (
            row["event_hash"],
            row["season"],
            row["game_id"],
            model_version,
            float(xg),
         )
         for row, xg in zip(scoring_rows, xg_values, strict=True)
      ]
      cursor.executemany(
         """
         INSERT INTO shot_xg (event_hash, season, game_id, model_version, xg)
         VALUES (?, ?, ?, ?, ?)
         """,
         shot_rows,
      )

      player_count = _persist_player_season_metrics(cursor, model_version, dirty_seasons, config.min_shots_for_comparison)
      team_count = _persist_team_season_metrics(cursor, model_version, dirty_seasons)
      goalie_count = _persist_goalie_season_metrics(cursor, model_version, dirty_seasons)

      # Compute team strength metrics for opponent adjustment
      team_strength_count = _compute_team_strength_metrics(connection, cursor, model_version, dirty_seasons)

      # Compute advanced rate-based metrics
      if config.compute_rate_metrics:
         advanced_count = _compute_player_season_advanced_metrics(
            connection, cursor, model_version, dirty_seasons,
            config.min_shots_for_comparison, config.rolling_window_seasons,
            config.include_age_adjusted
         )
      else:
         advanced_count = 0

      # Compute career trajectory tables
      career_lookback = max(1, int(config.career_lookback_seasons))
      player_career_count = _compute_player_career_trajectory(connection, cursor, model_version, dirty_seasons, career_lookback)
      goalie_career_count = _compute_goalie_career_trajectory(connection, cursor, model_version, dirty_seasons, career_lookback)

      # Compute career advanced metrics
      if config.include_age_adjusted:
         career_advanced_count = _compute_player_career_advanced(connection, cursor, model_version, dirty_seasons)
      else:
         career_advanced_count = 0

      for season in dirty_seasons:
         fingerprint = score_fingerprint[season]
         _upsert_season_state(
            cursor,
            season,
            config_signature,
            train_signature,
            int(fingerprint["row_count"]),
            int(fingerprint["goal_count"]),
            int(fingerprint["max_shot_id"]),
            model_version,
         )

      connection.commit()

   summary = {
      "model_version": model_version,
      "model_type": "player_adaptive" if use_player_features else "base_situation",
      "training_rows": len(training_rows),
      "scored_shots": len(scoring_rows),
      "player_season_rows": player_count,
      "team_season_rows": team_count,
      "goalie_season_rows": goalie_count,
      "player_career_rows": player_career_count,
      "goalie_career_rows": goalie_career_count,
      "team_strength_rows": team_strength_count,
      "advanced_metrics_rows": advanced_count,
      "career_advanced_rows": career_advanced_count,
      "scored_seasons": dirty_seasons,
      "trained_seasons": train_seasons,
      "skipped_seasons": skipped_seasons,
      "trained_new_model": trained_new_model,
      "validation": validation_metrics,
      "calibration_method": config.calibration_method,
      "top_features": feature_importance.get("ranked", [])[:10],
      "feature_pruning_candidates": feature_importance.get("near_zero", []),
   }
   return summary


def parse_args() -> argparse.Namespace:
   parser = argparse.ArgumentParser(description="Build xG/xSV derived metrics from scraped shot data.")
   parser.add_argument("--db-path", default="hockey_data.db", help="SQLite database path from Main.py scrape output.")
   parser.add_argument("--season", type=int, required=True, help="Start season to score (for example 2024 for 2024-25).")
   parser.add_argument("--end-season", type=int, default=None, help="Optional end season to score.")
   parser.add_argument("--train-start-season", type=int, default=None, help="Optional start season for model training window.")
   parser.add_argument("--train-end-season", type=int, default=None, help="Optional end season for model training window.")
   parser.add_argument("--min-shots", type=int, default=50, help="Minimum shots for league-comparison ranking in player metrics.")
   parser.add_argument("--learning-rate", type=float, default=0.05, help="Learning rate for XGBoost trees.")
   parser.add_argument("--epochs", type=int, default=400, help="Number of boosting rounds (n_estimators).")
   parser.add_argument("--l2", type=float, default=0.0005, help="L2 regularization strength (XGBoost reg_lambda).")
   parser.add_argument("--validation-split", type=float, default=0.2, help="Validation split fraction used for monitoring and calibration.")
   parser.add_argument("--validation-split-strategy", default="random", choices=["random", "temporal"], help="Validation split strategy: random or temporal (by game date).")
   parser.add_argument("--calibration-method", default="sigmoid", choices=["sigmoid"], help="Probability calibration method.")
   parser.add_argument("--calibration-bins", type=int, default=10, help="Number of bins for calibration monitoring.")
   parser.add_argument("--random-seed", type=int, default=42, help="Random seed for train/validation split and model training.")
   parser.add_argument("--xgb-max-depth", type=int, default=4, help="Maximum tree depth for XGBoost.")
   parser.add_argument("--xgb-subsample", type=float, default=0.9, help="Row subsample ratio for XGBoost.")
   parser.add_argument("--xgb-colsample-bytree", type=float, default=0.9, help="Feature subsample ratio per tree for XGBoost.")
   parser.add_argument("--xgb-min-child-weight", type=float, default=1.0, help="Minimum child weight for XGBoost splits.")
   parser.add_argument("--no-player-effects", action="store_true", help="Disable shooter_id and goalie_id features (situation-only model).")
   parser.add_argument("--career-lookback", type=int, default=3, help="Number of trailing seasons for career trajectory tracking.")
   parser.add_argument("--rolling-window", type=int, default=3, help="Number of seasons for rolling benchmark metrics.")
   parser.add_argument("--no-rate-metrics", action="store_true", help="Disable rate-based metrics (per-60).")
   parser.add_argument("--no-age-adjusted", action="store_true", help="Disable age-adjusted career trajectory metrics.")
   parser.add_argument("--force", action="store_true", help="Skip staleness check; force re-score and replace all output tables for the target seasons.")
   parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
   return parser.parse_args()


def main() -> None:
   args = parse_args()
   logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")

   config = MetricsConfig(
      db_path=args.db_path,
      score_start_season=args.season,
      score_end_season=args.end_season,
      train_start_season=args.train_start_season,
      train_end_season=args.train_end_season,
      min_shots_for_comparison=args.min_shots,
      learning_rate=args.learning_rate,
      epochs=args.epochs,
      l2_regularization=args.l2,
      validation_split=args.validation_split,
      calibration_method=args.calibration_method,
      calibration_bins=args.calibration_bins,
      random_seed=args.random_seed,
      xgb_max_depth=args.xgb_max_depth,
      xgb_subsample=args.xgb_subsample,
      xgb_colsample_bytree=args.xgb_colsample_bytree,
      xgb_min_child_weight=args.xgb_min_child_weight,
      use_player_effects=not args.no_player_effects,
      career_lookback_seasons=args.career_lookback,
      validation_split_strategy=args.validation_split_strategy,
      compute_rate_metrics=not args.no_rate_metrics,
      rolling_window_seasons=args.rolling_window,
      include_age_adjusted=not args.no_age_adjusted,
      force_refresh=args.force,
   )

   summary = run_metrics_refresh(config)
   logging.info("Metrics refresh complete: %s", json.dumps(summary))


if __name__ == "__main__":
   main()
