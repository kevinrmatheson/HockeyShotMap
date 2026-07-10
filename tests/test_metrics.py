import sqlite3
import tempfile
import unittest
import uuid
from pathlib import Path

import pandas as pd

import Main
import Metrics


class TestMetricsPipeline(unittest.TestCase):
   def test_metrics_refresh_uses_player_season_stats_for_names_and_games(self):
      db_path = str(Path(tempfile.gettempdir()) / f"hockeyshotmap_metrics_stats_{uuid.uuid4().hex}.db")
      Main.initialize_database(db_path)

      rows = []
      for game_id in range(1, 11):
         shooter_id = 101 if game_id <= 5 else 202
         shooter_name = "Real Player A" if shooter_id == 101 else "Real Player B"
         goalie_id = 301
         goalie_name = "Real Goalie"
         for shot_index in range(1, 26):
            rows.append(
               {
                  "Shot": "Goal" if shot_index == 10 else "ngshot",
                  "X": 60.0 + shot_index,
                  "Y": -20.0 + shot_index,
                  "Shot_Type": "Wrist",
                  "Shooter": shooter_name,
                  "Shooter_ID": shooter_id,
                  "Team": "TOR",
                  "Home_Away": 1,
                  "Period": 1,
                  "Period_Time": "10:00",
                  "Period_Time_Remaining": "10:00",
                  "Year": "2024",
                  "GameID": game_id,
                  "API_Source": "web",
                  "Goalie": goalie_name,
                  "Goalie_ID": goalie_id,
                  "Shot_Distance": 25.0,
                  "Shot_Angle": 15.0,
                  "Is_Empty_Net": 0,
                  "Strength_State": "5v5",
                  "Score_Differential": 0,
                  "Zone": "OZ",
                  "Event_ID": (game_id * 1000) + shot_index,
               }
            )

      Main.persist_rows(db_path, rows)

      with sqlite3.connect(db_path) as connection:
         connection.execute(
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
               101, "Real Player A", "2024", "02", "TOR", "C", 10, 12345, 0, 0, 0, 10,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0, 0.0, 0,
            ),
         )
         connection.execute(
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
               202, "Real Player B", "2024", "02", "TOR", "W", 10, 12345, 0, 0, 0, 10,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0, 0.0, 0,
            ),
         )
         connection.execute(
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
               301, "Real Goalie", "2024", "02", "TOR", "G", 10, 18000, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.915, 2.15, 1,
            ),
         )

      summary = Metrics.run_metrics_refresh(
         Metrics.MetricsConfig(
            db_path=db_path,
            score_start_season=2024,
            min_shots_for_comparison=1,
            validation_split=0.2,
         )
      )

      self.assertEqual(summary["scored_seasons"], ["2024"])

      with sqlite3.connect(db_path) as connection:
         row = connection.execute(
            """
            SELECT shooter, games, toi
            FROM player_season_advanced_metrics
            WHERE shooter_id = 101
            ORDER BY season DESC
            LIMIT 1
            """
         ).fetchone()

      self.assertIsNotNone(row)
      self.assertEqual(row[0], "Real Player A")
      self.assertEqual(row[1], 10)
      self.assertGreater(row[2], 0)

   def test_metrics_refresh_creates_derived_outputs(self):
      db_path = str(Path(tempfile.gettempdir()) / f"hockeyshotmap_metrics_{uuid.uuid4().hex}.db")
      Main.initialize_database(db_path)

      rows = []
      for game_id in range(1, 11):
         for shot_index in range(1, 31):
            is_goal = 1 if shot_index % 7 == 0 else 0
            rows.append(
               {
                  "Shot": "Goal" if is_goal else "ngshot",
                  "X": 65.0 + (shot_index % 20),
                  "Y": -15.0 + (shot_index % 30),
                  "Shot_Type": "Wrist" if shot_index % 2 == 0 else "Slap",
                  "Shooter": "Player A" if shot_index % 3 == 0 else "Player B",
                  "Shooter_ID": 101 if shot_index % 3 == 0 else 202,
                  "Team": "TOR" if shot_index % 2 == 0 else "MTL",
                  "Home_Away": 1 if game_id % 2 == 0 else 0,
                  "Period": 1 + (shot_index % 3),
                  "Period_Time": "10:00",
                  "Period_Time_Remaining": "10:00",
                  "Year": "2024",
                  "GameID": game_id,
                  "API_Source": "web",
                  "Goalie": "Goalie X" if shot_index % 2 == 0 else "Goalie Y",
                  "Goalie_ID": 301 if shot_index % 2 == 0 else 302,
                  "Shot_Distance": 20.0 + (shot_index % 40),
                  "Shot_Angle": 5.0 + (shot_index % 55),
                  "Is_Empty_Net": 0,
                  "Strength_State": "5v5" if shot_index % 4 else "5v4",
                  "Score_Differential": (shot_index % 5) - 2,
                  "Zone": "OZ" if shot_index % 2 == 0 else "DZ",
                  "Event_ID": (game_id * 1000) + shot_index,
               }
            )

      inserted = Main.persist_rows(db_path, rows)
      self.assertGreater(inserted, 200)

      summary = Metrics.run_metrics_refresh(
         Metrics.MetricsConfig(
            db_path=db_path,
            score_start_season=2024,
            min_shots_for_comparison=25,
            learning_rate=0.05,
            epochs=300,
            validation_split=0.2,
            calibration_bins=8,
         )
      )

      self.assertEqual(summary["scored_seasons"], ["2024"])
      self.assertGreater(summary["scored_shots"], 0)
      self.assertGreater(summary["player_season_rows"], 0)
      self.assertGreater(summary["team_season_rows"], 0)
      self.assertGreater(summary["goalie_season_rows"], 0)
      self.assertIsNotNone(summary["validation"])
      self.assertEqual(summary["calibration_method"], "sigmoid")
      self.assertIn("top_features", summary)
      self.assertIn("feature_pruning_candidates", summary)
      self.assertEqual(summary["model_type"], "player_adaptive")
      self.assertGreater(summary.get("player_career_rows", 0), 0)

      with sqlite3.connect(db_path) as connection:
         player_rows = connection.execute("SELECT COUNT(*) FROM player_season_metrics").fetchone()[0]
         team_rows = connection.execute("SELECT COUNT(*) FROM team_season_metrics").fetchone()[0]
         goalie_rows = connection.execute("SELECT COUNT(*) FROM goalie_season_metrics").fetchone()[0]
         shot_rows = connection.execute("SELECT COUNT(*) FROM shot_xg").fetchone()[0]
         model_row = connection.execute(
            """
            SELECT model_type, validation_auc, validation_log_loss, validation_brier,
                   validation_ece, calibration_method
            FROM metrics_model_runs
            ORDER BY created_at DESC
            LIMIT 1
            """
         ).fetchone()
         bin_rows = connection.execute("SELECT COUNT(*) FROM metrics_model_validation_bins").fetchone()[0]
         player_career_rows = connection.execute("SELECT COUNT(*) FROM player_career_trajectory").fetchone()[0]
         goalie_career_rows = connection.execute("SELECT COUNT(*) FROM goalie_career_trajectory").fetchone()[0]

      self.assertGreater(player_rows, 0)
      self.assertGreater(team_rows, 0)
      self.assertGreater(goalie_rows, 0)
      self.assertGreater(shot_rows, 0)
      self.assertIsNotNone(model_row)
      self.assertEqual(model_row[0], "xgboost_player_adaptive_v1")
      self.assertEqual(model_row[5], "sigmoid")
      self.assertIsNotNone(model_row[1])
      self.assertIsNotNone(model_row[2])
      self.assertIsNotNone(model_row[3])
      self.assertIsNotNone(model_row[4])
      self.assertGreater(bin_rows, 0)
      self.assertGreater(player_career_rows, 0)
      self.assertGreater(goalie_career_rows, 0)

   def test_metrics_refresh_skips_unchanged_season(self):
      db_path = str(Path(tempfile.gettempdir()) / f"hockeyshotmap_metrics_skip_{uuid.uuid4().hex}.db")
      Main.initialize_database(db_path)

      rows = []
      for game_id in range(1, 9):
         for shot_index in range(1, 26):
            is_goal = 1 if shot_index % 6 == 0 else 0
            rows.append(
               {
                  "Shot": "Goal" if is_goal else "ngshot",
                  "X": 62.0 + (shot_index % 25),
                  "Y": -18.0 + (shot_index % 35),
                  "Shot_Type": "Wrist" if shot_index % 2 == 0 else "Snap",
                  "Shooter": "Player C" if shot_index % 3 == 0 else "Player D",
                  "Shooter_ID": 401 if shot_index % 3 == 0 else 402,
                  "Team": "COL" if shot_index % 2 == 0 else "DAL",
                  "Home_Away": 1 if game_id % 2 == 0 else 0,
                  "Period": 1 + (shot_index % 3),
                  "Period_Time": "10:00",
                  "Period_Time_Remaining": "10:00",
                  "Year": "2024",
                  "GameID": game_id,
                  "API_Source": "web",
                  "Goalie": "Goalie M" if shot_index % 2 == 0 else "Goalie N",
                  "Goalie_ID": 501 if shot_index % 2 == 0 else 502,
                  "Shot_Distance": 18.0 + (shot_index % 38),
                  "Shot_Angle": 4.0 + (shot_index % 50),
                  "Is_Empty_Net": 0,
                  "Strength_State": "5v5",
                  "Score_Differential": (shot_index % 5) - 2,
                  "Zone": "OZ" if shot_index % 2 == 0 else "DZ",
                  "Event_ID": (game_id * 1000) + shot_index,
               }
            )

      Main.persist_rows(db_path, rows)

      first_summary = Metrics.run_metrics_refresh(
         Metrics.MetricsConfig(
            db_path=db_path,
            score_start_season=2024,
            min_shots_for_comparison=20,
            epochs=250,
            use_player_effects=False,
         )
      )
      self.assertEqual(first_summary["scored_seasons"], ["2024"])

      second_summary = Metrics.run_metrics_refresh(
         Metrics.MetricsConfig(
            db_path=db_path,
            score_start_season=2024,
            min_shots_for_comparison=20,
            epochs=250,
            use_player_effects=False,
         )
      )

      self.assertEqual(second_summary["scored_seasons"], [])
      self.assertEqual(second_summary["skipped_seasons"], ["2024"])
      self.assertEqual(second_summary["scored_shots"], 0)


class TestPriorEventGrouping(unittest.TestCase):
   def test_group_prior_event_types_goal_override_resets_tracking_fields(self):
      frame = pd.DataFrame(
         {
            "prior_event_type": ["Goal", "TaKeAwAy"],
            "seconds_since_last_event": [12.5, 3.0],
            "puck_velocity": [24.8, 11.2],
            "crossed_royal_road": [True, True],
         }
      )

      grouped = Metrics.group_prior_event_types(frame, "prior_event_type")

      self.assertEqual(str(grouped.loc[0, "prior_event_grouped"]), "set_play")
      self.assertEqual(grouped.loc[0, "seconds_since_last_event"], 0)
      self.assertEqual(grouped.loc[0, "puck_velocity"], 0)
      self.assertEqual(grouped.loc[0, "crossed_royal_road"], 0)
      self.assertEqual(str(grouped.loc[1, "prior_event_grouped"]), "rush")

   def test_group_prior_event_types_case_insensitive_and_safe_fallback(self):
      frame = pd.DataFrame(
         {
            "prior_event_type": [
               "SHOT-ON-GOAL",
               "MiSsEd-ShOt",
               "BLoCkEd-ShOt",
               "FAILED-SHOT-ATTEMPT",
               "gIvEaWaY",
               "TAKEAWAY",
               "Faceoff",
               "PENALTY",
               "Stoppage",
               "HiT",
               "Something-Unmapped",
               None,
            ]
         }
      )

      grouped = Metrics.group_prior_event_types(frame, "prior_event_type")

      expected = [
         "rebound",
         "rebound",
         "rebound",
         "rebound",
         "rush",
         "rush",
         "set_play",
         "set_play",
         "set_play",
         "locational_clash",
         "set_play",
         "set_play",
      ]
      self.assertEqual(grouped["prior_event_grouped"].astype(str).tolist(), expected)


if __name__ == "__main__":
   unittest.main()
