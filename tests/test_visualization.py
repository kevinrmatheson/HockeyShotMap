import sqlite3
import tempfile
import unittest
import uuid
from pathlib import Path

import Main

from visualization.query_engine import (
   HeatmapFilters,
   build_filters_from_args,
   get_filter_options,
   get_heatmap_bins,
   get_summary,
   latest_season,
)


def _sample_row(**overrides):
   row = {
      "Shot": "Goal",
      "X": 20.1,
      "Y": 10.1,
      "Shot_Type": "Wrist Shot",
      "Shooter": "Shooter One",
      "Shooter_ID": 101,
      "Team": "TOR",
      "Home_Away": 1,
      "Period": 1,
      "Period_Time": "12:34",
      "Period_Time_Remaining": "07:26",
      "Year": "2024",
      "GameID": 10,
      "API_Source": "web",
      "Goalie": "Goalie One",
      "Goalie_ID": 201,
      "Shot_Distance": 18.5,
      "Shot_Angle": 22.0,
      "Is_Empty_Net": 0,
      "Strength_State": "5v5",
      "Score_Differential": 0,
      "Zone": "OZ",
      "Event_ID": 5001,
   }
   row.update(overrides)
   return row


class TestVisualizationQueryEngine(unittest.TestCase):
   def setUp(self):
      self.db_path = str(Path(tempfile.gettempdir()) / f"hockeyshotmap_viz_{uuid.uuid4().hex}.db")
      Main.initialize_database(self.db_path)
      Main.persist_rows(
         self.db_path,
         [
            _sample_row(),
            _sample_row(
               Shot="ngshot",
               X=21.3,
               Y=11.0,
               Shooter="Shooter Two",
               Shooter_ID=102,
               Team="MTL",
               Home_Away=0,
               Period=2,
               Year="2024",
               GameID=11,
               Shot_Type="Slap Shot",
               Strength_State="5v4",
               Score_Differential=-1,
               Zone="DZ",
               Event_ID=5002,
            ),
            _sample_row(
               Shot="Goal",
               X=-10.5,
               Y=-6.2,
               Shooter="Shooter Three",
               Shooter_ID=103,
               Team="TOR",
               Home_Away=1,
               Period=3,
               Year="2023",
               GameID=12,
               Strength_State="5v5",
               Score_Differential=1,
               Zone="NZ",
               Event_ID=5003,
            ),
         ],
      )

   def test_latest_season(self):
      self.assertEqual(latest_season(self.db_path), "2024")

   def test_get_filter_options(self):
      options = get_filter_options(self.db_path)
      self.assertIn("2024", options["seasons"])
      self.assertIn("TOR", options["teams"])
      self.assertIn("Shooter One", options["players"])
      self.assertIn("5v5", options["strength_states"])

   def test_get_summary_with_filters(self):
      summary = get_summary(self.db_path, HeatmapFilters(season="2024", team="TOR"))
      self.assertEqual(summary["shot_count"], 1)
      self.assertEqual(summary["goal_count"], 1)
      self.assertEqual(summary["goal_pct"], 100.0)

   def test_get_heatmap_bins_aggregates_shots(self):
      bins = get_heatmap_bins(self.db_path, HeatmapFilters(season="2024", team="TOR"), bin_size=5.0)
      self.assertEqual(len(bins), 1)
      self.assertEqual(bins[0]["shot_count"], 1)
      self.assertEqual(bins[0]["goal_count"], 1)

   def test_build_filters_from_args(self):
      filters = build_filters_from_args(
         {
            "season": "2024",
            "team": "TOR",
            "player": "Shooter One",
            "strength_state": "5v5",
            "shot_result": "Goal",
            "home_away": "home",
            "period": "1",
         }
      )
      self.assertEqual(filters.season, "2024")
      self.assertEqual(filters.team, "TOR")
      self.assertEqual(filters.player, "Shooter One")
      self.assertEqual(filters.strength_state, "5v5")
      self.assertEqual(filters.shot_result, "Goal")
      self.assertEqual(filters.home_away, 1)
      self.assertEqual(filters.period, 1)


if __name__ == "__main__":
   unittest.main()
