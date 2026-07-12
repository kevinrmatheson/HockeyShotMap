import sqlite3
import sys
import tempfile
import unittest
import uuid
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import Main


class TestMainPipeline(unittest.TestCase):
    def test_build_web_play_by_play_url(self):
        url = Main.build_web_play_by_play_url("2013", Main.REGULAR_SEASON, 7)
        self.assertEqual(
            url,
            "https://api-web.nhle.com/v1/gamecenter/2013020007/play-by-play",
        )

    def test_build_stats_shiftcharts_url(self):
        self.assertEqual(
            Main.build_stats_shiftcharts_url(),
            "https://api.nhle.com/stats/rest/en/shiftcharts",
        )

    def test_build_stats_games_url(self):
        self.assertEqual(
            Main.build_stats_games_url(),
            "https://api.nhle.com/stats/rest/en/game",
        )

    def test_fetch_season_game_numbers_from_stats_filters_and_sorts(self):
        class FakeResponse:
            status_code = 200

            def raise_for_status(self):
                return None

            def json(self):
                return {
                    "data": [
                        {"gameNumber": 2, "gameStateId": 7},
                        {"gameNumber": 1, "gameStateId": 6},
                        {"gameNumber": 3, "gameStateId": 1},
                        {"id": 2023020042, "gameStateId": 7},
                    ]
                }

        with patch.object(Main._HTTP_SESSION, "get", return_value=FakeResponse()):
            numbers = Main.fetch_season_game_numbers_from_stats("2023", Main.REGULAR_SEASON, 10)

        self.assertEqual(numbers, [1, 2, 42])

    def test_fetch_and_parse_game_rows_stats_source(self):
        with patch("Main._fetch_json", return_value={"data": [{"eventType": "shot", "xCoord": 10, "yCoord": 5}]}) as fetch_mock:
            rows = Main._fetch_and_parse_game_rows("2025", Main.REGULAR_SEASON, 1, 10, ["stats"])

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["API_Source"], "stats")
        self.assertEqual(rows[0]["GameID"], 1)
        self.assertEqual(rows[0]["Year"], "2025")
        fetch_mock.assert_called_once()

    def test_fetch_and_parse_game_rows_web_rows_skip_stats_fallback(self):
        with patch("Main._fetch_json_allow_404", return_value={"plays": [{"typeDescKey": "goal", "details": {"xCoord": 10, "yCoord": 5}}]}), patch("Main._fetch_json") as fetch_mock:
            rows = Main._fetch_and_parse_game_rows("2025", Main.REGULAR_SEASON, 1, 10, ["web", "stats"])

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["API_Source"], "web")
        fetch_mock.assert_not_called()

    def test_shot_sources_for_season_pre_coordinate_era(self):
        self.assertEqual(Main.shot_sources_for_season("2008"), ["stats"])

    def test_shot_sources_for_season_coordinate_era_and_newer(self):
        self.assertEqual(Main.shot_sources_for_season("2009"), ["web", "stats"])
        self.assertEqual(Main.shot_sources_for_season("2024"), ["web", "stats"])

    def test_parse_web_shot_events(self):
        payload = {
            "homeTeam": {"abbrev": "TOR"},
            "awayTeam": {"abbrev": "MTL"},
            "plays": [
                {
                    "typeDescKey": "goal",
                    "eventId": 101,
                    "periodDescriptor": {"number": 1},
                    "about": {"periodTime": "12:34", "periodTimeRemaining": "07:26", "period": 1},
                    "details": {
                        "xCoord": 30,
                        "yCoord": -6,
                        "shotType": "Wrist",
                        "eventOwnerTeamTricode": "TOR",
                        "scoringPlayerName": "Scorer A",
                        "scoringPlayerId": 11,
                        "goalieInNetName": "Goalie X",
                        "goalieInNetId": 41,
                        "shotDistance": 18.4,
                        "shotAngle": 22.1,
                        "emptyNet": False,
                        "situationCode": "5v5",
                        "scoreDifferential": 0,
                        "zoneCode": "OZ",
                    },
                },
                {
                    "typeDescKey": "shot-on-goal",
                    "eventId": 102,
                    "periodDescriptor": {"number": 2},
                    "about": {"periodTime": "03:21", "periodTimeRemaining": "16:39", "period": 2},
                    "details": {
                        "xCoord": -20,
                        "yCoord": 10,
                        "shotType": "Slap",
                        "eventOwnerTeamTricode": "MTL",
                        "shootingPlayerName": "Shooter B",
                        "shootingPlayerId": 22,
                        "goalieInNetName": "Goalie Y",
                        "goalieInNetId": 42,
                        "shotDistance": 31.0,
                        "shotAngle": 35.5,
                        "isEmptyNet": 0,
                        "strength": "5v4",
                        "goalDifferential": -1,
                        "zone": "DZ",
                    },
                },
            ],
        }

        rows = Main.parse_web_shot_events(payload, "2023", 204)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["Shot"], "Goal")
        self.assertEqual(rows[0]["API_Source"], "web")
        self.assertEqual(rows[0]["Shooter_ID"], 11)
        self.assertEqual(rows[0]["Goalie"], "Goalie X")
        self.assertEqual(rows[0]["Event_ID"], 101)
        self.assertEqual(rows[1]["Shot"], "ngshot")
        self.assertEqual(rows[1]["Home_Away"], 0)
        self.assertEqual(rows[1]["Goalie_ID"], 42)
        self.assertEqual(rows[1]["Is_Empty_Net"], 0)
        self.assertEqual(rows[1]["Score_Differential"], -1)
        self.assertEqual(rows[1]["Zone"], "DZ")

    def test_parse_web_shot_events_captures_previous_coordinate_event(self):
        payload = {
            "homeTeam": {"abbrev": "TOR"},
            "awayTeam": {"abbrev": "MTL"},
            "plays": [
                {
                    "typeDescKey": "shot-on-goal",
                    "sortOrder": 20,
                    "eventId": 202,
                    "periodDescriptor": {"number": 1},
                    "about": {"periodTime": "01:00", "periodTimeRemaining": "19:00", "period": 1},
                    "details": {
                        "xCoord": 25,
                        "yCoord": 4,
                        "shotType": "Wrist",
                        "eventOwnerTeamTricode": "TOR",
                        "shootingPlayerName": "Shooter A",
                    },
                },
                {
                    "typeDescKey": "faceoff",
                    "sortOrder": 10,
                    "eventId": 201,
                    "periodDescriptor": {"number": 1},
                    "about": {"periodTime": "00:58", "periodTimeRemaining": "19:02", "period": 1},
                    "details": {
                        "xCoord": 18,
                        "yCoord": -3,
                    },
                },
            ],
        }

        rows = Main.parse_web_shot_events(payload, "2023", 205)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["Prev_Event_Type"], "faceoff")
        self.assertEqual(rows[0]["Prev_Event_X"], 18.0)
        self.assertEqual(rows[0]["Prev_Event_Y"], -3.0)

    def test_parse_web_shot_events_derives_missing_2025_metadata(self):
        payload = {
            "homeTeam": {"abbrev": "TOR"},
            "awayTeam": {"abbrev": "MTL"},
            "plays": [
                {
                    "typeDescKey": "goal",
                    "sortOrder": 10,
                    "eventId": 301,
                    "timeInPeriod": "00:05",
                    "timeRemaining": "19:55",
                    "periodDescriptor": {"number": 1},
                    "homeTeamDefendingSide": "right",
                    "details": {
                        "xCoord": 20,
                        "yCoord": 3,
                        "shotType": "wrist",
                        "eventOwnerTeamTricode": "TOR",
                        "scoringPlayerName": "Scorer A",
                    },
                },
                {
                    "typeDescKey": "faceoff",
                    "sortOrder": 20,
                    "eventId": 302,
                    "timeInPeriod": "00:10",
                    "timeRemaining": "19:50",
                    "periodDescriptor": {"number": 1},
                    "homeTeamDefendingSide": "right",
                    "details": {
                        "xCoord": 0,
                        "yCoord": 0,
                    },
                },
                {
                    "typeDescKey": "shot-on-goal",
                    "sortOrder": 30,
                    "eventId": 303,
                    "timeInPeriod": "00:20",
                    "timeRemaining": "19:40",
                    "periodDescriptor": {"number": 1},
                    "homeTeamDefendingSide": "right",
                    "details": {
                        "xCoord": -30,
                        "yCoord": 10,
                        "shotType": "wrist",
                        "eventOwnerTeamTricode": "MTL",
                        "shootingPlayerName": "Shooter B",
                    },
                },
            ],
        }

        rows = Main.parse_web_shot_events(payload, "2025", 206)
        self.assertEqual(len(rows), 2)

        shot_row = rows[1]
        self.assertEqual(shot_row["Prev_Event_Type"], "faceoff")
        self.assertEqual(shot_row["Prev_Event_Seconds_Ago"], 10)
        self.assertEqual(shot_row["Score_Differential"], -1)
        self.assertAlmostEqual(shot_row["Shot_Distance"], 59.84, places=1)
        self.assertAlmostEqual(shot_row["Shot_Angle"], 9.6, places=1)

    def test_persist_rows_updates_existing_row_with_previous_event_data(self):
        db_path = str(Path(tempfile.gettempdir()) / f"hockeyshotmap_prev_event_{uuid.uuid4().hex}.db")
        Main.initialize_database(db_path)

        base_row = {
            "Shot": "Goal",
            "X": 20.0,
            "Y": -3.0,
            "Shot_Type": "Wrist Shot",
            "Shooter": "Test Shooter",
            "Team": "TOR",
            "Home_Away": 1,
            "Period": 1,
            "Year": "2013",
            "GameID": 1,
        }
        enriched_row = dict(base_row)
        enriched_row.update(
            {
                "Prev_Event_Type": "faceoff",
                "Prev_Event_X": 12.0,
                "Prev_Event_Y": -4.0,
                "Prev_Event_Seconds_Ago": 6,
            }
        )

        inserted_first = Main.persist_rows(db_path, [base_row])
        inserted_second = Main.persist_rows(db_path, [enriched_row])

        self.assertEqual(inserted_first, 1)
        self.assertEqual(inserted_second, 1)

        with sqlite3.connect(db_path) as connection:
            row = connection.execute(
                "SELECT prev_event_type, prev_event_x, prev_event_y, prev_event_seconds_ago, COUNT(*) FROM shots"
            ).fetchone()
        self.assertEqual(row[0], "faceoff")
        self.assertEqual(row[1], 12.0)
        self.assertEqual(row[2], -4.0)
        self.assertEqual(row[3], 6)
        self.assertEqual(row[4], 1)

    def test_roster_player_helpers(self):
        roster_payload = {
            "forwards": [
                {
                    "person": {"id": 101, "fullName": "Skater One"},
                    "position": {"code": "C"},
                }
            ],
            "goalies": [
                {
                    "person": {"id": 202, "fullName": "Goalie One"},
                    "position": {"code": "G"},
                }
            ],
        }

        players = Main._extract_roster_players(roster_payload)
        self.assertEqual(len(players), 2)
        self.assertEqual(Main._player_id_from_roster_entry(players[0]), 101)
        self.assertEqual(Main._player_position_code(players[0]), "C")
        self.assertEqual(Main._player_position_code(players[1]), "G")

    def test_edge_supported_for_season_threshold(self):
        self.assertFalse(Main.edge_supported_for_season("2020"))
        self.assertTrue(Main.edge_supported_for_season("2021"))

    def test_resolve_capture_edge_setting_defaults_by_season(self):
        self.assertFalse(Main.resolve_capture_edge_setting("2020", None))
        self.assertTrue(Main.resolve_capture_edge_setting("2021", None))
        self.assertTrue(Main.resolve_capture_edge_setting("2025", None))

    def test_resolve_capture_edge_setting_honors_explicit_override(self):
        self.assertFalse(Main.resolve_capture_edge_setting("2025", False))
        self.assertTrue(Main.resolve_capture_edge_setting("2020", True))

    def test_parse_args_defaults_capture_edge_to_auto(self):
        with patch.object(sys, "argv", ["Main.py"]):
            args = Main.parse_args()

        self.assertIsNone(args.capture_edge)

    def test_parse_args_supports_no_capture_edge_override(self):
        with patch.object(sys, "argv", ["Main.py", "--no-capture-edge"]):
            args = Main.parse_args()

        self.assertFalse(args.capture_edge)

    def test_edge_detail_url_builders(self):
        self.assertEqual(
            Main.build_web_edge_team_detail_url(9, "2024", Main.REGULAR_SEASON),
            "https://api-web.nhle.com/v1/edge/team-detail/9/20242025/2",
        )
        self.assertEqual(
            Main.build_web_edge_skater_detail_url(8482116, "2024", Main.REGULAR_SEASON),
            "https://api-web.nhle.com/v1/edge/skater-detail/8482116/20242025/2",
        )
        self.assertEqual(
            Main.build_web_edge_goalie_detail_url(8476999, "2024", Main.REGULAR_SEASON),
            "https://api-web.nhle.com/v1/edge/goalie-detail/8476999/20242025/2",
        )

    def test_persist_rows_is_idempotent(self):
        row = {
            "Shot": "Goal",
            "X": 20.0,
            "Y": -3.0,
            "Shot_Type": "Wrist Shot",
            "Shooter": "Test Shooter",
            "Team": "TOR",
            "Home_Away": 1,
            "Period": 1,
            "Year": "2013",
            "GameID": 1,
        }

        db_path = str(Path(tempfile.gettempdir()) / f"hockeyshotmap_test_{uuid.uuid4().hex}.db")
        Main.initialize_database(db_path)

        inserted_first = Main.persist_rows(db_path, [row])
        inserted_second = Main.persist_rows(db_path, [row])

        self.assertEqual(inserted_first, 1)
        self.assertEqual(inserted_second, 0)

        with sqlite3.connect(db_path) as connection:
            count = connection.execute("SELECT COUNT(*) FROM shots").fetchone()[0]
        self.assertEqual(count, 1)

    def test_current_nhl_season_start_year_rollover(self):
        self.assertEqual(Main.current_nhl_season_start_year(datetime(2026, 1, 15)), 2025)
        self.assertEqual(Main.current_nhl_season_start_year(datetime(2026, 10, 1)), 2026)

    def test_season_range_with_explicit_end(self):
        seasons = Main.season_range(2013, 2015)
        self.assertEqual(seasons, ["2013", "2014", "2015"])

    def test_season_range_uses_current_when_end_missing(self):
        original_fn = Main.current_nhl_season_start_year
        try:
            Main.current_nhl_season_start_year = lambda today=None: 2020
            seasons = Main.season_range(2018, None)
            self.assertEqual(seasons, ["2018", "2019", "2020"])
        finally:
            Main.current_nhl_season_start_year = original_fn

    def test_season_range_rejects_invalid_bounds(self):
        with self.assertRaises(ValueError):
            Main.season_range(2022, 2021)

    def test_parse_web_shot_events_uses_timeInPeriod_for_period_time(self):
        """Test that period_time is extracted from timeInPeriod when periodTime is not in about."""
        payload = {
            "homeTeam": {"abbrev": "TOR"},
            "awayTeam": {"abbrev": "MTL"},
            "plays": [
                {
                    "typeDescKey": "shot-on-goal",
                    "eventId": 101,
                    "periodDescriptor": {"number": 1},
                    "timeInPeriod": "05:30",
                    "timeRemaining": "14:30",
                    "details": {
                        "xCoord": 20,
                        "yCoord": 5,
                        "shotType": "Wrist",
                        "eventOwnerTeamTricode": "TOR",
                        "shootingPlayerId": 11,
                    },
                },
            ],
        }

        rows = Main.parse_web_shot_events(payload, "2023", 101)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["Period_Time"], "05:30")
        self.assertEqual(rows[0]["Period_Time_Remaining"], "14:30")

    def test_backfill_player_names_returns_zero_when_no_players_table(self):
        """Test that backfill returns (0, 0) when players table doesn't exist."""
        db_path = str(Path(tempfile.gettempdir()) / f"hockeyshotmap_backfill_{uuid.uuid4().hex}.db")
        Main.initialize_database(db_path)

        # No players table exists
        shooter_filled, goalie_filled = Main.backfill_player_names(db_path)
        self.assertEqual(shooter_filled, 0)
        self.assertEqual(goalie_filled, 0)

    def test_backfill_player_names_fills_from_players_table(self):
        """Test that backfill correctly fills shooter and goalie names from players table."""
        db_path = str(Path(tempfile.gettempdir()) / f"hockeyshotmap_backfill2_{uuid.uuid4().hex}.db")
        Main.initialize_database(db_path)

        # Insert a row with shooter_id and goalie_id but no names
        row = {
            "Shot": "Goal",
            "X": 20.0,
            "Y": -3.0,
            "Shot_Type": "Wrist Shot",
            "Shooter": "Unknown",
            "Shooter_ID": 101,
            "Team": "TOR",
            "Home_Away": 1,
            "Period": 1,
            "Year": "2023",
            "GameID": 1,
            "Goalie": None,
            "Goalie_ID": 201,
        }
        Main.persist_rows(db_path, [row])

        # Create players table and insert player records
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("CREATE TABLE players (player_id INTEGER PRIMARY KEY, full_name TEXT)")
            cursor.execute("INSERT INTO players (player_id, full_name) VALUES (101, 'John Doe'), (201, 'Jane Smith')")
            conn.commit()

        # Run backfill
        shooter_filled, goalie_filled = Main.backfill_player_names(db_path)
        self.assertEqual(shooter_filled, 1)
        self.assertEqual(goalie_filled, 1)

        # Verify the names were filled
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT shooter, goalie FROM shots")
            result = cursor.fetchone()
            self.assertEqual(result[0], "John Doe")
            self.assertEqual(result[1], "Jane Smith")

    def test_check_players_table_warning_logs_warning_when_missing(self):
        """Test that check_players_table_warning logs a warning when players table is missing."""
        db_path = str(Path(tempfile.gettempdir()) / f"hockeyshotmap_warning_{uuid.uuid4().hex}.db")
        Main.initialize_database(db_path)

        # Should return True (warning logged) when players table doesn't exist
        result = Main.check_players_table_warning(db_path)
        self.assertTrue(result)

        # Create players table
        with sqlite3.connect(db_path) as conn:
            conn.execute("CREATE TABLE players (player_id INTEGER PRIMARY KEY, full_name TEXT)")
            conn.commit()

        # Should return False (no warning) when players table exists
        result = Main.check_players_table_warning(db_path)
        self.assertFalse(result)

if __name__ == "__main__":
    unittest.main()
