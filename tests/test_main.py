import sqlite3
import tempfile
import unittest
import uuid
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import Main
import requests


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

    def test_season_has_coordinate_shots_true(self):
        with patch("Main.fetch_season_game_numbers_from_stats", return_value=[1, 2, 3]):
            with patch(
                "Main._fetch_json_allow_404",
                side_effect=[
                    {"plays": [{"typeDescKey": "goal", "details": {"eventOwnerTeamId": 1}}]},
                    {"plays": [{"typeDescKey": "shot-on-goal", "details": {"xCoord": 10, "yCoord": -5}}]},
                ],
            ):
                supported = Main.season_has_coordinate_shots("2013", Main.REGULAR_SEASON, 10, sample_games=3)
        self.assertTrue(supported)

    def test_season_has_coordinate_shots_false(self):
        with patch("Main.fetch_season_game_numbers_from_stats", return_value=[1, 2]):
            with patch(
                "Main._fetch_json_allow_404",
                side_effect=[
                    {"plays": [{"typeDescKey": "goal", "details": {"eventOwnerTeamId": 1}}]},
                    {"plays": [{"typeDescKey": "shot", "details": {"xCoord": None, "yCoord": None}}]},
                ],
            ):
                supported = Main.season_has_coordinate_shots("2013", Main.REGULAR_SEASON, 10, sample_games=2)
        self.assertFalse(supported)

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
        self.assertEqual(rows[0]["Period_Time"], "12:34")
        self.assertEqual(rows[0]["Strength_State"], "5v5")
        self.assertEqual(rows[0]["Event_ID"], 101)
        self.assertEqual(rows[1]["Shot"], "ngshot")
        self.assertEqual(rows[1]["Home_Away"], 0)
        self.assertEqual(rows[1]["Goalie_ID"], 42)
        self.assertEqual(rows[1]["Is_Empty_Net"], 0)
        self.assertEqual(rows[1]["Score_Differential"], -1)
        self.assertEqual(rows[1]["Zone"], "DZ")

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

    def test_edge_is_supported_for_season_true(self):
        class FakeResponse:
            status_code = 200

            def raise_for_status(self):
                return None

            def json(self):
                return {
                    "seasonsWithEdgeStats": [
                        {"id": 20212022, "gameTypes": [2, 3]},
                        {"id": 20222023, "gameTypes": [2, 3]},
                    ]
                }

        with patch.object(Main._HTTP_SESSION, "get", return_value=FakeResponse()):
            supported = Main.edge_is_supported_for_season("2022", Main.REGULAR_SEASON, 10)
        self.assertTrue(supported)

    def test_edge_is_supported_for_season_false(self):
        class FakeResponse:
            status_code = 200

            def raise_for_status(self):
                return None

            def json(self):
                return {
                    "seasonsWithEdgeStats": [
                        {"id": 20232024, "gameTypes": [2, 3]},
                    ]
                }

        with patch.object(Main._HTTP_SESSION, "get", return_value=FakeResponse()):
            supported = Main.edge_is_supported_for_season("2022", Main.REGULAR_SEASON, 10)
        self.assertFalse(supported)

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

    def test_discover_earliest_full_season_returns_first_valid(self):
        class FakeResponse:
            def __init__(self, payload):
                self._payload = payload

            def raise_for_status(self):
                return None

            def json(self):
                return self._payload

        def fake_get(url, timeout=None, params=None):
            if url.endswith("/season"):
                return FakeResponse(
                    ["20082009", "20092010"]
                )
            raise AssertionError(f"Unexpected URL call: {url} params={params}")

        with patch.object(Main._HTTP_SESSION, "get", side_effect=fake_get):
            earliest = Main.discover_earliest_full_season(Main.REGULAR_SEASON, 10)
        self.assertEqual(earliest, 2008)

    def test_discover_earliest_full_season_stats_source(self):
        class FakeResponse:
            def __init__(self, payload):
                self._payload = payload

            def raise_for_status(self):
                return None

            def json(self):
                return self._payload

        def fake_get(url, timeout=None, params=None):
            if url.endswith("/en/season"):
                return FakeResponse({"data": [{"id": "20102011"}, {"id": "20112012"}]})
            raise AssertionError(f"Unexpected URL call: {url} params={params}")

        with patch.object(Main._HTTP_SESSION, "get", side_effect=fake_get):
            earliest = Main.discover_earliest_full_season(Main.REGULAR_SEASON, 10, preferred_source="stats")
        self.assertEqual(earliest, 2010)

    def test_discover_earliest_full_season_clamps_to_modern_floor(self):
        class FakeResponse:
            def __init__(self, payload):
                self._payload = payload

            def raise_for_status(self):
                return None

            def json(self):
                return self._payload

        def fake_get(url, timeout=None, params=None):
            if url.endswith("/season"):
                return FakeResponse(["19171918", "19791980"])
            raise AssertionError(f"Unexpected URL call: {url} params={params}")

        with patch.object(Main._HTTP_SESSION, "get", side_effect=fake_get):
            earliest = Main.discover_earliest_full_season(Main.REGULAR_SEASON, 10)
        self.assertEqual(earliest, Main.MODERN_ERA_START_SEASON)

    def test_discover_earliest_full_season_fallback_on_failure(self):
        with patch.object(Main._HTTP_SESSION, "get", side_effect=requests.RequestException("network down")):
            earliest = Main.discover_earliest_full_season(Main.REGULAR_SEASON, 10)
        self.assertEqual(earliest, Main.DEFAULT_START_SEASON_FALLBACK)

    def test_run_state_helpers_skip_completed_range(self):
        db_path = str(Path(tempfile.gettempdir()) / f"hockeyshotmap_runstate_{uuid.uuid4().hex}.db")
        Main.initialize_database(db_path)

        with sqlite3.connect(db_path) as connection:
            cursor = connection.cursor()
            Main._upsert_run_state(
                cursor,
                "2024",
                Main.REGULAR_SEASON,
                1,
                10,
                10,
                10,
                100,
                "complete",
            )
            connection.commit()

        self.assertTrue(Main._season_run_is_complete(db_path, "2024", Main.REGULAR_SEASON, 1, 10))
        self.assertEqual(Main._resume_start_game(db_path, "2024", Main.REGULAR_SEASON, 1, 10), 11)
        self.assertFalse(Main._season_run_is_complete(db_path, "2024", Main.REGULAR_SEASON, 1, 11))


if __name__ == "__main__":
    unittest.main()
