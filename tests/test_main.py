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
    def test_build_game_url(self):
        url = Main.build_game_url("2013", Main.REGULAR_SEASON, 7)
        self.assertEqual(
            url,
            "https://statsapi.web.nhl.com/api/v1/game/2013020007/feed/live",
        )

    def test_parse_shot_events_valid_and_skips_missing_coordinates(self):
        payload = {
            "gameData": {
                "teams": {
                    "home": {"triCode": "TOR"},
                    "away": {"triCode": "MTL"},
                }
            },
            "liveData": {
                "plays": {
                    "allPlays": [
                        {
                            "result": {"event": "Goal", "secondaryType": "Wrist Shot"},
                            "team": {"triCode": "TOR"},
                            "about": {"period": 1},
                            "coordinates": {"x": 45, "y": -10},
                            "players": [{"player": {"fullName": "Player One"}}],
                        },
                        {
                            "result": {"event": "Shot", "secondaryType": "Slap Shot"},
                            "team": {"triCode": "MTL"},
                            "about": {"period": 2},
                            "coordinates": {"x": -33, "y": 8},
                            "players": [{"player": {"fullName": "Player Two"}}],
                        },
                        {
                            "result": {"event": "Shot", "secondaryType": "Backhand"},
                            "team": {"triCode": "TOR"},
                            "about": {"period": 3},
                        },
                    ]
                }
            },
        }

        rows = Main.parse_shot_events(payload, "2013", 99)
        self.assertEqual(len(rows), 2)

        goal_row = rows[0]
        shot_row = rows[1]

        self.assertEqual(goal_row["Shot"], "Goal")
        self.assertEqual(goal_row["Home_Away"], 1)
        self.assertEqual(shot_row["Shot"], "ngshot")
        self.assertEqual(shot_row["Home_Away"], 0)

    def test_parse_shot_events_handles_missing_optional_fields(self):
        payload = {
            "gameData": {
                "teams": {
                    "home": {"triCode": "NYR"},
                    "away": {"triCode": "BOS"},
                }
            },
            "liveData": {
                "plays": {
                    "allPlays": [
                        {
                            "result": {"event": "Shot"},
                            "team": {"triCode": "BOS"},
                            "about": {"period": 1},
                            "coordinates": {"x": 10, "y": 5},
                            "players": [],
                        }
                    ]
                }
            },
        }

        rows = Main.parse_shot_events(payload, "2016", 1)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["Shot_Type"], "Unknown")
        self.assertEqual(rows[0]["Shooter"], "Unknown")

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
            if url.endswith("/seasons"):
                return FakeResponse(
                    {
                        "seasons": [
                            {"seasonId": "20082009"},
                            {"seasonId": "20092010"},
                        ]
                    }
                )
            if url.endswith("/schedule") and params == {"season": "20082009", "gameType": Main.REGULAR_SEASON}:
                return FakeResponse({"totalItems": 0, "dates": []})
            if url.endswith("/schedule") and params == {"season": "20092010", "gameType": Main.REGULAR_SEASON}:
                return FakeResponse(
                    {
                        "totalItems": 2,
                        "dates": [
                            {"games": [{"gamePk": 2009020001}]},
                            {"games": [{"gamePk": 2009021230}]},
                        ],
                    }
                )
            if url.endswith("/game/2009020001/feed/live"):
                return FakeResponse({"ok": True})
            if url.endswith("/game/2009021230/feed/live"):
                return FakeResponse({"ok": True})
            raise AssertionError(f"Unexpected URL call: {url} params={params}")

        with patch("Main.requests.get", side_effect=fake_get):
            earliest = Main.discover_earliest_full_season(Main.REGULAR_SEASON, 10)
        self.assertEqual(earliest, 2009)

    def test_discover_earliest_full_season_fallback_on_failure(self):
        with patch("Main.requests.get", side_effect=requests.RequestException("network down")):
            earliest = Main.discover_earliest_full_season(Main.REGULAR_SEASON, 10)
        self.assertEqual(earliest, Main.DEFAULT_START_SEASON_FALLBACK)


if __name__ == "__main__":
    unittest.main()
