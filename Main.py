#!/usr/bin/env python3
"""
NHL Data Scraper Orchestrator.

This is the unified entry point for scraping NHL data from multiple API endpoints.
It can run individual scrapers on demand or run all scrapers in sequence.

Usage:
    python main.py --scraper shots --season 2023
    python main.py --scraper player_season --season 2023 --game-type 02
    python main.py --all --start-season 2020 --end-season 2023
"""

import argparse
import logging
import sys
from typing import Callable

# Import scraper modules
from shots_scraper import (
    run_season_scrape as run_shots_scrape,
    ScrapeConfig,
    resolve_capture_edge_setting,
    COORDINATE_DATA_START_SEASON,
    DEFAULT_REQUEST_DELAY_SECONDS,
)
from player_bio import (
    fetch_player_bios,
    PlayerBioConfig,
)
from player_season import (
    fetch_and_store_player_season_stats,
)
from team_season import (
    fetch_and_store_team_season_stats,
)
from game_shifts import (
    fetch_and_store_game_shifts,
)

# Supported game type codes
PRESEASON = "01"
REGULAR_SEASON = "02"
PLAYOFFS = "03"
ALLSTAR = "04"


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the orchestrator."""
    parser = argparse.ArgumentParser(
        description="NHL Data Scraper Orchestrator - Scrape NHL API data into SQLite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scrape shots for a single season
  python main.py --scraper shots --season 2023

  # Scrape player season stats
  python main.py --scraper player_season --season 2023

  # Scrape team season stats
  python main.py --scraper team_season --season 2023

  # Scrape game shifts
  python main.py --scraper game_shifts --season 2023

  # Run all scrapers for a season range
  python main.py --all --start-season 2020 --end-season 2023

  # Run with custom options
  python main.py --scraper shots --season 2023 --game-type 03 --max-workers 4 --request-delay 0.5
        """,
    )

    # Scraper selection
    parser.add_argument(
        "--scraper",
        choices=["shots", "player_season", "team_season", "game_shifts", "player_bio"],
        help="Specific scraper to run. If not specified with --all, must be provided.",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all scrapers in sequence.",
    )

    # Season options
    parser.add_argument(
        "--season",
        type=str,
        default=None,
        help="Single season start year (e.g., 2013 for 2013-2014).",
    )

    parser.add_argument(
        "--start-season",
        type=int,
        default=None,
        help="Start season for multi-season run. Defaults to 2009 when --season is not used.",
    )

    parser.add_argument(
        "--end-season",
        type=int,
        default=None,
        help="End season for multi-season run. Defaults to current NHL season.",
    )

    # Game type options
    parser.add_argument(
        "--game-type",
        choices=[PRESEASON, REGULAR_SEASON, PLAYOFFS, ALLSTAR],
        default=REGULAR_SEASON,
        help="NHL game type code. Default: 02 (Regular Season).",
    )

    # Performance options
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum concurrent workers for parallel requests. Default: 4",
    )

    parser.add_argument(
        "--request-delay",
        type=float,
        default=DEFAULT_REQUEST_DELAY_SECONDS,
        help=f"Delay between API requests in seconds. Default: {DEFAULT_REQUEST_DELAY_SECONDS}",
    )

    # Database options
    parser.add_argument(
        "--db-path",
        type=str,
        default="hockey_data.db",
        help="SQLite database path. Default: hockey_data.db",
    )

    # Edge data options
    parser.add_argument(
        "--capture-edge",
        action="store_true",
        default=None,
        help="Capture NHL Edge summary payloads (enabled by default for 2021+ seasons).",
    )

    parser.add_argument(
        "--no-capture-edge",
        dest="capture_edge",
        action="store_false",
        help="Disable NHL Edge summary payload capture.",
    )

    parser.add_argument(
        "--capture-edge-deep",
        action="store_true",
        help="Capture NHL Edge team and player detail payloads.",
    )

    # Player bio options
    parser.add_argument(
        "--start-player-id",
        type=int,
        default=0,
        help="Start player ID for player_bio scraper (0 = auto-detect from shots table).",
    )

    parser.add_argument(
        "--end-player-id",
        type=int,
        default=0,
        help="End player ID for player_bio scraper (0 = auto-detect from shots table).",
    )

    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Force re-fetching all data, even if already exists.",
    )

    # Logging
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level. Default: INFO",
    )

    return parser


def get_season_range(args: argparse.Namespace) -> list[str]:
    """Determine the season range to process."""
    if args.season is not None:
        return [str(args.season)]

    start = args.start_season if args.start_season is not None else COORDINATE_DATA_START_SEASON
    end = args.end_season

    if end is None:
        from datetime import datetime
        end = datetime.now().year if datetime.now().month >= 9 else datetime.now().year - 1

    if end < start:
        raise ValueError(f"end-season ({end}) must be >= start-season ({start})")

    return [str(year) for year in range(start, end + 1)]


def run_shots_scraper(args: argparse.Namespace, season: str) -> dict:
    """Run the shots scraper for a given season."""
    logging.info("Running shots scraper for season %s", season)

    capture_edge = resolve_capture_edge_setting(season, args.capture_edge)

    config = ScrapeConfig(
        season=season,
        game_type=args.game_type,
        start_game=args.start_game if hasattr(args, 'start_game') else 1,
        end_game=args.end_game if hasattr(args, 'end_game') else 1271,
        timeout_seconds=args.timeout if hasattr(args, 'timeout') else 10,
        max_workers=args.max_workers,
        request_delay_seconds=args.request_delay,
        db_path=args.db_path,
        empty_game_streak=args.empty_game_streak if hasattr(args, 'empty_game_streak') else 200,
    )

    games_processed, rows_inserted, edge_payloads, edge_detail_payloads, player_stats = run_shots_scrape(
        config,
        capture_edge,
        args.capture_edge_deep,
    )

    return {
        "games_processed": games_processed,
        "rows_inserted": rows_inserted,
        "edge_payloads": edge_payloads,
        "edge_detail_payloads": edge_detail_payloads,
        "player_stats": player_stats,
    }


def run_player_bio_scraper(args: argparse.Namespace) -> dict:
    """Run the player bio scraper."""
    logging.info("Running player bio scraper")

    config = PlayerBioConfig(
        start_player_id=args.start_player_id,
        end_player_id=args.end_player_id,
        timeout_seconds=args.timeout if hasattr(args, 'timeout') else 10,
        max_workers=args.max_workers,
        request_delay_seconds=args.request_delay,
        db_path=args.db_path,
        force_refresh=args.force_refresh,
    )

    players_processed, bios_inserted = fetch_player_bios(config)

    return {
        "players_processed": players_processed,
        "bios_inserted": bios_inserted,
    }


def run_player_season_scraper(args: argparse.Namespace, season: str) -> dict:
    """Run the player season stats scraper."""
    logging.info("Running player season stats scraper for season %s", season)

    inserted = fetch_and_store_player_season_stats(
        args.db_path,
        season,
        args.game_type,
        args.timeout if hasattr(args, 'timeout') else 10,
    )

    return {
        "rows_inserted": inserted,
    }


def run_team_season_scraper(args: argparse.Namespace, season: str) -> dict:
    """Run the team season stats scraper."""
    logging.info("Running team season stats scraper for season %s", season)

    inserted = fetch_and_store_team_season_stats(
        args.db_path,
        season,
        args.game_type,
        args.timeout if hasattr(args, 'timeout') else 10,
    )

    return {
        "rows_inserted": inserted,
    }


def run_game_shifts_scraper(args: argparse.Namespace, season: str) -> dict:
    """Run the game shifts scraper."""
    logging.info("Running game shifts scraper for season %s", season)

    inserted = fetch_and_store_game_shifts(
        args.db_path,
        season,
        args.game_type,
        args.timeout if hasattr(args, 'timeout') else 10,
    )

    return {
        "rows_inserted": inserted,
    }


# Scraper registry
SCRAPERS: dict[str, Callable] = {
    "shots": run_shots_scraper,
    "player_bio": run_player_bio_scraper,
    "player_season": run_player_season_scraper,
    "team_season": run_team_season_scraper,
    "game_shifts": run_game_shifts_scraper,
}


def main() -> int:
    """Main entry point for the orchestrator."""
    parser = create_parser()
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    # Validate arguments
    if not args.scraper and not args.all:
        parser.error("Must specify --scraper or --all")

    if args.scraper and args.all:
        parser.error("Cannot use both --scraper and --all")

    if args.season is not None and (args.start_season is not None or args.end_season is not None):
        parser.error("Use either --season or --start-season/--end-season, not both")

    # Get season range
    try:
        seasons = get_season_range(args)
    except ValueError as e:
        parser.error(str(e))

    # Track results
    total_results = {}

    # Run scrapers
    scrapers_to_run = [args.scraper] if args.scraper else list(SCRAPERS.keys())

    for scraper_name in scrapers_to_run:
        if scraper_name not in SCRAPERS:
            logging.error("Unknown scraper: %s", scraper_name)
            continue

        scraper_func = SCRAPERS[scraper_name]

        for season in seasons:
            try:
                results = scraper_func(args, season)
                total_results[f"{scraper_name}_{season}"] = results
                logging.info("Completed %s for season %s: %s", scraper_name, season, results)
            except Exception as e:
                logging.error("Error running %s for season %s: %s", scraper_name, season, e)
                if args.force_refresh:
                    raise

    # Print summary
    logging.info("=" * 60)
    logging.info("SCRAPE SUMMARY")
    logging.info("=" * 60)
    for key, results in total_results.items():
        logging.info("%s: %s", key, results)

    return 0


if __name__ == "__main__":
    sys.exit(main())