"""NHL Player Biographical Data Fetcher.

This module fetches player biographical data from the NHL Web API
and stores it in a local SQLite database. It is designed to be:
- Rate limited (respects NHL API limits)
- Resumable (tracks last successfully fetched player ID)
- Idempotent (safe to re-run, uses INSERT OR REPLACE)
- Configurable (can force full re-fetch if needed)
"""

import argparse
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Reuse the same base URL and HTTP constants from Main.py
WEB_API_BASE_URL = "https://api-web.nhle.com/v1"

# Player bio fetching defaults
PLAYER_BIO_REQUEST_DELAY_SECONDS = 0.1  # Slightly more conservative than game scraping
PLAYER_BIO_HTTP_RETRY_TOTAL = 5
PLAYER_BIO_HTTP_RETRY_BACKOFF_FACTOR = 0.6
PLAYER_BIO_HTTP_RATE_LIMIT_STATUSES = {429, 500, 502, 503, 504}
PLAYER_BIO_DB_COMMIT_INTERVAL = 50  # Commit every N players
PLAYER_BIO_HEARTBEAT_INTERVAL_SECONDS = 30
PLAYER_BIO_DEFAULT_MAX_WORKERS = 2  # Conservative for player bios
PLAYER_BIO_DEFAULT_TIMEOUT_SECONDS = 10

# State tracking table name
PLAYER_BIO_STATE_TABLE = "player_bio_fetch_state"

# Player bio table name
PLAYER_BIO_TABLE = "players"


# Thread-local HTTP session with retry adapter (mirrors Main.py pattern)
_PLAYER_BIO_HTTP_SESSION = requests.Session()
_PLAYER_BIO_THREAD_LOCAL = threading.local()


def _build_player_bio_retry_adapter() -> HTTPAdapter:
    retry = Retry(
        total=PLAYER_BIO_HTTP_RETRY_TOTAL,
        connect=PLAYER_BIO_HTTP_RETRY_TOTAL,
        read=PLAYER_BIO_HTTP_RETRY_TOTAL,
        status=PLAYER_BIO_HTTP_RETRY_TOTAL,
        backoff_factor=PLAYER_BIO_HTTP_RETRY_BACKOFF_FACTOR,
        status_forcelist=PLAYER_BIO_HTTP_RATE_LIMIT_STATUSES,
        allowed_methods=frozenset({"GET"}),
        raise_on_status=False,
    )
    return HTTPAdapter(
        max_retries=retry,
        pool_connections=PLAYER_BIO_DEFAULT_MAX_WORKERS * 2,
        pool_maxsize=PLAYER_BIO_DEFAULT_MAX_WORKERS * 2,
    )


_PLAYER_BIO_HTTP_SESSION.mount("http://", _build_player_bio_retry_adapter())
_PLAYER_BIO_HTTP_SESSION.mount("https://", _build_player_bio_retry_adapter())


class _PlayerBioRateLimiter:
    """Thread-safe rate limiter for player bio API requests."""

    def __init__(self, minimum_interval_seconds: float) -> None:
        self.minimum_interval_seconds = max(0.0, minimum_interval_seconds)
        self._lock = threading.Lock()
        self._next_allowed_time = 0.0

    def acquire(self) -> None:
        if self.minimum_interval_seconds <= 0:
            return

        with self._lock:
            now = time.monotonic()
            if now < self._next_allowed_time:
                sleep_seconds = self._next_allowed_time - now
                self._next_allowed_time += self.minimum_interval_seconds
            else:
                sleep_seconds = 0.0
                self._next_allowed_time = now + self.minimum_interval_seconds

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)


_PLAYER_BIO_RATE_LIMITER = _PlayerBioRateLimiter(PLAYER_BIO_REQUEST_DELAY_SECONDS)


def configure_player_bio_rate_limit(minimum_interval_seconds: float) -> None:
    """Configure the minimum interval between player bio API requests."""
    _PLAYER_BIO_RATE_LIMITER.minimum_interval_seconds = max(0.0, minimum_interval_seconds)


def _get_player_bio_http_session() -> requests.Session:
    """Get a thread-local HTTP session with retry adapter."""
    if threading.current_thread() is threading.main_thread():
        return _PLAYER_BIO_HTTP_SESSION

    session = getattr(_PLAYER_BIO_THREAD_LOCAL, "session", None)
    if session is None:
        session = requests.Session()
        adapter = _build_player_bio_retry_adapter()
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        _PLAYER_BIO_THREAD_LOCAL.session = session
    return session


@dataclass
class PlayerBioConfig:
    """Runtime settings for player bio fetching."""
    start_player_id: int = 1
    end_player_id: int = 9999999  # Practical upper bound for NHL player IDs
    timeout_seconds: int = PLAYER_BIO_DEFAULT_TIMEOUT_SECONDS
    max_workers: int = PLAYER_BIO_DEFAULT_MAX_WORKERS
    request_delay_seconds: float = PLAYER_BIO_REQUEST_DELAY_SECONDS
    db_path: str = "hockey_shots.db"
    force_refresh: bool = False  # If True, re-fetch all players even if already stored


def build_web_player_landing_url(player_id: int) -> str:
    """Build the NHL Web API player landing page URL."""
    return f"{WEB_API_BASE_URL}/player/{player_id}/landing"


def _fetch_player_bio_json(url: str, timeout_seconds: int) -> dict | None:
    """Fetch player bio JSON from NHL API with rate limiting and retries."""
    try:
        _PLAYER_BIO_RATE_LIMITER.acquire()
        response = _get_player_bio_http_session().get(url, timeout=timeout_seconds)
        if response.status_code == 404:
            # Player ID doesn't exist - this is normal for gaps in ID space
            return None
        response.raise_for_status()
        return response.json()
    except (requests.RequestException, ValueError) as exc:
        logging.warning("Failed to fetch player bio from %s: %s", url, exc)
        return None


def _normalize_text(value: object | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _coerce_int(value: object | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return None


def parse_player_bio(payload: dict) -> dict | None:
    """Extract biographical fields from player landing payload."""
    if not isinstance(payload, dict):
        return None

    player_id = payload.get("playerId")
    if player_id is None:
        return None

    try:
        player_id = int(player_id)
    except (TypeError, ValueError):
        return None

    # Extract nested name objects (they have "default" key for English)
    first_name_obj = payload.get("firstName", {})
    last_name_obj = payload.get("lastName", {})
    first_name = _normalize_text(first_name_obj.get("default") if isinstance(first_name_obj, dict) else first_name_obj)
    last_name = _normalize_text(last_name_obj.get("default") if isinstance(last_name_obj, dict) else last_name_obj)

    # Birth place objects
    birth_city_obj = payload.get("birthCity", {})
    birth_state_obj = payload.get("birthStateProvince", {})
    birth_city = _normalize_text(birth_city_obj.get("default") if isinstance(birth_city_obj, dict) else birth_city_obj)
    birth_state = _normalize_text(birth_state_obj.get("default") if isinstance(birth_state_obj, dict) else birth_state_obj)

    # Draft details
    draft_details = payload.get("draftDetails", {})
    if not isinstance(draft_details, dict):
        draft_details = {}

    full_name = f"{first_name or ''} {last_name or ''}".strip()

    return {
        "player_id": player_id,
        "first_name": first_name,
        "last_name": last_name,
        "full_name": full_name or None,
        "height_inches": _coerce_int(payload.get("heightInInches")),
        "height_cm": _coerce_int(payload.get("heightInCentimeters")),
        "weight_lbs": _coerce_int(payload.get("weightInPounds")),
        "weight_kg": _coerce_int(payload.get("weightInKilograms")),
        "birth_date": _normalize_text(payload.get("birthDate")),
        "birth_city": birth_city,
        "birth_state_province": birth_state,
        "birth_country": _normalize_text(payload.get("birthCountry")),
        "shoots_catches": _normalize_text(payload.get("shootsCatches")),
        "position": _normalize_text(payload.get("position")),
        "draft_year": _coerce_int(draft_details.get("year")),
        "draft_team": _normalize_text(draft_details.get("teamAbbrev")),
        "draft_round": _coerce_int(draft_details.get("round")),
        "draft_pick_in_round": _coerce_int(draft_details.get("pickInRound")),
        "draft_overall_pick": _coerce_int(draft_details.get("overallPick")),
    }


def initialize_player_bio_database(db_path: str) -> None:
    """Create player bio table and state tracking table if they don't exist."""
    with sqlite3.connect(db_path) as connection:
        cursor = connection.cursor()

        # Player biographical data table
        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {PLAYER_BIO_TABLE} (
                player_id INTEGER PRIMARY KEY,
                first_name TEXT,
                last_name TEXT,
                full_name TEXT,
                height_inches INTEGER,
                height_cm INTEGER,
                weight_lbs INTEGER,
                weight_kg INTEGER,
                birth_date TEXT,
                birth_city TEXT,
                birth_state_province TEXT,
                birth_country TEXT,
                shoots_catches TEXT,
                position TEXT,
                draft_year INTEGER,
                draft_team TEXT,
                draft_round INTEGER,
                draft_pick_in_round INTEGER,
                draft_overall_pick INTEGER,
                fetched_at TEXT NOT NULL
            )
            """
        )

        # Index for common queries
        cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_players_last_name ON {PLAYER_BIO_TABLE}(last_name)")
        cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_players_position ON {PLAYER_BIO_TABLE}(position)")
        cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_players_shoots_catches ON {PLAYER_BIO_TABLE}(shoots_catches)")
        cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_players_draft_year ON {PLAYER_BIO_TABLE}(draft_year)")

        # State tracking table for resumable fetching
        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {PLAYER_BIO_STATE_TABLE} (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                last_completed_player_id INTEGER NOT NULL DEFAULT 0,
                last_attempted_player_id INTEGER NOT NULL DEFAULT 0,
                total_fetched INTEGER NOT NULL DEFAULT 0,
                total_skipped_existing INTEGER NOT NULL DEFAULT 0,
                total_not_found INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'idle',
                updated_at TEXT NOT NULL
            )
            """
        )

        # Initialize state row if it doesn't exist
        cursor.execute(
            f"""
            INSERT OR IGNORE INTO {PLAYER_BIO_STATE_TABLE} (id, last_completed_player_id, last_attempted_player_id, status, updated_at)
            VALUES (1, 0, 0, 'idle', ?)
            """,
            (datetime.now(UTC).isoformat(timespec="seconds"),),
        )

        connection.commit()


def _load_fetch_state(db_path: str) -> dict:
    """Load the current fetch state from the database."""
    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            f"SELECT * FROM {PLAYER_BIO_STATE_TABLE} WHERE id = 1"
        ).fetchone()
    return dict(row) if row else {}


def _save_fetch_state(
    cursor: sqlite3.Cursor,
    last_completed_player_id: int,
    last_attempted_player_id: int,
    total_fetched: int,
    total_skipped_existing: int,
    total_not_found: int,
    status: str,
) -> None:
    """Save the fetch state to the database."""
    updated_at = datetime.now(UTC).isoformat(timespec="seconds")
    cursor.execute(
        f"""
        UPDATE {PLAYER_BIO_STATE_TABLE}
        SET last_completed_player_id = ?,
            last_attempted_player_id = ?,
            total_fetched = ?,
            total_skipped_existing = ?,
            total_not_found = ?,
            status = ?,
            updated_at = ?
        WHERE id = 1
        """,
        (
            last_completed_player_id,
            last_attempted_player_id,
            total_fetched,
            total_skipped_existing,
            total_not_found,
            status,
            updated_at,
        ),
    )


def _player_exists_in_db(db_path: str, player_id: int) -> bool:
    """Check if a player bio already exists in the database."""
    with sqlite3.connect(db_path) as connection:
        cursor = connection.cursor()
        cursor.execute(
            f"SELECT 1 FROM {PLAYER_BIO_TABLE} WHERE player_id = ? LIMIT 1",
            (player_id,),
        )
        return cursor.fetchone() is not None


def upsert_player_bio(db_path: str, bio: dict) -> bool:
    """Insert or update player biographical data. Returns True if inserted/updated."""
    if not bio or bio.get("player_id") is None:
        return False

    fetched_at = datetime.now(UTC).isoformat(timespec="seconds")

    with sqlite3.connect(db_path) as connection:
        cursor = connection.cursor()
        cursor.execute(
            f"""
            INSERT INTO {PLAYER_BIO_TABLE} (
                player_id, first_name, last_name, full_name,
                height_inches, height_cm, weight_lbs, weight_kg,
                birth_date, birth_city, birth_state_province, birth_country,
                shoots_catches, position,
                draft_year, draft_team, draft_round, draft_pick_in_round, draft_overall_pick,
                fetched_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(player_id) DO UPDATE SET
                first_name = excluded.first_name,
                last_name = excluded.last_name,
                full_name = excluded.full_name,
                height_inches = excluded.height_inches,
                height_cm = excluded.height_cm,
                weight_lbs = excluded.weight_lbs,
                weight_kg = excluded.weight_kg,
                birth_date = excluded.birth_date,
                birth_city = excluded.birth_city,
                birth_state_province = excluded.birth_state_province,
                birth_country = excluded.birth_country,
                shoots_catches = excluded.shoots_catches,
                position = excluded.position,
                draft_year = excluded.draft_year,
                draft_team = excluded.draft_team,
                draft_round = excluded.draft_round,
                draft_pick_in_round = excluded.draft_pick_in_round,
                draft_overall_pick = excluded.draft_overall_pick,
                fetched_at = excluded.fetched_at
            """,
            (
                bio["player_id"],
                bio.get("first_name"),
                bio.get("last_name"),
                bio.get("full_name"),
                bio.get("height_inches"),
                bio.get("height_cm"),
                bio.get("weight_lbs"),
                bio.get("weight_kg"),
                bio.get("birth_date"),
                bio.get("birth_city"),
                bio.get("birth_state_province"),
                bio.get("birth_country"),
                bio.get("shoots_catches"),
                bio.get("position"),
                bio.get("draft_year"),
                bio.get("draft_team"),
                bio.get("draft_round"),
                bio.get("draft_pick_in_round"),
                bio.get("draft_overall_pick"),
                fetched_at,
            ),
        )
        connection.commit()
    return True


def fetch_single_player_bio(player_id: int, timeout_seconds: int) -> dict | None:
    """Fetch and parse a single player's bio. Returns None if not found or error."""
    url = build_web_player_landing_url(player_id)
    payload = _fetch_player_bio_json(url, timeout_seconds)
    if payload is None:
        return None
    return parse_player_bio(payload)


def _fetch_and_store_player_bio(
    player_id: int,
    timeout_seconds: int,
    db_path: str,
    force_refresh: bool,
) -> tuple[bool, str]:
    """
    Fetch and store a single player's bio.
    Returns (success, status) where status is one of: 'fetched', 'skipped_existing', 'not_found', 'error'
    """
    # Check if already exists (unless force_refresh)
    if not force_refresh and _player_exists_in_db(db_path, player_id):
        return True, "skipped_existing"

    bio = fetch_single_player_bio(player_id, timeout_seconds)
    if bio is None:
        return True, "not_found"  # Not an error - player ID just doesn't exist

    upsert_player_bio(db_path, bio)
    return True, "fetched"


def run_player_bio_fetch(config: PlayerBioConfig) -> tuple[int, int, int, int]:
    """
    Orchestrates fetching player bios for a range of player IDs.
    Returns (total_fetched, total_skipped, total_not_found, total_errors).
    """
    initialize_player_bio_database(config.db_path)
    configure_player_bio_rate_limit(config.request_delay_seconds)

    # Load previous state to resume
    state = _load_fetch_state(config.db_path)
    last_completed = state.get("last_completed_player_id", 0)

    # Determine start point
    if config.force_refresh:
        start_id = config.start_player_id
        logging.info("Force refresh enabled: starting from player ID %d", start_id)
    else:
        start_id = max(config.start_player_id, last_completed + 1)
        if start_id > config.start_player_id:
            logging.info("Resuming from player ID %d (last completed: %d)", start_id, last_completed)

    if start_id > config.end_player_id:
        logging.info("No players to fetch: start_id %d > end_player_id %d", start_id, config.end_player_id)
        return 0, 0, 0, 0

    player_ids = list(range(start_id, config.end_player_id + 1))
    total_players = len(player_ids)

    logging.info(
        "Fetching player bios for %d players (IDs %d to %d) with %d workers",
        total_players,
        start_id,
        config.end_player_id,
        config.max_workers,
    )

    total_fetched = 0
    total_skipped = 0
    total_not_found = 0
    total_errors = 0
    last_heartbeat = time.monotonic()

    def log_heartbeat(processed: int, current_id: int) -> None:
        nonlocal last_heartbeat
        now = time.monotonic()
        if now - last_heartbeat >= PLAYER_BIO_HEARTBEAT_INTERVAL_SECONDS:
            logging.info(
                "Player bio fetch progress: %d/%d processed, fetched=%d, skipped=%d, not_found=%d, errors=%d, current_id=%d",
                processed,
                total_players,
                total_fetched,
                total_skipped,
                total_not_found,
                total_errors,
                current_id,
            )
            last_heartbeat = now

    with sqlite3.connect(config.db_path) as connection:
        cursor = connection.cursor()
        _save_fetch_state(
            cursor,
            last_completed_player_id=start_id - 1,
            last_attempted_player_id=start_id - 1,
            total_fetched=0,
            total_skipped_existing=0,
            total_not_found=0,
            status="running",
        )
        connection.commit()

        try:
            if config.max_workers <= 1 or total_players <= 1:
                # Sequential processing
                for idx, player_id in enumerate(player_ids, 1):
                    success, status = _fetch_and_store_player_bio(
                        player_id, config.timeout_seconds, config.db_path, config.force_refresh
                    )
                    if not success:
                        total_errors += 1
                    elif status == "fetched":
                        total_fetched += 1
                    elif status == "skipped_existing":
                        total_skipped += 1
                    elif status == "not_found":
                        total_not_found += 1

                    # Update state periodically
                    if idx % PLAYER_BIO_DB_COMMIT_INTERVAL == 0 or idx == total_players:
                        _save_fetch_state(
                            cursor,
                            last_completed_player_id=player_id,
                            last_attempted_player_id=player_id,
                            total_fetched=total_fetched,
                            total_skipped_existing=total_skipped,
                            total_not_found=total_not_found,
                            status="running",
                        )
                        connection.commit()

                    log_heartbeat(idx, player_id)

            else:
                # Parallel processing with ThreadPoolExecutor
                from concurrent.futures import ThreadPoolExecutor, as_completed

                with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
                    futures = {
                        executor.submit(
                            _fetch_and_store_player_bio,
                            player_id,
                            config.timeout_seconds,
                            config.db_path,
                            config.force_refresh,
                        ): player_id
                        for player_id in player_ids
                    }

                    completed_count = 0
                    for future in as_completed(futures):
                        player_id = futures[future]
                        completed_count += 1

                        try:
                            success, status = future.result()
                            if not success:
                                total_errors += 1
                            elif status == "fetched":
                                total_fetched += 1
                            elif status == "skipped_existing":
                                total_skipped += 1
                            elif status == "not_found":
                                total_not_found += 1
                        except Exception as exc:
                            logging.error("Error fetching player %d: %s", player_id, exc)
                            total_errors += 1

                        # Update state periodically
                        if completed_count % PLAYER_BIO_DB_COMMIT_INTERVAL == 0 or completed_count == total_players:
                            _save_fetch_state(
                                cursor,
                                last_completed_player_id=player_id,
                                last_attempted_player_id=player_id,
                                total_fetched=total_fetched,
                                total_skipped_existing=total_skipped,
                                total_not_found=total_not_found,
                                status="running",
                            )
                            connection.commit()

                        log_heartbeat(completed_count, player_id)

            # Final state update
            final_status = "complete" if total_errors == 0 else "partial"
            _save_fetch_state(
                cursor,
                last_completed_player_id=config.end_player_id,
                last_attempted_player_id=config.end_player_id,
                total_fetched=total_fetched,
                total_skipped_existing=total_skipped,
                total_not_found=total_not_found,
                status=final_status,
            )
            connection.commit()

        except Exception:
            _save_fetch_state(
                cursor,
                last_completed_player_id=start_id - 1,
                last_attempted_player_id=config.end_player_id,
                total_fetched=total_fetched,
                total_skipped_existing=total_skipped,
                total_not_found=total_not_found,
                status="failed",
            )
            connection.commit()
            raise

    logging.info(
        "Player bio fetch complete: fetched=%d, skipped_existing=%d, not_found=%d, errors=%d",
        total_fetched,
        total_skipped,
        total_not_found,
        total_errors,
    )

    return total_fetched, total_skipped, total_not_found, total_errors


def get_player_bio(db_path: str, player_id: int) -> dict | None:
    """Retrieve a player's bio from the local database."""
    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            f"SELECT * FROM {PLAYER_BIO_TABLE} WHERE player_id = ?",
            (player_id,),
        ).fetchone()
    return dict(row) if row else None


def get_players_by_team_draft(db_path: str, draft_team: str, draft_year: int | None = None) -> list[dict]:
    """Retrieve players drafted by a specific team (optionally in a specific year)."""
    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        if draft_year is not None:
            rows = connection.execute(
                f"SELECT * FROM {PLAYER_BIO_TABLE} WHERE draft_team = ? AND draft_year = ? ORDER BY draft_overall_pick",
                (draft_team.upper(), draft_year),
            ).fetchall()
        else:
            rows = connection.execute(
                f"SELECT * FROM {PLAYER_BIO_TABLE} WHERE draft_team = ? ORDER BY draft_year, draft_overall_pick",
                (draft_team.upper(),),
            ).fetchall()
    return [dict(row) for row in rows]


def get_fetch_status(db_path: str) -> dict:
    """Get the current fetch status/state."""
    state = _load_fetch_state(db_path)
    with sqlite3.connect(db_path) as connection:
        cursor = connection.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {PLAYER_BIO_TABLE}")
        total_players = cursor.fetchone()[0]
    state["total_players_in_db"] = total_players
    return state


def parse_args() -> argparse.Namespace:
    """CLI argument parser for player bio fetching."""
    parser = argparse.ArgumentParser(description="Fetch NHL player biographical data from NHL Web API.")
    parser.add_argument("--start-player-id", type=int, default=1, help="Starting player ID (default: 1)")
    parser.add_argument("--end-player-id", type=int, default=9999999, help="Ending player ID (default: 9999999)")
    parser.add_argument("--force-refresh", action="store_true", help="Re-fetch all players even if already in database")
    parser.add_argument("--timeout", type=int, default=PLAYER_BIO_DEFAULT_TIMEOUT_SECONDS, help="HTTP timeout in seconds")
    parser.add_argument("--max-workers", type=int, default=PLAYER_BIO_DEFAULT_MAX_WORKERS, help="Max concurrent workers")
    parser.add_argument("--request-delay", type=float, default=PLAYER_BIO_REQUEST_DELAY_SECONDS, help="Min delay between requests (seconds)")
    parser.add_argument("--db-path", default="hockey_shots.db", help="SQLite database path")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--status", action="store_true", help="Show current fetch status and exit")
    return parser.parse_args()


def main() -> None:
    """CLI entry point for player bio fetching."""
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")

    if args.status:
        status = get_fetch_status(args.db_path)
        print("Player Bio Fetch Status:")
        for key, value in status.items():
            print(f"  {key}: {value}")
        return

    config = PlayerBioConfig(
        start_player_id=args.start_player_id,
        end_player_id=args.end_player_id,
        timeout_seconds=args.timeout,
        max_workers=args.max_workers,
        request_delay_seconds=args.request_delay,
        db_path=args.db_path,
        force_refresh=args.force_refresh,
    )

    run_player_bio_fetch(config)


if __name__ == "__main__":
    main()