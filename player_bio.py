"""NHL Player Biographical Data Fetcher.

This module fetches player biographical data from the NHL Web API
and stores it in a local SQLite database. It is designed to be:
- Rate limited (respects NHL API limits)
- Targeted (default collects player IDs from the shots table)
- Idempotent (safe to re-run, skips existing entries)
- Configurable (--range to manually specify IDs, --force-refresh to re-fetch)
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
PLAYER_BIO_REQUEST_DELAY_SECONDS = 0.2  # Conservative delay to avoid NHL API rate limits
PLAYER_BIO_HTTP_RETRY_TOTAL = 5
PLAYER_BIO_HTTP_RETRY_BACKOFF_FACTOR = 0.6
PLAYER_BIO_HTTP_RATE_LIMIT_STATUSES = {500, 502, 503, 504}  # 429 handled by our own retry logic
PLAYER_BIO_DB_COMMIT_INTERVAL = 50  # Commit every N players
PLAYER_BIO_HEARTBEAT_INTERVAL_SECONDS = 30
PLAYER_BIO_DEFAULT_MAX_WORKERS = 1  # Single worker is safer for rate-limited APIs
PLAYER_BIO_DEFAULT_TIMEOUT_SECONDS = 10

# State tracking table name
PLAYER_BIO_STATE_TABLE = "player_bio_fetch_state"

# Player bio table name
PLAYER_BIO_TABLE = "players"


# Thread-local HTTP session with retry adapter (mirrors Main.py pattern)
_PLAYER_BIO_HTTP_SESSION = requests.Session()
_PLAYER_BIO_THREAD_LOCAL = threading.local()


def _build_player_bio_retry_adapter() -> HTTPAdapter:
    """Minimal adapter — no automatic retries on status codes (we handle that ourselves)."""
    retry = Retry(
        total=0,  # Don't auto-retry anything — our code manages retries
        connect=0,
        read=0,
        status=0,
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
    """Thread-safe rate limiter for player bio API requests.

    Automatically adapts: when a 429 is encountered, the delay is doubled.
    On success, it gradually relaxes back toward the configured minimum.
    """

    def __init__(self, minimum_interval_seconds: float) -> None:
        self._configured_minimum = max(0.0, minimum_interval_seconds)
        self._current_interval = max(0.0, minimum_interval_seconds)
        self._lock = threading.Lock()
        self._next_allowed_time = 0.0

    def acquire(self) -> None:
        if self._current_interval <= 0:
            return

        with self._lock:
            now = time.monotonic()
            wait = self._next_allowed_time - now
            if wait > 0:
                self._next_allowed_time += self._current_interval
            else:
                wait = 0.0
                self._next_allowed_time = now + self._current_interval

        if wait > 0:
            time.sleep(wait)

    def report_success(self) -> None:
        """Called after a successful API response. Gradually relaxes delay."""
        with self._lock:
            # Relax toward configured minimum
            self._current_interval = max(
                self._configured_minimum,
                self._current_interval * 0.9,
            )

    def report_throttled(self) -> None:
        """Called after a 429 response. Doubles the delay (capped at 30s)."""
        with self._lock:
            self._current_interval = min(30.0, self._current_interval * 2)
            logging.warning(
                "Rate limit hit — backing off to %.1fs between requests",
                self._current_interval,
            )


_PLAYER_BIO_RATE_LIMITER = _PlayerBioRateLimiter(PLAYER_BIO_REQUEST_DELAY_SECONDS)


def configure_player_bio_rate_limit(minimum_interval_seconds: float) -> None:
    """Configure the minimum interval between player bio API requests."""
    _PLAYER_BIO_RATE_LIMITER._configured_minimum = max(0.0, minimum_interval_seconds)


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
@dataclass
class PlayerBioConfig:
    """Runtime settings for player bio fetching."""
    start_player_id: int = 0    # 0 = collect IDs from shots table
    end_player_id: int = 0      # 0 = collect IDs from shots table
    timeout_seconds: int = PLAYER_BIO_DEFAULT_TIMEOUT_SECONDS
    max_workers: int = PLAYER_BIO_DEFAULT_MAX_WORKERS
    request_delay_seconds: float = PLAYER_BIO_REQUEST_DELAY_SECONDS
    db_path: str = "hockey_data.db"
    force_refresh: bool = False  # If True, re-fetch all players even if already stored


def build_web_player_landing_url(player_id: int) -> str:
    """Build the NHL Web API player landing page URL."""
    return f"{WEB_API_BASE_URL}/player/{player_id}/landing"


def _fetch_player_bio_json(url: str, timeout_seconds: int) -> tuple[dict | None, int | None]:
    """
    Fetch player bio JSON from NHL API with rate limiting and retries.
    Returns (parsed_json, status_code).
    On rate-limit (429), returns (None, 429) so the caller can retry later.
    On 404, returns (None, 404) — ID doesn't exist.
    On other errors, returns (None, status_code) for diagnostics.
    """
    try:
        _PLAYER_BIO_RATE_LIMITER.acquire()
        response = _get_player_bio_http_session().get(url, timeout=timeout_seconds)
        status = response.status_code
        if status == 429:
            _PLAYER_BIO_RATE_LIMITER.report_throttled()
            return None, 429
        if status == 404:
            _PLAYER_BIO_RATE_LIMITER.report_success()
            return None, 404
        response.raise_for_status()
        _PLAYER_BIO_RATE_LIMITER.report_success()
        return response.json(), status
    except requests.exceptions.RetryError as exc:
        logging.warning("Retry exhausted for %s: %s", url, exc)
        _PLAYER_BIO_RATE_LIMITER.report_throttled()
        return None, 429
    except (requests.RequestException, ValueError) as exc:
        logging.warning("Failed to fetch player bio from %s: %s", url, exc)
        return None, None


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


def fetch_single_player_bio(player_id: int, timeout_seconds: int) -> tuple[dict | None, int | None]:
    """Fetch and parse a single player's bio.
    Returns (parsed_bio, status_code).
    Status 429 means rate-limited and worth retrying; 404 means ID doesn't exist.
    """
    url = build_web_player_landing_url(player_id)
    payload, status = _fetch_player_bio_json(url, timeout_seconds)
    if payload is None:
        return None, status
    return parse_player_bio(payload), status


def _collect_player_ids_from_shots(db_path: str) -> list[int]:
    """Collect distinct player IDs (shooters + goalies) from the shots table."""
    with sqlite3.connect(db_path) as connection:
        cursor = connection.cursor()
        cursor.execute("""
            SELECT DISTINCT id FROM (
                SELECT Shooter_ID AS id FROM shots WHERE Shooter_ID IS NOT NULL
                UNION
                SELECT Goalie_ID AS id FROM shots WHERE Goalie_ID IS NOT NULL
            )
            ORDER BY id
        """)
        ids = [row[0] for row in cursor.fetchall()]
    if not ids:
        logging.warning(
            "No player IDs found in shots table. "
            "Run Main.py to scrape shot data first, or use --range to specify a manual range."
        )
    else:
        logging.info("Collected %d unique player IDs from shots table (range %d–%d)", len(ids), ids[0], ids[-1])
    return ids


def _fetch_and_store_player_bio(
    player_id: int,
    timeout_seconds: int,
    db_path: str,
    force_refresh: bool,
) -> tuple[bool, str]:
    """
    Fetch and store a single player's bio.
    Returns (success, status) where status is one of:
    'fetched', 'skipped_existing', 'not_found', 'rate_limited', 'error'
    """
    # Check if already exists (unless force_refresh)
    if not force_refresh and _player_exists_in_db(db_path, player_id):
        return True, "skipped_existing"

    bio, status_code = fetch_single_player_bio(player_id, timeout_seconds)
    if bio is None:
        if status_code == 429:
            return False, "rate_limited"
        if status_code == 404:
            return True, "not_found"  # Not an error - player ID just doesn't exist
        return False, "error"

    upsert_player_bio(db_path, bio)
    return True, "fetched"


def run_player_bio_fetch(config: PlayerBioConfig) -> tuple[int, int, int, int]:
    """
    Orchestrates fetching player bios. By default, collects player IDs from the
    shots table. If start_player_id and end_player_id are set (non-zero), fetches
    all IDs in that range instead.
    Returns (total_fetched, total_skipped, total_not_found, total_errors).
    """
    initialize_player_bio_database(config.db_path)
    configure_player_bio_rate_limit(config.request_delay_seconds)

    # Determine which player IDs to fetch
    if config.start_player_id > 0 and config.end_player_id > 0:
        if config.start_player_id > config.end_player_id:
            logging.error("Invalid range: start %d > end %d", config.start_player_id, config.end_player_id)
            return 0, 0, 0, 0
        player_ids = list(range(config.start_player_id, config.end_player_id + 1))
        logging.info("Fetching bios for %d IDs in range %d–%d", len(player_ids), config.start_player_id, config.end_player_id)
    else:
        player_ids = _collect_player_ids_from_shots(config.db_path)
        if not player_ids:
            return 0, 0, 0, 0

    # When not forcing refresh, skip IDs already in the players table
    if not config.force_refresh:
        with sqlite3.connect(config.db_path) as connection:
            cursor = connection.cursor()
            placeholders = ",".join("?" for _ in player_ids)
            cursor.execute(
                f"SELECT player_id FROM {PLAYER_BIO_TABLE} WHERE player_id IN ({placeholders})",
                player_ids,
            )
            existing = {row[0] for row in cursor.fetchall()}

        skipped = [pid for pid in player_ids if pid in existing]
        to_fetch = [pid for pid in player_ids if pid not in existing]
        logging.info(
            "Skipping %d already-stored players, fetching %d new players",
            len(skipped), len(to_fetch),
        )
        player_ids = to_fetch

    total_players = len(player_ids)
    if total_players == 0:
        logging.info("All player bios already exist in the database. Nothing to do.")
        return 0, 0, 0, 0

    logging.info(
        "Fetching player bios for %d players with %d workers",
        total_players, config.max_workers,
    )

    total_fetched = 0
    total_skipped = 0
    total_not_found = 0
    total_errors = 0
    total_rate_limited = 0
    last_heartbeat = time.monotonic()

    def log_heartbeat(processed: int, current_id: int) -> None:
        nonlocal last_heartbeat
        now = time.monotonic()
        if now - last_heartbeat >= PLAYER_BIO_HEARTBEAT_INTERVAL_SECONDS:
            logging.info(
                "Player bio fetch progress: %d/%d processed, fetched=%d, skipped=%d, not_found=%d, rate_limited=%d, errors=%d, current_id=%d",
                processed,
                total_players,
                total_fetched,
                total_skipped,
                total_not_found,
                total_rate_limited,
                total_errors,
                current_id,
            )
            last_heartbeat = now

    with sqlite3.connect(config.db_path) as connection:
        cursor = connection.cursor()
        _save_fetch_state(
            cursor,
            last_completed_player_id=0,
            last_attempted_player_id=0,
            total_fetched=0,
            total_skipped_existing=0,
            total_not_found=0,
            status="running",
        )
        connection.commit()

        try:
            if config.max_workers <= 1 or total_players <= 1:
                # Sequential processing with retry on rate-limit
                remaining = list(player_ids)
                retry_delay = config.request_delay_seconds * 4  # Start with a more conservative backoff
                max_retries = 3

                for attempt in range(max_retries + 1):
                    if not remaining:
                        break

                    if attempt > 0:
                        logging.info(
                            "Retry attempt %d/%d for %d rate-limited players (waiting %.1fs)...",
                            attempt, max_retries, len(remaining), retry_delay,
                        )
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff

                    still_remaining = []
                    for idx, player_id in enumerate(remaining, 1):
                        success, status = _fetch_and_store_player_bio(
                            player_id, config.timeout_seconds, config.db_path, config.force_refresh
                        )
                        if status == "rate_limited":
                            still_remaining.append(player_id)
                        elif status == "fetched":
                            total_fetched += 1
                        elif status == "skipped_existing":
                            total_skipped += 1
                        elif status == "not_found":
                            total_not_found += 1
                        else:
                            total_errors += 1

                        log_heartbeat(len(remaining), player_id)

                    remaining = still_remaining
                    total_rate_limited += len(remaining)

                if remaining:
                    logging.warning(
                        "Gave up on %d players after %d retries due to rate limiting: %s",
                        len(remaining), max_retries, remaining[:10],
                    )
                    total_errors += len(remaining)

                    # Final state update
            final_status = "complete" if total_errors == 0 else "partial"
            _save_fetch_state(
                cursor,
                last_completed_player_id=player_ids[-1] if player_ids else 0,
                last_attempted_player_id=player_ids[-1] if player_ids else 0,
                total_fetched=total_fetched,
                total_skipped_existing=total_skipped,
                total_not_found=total_not_found,
                status=final_status,
            )
            connection.commit()

        except Exception:
            _save_fetch_state(
                cursor,
                last_completed_player_id=0,
                last_attempted_player_id=player_ids[-1] if player_ids else 0,
                total_fetched=total_fetched,
                total_skipped_existing=total_skipped,
                total_not_found=total_not_found,
                status="failed",
            )
            connection.commit()
            raise

    logging.info(
        "Player bio fetch complete: fetched=%d, skipped_existing=%d, not_found=%d, rate_limited=%d, errors=%d",
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
    parser.add_argument("--start", type=int, default=0, metavar="ID",
                        help="Start of player ID range. Default: collect IDs from shots table.")
    parser.add_argument("--end", type=int, default=0, metavar="ID",
                        help="End of player ID range. Requires --start.")
    parser.add_argument("--force-refresh", action="store_true", help="Re-fetch all players even if already in database")
    parser.add_argument("--timeout", type=int, default=PLAYER_BIO_DEFAULT_TIMEOUT_SECONDS, help="HTTP timeout in seconds")
    parser.add_argument("--max-workers", type=int, default=PLAYER_BIO_DEFAULT_MAX_WORKERS, help="Max concurrent workers")
    parser.add_argument("--request-delay", type=float, default=PLAYER_BIO_REQUEST_DELAY_SECONDS, help="Min delay between requests (seconds)")
    parser.add_argument("--db-path", default="hockey_data.db", help="SQLite database path")
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

    if (args.start > 0) != (args.end > 0):
        logging.error("Both --start and --end must be provided together, or neither.")
        return

    config = PlayerBioConfig(
        start_player_id=args.start,
        end_player_id=args.end,
        timeout_seconds=args.timeout,
        max_workers=args.max_workers,
        request_delay_seconds=args.request_delay,
        db_path=args.db_path,
        force_refresh=args.force_refresh,
    )

    run_player_bio_fetch(config)


def fetch_player_bios(config: PlayerBioConfig) -> tuple[int, int]:
    """
    Wrapper function for use by the orchestrator.
    Returns (players_processed, bios_inserted).
    """
    total_fetched, total_skipped, total_not_found, total_errors = run_player_bio_fetch(config)
    return total_fetched, total_fetched - total_skipped - total_not_found - total_errors


if __name__ == "__main__":
    main()