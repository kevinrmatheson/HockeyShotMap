# Copilot Instructions for HockeyShotMap

## Repository purpose
- Python project for collecting NHL shot events into SQLite (`Main.py`), generating derived metrics (`Metrics.py`), enriching players (`player_bio.py`), and serving a read-only Flask dashboard (`visualization/`).
- `Scraper.py` is an explicit prototype scaffold and is not part of the active ingestion pipeline.

## High-value files
- `Main.py`: primary ETL and DB schema initialization.
- `Metrics.py`: xG/xSV model training/scoring and derived tables.
- `player_bio.py`: player bio fetcher into SQLite.
- `visualization/query_engine.py`: SQL-backed dashboard queries.
- `visualization/app.py`: Flask app entrypoint.
- `tests/`: unit tests (`unittest` style).

## Environment and setup
1. Install dependencies before running tests or scripts:
   - `pip install -r requirements.txt`
2. Main local database artifact is `hockey_data.db` (already gitignored).
3. No dedicated lint config (`pyproject.toml`, `setup.cfg`, `tox.ini`, `.flake8`, `pytest.ini`) is currently present.

## Validation workflow
- Primary test command:
  - `python -m unittest discover -s tests -p "test_*.py"`
- For targeted checks:
  - `python -m unittest tests.test_main`
  - `python -m unittest tests.test_visualization`
  - `python -m unittest tests.test_metrics`

## Known errors encountered during onboarding
1. **Missing dependencies in fresh environment**
   - Error: `ModuleNotFoundError: No module named 'numpy'` when running tests.
   - Workaround: run `pip install -r requirements.txt` first.

2. **Current baseline metrics test failure (pre-existing)**
   - Error from `tests/test_metrics.py`: `sqlite3.OperationalError: no such column: team`
   - Trace path: `Metrics.run_metrics_refresh` -> `_compute_team_strength_metrics`.
   - Workaround for unrelated tasks: validate with non-metrics suites (`tests.test_main`, `tests.test_visualization`) and treat `test_metrics` failure as known baseline until fixed in `Metrics.py`.

## Change guidance for agents
- Keep changes surgical; this repo uses large single-file modules, so avoid broad refactors unless required.
- Preserve CLI behavior and argparse flags in `Main.py`, `Metrics.py`, and `player_bio.py`.
- Prefer adding tests in existing `unittest` style and colocate in `tests/`.
- If changing SQL/table logic, verify compatibility with both ingestion (`Main.py`) and query/metrics consumers (`Metrics.py`, `visualization/query_engine.py`).
- Avoid using full-season scrape runs for quick validation; they are network-heavy and slow.
