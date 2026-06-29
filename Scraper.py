"""Prototype multi-league player-level scraper scaffold.

This module is intentionally separate from the active shot-map ETL in Main.py.
It exists as an incubation area for future player-level and cross-league data
projects, for example:

- Hall of Fame style player ranking research.
- Draft prospect modeling across junior/college/international leagues.
- Rich historical player profile datasets beyond shot events.

Current status:
- Not production-ready.
- Not integrated into scheduled/data-pipeline runs.
- Kept in this repository for now, with an easy path to split into its own project later.
"""

from __future__ import annotations

from typing import Final

# Map short league codes to HockeyDB league names.
LEAGUE_NAME_LOOKUP: Final[dict[str, str]] = {
    "OHL": "Ontario Hockey League",
    "NHL": "National Hockey League",
    "AHL": "American Hockey League",
    "ECHL": "ECHL",
    "FHL": "Federal Prospects Hockey League",
    "LNAH": "Ligue Nord-Americaine de Hockey",
    "SPHL": "Southern Professional Hockey League",
    "QMJHL": "Quebec Major Junior Hockey League",
    "WHL": "Western Hockey League",
    "CWUAA": "CIS - Canada West Universities Athletic Assn",
    "OUAA": "CIS - Ontario University Athletic Association",
    "NCAA": "National Collegiate Athletic Association",
    "AHA": "NCAA - Atlantic Hockey Association - Div. 1",
    "Big-10": "NCAA - Big 10 - Div. 1",
    "CCC": "NCAA - Commonwealth Coast Conference",
    "ECAC": "NCAA - ECAC - Div. 1",
    "H-East": "NCAA - Hockey East - Div. 1",
    "MASCAC": "NCAA - MASCAC",
    "MIAC": "NCAA - Minnesota Intercollegiate Athletic Conf.",
    "NCHC": "NCAA - National Collegiate Hockey Conf. - Div. 1",
    "NESCAC": "NCAA - NESCAC",
    "NEHC": "NCAA - New England Hockey Conference",
    "NE-10": "NCAA - Northeast 10",
    "NCHA": "NCAA - Northern Collegiate Hockey Association",
    "SUNYAC": "NCAA - SUNYAC",
    "UCHC": "NCAA - United Collegiate Hockey Conference",
    "WCHA": "NCAA - Western Collegiate Hockey Assn. - Div. 1",
    "WIAC": "NCAA - Wisconsin Intercollegiate Athletic Conf.",
    "AJHL": "Alberta Junior Hockey League",
    "BCHL": "British Columbia Hockey League",
    "MHL": "Maritime Hockey League",
    "NOJHL": "Northern Ontario Junior Hockey League",
    "OJHL": "Ontario Junior Hockey League",
    "QJAAAHL": "Quebec Junior Hockey League",
    "SIJHL": "Superior International Jr Hockey League",
}


def league_year_scrape(league_code: str, year: int, output_path: str | None = None) -> None:
    """Plan-level entry point for a future league-season scrape.

    Intended flow:
    1. Resolve league from LEAGUE_NAME_LOOKUP.
    2. Fetch league season table.
    3. Discover team links for the selected season.
    4. Call team_scrape for each team.
    5. Persist normalized player rows.
    """
    _ = (league_code, year, output_path)
    raise NotImplementedError(
        "Scraper prototype: league_year_scrape is a design scaffold and is not implemented yet."
    )


def team_scrape(team_page_url: str, output_path: str | None = None) -> None:
    """Placeholder for one-team roster/stat extraction logic."""
    _ = (team_page_url, output_path)
    raise NotImplementedError(
        "Scraper prototype: team_scrape is a design scaffold and is not implemented yet."
    )


def player_scrape(player_page_url: str, output_path: str | None = None) -> None:
    """Placeholder for detailed player profile extraction.

    Planned examples: bio, league history, production rates, awards, and other
    career-level features useful for ranking/modeling workflows.
    """
    _ = (player_page_url, output_path)
    raise NotImplementedError(
        "Scraper prototype: player_scrape is a design scaffold and is not implemented yet."
    )


if __name__ == "__main__":
    print(
        "Scraper.py is an intentional prototype scaffold for future player-level/multi-league "
        "research and is not wired into the active Main.py pipeline."
    )