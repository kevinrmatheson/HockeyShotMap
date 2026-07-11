from __future__ import annotations

import argparse
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from .query_engine import (
   DEFAULT_BIN_SIZE,
   build_filters_from_args,
   get_filter_options,
   get_heatmap_bins,
   get_summary,
   latest_season,
)


def create_app(db_path: str) -> Flask:
   app = Flask(
      __name__,
      template_folder=str(Path(__file__).with_name("templates")),
      static_folder=str(Path(__file__).with_name("static")),
      static_url_path="/static",
   )
   app.config["DB_PATH"] = db_path

   @app.get("/")
   def index() -> str:
      return render_template(
         "index.html",
         db_path=db_path,
         latest_season=latest_season(db_path) or "",
      )

   @app.get("/api/options")
   def options():
      return jsonify(get_filter_options(db_path))

   @app.get("/api/dashboard")
   def dashboard():
      filters = build_filters_from_args(request.args)
      try:
         bin_size = float(request.args.get("bin_size", DEFAULT_BIN_SIZE))
      except ValueError:
         bin_size = DEFAULT_BIN_SIZE

      return jsonify(
         {
            "filters": {
               "season": filters.season,
               "team": filters.team,
               "player": filters.player,
               "shot_result": filters.shot_result,
               "home_away": filters.home_away,
               "period": filters.period,
               "bin_size": bin_size,
            },
            "summary": get_summary(db_path, filters),
            "bins": get_heatmap_bins(db_path, filters, bin_size=bin_size),
         }
      )

   return app


def parse_args() -> argparse.Namespace:
   parser = argparse.ArgumentParser(description="Run the HockeyShotMap visualization app.")
   parser.add_argument("--db-path", default="hockey_data.db", help="Path to the SQLite shot database.")
   parser.add_argument("--host", default="127.0.0.1", help="Host to bind the dashboard to.")
   parser.add_argument("--port", type=int, default=5000, help="Port to bind the dashboard to.")
   parser.add_argument("--debug", action="store_true", help="Run the Flask app in debug mode.")
   return parser.parse_args()


def main() -> None:
   args = parse_args()
   app = create_app(args.db_path)
   app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
   main()
