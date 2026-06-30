import argparse
import json
import logging
import math
import sqlite3
import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np


@dataclass
class MetricsConfig:
   db_path: str
   score_start_season: int
   score_end_season: int | None = None
   train_start_season: int | None = None
   train_end_season: int | None = None
   min_shots_for_comparison: int = 50
   learning_rate: float = 0.05
   epochs: int = 400
   l2_regularization: float = 0.0005


FEATURE_SPEC_VERSION = "v1"


def _season_range(start_season: int, end_season: int | None) -> list[str]:
   resolved_end = start_season if end_season is None else end_season
   if resolved_end < start_season:
      raise ValueError("end season must be greater than or equal to start season")
   return [str(year) for year in range(start_season, resolved_end + 1)]


def initialize_metrics_tables(db_path: str) -> None:
   with sqlite3.connect(db_path) as connection:
      cursor = connection.cursor()
      cursor.execute(
         """
         CREATE TABLE IF NOT EXISTS metrics_model_runs (
            model_version TEXT PRIMARY KEY,
            model_type TEXT NOT NULL,
            created_at TEXT NOT NULL,
            train_start_season TEXT NOT NULL,
            train_end_season TEXT NOT NULL,
            score_start_season TEXT NOT NULL,
            score_end_season TEXT NOT NULL,
            row_count INTEGER NOT NULL,
            feature_spec_json TEXT NOT NULL,
            weights_json TEXT NOT NULL,
            bias REAL NOT NULL,
            config_signature TEXT,
            train_signature TEXT
         )
         """
      )
      cursor.execute("PRAGMA table_info(metrics_model_runs)")
      model_run_columns = {row[1] for row in cursor.fetchall()}
      if "config_signature" not in model_run_columns:
         cursor.execute("ALTER TABLE metrics_model_runs ADD COLUMN config_signature TEXT")
      if "train_signature" not in model_run_columns:
         cursor.execute("ALTER TABLE metrics_model_runs ADD COLUMN train_signature TEXT")
      cursor.execute(
         """
         CREATE TABLE IF NOT EXISTS shot_xg (
            event_hash TEXT NOT NULL,
            season TEXT NOT NULL,
            game_id INTEGER NOT NULL,
            model_version TEXT NOT NULL,
            xg REAL NOT NULL,
            PRIMARY KEY (event_hash, model_version)
         )
         """
      )
      cursor.execute("CREATE INDEX IF NOT EXISTS idx_shot_xg_model_season ON shot_xg(model_version, season)")

      cursor.execute(
         """
         CREATE TABLE IF NOT EXISTS player_season_metrics (
            season TEXT NOT NULL,
            model_version TEXT NOT NULL,
            shooter TEXT NOT NULL,
            shooter_id INTEGER NOT NULL,
            team TEXT NOT NULL,
            shots INTEGER NOT NULL,
            goals INTEGER NOT NULL,
            xg REAL NOT NULL,
            goals_above_expected REAL NOT NULL,
            shooting_pct REAL NOT NULL,
            xg_per_shot REAL NOT NULL,
            league_avg_xg_per_shot REAL,
            league_avg_shooting_pct REAL,
            delta_xg_per_shot REAL,
            delta_shooting_pct REAL,
            qualified INTEGER NOT NULL,
            rank_gax INTEGER,
            PRIMARY KEY (season, model_version, shooter, shooter_id, team)
         )
         """
      )

      cursor.execute(
         """
         CREATE TABLE IF NOT EXISTS team_season_metrics (
            season TEXT NOT NULL,
            model_version TEXT NOT NULL,
            team TEXT NOT NULL,
            shots INTEGER NOT NULL,
            goals INTEGER NOT NULL,
            xg REAL NOT NULL,
            goals_above_expected REAL NOT NULL,
            shooting_pct REAL NOT NULL,
            xg_per_shot REAL NOT NULL,
            PRIMARY KEY (season, model_version, team)
         )
         """
      )

      cursor.execute(
         """
         CREATE TABLE IF NOT EXISTS goalie_season_metrics (
            season TEXT NOT NULL,
            model_version TEXT NOT NULL,
            goalie TEXT NOT NULL,
            goalie_id INTEGER NOT NULL,
            shots_against INTEGER NOT NULL,
            goals_against INTEGER NOT NULL,
            xga REAL NOT NULL,
            expected_save_pct REAL NOT NULL,
            save_pct REAL NOT NULL,
            saves_above_average REAL NOT NULL,
            PRIMARY KEY (season, model_version, goalie, goalie_id)
         )
         """
      )

      cursor.execute(
         """
         CREATE TABLE IF NOT EXISTS metrics_season_refresh_state (
            season TEXT PRIMARY KEY,
            config_signature TEXT NOT NULL,
            train_signature TEXT NOT NULL,
            source_row_count INTEGER NOT NULL,
            source_goal_count INTEGER NOT NULL,
            source_max_shot_id INTEGER NOT NULL,
            model_version TEXT NOT NULL,
            updated_at TEXT NOT NULL
         )
         """
      )
      connection.commit()


def _config_signature(config: MetricsConfig, train_seasons: list[str]) -> str:
   payload = {
      "feature_spec_version": FEATURE_SPEC_VERSION,
      "train_seasons": train_seasons,
      "learning_rate": config.learning_rate,
      "epochs": config.epochs,
      "l2_regularization": config.l2_regularization,
      "min_shots_for_comparison": config.min_shots_for_comparison,
   }
   encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
   return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _collect_source_fingerprint(connection: sqlite3.Connection, seasons: list[str]) -> dict[str, dict[str, int]]:
   placeholders = ",".join("?" for _ in seasons)
   rows = connection.execute(
      f"""
      SELECT season,
             COUNT(*) AS row_count,
             SUM(CASE WHEN shot_result = 'Goal' THEN 1 ELSE 0 END) AS goal_count,
             COALESCE(MAX(id), 0) AS max_shot_id
      FROM shots
      WHERE season IN ({placeholders})
        AND shot_result IN ('Goal', 'ngshot')
        AND x IS NOT NULL
        AND y IS NOT NULL
        AND COALESCE(is_empty_net, 0) = 0
      GROUP BY season
      """,
      seasons,
   ).fetchall()

   by_season = {
      str(row[0]): {
         "row_count": int(row[1] or 0),
         "goal_count": int(row[2] or 0),
         "max_shot_id": int(row[3] or 0),
      }
      for row in rows
   }
   for season in seasons:
      by_season.setdefault(str(season), {"row_count": 0, "goal_count": 0, "max_shot_id": 0})
   return by_season


def _train_signature(train_fingerprint: dict[str, dict[str, int]]) -> str:
   encoded = json.dumps(train_fingerprint, sort_keys=True, separators=(",", ":"))
   return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _lookup_cached_model(
   connection: sqlite3.Connection,
   config_signature: str,
   train_signature: str,
) -> tuple[str, dict, np.ndarray, float] | None:
   connection.row_factory = sqlite3.Row
   row = connection.execute(
      """
      SELECT model_version, feature_spec_json, weights_json, bias
      FROM metrics_model_runs
      WHERE config_signature = ? AND train_signature = ?
      ORDER BY created_at DESC
      LIMIT 1
      """,
      (config_signature, train_signature),
   ).fetchone()
   if row is None:
      return None

   feature_spec_payload = json.loads(row["feature_spec_json"])
   weight_vector = np.array(json.loads(row["weights_json"]), dtype=np.float64)
   bias = float(row["bias"])
   return str(row["model_version"]), feature_spec_payload, weight_vector, bias


def _season_state_row(connection: sqlite3.Connection, season: str) -> sqlite3.Row | None:
   connection.row_factory = sqlite3.Row
   return connection.execute(
      """
      SELECT season, config_signature, train_signature, source_row_count, source_goal_count,
             source_max_shot_id, model_version
      FROM metrics_season_refresh_state
      WHERE season = ?
      """,
      (season,),
   ).fetchone()


def _upsert_season_state(
   cursor: sqlite3.Cursor,
   season: str,
   config_signature: str,
   train_signature: str,
   source_row_count: int,
   source_goal_count: int,
   source_max_shot_id: int,
   model_version: str,
) -> None:
   cursor.execute(
      """
      INSERT INTO metrics_season_refresh_state (
         season, config_signature, train_signature, source_row_count,
         source_goal_count, source_max_shot_id, model_version, updated_at
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
      ON CONFLICT(season) DO UPDATE SET
         config_signature = excluded.config_signature,
         train_signature = excluded.train_signature,
         source_row_count = excluded.source_row_count,
         source_goal_count = excluded.source_goal_count,
         source_max_shot_id = excluded.source_max_shot_id,
         model_version = excluded.model_version,
         updated_at = excluded.updated_at
      """,
      (
         season,
         config_signature,
         train_signature,
         source_row_count,
         source_goal_count,
         source_max_shot_id,
         model_version,
         datetime.now(UTC).isoformat(timespec="seconds"),
      ),
   )


def _derived_distance_and_angle(x_coord: float, y_coord: float) -> tuple[float, float]:
   # Approximate distance/angle to the attacking net using nearest net at +/-89 feet.
   net_x = 89.0
   dx = abs(net_x - abs(x_coord))
   distance = math.hypot(dx, y_coord)
   angle = math.degrees(math.atan2(abs(y_coord), max(dx, 1e-6)))
   return distance, angle


def _safe_float(value: object, default: float = 0.0) -> float:
   try:
      if value is None:
         return default
      return float(value)
   except (TypeError, ValueError):
      return default


def _load_shot_rows_for_features(connection: sqlite3.Connection, seasons: list[str]) -> list[sqlite3.Row]:
   placeholders = ",".join("?" for _ in seasons)
   query = f"""
      SELECT event_hash, season, game_id, shot_result, x, y, shot_type, strength_state, zone,
             score_differential, period, home_away, is_empty_net, shot_distance, shot_angle
      FROM shots
      WHERE season IN ({placeholders})
        AND shot_result IN ('Goal', 'ngshot')
        AND x IS NOT NULL
        AND y IS NOT NULL
        AND COALESCE(is_empty_net, 0) = 0
   """
   connection.row_factory = sqlite3.Row
   return list(connection.execute(query, seasons).fetchall())


def _build_feature_spec(rows: list[sqlite3.Row]) -> dict:
   shot_types = sorted({(row["shot_type"] or "Unknown").strip() for row in rows})
   strength_states = sorted({(row["strength_state"] or "Unknown").strip() for row in rows})
   zones = sorted({(row["zone"] or "Unknown").strip() for row in rows})
   return {
      "numeric": ["shot_distance", "shot_angle", "score_differential", "period", "home_away"],
      "shot_type": shot_types,
      "strength_state": strength_states,
      "zone": zones,
   }


def _vectorize_rows(rows: list[sqlite3.Row], feature_spec: dict, fit_scaler: bool, scaler: dict | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
   numeric_matrix: list[list[float]] = []
   categorical_matrix: list[list[float]] = []
   labels: list[float] = []

   shot_type_to_idx = {value: idx for idx, value in enumerate(feature_spec["shot_type"])}
   strength_to_idx = {value: idx for idx, value in enumerate(feature_spec["strength_state"])}
   zone_to_idx = {value: idx for idx, value in enumerate(feature_spec["zone"])}

   cat_width = len(shot_type_to_idx) + len(strength_to_idx) + len(zone_to_idx)

   for row in rows:
      dist = _safe_float(row["shot_distance"], default=float("nan"))
      angle = _safe_float(row["shot_angle"], default=float("nan"))
      if math.isnan(dist) or math.isnan(angle):
         dist_fallback, angle_fallback = _derived_distance_and_angle(_safe_float(row["x"]), _safe_float(row["y"]))
         if math.isnan(dist):
            dist = dist_fallback
         if math.isnan(angle):
            angle = angle_fallback

      numeric_values = [
         dist,
         angle,
         _safe_float(row["score_differential"], 0.0),
         _safe_float(row["period"], 0.0),
         _safe_float(row["home_away"], 0.0),
      ]
      numeric_matrix.append(numeric_values)

      category_vec = [0.0] * cat_width
      shot_type = (row["shot_type"] or "Unknown").strip()
      strength = (row["strength_state"] or "Unknown").strip()
      zone = (row["zone"] or "Unknown").strip()

      offset = 0
      if shot_type in shot_type_to_idx:
         category_vec[offset + shot_type_to_idx[shot_type]] = 1.0
      offset += len(shot_type_to_idx)

      if strength in strength_to_idx:
         category_vec[offset + strength_to_idx[strength]] = 1.0
      offset += len(strength_to_idx)

      if zone in zone_to_idx:
         category_vec[offset + zone_to_idx[zone]] = 1.0

      categorical_matrix.append(category_vec)
      labels.append(1.0 if row["shot_result"] == "Goal" else 0.0)

   numeric_array = np.array(numeric_matrix, dtype=np.float64)
   categorical_array = np.array(categorical_matrix, dtype=np.float64)

   if fit_scaler:
      means = numeric_array.mean(axis=0)
      stds = numeric_array.std(axis=0)
      stds = np.where(stds < 1e-9, 1.0, stds)
      scaler = {"means": means.tolist(), "stds": stds.tolist()}
   if scaler is None:
      raise ValueError("scaler must be provided when fit_scaler is False")

   means_np = np.array(scaler["means"], dtype=np.float64)
   stds_np = np.array(scaler["stds"], dtype=np.float64)
   standardized_numeric = (numeric_array - means_np) / stds_np

   features = np.concatenate([standardized_numeric, categorical_array], axis=1)
   targets = np.array(labels, dtype=np.float64)
   return features, targets, scaler


def _sigmoid(values: np.ndarray) -> np.ndarray:
   clipped = np.clip(values, -50.0, 50.0)
   return 1.0 / (1.0 + np.exp(-clipped))


def _fit_logistic_regression(features: np.ndarray, targets: np.ndarray, learning_rate: float, epochs: int, l2_regularization: float) -> tuple[np.ndarray, float]:
   if features.shape[0] == 0:
      raise ValueError("cannot fit model with zero rows")

   weight_vector = np.zeros(features.shape[1], dtype=np.float64)
   bias = 0.0
   row_count = float(features.shape[0])

   for _ in range(epochs):
      logits = features @ weight_vector + bias
      probs = _sigmoid(logits)

      error = probs - targets
      grad_w = (features.T @ error) / row_count + (l2_regularization * weight_vector)
      grad_b = float(error.mean())

      weight_vector -= learning_rate * grad_w
      bias -= learning_rate * grad_b

   return weight_vector, bias


def _score_rows(features: np.ndarray, weight_vector: np.ndarray, bias: float) -> np.ndarray:
   logits = features @ weight_vector + bias
   return _sigmoid(logits)


def _delete_existing_model_outputs(cursor: sqlite3.Cursor, seasons: list[str]) -> None:
   placeholders = ",".join("?" for _ in seasons)
   cursor.execute(f"DELETE FROM shot_xg WHERE season IN ({placeholders})", seasons)
   cursor.execute(f"DELETE FROM player_season_metrics WHERE season IN ({placeholders})", seasons)
   cursor.execute(f"DELETE FROM team_season_metrics WHERE season IN ({placeholders})", seasons)
   cursor.execute(f"DELETE FROM goalie_season_metrics WHERE season IN ({placeholders})", seasons)


def _persist_player_season_metrics(cursor: sqlite3.Cursor, model_version: str, seasons: list[str], min_shots: int) -> int:
   placeholders = ",".join("?" for _ in seasons)
   query = f"""
      SELECT s.season,
             COALESCE(s.shooter, 'Unknown') AS shooter,
             s.shooter_id,
             s.team,
             COUNT(*) AS shots,
             SUM(CASE WHEN s.shot_result = 'Goal' THEN 1 ELSE 0 END) AS goals,
             SUM(x.xg) AS xg
      FROM shots s
      JOIN shot_xg x ON x.event_hash = s.event_hash
      WHERE x.model_version = ?
        AND s.season IN ({placeholders})
      GROUP BY s.season, shooter, s.shooter_id, s.team
   """
   rows = [dict(row) for row in cursor.execute(query, [model_version, *seasons]).fetchall()]
   if not rows:
      return 0

   league_by_season: dict[str, dict[str, float]] = {}
   for season in seasons:
      qualified = [row for row in rows if row["season"] == season and int(row["shots"] or 0) >= min_shots]
      if not qualified:
         league_by_season[season] = {"xg_per_shot": 0.0, "shooting_pct": 0.0}
         continue
      total_xg = sum(float(row["xg"] or 0.0) for row in qualified)
      total_goals = sum(float(row["goals"] or 0.0) for row in qualified)
      total_shots = sum(float(row["shots"] or 0.0) for row in qualified)
      league_by_season[season] = {
         "xg_per_shot": total_xg / total_shots if total_shots else 0.0,
         "shooting_pct": total_goals / total_shots if total_shots else 0.0,
      }

   rank_map: dict[tuple[str, str, object, object], int] = {}
   for season in seasons:
      ranked_rows = [row for row in rows if row["season"] == season and int(row["shots"] or 0) >= min_shots]
      ranked_rows.sort(key=lambda row: float(row["goals"] or 0.0) - float(row["xg"] or 0.0), reverse=True)
      for index, row in enumerate(ranked_rows, start=1):
         rank_map[(row["season"], row["shooter"], row["shooter_id"], row["team"])] = index

   insert_rows = []
   for row in rows:
      shots = int(row["shots"] or 0)
      goals = int(row["goals"] or 0)
      xg = float(row["xg"] or 0.0)
      gax = goals - xg
      shooting_pct = goals / shots if shots else 0.0
      xg_per_shot = xg / shots if shots else 0.0
      season = row["season"]
      qualified = 1 if shots >= min_shots else 0
      league_avg = league_by_season.get(season, {"xg_per_shot": 0.0, "shooting_pct": 0.0})
      rank = rank_map.get((row["season"], row["shooter"], row["shooter_id"], row["team"]))
      insert_rows.append(
         (
            season,
            model_version,
            row["shooter"],
            -1 if row["shooter_id"] is None else int(row["shooter_id"]),
            "" if row["team"] is None else str(row["team"]),
            shots,
            goals,
            xg,
            gax,
            shooting_pct,
            xg_per_shot,
            league_avg["xg_per_shot"],
            league_avg["shooting_pct"],
            xg_per_shot - league_avg["xg_per_shot"],
            shooting_pct - league_avg["shooting_pct"],
            qualified,
            rank,
         )
      )

   cursor.executemany(
      """
      INSERT INTO player_season_metrics (
         season, model_version, shooter, shooter_id, team, shots, goals, xg,
         goals_above_expected, shooting_pct, xg_per_shot, league_avg_xg_per_shot,
         league_avg_shooting_pct, delta_xg_per_shot, delta_shooting_pct, qualified, rank_gax
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
      """,
      insert_rows,
   )
   return len(insert_rows)


def _persist_team_season_metrics(cursor: sqlite3.Cursor, model_version: str, seasons: list[str]) -> int:
   placeholders = ",".join("?" for _ in seasons)
   rows = cursor.execute(
      f"""
      SELECT s.season,
             COALESCE(s.team, 'Unknown') AS team,
             COUNT(*) AS shots,
             SUM(CASE WHEN s.shot_result = 'Goal' THEN 1 ELSE 0 END) AS goals,
             SUM(x.xg) AS xg
      FROM shots s
      JOIN shot_xg x ON x.event_hash = s.event_hash
      WHERE x.model_version = ?
        AND s.season IN ({placeholders})
      GROUP BY s.season, team
      """,
      [model_version, *seasons],
   ).fetchall()

   insert_rows = []
   for row in rows:
      shots = int(row[2] or 0)
      goals = int(row[3] or 0)
      xg = float(row[4] or 0.0)
      insert_rows.append(
         (
            row[0],
            model_version,
            row[1],
            shots,
            goals,
            xg,
            goals - xg,
            goals / shots if shots else 0.0,
            xg / shots if shots else 0.0,
         )
      )

   if insert_rows:
      cursor.executemany(
         """
         INSERT INTO team_season_metrics (
            season, model_version, team, shots, goals, xg, goals_above_expected,
            shooting_pct, xg_per_shot
         ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
         """,
         insert_rows,
      )
   return len(insert_rows)


def _persist_goalie_season_metrics(cursor: sqlite3.Cursor, model_version: str, seasons: list[str]) -> int:
   placeholders = ",".join("?" for _ in seasons)
   rows = cursor.execute(
      f"""
      SELECT s.season,
             COALESCE(s.goalie, 'Unknown') AS goalie,
             s.goalie_id,
             COUNT(*) AS shots_against,
             SUM(CASE WHEN s.shot_result = 'Goal' THEN 1 ELSE 0 END) AS goals_against,
             SUM(x.xg) AS xga
      FROM shots s
      JOIN shot_xg x ON x.event_hash = s.event_hash
      WHERE x.model_version = ?
        AND s.season IN ({placeholders})
        AND s.goalie IS NOT NULL
      GROUP BY s.season, goalie, s.goalie_id
      """,
      [model_version, *seasons],
   ).fetchall()

   insert_rows = []
   for row in rows:
      shots_against = int(row[3] or 0)
      goals_against = int(row[4] or 0)
      xga = float(row[5] or 0.0)
      saves = shots_against - goals_against
      expected_saves = shots_against - xga
      save_pct = saves / shots_against if shots_against else 0.0
      expected_save_pct = expected_saves / shots_against if shots_against else 0.0
      saves_above_average = saves - expected_saves
      insert_rows.append(
         (
            row[0],
            model_version,
            row[1],
            -1 if row[2] is None else int(row[2]),
            shots_against,
            goals_against,
            xga,
            expected_save_pct,
            save_pct,
            saves_above_average,
         )
      )

   if insert_rows:
      cursor.executemany(
         """
         INSERT INTO goalie_season_metrics (
            season, model_version, goalie, goalie_id, shots_against, goals_against,
            xga, expected_save_pct, save_pct, saves_above_average
         ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
         """,
         insert_rows,
      )
   return len(insert_rows)


def run_metrics_refresh(config: MetricsConfig) -> dict:
   initialize_metrics_tables(config.db_path)

   score_seasons = _season_range(config.score_start_season, config.score_end_season)
   train_start = config.score_start_season if config.train_start_season is None else config.train_start_season
   train_end = (config.score_end_season if config.score_end_season is not None else config.score_start_season) if config.train_end_season is None else config.train_end_season
   train_seasons = _season_range(train_start, train_end)

   with sqlite3.connect(config.db_path) as connection:
      connection.row_factory = sqlite3.Row
      score_fingerprint = _collect_source_fingerprint(connection, score_seasons)
      train_fingerprint = _collect_source_fingerprint(connection, train_seasons)

      config_signature = _config_signature(config, train_seasons)
      train_signature = _train_signature(train_fingerprint)

      cached_model = _lookup_cached_model(connection, config_signature, train_signature)
      trained_new_model = False
      if cached_model is None:
         training_rows = _load_shot_rows_for_features(connection, train_seasons)
         if len(training_rows) < 200:
            raise ValueError("not enough training rows to fit xG model; scrape more games/seasons first")

         feature_spec = _build_feature_spec(training_rows)
         training_features, training_targets, scaler = _vectorize_rows(training_rows, feature_spec, fit_scaler=True)
         weight_vector, bias = _fit_logistic_regression(
            training_features,
            training_targets,
            config.learning_rate,
            config.epochs,
            config.l2_regularization,
         )

         model_version = f"xg_logreg_v1_{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}"
         feature_spec_payload = {
            "feature_spec": feature_spec,
            "scaler": scaler,
         }

         cursor = connection.cursor()
         cursor.execute(
            """
            INSERT INTO metrics_model_runs (
               model_version, model_type, created_at, train_start_season, train_end_season,
               score_start_season, score_end_season, row_count, feature_spec_json,
               weights_json, bias, config_signature, train_signature
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
               model_version,
               "logistic_regression_v1",
               datetime.now(UTC).isoformat(timespec="seconds"),
               train_seasons[0],
               train_seasons[-1],
               score_seasons[0],
               score_seasons[-1],
               len(training_rows),
               json.dumps(feature_spec_payload),
               json.dumps(weight_vector.tolist()),
               float(bias),
               config_signature,
               train_signature,
            ),
         )
         connection.commit()
         trained_new_model = True
      else:
         model_version, feature_spec_payload, weight_vector, bias = cached_model
         training_rows = _load_shot_rows_for_features(connection, train_seasons)

      feature_spec = feature_spec_payload["feature_spec"]
      scaler = feature_spec_payload["scaler"]

      dirty_seasons: list[str] = []
      skipped_seasons: list[str] = []
      for season in score_seasons:
         state = _season_state_row(connection, season)
         fingerprint = score_fingerprint[season]
         if state is None:
            dirty_seasons.append(season)
            continue

         unchanged_source = (
            int(state["source_row_count"]) == int(fingerprint["row_count"])
            and int(state["source_goal_count"]) == int(fingerprint["goal_count"])
            and int(state["source_max_shot_id"]) == int(fingerprint["max_shot_id"])
         )
         same_signatures = (
            str(state["config_signature"]) == config_signature
            and str(state["train_signature"]) == train_signature
            and str(state["model_version"]) == model_version
         )
         if unchanged_source and same_signatures:
            skipped_seasons.append(season)
         else:
            dirty_seasons.append(season)

      if not dirty_seasons:
         return {
            "model_version": model_version,
            "training_rows": len(training_rows),
            "scored_shots": 0,
            "player_season_rows": 0,
            "team_season_rows": 0,
            "goalie_season_rows": 0,
            "scored_seasons": [],
            "trained_seasons": train_seasons,
            "skipped_seasons": skipped_seasons,
            "trained_new_model": trained_new_model,
         }

      scoring_rows = _load_shot_rows_for_features(connection, dirty_seasons)
      scoring_features, _, _ = _vectorize_rows(scoring_rows, feature_spec, fit_scaler=False, scaler=scaler)
      xg_values = _score_rows(scoring_features, weight_vector, bias)

      cursor = connection.cursor()
      _delete_existing_model_outputs(cursor, dirty_seasons)

      shot_rows = [
         (
            row["event_hash"],
            row["season"],
            row["game_id"],
            model_version,
            float(xg),
         )
         for row, xg in zip(scoring_rows, xg_values, strict=True)
      ]
      cursor.executemany(
         """
         INSERT INTO shot_xg (event_hash, season, game_id, model_version, xg)
         VALUES (?, ?, ?, ?, ?)
         """,
         shot_rows,
      )

      player_count = _persist_player_season_metrics(cursor, model_version, dirty_seasons, config.min_shots_for_comparison)
      team_count = _persist_team_season_metrics(cursor, model_version, dirty_seasons)
      goalie_count = _persist_goalie_season_metrics(cursor, model_version, dirty_seasons)

      for season in dirty_seasons:
         fingerprint = score_fingerprint[season]
         _upsert_season_state(
            cursor,
            season,
            config_signature,
            train_signature,
            int(fingerprint["row_count"]),
            int(fingerprint["goal_count"]),
            int(fingerprint["max_shot_id"]),
            model_version,
         )

      connection.commit()

   summary = {
      "model_version": model_version,
      "training_rows": len(training_rows),
      "scored_shots": len(scoring_rows),
      "player_season_rows": player_count,
      "team_season_rows": team_count,
      "goalie_season_rows": goalie_count,
      "scored_seasons": dirty_seasons,
      "trained_seasons": train_seasons,
      "skipped_seasons": skipped_seasons,
      "trained_new_model": trained_new_model,
   }
   return summary


def parse_args() -> argparse.Namespace:
   parser = argparse.ArgumentParser(description="Build xG/xSV derived metrics from scraped shot data.")
   parser.add_argument("--db-path", default="hockey_shots.db", help="SQLite database path from Main.py scrape output.")
   parser.add_argument("--season", type=int, required=True, help="Start season to score (for example 2024 for 2024-25).")
   parser.add_argument("--end-season", type=int, default=None, help="Optional end season to score.")
   parser.add_argument("--train-start-season", type=int, default=None, help="Optional start season for model training window.")
   parser.add_argument("--train-end-season", type=int, default=None, help="Optional end season for model training window.")
   parser.add_argument("--min-shots", type=int, default=50, help="Minimum shots for league-comparison ranking in player metrics.")
   parser.add_argument("--learning-rate", type=float, default=0.05, help="Gradient descent learning rate for logistic regression.")
   parser.add_argument("--epochs", type=int, default=400, help="Training epochs for logistic regression.")
   parser.add_argument("--l2", type=float, default=0.0005, help="L2 regularization strength.")
   parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
   return parser.parse_args()


def main() -> None:
   args = parse_args()
   logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")

   config = MetricsConfig(
      db_path=args.db_path,
      score_start_season=args.season,
      score_end_season=args.end_season,
      train_start_season=args.train_start_season,
      train_end_season=args.train_end_season,
      min_shots_for_comparison=args.min_shots,
      learning_rate=args.learning_rate,
      epochs=args.epochs,
      l2_regularization=args.l2,
   )

   summary = run_metrics_refresh(config)
   logging.info("Metrics refresh complete: %s", json.dumps(summary))


if __name__ == "__main__":
   main()
