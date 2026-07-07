"""Age curve model for hockey shot xG predictions.

This module contains the age curve model that was previously embedded in Metrics.py.
It provides functions to compute age-adjusted metrics for player performance.

The model includes:
- A fitted age curve based on historical player data
- Player trend multipliers from recent performance
- Age projection factors for adjusting metrics based on age
"""

import numpy as np
from typing import Callable, Dict, Any


def _build_age_xg_curve(player_season_data: Dict[int, Dict[str, Dict[str, Any]]]) -> Callable[[float], float]:
   """Fit a smooth league-wide xG-per-game age curve.

   The curve is intentionally low-dimensional (4th-degree polynomial with
   ridge regularization) so it captures broad lifecycle shape without
   overfitting noise from a single season.
   """
   # Prior lifecycle curve (relative scale) matching typical NHL development:
   # rapid growth early 20s, prime plateau, slow decline, then steeper decline.
   prior_points = {
      18: 0.68,
      20: 0.80,
      22: 0.92,
      24: 1.00,
      27: 1.02,
      30: 0.98,
      33: 0.90,
      36: 0.76,
      39: 0.60,
   }

   def _prior_curve(age: float) -> float:
      clamped = float(np.clip(age, 18.0, 39.0))
      low = int(np.floor(clamped))
      high = int(np.ceil(clamped))
      low_anchor = max(k for k in prior_points if k <= low)
      high_anchor = min(k for k in prior_points if k >= high)
      if low_anchor == high_anchor:
         return float(prior_points[low_anchor])
      low_val = float(prior_points[low_anchor])
      high_val = float(prior_points[high_anchor])
      span = float(high_anchor - low_anchor)
      t = (clamped - float(low_anchor)) / span if span else 0.0
      return float(low_val + t * (high_val - low_val))

   age_samples: list[float] = []
   xg_per_game_samples: list[float] = []
   sample_weights: list[float] = []

   for season_map in player_season_data.values():
      for data in season_map.values():
         age = int(data.get("age") or 0)
         games = int(data.get("games") or 0)
         xg = float(data.get("xg") or 0.0)
         if age <= 0 or games <= 0:
            continue
         xg_per_game = xg / games
         age_samples.append(float(age))
         xg_per_game_samples.append(float(xg_per_game))
         sample_weights.append(float(min(games, 82)))

   if len(age_samples) < 20:
      return _prior_curve

   x = np.asarray(age_samples, dtype=float)
   y = np.asarray(xg_per_game_samples, dtype=float)
   w = np.asarray(sample_weights, dtype=float)

   # Polynomial basis centered at league-prime-ish age to improve stability.
   centered = x - 27.0
   design = np.column_stack(
      [
         np.ones_like(centered),
         centered,
         centered**2,
         centered**3,
         centered**4,
      ]
   )

   sqrt_w = np.sqrt(np.clip(w, 1.0, None))
   wx = design * sqrt_w[:, None]
   wy = y * sqrt_w
   ridge_lambda = 0.08
   ridge = ridge_lambda * np.eye(design.shape[1])

   try:
      beta = np.linalg.solve(wx.T @ wx + ridge, wx.T @ wy)
   except np.linalg.LinAlgError:
      beta = np.linalg.lstsq(wx, wy, rcond=None)[0]

   fit_grid = {age: max(float(np.asarray([1.0, age - 27.0, (age - 27.0) ** 2, (age - 27.0) ** 3, (age - 27.0) ** 4]) @ beta), 0.02)
               for age in range(18, 41)}
   fit_peak = max(fit_grid.values()) if fit_grid else 1.0
   prior_peak = max(_prior_curve(float(age)) for age in range(24, 31))

   def _curve(age: float) -> float:
      clamped_age = float(np.clip(age, 18.0, 40.0))
      c = clamped_age - 27.0
      features = np.asarray([1.0, c, c**2, c**3, c**4], dtype=float)
      fit_value = max(float(features @ beta), 0.02)

      # Blend empirical fit with lifecycle prior so single-season noise does
      # not invert early-career growth or overstate late decline.
      fit_relative = fit_value / max(fit_peak, 1e-6)
      prior_relative = _prior_curve(clamped_age) / max(prior_peak, 1e-6)
      blended_relative = (0.45 * fit_relative) + (0.55 * prior_relative)
      return float(np.clip(blended_relative, 0.45, 1.10))

   return _curve


def _player_trend_multiplier(season_map: Dict[str, Dict], season: str) -> float:
   """Compute a player-specific trajectory multiplier from recent xG/GAX trend."""
   seasons_sorted = sorted(season_map.keys())
   if season not in seasons_sorted:
      return 1.0

   current_idx = seasons_sorted.index(season)
   history = seasons_sorted[max(0, current_idx - 3):current_idx + 1]
   if len(history) < 2:
      return 1.0

   x_vals = np.arange(len(history), dtype=float)
   xg_pg_vals: list[float] = []
   gax_pg_vals: list[float] = []
   for hs in history:
      data = season_map[hs]
      games = max(int(data.get("games") or 0), 1)
      xg_pg_vals.append(float(data.get("xg") or 0.0) / float(games))
      gax_pg_vals.append(float(data.get("gax") or 0.0) / float(games))

   xg_slope = float(np.polyfit(x_vals, np.asarray(xg_pg_vals, dtype=float), 1)[0])
   gax_slope = float(np.polyfit(x_vals, np.asarray(gax_pg_vals, dtype=float), 1)[0])

   # Positive trends (including elite outliers aging well) reduce decline.
   xg_component = 0.10 * float(np.tanh(xg_slope / 0.03))
   gax_component = 0.12 * float(np.tanh(gax_slope / 0.04))
   return float(np.clip(1.0 + xg_component + gax_component, 0.88, 1.20))


def _apply_projection_factor(value: float, factor: float) -> float:
   """Apply age projection factor while preserving intuitive direction.

   For improving ages (factor > 1):
   - positive GAX is boosted upward
   - negative GAX is softened toward 0
   For declining ages (factor < 1):
   - positive GAX is damped
   - negative GAX becomes more negative
   """
   if value >= 0:
      return value * factor
   safe_factor = max(factor, 1e-6)
   return value / safe_factor


def compute_age_adjusted_gax(gax: float, age: float, age_curve: Callable[[float], float], 
                           trend_factor: float = 1.0) -> float:
   """Compute age-adjusted goals above expected (GAx) for a player.

   Args:
      gax: Goals above expected for the player
      age: Player's age
      age_curve: Age curve function from _build_age_xg_curve
      trend_factor: Player trend multiplier from _player_trend_multiplier

   Returns:
      Age-adjusted GAx value
   """
   if age <= 0:
      return gax

   current_curve = age_curve(float(age))
   horizon_age = min(float(age + 3), 40.0)
   future_curve = age_curve(horizon_age)
   base_age_factor = float(np.clip(future_curve / max(current_curve, 1e-6), 0.75, 1.30))

   age_factor = float(np.clip(base_age_factor * trend_factor, 0.70, 1.35))

   # Young players should carry explicit upside unless the age is unknown.
   if age <= 22:
      age_factor = max(age_factor, 1.05)
   elif age <= 24:
      age_factor = max(age_factor, 1.00)

   return _apply_projection_factor(gax, age_factor)


def compute_age_adjusted_gax_per_60(gax_per_60: float, age: float, age_curve: Callable[[float], float],
                                   trend_factor: float = 1.0) -> float:
   """Compute age-adjusted GAx per 60 minutes for a player.

   Args:
      gax_per_60: Goals above expected per 60 minutes for the player
      age: Player's age
      age_curve: Age curve function from _build_age_xg_curve
      trend_factor: Player trend multiplier from _player_trend_multiplier

   Returns:
      Age-adjusted GAx per 60 value
   """
   return compute_age_adjusted_gax(gax_per_60, age, age_curve, trend_factor)