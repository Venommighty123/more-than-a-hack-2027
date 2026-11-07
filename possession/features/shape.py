"""Team shape and movement per frame -- the signal for WHO has the ball when nobody sees it.

The game plan's gap idea: the team in possession spreads out and pushes forward, the defending team
compacts and drops. Everything is measured along each team's own attacking direction
(``x_att``: 0 = own goal line, pitch length = opponent goal line), so the halves and both teams are
comparable. Shape lags a turnover by about a second (viability report): use it to say who, never to
time a change.

Per team (suffix _home / _away):
  cx_att       outfield centroid along the attacking direction (m from own goal line)
  cy           centroid across the pitch
  spread       RMS distance of the outfield players from their centroid
  length       x_att extent of the outfield players (90th - 10th percentile)
  width        y extent (90th - 10th percentile)
  def_line     x_att of the deepest outfield player
  advance      mean velocity along the attacking direction (m/s, + = pushing forward)
  n_tracked    players tracked
and differences home - away for advance, spread and cx_att.
The goalkeeper is the player closest to his own goal line and is left out of the outfield numbers.
"""
from __future__ import annotations

import warnings

import numpy as np

from possession.preprocess.players import attack_sign_per_frame

HOME, AWAY = 0, 1


def _nanpct(a, q):
    """Row-wise percentile ignoring NaN, linear interpolation (= np.nanpercentile, ~50x faster)."""
    s = np.sort(a, axis=1)                         # NaN sort to the end
    k = (~np.isnan(a)).sum(1)
    pos = (k - 1) * (q / 100.0)
    lo = np.clip(np.floor(pos).astype(int), 0, a.shape[1] - 1)
    hi = np.clip(lo + 1, 0, a.shape[1] - 1)
    r = np.arange(len(a))
    frac = pos - np.floor(pos)
    hi_v = np.where(lo + 1 < k, s[r, hi], s[r, lo])
    out = s[r, lo] + frac * (hi_v - s[r, lo])
    return np.where(k > 0, out, np.nan)


def _drop_keeper(x_att):
    """NaN out, per frame, the player with the smallest x_att (closest to own goal line)."""
    out = x_att.copy()
    filled = np.where(np.isnan(out), np.inf, out)
    j = np.argmin(filled, axis=1)
    ok = np.isfinite(filled[np.arange(len(out)), j])
    out[np.flatnonzero(ok), j[ok]] = np.nan
    return out


def team_shape(px, py, pvx, team, signs, blocks, *, lo=0, pitch=(105.3, 68.0)) -> dict:
    """Named (n,) float32 columns. ``signs`` / ``blocks`` as in preprocess.players.attack_signs."""
    team = np.asarray(team)
    n = len(px)
    length = pitch[0]
    cols = {}
    for side, tag in ((HOME, "home"), (AWAY, "away")):
        m = team == side
        sign = attack_sign_per_frame(signs, blocks, n, side, lo).astype(np.float64)[:, None]
        x, y = px[:, m].astype(np.float64), py[:, m].astype(np.float64)
        x_att = np.where(sign > 0, x, length - x)
        x_att[(sign == 0).ravel()] = np.nan                      # outside the halves
        field = _drop_keeper(x_att)
        yf = np.where(np.isnan(field), np.nan, y)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            cx, cy = np.nanmean(field, 1), np.nanmean(yf, 1)
            spread = np.sqrt(np.nanmean((field - cx[:, None]) ** 2 + (yf - cy[:, None]) ** 2, 1))
            adv = np.nanmean(pvx[:, m] * sign, 1) if pvx is not None else np.full(n, np.nan)
            def_line = np.nanmin(field, 1)
        cols["cx_att_" + tag] = cx
        cols["cy_" + tag] = cy
        cols["spread_" + tag] = spread
        cols["length_" + tag] = _nanpct(field, 90) - _nanpct(field, 10)
        cols["width_" + tag] = _nanpct(yf, 90) - _nanpct(yf, 10)
        cols["def_line_" + tag] = def_line
        cols["advance_" + tag] = adv
        cols["n_tracked_" + tag] = (~np.isnan(x)).sum(1).astype(np.float64)
    for k in ("advance", "spread", "cx_att"):
        cols[k + "_diff"] = cols[k + "_home"] - cols[k + "_away"]
    return {k: v.astype(np.float32) for k, v in cols.items()}
