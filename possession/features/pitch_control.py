"""Pitch control: the probability that each team would win a ball arriving at a point.

The simplified Spearman model: both teams' fastest arrival times (features/time_to_ball.py) go
through a logistic on their difference,

    P_home(x) = 1 / (1 + exp(-pi / (sqrt(3) * sigma) * (T_away(x) - T_home(x))))

with sigma the time-uncertainty (0.45 s in Spearman 2017). This is the collective version of the
zone of influence: it combines every player's zone into one number per point.

  pitch_control_at_ball  per frame, P_home at the ball's position  (a feature column)
  pitch_control_grid     one frame, a (ny, nx) grid over the pitch  (for viz / analysis)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .time_to_ball import TimeToBallConfig, player_times

HOME, AWAY = 0, 1


@dataclass
class PitchControlConfig:
    sigma_s: float = 0.45
    ttb: TimeToBallConfig = None

    def __post_init__(self):
        if self.ttb is None:
            self.ttb = TimeToBallConfig()

    @classmethod
    def from_dict(cls, d: dict | None) -> "PitchControlConfig":
        d = dict(d or {})
        ttb = TimeToBallConfig.from_dict(d.pop("ttb", None))
        return cls(ttb=ttb, **{k: v for k, v in d.items() if k in ("sigma_s",)})

    def to_dict(self) -> dict:
        return asdict(self)


def _logistic(t_home, t_away, sigma_s):
    k = np.pi / (np.sqrt(3.0) * sigma_s)
    with np.errstate(invalid="ignore", over="ignore"):
        diff = np.clip(t_away - t_home, -30.0, 30.0)   # inf - inf = nan -> 0.5 below
        p = 1.0 / (1.0 + np.exp(-k * diff))
    return np.where(np.isnan(p), 0.5, p).astype(np.float32)


def pitch_control_at_ball(px, py, pvx, pvy, bx, by, team, cfg: PitchControlConfig | None = None):
    """(n,) P_home at the ball; NaN where the ball is unseen."""
    cfg = cfg or PitchControlConfig()
    team = np.asarray(team)
    t = player_times(px, py, pvx, pvy, bx, by, cfg.ttb)
    th = np.min(np.where((team == HOME)[None, :], t, np.inf), axis=1)
    ta = np.min(np.where((team == AWAY)[None, :], t, np.inf), axis=1)
    p = _logistic(th, ta, cfg.sigma_s)
    p[np.isnan(bx) | np.isnan(by)] = np.nan
    return p


def pitch_control_grid(px, py, pvx, pvy, team, pitch=(105.3, 68.0), step_m=1.0,
                       cfg: PitchControlConfig | None = None):
    """One frame (px, py, pvx, pvy are (slots,)): returns (xs, ys, P_home grid of shape (ny, nx))."""
    cfg = cfg or PitchControlConfig()
    team = np.asarray(team)
    xs = np.arange(step_m / 2, pitch[0], step_m)
    ys = np.arange(step_m / 2, pitch[1], step_m)
    gx, gy = np.meshgrid(xs, ys)
    flat_x, flat_y = gx.ravel(), gy.ravel()
    m = len(flat_x)
    rep = lambda v: np.broadcast_to(np.asarray(v, dtype=np.float64), (m, len(team)))  # noqa: E731
    t = player_times(rep(px), rep(py), rep(pvx), rep(pvy), flat_x, flat_y, cfg.ttb)
    th = np.min(np.where((team == HOME)[None, :], t, np.inf), axis=1)
    ta = np.min(np.where((team == AWAY)[None, :], t, np.inf), axis=1)
    return xs, ys, _logistic(th, ta, cfg.sigma_s).reshape(gx.shape)
