"""Time for each player to reach the ball (a simple intercept model), per frame.

A distance treats a player sprinting at the ball and one running away from it the same. The
time-to-ball does not: each player keeps running with their current velocity for a reaction time,
then runs straight at the ball at top speed (Spearman 2017, the usual simplification):

    t_i = reaction_s + | ball - (p_i + v_i * reaction_s) | / max_speed_ms

Returned per frame: the smallest time per team, the slot achieving it, and the difference
(+ = home gets there first). The ball is taken where it is now (no ball flight model). Untuned
defaults from the literature: reaction 0.7 s, top speed 7 m/s.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

HOME, AWAY = 0, 1


@dataclass
class TimeToBallConfig:
    reaction_s: float = 0.7
    max_speed_ms: float = 7.0
    chunk: int = 20000

    @classmethod
    def from_dict(cls, d: dict | None) -> "TimeToBallConfig":
        d = d or {}
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict:
        return asdict(self)


def player_times(px, py, pvx, pvy, bx, by, cfg: TimeToBallConfig | None = None) -> np.ndarray:
    """(n, slots) seconds to reach the ball; inf where the player or the ball is unseen."""
    cfg = cfg or TimeToBallConfig()
    n = len(bx)
    out = np.full(np.shape(px), np.inf, dtype=np.float32)
    for a in range(0, n, cfg.chunk):
        b = min(n, a + cfg.chunk)
        rx = px[a:b] + np.nan_to_num(pvx[a:b]) * cfg.reaction_s
        ry = py[a:b] + np.nan_to_num(pvy[a:b]) * cfg.reaction_s
        with np.errstate(invalid="ignore"):
            t = cfg.reaction_s + np.hypot(bx[a:b, None] - rx, by[a:b, None] - ry) / cfg.max_speed_ms
        out[a:b] = np.where(np.isnan(t), np.inf, t)
    return out


def team_times(times, team) -> dict:
    """Per-frame minimum time per team, the slot, and the difference (+ = home first)."""
    team = np.asarray(team)
    cols = {}
    for side, tag in ((HOME, "home"), (AWAY, "away")):
        m = np.flatnonzero(team == side)
        if not len(m):
            cols["ttb_" + tag] = np.full(len(times), np.inf, dtype=np.float32)
            cols["ttb_slot_" + tag] = np.full(len(times), -1, dtype=np.int16)
            continue
        j = np.argmin(times[:, m], axis=1)
        t = times[np.arange(len(times)), m[j]]
        cols["ttb_" + tag] = t.astype(np.float32)
        cols["ttb_slot_" + tag] = np.where(np.isfinite(t), m[j], -1).astype(np.int16)
    with np.errstate(invalid="ignore"):
        cols["ttb_diff"] = (cols["ttb_away"] - cols["ttb_home"]).astype(np.float32)
    return cols


def time_to_ball(px, py, pvx, pvy, bx, by, team, cfg: TimeToBallConfig | None = None) -> dict:
    return team_times(player_times(px, py, pvx, pvy, bx, by, cfg), team)
