"""Zone of influence (ZoI): the area around a player where they can play the ball.

Merges possession_test_1.zoi_geometry / zoi_q and event_detection.zoi_q (same model, two copies).

The zone is an ellipse centred on the player and oriented along their velocity. At standstill it
is a circle of radius ``r0_m``. As speed rises towards ``speed_max_ms`` it stretches forward and
shrinks sideways and backwards:

    front = r0 * (1 + front_gain * s / smax)
    rear  = r0 * (1 - rear_shrink * s / smax)
    side  = r0 * (1 - side_shrink * s / smax)

``q`` is the normalised zone distance of the ball, (along / r_along)^2 + (lateral / side)^2:
q <= 1 means the ball is inside the zone. It is +inf where the player or the ball is not seen.
Untuned defaults; tune on the bench (RULES.md).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class ZoiConfig:
    r0_m: float = 1.5             # radius at standstill
    speed_max_ms: float = 8.0     # speed at which the deformation is complete
    front_gain: float = 0.8
    side_shrink: float = 0.35
    rear_shrink: float = 0.5
    moving_min_ms: float = 0.2    # below this the heading is noise: use a circle
    chunk: int = 20000            # frames per block, to bound memory on the full match

    @classmethod
    def from_dict(cls, d: dict | None) -> "ZoiConfig":
        d = d or {}
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict:
        return asdict(self)


def zoi_geometry(vx, vy, cfg: ZoiConfig | None = None):
    """Unit heading (ux, uy) and front / rear / side radii in metres, per frame and slot."""
    cfg = cfg or ZoiConfig()
    with np.errstate(invalid="ignore"):
        speed = np.nan_to_num(np.hypot(vx, vy))
    ratio = np.clip(speed / cfg.speed_max_ms, 0.0, 1.0)
    moving = speed > cfg.moving_min_ms
    ux = np.where(moving, np.nan_to_num(vx) / np.maximum(speed, 1e-6), 1.0)
    uy = np.where(moving, np.nan_to_num(vy) / np.maximum(speed, 1e-6), 0.0)
    front = cfg.r0_m * (1.0 + cfg.front_gain * ratio)
    rear = cfg.r0_m * (1.0 - cfg.rear_shrink * ratio)
    side = cfg.r0_m * (1.0 - cfg.side_shrink * ratio)
    return ux, uy, front, rear, side


def _q_block(px, py, vx, vy, bx, by, cfg):
    ux, uy, front, rear, side = zoi_geometry(vx, vy, cfg)
    with np.errstate(invalid="ignore", divide="ignore"):
        dx, dy = bx[:, None] - px, by[:, None] - py
        along = dx * ux + dy * uy
        lateral = -dx * uy + dy * ux
        q = (along / np.where(along >= 0, front, rear)) ** 2 + (lateral / side) ** 2
    return np.nan_to_num(q, nan=np.inf)


def zoi_q(px, py, vx, vy, bx, by, cfg: ZoiConfig | None = None) -> np.ndarray:
    """(frames, slots) float32 normalised zone distance of the ball; <= 1 inside, inf unseen."""
    cfg = cfg or ZoiConfig()
    n = len(bx)
    q = np.full(np.shape(px), np.inf, dtype=np.float32)
    for a in range(0, n, cfg.chunk):
        b = min(n, a + cfg.chunk)
        q[a:b] = _q_block(px[a:b], py[a:b], vx[a:b], vy[a:b], bx[a:b], by[a:b], cfg)
    return q


def zoi_inside(px, py, vx, vy, bx, by, cfg: ZoiConfig | None = None) -> np.ndarray:
    """(frames, slots) bool: is the ball inside each player's zone?"""
    return zoi_q(px, py, vx, vy, bx, by, cfg) <= 1.0


def team_min_q(q, team, side):
    """(frames,) smallest zone distance among one team's slots, and the slot that achieves it."""
    cols = np.flatnonzero(np.asarray(team) == side)
    if not len(cols):
        n = len(q)
        return np.full(n, np.inf, dtype=np.float32), np.full(n, -1, dtype=np.int16)
    sub = q[:, cols]
    j = np.argmin(sub, axis=1)
    best = sub[np.arange(len(sub)), j]
    return best, np.where(np.isfinite(best), cols[j], -1).astype(np.int16)


def zoi_polygon(x, y, vx, vy, cfg: ZoiConfig | None = None, points: int = 24):
    """Outline of one player's zone as (points, 2) metres -- for drawing (viz/)."""
    ux, uy, front, rear, side = (np.asarray(v).item() for v in zoi_geometry(
        np.array([vx]), np.array([vy]), cfg))
    t = np.linspace(0, 2 * np.pi, points, endpoint=False)
    along = np.where(np.cos(t) >= 0, front, rear) * np.cos(t)
    lateral = side * np.sin(t)
    return np.stack([x + along * ux - lateral * uy, y + along * uy + lateral * ux], axis=1)
