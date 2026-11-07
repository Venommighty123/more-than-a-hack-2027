"""Player-to-ball relations per frame: who is near the ball, how close the other team is, who is
closing in, how much pressure there is.

These are the inputs the first ideation round asked for (nearest per team, the gap between them,
closing speed, pressure counts). detectors/control.py already uses the nearest distances; this
module adds the rest and returns them as named per-frame columns for features/build.py.

All distances in metres, speeds in m/s. NaN / inf where the ball (or a whole team) is not seen.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

HOME, AWAY = 0, 1


@dataclass
class RelationalConfig:
    radii_m: tuple = (3.0, 5.0, 10.0)   # pressure counts within these distances of the ball
    chunk: int = 20000

    @classmethod
    def from_dict(cls, d: dict | None) -> "RelationalConfig":
        d = dict(d or {})
        if "radii_m" in d:
            d["radii_m"] = tuple(d["radii_m"])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict:
        return asdict(self)


def _two_nearest(d):
    """Smallest and second-smallest value per row, and the column of the smallest."""
    if d.shape[1] == 0:
        n = len(d)
        return np.full(n, np.inf), np.full(n, np.inf), np.full(n, -1)
    if d.shape[1] == 1:
        return d[:, 0], np.full(len(d), np.inf), np.zeros(len(d), dtype=int)
    part = np.partition(d, 1, axis=1)
    return part[:, 0], part[:, 1], np.argmin(d, axis=1)


def relational(px, py, bx, by, team, *, pvx=None, pvy=None, bvx=None, bvy=None,
               cfg: RelationalConfig | None = None) -> dict:
    """Named (n,) float32 columns. Velocities are optional (closing / relative speeds need them)."""
    cfg = cfg or RelationalConfig()
    team = np.asarray(team)
    n = len(bx)
    cols = {}
    names = []
    for side, tag in ((HOME, "home"), (AWAY, "away")):
        names += ["d1_" + tag, "d2_" + tag, "closing_" + tag, "relspeed_" + tag]
        names += ["n{}_{}".format(int(r), tag) for r in cfg.radii_m]
    for k in names:
        cols[k] = np.full(n, np.nan, dtype=np.float32)
    for a in range(0, n, cfg.chunk):
        b = min(n, a + cfg.chunk)
        with np.errstate(invalid="ignore"):
            dx, dy = px[a:b] - bx[a:b, None], py[a:b] - by[a:b, None]
            d = np.hypot(dx, dy)
        d = np.where(np.isnan(d), np.inf, d)
        for side, tag in ((HOME, "home"), (AWAY, "away")):
            m = np.flatnonzero(team == side)
            dm = d[:, m]
            d1, d2, j = _two_nearest(dm)
            cols["d1_" + tag][a:b] = d1
            cols["d2_" + tag][a:b] = d2
            for r in cfg.radii_m:
                cols["n{}_{}".format(int(r), tag)][a:b] = (dm <= r).sum(1)
            if pvx is not None and len(m):
                rows = np.arange(b - a)
                slot = m[np.clip(j, 0, None)]
                vx, vy = pvx[a:b][rows, slot], pvy[a:b][rows, slot]
                ux, uy = dx[rows, slot], dy[rows, slot]
                with np.errstate(invalid="ignore", divide="ignore"):
                    # closing speed: rate at which the nearest player approaches the ball (+ = closer)
                    cols["closing_" + tag][a:b] = -(vx * ux + vy * uy) / np.maximum(d1, 1e-3)
                    if bvx is not None:
                        cols["relspeed_" + tag][a:b] = np.hypot(bvx[a:b] - vx, bvy[a:b] - vy)
        seen = ~(np.isnan(bx[a:b]) | np.isnan(by[a:b]))
        for k in names:
            cols[k][a:b][~seen] = np.nan
    with np.errstate(invalid="ignore"):
        cols["gap_d1"] = (cols["d1_away"] - cols["d1_home"]).astype(np.float32)  # + = home closer
    return cols
