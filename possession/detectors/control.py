"""Who controls the ball on each frame -- the WHEN of a possession change.

Merges the three control rules the team wrote:

  method="distance"   Ashwyn's labeller_v1.control (best on the bench so far). A team controls the
                      ball when its nearest player is within ``r_m``, the other team's nearest is
                      at least ``margin_m`` further away, and the ball is slower than
                      ``max_ball_speed_ms``. gap_classification.control_team is the same rule
                      without the margin (margin_m = 0, contest_as_none).
  method="zone"       possession_test_1 / event_detection: the ball is inside a player's zone of
                      influence (features/zoi.py) AND moves about as fast as that player
                      (|v_ball - v_player| <= ``max_rel_speed_ms``). A fast ball crossing a zone
                      (a pass in flight) is not control. Both teams with the ball in their zones,
                      whatever its speed, is a contest.

Output per frame: HOME / AWAY (sole control), BOTH (contest), NONE (free ball, ball unseen).
This module does not decide possession: a touch, a deflection or a lost duel never confirms a
change by itself. That is model/resolve.py (confirm for N frames, then backdate the change to the
first frame of control, because spellF1 punishes lateness).

Untuned defaults; tune on the bench, never on BAR-ATH or the public 5 minutes (RULES.md).

    from possession.detectors.control import ControlConfig, detect_control
    res = detect_control(tr.px, tr.py, track.x, track.y, tr.teams.team, cfg,
                         ball_v=(track.vx, track.vy), player_v=(ps.vx, ps.vy))
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from possession.features.zoi import ZoiConfig, zoi_q
from possession.utils.frames import runs

HOME, AWAY, NONE, BOTH = 0, 1, -1, 2
NAMES = {HOME: "home", AWAY: "away", NONE: "none", BOTH: "contest"}


@dataclass
class ControlConfig:
    method: str = "distance"          # "distance" | "zone"
    # distance method (labeller_v1 defaults)
    r_m: float = 1.5                  # nearest player within this of the ball
    margin_m: float = 0.3             # ... and the other team's nearest this much further away
    max_ball_speed_ms: float = 12.0   # ball faster than this is in flight, nobody controls it
    # zone method (possession_test_1 defaults)
    max_rel_speed_ms: float = 4.0     # |v_ball - v_player| still counted as control
    zoi: ZoiConfig = field(default_factory=ZoiConfig)
    # both
    contest_as_none: bool = False     # True: report contests as NONE (labeller_v1 behaviour)
    chunk: int = 20000

    @classmethod
    def from_dict(cls, d: dict | None) -> "ControlConfig":
        d = dict(d or {})
        zoi = ZoiConfig.from_dict(d.pop("zoi", None))
        return cls(zoi=zoi, **{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ControlResult:
    ctrl: np.ndarray        # int8: HOME / AWAY / BOTH / NONE
    holder: np.ndarray      # int16 slot of the controlling player (sole control), else -1
    near_home: np.ndarray   # int16 nearest home slot to the ball (-1 if none / no ball)
    near_away: np.ndarray
    d_home: np.ndarray      # float32 distance (m) of the nearest home player, inf if unknown
    d_away: np.ndarray
    q_home: np.ndarray | None = None   # zone method: smallest zone distance per team
    q_away: np.ndarray | None = None
    config: ControlConfig | None = None

    @property
    def n(self) -> int:
        return len(self.ctrl)

    def share(self) -> dict:
        n = max(1, self.n)
        return {NAMES[k]: round(100.0 * float((self.ctrl == k).sum()) / n, 2) for k in NAMES}


def ball_speed_from_positions(bx, by, fps=25, half=4):
    """Fallback ball speed when no BallTrack is given (labeller_v1.ball_speed)."""
    v = np.full(len(bx), np.nan)
    if len(bx) > 2 * half:
        v[half:-half] = np.hypot(bx[2 * half:] - bx[:-2 * half],
                                 by[2 * half:] - by[:-2 * half]) * fps / (2 * half)
    return v


def _nearest(d, cols):
    """Smallest distance among ``cols`` and the slot index achieving it."""
    if not len(cols):
        n = len(d)
        return np.full(n, np.inf, dtype=np.float32), np.full(n, -1, dtype=np.int16)
    sub = d[:, cols]
    j = np.argmin(sub, axis=1)
    best = sub[np.arange(len(sub)), j].astype(np.float32)
    return best, np.where(np.isfinite(best), cols[j], -1).astype(np.int16)


def detect_control(px, py, bx, by, team, cfg: ControlConfig | None = None, *, ball_v=None,
                   player_v=None, fps: int = 25) -> ControlResult:
    """Per-frame control.

    px, py      (n, slots) player positions, metres (smoothed or raw)
    bx, by      (n,) ball positions, metres, NaN when unseen (use the cleaned BallTrack)
    team        (slots,) HOME / AWAY / UNKNOWN per slot; UNKNOWN slots never control
    ball_v      (vx, vy) of the ball; required by the zone method, speed source for distance
    player_v    (vx, vy) of the players, (n, slots); required by the zone method
    """
    cfg = cfg or ControlConfig()
    team = np.asarray(team)
    home_cols, away_cols = np.flatnonzero(team == HOME), np.flatnonzero(team == AWAY)
    n = len(bx)
    seen = ~(np.isnan(bx) | np.isnan(by))

    d_home = np.full(n, np.inf, dtype=np.float32)
    d_away = np.full(n, np.inf, dtype=np.float32)
    near_home = np.full(n, -1, dtype=np.int16)
    near_away = np.full(n, -1, dtype=np.int16)
    for a in range(0, n, cfg.chunk):
        b = min(n, a + cfg.chunk)
        with np.errstate(invalid="ignore"):
            d = np.hypot(px[a:b] - bx[a:b, None], py[a:b] - by[a:b, None])
        d = np.where(np.isnan(d), np.inf, d)
        d_home[a:b], near_home[a:b] = _nearest(d, home_cols)
        d_away[a:b], near_away[a:b] = _nearest(d, away_cols)

    ctrl = np.full(n, NONE, dtype=np.int8)
    holder = np.full(n, -1, dtype=np.int16)
    q_home = q_away = None

    if cfg.method == "distance":
        if ball_v is not None:
            speed = np.hypot(*ball_v)
        else:
            speed = ball_speed_from_positions(bx, by, fps)
        with np.errstate(invalid="ignore"):
            slow = ~(speed > cfg.max_ball_speed_ms)          # NaN speed counts as slow
        h = (d_home < cfg.r_m) & (d_away > d_home + cfg.margin_m) & slow & seen
        w = (d_away < cfg.r_m) & (d_home > d_away + cfg.margin_m) & slow & seen
        both = (d_home < cfg.r_m) & (d_away < cfg.r_m) & seen & ~h & ~w
        ctrl[h], ctrl[w], ctrl[both] = HOME, AWAY, BOTH
        holder[h], holder[w] = near_home[h], near_away[w]

    elif cfg.method == "zone":
        if ball_v is None or player_v is None:
            raise ValueError("the zone method needs ball_v and player_v")
        bvx, bvy = ball_v
        pvx, pvy = player_v
        q = zoi_q(px, py, pvx, pvy, bx, by, cfg.zoi)
        inside = q <= 1.0
        with np.errstate(invalid="ignore"):
            rel = np.hypot(bvx[:, None] - np.nan_to_num(pvx), bvy[:, None] - np.nan_to_num(pvy))
        slow = np.where(np.isnan(rel), True, rel <= cfg.max_rel_speed_ms)  # unknown -> position only
        controls = inside & slow
        h = controls[:, home_cols].any(axis=1) if len(home_cols) else np.zeros(n, bool)
        w = controls[:, away_cols].any(axis=1) if len(away_cols) else np.zeros(n, bool)
        contest = (inside[:, home_cols].any(axis=1) if len(home_cols) else np.zeros(n, bool)) & \
                  (inside[:, away_cols].any(axis=1) if len(away_cols) else np.zeros(n, bool))
        ctrl[h & ~w] = HOME
        ctrl[w & ~h] = AWAY
        ctrl[contest] = BOTH
        best = np.argmin(np.where(controls, q, np.inf), axis=1)
        sole = np.isin(ctrl, (HOME, AWAY))
        holder[sole] = best[sole]
        valid = team != -1
        q_home = np.min(np.where((team == HOME)[None, :], q, np.inf), axis=1) if valid.any() else None
        q_away = np.min(np.where((team == AWAY)[None, :], q, np.inf), axis=1) if valid.any() else None
    else:
        raise ValueError("method must be 'distance' or 'zone', got {!r}".format(cfg.method))

    if cfg.contest_as_none:
        ctrl[ctrl == BOTH] = NONE
    return ControlResult(ctrl, holder, near_home, near_away, d_home, d_away, q_home, q_away, cfg)


# ------------------------------------------------------------------------------------------------
# Spells of control (input to model/resolve.py and to the event detector)
# ------------------------------------------------------------------------------------------------

def control_spells(ctrl, min_frames=1, hole=0, lo=0):
    """Runs of sole control by one team, tolerating gaps of NONE up to ``hole`` frames.

    Returns dicts: team, start, end (exclusive, video frames with ``lo``), frames (controlled
    frames inside the spell). A contest (BOTH) or the other team's control always ends a spell.
    """
    ctrl = np.asarray(ctrl)
    spells = []
    for side in (HOME, AWAY):
        mine = ctrl == side
        if hole > 0:
            filled = mine.copy()
            s_none, e_none = runs(ctrl == NONE)
            for s, e in zip(s_none, e_none):
                if e - s <= hole and s > 0 and e < len(ctrl) and mine[s - 1] and mine[e]:
                    filled[s:e] = True
            mine_run = filled
        else:
            mine_run = mine
        for s, e in zip(*runs(mine_run)):
            k = int(mine[s:e].sum())
            if k >= min_frames:
                spells.append({"team": side, "start": int(s + lo), "end": int(e + lo), "frames": k})
    return sorted(spells, key=lambda r: r["start"])


def first_frame_of_control(ctrl, side, start, end):
    """First frame in [start, end) where ``side`` has sole control, or None (for backdating)."""
    idx = np.flatnonzero(np.asarray(ctrl[start:end]) == side)
    return int(start + idx[0]) if len(idx) else None
