"""Player track clean-up, velocities, team summaries and attacking direction.

Merges:
  smoothing             event_detection / possession_test_1 nan_smooth
  velocity              possession_test_1.velocity (centred difference) and labeller_v1.player_speed
                        (displacement over +/- w frames, jumps > 12 m/s treated as identity switches)
  mean speed            labeller_v1.player_speed (the dead-ball "players are slow" signal)
  team centroid         Krish's compute_player_centroid_possession (ball_reconstruction.py)
  attacking direction   pass_detection.attack_signs / to_attacking

Slots are tracklets, not people (DATA.md): an identity switch shows up as a position jump, which
is why speeds above ``max_speed_ms`` are discarded rather than trusted. Untuned defaults.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import warnings

import numpy as np

from .ball import nan_smooth, windowed_velocity

FPS = 25
PITCH = (105.3, 68.0)
HOME, AWAY = 0, 1


@dataclass
class PlayerConfig:
    smooth_half: int = 2            # centred moving average on positions (frames)
    speed_half_window: int = 12     # velocity from displacement over +/- this many frames
    max_speed_ms: float = 12.0      # faster than any footballer: an identity switch, not a sprint
    mean_speed_smooth_s: float = 1.0

    @classmethod
    def from_dict(cls, d: dict | None) -> "PlayerConfig":
        d = d or {}
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict:
        return asdict(self)


# ------------------------------------------------------------------------------------------------
# Per-slot signals
# ------------------------------------------------------------------------------------------------

def smooth_positions(px, py, half=2):
    return nan_smooth(px, half), nan_smooth(py, half)


def velocity(px, py, fps=FPS, half=12, max_speed_ms=12.0):
    """(vx, vy, speed) per frame and slot. Implausible speeds (identity switches) become NaN."""
    vx, vy = windowed_velocity(px, py, fps, half)
    speed = np.hypot(vx, vy)
    with np.errstate(invalid="ignore"):
        bad = speed > max_speed_ms
    vx[bad] = vy[bad] = speed[bad] = np.nan
    return vx, vy, speed


def _moving_average(values, width):
    if width <= 1:
        return values
    return np.convolve(values, np.ones(width) / width, mode="same")


def mean_speed(speed, slots=None, fps=FPS, smooth_s=1.0, fill=True):
    """Mean speed over the (selected) slots per frame, interpolated over empty frames, smoothed.

    ``slots`` is a bool mask over slots (e.g. ``teams.home``); None = all players.
    """
    s = speed if slots is None else speed[:, slots]
    with np.errstate(invalid="ignore"):
        cnt = (~np.isnan(s)).sum(1)
        m = np.where(cnt > 0, np.nansum(s, 1) / np.maximum(cnt, 1), np.nan)
    if fill:
        good = ~np.isnan(m)
        m = np.interp(np.arange(len(m)), np.flatnonzero(good), m[good]) if good.any() \
            else np.zeros(len(m))
        m = _moving_average(m, int(round(smooth_s * fps)))
    return m


def tracked_count(px, slots=None):
    ok = ~np.isnan(px if slots is None else px[:, slots])
    return ok.sum(1)


# ------------------------------------------------------------------------------------------------
# Team shape
# ------------------------------------------------------------------------------------------------

def centroid(px, py, slots):
    """(cx, cy) per frame over the selected slots; NaN when none is tracked."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(px[:, slots], 1), np.nanmean(py[:, slots], 1)


def spread(px, py, slots):
    """Root-mean-square distance of the team's players from their centroid (m)."""
    cx, cy = centroid(px, py, slots)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        d2 = (px[:, slots] - cx[:, None]) ** 2 + (py[:, slots] - cy[:, None]) ** 2
        return np.sqrt(np.nanmean(d2, 1))


# ------------------------------------------------------------------------------------------------
# Attacking direction
# ------------------------------------------------------------------------------------------------

def attack_signs(px, team, blocks, lo=0):
    """{(half, side): +1 if that side attacks towards +x in that half, else -1}.

    The team whose players average the lower x defends the x = 0 goal and attacks +x.
    ``blocks`` = [(half, first_video_frame, last_video_frame)], ``lo`` = video frame of row 0.
    """
    signs = {}
    for half, first, last in blocks:
        a, b = max(first - lo, 0), min(last - lo + 1, len(px))
        if b <= a:
            continue
        cols = [px[a:b][:, team == side] for side in (HOME, AWAY)]
        if not all(np.isfinite(c).any() for c in cols):
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            means = [np.nanmean(c) for c in cols]
        home_plus = means[HOME] < means[AWAY]
        signs[(half, HOME)] = 1 if home_plus else -1
        signs[(half, AWAY)] = -1 if home_plus else 1
    return signs


def attack_sign_per_frame(signs, blocks, n, side, lo=0):
    """int8 per row: +1 / -1 inside a half, 0 outside the halves."""
    out = np.zeros(n, dtype=np.int8)
    for half, first, last in blocks:
        a, b = max(first - lo, 0), min(last - lo + 1, n)
        if b > a and (half, side) in signs:
            out[a:b] = signs[(half, side)]
    return out


def to_attacking(x, y, sign, pitch=PITCH):
    """Rotate coordinates so the team attacks towards +x (works element-wise with array signs)."""
    length, width = pitch
    sign = np.asarray(sign)
    return np.where(sign < 0, length - x, x), np.where(sign < 0, width - y, y)


# ------------------------------------------------------------------------------------------------
# The pipeline
# ------------------------------------------------------------------------------------------------

@dataclass
class PlayerState:
    x: np.ndarray              # smoothed positions (n, slots), metres
    y: np.ndarray
    vx: np.ndarray             # m/s, NaN on identity switches
    vy: np.ndarray
    speed: np.ndarray
    mean_speed: np.ndarray     # (n,) all players, smoothed
    team_speed: dict           # side -> (n,)
    signs: dict                # (half, side) -> +1 / -1
    config: PlayerConfig


def prepare_players(px, py, team, blocks=None, lo=0, cfg: PlayerConfig | None = None,
                    fps: int = FPS) -> PlayerState:
    """Smooth -> velocity -> mean speeds -> attacking direction (if ``blocks`` is given)."""
    cfg = cfg or PlayerConfig()
    team = np.asarray(team)
    sx, sy = smooth_positions(px, py, cfg.smooth_half)
    vx, vy, sp = velocity(sx, sy, fps, cfg.speed_half_window, cfg.max_speed_ms)
    team_speed = {side: mean_speed(sp, team == side, fps, cfg.mean_speed_smooth_s)
                  for side in (HOME, AWAY)}
    return PlayerState(x=sx, y=sy, vx=vx, vy=vy, speed=sp,
                       mean_speed=mean_speed(sp, None, fps, cfg.mean_speed_smooth_s),
                       team_speed=team_speed,
                       signs=attack_signs(sx, team, blocks, lo) if blocks else {}, config=cfg)
