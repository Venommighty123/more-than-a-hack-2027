"""Ball track clean-up and kinematics.

Merges what was copied across the repository:

  spikes / teleports  labeller_v1.clean_ball (two-sided spike), event_detection.remove_teleports
                      and pass_detection.clean_ball (one-sided speed ceiling)
  short-gap bridging  labeller_v1.clean_ball, event_detection.fill_short_gaps
  smoothing           event_detection / possession_test_1 nan_smooth
  speed               labeller_v1.ball_speed (displacement over +/- w frames: one-frame
                      differences turn 0.3 m of detection jitter into ~5 m/s of fake speed)
  touch signal        pass_detection.ball_kinematics (change of the VELOCITY VECTOR across a
                      frame; heading alone is noise when the ball is slow)
  out of pitch        labeller_v1.dead_mask / event_detection OUT_MARGIN_M

Long gaps are NOT filled here: that is a modelling decision (preprocess/reconstruct.py and
detectors/gap_fill.py). Every threshold is an untuned starting point -- tune on the bench, never
on the public 5 minutes (RULES.md).

    from possession.preprocess import clean_ball
    track = clean_ball(tr.bx, tr.by, fps=tr.fps, pitch=tr.pitch)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

FPS = 25
PITCH = (105.3, 68.0)


@dataclass
class BallConfig:
    max_step_m: float = 2.0         # a point whose jump in AND out exceeds this (per frame) is a spike
    max_speed_ms: float | None = 40.0   # ceiling on frame-to-frame speed; above = teleport (None = off)
    bridge_frames: int = 6          # interior gaps up to this many frames are linearly bridged
    speed_half_window: int = 4      # speed from displacement over +/- this many frames
    touch_half_window: int = 3      # touch = |v(t+h) - v(t-h)|
    smooth_half: int = 0            # optional centred moving average before velocities (0 = off)
    out_margin_m: float = 0.5       # ball beyond a line by more than this = outside the pitch

    @classmethod
    def from_dict(cls, d: dict | None) -> "BallConfig":
        d = d or {}
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict:
        return asdict(self)


# ------------------------------------------------------------------------------------------------
# Building blocks
# ------------------------------------------------------------------------------------------------

def runs(mask) -> tuple[np.ndarray, np.ndarray]:
    """(starts, ends) of the True runs of ``mask``; ends are exclusive."""
    e = np.diff(np.concatenate(([0], np.asarray(mask, dtype=np.int8), [0])))
    return np.flatnonzero(e == 1), np.flatnonzero(e == -1)


def remove_spikes(x, y, max_step_m=2.0):
    """NaN out isolated detections that jump away and back (false blobs, one-frame swaps).

    A point is a spike when the jump from the previous observed point AND to the next one both
    exceed ``max_step_m`` per frame elapsed. A genuine fast kick only has one big jump.
    Returns (x, y, removed_mask).
    """
    x, y = np.array(x, dtype=np.float64), np.array(y, dtype=np.float64)
    removed = np.zeros(len(x), dtype=bool)
    idx = np.flatnonzero(~(np.isnan(x) | np.isnan(y)))
    if len(idx) > 2:
        step = np.hypot(np.diff(x[idx]), np.diff(y[idx])) / np.maximum(1, np.diff(idx))
        prev = np.r_[np.inf, step]
        nxt = np.r_[step, np.inf]
        spike = (prev > max_step_m) & (nxt > max_step_m)
        removed[idx[spike]] = True
        x[removed] = np.nan
        y[removed] = np.nan
    return x, y, removed


def remove_teleports(x, y, fps=FPS, max_speed_ms=40.0):
    """NaN out frames reached at an impossible speed from the previous frame. Returns (x, y, mask)."""
    x, y = np.array(x, dtype=np.float64), np.array(y, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        step = np.hypot(np.diff(x, prepend=np.nan), np.diff(y, prepend=np.nan)) * fps
        jump = step > max_speed_ms
    x[jump] = np.nan
    y[jump] = np.nan
    return x, y, jump


def bridge_gaps(x, y, max_gap):
    """Linearly fill interior NaN runs of at most ``max_gap`` frames. Returns (x, y, bridged)."""
    x, y = np.array(x, dtype=np.float64), np.array(y, dtype=np.float64)
    bridged = np.zeros(len(x), dtype=bool)
    starts, ends = runs(np.isnan(x) | np.isnan(y))
    for s, e in zip(starts, ends):
        if s == 0 or e >= len(x) or e - s > max_gap:
            continue
        t = np.arange(s, e)
        x[s:e] = np.interp(t, [s - 1, e], [x[s - 1], x[e]])
        y[s:e] = np.interp(t, [s - 1, e], [y[s - 1], y[e]])
        bridged[s:e] = True
    return x, y, bridged


def nan_smooth(values, half):
    """Centred moving average along axis 0 that ignores NaN. NaN stays NaN."""
    values = np.asarray(values)
    if half <= 0:
        return values
    ok = ~np.isnan(values)
    filled = np.where(ok, values, 0.0).astype(np.float64)
    pad = [(half + 1, half)] + [(0, 0)] * (values.ndim - 1)
    csum = np.cumsum(np.pad(filled, pad), axis=0)
    ccnt = np.cumsum(np.pad(ok.astype(np.float64), pad), axis=0)
    width = 2 * half + 1
    total, count = csum[width:] - csum[:-width], ccnt[width:] - ccnt[:-width]
    out = np.where(count > 0, total / np.maximum(count, 1), np.nan)
    return np.where(ok, out, np.nan)


def windowed_velocity(x, y, fps=FPS, half=4):
    """(vx, vy) in m/s from the displacement between t-half and t+half. NaN at the edges and
    wherever either end of the window is missing. Works on (n,) or (n, slots) arrays."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    vx, vy = np.full(x.shape, np.nan), np.full(y.shape, np.nan)
    if half < 1 or len(x) <= 2 * half:
        return vx, vy
    k = fps / (2.0 * half)
    vx[half:-half] = (x[2 * half:] - x[:-2 * half]) * k
    vy[half:-half] = (y[2 * half:] - y[:-2 * half]) * k
    return vx, vy


def speed_of(vx, vy):
    return np.hypot(vx, vy)


def heading_of(vx, vy):
    """Direction of travel in radians (atan2(vy, vx)); meaningless when the speed is tiny."""
    return np.arctan2(vy, vx)


def touch_signal(vx, vy, half=3):
    """|v(t+half) - v(t-half)| in m/s: large when something changed the ball's velocity."""
    vx, vy = np.asarray(vx), np.asarray(vy)
    out = np.full(len(vx), np.nan)
    if half >= 1 and len(vx) > 2 * half:
        out[half:-half] = np.hypot(vx[2 * half:] - vx[:-2 * half], vy[2 * half:] - vy[:-2 * half])
    return out


def acceleration(speed, fps=FPS, half=2):
    """d(speed)/dt in m/s^2, centred over +/- half frames."""
    speed = np.asarray(speed, dtype=np.float64)
    out = np.full(len(speed), np.nan)
    if half >= 1 and len(speed) > 2 * half:
        out[half:-half] = (speed[2 * half:] - speed[:-2 * half]) * fps / (2.0 * half)
    return out


def outside_pitch(x, y, pitch=PITCH, margin=0.5):
    """True where the ball is beyond a touchline or goal line by more than ``margin``. NaN -> False."""
    length, width = pitch
    with np.errstate(invalid="ignore"):
        return (x < -margin) | (x > length + margin) | (y < -margin) | (y > width + margin)


def distance_to_lines(x, y, pitch=PITCH):
    """Distance in metres to the nearest touchline or goal line (negative = outside)."""
    length, width = pitch
    return np.minimum(np.minimum(x, length - x), np.minimum(y, width - y))


def gap_table(present, lo=0, min_len=1):
    """Missing-ball runs as dicts: start / end (video frames, end exclusive) and length."""
    starts, ends = runs(~np.asarray(present, dtype=bool))
    return [{"start": int(s + lo), "end": int(e + lo), "length": int(e - s)}
            for s, e in zip(starts, ends) if e - s >= min_len]


# ------------------------------------------------------------------------------------------------
# The pipeline
# ------------------------------------------------------------------------------------------------

@dataclass
class BallTrack:
    x: np.ndarray              # cleaned + short gaps bridged, metres
    y: np.ndarray
    observed: np.ndarray       # the tracker saw the ball on this frame (before cleaning)
    removed: np.ndarray        # dropped as a spike or teleport
    bridged: np.ndarray        # filled by short-gap interpolation
    vx: np.ndarray             # m/s
    vy: np.ndarray
    speed: np.ndarray
    touch: np.ndarray          # m/s change of velocity vector
    outside: np.ndarray        # beyond a line by more than out_margin_m
    config: BallConfig

    @property
    def present(self) -> np.ndarray:
        """A usable position (observed and kept, or bridged)."""
        return ~(np.isnan(self.x) | np.isnan(self.y))

    @property
    def xy(self) -> np.ndarray:
        return np.stack([self.x, self.y], axis=1)

    def summary(self) -> dict:
        n = max(1, len(self.x))
        return {"frames": len(self.x), "observed_pct": round(100.0 * float(self.observed.sum()) / n, 2),
                "present_pct": round(100.0 * float(self.present.sum()) / n, 2),
                "removed": int(self.removed.sum()), "bridged": int(self.bridged.sum()),
                "outside": int(self.outside.sum())}


def clean_ball(bx, by, cfg: BallConfig | None = None, fps: int = FPS, pitch=PITCH) -> BallTrack:
    """Spikes -> teleports -> short-gap bridge -> (optional smoothing) -> kinematics."""
    cfg = cfg or BallConfig()
    observed = ~(np.isnan(bx) | np.isnan(by))
    x, y, spikes = remove_spikes(bx, by, cfg.max_step_m)
    if cfg.max_speed_ms:
        x, y, jumps = remove_teleports(x, y, fps, cfg.max_speed_ms)
    else:
        jumps = np.zeros(len(x), dtype=bool)
    x, y, bridged = bridge_gaps(x, y, cfg.bridge_frames)
    sx, sy = (nan_smooth(x, cfg.smooth_half), nan_smooth(y, cfg.smooth_half)) \
        if cfg.smooth_half > 0 else (x, y)
    vx, vy = windowed_velocity(sx, sy, fps, cfg.speed_half_window)
    return BallTrack(x=x, y=y, observed=observed, removed=spikes | jumps, bridged=bridged,
                     vx=vx, vy=vy, speed=speed_of(vx, vy),
                     touch=touch_signal(vx, vy, cfg.touch_half_window),
                     outside=outside_pitch(x, y, pitch, cfg.out_margin_m), config=cfg)
