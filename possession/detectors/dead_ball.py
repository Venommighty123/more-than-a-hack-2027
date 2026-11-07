"""Dead-ball detector: the stretches where the ball is not in play (label D).

Ported from Ashwyn's labeller_v1.dead_mask (logic unchanged, parameters named), with the rules of
possession_test_1 / event_detection available as options:

  out of pitch     the (cleaned) ball is beyond a line by more than ``out_margin_m``
  stoppage gap     a ball gap of at least ``min_gap_s`` is dead when the ball vanished or came back
                   within ``line_m`` of a line, OR the players were slow through it (median mean
                   player speed < ``slow_speed_ms``; the gap audit found dead gaps ~1.1 m/s, live
                   gaps faster). D lasts from the moment the ball vanished to the RESTART: the first
                   frame after it reappears where it moves faster than ``restart_speed_ms`` (it often
                   sits at the restarter's feet first), searched for at most ``restart_search_s``.
  long gap         (option, possession_test_1) any gap of at least ``long_gap_s`` is dead
  join             live slivers shorter than ``join_s`` between two dead stretches are noise

Why it matters: spellF1 scores possessions, and ``H H D D H H`` is two of them. A missed stoppage
merges two true possessions into one prediction that matches neither (game plan, "Dead balls
split possessions").

The second-half pre-kickoff stretch (published in FAQ.md) is not detected here: scripts/run.py
writes it as D directly. Untuned defaults; tune on the bench (RULES.md).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from possession.utils.frames import runs

FPS = 25
PITCH = (105.3, 68.0)

# reason codes per frame (0 = alive)
ALIVE, OUT_OF_PITCH, GAP_NEAR_LINE, GAP_SLOW, LONG_GAP, JOINED = 0, 1, 2, 3, 4, 5
REASONS = {ALIVE: "", OUT_OF_PITCH: "out_of_pitch", GAP_NEAR_LINE: "gap_near_line",
           GAP_SLOW: "gap_players_slow", LONG_GAP: "long_gap", JOINED: "joined"}


@dataclass
class DeadBallConfig:
    out_margin_m: float = 0.5
    min_gap_s: float = 1.5
    line_m: float = 2.0
    slow_speed_ms: float = 1.8
    restart_speed_ms: float = 5.0
    restart_search_s: float = 30.0
    long_gap_s: float | None = None     # possession_test_1 used 10.0; None = off (labeller_v1)
    join_s: float = 1.0

    @classmethod
    def from_dict(cls, d: dict | None) -> "DeadBallConfig":
        d = d or {}
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict:
        return asdict(self)


def _near_line(x, y, pitch, line_m):
    if np.isnan(x) or np.isnan(y):
        return False
    length, width = pitch
    return x < line_m or x > length - line_m or y < line_m or y > width - line_m


def dead_ball(bx, by, ball_speed, mean_speed, lo, hi, cfg: DeadBallConfig | None = None,
              fps: int = FPS, pitch=PITCH):
    """Dead-ball mask over the whole array, with gaps considered only inside rows lo..hi.

    bx, by        (n,) cleaned ball, metres (NaN = missing; short gaps already bridged)
    ball_speed    (n,) m/s, e.g. BallTrack.speed (displacement over +/- 4 frames)
    mean_speed    (n,) mean player speed, e.g. PlayerState.mean_speed / players.mean_speed
    lo, hi        row range of one graded block (inclusive), usually kickoff .. end of half
    Returns (dead bool (n,), reason int8 (n,)).
    """
    cfg = cfg or DeadBallConfig()
    n = len(bx)
    length, width = pitch
    dead = np.zeros(n, dtype=bool)
    reason = np.zeros(n, dtype=np.int8)

    seen = ~np.isnan(bx)
    with np.errstate(invalid="ignore"):
        out = ((bx < -cfg.out_margin_m) | (bx > length + cfg.out_margin_m) |
               (by < -cfg.out_margin_m) | (by > width + cfg.out_margin_m)) & seen
    dead |= out
    reason[out] = OUT_OF_PITCH

    miss = np.isnan(bx).copy()
    miss[:lo] = False
    miss[hi + 1:] = False
    min_len = cfg.min_gap_s * fps
    for s, e in zip(*runs(miss)):
        if e - s < min_len:
            continue
        before = (bx[s - 1], by[s - 1]) if s > 0 else (np.nan, np.nan)
        after = (bx[e], by[e]) if e < n else (np.nan, np.nan)
        line = _near_line(*before, pitch, cfg.line_m) or _near_line(*after, pitch, cfg.line_m)
        slow = np.median(mean_speed[s:e]) < cfg.slow_speed_ms
        long_gap = cfg.long_gap_s is not None and e - s >= cfg.long_gap_s * fps
        if not (line or slow or long_gap):
            continue
        # restart: first frame after the ball reappears where it moves
        r = e
        lim = min(n, hi + 1, e + int(cfg.restart_search_s * fps))
        while r < lim and not np.isnan(bx[r]) and not (ball_speed[r] > cfg.restart_speed_ms):
            r += 1
        why = GAP_NEAR_LINE if line else GAP_SLOW if slow else LONG_GAP
        reason[s:r][~dead[s:r]] = why
        dead[s:r] = True

    # a stoppage is one stretch: alive slivers shorter than join_s inside it are noise
    for s, e in zip(*runs(~dead)):
        if s > 0 and e < n and dead[s - 1] and dead[e] and e - s < cfg.join_s * fps:
            dead[s:e] = True
            reason[s:e] = JOINED
    return dead, reason


def stoppages(dead, reason=None, lo=0):
    """Dead stretches as dicts: start, end (exclusive, video frames with ``lo``), seconds, reason."""
    rows = []
    for s, e in zip(*runs(dead)):
        why = ""
        if reason is not None:
            vals = reason[s:e][reason[s:e] > 0]
            why = REASONS[int(np.bincount(vals).argmax())] if len(vals) else ""
        rows.append({"start": int(s + lo), "end": int(e + lo), "seconds": (e - s) / FPS,
                     "reason": why})
    return rows
