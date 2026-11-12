"""Possession labeller v6 (Ashwyn, Nov 8-10), rebuilt on the shared modules. Submission candidate.

Original: backup/ashwyn_labeller_v6.py, run by backup/ashwyn_run_labeller_v6.py (Ashwyn's branch,
commit 17685a6 "labeller v6 confirmed"). v6 = v4 plus three changes aimed at false possession
changes and false stoppages (diagnosis and numbers: docs/changelog/labeller.md, section v6):

  1. SMOOTH_K   centred, NaN-aware moving average of ball and player positions over +/- SMOOTH_K
                frames before anything else (0 = off). The ~1 m ball offset wobbles frame to frame;
                smoothing stops the nearest player from flickering.
  2. SW_MARGIN  hysteresis: to TAKE the ball from the team in possession, the other team must be
                SW_MARGIN m clearer (MARGIN still applies to keeping it). None = off.
                Implemented as model/resolve.py's ``ctrl_sw``.
  3. MIN_D      a D stretch shorter than MIN_D s is live play (0 = off). MIN_D_KEEP_OUT keeps short
                stretches where the ball is off the pitch (default False: all are dropped).

Defaults SMOOTH_K=3, SW_MARGIN=0.6, MIN_D=2.0 (Ashwyn's Nov 8 bench verdict). With
``params=dict(SMOOTH_K=0, SW_MARGIN=None, MIN_D=0.0)`` it is exactly the package's v4.

Bench (Ashwyn, mean of Metrica games 1-3, spellF1): event truth real 0.542 / real_occ 0.478,
tracking truth 0.596 / 0.525 (v4: 0.516 / 0.462 / 0.563 / 0.518). DFL (7 matches): 0.464 vs v4
0.468, a tie. Rejected v6 switches (WEAK_MERGE, FOLLOW, density switching, gap split, gap start
delay, shorter walk-back) are NOT ported; they live in backup/ashwyn_labeller_v6.py.

    label(px, py, ball, team, blocks, params=None, *, fps=25, pitch=(105, 68), tags=None)
"""
from __future__ import annotations

import numpy as np

from possession.detectors import stoppage
from possession.detectors.control import detect_control
from possession.model.resolve import resolve_block
from possession.preprocess.ball import clean_ball
from possession.utils.frames import runs

from . import v4

DEFAULTS = dict(v4.DEFAULTS)
DEFAULTS.update(
    SMOOTH_K=3,          # +/- frames of position smoothing (0 = off)
    SW_MARGIN=0.6,       # m clearer needed to take the ball (None = off)
    MIN_D=2.0,           # s, shortest stoppage kept (0 = off)
    MIN_D_KEEP_OUT=False,
)
LETTERS = v4.LETTERS
AS_V4 = dict(SMOOTH_K=0, SW_MARGIN=None, MIN_D=0.0)


def smooth(a, k):
    """NaN-aware centred moving average along axis 0; missing frames stay missing."""
    if k <= 0:
        return a
    ok = ~np.isnan(a)
    z = np.where(ok, a, 0.0)
    pad = np.zeros((1,) + a.shape[1:])
    cs = np.cumsum(np.concatenate([pad, z]), 0)
    cn = np.cumsum(np.concatenate([pad, ok.astype(float)]), 0)
    n = len(a)
    lo = np.clip(np.arange(n) - k, 0, n)
    hi = np.clip(np.arange(n) + k + 1, 0, n)
    s = cs[hi] - cs[lo]
    c = cn[hi] - cn[lo]
    return np.where(ok, s / np.maximum(c, 1), np.nan)


def drop_short_dead(dead, b, p, lo, hi, fps=25, pitch=(105.0, 68.0)):
    """MIN_D: D stretches shorter than MIN_D s inside the block become live play."""
    d = dead.copy()
    if not p["MIN_D"]:
        return d
    L, W = pitch
    m = p["OUT_M"]
    with np.errstate(invalid="ignore"):
        out = ~np.isnan(b[:, 0]) & ((b[:, 0] < -m) | (b[:, 0] > L + m) |
                                    (b[:, 1] < -m) | (b[:, 1] > W + m))
    for s, e in zip(*runs(d)):
        if s < lo or e > hi + 1:
            continue
        if e - s < p["MIN_D"] * fps and not (p["MIN_D_KEEP_OUT"] and out[s:e].any()):
            d[s:e] = False
    return d


def run(px, py, ball, team, blocks, params=None, *, fps=25, pitch=(105.0, 68.0), tags=None):
    """Labels plus the intermediate signals (ball track, control, rel speed, dead mask)."""
    p = dict(DEFAULTS, **(params or {}))
    team = np.asarray(team)
    if p["SMOOTH_K"]:
        k = int(p["SMOOTH_K"])
        px, py, ball = smooth(px, k), smooth(py, k), smooth(ball, k)
    track = clean_ball(ball[:, 0], ball[:, 1], v4.ball_config(p), fps=fps, pitch=pitch)
    b = track.xy
    ccfg = v4.control_config(p)
    ctrl = detect_control(px, py, track.x, track.y, team, ccfg,
                          ball_v=(track.vx, track.vy), fps=fps)
    ctrl_sw = None
    if p["SW_MARGIN"] is not None:
        ccfg.margin_m = p["SW_MARGIN"]
        ctrl_sw = detect_control(px, py, track.x, track.y, team, ccfg,
                                 ball_v=(track.vx, track.vy), fps=fps).ctrl
    rel = stoppage.rel_speed(px, py, blocks, p, fps)
    out = np.full(len(ball), "X", dtype="<U1")
    dead_all = np.zeros(len(ball), dtype=bool)
    for _, lo, hi in blocks:
        lo, hi = int(lo), int(hi)
        dead = stoppage.dead_mask(b, rel, p, lo, hi, tags if p["USE_TAGS"] else None, fps, pitch)
        dead = drop_short_dead(dead, b, p, lo, hi, fps, pitch)
        codes = resolve_block(ctrl.ctrl, dead, lo, hi, v4.resolve_config(p), fps, ctrl_sw=ctrl_sw)
        out[lo:hi + 1] = LETTERS[codes]
        dead_all[lo:hi + 1] = dead[lo:hi + 1]
    return {"labels": out, "ball": track, "control": ctrl, "ctrl_sw": ctrl_sw, "rel": rel,
            "dead": dead_all, "params": p}


def label(px, py, ball, team, blocks, params=None, *, fps=25, pitch=(105.0, 68.0), tags=None):
    return run(px, py, ball, team, blocks, params, fps=fps, pitch=pitch, tags=tags)["labels"]
