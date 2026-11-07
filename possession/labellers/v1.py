"""Possession labeller v1 (Ashwyn), rebuilt on the shared modules. Per-frame H / A / D from
tracking geometry alone.

Recipe, one module per step (each answers one rule of CHALLENGE.md):

  1. clean the ball           preprocess/ball.py      spikes out, holes <= BRIDGE frames bridged
  2. control per frame        detectors/control.py    distance method: nearest within CTRL_R m,
                                                      other team MARGIN m further, ball < CTRL_V
  3. dead ball                detectors/dead_ball.py  out of pitch; gaps >= DEAD_MIN s near a
                                                      line or with slow players, until restart
  4. possessions              model/resolve.py        confirm CONFIRM frames (HOLE allowed),
                                                      backdate; merge spells < MIN_SPELL s

DEFAULTS are Ashwyn's values, tuned ONLY on the Metrica bench (docs/changelog/labeller.md). With
``pitch=(105.0, 68.0)`` the output is identical to the original labeller_v1.label (checked).
For BAR-ATH scripts/run.py passes the real pitch (105.3 x 68.0); that only moves the touchline
tests by 30 cm. Change a value only with a bench score, and log it in the changelog.
"""
from __future__ import annotations

import numpy as np

from possession.detectors.control import ControlConfig, detect_control
from possession.detectors.dead_ball import DeadBallConfig, dead_ball
from possession.model.resolve import ResolveConfig, resolve_block
from possession.preprocess.ball import BallConfig, clean_ball
from possession.preprocess.players import mean_speed, velocity

DEFAULTS = dict(
    MAX_STEP=2.0, BRIDGE=6,
    CTRL_R=1.5, MARGIN=0.3, CTRL_V=12.0,
    CONFIRM=6, HOLE=3,
    DEAD_MIN=1.5, DEAD_LINE=2.0, DEAD_SPEED=1.8, RESTART_V=5.0, OUT_M=0.5, D_JOIN=1.0,
    MIN_SPELL=2.0,
)
LETTERS = np.array(["H", "A", "D"])


def configs(params=None):
    """The v1 parameter dict mapped onto the module configs."""
    p = dict(DEFAULTS, **(params or {}))
    ball = BallConfig(max_step_m=p["MAX_STEP"], max_speed_ms=None, bridge_frames=int(p["BRIDGE"]),
                      speed_half_window=4, out_margin_m=p["OUT_M"])
    control = ControlConfig(method="distance", r_m=p["CTRL_R"], margin_m=p["MARGIN"],
                            max_ball_speed_ms=p["CTRL_V"], contest_as_none=True)
    dead = DeadBallConfig(out_margin_m=p["OUT_M"], min_gap_s=p["DEAD_MIN"], line_m=p["DEAD_LINE"],
                          slow_speed_ms=p["DEAD_SPEED"], restart_speed_ms=p["RESTART_V"],
                          join_s=p["D_JOIN"], long_gap_s=None)
    res = ResolveConfig(confirm=int(p["CONFIRM"]), hole=int(p["HOLE"]), min_spell_s=p["MIN_SPELL"])
    return ball, control, dead, res


def run(px, py, ball, team, blocks, params=None, *, fps=25, pitch=(105.0, 68.0)):
    """Everything v1 computes: labels plus the intermediate signals, for debugging and viz."""
    ball_cfg, ctrl_cfg, dead_cfg, res_cfg = configs(params)
    team = np.asarray(team)
    track = clean_ball(ball[:, 0], ball[:, 1], ball_cfg, fps=fps, pitch=pitch)
    ctrl = detect_control(px, py, track.x, track.y, team, ctrl_cfg,
                          ball_v=(track.vx, track.vy), fps=fps)
    _, _, sp = velocity(px, py, fps, half=12, max_speed_ms=12.0)
    ms = mean_speed(sp, None, fps, smooth_s=1.0)
    out = np.full(len(ball), "X", dtype="<U1")
    dead_all = np.zeros(len(ball), dtype=bool)
    reason_all = np.zeros(len(ball), dtype=np.int8)
    for _, lo, hi in blocks:
        lo, hi = int(lo), int(hi)
        dead, reason = dead_ball(track.x, track.y, track.speed, ms, lo, hi, dead_cfg, fps, pitch)
        codes = resolve_block(ctrl.ctrl, dead, lo, hi, res_cfg, fps)
        out[lo:hi + 1] = LETTERS[codes]
        dead_all[lo:hi + 1] = dead[lo:hi + 1]
        reason_all[lo:hi + 1] = reason[lo:hi + 1]
    return {"labels": out, "ball": track, "control": ctrl, "mean_speed": ms, "dead": dead_all,
            "dead_reason": reason_all}


def label(px, py, ball, team, blocks, params=None, *, fps=25, pitch=(105.0, 68.0)):
    return run(px, py, ball, team, blocks, params, fps=fps, pitch=pitch)["labels"]
