"""Possession labeller v4 (Ashwyn, Nov 7-8), rebuilt on the shared modules. Submission candidate.

Original: backup/ashwyn_labeller_v4.py (+ ashwyn_labeller_v1/v2.py it reuses), run by
backup/ashwyn_run_labeller_v4.py. Output is identical to the original on every bench copy
(checked Nov 9, docs/changelog/labeller.md). Recipe:

  1. clean the ball           preprocess/ball.py       spikes out, holes <= BRIDGE frames bridged
  2. control per frame        detectors/control.py     distance method, CTRL_R = 2.0 m (v1: 1.5):
                                                       the real ball sits ~1 m off the player on it
  3. dead ball                detectors/stoppage.py    Ashwyn's v2.1 rules A/B/D relative to the
                                                       match's own pace; rule C OFF (v4)
  4. possessions              model/resolve.py         confirm CONFIRM frames, backdate, merge
                                                       spells < MIN_SPELL s

Bench (Ashwyn, mean of Metrica games 1-3, spellF1): real 0.516, real_occ 0.462, hard 0.443,
R+O 0.489 (v2.1: 0.461). Other control shapes (shift, ellipse, auto radius) and rule E were
tested and rejected; their switches live in backup/ashwyn_labeller_v4.py.

    label(px, py, ball, team, blocks, params=None, *, fps=25, pitch=(105, 68), tags=None)
"""
from __future__ import annotations

import numpy as np

from possession.detectors import stoppage
from possession.detectors.control import ControlConfig, detect_control
from possession.model.resolve import ResolveConfig, resolve_block
from possession.preprocess.ball import BallConfig, clean_ball

DEFAULTS = dict(stoppage.DEFAULTS)
DEFAULTS.update(
    MAX_STEP=2.0, BRIDGE=6,
    CTRL_R=2.0, MARGIN=0.3, CTRL_V=12.0,
    CONFIRM=6, HOLE=3, MIN_SPELL=2.0,
)
LETTERS = np.array(["H", "A", "D"])


def ball_config(p) -> BallConfig:
    return BallConfig(max_step_m=p["MAX_STEP"], max_speed_ms=None, bridge_frames=int(p["BRIDGE"]),
                      speed_half_window=4, out_margin_m=p["OUT_M"])


def control_config(p) -> ControlConfig:
    return ControlConfig(method="distance", r_m=p["CTRL_R"], margin_m=p["MARGIN"],
                         max_ball_speed_ms=p["CTRL_V"], contest_as_none=True)


def resolve_config(p) -> ResolveConfig:
    return ResolveConfig(confirm=int(p["CONFIRM"]), hole=int(p["HOLE"]), min_spell_s=p["MIN_SPELL"])


def run(px, py, ball, team, blocks, params=None, *, fps=25, pitch=(105.0, 68.0), tags=None):
    """Labels plus the intermediate signals (ball track, control, rel speed, dead mask)."""
    p = dict(DEFAULTS, **(params or {}))
    team = np.asarray(team)
    track = clean_ball(ball[:, 0], ball[:, 1], ball_config(p), fps=fps, pitch=pitch)
    b = track.xy
    ctrl = detect_control(px, py, track.x, track.y, team, control_config(p),
                          ball_v=(track.vx, track.vy), fps=fps)
    rel = stoppage.rel_speed(px, py, blocks, p, fps)
    out = np.full(len(ball), "X", dtype="<U1")
    dead_all = np.zeros(len(ball), dtype=bool)
    for _, lo, hi in blocks:
        lo, hi = int(lo), int(hi)
        dead = stoppage.dead_mask(b, rel, p, lo, hi, tags if p["USE_TAGS"] else None, fps, pitch)
        codes = resolve_block(ctrl.ctrl, dead, lo, hi, resolve_config(p), fps)
        out[lo:hi + 1] = LETTERS[codes]
        dead_all[lo:hi + 1] = dead[lo:hi + 1]
    return {"labels": out, "ball": track, "control": ctrl, "rel": rel, "dead": dead_all,
            "params": p}


def label(px, py, ball, team, blocks, params=None, *, fps=25, pitch=(105.0, 68.0), tags=None):
    return run(px, py, ball, team, blocks, params, fps=fps, pitch=pitch, tags=tags)["labels"]
