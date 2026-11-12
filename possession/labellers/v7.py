"""Possession labeller v7 ("event-guided", Nov 10): v6's possession definition fed with our new data.

Same definition of possession as v6 (Ashwyn): a team controls the ball when its nearest player is
within CTRL_R m, MARGIN m clearer than the other team, ball slower than CTRL_V m/s; a change needs
CONFIRM frames of the other team's control (holes <= HOLE), backdated, and SW_MARGIN m to TAKE the
ball; loose / contested = last team; dead ball = v4's rules A / B / D; D shorter than MIN_D s and
possessions shorter than MIN_SPELL s are merged away. What is new is the EVIDENCE fed into it:

  ball pipeline
    BALL="prepared"  preprocess/reconstruct.prepare_ball (spikes out, gaps <= 25 frames Hermite-filled
                     when physically possible), then v6's +/- SMOOTH_K smoothing. "v6" = v6's ball.
    FLIGHTS          labeller v5's kinematic flights: inside a flight the team that did NOT kick
                     loses its control (a ball flying past a defender is not his)

  events (possession/events/detect.py; the same detector on BAR-ATH and on the bench)
    E1 PASS_SHIELD    completed pass of team T: inside it the other team's control is ignored
    E2 ANCHOR         interception, tackle, possession change (kinds in ANCHOR_KINDS) won by team T:
                      T controls the ball for ANCHOR_N frames from the event's first frame, so the
                      change is confirmed AT the event (v4's false changes were mostly late). With
                      ANCHOR_NEED_CTRL the distance rule must have seen T on the ball within
                      ANCHOR_WIN frames of the event, so a detector error alone cannot flip it.
    E3                possession changes of kind "contest" are not anchors (contested = last team)
    E4 DUEL_GUARD     duel with no winner: nobody can TAKE the ball during it
    E5 HOLD_FILL      carry / hold of team T: T controls the frames where the distance rule saw nobody
                      (the detector's zone around a moving player tolerates the ~1 m ball offset)
    E6 RESTART        restart with a team from RESTART_SOURCES: that team controls the restart frames
                      and the other team's control just before it (ball being placed) is ignored

All switches off (``AS_V6``) is exactly labeller v6. Events are ``events=[...]`` (dicts as in
outputs/events/events_full.json; frames are row indices, i.e. video frames for BAR-ATH from row 0).
Without events only the ball pipeline changes.

    label(px, py, ball, team, blocks, params=None, *, fps=25, pitch=(105, 68), tags=None,
          events=None, ball_source=None)
"""
from __future__ import annotations

import numpy as np

from possession.detectors import stoppage
from possession.detectors.control import BOTH, NONE, detect_control
from possession.features.flight import FlightConfig, flight_mask
from possession.model.resolve import resolve_block
from possession.preprocess.ball import clean_ball, windowed_velocity

from . import v4, v5, v6

HOME, AWAY = 0, 1
TEAM_CODE = {"H": HOME, "A": AWAY}

DEFAULTS = dict(v5.DEFAULTS)                 # ball-fill and flight settings of v5 ...
DEFAULTS.update(v6.DEFAULTS)                 # ... under v6's possession definition (CONFIRM 6 etc.)
DEFAULTS.update(
    BALL="prepared",                # "prepared" | "v6"
    FLIGHTS=True,
    PASS_SHIELD=True,
    ANCHOR=True, ANCHOR_N=6, ANCHOR_NEED_CTRL=True, ANCHOR_WIN=25,
    ANCHOR_KINDS=("interception", "tackle", "loose_ball_recovery"),
    DUEL_GUARD=True,
    HOLD_FILL=True, HOLD_TYPES=("carry", "hold"),
    RESTART=True, RESTART_N=6, RESTART_PRE=50, RESTART_SOURCES=("smart_tagging", "first_control"),
)
AS_V6 = dict(BALL="v6", FLIGHTS=False, PASS_SHIELD=False, ANCHOR=False, DUEL_GUARD=False,
             HOLD_FILL=False, RESTART=False)
LETTERS = v6.LETTERS


# ------------------------------------------------------------------------------------------------
# ball
# ------------------------------------------------------------------------------------------------

def ball_track(ball, p, fps, pitch, ball_source=None):
    """(x, y, vx, vy) of the ball the labeller uses."""
    if p["BALL"] == "v6":
        b = v6.smooth(ball, int(p["SMOOTH_K"])) if p["SMOOTH_K"] else ball
        t = clean_ball(b[:, 0], b[:, 1], v4.ball_config(p), fps=fps, pitch=pitch)
        return t.x, t.y, t.vx, t.vy
    if p["BALL"] != "prepared":
        raise ValueError("BALL must be 'prepared' or 'v6'")
    x, y, _, _ = v5.ball_input(ball, p, fps, ball_source)
    if p["SMOOTH_K"]:
        s = v6.smooth(np.stack([x, y], axis=1), int(p["SMOOTH_K"]))
        x, y = s[:, 0], s[:, 1]
    vx, vy = windowed_velocity(x, y, fps, 4)
    return x, y, vx, vy


# ------------------------------------------------------------------------------------------------
# events -> control evidence (each function changes ctrl / ctrl_sw in place and returns a count)
# ------------------------------------------------------------------------------------------------

def _team(e, key="team"):
    return TEAM_CODE.get(e.get(key) or "", None)


def _span(e, n):
    return max(0, int(e["frame_start"])), min(n, int(e["frame_end"]) + 1)


def pass_shield(ctrl, ctrl_sw, events):
    n, k = len(ctrl), 0
    for e in events:
        if e["type"] != "pass" or (e.get("details") or {}).get("outcome") != "complete":
            continue
        t = _team(e)
        if t is None:
            continue
        a, b = _span(e, n)
        if b - a < 3:
            continue
        for c in (ctrl, ctrl_sw):
            if c is not None:
                seg = c[a + 1:b - 1]
                k += int((seg == 1 - t).sum())
                seg[seg == 1 - t] = NONE
    return k


def anchor_events(events, kinds):
    """(frame, team) of every change-of-possession event used as an anchor."""
    out = []
    for e in events:
        t = None
        if e["type"] in ("interception", "tackle"):
            t = _team(e)
        elif e["type"] == "possession_change" and (e.get("details") or {}).get("kind") in kinds:
            t = TEAM_CODE.get((e.get("details") or {}).get("to_team") or "", _team(e))
        if t is not None:
            out.append((int(e["frame_start"]), t))
    return sorted(set(out))


def anchor(ctrl, ctrl_sw, raw_ctrl, events, p):
    n, used = len(ctrl), 0
    for f, t in anchor_events(events, p["ANCHOR_KINDS"]):
        if not 0 <= f < n:
            continue
        if p["ANCHOR_NEED_CTRL"]:
            w = raw_ctrl[max(0, f - p["ANCHOR_WIN"]):min(n, f + p["ANCHOR_WIN"] + 1)]
            if not (w == t).any():
                continue
        a, b = f, min(n, f + int(p["ANCHOR_N"]))
        ctrl[a:b] = t
        if ctrl_sw is not None:
            ctrl_sw[a:b] = t
        used += 1
    return used


def duel_guard(ctrl_sw, events):
    if ctrl_sw is None:
        return 0
    n, k = len(ctrl_sw), 0
    for e in events:
        if e["type"] != "duel" or (e.get("details") or {}).get("winner_team"):
            continue
        a, b = _span(e, n)
        ctrl_sw[a:b] = NONE
        k += 1
    return k


def hold_fill(ctrl, ctrl_sw, events, types):
    n, k = len(ctrl), 0
    for e in events:
        if e["type"] not in types:
            continue
        t = _team(e)
        if t is None:
            continue
        a, b = _span(e, n)
        for c in (ctrl, ctrl_sw):
            if c is not None:
                seg = c[a:b]
                free = seg == NONE
                if c is ctrl:
                    k += int(free.sum())
                seg[free] = t
    return k


def restart_team(ctrl, ctrl_sw, events, p):
    n, k = len(ctrl), 0
    for e in events:
        if e["type"] != "restart":
            continue
        d = e.get("details") or {}
        if d.get("team_source") not in p["RESTART_SOURCES"]:
            continue
        t = TEAM_CODE.get(d.get("restart_team") or "", _team(e))
        if t is None:
            continue
        f = int(e["frame_start"])
        if not 0 <= f < n:
            continue
        pre = max(0, f - int(p["RESTART_PRE"]))
        for c in (ctrl, ctrl_sw):
            if c is not None:
                seg = c[pre:f]
                seg[seg == 1 - t] = NONE
                c[f:min(n, f + int(p["RESTART_N"]))] = t
        k += 1
    return k


# ------------------------------------------------------------------------------------------------
# labeller
# ------------------------------------------------------------------------------------------------

def run(px, py, ball, team, blocks, params=None, *, fps=25, pitch=(105.0, 68.0), tags=None,
        events=None, ball_source=None):
    p = dict(DEFAULTS, **(params or {}))
    team = np.asarray(team)
    ball = np.asarray(ball, dtype=np.float64)
    if p["SMOOTH_K"]:
        k = int(p["SMOOTH_K"])
        px, py = v6.smooth(px, k), v6.smooth(py, k)
    x, y, vx, vy = ball_track(ball, p, fps, pitch, ball_source)
    b = np.stack([x, y], axis=1)

    ccfg = v4.control_config(p)
    raw = detect_control(px, py, x, y, team, ccfg, ball_v=(vx, vy), fps=fps).ctrl.copy()
    raw[raw == BOTH] = NONE
    ctrl = raw.copy()
    ctrl_sw = None
    if p["SW_MARGIN"] is not None:
        ccfg.margin_m = p["SW_MARGIN"]
        ctrl_sw = detect_control(px, py, x, y, team, ccfg, ball_v=(vx, vy), fps=fps).ctrl.copy()
        ctrl_sw[ctrl_sw == BOTH] = NONE

    info = {}
    if p["FLIGHTS"]:
        fcfg = FlightConfig(fps=fps, pitch=tuple(pitch), start_speed=p["FLIGHT_START_V"],
                            stop_speed=p["FLIGHT_STOP_V"], min_flight_frames=int(p["FLIGHT_MIN_FRAMES"]),
                            max_flight_s=p["FLIGHT_MAX_S"], receive_r=p["FLIGHT_RECEIVE_R"],
                            receive_speed=p["FLIGHT_RECEIVE_V"])
        flights, _ = flight_mask(x, y, fcfg, px, py)
        ctrl = v5.mask_flights(ctrl, flights, "opponent", int(p["FLIGHT_KICK_LOOKBACK"]))
        if ctrl_sw is not None:
            ctrl_sw = v5.mask_flights(ctrl_sw, flights, "opponent", int(p["FLIGHT_KICK_LOOKBACK"]))
        info["flights"] = len(flights)
    ev = list(events or [])
    if ev:
        # order matters: fills first, then the stricter edits, then the anchors and restarts
        if p["HOLD_FILL"]:
            info["hold_fill_frames"] = hold_fill(ctrl, ctrl_sw, ev, p["HOLD_TYPES"])
        if p["PASS_SHIELD"]:
            info["pass_shield_frames"] = pass_shield(ctrl, ctrl_sw, ev)
        if p["DUEL_GUARD"]:
            info["duel_guard"] = duel_guard(ctrl_sw, ev)
        if p["ANCHOR"]:
            info["anchors"] = anchor(ctrl, ctrl_sw, raw, ev, p)
        if p["RESTART"]:
            info["restarts"] = restart_team(ctrl, ctrl_sw, ev, p)

    rel = stoppage.rel_speed(px, py, blocks, p, fps)
    out = np.full(len(ball), "X", dtype="<U1")
    dead_all = np.zeros(len(ball), dtype=bool)
    for _, lo, hi in blocks:
        lo, hi = int(lo), int(hi)
        dead = stoppage.dead_mask(b, rel, p, lo, hi, tags if p["USE_TAGS"] else None, fps, pitch)
        dead = v6.drop_short_dead(dead, b, p, lo, hi, fps, pitch)
        codes = resolve_block(ctrl, dead, lo, hi, v4.resolve_config(p), fps, ctrl_sw=ctrl_sw)
        out[lo:hi + 1] = LETTERS[codes]
        dead_all[lo:hi + 1] = dead[lo:hi + 1]
    return {"labels": out, "ball_xy": b, "control_raw": raw, "control": ctrl, "ctrl_sw": ctrl_sw,
            "rel": rel, "dead": dead_all, "info": info, "params": p}


def label(px, py, ball, team, blocks, params=None, *, fps=25, pitch=(105.0, 68.0), tags=None,
          events=None, ball_source=None):
    return run(px, py, ball, team, blocks, params, fps=fps, pitch=pitch, tags=tags, events=events,
               ball_source=ball_source)["labels"]
