"""Possession labeller v5 = v4 + reconstructed ball + zone of influence + 15-frame confirm + flights.

    label(px, py, ball, team, blocks, params=None, *, fps=25, pitch=(105, 68), tags=None,
          ball_source=None) -> (n,) "H" / "A" / "D" / "X"

Every addition has a switch; with FILL_METHOD="linear", FILL_MAX=6, FILL_MAX_SPEED=None,
ZOI_MODE="off", CONFIRM=6 and USE_FLIGHTS=False the output is exactly v4 (checked on the bench).

1. BALL SOURCE  (preprocess/reconstruct.prepare_ball, scripts/reconstruct_ball.py)
   Spikes out, then every gap of at most FILL_MAX frames filled with FILL_METHOD (default Hermite,
   25 frames = 1 s; v4 bridged 6 frames = 0.24 s linearly). A gap whose ends imply a ball faster
   than FILL_MAX_SPEED (35 m/s) is a re-detection elsewhere and stays NaN. On BAR-ATH,
   scripts/run.py passes the stored file outputs/ball/ball_reconstructed.npz as ``ball_source``;
   it is used when its settings equal these parameters, otherwise the same thing is computed
   here (the bench always computes it).

2. FLIGHTS  (features/flight.flight_mask, kinematic part only: no homography needed)
   A flight runs from the kick (contact frame) to the landing / trap / reception. While the ball
   is in flight, nobody controls it: frames strictly inside a flight get no control, whatever
   the distance or zone rules say. Possession stays with the last team through the flight
   (model/resolve.py holds it while nobody controls). This removes false "control" by a player
   the ball merely passes (a 4-12 m/s ball within 2 m of an opponent).

3. CONTROL = DISTANCE RULE (the decider) + ZONE OF INFLUENCE (a check, never the sole decider)
   distance  D(t) = team T if T's nearest player is within CTRL_R m, the other team's nearest is
             at least MARGIN m further away and the ball is slower than CTRL_V m/s (v4).
   zone      Z(t) from detectors/control.py method "zone": the ball is inside a player's zone of
             influence (features/zoi.py: an ellipse of radius ZOI_R0 at standstill, stretched
             forward / shrunk sideways and backwards with his speed) AND moves with him
             (|v_ball - v_player| <= ZOI_REL_V m/s). Z = T (only T has it), BOTH (both teams'
             zones hold the ball: a contest) or NONE.
   ZOI_MODE  how Z is combined with D, frame by frame:
     "off"     control = D                                         (v4)
     "veto"    control = D, unless Z says BOTH or the OTHER team   (Z can only remove control)
     "confirm" control = D only where Z agrees (Z == D)            (both rules must hold)
   In every mode Z alone never creates control, as required: the zone adds evidence about
   direction of play and ball-player motion that the plain distance cannot see (a ball 1.8 m
   BEHIND a sprinting player is not his; two players from different teams both on the ball is a
   duel, not control), but a team only ever gains control where the distance rule fires.

4. POSSESSION CHANGE  (model/resolve.py)
   The other team must control the ball for CONFIRM frames (holes of nobody up to HOLE frames)
   before possession switches; the switch is backdated to the first frame of that control.
   v5: CONFIRM = 15 frames = 0.6 s at 25 fps (v4: 6 = 0.24 s). A touch, block or deflection
   shorter than that never switches possession.

5. DEAD BALL  (detectors/stoppage.py): v4 unchanged, on the reconstructed ball.
"""
from __future__ import annotations

import numpy as np

from possession.detectors import stoppage
from possession.detectors.control import AWAY, BOTH, HOME, NONE, ControlConfig, detect_control
from possession.features.flight import FlightConfig, flight_mask
from possession.features.zoi import ZoiConfig
from possession.labellers import v4
from possession.model.resolve import resolve_block
from possession.preprocess.ball import windowed_velocity
from possession.preprocess.players import velocity
from possession.preprocess.reconstruct import prepare_ball

DEFAULTS = dict(v4.DEFAULTS)
DEFAULTS.update(
    # 1. ball source
    FILL_METHOD="hermite", FILL_MAX=25, FILL_MAX_SPEED=35.0,
    # 2. flights
    USE_FLIGHTS=True, FLIGHT_MODE="opponent", FLIGHT_KICK_LOOKBACK=10, FLIGHT_START_V=12.0, FLIGHT_STOP_V=4.0, FLIGHT_MIN_FRAMES=10,
    FLIGHT_MAX_S=4.0, FLIGHT_RECEIVE_R=1.5, FLIGHT_RECEIVE_V=8.0,
    # 3. zone of influence
    ZOI_MODE="veto", ZOI_R0=1.5, ZOI_REL_V=4.0, ZOI_FRONT=0.8, ZOI_SIDE=0.35, ZOI_REAR=0.5,
    ZOI_VMAX=8.0,
    # 4. possession change
    CONFIRM=15,
)
LETTERS = np.array(["H", "A", "D"])
ZOI_MODES = ("off", "contest", "veto", "confirm")
FLIGHT_MODES = ("all", "opponent")


def ball_input(ball, p, fps, ball_source=None):
    """(x, y, filled, source) of the prepared ball: the stored file when its settings match."""
    if ball_source is not None and len(ball_source["x"]) == len(ball) and \
            ball_source["method"] == p["FILL_METHOD"] and ball_source["max_gap"] == int(p["FILL_MAX"]) \
            and abs(ball_source["max_step_m"] - p["MAX_STEP"]) < 1e-9 \
            and abs(ball_source.get("max_speed_ms", 0.0) - (p["FILL_MAX_SPEED"] or 0.0)) < 1e-9:
        return ball_source["x"], ball_source["y"], ball_source["filled"], "file"
    x, y, _, _, filled = prepare_ball(ball[:, 0], ball[:, 1], p["FILL_METHOD"], int(p["FILL_MAX"]),
                                      p["MAX_STEP"], fps, p["FILL_MAX_SPEED"])
    return x, y, filled, "computed"


def zone_control(px, py, x, y, bvx, bvy, team, p, fps, chunk=20000):
    """Per-frame zone evidence Z: HOME / AWAY / BOTH / NONE (computed in chunks to bound memory)."""
    pvx, pvy, _ = velocity(px, py, fps, half=12, max_speed_ms=12.0)
    zcfg = ZoiConfig(r0_m=p["ZOI_R0"], speed_max_ms=p["ZOI_VMAX"], front_gain=p["ZOI_FRONT"],
                     side_shrink=p["ZOI_SIDE"], rear_shrink=p["ZOI_REAR"])
    cfg = ControlConfig(method="zone", max_rel_speed_ms=p["ZOI_REL_V"], zoi=zcfg,
                        contest_as_none=False)
    z = np.full(len(x), NONE, dtype=np.int8)
    for a in range(0, len(x), chunk):
        b = min(len(x), a + chunk)
        r = detect_control(px[a:b], py[a:b], x[a:b], y[a:b], team, cfg,
                           ball_v=(bvx[a:b], bvy[a:b]), player_v=(pvx[a:b], pvy[a:b]), fps=fps)
        z[a:b] = r.ctrl
    return z


def combine(d, z, mode):
    """Control from the distance rule ``d`` (HOME/AWAY/NONE) checked by the zone evidence ``z``."""
    if mode == "off":
        return d.copy()
    out = d.copy()
    other = np.where(d == HOME, AWAY, np.where(d == AWAY, HOME, -9))
    if mode == "contest":
        out[(d >= 0) & (z == BOTH)] = NONE
    elif mode == "veto":
        out[(d >= 0) & ((z == BOTH) | (z == other))] = NONE
    elif mode == "confirm":
        out[(d >= 0) & (z != d)] = NONE
    else:
        raise ValueError("ZOI_MODE must be one of {}".format(ZOI_MODES))
    return out


def kicker_team(ctrl, s, lookback):
    """Team that last controlled the ball at or just before a flight's kick frame, else NONE."""
    seg = ctrl[max(0, s - lookback):s + 1]
    idx = np.flatnonzero((seg == HOME) | (seg == AWAY))
    return int(seg[idx[-1]]) if len(idx) else NONE


def mask_flights(ctrl, flights, mode, lookback=10):
    """Remove control inside flights (kick and reception frames keep theirs).

    "all"       nobody controls a ball in flight
    "opponent"  only the team that did NOT kick loses its control inside the flight: a ball
                passing within reach of an opponent is not his. The kicker's own team keeps its
                evidence (its receiver's first touches), and a real interception still ends the
                flight (reception / landing / direction-change exits), so it is not hidden.
    """
    out = ctrl.copy()
    for s, e in flights:
        if e - s < 2:
            continue
        if mode == "all":
            out[s + 1:e] = NONE
        elif mode == "opponent":
            k = kicker_team(ctrl, s, lookback)
            if k == NONE:
                continue
            seg = out[s + 1:e]
            seg[seg == (AWAY if k == HOME else HOME)] = NONE
        else:
            raise ValueError("FLIGHT_MODE must be one of {}".format(FLIGHT_MODES))
    return out


def run(px, py, ball, team, blocks, params=None, *, fps=25, pitch=(105.0, 68.0), tags=None,
        ball_source=None):
    p = dict(DEFAULTS, **(params or {}))
    team = np.asarray(team)
    x, y, filled, source = ball_input(ball, p, fps, ball_source)
    bvx, bvy = windowed_velocity(x, y, fps, 4)
    b = np.stack([x, y], axis=1)

    # 3. distance rule (the decider), then the zone check
    dist = detect_control(px, py, x, y, team, v4.control_config(p), ball_v=(bvx, bvy), fps=fps)
    d = dist.ctrl.copy()
    d[d == BOTH] = NONE
    z = zone_control(px, py, x, y, bvx, bvy, team, p, fps) if p["ZOI_MODE"] != "off" else None
    ctrl = combine(d, z, p["ZOI_MODE"]) if z is not None else d

    # 2. flights: nobody controls a ball in flight (kick and reception frames keep their control)
    flights, in_flight = [], np.zeros(len(x), dtype=bool)
    if p["USE_FLIGHTS"]:
        fcfg = FlightConfig(fps=fps, pitch=tuple(pitch), start_speed=p["FLIGHT_START_V"],
                            stop_speed=p["FLIGHT_STOP_V"], min_flight_frames=int(p["FLIGHT_MIN_FRAMES"]),
                            max_flight_s=p["FLIGHT_MAX_S"], receive_r=p["FLIGHT_RECEIVE_R"],
                            receive_speed=p["FLIGHT_RECEIVE_V"])
        flights, in_flight = flight_mask(x, y, fcfg, px, py)
        ctrl = mask_flights(ctrl, flights, p["FLIGHT_MODE"], int(p["FLIGHT_KICK_LOOKBACK"]))

    # 5. dead ball (v4) and 4. possession changes
    rel = stoppage.rel_speed(px, py, blocks, p, fps)
    out = np.full(len(ball), "X", dtype="<U1")
    dead_all = np.zeros(len(ball), dtype=bool)
    for _, lo, hi in blocks:
        lo, hi = int(lo), int(hi)
        dead = stoppage.dead_mask(b, rel, p, lo, hi, tags if p["USE_TAGS"] else None, fps, pitch)
        codes = resolve_block(ctrl, dead, lo, hi, v4.resolve_config(p), fps)
        out[lo:hi + 1] = LETTERS[codes]
        dead_all[lo:hi + 1] = dead[lo:hi + 1]
    return {"labels": out, "ball_xy": b, "ball_filled": filled, "ball_source": source,
            "control_distance": d, "control_zone": z, "control": ctrl, "flights": flights,
            "in_flight": in_flight, "rel": rel, "dead": dead_all, "params": p}


def label(px, py, ball, team, blocks, params=None, *, fps=25, pitch=(105.0, 68.0), tags=None,
          ball_source=None):
    return run(px, py, ball, team, blocks, params, fps=fps, pitch=pitch, tags=tags,
               ball_source=ball_source)["labels"]
