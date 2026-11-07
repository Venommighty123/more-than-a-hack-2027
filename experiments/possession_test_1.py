"""FROZEN EXPERIMENT (2026-11-04), kept as a record; its control logic now lives in
possession/detectors/control.py (method="zone", checked identical) and is scored through the labellers.

Possession test 1: zone-of-influence ball control with a contest rule, plus an event log.

Tests the team's carry / pass / interception logic (thread of 2026-11-04):

  CONTROL      A player controls the ball when it sits inside their zone of influence (ZoI) and
               moves at roughly their speed. A fast ball crossing a zone (a pass in flight, a
               lofted ball) is not control.
  CARRY/PASS   Possession stays with the team until the other team takes control. A pass in
               flight, or a pass received by a teammate, changes nothing.
  INTERCEPTION The other team takes sole control with no contest just before it. The change is
               accepted after a short control check (INTERCEPT_HOLD_S, default 0.3 s) so that a
               deflection or block does not flip the label.
  CONTEST      Players of both teams have the ball in their zones. Possession stays with the team
               that had it. If the other team then keeps the ball for CONTEST_HOLD_S (default 5 s)
               the change is accepted.
  BACKDATING   Every accepted change is written at the frame the new team FIRST had sole control,
               not at the frame the check passed. Late changes are what spellF1 punishes most.

  ZONE OF INFLUENCE  An ellipse around the player, oriented along their smoothed velocity. It
               stretches forward and shrinks sideways and backwards as speed rises. At standstill
               it is a circle of radius ZOI_R0_M.

Gaps and dead balls are handled crudely on purpose, so this test isolates the control logic:
  - ball missing for less than DEAD_GAP_S: possession carries on;
  - ball missing for DEAD_GAP_S or more, or ball outside the pitch: D until the next control;
  - the published pre-kickoff stretch of the 2nd half (110808..113741) is D.
No X is written on graded frames (X is forward-filled by the scorer anyway and costs macroF1).

EVENTS (written to the JSON, StatsBomb-style names where one exists)
  carry / hold       one player keeps the ball in their zone; carry if the ball moves >= 2 m
  pass               the ball goes from one player to another. outcome "complete" (teammate) or
                     "intercepted" (opponent)
  dispossessed       the ball goes straight to an opponent with no flight (tackle, steal)
  contest            players of both teams have the ball in their zones at the same time
  possession_change  a change the state machine accepted (kind interception or contest)
  change_cancelled   a pending change undone because the old team got the ball back
  dead_ball          a D stretch, with its reason, and restart (the team that restarts play)
  ball_not_tracked   a gap of 5+ frames with no ball position
Player ids are track SLOTS ("Track 7" = slot 7), not people: slots swap between players.

Runs on a SHORT window only: by default the first 60 s of the match (video frames 11185..12684,
inside the public ground truth). Run from the repo root (numpy + pandas only, a few seconds):

    python experiments/possession_test_1.py
    python experiments/possession_test_1.py --start 13000 --seconds 60      # another minute
    python experiments/possession_test_1.py --contest-hold 2 --zoi-r0 2.0   # try variants
    python experiments/possession_test_1.py --no-frame-detail               # smaller JSON

Writes
  evaluation/results/possession_test_1.csv   the window labelled, every other frame X
  outputs/possession_test_1/events_<start>.json   (readable)
       meta (window, settings, units), summary, possessions and the event list
  outputs/possession_test_1/frames_<start>.json   (compact; skipped with --no-frame-detail)
       every frame: label, control state, holder, ball position and speed, and for every
       tracked player position, velocity, zone-of-influence shape and ball-in-zone
  The JSON files hold frame-level tracking, which the rules say must not be published: keep it out of
  the public repo (outputs/possession_test_1/ is git-ignored) and do not upload it to any cloud
  AI tool.

and prints
  1. the format check (same checks as the official scorer),
  2. inside the window: possession count and lengths, control-state mix, change counts by kind,
     event counts, and every predicted label change with its video time,
  3. if the window overlaps the public 5 minutes, the true label changes in the same window and
     the share of frames that agree. A SANITY check only: do NOT tune thresholds on it (the
     rules forbid it, and one minute is far too little to tune on anyway).

Only aggregate numbers are printed, so the printed output is safe to paste into the project thread.
test_scripts/visualize_events.py draws all of this onto the video.
"""

import argparse
import importlib.util
import json
import os
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402

# Params (all overridable from the command line)
ZOI_R0_M = 1.5            # zone radius at standstill, metres
ZOI_SPEED_MAX = 8.0       # speed (m/s) at which the zone reaches its full deformation
ZOI_FRONT_GAIN = 0.8      # front radius = r0 * (1 + gain * s/smax)
ZOI_SIDE_SHRINK = 0.35    # side radius  = r0 * (1 - shrink * s/smax)
ZOI_REAR_SHRINK = 0.5     # rear radius  = r0 * (1 - shrink * s/smax)
REL_SPEED_MAX = 4.0       # ball speed relative to the player (m/s) still counted as control
INTERCEPT_HOLD_S = 0.3    # control check for an uncontested change
CONTEST_HOLD_S = 5.0      # how long the other team must keep the ball after a contest
CONTEST_MEMORY_S = 1.0    # a change within this long after a contest counts as contested
MIN_CTRL_S = 0.2          # sole-control time the new team needs inside the hold window
DEAD_GAP_S = 10.0         # a ball gap this long or longer is treated as dead ball
OUT_MARGIN_M = 0.5        # ball this far past a pitch line is out of play
SMOOTH_HALF = 2           # +/- frames of moving-average smoothing on positions
MAX_BALL_SPEED = 40.0     # m/s; faster frame-to-frame jumps are tracking errors
CHUNK = 20000             # frames processed at a time
# Event detection
TOUCH_GAP_FRAMES = 3      # a holder may drop out this many frames and still be one touch spell
CARRY_MIN_M = 2.0         # ball displacement during a touch spell that makes it a carry
FLIGHT_MIN_FRAMES = 3     # frames of free ball between two players that make it a pass
SLOT_SWITCH_M = 1.0       # a "pass" to a teammate this close in under 3 frames is a slot switch
MISSING_EVENT_FRAMES = 5  # ball gaps at least this long are logged as ball_not_tracked
DETAIL_MAX_S = 180.0      # per-frame detail is written only for windows up to this long

FPS = GAMES[GAME]["fps"]
PITCH_L, PITCH_W = GAMES[GAME]["pitch"]
SECOND_HALF_KICKOFF = 113742       # published in FAQ.md: graded from 110808, all D until here
HOME, AWAY, NONE, BOTH = 0, 1, -1, 2
TEAM_LETTER = {HOME: "H", AWAY: "A"}
CTRL_NAME = {HOME: "home", AWAY: "away", NONE: "none", BOTH: "contest"}
OUT_NAME = "possession_test_1"
WINDOW_START = 11185             # 1st-half kickoff, inside the public ground truth
WINDOW_SECONDS = 60.0
REPORT_DIR = os.path.join(ROOT, "outputs", OUT_NAME)


# -- loading ---------------------------------------------------------------------------------

def team_of_each_track(meta_path):
    """0 for home slots, 1 for away, in Player order (first Team in the XML is home)."""
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    if len(team_ids) != 2:
        raise ValueError("expected 2 teams in {}, found {}".format(meta_path, len(team_ids)))
    slots = [player.get("teamId") for player in root.iter("Player")]
    return np.array([HOME if t == team_ids[0] else AWAY for t in slots], dtype=np.int8)


def team_names(meta_path):
    root = ET.parse(meta_path).getroot()
    return [team.findtext("Name") for team in root.iter("Team")]


def load_tracking(raw_path, n_slots, lo, hi):
    """Positions in metres for video frames lo..hi inclusive: px, py (frames x slots) and bx, by.

    Lines outside the window are skipped without parsing, and reading stops after hi, so a
    one-minute window takes seconds. NaN where an object is not seen.
    """
    n = hi - lo + 1
    px = np.full((n, n_slots), np.nan, dtype=np.float32)
    py = np.full((n, n_slots), np.nan, dtype=np.float32)
    bx = np.full(n, np.nan, dtype=np.float32)
    by = np.full(n, np.nan, dtype=np.float32)
    per_slot = None

    with open(raw_path) as handle:
        for line in handle:
            head = line[:line.index(":")] if ":" in line else ""
            if not head.isdigit():
                continue
            frame = int(head) - 1          # ATD counts from 1, video frames from 0
            if frame < lo:
                continue
            if frame > hi:
                break
            parts = line.strip().split(":")
            if len(parts) != 3:
                continue
            values = np.array(parts[1].replace(";", ",").split(","), dtype=np.float32)
            if per_slot is None:
                per_slot = len(values) // n_slots
                if per_slot < 2 or per_slot * n_slots != len(values):
                    raise ValueError("cannot split {} values into {} slots".format(
                        len(values), n_slots))
            values = values.reshape(n_slots, per_slot)
            k = frame - lo
            px[k] = values[:, 0] * PITCH_L
            py[k] = values[:, 1] * PITCH_W
            ball = parts[2].split(",")
            if len(ball) >= 2 and ball[0][:1] != "N":
                bx[k] = float(ball[0]) * PITCH_L
                by[k] = float(ball[1]) * PITCH_W
    return px, py, bx, by


# -- signal processing -----------------------------------------------------------------------

def nan_smooth(values, half):
    """Centred moving average that ignores NaN, along axis 0. NaN stays NaN."""
    if half <= 0:
        return values
    ok = ~np.isnan(values)
    filled = np.where(ok, values, 0.0).astype(np.float64)
    pad = [(half + 1, half)] + [(0, 0)] * (values.ndim - 1)
    csum = np.cumsum(np.pad(filled, pad), axis=0)
    ccnt = np.cumsum(np.pad(ok.astype(np.float64), pad), axis=0)
    width = 2 * half + 1
    total = csum[width:] - csum[:-width]
    count = ccnt[width:] - ccnt[:-width]
    out = np.where(count > 0, total / np.maximum(count, 1), np.nan)
    return np.where(ok, out, np.nan).astype(np.float32)


def velocity(pos):
    """Central-difference velocity in m/s along axis 0."""
    vel = np.full_like(pos, np.nan)
    vel[1:-1] = (pos[2:] - pos[:-2]) * (FPS / 2.0)
    return vel


def clean_ball(bx, by):
    """Drop single-frame teleports (implausible speed) from the ball track."""
    step = np.hypot(np.diff(bx, prepend=np.nan), np.diff(by, prepend=np.nan)) * FPS
    jump = step > MAX_BALL_SPEED
    bx, by = bx.copy(), by.copy()
    bx[jump] = np.nan
    by[jump] = np.nan
    return bx, by


def zoi_geometry(vx, vy, args):
    """Zone of influence shape per frame and slot: unit heading (ux, uy) and the front, rear and
    side radii in metres. A standing player gets an arbitrary heading (the zone is a circle)."""
    speed = np.hypot(vx, vy)
    speed = np.where(np.isnan(speed), 0.0, speed)
    ratio = np.clip(speed / args.zoi_speed_max, 0.0, 1.0)
    moving = speed > 0.2
    ux = np.where(moving, np.nan_to_num(vx) / np.maximum(speed, 1e-6), 1.0)
    uy = np.where(moving, np.nan_to_num(vy) / np.maximum(speed, 1e-6), 0.0)
    r0 = args.zoi_r0
    front = r0 * (1.0 + ZOI_FRONT_GAIN * ratio)
    rear = r0 * (1.0 - ZOI_REAR_SHRINK * ratio)
    side = r0 * (1.0 - ZOI_SIDE_SHRINK * ratio)
    return ux, uy, front, rear, side


def zoi_q(px, py, vx, vy, bx, by, args):
    """Normalised zone distance of the ball for each frame and slot: <= 1 means inside the zone.
    inf where the player or ball is not seen."""
    ux, uy, front, rear, side = zoi_geometry(vx, vy, args)
    dx = bx[:, None] - px
    dy = by[:, None] - py
    along = dx * ux + dy * uy
    lateral = -dx * uy + dy * ux
    radius_along = np.where(along >= 0, front, rear)
    q = (along / radius_along) ** 2 + (lateral / side) ** 2
    return np.nan_to_num(q, nan=np.inf)


def zoi_inside(px, py, vx, vy, bx, by, args):
    """Boolean (n_frames x n_slots): is the ball inside each player's zone of influence?"""
    return zoi_q(px, py, vx, vy, bx, by, args) <= 1.0


def control_per_frame(px, py, bx, by, team_of_slot, args):
    """Per-frame control. Returns a dict:

      ctrl         HOME / AWAY (sole control), BOTH (contest), NONE (free ball or no ball)
      holder       slot of the controlling player (sole control only), else -1
      contest_h/a  slot of the home / away player nearest the ball in a contest, else -1
      sbx, sby, bvx, bvy   smoothed ball position (m) and velocity (m/s)
      spx, spy, vx, vy     smoothed player positions (m) and velocities (m/s)
      q            normalised zone distance (frames x slots), <= 1 inside
      slow         ball moving at roughly the player's speed (frames x slots)
    """
    spx, spy = nan_smooth(px, SMOOTH_HALF), nan_smooth(py, SMOOTH_HALF)
    vx, vy = velocity(spx), velocity(spy)
    sbx, sby = nan_smooth(bx, SMOOTH_HALF), nan_smooth(by, SMOOTH_HALF)
    bvx, bvy = velocity(sbx), velocity(sby)

    home_slots = team_of_slot == HOME
    n = len(bx)
    q_all = np.full(px.shape, np.inf, dtype=np.float32)
    slow_all = np.zeros(px.shape, dtype=bool)
    for a in range(0, n, CHUNK):        # chunks keep memory down on long windows
        b = min(n, a + CHUNK)
        q_all[a:b] = zoi_q(spx[a:b], spy[a:b], vx[a:b], vy[a:b], sbx[a:b], sby[a:b], args)
        rel = np.hypot(bvx[a:b, None] - np.nan_to_num(vx[a:b]),
                       bvy[a:b, None] - np.nan_to_num(vy[a:b]))
        # Where the ball velocity is unknown (track edges), accept position alone.
        slow_all[a:b] = np.where(np.isnan(rel), True, rel <= args.rel_speed_max)

    inside = q_all <= 1.0
    controls = inside & slow_all
    home = controls[:, home_slots].any(axis=1)
    away = controls[:, ~home_slots].any(axis=1)
    # A contest is both teams having the ball in their zones, whatever its speed.
    contest = inside[:, home_slots].any(axis=1) & inside[:, ~home_slots].any(axis=1)

    ctrl = np.full(n, NONE, dtype=np.int8)
    ctrl[home & ~away] = HOME
    ctrl[away & ~home] = AWAY
    ctrl[contest] = BOTH

    # Holder: the controlling player deepest inside their zone.
    q_ctrl = np.where(controls, q_all, np.inf)
    best = np.argmin(q_ctrl, axis=1)
    holder = np.where(np.isin(ctrl, (HOME, AWAY)), best, -1).astype(np.int16)

    def nearest_of(mask):
        q_team = np.where(mask[None, :], q_all, np.inf)
        slot = np.argmin(q_team, axis=1)
        return np.where(contest, slot, -1).astype(np.int16)

    return {"ctrl": ctrl, "holder": holder,
            "contest_h": nearest_of(home_slots), "contest_a": nearest_of(~home_slots),
            "sbx": sbx, "sby": sby, "bvx": bvx, "bvy": bvy,
            "spx": spx, "spy": spy, "vx": vx, "vy": vy, "q": q_all, "slow": slow_all}


DEAD_REASONS = ("", "out_of_pitch", "long_ball_gap", "before_kickoff", "until_restart")


def dead_mask(sbx, sby, ctrl, n_frames, args):
    """True where the ball is taken as dead: long gaps and ball out of the pitch, each until the
    next frame some team controls the ball (the restart). Also returns a reason code per frame
    (index into DEAD_REASONS) and the number of out-of-pitch frames."""
    missing = np.isnan(sbx)
    out = (~missing) & ((sbx < -OUT_MARGIN_M) | (sbx > PITCH_L + OUT_MARGIN_M)
                        | (sby < -OUT_MARGIN_M) | (sby > PITCH_W + OUT_MARGIN_M))
    dead = out.copy()
    reason = np.where(out, 1, 0).astype(np.int8)

    # Long gaps.
    edges = np.diff(np.concatenate(([0], missing.astype(np.int8), [0])))
    for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if stop - start >= args.dead_gap * FPS:
            dead[start:stop] = True
            reason[start:stop] = 2

    # Extend each dead stretch until the next controlled frame.
    controlled = (ctrl == HOME) | (ctrl == AWAY)
    i = 0
    while i < n_frames:
        if dead[i]:
            j = i
            while j < n_frames and (dead[j] or not controlled[j]):
                if not dead[j]:
                    reason[j] = 4
                j += 1
            dead[i:j] = True
            i = j
        else:
            i += 1
    return dead, reason, int(out.sum())


# -- the possession state machine --------------------------------------------------------------

def run_state_machine(ctrl, ball_x, dead, first, last, args, stats, log=None):
    """Labels (0 H, 1 A, 2 D) for frames first..last inclusive.

    If log is a list, every accepted and cancelled change is appended to it as a dict with
    window-relative frame indices (start = first sole-control frame, decided = frame the check
    passed or failed)."""
    n = last - first + 1
    labels = np.full(n, -1, dtype=np.int8)
    intercept_hold = max(1, int(round(args.intercept_hold * FPS)))
    contest_hold = max(1, int(round(args.contest_hold * FPS)))
    contest_memory = int(round(CONTEST_MEMORY_S * FPS))
    min_ctrl = max(1, int(round(MIN_CTRL_S * FPS)))

    team = None                      # current team in possession
    pending = None                   # dict(team, start, kind, ctrl, observed)
    last_contest = -10 ** 9

    for k in range(n):
        f = first + k
        c = ctrl[f]

        if dead[f]:
            labels[k] = 2
            team, pending = None, None
            continue

        if c == BOTH:
            last_contest = k
            if pending is not None and pending["kind"] == "interception":
                pending["kind"] = "contest"

        if team is None:
            if c in (HOME, AWAY):
                team = int(c)            # restart: first team to control the ball
                labels[k] = team
            continue                     # frames before the first control are back-filled

        if c in (HOME, AWAY) and c != team:
            if pending is None:
                kind = "contest" if k - last_contest <= contest_memory else "interception"
                pending = {"team": int(c), "start": k, "kind": kind, "ctrl": 0, "observed": 0}
            pending["ctrl"] += 1
        elif c == team and pending is not None:
            stats["cancelled_" + pending["kind"]] += 1
            if log is not None:
                log.append({"type": "change_cancelled", "kind": pending["kind"],
                            "start": first + pending["start"], "decided": f,
                            "from": team, "to": pending["team"]})
            pending = None

        if pending is not None:
            if not np.isnan(ball_x[f]):
                pending["observed"] += 1
            hold = contest_hold if pending["kind"] == "contest" else intercept_hold
            if pending["observed"] >= hold and pending["ctrl"] >= min(min_ctrl, hold):
                # Backdate: the new team has had the ball since pending["start"].
                labels[pending["start"]:k + 1] = pending["team"]
                stats["changes_" + pending["kind"]] += 1
                if log is not None:
                    log.append({"type": "possession_change", "kind": pending["kind"],
                                "start": first + pending["start"], "decided": f,
                                "from": team, "to": pending["team"]})
                team, pending = pending["team"], None
                continue

        labels[k] = team

    # Back-fill the frames before the first control in each live stretch.
    nxt = -1
    for k in range(n - 1, -1, -1):
        if labels[k] == -1:
            labels[k] = nxt if nxt != -1 else 2
        else:
            nxt = labels[k] if labels[k] != 2 else -1
    return labels


# -- events ----------------------------------------------------------------------------------

def runs_of(mask):
    """[(start, stop)] half-open runs where a boolean array is True."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def r2(value):
    """Round for JSON; NaN becomes None."""
    if value is None:
        return None
    value = float(value)
    return None if not np.isfinite(value) else round(value, 2)


def touch_spells(holder, dead):
    """[(start, stop, slot)] runs where one slot holds the ball, bridging gaps of up to
    TOUCH_GAP_FRAMES frames with no holder (never across a dead ball)."""
    spells = []
    k, n = 0, len(holder)
    while k < n:
        slot = holder[k]
        if slot < 0:
            k += 1
            continue
        start, end = k, k + 1
        j = k + 1
        while j < n and not dead[j]:
            if holder[j] == slot:
                end = j + 1
            elif holder[j] >= 0 or j - end >= TOUCH_GAP_FRAMES:
                break
            j += 1
        spells.append((start, end, int(slot)))
        k = end
    return spells


def detect_events(res, team_of_slot, lo):
    """StatsBomb-like event list from the per-frame control signals in res (window-relative)."""
    holder, ctrl, dead = res["holder"], res["ctrl"], res["dead"]
    sbx, sby = res["sbx"], res["sby"]
    bspeed = np.hypot(res["bvx"], res["bvy"])
    spx, spy = res["spx"], res["spy"]
    events = []

    def frame_fields(k0, k1):
        """Absolute frames and video times of a half-open window-relative span."""
        return {"frame_start": lo + k0, "frame_end": lo + k1 - 1,
                "time_start": clock(lo + k0), "duration_s": r2((k1 - k0) / float(FPS))}

    def loc(k):
        return {"x": r2(sbx[k]), "y": r2(sby[k])}

    # Merge touch spells where a teammate's slot takes over at the same spot (slot switch).
    spells = touch_spells(holder, dead)
    merged = []
    for s in spells:
        if merged:
            a0, a1, aslot = merged[-1]
            b0, b1, bslot = s
            same_team = team_of_slot[aslot] == team_of_slot[bslot]
            close = (b0 - a1 < FLIGHT_MIN_FRAMES and
                     np.hypot(spx[a1 - 1, aslot] - spx[b0, bslot],
                              spy[a1 - 1, aslot] - spy[b0, bslot]) < SLOT_SWITCH_M)
            if aslot != bslot and same_team and close:
                merged[-1] = (a0, b1, bslot)
                continue
        merged.append(s)

    # Carries and holds.
    for k0, k1, slot in merged:
        dist = np.hypot(sbx[k1 - 1] - sbx[k0], sby[k1 - 1] - sby[k0])
        kind = "carry" if np.isfinite(dist) and dist >= CARRY_MIN_M else "hold"
        events.append(dict(type=kind, team=TEAM_LETTER[int(team_of_slot[slot])],
                           player=slot + 1, start=loc(k0), end=loc(k1 - 1),
                           distance_m=r2(dist), **frame_fields(k0, k1)))

    # Transfers between consecutive spells.
    for (a0, a1, aslot), (b0, b1, bslot) in zip(merged, merged[1:]):
        if aslot == bslot or dead[a1:b0].any():
            continue
        same = team_of_slot[aslot] == team_of_slot[bslot]
        flight = b0 - a1
        length = np.hypot(sbx[b0] - sbx[a1 - 1], sby[b0] - sby[a1 - 1])
        seen = ~np.isnan(sbx[a1:b0]) if flight > 0 else np.array([True])
        top = np.nanmax(bspeed[a1 - 1:b0 + 1]) if np.isfinite(bspeed[a1 - 1:b0 + 1]).any() \
            else np.nan
        base = dict(team=TEAM_LETTER[int(team_of_slot[aslot])], player=aslot + 1,
                    recipient=bslot + 1, recipient_team=TEAM_LETTER[int(team_of_slot[bslot])],
                    start=loc(a1 - 1), end=loc(b0), length_m=r2(length),
                    max_ball_speed_ms=r2(top), ball_seen_share=r2(seen.mean()),
                    **frame_fields(a1 - 1, b0 + 1))
        if same:
            events.append(dict(type="pass", outcome="complete", **base))
        elif flight >= FLIGHT_MIN_FRAMES:
            events.append(dict(type="pass", outcome="intercepted", **base))
        else:
            events.append(dict(type="dispossessed", **base))

    # Contests.
    for k0, k1 in runs_of(ctrl == BOTH):
        after = ctrl[k1:]
        sole = after[np.isin(after, (HOME, AWAY))]
        before = ctrl[:k0][np.isin(ctrl[:k0], (HOME, AWAY))]
        events.append(dict(type="contest",
                           home_player=int(res["contest_h"][k0]) + 1,
                           away_player=int(res["contest_a"][k0]) + 1,
                           location=loc(k0),
                           team_before=TEAM_LETTER[int(before[-1])] if len(before) else None,
                           won_by=TEAM_LETTER[int(sole[0])] if len(sole) else None,
                           **frame_fields(k0, k1)))

    # Possession changes and cancellations from the state machine.
    for item in res["log"]:
        k0, k1 = item["start"] - lo, item["decided"] - lo
        events.append(dict(type=item["type"], kind=item["kind"],
                           team_from=TEAM_LETTER[item["from"]], team_to=TEAM_LETTER[item["to"]],
                           frame=item["start"], time=clock(item["start"]),
                           decided_frame=item["decided"],
                           decision_delay_s=r2((k1 - k0) / float(FPS)), location=loc(k0)))

    # Dead balls.
    for k0, k1 in runs_of(dead):
        reasons = res["dead_reason"][k0:k1]
        main = [DEAD_REASONS[r] for r in (1, 2, 3) if (reasons == r).any()]
        nxt = ctrl[k1:k1 + 1]
        restart = TEAM_LETTER.get(int(nxt[0])) if len(nxt) else None
        events.append(dict(type="dead_ball", reason="+".join(main) or "until_restart",
                           restart_team=restart, **frame_fields(k0, k1)))

    # Ball not tracked.
    for k0, k1 in runs_of(np.isnan(sbx)):
        if k1 - k0 >= MISSING_EVENT_FRAMES:
            events.append(dict(type="ball_not_tracked", last_seen=loc(k0 - 1) if k0 else None,
                               **frame_fields(k0, k1)))

    def key(e):
        return e.get("frame_start", e.get("frame"))
    events.sort(key=key)
    for i, e in enumerate(events):
        e["id"] = i + 1
    return events


# -- one window, end to end --------------------------------------------------------------------

def analyse_window(lo, hi, args):
    """Run the whole test on video frames lo..hi (inside one half). Returns a dict of arrays
    (window-relative) plus the event list. Used by main() and by visualize_events.py."""
    block = [(h, a, b) for h, a, b in half_video_blocks(GAME) if a <= lo <= b]
    if not block:
        raise ValueError("start frame {} is not inside a half; halves are {}".format(
            lo, [(a, b) for _, a, b in half_video_blocks(GAME)]))
    half, _, half_end = block[0]
    hi = min(hi, half_end)

    meta_path, raw_path = require(["atd_meta", "atd_raw"])
    team_of_slot = team_of_each_track(meta_path)
    px, py, bx, by = load_tracking(raw_path, len(team_of_slot), lo, hi)
    bx, by = clean_ball(bx, by)
    n = hi - lo + 1

    res = control_per_frame(px, py, bx, by, team_of_slot, args)
    dead, reason, n_out = dead_mask(res["sbx"], res["sby"], res["ctrl"], n, args)
    if half == 2:
        k_end = min(n, SECOND_HALF_KICKOFF - lo)
        if k_end > 0:
            dead[:k_end] = True
            reason[:k_end] = 3

    stats = {k: 0 for k in ("changes_interception", "changes_contest",
                            "cancelled_interception", "cancelled_contest")}
    log = []
    labels = run_state_machine(res["ctrl"], res["sbx"], dead, 0, n - 1, args, stats, log)
    for item in log:                       # window-relative -> absolute frames
        item["start"] += lo
        item["decided"] += lo

    res.update({"lo": lo, "hi": hi, "n": n, "half": half, "labels": labels, "dead": dead,
                "dead_reason": reason, "n_out": n_out, "stats": stats, "log": log,
                "team_of_slot": team_of_slot, "team_names": team_names(meta_path),
                "ball_seen": float(np.mean(~np.isnan(bx)))})
    res["events"] = detect_events(res, team_of_slot, lo)
    return res


def possessions_of(labels, lo):
    out = []
    for k0, k1 in runs_of(labels != 2):
        sub = labels[k0:k1]
        breaks = np.flatnonzero(sub[1:] != sub[:-1]) + 1
        bounds = np.concatenate(([0], breaks, [len(sub)]))
        for a, b in zip(bounds[:-1], bounds[1:]):
            out.append({"team": TEAM_LETTER[int(sub[a])], "frame_start": lo + k0 + int(a),
                        "frame_end": lo + k0 + int(b) - 1, "time_start": clock(lo + k0 + int(a)),
                        "duration_s": r2((b - a) / float(FPS))})
    for i, p in enumerate(out):
        p["id"] = i + 1
    return out


def write_json(res, args, path, detail):
    lo, n = res["lo"], res["n"]
    team_of_slot = res["team_of_slot"]
    ev_counts = {}
    for e in res["events"]:
        name = e["type"] + ("_" + e["outcome"] if "outcome" in e else "")
        ev_counts[name] = ev_counts.get(name, 0) + 1
    labels = res["labels"]
    doc = {
        "meta": {
            "script": "test_scripts/possession_test_1.py",
            "game": GAME,
            "teams": {"H": res["team_names"][0] if res["team_names"] else "home",
                      "A": res["team_names"][1] if len(res["team_names"]) > 1 else "away"},
            "window": {"frame_start": lo, "frame_end": res["hi"], "half": res["half"],
                       "time_start": clock(lo), "time_end": clock(res["hi"]),
                       "seconds": r2(n / float(FPS))},
            "settings": {k: v for k, v in vars(args).items() if k not in ("out",)},
            "constants": {"zoi_front_gain": ZOI_FRONT_GAIN, "zoi_side_shrink": ZOI_SIDE_SHRINK,
                          "zoi_rear_shrink": ZOI_REAR_SHRINK, "min_ctrl_s": MIN_CTRL_S,
                          "contest_memory_s": CONTEST_MEMORY_S, "carry_min_m": CARRY_MIN_M,
                          "touch_gap_frames": TOUCH_GAP_FRAMES,
                          "flight_min_frames": FLIGHT_MIN_FRAMES},
            "units": {"x": "metres along the pitch, 0 to {}".format(PITCH_L),
                      "y": "metres across the pitch, 0 to {}".format(PITCH_W),
                      "speed": "m/s", "frame": "tactical video frame, 25 per second",
                      "time": "video time mm:ss.ss from the first video frame",
                      "player": "track slot 1..46 (not a person; slots swap players)",
                      "zoi": "ellipse: front / rear radius along heading_deg, side radius "
                             "across it; ball_in_zone means q <= 1"},
            "warning": "Frame-level tracking derived from the match data. Do not publish or "
                       "upload (RULES.md).",
        },
        "summary": dict(
            possessions=int(len(possessions_of(labels, lo))),
            label_share={L: r2(np.mean(labels == i)) for i, L in enumerate("HAD")},
            control_share={CTRL_NAME[v]: r2(np.mean(res["ctrl"] == v))
                           for v in (HOME, AWAY, BOTH, NONE)},
            ball_tracked_share=r2(res["ball_seen"]),
            ball_out_of_pitch_frames=res["n_out"],
            event_counts=ev_counts, **res["stats"]),
        "possessions": possessions_of(labels, lo),
        "events": res["events"],
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if detail:
        ux, uy, front, rear, side = zoi_geometry(res["vx"], res["vy"], args)
        heading = np.degrees(np.arctan2(uy, ux))
        frames = []
        for k in range(n):
            players = []
            for s in np.flatnonzero(~np.isnan(res["spx"][k])):
                q = res["q"][k, s]
                players.append({
                    "player": int(s) + 1, "team": TEAM_LETTER[int(team_of_slot[s])],
                    "x": r2(res["spx"][k, s]), "y": r2(res["spy"][k, s]),
                    "vx": r2(res["vx"][k, s]), "vy": r2(res["vy"][k, s]),
                    "speed": r2(np.hypot(res["vx"][k, s], res["vy"][k, s])),
                    "zoi": {"heading_deg": r2(heading[k, s]), "front_m": r2(front[k, s]),
                            "rear_m": r2(rear[k, s]), "side_m": r2(side[k, s])},
                    "zoi_q": r2(q) if np.isfinite(q) and q < 25 else None,
                    "ball_in_zone": bool(q <= 1.0),
                    "ball_speed_matches": bool(res["slow"][k, s]) if q <= 1.0 else None,
                })
            hold = int(res["holder"][k])
            frames.append({
                "frame": lo + k, "time": clock(lo + k), "label": "HAD"[labels[k]],
                "control": CTRL_NAME[int(res["ctrl"][k])],
                "holder": hold + 1 if hold >= 0 else None,
                "dead": bool(res["dead"][k]),
                "dead_reason": DEAD_REASONS[int(res["dead_reason"][k])] or None,
                "ball": {"x": r2(res["sbx"][k]), "y": r2(res["sby"][k]),
                         "vx": r2(res["bvx"][k]), "vy": r2(res["bvy"][k]),
                         "speed": r2(np.hypot(res["bvx"][k], res["bvy"][k])),
                         "tracked": bool(np.isfinite(res["sbx"][k]))},
                "players": players,
            })
        frames_path = path.replace("events_", "frames_") if "events_" in path \
            else path.replace(".json", "_frames.json")
        doc["meta"]["frames_file"] = os.path.basename(frames_path)
        with open(frames_path, "w") as handle:      # compact: about 5 MB a minute
            json.dump({"meta": doc["meta"], "frames": frames}, handle, separators=(",", ":"))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        json.dump(doc, handle, indent=2)
    return ev_counts


# -- reporting -------------------------------------------------------------------------------

def load_metrics():
    """evaluation/possession_metrics.py, imported by path so the folder layout does not matter."""
    path = os.path.join(ROOT, "evaluation", "possession_metrics.py")
    spec = importlib.util.spec_from_file_location("possession_metrics", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def describe(pm, codes, runs, label):
    spells = pm.possession_spells(codes, runs)
    lengths = np.array([b - a for a, b, _ in spells]) / float(FPS)
    print("  {}: {} possessions, mean {:.1f} s, median {:.1f} s, under 2 s: {}".format(
        label, len(spells), lengths.mean() if len(lengths) else 0,
        np.median(lengths) if len(lengths) else 0, int((lengths < 2).sum())))


def clock(frame):
    """Video time mm:ss.ss of a video frame, to find it in the tactical video."""
    seconds = frame / float(FPS)
    return "{:02d}:{:05.2f}".format(int(seconds // 60), seconds % 60)


def print_changes(pm, codes, runs, title):
    names = "HADX"
    changes = pm.transitions(codes, runs)
    print("  {} ({}):".format(title, len(changes)))
    for frame, frm, to in changes:
        print("    frame {:>6}  video {}  {} -> {}".format(frame, clock(frame), names[frm],
                                                       names[to]))


def add_settings_args(parser):
    """The thresholds, shared with visualize_events.py."""
    parser.add_argument("--zoi-r0", type=float, default=ZOI_R0_M)
    parser.add_argument("--zoi-speed-max", type=float, default=ZOI_SPEED_MAX)
    parser.add_argument("--rel-speed-max", type=float, default=REL_SPEED_MAX)
    parser.add_argument("--intercept-hold", type=float, default=INTERCEPT_HOLD_S)
    parser.add_argument("--contest-hold", type=float, default=CONTEST_HOLD_S)
    parser.add_argument("--dead-gap", type=float, default=DEAD_GAP_S)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--start", type=int, default=WINDOW_START,
                        help="first video frame of the test window (default: 1st-half kickoff)")
    parser.add_argument("--seconds", type=float, default=WINDOW_SECONDS,
                        help="length of the test window in seconds (default 60)")
    add_settings_args(parser)
    parser.add_argument("--out", default=None, help="output CSV path")
    parser.add_argument("--json", default=None, help="output JSON path")
    parser.add_argument("--no-frame-detail", action="store_true",
                        help="leave the per-frame, per-player section out of the JSON")
    args = parser.parse_args()

    lo = args.start
    hi = lo + int(round(args.seconds * FPS)) - 1
    t0 = time.time()
    try:
        res = analyse_window(lo, hi, args)
    except ValueError as error:
        sys.exit(str(error))
    hi, n = res["hi"], res["n"]
    print("test window: video frames {}..{} ({:.0f} s, half {}, video {} to {})".format(
        lo, hi, n / float(FPS), res["half"], clock(lo), clock(hi)))
    print("  analysed in {:.1f} s; ball visible on {:.0%} of frames".format(
        time.time() - t0, res["ball_seen"]))
    labels, ctrl, stats = res["labels"], res["ctrl"], res["stats"]

    # Full-length file: the window is labelled, everything else is X, so it still validates.
    n_frames = source_frames("tactical")
    letters = np.full(n_frames, "X", dtype="<U1")
    letters[lo:hi + 1] = np.array(["H", "A", "D"])[labels]
    out_path = args.out or os.path.join(ROOT, "evaluation", "results", OUT_NAME + ".csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, v) for i, v in enumerate(letters))
    print("  wrote {}".format(out_path))

    json_path = args.json or os.path.join(REPORT_DIR, "events_{}.json".format(lo))
    detail = not args.no_frame_detail and n / float(FPS) <= DETAIL_MAX_S
    if not args.no_frame_detail and not detail:
        print("  note: window over {:.0f} s, per-frame detail left out of the JSON".format(
            DETAIL_MAX_S))
    ev_counts = write_json(res, args, json_path, detail)
    print("  wrote {}".format(json_path))
    if detail:
        print("  wrote {}".format(json_path.replace("events_", "frames_")))

    pm = load_metrics()
    table = pm.read_label_csv(out_path)
    problem = pm.validate_submission(table, n_frames)
    print("\n1. file format check: {}".format("OK" if problem is None else "FAILED - " + problem))
    if problem is not None:
        return
    codes = pm.label_codes(table)
    runs = [(lo, hi + 1)]

    print("\n2. inside the window")
    describe(pm, codes, runs, "predicted")
    print("  label mix: H {:.0%}  A {:.0%}  D {:.0%}".format(
        *[float(np.mean(labels == i)) for i in range(3)]))
    print("  control frames: home {:.0%}, away {:.0%}, contest {:.0%}, free/no ball {:.0%}".format(
        *[float(np.mean(ctrl == v)) for v in (HOME, AWAY, BOTH, NONE)]))
    print("  ball off the pitch on {} frames".format(res["n_out"]))
    print("  changes accepted: {} interceptions, {} after contests".format(
        stats["changes_interception"], stats["changes_contest"]))
    print("  changes cancelled (old team got it back): {} interceptions, {} contests".format(
        stats["cancelled_interception"], stats["cancelled_contest"]))
    print("  events: " + ", ".join("{} {}".format(v, k) for k, v in sorted(ev_counts.items())))
    print_changes(pm, codes, runs, "predicted label changes")

    # Public ground truth, restricted to the window. Sanity check only.
    gt_path = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
    if os.path.isfile(gt_path):
        gt = pm.label_codes(pm.read_label_csv(gt_path))
        inside = np.zeros(n_frames, dtype=bool)
        inside[lo:hi + 1] = True
        gt = np.where(inside, gt, 3).astype(gt.dtype)
        gt_runs = pm.valid_runs(gt)
        if not gt_runs:
            print("\n3. no public ground truth in this window (it covers frames 11185..18684)")
        else:
            print("\n3. public ground truth in this window - SANITY CHECK ONLY, do not tune on it")
            describe(pm, gt, gt_runs, "truth    ")
            print_changes(pm, gt, gt_runs, "true label changes")
            filled = pm.fill_abstentions(codes, gt_runs)
            agree = float(np.mean(filled[inside & (gt != 3)] == gt[inside & (gt != 3)]))
            print("  frames where our label matches the truth: {:.0%}".format(agree))
            print("  (spellF1 is not shown: on one minute it is too noisy to mean anything)")

    print("\nsettings: " + ", ".join("{}={}".format(k, v) for k, v in vars(args).items()
                                     if k not in ("out", "json")))
    print("total time {:.1f} s".format(time.time() - t0))


if __name__ == "__main__":
    main()
