"""Pass detection: every pass in the match as JSON, built on starter/ball_flight.py.

WHY ball_flight.py ALONE IS NOT ENOUGH
  ball_flight.py only looks at the ball. It finds "the ball is moving fast" stretches (a speed
  hysteresis on the ball track) and fits a parabola to decide whether each one was in the air. It
  never looks at the players, so it cannot say who kicked the ball, who got it, or whether that
  was a teammate. Its stretches also stop when the ball lands or slows, not when someone receives
  it, so one long pass is often several "flights", and slow ground passes (under 12 m/s) are
  never flights at all.

WHAT THIS SCRIPT ADDS
  1. KICKS       ball_flight.detect_candidates() with a lower start speed (KICK_SPEED, 7 m/s)
                 gives the frames where the ball starts moving fast.
  2. PASSER      the player nearest the ball in the KICK_LOOKBACK frames before that, within
                 KICK_R_M. No one close: the movement is "unattributed" (a bounce, a deflection,
                 or tracking noise) and is not a pass.
  3. RECEPTION   follow the ball forward. The pass ends at the first frame where a player is
                 within RECEIVE_R_M of the ball AND the ball is slow enough to be controlled
                 (< RECEIVE_SPEED) or shows a touch (its velocity vector changes by TOUCH_DVEC). The
                 passer only counts again after the ball has left them by LEAVE_R_M.
                 Other endings: ball leaves the pitch (out_of_play), ball lost by the tracking for
                 more than ball_flight's max_gap frames (ball_lost), or nothing after
                 MAX_PASS_S seconds (no_receiver).
  4. COMPLETION  pass_completed is inferred from the TEAM OF THE RECEIVER: same team -> true
                 (outcome "complete"), other team -> false ("intercepted"), out of play -> false,
                 ball lost / no receiver -> null (unknown). There is no event data to check this
                 against: it is a tracking-based inference, see LIMITS.
  5. TYPE        pass, or one of the non-passes the same machinery finds:
                   touch         the passer keeps the ball (dribble touch, first touch, slot jitter)
                   shot          aimed at goal from within SHOT_MAX_DIST_M, fast, and ending at
                                 the goal line or with the opponent keeper area
                   unattributed  no passer found
                 is_pass is true only for type "pass". All rows are kept so you can audit them.
  6. HEIGHT      (optional, --height) ball_flight's parabola fit on each pass: max_height_m,
                 is_airborne, pass_height "ground"/"high". Needs scipy + the homography file, and
                 ball_flight.py's KEY ASSUMPTION must hold (see its docstring); unverified.

COORDINATES
  Metres, origin at the pitch corner, x along the 105.3 m length, y along the 68 m width, same
  as games.py and ball_flight.py (ATD normalised * pitch size). The *_att fields rotate the pitch
  so the passing team always attacks towards +x (like StatsBomb), using each team's average x in
  that half to tell which way they attack.

LIMITS (read before trusting the numbers)
  - Player slots are tracklets, not people (DATA.md). passer_slot / receiver_slot are slots.
  - A slow pass rolling past a defender within RECEIVE_R_M counts as intercepted. A lofted pass
    flying over a defender does not (it is too fast), which is the intended behaviour.
  - Ball gaps: 41% of graded frames have no ball, often long stretches. Passes inside those are
    invisible here. Gaps up to max_gap (12 frames) are bridged by interpolation.
  - Shots and clearances are heuristics; clearances are not separated from long passes.
  - Thresholds are UNVALIDATED starting points. Do NOT compare or tune them against published
    match statistics or the public 5 minutes (RULES.md rule 2: human-produced data about this game).
    The defaults below were picked earlier by ranking against published stats and must be
    re-derived on other games (docs/compliance.md).

Run from the repo root (numpy + scipy; pandas only for --height):
    python scripts/detect_passes.py                       # whole match
    python scripts/detect_passes.py --start 11185 --seconds 120   # a window, for checking
    python scripts/detect_passes.py --height               # add ball height (slower)

Writes outputs/passes/passes[_<start>].json (+ .csv with the same rows). These files are
derived from the tracking data: keep them out of the public repo (the folder is git-ignored) and
do not upload them to any cloud AI tool. The printed summary has only aggregates and is safe to
paste into the project thread.
"""

import argparse
import csv
import heapq
import json
import os
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import replace

import numpy as np

import possession.io  # noqa: F401,E402  (repository root on sys.path)
from possession.io import ROOT  # noqa: E402
from games import (GAME, GAMES, anchor_video_frame, half_video_blocks, require,  # noqa: E402
                   source_frames)
from possession.features.flight import (FlightConfig, fill_short_gaps,  # noqa: E402
                                        finite_runs)
from possession.features.flight import detect_candidates_legacy as detect_candidates  # noqa: E402

# -- params (all overridable from the command line) ------------------------------------------
KICK_SPEED = 5.0         # m/s   ball faster than this starts a kick candidate (ball_flight: 12)
KICK_STOP_SPEED = 3.0    # m/s   candidate ends below this
KICK_MIN_FRAMES = 3      # frames a candidate must last (ball_flight: 10, too long for short passes)
KICK_R_M = 2.0           # passer must be this close to the ball at the kick
KICK_LOOKBACK = 6        # frames before the candidate start searched for the passer
MIN_PASS_M = 3.0         # a "reception" closer than this to the kick point is a duel / tackle /
                         # block, not a pass (spot check Nov 6: 5 of 20 "interceptions" were
                         # under 0.3 m: the kick's own velocity change read as an opponent touch)
LEAVE_R_M = 3.0          # ball must get this far from the passer before the passer can receive
RECEIVE_R_M = 1.5        # receiver must be this close to the ball
RECEIVE_SPEED = 4.0      # m/s   ball this slow near a player = controlled (was 8: a pass
                         #       rolling past a defender at 5-8 m/s counted as intercepted)
TOUCH_DVEC = 3.0         # m/s   velocity-vector change that marks a touch (see ball_kinematics)
TOUCH_HALF = 5           # frames either side used for that change
TOUCH_DVEC_MIN = 3.0     # m/s   floor for the calibrated threshold
TOUCH_NOISE_PCT = 95.0   # threshold = this percentile of the free-ball noise (calibrate_touch)
FREE_BALL_M = 5.0        # ball this far from every player cannot be touched
TOUCH_AUTO = False       # calibrate TOUCH_DVEC from the data (--touch N fixes it instead)
CONFIRM_PASSER = False   # vote on the passer's team before the kick (sweep 3: made it worse)
CONFIRM_FRAMES = 8       # who has the ball over this many frames after the reception decides
                         # the receiving TEAM (majority), not the first player it reaches
DWELL_FRAMES = 0         # ball this many consecutive frames within RECEIVE_R_M of one player
                         # also counts as received (0 = off)
CHAIN_WINDOW = 8         # frames after a reception in which a fast ball means a one-touch pass
MAX_PASS_S = 6.0         # give up following the ball after this long
OUT_MARGIN_M = 0.5       # ball this far past a line is out of play
MAX_BALL_SPEED = 40.0    # m/s   faster frame-to-frame jumps are tracking errors
SHOT_MAX_DIST_M = 35.0   # shots are only taken from this close to goal
SHOT_MIN_KMH = 50.0      # and are at least this fast
GOAL_HALF_W = 3.66       # half the goal width, metres
SHOT_AIM_MARGIN_M = 2.0  # aim line may miss the goal by this much and still be a shot attempt

FPS = GAMES[GAME]["fps"]
PITCH_L, PITCH_W = GAMES[GAME]["pitch"]
HOME, AWAY = 0, 1
TEAM_LETTER = {HOME: "H", AWAY: "A"}
OUT_DIR = os.path.join(ROOT, "outputs", "passes")


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
    """Player (frames x slots) and ball positions in metres, video frames lo..hi inclusive.

    Same parse as possession_test_1.py: '<frame>:<x,y,s;x,y,s;...>:<ball x,y,...>', ATD frames
    count from 1, video frames from 0. NaN where an object is not seen.
    """
    n = hi - lo + 1
    px = np.full((n, n_slots), np.nan, dtype=np.float32)
    py = np.full((n, n_slots), np.nan, dtype=np.float32)
    ball = np.full((n, 2), np.nan)
    per_slot = None
    with open(raw_path) as handle:
        for line in handle:
            head = line[:line.index(":")] if ":" in line else ""
            if not head.isdigit():
                continue
            frame = int(head) - 1
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
            b = parts[2].split(",")
            if len(b) >= 2 and b[0][:1] != "N":
                ball[k] = (float(b[0]) * PITCH_L, float(b[1]) * PITCH_W)
    return px, py, ball


def clean_ball(ball):
    """Drop single-frame teleports (implausible speed) from the ball track."""
    step = np.hypot(*np.diff(ball, axis=0, prepend=np.nan).T) * FPS
    out = ball.copy()
    out[step > MAX_BALL_SPEED] = np.nan
    return out


# -- ball kinematics ---------------------------------------------------------------------------

def ball_kinematics(filled, cfg):
    """Per-frame smoothed ball speed (m/s) and velocity change (m/s) across the frame.

    touch[j] = |v(j + TOUCH_HALF) - v(j - TOUCH_HALF)|, the change of the ball's VELOCITY VECTOR
    around frame j. A ball rolling past a player keeps its velocity (friction takes well under
    1 m/s off in 6 frames), a touch changes it (stop, redirect, speed up). Using the vector, not
    speed and heading separately, matters: heading is pure noise when the ball is slow, which
    made the first version see "touches" on balls that were only rolling past a defender.

    Same smoothing as ball_flight.detect_candidates, done once for the whole track. NaN outside
    gap-free runs.
    """
    from scipy.signal import savgol_filter
    n = len(filled)
    speed = np.full(n, np.nan)
    vel = np.full((n, 2), np.nan)
    for a, b in finite_runs(~np.isnan(filled).any(axis=1)):
        seg = filled[a:b]
        s = savgol_filter(seg, cfg.smooth_window, 2, axis=0) if len(seg) >= cfg.smooth_window \
            else seg
        d = np.diff(s, axis=0, prepend=s[:1])
        if len(d) > 1:
            d[0] = d[1]
        vel[a:b] = d * FPS
        speed[a:b] = np.hypot(d[:, 0], d[:, 1]) * FPS
    h = TOUCH_HALF
    touch = np.full(n, np.nan)
    touch[h:-h] = np.hypot(*(vel[2 * h:] - vel[:-2 * h]).T)
    return speed, touch


# -- pitch helpers -----------------------------------------------------------------------------

def half_of(frame, blocks):
    for half, first, last in blocks:
        if first <= frame <= last:
            return half
    return None


def attack_signs(px, team_of_slot, lo, blocks):
    """{(half, team): +1 if the team attacks towards +x in that half, else -1}.

    The team whose players average the lower x defends the x=0 goal and attacks +x.
    """
    signs = {}
    for half, first, last in blocks:
        a, b = max(first - lo, 0), min(last - lo + 1, len(px))
        if b <= a:
            continue
        cols = [px[a:b][:, team_of_slot == t] for t in (HOME, AWAY)]
        if not all(np.isfinite(c).any() for c in cols):
            continue
        means = [np.nanmean(c) for c in cols]
        home_plus = means[HOME] < means[AWAY]
        signs[(half, HOME)] = 1 if home_plus else -1
        signs[(half, AWAY)] = -1 if home_plus else 1
    return signs


def to_attacking(x, y, sign):
    """Rotate so the team attacks towards +x (StatsBomb-style)."""
    return (x, y) if sign > 0 else (PITCH_L - x, PITCH_W - y)


def outside(x, y):
    return (x < -OUT_MARGIN_M or x > PITCH_L + OUT_MARGIN_M or
            y < -OUT_MARGIN_M or y > PITCH_W + OUT_MARGIN_M)


def out_line(x, y):
    return "goal_line" if (x < -OUT_MARGIN_M or x > PITCH_L + OUT_MARGIN_M) else "touchline"


def in_box(x_att, y_att):
    """Opponent penalty area, in attacking coordinates."""
    return bool(x_att >= PITCH_L - 16.5 and abs(y_att - PITCH_W / 2) <= 20.16)


def clock(frame, kickoffs):
    """Match clock 'MM:SS' (2nd half starts at 45:00)."""
    if frame >= kickoffs[2]:
        sec = 45 * 60 + (frame - kickoffs[2]) / float(FPS)
    else:
        sec = (frame - kickoffs[1]) / float(FPS)
    sec = max(sec, 0.0)
    return "{:02d}:{:02d}".format(int(sec // 60), int(sec % 60))


def r2(value):
    if value is None:
        return None
    value = float(value)
    return round(value, 2) if np.isfinite(value) else None


# -- the detector ------------------------------------------------------------------------------

def find_passer(k, filled, px, py, lo_k):
    """(kick frame, slot, distance) of the nearest player in the lookback window, or None."""
    a = max(k - KICK_LOOKBACK, lo_k)
    b = k + 1
    d = np.hypot(px[a:b] - filled[a:b, 0:1], py[a:b] - filled[a:b, 1:2])
    if not np.isfinite(d).any():
        return None
    d = np.where(np.isfinite(d), d, np.inf)
    best = d.min(axis=1)
    close = np.flatnonzero(best <= KICK_R_M)
    if not len(close):
        return None
    j = close[-1]                       # last frame the passer was still on the ball = release
    slot = int(np.argmin(d[j]))
    return a + j, slot, float(d[j, slot])


def follow_ball(k0, passer, filled, observed, px, py, speed, touch, team_of_slot, kick_speed):
    """Walk forward from the kick until reception / out of play / ball lost / timeout.

    An OPPONENT only ends the pass by controlling the ball: it is slow next to them, or they
    touch it and it does not fly on. A fast ball that merely changes velocity near an opponent
    (a deflection, a block that loops on, or tracking jitter as the ball passes them) is
    followed further: wherever it ends up decides the outcome. Without this rule every ball
    rolling past a defender was an "interception" AND, through the one-touch chaining, a fake
    pass by the defender's team.

    Returns dict(end, ending, receiver, receiver_dist, deflected_by).
    """
    n = len(filled)
    stop = min(n, k0 + 1 + int(MAX_PASS_S * FPS))
    seg = slice(k0 + 1, stop)
    dist = np.hypot(px[seg] - filled[seg, 0:1], py[seg] - filled[seg, 1:2])
    dist = np.where(np.isfinite(dist), dist, np.inf)
    left = passer is None
    last_seen = k0
    deflected_by = None
    ignore = np.zeros(dist.shape[1], dtype=bool)   # opponents already passed (deflectors)
    near_slot, near_run = -1, 0

    def done(**kw):
        kw.setdefault("receiver", None)
        kw.setdefault("receiver_dist", None)
        return dict(kw, deflected_by=deflected_by)
    for i, j in enumerate(range(k0 + 1, stop)):
        if np.isnan(filled[j]).any():
            return done(end=last_seen, ending="ball_lost")
        if observed[j]:
            last_seen = j
        if outside(*filled[j]):
            return done(end=j, ending="out_of_play")
        row = dist[i].copy()
        row[ignore] = np.inf
        if passer is not None and not left:
            if row[passer] >= LEAVE_R_M:
                left = True
            else:
                row[passer] = np.inf      # passer cannot receive their own pass yet
        q = int(np.argmin(row))
        if row[q] <= RECEIVE_R_M:
            near_run = near_run + 1 if q == near_slot else 1
            near_slot = q
            if DWELL_FRAMES and near_run >= DWELL_FRAMES:
                first = j - near_run + 1     # ball has stayed with this player: control
                return done(end=first, ending="received", receiver=q,
                            receiver_dist=float(dist[first - k0 - 1, q]))
            slow = np.isfinite(speed[j]) and speed[j] <= RECEIVE_SPEED
            near = touch[max(j - 1, 0):j + 2]
            touched = np.isfinite(near).any() and np.nanmax(near) >= TOUCH_DVEC
            opponent = passer is not None and team_of_slot[q] != team_of_slot[passer]
            if opponent and not slow and touched:
                ahead = speed[j + 1:j + 1 + CHAIN_WINDOW]
                if np.isfinite(ahead).any() and np.nanmin(ahead) > kick_speed:
                    deflected_by = q            # ball flies on: deflection, keep following
                    ignore[q] = True
                    continue
            if slow or touched:
                return done(end=j, ending="received", receiver=q, receiver_dist=float(row[q]))
        else:
            near_slot, near_run = -1, 0
        if passer is not None and not left and j - k0 > FPS:
            # still at the passer's feet after 1 s: a dribble touch, not a pass
            return done(end=j, ending="kept", receiver=passer)
    return done(end=stop - 1, ending="no_receiver")


def confirm_passer(k0, filled, px, py, team_of_slot, first):
    """Passer's TEAM decided by who had the ball over the CONFIRM_FRAMES frames up to the kick.

    Same idea as confirm_receiver, looking backwards: in a duel the player nearest the ball at
    the release frame is often the tackler, which turned a home pass into an away "pass" that
    the home team "intercepted" (one error on each side's count and accuracy).
    """
    return _vote(slice(max(0, k0 - CONFIRM_FRAMES + 1), k0 + 1), filled, px, py, team_of_slot,
                 first, KICK_R_M + 0.5)


def confirm_receiver(k1, filled, px, py, team_of_slot, first):
    """Receiver decided by who is on the ball over the next CONFIRM_FRAMES frames.

    The first player the ball comes within RECEIVE_R_M of is often a marker standing next to
    the real receiver: with ~1 m of position noise on ball and players, "nearest at one frame"
    is close to a coin toss in a duel. Counting, frame by frame, which team's player is nearest
    (within RECEIVE_R_M + 0.5 m) and taking the majority is far more stable. The receiver is
    that team's most frequent nearest player.
    """
    return _vote(slice(k1, min(len(filled), k1 + CONFIRM_FRAMES)), filled, px, py,
                 team_of_slot, first, RECEIVE_R_M + 0.5)


def _vote(seg, filled, px, py, team_of_slot, first, radius):
    """Majority team nearest the ball (within radius) over seg; keeps first on a tie."""
    d = np.hypot(px[seg] - filled[seg, 0:1], py[seg] - filled[seg, 1:2])
    d = np.where(np.isfinite(d), d, np.inf)
    if not len(d):
        return first
    near = np.argmin(d, axis=1)
    ok = d[np.arange(len(d)), near] <= radius
    if not ok.any():
        return first
    slots = near[ok]
    teams = team_of_slot[slots]
    votes = np.bincount(teams, minlength=2)
    first_team = team_of_slot[first]
    if votes[first_team] >= votes[1 - first_team]:
        return first                    # ties keep the first contact
    win = slots[teams == 1 - first_team]
    return int(np.bincount(win).argmax())


def calibrate_touch(filled, speed, touch, px, py, verbose=True):
    """Set TOUCH_DVEC from this match's own noise floor (no ground truth involved).

    On frames where the ball moves and no player is within FREE_BALL_M, nobody can touch it,
    so any velocity change there is tracking noise (plus bounces and friction). The touch
    threshold is the TOUCH_NOISE_PCT percentile of that, never below TOUCH_DVEC_MIN.
    """
    global TOUCH_DVEC
    ok = np.isfinite(touch) & (speed > KICK_STOP_SPEED)
    idx = np.flatnonzero(ok)
    if len(idx) > 200000:
        idx = idx[::len(idx) // 200000 + 1]
    d = np.hypot(px[idx] - filled[idx, 0:1], py[idx] - filled[idx, 1:2])
    nearest = np.min(np.where(np.isfinite(d), d, np.inf), axis=1)
    free = touch[idx[nearest > FREE_BALL_M]]
    if len(free) < 500:
        if verbose:
            print("touch threshold: too few free-ball frames ({}), keeping {:.1f} m/s".format(
                len(free), TOUCH_DVEC))
        return
    p50, p90, p99 = np.percentile(free, [50, 90, 99])
    TOUCH_DVEC = max(TOUCH_DVEC_MIN, float(np.percentile(free, TOUCH_NOISE_PCT)))
    if verbose:
        print("touch threshold: free-ball velocity noise p50 {:.1f}, p90 {:.1f}, p99 {:.1f} m/s "
          "({} frames) -> threshold {:.1f} m/s".format(p50, p90, p99, len(free), TOUCH_DVEC))


def aimed_at_goal(x0, y0, x1, y1, sign):
    """Does the line from kick to end cross the attacked goal line inside the goal (+margin)?"""
    xa0, ya0 = to_attacking(x0, y0, sign)
    xa1, ya1 = to_attacking(x1, y1, sign)
    if xa1 <= xa0 or PITCH_L - xa0 > SHOT_MAX_DIST_M:
        return False
    t = (PITCH_L - xa0) / (xa1 - xa0)
    y_goal = ya0 + t * (ya1 - ya0)
    return abs(y_goal - PITCH_W / 2) <= GOAL_HALF_W + SHOT_AIM_MARGIN_M


def detect_passes(px, py, ball, team_of_slot, lo, blocks, kickoffs, names, args, height=None):
    cfg = replace(FlightConfig(fps=FPS, pitch=(PITCH_L, PITCH_W)),
                  start_speed=args.kick_speed, stop_speed=KICK_STOP_SPEED,
                  min_flight_frames=KICK_MIN_FRAMES, smooth_window=5)   # Nov 7 detector settings
    ball = clean_ball(ball)
    observed = ~np.isnan(ball).any(axis=1)
    filled, _ = fill_short_gaps(ball, cfg.max_gap)
    speed, touch = ball_kinematics(filled, cfg)
    if TOUCH_AUTO:
        calibrate_touch(filled, speed, touch, px, py, verbose=True)
    signs = attack_signs(px, team_of_slot, lo, blocks)

    # Kick candidates: ball_flight's flight starts, plus (added while walking) every reception
    # that is followed at once by a fast ball. ball_flight never starts a new flight there,
    # because a one-touch pass keeps the ball fast through the touch: without this, every
    # one-touch pass is lost (the previous pass is credited, the next one is not).
    starts = []
    run_start_of = np.full(len(filled), -1)
    for a, b in finite_runs(~np.isnan(filled).any(axis=1)):
        run_start_of[a:b] = a
        for i0, _ in detect_candidates(filled[a:b], cfg):
            starts.append((a + i0, a, "flight"))
    heapq.heapify(starts)
    diag = dict(flight_starts=len(starts), chained_starts=0, skipped_inside_previous=0,
                deflections_followed=0, receiver_changed_by_confirm=0,
                passer_changed_by_confirm=0, touch_threshold_ms=r2(TOUCH_DVEC))
    rows, busy_until = [], -1
    while starts:
        s, run_start, origin = heapq.heappop(starts)
        if s <= busy_until:
            diag["skipped_inside_previous"] += 1
            continue                    # still inside the previous pass (landing, bounce, roll)
        found = find_passer(s, filled, px, py, run_start)
        if found is None:
            k0, passer, passer_dist = s, None, None
        else:
            k0, passer, passer_dist = found
            if CONFIRM_FRAMES and CONFIRM_PASSER:
                q = confirm_passer(k0, filled, px, py, team_of_slot, passer)
                if q != passer:
                    diag["passer_changed_by_confirm"] += 1
                    passer = q
        f = follow_ball(k0, passer, filled, observed, px, py, speed, touch, team_of_slot,
                        args.kick_speed)
        if f["deflected_by"] is not None:
            diag["deflections_followed"] += 1
        k1 = f["end"]
        busy_until = k1
        if f["ending"] == "received" and CONFIRM_FRAMES and passer is not None:
            q = confirm_receiver(k1, filled, px, py, team_of_slot, f["receiver"])
            if q != f["receiver"]:
                diag["receiver_changed_by_confirm"] += 1
                f["receiver"] = q
        if f["ending"] == "received":
            busy_until = k1 - 1         # the reception frame may also be the next kick
            after = speed[k1:k1 + CHAIN_WINDOW + 1]
            if np.isfinite(after).any() and np.nanmax(after) > args.kick_speed and \
                    run_start_of[k1] >= 0:
                heapq.heappush(starts, (k1, int(run_start_of[k1]), "chain"))
                diag["chained_starts"] += 1

        frame0, frame1 = int(lo + k0), int(lo + k1)
        half = half_of(frame0, blocks)
        x0, y0 = filled[k0]
        x1, y1 = filled[k1]
        seg_speed = speed[k0:k1 + 1]
        top_kmh = float(np.nanmax(seg_speed)) * 3.6 if np.isfinite(seg_speed).any() else np.nan
        length = float(np.hypot(x1 - x0, y1 - y0))
        team = int(team_of_slot[passer]) if passer is not None else None
        sign = signs.get((half, team), 1) if team is not None else 1
        xa0, ya0 = to_attacking(x0, y0, sign)
        xa1, ya1 = to_attacking(x1, y1, sign)
        receiver = f["receiver"]
        recv_team = int(team_of_slot[receiver]) if receiver is not None else None

        # type
        if passer is None:
            kind = "unattributed"
        elif f["ending"] == "kept" or receiver == passer:
            kind = "touch"
        elif f["ending"] == "received" and length < MIN_PASS_M:
            kind = "duel" if recv_team != team else "touch"
        elif (top_kmh >= SHOT_MIN_KMH and aimed_at_goal(x0, y0, x1, y1, sign) and
              (f["ending"] == "out_of_play" and out_line(x1, y1) == "goal_line" or
               recv_team is not None and recv_team != team and xa1 >= PITCH_L - 6.0)):
            kind = "shot"
        else:
            kind = "pass"

        # outcome and completion
        if f["ending"] == "received":
            outcome = "complete" if recv_team == team else "intercepted"
            completed = recv_team == team
        elif f["ending"] == "out_of_play":
            outcome, completed = "out_of_play", False
        elif f["ending"] == "kept":
            outcome, completed = "kept", None
        else:
            outcome, completed = f["ending"], None      # ball_lost / no_receiver: unknown
        if passer is None:
            completed = None

        # pressure: nearest opponent to the passer at the kick, and to the receiver at reception
        def nearest_opp(k, slot):
            if slot is None:
                return None
            opp = team_of_slot != team_of_slot[slot]
            d = np.hypot(px[k, opp] - px[k, slot], py[k, opp] - py[k, slot])
            return float(np.nanmin(d)) if np.isfinite(d).any() else None

        seen = observed[k0:k1 + 1].mean() if k1 >= k0 else 0.0
        row = dict(
            is_pass=kind == "pass", type=kind, outcome=outcome, pass_completed=completed,
            team=TEAM_LETTER.get(team), team_name=names[team] if team is not None else None,
            passer_slot=passer + 1 if passer is not None else None,
            receiver_slot=receiver + 1 if receiver is not None and kind != "touch" else None,
            receiver_team=TEAM_LETTER.get(recv_team) if kind != "touch" else None,
            half=half, frame_start=frame0, frame_end=frame1,
            match_clock=clock(frame0, kickoffs), video_time_s=r2(frame0 / float(FPS)),
            duration_s=r2((k1 - k0) / float(FPS)),
            pass_begin_x=r2(x0), pass_begin_y=r2(y0), pass_end_x=r2(x1), pass_end_y=r2(y1),
            pass_begin_x_att=r2(xa0), pass_begin_y_att=r2(ya0),
            pass_end_x_att=r2(xa1), pass_end_y_att=r2(ya1),
            length_m=r2(length),
            angle_deg=r2(np.degrees(np.arctan2(ya1 - ya0, xa1 - xa0))),
            progress_m=r2(xa1 - xa0),
            direction=("forward" if xa1 - xa0 > max(2.0, 0.3 * length)
                       else "backward" if xa0 - xa1 > max(2.0, 0.3 * length) else "lateral"),
            into_final_third=bool(xa0 < 2 * PITCH_L / 3 <= xa1),
            into_box=bool(in_box(xa1, ya1) and not in_box(xa0, ya0)),
            out_line=out_line(x1, y1) if outcome == "out_of_play" else None,
            max_ball_speed_kmh=r2(top_kmh),
            passer_ball_dist_m=r2(passer_dist), receiver_ball_dist_m=r2(f["receiver_dist"]),
            passer_nearest_opponent_m=r2(nearest_opp(k0, passer)),
            receiver_nearest_opponent_m=r2(nearest_opp(k1, receiver)) if kind == "pass"
            and outcome == "complete" else None,
            ball_tracked_share=r2(seen),
            deflected_by_slot=f["deflected_by"] + 1 if f["deflected_by"] is not None else None,
        )
        if height is not None:
            row.update(height(k0, k1, filled, observed))
        rows.append(row)

    for i, row in enumerate(rows):
        row["id"] = i + 1
    return rows, diag


def make_height_fn(lo, n, cfg):
    """Wrap ball_flight's parabola fit for one pass. Needs pandas + scipy + the homography."""
    from possession.features.flight import _characterise, estimate_focal
    from starter.load_homography import load as load_h
    Hn, _, valid = load_h()
    Hn, valid = Hn[lo:lo + n], valid[lo:lo + n]
    f, spread = estimate_focal(Hn, valid, cfg)
    print("height fit: focal length {:.0f} px (relative IQR {:.2f})".format(f, spread))

    def height(k0, k1, filled, observed):
        if k1 - k0 < cfg.min_obs:
            return dict(max_height_m=None, is_airborne=None, pass_height=None)
        try:
            fl = _characterise(k0, k1, filled, observed, Hn, valid, f, cfg)
        except Exception:               # a bad fit should not stop the whole match
            return dict(max_height_m=None, is_airborne=None, pass_height=None)
        if not fl.fit_ok:
            return dict(max_height_m=None, is_airborne=None, pass_height=None)
        return dict(max_height_m=r2(fl.max_height_m), is_airborne=fl.is_airborne,
                    pass_height="high" if fl.is_airborne else "ground")
    return height


# -- reporting ---------------------------------------------------------------------------------

def summarise(rows, names):
    summary = {}
    for t, letter in TEAM_LETTER.items():
        mine = [r for r in rows if r["team"] == letter]
        passes = [r for r in mine if r["is_pass"]]
        known = [r for r in passes if r["pass_completed"] is not None]
        done = [r for r in known if r["pass_completed"]]
        lengths = [r["length_m"] for r in passes if r["length_m"] is not None]
        summary[letter] = dict(
            team_name=names[t], passes=len(passes), completed=len(done),
            completion_known=len(known),
            completion_pct=r2(100.0 * len(done) / len(known)) if known else None,
            outcomes={o: sum(r["outcome"] == o for r in passes) for o in
                      ("complete", "intercepted", "out_of_play", "ball_lost", "no_receiver")},
            by_half={h: sum(r["half"] == h for r in passes) for h in (1, 2)},
            median_length_m=r2(np.median(lengths)) if lengths else None,
            forward=sum(r["direction"] == "forward" for r in passes),
            into_final_third=sum(r["into_final_third"] for r in passes),
            into_box=sum(r["into_box"] for r in passes),
            shots=sum(r["type"] == "shot" for r in mine),
            touches=sum(r["type"] == "touch" for r in mine),
            duels=sum(r["type"] == "duel" for r in mine),
        )
    summary["unattributed"] = sum(r["type"] == "unattributed" for r in rows)
    return summary


def print_summary(summary, n_rows, seconds, elapsed):
    print("\n{} ball movements in {:.0f} s of play, {:.1f} s to run".format(
        n_rows, seconds, elapsed))
    print("{:<26}{:>8}{:>10}{:>9}{:>10}{:>8}{:>8}{:>9}{:>8}".format(
        "team", "passes", "complete", "pct", "intercept", "out", "lost", "noRecv", "med m"))
    for letter in TEAM_LETTER.values():
        s = summary[letter]
        o = s["outcomes"]
        print("{:<26}{:>8}{:>10}{:>9}{:>10}{:>8}{:>8}{:>9}{:>8}".format(
            "{} ({})".format(letter, (s["team_name"] or "")[:20]), s["passes"], s["completed"],
            "{}%".format(s["completion_pct"]), o["intercepted"], o["out_of_play"],
            o["ball_lost"], o["no_receiver"], s["median_length_m"]))
    for letter in TEAM_LETTER.values():
        s = summary[letter]
        print("{}: per half {}, forward {}, into final third {}, into box {}, shots {}, "
              "dribble touches {}, duels (ball lost within {:.0f} m) {}".format(
                  letter, s["by_half"], s["forward"], s["into_final_third"], s["into_box"],
                  s["shots"], s["touches"], MIN_PASS_M, s["duels"]))
    print("unattributed ball movements (no player near the kick): {}".format(
        summary["unattributed"]))
    d = summary["kick_candidates"]
    print("kick candidates: {} from ball_flight, {} one-touch (chained from a reception), "
          "{} skipped as inside the previous pass".format(
              d["flight_starts"], d["chained_starts"], d["skipped_inside_previous"]))
    print("deflections followed through an opponent: {}; touch threshold {} m/s; receivers "
          "changed by the confirm window: {}".format(
              d["deflections_followed"], d["touch_threshold_ms"],
              d["receiver_changed_by_confirm"]))
    print("passers changed by the confirm window: {}".format(d["passer_changed_by_confirm"]))


def _plain(value):
    """numpy scalars -> plain Python for json."""
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return r2(value)
    raise TypeError(type(value))


def write_outputs(rows, meta, summary, stem):
    os.makedirs(OUT_DIR, exist_ok=True)
    json_path = os.path.join(OUT_DIR, stem + ".json")
    with open(json_path, "w") as handle:
        json.dump(dict(meta=meta, summary=summary, passes=rows), handle, indent=1, default=_plain)
    csv_path = os.path.join(OUT_DIR, stem + ".csv")
    if rows:
        keys = ["id"] + [k for k in rows[0] if k != "id"]
        with open(csv_path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
    print("\nwrote {}\n      {}".format(os.path.relpath(json_path, ROOT),
                                       os.path.relpath(csv_path, ROOT)))


def set_radii(args):
    global RECEIVE_R_M, KICK_R_M
    if args.receive_r is not None:
        RECEIVE_R_M = args.receive_r
    if args.kick_r is not None:
        KICK_R_M = args.kick_r
    global RECEIVE_SPEED, TOUCH_DVEC, TOUCH_AUTO, TOUCH_HALF, DWELL_FRAMES
    if args.receive_speed is not None:
        RECEIVE_SPEED = args.receive_speed
    if args.touch == "auto":
        TOUCH_AUTO = True
    elif args.touch is not None:
        TOUCH_AUTO, TOUCH_DVEC = False, float(args.touch)
    if args.touch_half is not None:
        TOUCH_HALF = args.touch_half
    if args.dwell is not None:
        DWELL_FRAMES = args.dwell
    global CONFIRM_FRAMES
    if args.confirm is not None:
        CONFIRM_FRAMES = args.confirm
    global CONFIRM_PASSER
    if args.confirm_passer is not None:
        CONFIRM_PASSER = bool(args.confirm_passer)


# -- entry point -------------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--start", type=int, default=None, help="first video frame (default: "
                        "the whole match)")
    parser.add_argument("--seconds", type=float, default=None, help="window length")
    parser.add_argument("--height", action="store_true", help="add ball_flight's height fit")
    parser.add_argument("--kick-speed", type=float, default=KICK_SPEED)
    parser.add_argument("--receive-r", type=float, default=None, help="metres, default {}".format(
        RECEIVE_R_M))
    parser.add_argument("--kick-r", type=float, default=None, help="metres, default {}".format(
        KICK_R_M))
    parser.add_argument("--receive-speed", type=float, default=None)
    parser.add_argument("--touch", default=None, help="touch threshold m/s (default {}), or auto".format(TOUCH_DVEC))
    parser.add_argument("--touch-half", type=int, default=None)
    parser.add_argument("--dwell", type=int, default=None)
    parser.add_argument("--confirm", type=int, default=None,
                        help="frames used to confirm the receiving team (0 = off)")
    parser.add_argument("--confirm-passer", type=int, choices=[0, 1], default=None,
                        help="1 = also vote on the passing team before the kick")
    args = parser.parse_args(argv)
    set_radii(args)

    t0 = time.time()
    meta_path, raw_path = require(["atd_meta", "atd_raw"])
    team_of_slot = team_of_each_track(meta_path)
    names = team_names(meta_path)
    blocks = half_video_blocks(GAME)
    n_total = source_frames("tactical", GAME)
    if args.start is None:
        lo, hi = blocks[0][1], blocks[-1][2]
        stem = "passes"
    else:
        lo = args.start
        hi = min(n_total - 1, lo + int((args.seconds or 120.0) * FPS) - 1)
        stem = "passes_{}".format(lo)
    kickoffs = {1: anchor_video_frame(GAMES[GAME]["kickoff"]),
                2: anchor_video_frame(GAMES[GAME]["kickoff_2nd"])}

    print("loading tracking for video frames {}..{} ...".format(lo, hi))
    px, py, ball = load_tracking(raw_path, len(team_of_slot), lo, hi)
    # Keep only the two halves: tracking between them (halftime) or before kickoff is not
    # match play.
    in_match = np.zeros(hi - lo + 1, dtype=bool)
    for _, first, last in blocks:
        in_match[max(first - lo, 0):max(min(last - lo + 1, hi - lo + 1), 0)] = True
    print("ignoring {} loaded frames outside the two halves".format(
        int((~in_match & ~np.isnan(ball).any(axis=1)).sum())))
    px[~in_match], py[~in_match], ball[~in_match] = np.nan, np.nan, np.nan
    print("loaded in {:.1f} s; ball seen on {:.1f}% of frames".format(
        time.time() - t0, 100.0 * np.mean(~np.isnan(ball).any(axis=1))))

    height = None
    if args.height:
        cfg = FlightConfig(fps=FPS, pitch=(PITCH_L, PITCH_W))
        height = make_height_fn(lo, hi - lo + 1, cfg)

    rows, diag = detect_passes(px, py, ball, team_of_slot, lo, blocks, kickoffs, names, args, height)
    summary = summarise(rows, names)
    summary["kick_candidates"] = diag
    in_play = sum(max(0, min(b, hi) - max(a, lo) + 1) for _, a, b in blocks)
    meta = dict(
        game=GAME, frames=[lo, hi], fps=FPS, pitch_m=[PITCH_L, PITCH_W],
        units="metres, origin at pitch corner, x along length; *_att fields: passing team "
              "attacks +x; frames are tactical video frames (submission index)",
        completion_rule="pass_completed = receiver (first player within receive_r_m of a "
                        "controllable ball) is on the passer's team; null when the ball is "
                        "lost or nobody receives it",
        settings=dict(kick_speed_ms=args.kick_speed, kick_r_m=KICK_R_M, leave_r_m=LEAVE_R_M,
                      receive_r_m=RECEIVE_R_M, receive_speed_ms=RECEIVE_SPEED,
                      touch_dvec_ms=TOUCH_DVEC, touch_auto=TOUCH_AUTO, touch_half=TOUCH_HALF,
                      dwell_frames=DWELL_FRAMES, confirm_frames=CONFIRM_FRAMES, confirm_passer=CONFIRM_PASSER, max_pass_s=MAX_PASS_S,
                      height_fit=bool(args.height)),
        note="Player ids are track slots, not people. Derived from tracking: do not publish.",
    )
    write_outputs(rows, meta, summary, stem)
    print_summary(summary, len(rows), in_play / float(FPS), time.time() - t0)


if __name__ == "__main__":
    main()
