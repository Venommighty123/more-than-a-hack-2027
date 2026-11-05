"""Possession test 1: zone-of-influence ball control with a contest rule.

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

Runs on a SHORT window only: by default the first 60 s of the match (video frames 11185..12684,
inside the public ground truth). Run from the repo root (numpy + pandas only, a few seconds):

    python test_scripts/possession_test_1.py
    python test_scripts/possession_test_1.py --start 13000 --seconds 60      # another minute
    python test_scripts/possession_test_1.py --contest-hold 2 --zoi-r0 2.0   # try variants

Writes evaluation/results/possession_test_1.csv (the window labelled, every other frame X, so it
still passes the format check) and prints:
  1. the format check (same checks as the official scorer),
  2. inside the window: possession count and lengths, control-state mix, change counts by kind,
     and every predicted label change with its video time, so you can check it in the video,
  3. if the window overlaps the public 5 minutes, the true label changes in the same window and
     the share of frames that agree. A SANITY check only: do NOT tune thresholds on it (the
     rules forbid it, and one minute is far too little to tune on anyway).

Only aggregate numbers are printed, so the output is safe to paste back into the project thread.
"""

import argparse
import importlib.util
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

FPS = GAMES[GAME]["fps"]
PITCH_L, PITCH_W = GAMES[GAME]["pitch"]
SECOND_HALF_KICKOFF = 113742       # published in FAQ.md: graded from 110808, all D until here
HOME, AWAY, NONE, BOTH = 0, 1, -1, 2
OUT_NAME = "possession_test_1"
WINDOW_START = 11185             # 1st-half kickoff, inside the public ground truth
WINDOW_SECONDS = 60.0


# -- loading ---------------------------------------------------------------------------------

def team_of_each_track(meta_path):
    """0 for home slots, 1 for away, in Player order (first Team in the XML is home)."""
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    if len(team_ids) != 2:
        raise ValueError("expected 2 teams in {}, found {}".format(meta_path, len(team_ids)))
    slots = [player.get("teamId") for player in root.iter("Player")]
    return np.array([HOME if t == team_ids[0] else AWAY for t in slots], dtype=np.int8)


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


def zoi_inside(px, py, vx, vy, bx, by, args):
    """Boolean (n_frames x n_slots): is the ball inside each player's zone of influence?"""
    dx = bx[:, None] - px
    dy = by[:, None] - py
    speed = np.hypot(vx, vy)
    speed = np.where(np.isnan(speed), 0.0, speed)
    ratio = np.clip(speed / args.zoi_speed_max, 0.0, 1.0)

    # Unit direction of travel; a standing player gets an arbitrary axis (the zone is a circle).
    ux = np.where(speed > 0.2, np.nan_to_num(vx) / np.maximum(speed, 1e-6), 1.0)
    uy = np.where(speed > 0.2, np.nan_to_num(vy) / np.maximum(speed, 1e-6), 0.0)
    along = dx * ux + dy * uy
    side = -dx * uy + dy * ux

    r0 = args.zoi_r0
    front = r0 * (1.0 + ZOI_FRONT_GAIN * ratio)
    rear = r0 * (1.0 - ZOI_REAR_SHRINK * ratio)
    lateral = r0 * (1.0 - ZOI_SIDE_SHRINK * ratio)
    radius_along = np.where(along >= 0, front, rear)
    q = (along / radius_along) ** 2 + (side / lateral) ** 2
    return np.nan_to_num(q, nan=np.inf) <= 1.0


def control_per_frame(px, py, bx, by, team_of_slot, args):
    """Per frame: HOME / AWAY (sole control), BOTH (contest), NONE (free ball or no ball)."""
    spx, spy = nan_smooth(px, SMOOTH_HALF), nan_smooth(py, SMOOTH_HALF)
    vx, vy = velocity(spx), velocity(spy)
    sbx, sby = nan_smooth(bx, SMOOTH_HALF), nan_smooth(by, SMOOTH_HALF)
    bvx, bvy = velocity(sbx), velocity(sby)

    home_slots = team_of_slot == HOME
    n = len(bx)
    home = np.zeros(n, dtype=bool)
    away = np.zeros(n, dtype=bool)
    contest = np.zeros(n, dtype=bool)
    for a in range(0, n, CHUNK):        # chunks keep memory to a few hundred MB
        b = min(n, a + CHUNK)
        inside = zoi_inside(spx[a:b], spy[a:b], vx[a:b], vy[a:b], sbx[a:b], sby[a:b], args)
        rel = np.hypot(bvx[a:b, None] - np.nan_to_num(vx[a:b]),
                       bvy[a:b, None] - np.nan_to_num(vy[a:b]))
        # Where the ball velocity is unknown (track edges), accept position alone.
        slow = np.where(np.isnan(rel), True, rel <= args.rel_speed_max)
        controls = inside & slow
        home[a:b] = controls[:, home_slots].any(axis=1)
        away[a:b] = controls[:, ~home_slots].any(axis=1)
        # A contest is both teams having the ball in their zones, whatever its speed.
        contest[a:b] = inside[:, home_slots].any(axis=1) & inside[:, ~home_slots].any(axis=1)

    ctrl = np.full(len(bx), NONE, dtype=np.int8)
    ctrl[home & ~away] = HOME
    ctrl[away & ~home] = AWAY
    ctrl[contest] = BOTH
    return ctrl, sbx, sby


def dead_mask(sbx, sby, ctrl, n_frames, args):
    """True where the ball is taken as dead: long gaps and ball out of the pitch, each until the
    next frame some team controls the ball (the restart)."""
    missing = np.isnan(sbx)
    out = (~missing) & ((sbx < -OUT_MARGIN_M) | (sbx > PITCH_L + OUT_MARGIN_M)
                        | (sby < -OUT_MARGIN_M) | (sby > PITCH_W + OUT_MARGIN_M))
    dead = out.copy()

    # Long gaps.
    edges = np.diff(np.concatenate(([0], missing.astype(np.int8), [0])))
    for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if stop - start >= args.dead_gap * FPS:
            dead[start:stop] = True

    # Extend each dead stretch until the next controlled frame.
    controlled = (ctrl == HOME) | (ctrl == AWAY)
    i = 0
    while i < n_frames:
        if dead[i]:
            j = i
            while j < n_frames and (dead[j] or not controlled[j]):
                j += 1
            dead[i:j] = True
            i = j
        else:
            i += 1
    return dead, int(out.sum())


# -- the possession state machine --------------------------------------------------------------

def run_state_machine(ctrl, ball_x, dead, first, last, args, stats):
    """Labels (0 H, 1 A, 2 D) for frames first..last inclusive."""
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
            pending = None

        if pending is not None:
            if not np.isnan(ball_x[f]):
                pending["observed"] += 1
            hold = contest_hold if pending["kind"] == "contest" else intercept_hold
            if pending["observed"] >= hold and pending["ctrl"] >= min(min_ctrl, hold):
                # Backdate: the new team has had the ball since pending["start"].
                labels[pending["start"]:k + 1] = pending["team"]
                stats["changes_" + pending["kind"]] += 1
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


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--start", type=int, default=WINDOW_START,
                        help="first video frame of the test window (default: 1st-half kickoff)")
    parser.add_argument("--seconds", type=float, default=WINDOW_SECONDS,
                        help="length of the test window in seconds (default 60)")
    parser.add_argument("--zoi-r0", type=float, default=ZOI_R0_M)
    parser.add_argument("--zoi-speed-max", type=float, default=ZOI_SPEED_MAX)
    parser.add_argument("--rel-speed-max", type=float, default=REL_SPEED_MAX)
    parser.add_argument("--intercept-hold", type=float, default=INTERCEPT_HOLD_S)
    parser.add_argument("--contest-hold", type=float, default=CONTEST_HOLD_S)
    parser.add_argument("--dead-gap", type=float, default=DEAD_GAP_S)
    parser.add_argument("--out", default=None, help="output CSV path")
    args = parser.parse_args()

    # The window must sit inside one half.
    lo = args.start
    hi = lo + int(round(args.seconds * FPS)) - 1
    block = [(h, a, b) for h, a, b in half_video_blocks(GAME) if a <= lo <= b]
    if not block:
        sys.exit("start frame {} is not inside a half; halves are {}".format(
            lo, [(a, b) for _, a, b in half_video_blocks(GAME)]))
    half, _, half_end = block[0]
    hi = min(hi, half_end)
    print("test window: video frames {}..{} ({:.0f} s, half {}, video {} to {})".format(
        lo, hi, (hi - lo + 1) / float(FPS), half, clock(lo), clock(hi)))

    t0 = time.time()
    meta_path, raw_path = require(["atd_meta", "atd_raw"])
    n_frames = source_frames("tactical")
    team_of_slot = team_of_each_track(meta_path)
    px, py, bx, by = load_tracking(raw_path, len(team_of_slot), lo, hi)
    bx, by = clean_ball(bx, by)
    n = hi - lo + 1
    print("  read {} frames in {:.1f} s; ball visible on {:.0%} of them".format(
        n, time.time() - t0, float(np.mean(~np.isnan(bx)))))

    ctrl, sbx, sby = control_per_frame(px, py, bx, by, team_of_slot, args)
    dead, n_out = dead_mask(sbx, sby, ctrl, n, args)
    if half == 2:
        k_end = min(n, SECOND_HALF_KICKOFF - lo)
        if k_end > 0:
            dead[:k_end] = True

    stats = {k: 0 for k in ("changes_interception", "changes_contest",
                            "cancelled_interception", "cancelled_contest")}
    labels = run_state_machine(ctrl, sbx, dead, 0, n - 1, args, stats)

    # Full-length file: the window is labelled, everything else is X, so it still validates.
    letters = np.full(n_frames, "X", dtype="<U1")
    letters[lo:hi + 1] = np.array(["H", "A", "D"])[labels]
    out_path = args.out or os.path.join(ROOT, "evaluation", "results", OUT_NAME + ".csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, v) for i, v in enumerate(letters))
    print("  wrote {}".format(out_path))

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
    print("  ball off the pitch on {} frames".format(n_out))
    print("  changes accepted: {} interceptions, {} after contests".format(
        stats["changes_interception"], stats["changes_contest"]))
    print("  changes cancelled (old team got it back): {} interceptions, {} contests".format(
        stats["cancelled_interception"], stats["cancelled_contest"]))
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
                                     if k != "out"))
    print("total time {:.1f} s".format(time.time() - t0))


if __name__ == "__main__":
    main()
