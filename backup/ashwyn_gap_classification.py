"""Gap classification: what kind of ball gap is each one, and how much of the match is truly blind?

Add-on to ball_gap_analysis.py. That script finds every ball gap inside the halves and writes
reports/data_audit/ball_gaps.csv (half, start frame, length, ...). It does not say what a gap IS.
This script reads that file and classifies every gap longer than MIN_GAP_S:

  dead_kickoff    inside the published pre-kickoff stretch of a half (FAQ.md)
  dead_tagged     a Smart Tagging SET PIECES tag (throw-in, corner, goal kick, free kick, kick-off)
                  overlaps the gap or starts within TAG_SLACK_S of its end. Label D, and the tag
                  gives the restart team.
  dead_untagged   the ball disappeared or reappeared over / within NEAR_LINE_M of a pitch line AND
                  the players were slow during the gap (median team speed < DEAD_SPEED_MS)
  live            everything else: the ball is in play but nobody can see it. The blind part.

For each gap it also records
  team_before / team_after   who CONTROLLED the ball (player of that team within CONTROL_R_M,
                             ball slower than CONTROL_SPEED_MS) in the CONTEXT_S before / after
                             the gap; more robust than the nearest-player call in ball_gap_analysis
  flip                       team_before != team_after (only when both are known)
  phase_team                 the Smart Tagging phase team covering most of the gap (H / A / mixed
                             / none); phases are not possessions, but they are a free prior
  player_speed_ms            median over the gap of the mean speed of all tracked players

and prints, per class: gaps, seconds, share of the graded frames, flip rate, how often the phase
team agrees with team_after, and the player speed. Aggregates only: safe to paste in the thread.

    python ball_gap_analysis.py                    # once, if reports/data_audit/ball_gaps.csv is missing
    python test_scripts/gap_classification.py
    python test_scripts/gap_classification.py --dead-speed 2.0 --near-line 3   # sensitivity check

Writes reports/gap_classification/gaps_classified.csv (one row per gap: frames, class, teams, no
positions). Keep it out of the public repo with the rest of reports/.

Thresholds are UNVALIDATED starting points. The point of this run is the size of each class, so
also look at how the table moves with --dead-speed and --near-line before trusting one number.
"""

import argparse
import csv
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _p in (ROOT, os.path.join(ROOT, "starter"), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from games import GAME, GAMES, half_video_blocks, require  # noqa: E402
from pass_detection import load_tracking, team_of_each_track, clean_ball  # noqa: E402

# -- params (overridable from the command line) ----------------------------------------------
MIN_GAP_S = 1.0          # shorter gaps are bridged by interpolation already
NEAR_LINE_M = 2.0        # ball this close to (or past) a line when it vanishes / reappears
DEAD_SPEED_MS = 1.8      # median mean-player speed below this during the gap = play stopped
TAG_SLACK_S = 3.0        # a set-piece tag this close to the gap still belongs to it
CONTEXT_S = 5.0          # look this far before / after the gap for the controlling team
CONTROL_R_M = 1.5        # control = a player this close to the ball ...
CONTROL_SPEED_MS = 6.0   # ... and the ball this slow
PHASE_MAJORITY = 0.6     # phase team = a team covering at least this share of the gap

FPS = GAMES[GAME]["fps"]
PITCH_L, PITCH_W = GAMES[GAME]["pitch"]
HOME, AWAY = 0, 1
LETTER = {HOME: "H", AWAY: "A"}
KICKOFFS = {1: 11185, 2: 113742}                 # published (DATA.md / FAQ.md)
GAPS_CSV = os.path.join(ROOT, "reports", "data_audit", "ball_gaps.csv")
OUT_DIR = os.path.join(ROOT, "reports", "gap_classification")
CLASSES = ("dead_kickoff", "dead_tagged", "dead_untagged", "live")


# -- inputs -----------------------------------------------------------------------------------

def load_gaps(path):
    if not os.path.isfile(path):
        raise IOError("{} not found: run `python ball_gap_analysis.py` first".format(path))
    with open(path) as handle:
        rows = list(csv.DictReader(handle))
    return [(int(r["half"]), int(r["start_frame"]), int(r["length_frames"])) for r in rows]


def load_tags():
    from starter.load_smart_tagging import load
    try:
        tags = load()
    except IOError as error:
        print("note: no Smart Tagging ({}), dead_tagged will be empty".format(
            str(error).splitlines()[0]))
        return [], []
    pieces = [t for t in tags if (t["code"] or "").strip().lower() == "set pieces"]
    phases = [t for t in tags if (t["code"] or "").strip().lower() not in ("set pieces", "goals")]
    return pieces, phases


def tag_team(value, names):
    t = (value or "").strip().lower()
    if not t or t in ("n/a", "na"):
        return None
    for code, name in zip((HOME, AWAY), names):
        n = (name or "").strip().lower()
        if n and (n == t or n in t or t in n):
            return code
    if "barcelona" in t:
        return HOME
    if "atletic" in t or "athletic" in t:
        return AWAY
    return None


# -- per-frame signals ------------------------------------------------------------------------

def mean_player_speed(px, py):
    """Mean speed (m/s) of all tracked players per frame, from 5-frame displacement."""
    k = 5
    sp = np.full(len(px), np.nan)
    d = np.hypot(px[k:] - px[:-k], py[k:] - py[:-k]) * FPS / k
    d[d > 12.0] = np.nan                           # slot swaps / teleports
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)   # all-NaN frames -> NaN
        sp[k:] = np.nanmean(d, axis=1)
    return sp


def control_team(px, py, ball, team_of_slot):
    """Per frame: HOME / AWAY when one player is within CONTROL_R_M of a slow ball, else -1."""
    n = len(ball)
    out = np.full(n, -1, dtype=np.int8)
    step = np.hypot(*np.diff(ball, axis=0, prepend=np.nan).T) * FPS
    d = np.hypot(px - ball[:, 0:1], py - ball[:, 1:2])
    d = np.where(np.isfinite(d), d, np.inf)
    near = np.argmin(d, axis=1)
    ok = (d[np.arange(n), near] <= CONTROL_R_M) & ~(step > CONTROL_SPEED_MS)
    ok &= np.isfinite(ball[:, 0])
    out[ok] = team_of_slot[near[ok]]
    return out


def line_distance(x, y):
    """Distance to the nearest pitch line (negative = outside the pitch), and which line."""
    cand = {"goal_line": min(x, PITCH_L - x), "touchline": min(y, PITCH_W - y)}
    line = min(cand, key=cand.get)
    return cand[line], line


# -- classification ---------------------------------------------------------------------------

def classify(gaps, px, py, ball, lo, ctrl, speed, pieces, phases, names, args):
    rows = []
    slack = int(TAG_SLACK_S * FPS)
    ctx = int(CONTEXT_S * FPS)
    for half, g0, length in gaps:
        if length < args.min_gap * FPS:
            continue
        g1 = g0 + length                        # first frame after the gap (video frames)
        k0, k1 = g0 - lo, g1 - lo
        row = dict(half=half, start_frame=g0, end_frame=g1 - 1, seconds=round(length / FPS, 2))

        # ball where it vanished / reappeared
        last = ball[k0 - 1] if k0 - 1 >= 0 else np.array([np.nan, np.nan])
        first = ball[k1] if k1 < len(ball) else np.array([np.nan, np.nan])
        near = []
        for p, key in ((last, "vanish"), (first, "reappear")):
            if np.isfinite(p).all():
                dist, line = line_distance(*p)
                row[key + "_line"] = line
                row[key + "_line_m"] = round(float(dist), 2)
                near.append(dist <= args.near_line)
            else:
                row[key + "_line"], row[key + "_line_m"] = None, None
                near.append(False)
        row["near_line"] = bool(any(near))

        # players
        sp = speed[max(k0, 0):k1]
        row["player_speed_ms"] = round(float(np.nanmedian(sp)), 2) if np.isfinite(sp).any() \
            else None
        slow = row["player_speed_ms"] is not None and row["player_speed_ms"] < args.dead_speed

        # who had it before / after
        before = ctrl[max(0, k0 - ctx):max(k0, 0)]
        before = before[before >= 0]
        after = ctrl[k1:k1 + ctx]
        after = after[after >= 0]
        tb = int(before[-1]) if len(before) else None
        ta = int(after[0]) if len(after) else None
        row["team_before"], row["team_after"] = LETTER.get(tb), LETTER.get(ta)
        row["flip"] = (tb != ta) if tb is not None and ta is not None else None

        # Smart Tagging: set pieces near the gap, phase team over the gap
        hit = [p for p in pieces if p["start_frame"] <= g1 + slack and p["end_frame"] >= g0 - slack]
        if hit:
            best = min(hit, key=lambda p: abs(p["start_frame"] - g1))
            row["set_piece"] = best["tags"].get("Type")
            row["set_piece_team"] = LETTER.get(tag_team(best["tags"].get("Team"), names))
        else:
            row["set_piece"], row["set_piece_team"] = None, None
        cover = {HOME: 0, AWAY: 0}
        for p in phases:
            t = tag_team(p["tags"].get("Team"), names)
            if t is None:
                continue
            ov = min(p["end_frame"], g1) - max(p["start_frame"], g0)
            if ov > 0:
                cover[t] += ov
        share = {t: c / float(length) for t, c in cover.items()}
        if share[HOME] >= PHASE_MAJORITY and share[HOME] > share[AWAY]:
            row["phase_team"] = "H"
        elif share[AWAY] >= PHASE_MAJORITY and share[AWAY] > share[HOME]:
            row["phase_team"] = "A"
        else:
            row["phase_team"] = "mixed" if (share[HOME] + share[AWAY]) > 0 else "none"

        # class
        if g0 < KICKOFFS.get(half, -1):         # before (or across) the half's kickoff
            row["cls"] = "dead_kickoff"
        elif hit:
            row["cls"] = "dead_tagged"
        elif row["near_line"] and slow:
            row["cls"] = "dead_untagged"
        else:
            row["cls"] = "live"
        rows.append(row)
    return rows


# -- report -----------------------------------------------------------------------------------

def report(rows, graded_frames, n_short, short_frames):
    print("\ngaps shorter than {:.0f} s (already bridged / ignored): {}  ({:.0f} s)".format(
        MIN_GAP_S, n_short, short_frames / float(FPS)))
    print("\n{:<14}{:>6}{:>9}{:>9}{:>8}{:>9}{:>8}{:>10}{:>10}".format(
        "class", "gaps", "seconds", "% graded", "flips", "flip %", "phase=", "phase n/a",
        "speed m/s"))
    total_s = 0.0
    for cls in CLASSES:
        r = [x for x in rows if x["cls"] == cls]
        sec = sum(x["seconds"] for x in r)
        total_s += sec
        known = [x for x in r if x["flip"] is not None]
        flips = sum(1 for x in known if x["flip"])
        ph = [x for x in r if x["phase_team"] in ("H", "A") and x["team_after"]]
        agree = sum(1 for x in ph if x["phase_team"] == x["team_after"])
        no_phase = sum(1 for x in r if x["phase_team"] in ("none", "mixed"))
        speeds = [x["player_speed_ms"] for x in r if x["player_speed_ms"] is not None]
        print("{:<14}{:>6}{:>9.0f}{:>8.1f}%{:>8}{:>9}{:>8}{:>10}{:>10}".format(
            cls, len(r), sec, 100.0 * sec * FPS / graded_frames,
            "{}/{}".format(flips, len(known)),
            "{:.0f}%".format(100.0 * flips / len(known)) if known else "-",
            "{}/{}".format(agree, len(ph)), no_phase,
            "{:.2f}".format(float(np.median(speeds))) if speeds else "-"))
    print("{:<14}{:>6}{:>9.0f}{:>8.1f}%".format("all >= {:.0f} s".format(MIN_GAP_S), len(rows),
                                               total_s, 100.0 * total_s * FPS / graded_frames))

    live = [x for x in rows if x["cls"] == "live"]
    if live:
        lengths = sorted(x["seconds"] for x in live)
        print("\nlive gaps: median {:.1f} s, longest {:.1f} s; with a known team on both sides: "
              "{}; flips among those: {}".format(
                  lengths[len(lengths) // 2], lengths[-1],
                  sum(1 for x in live if x["flip"] is not None),
                  sum(1 for x in live if x["flip"])))
        why = {"near a line, players fast": 0, "mid-pitch, players slow": 0,
               "mid-pitch, players fast": 0}
        for x in live:
            slow = x["player_speed_ms"] is not None and x["player_speed_ms"] < DEAD_SPEED_MS
            key = ("near a line, players fast" if x["near_line"] else
                   "mid-pitch, players slow" if slow else "mid-pitch, players fast")
            why[key] += 1
        print("why live: {}".format(why))
    print("\nHow to read it:\n  % graded = share of the 158,857 graded frames inside gaps of that "
          "class.\n  flips = gaps where the controlling team after differs from before (of those "
          "with both known).\n  phase= = Smart Tagging phase team agrees with the team after the "
          "gap (of gaps with a clear phase team).\n  The 'live' row is the truly blind part; "
          "'live' flips are where a wrong guess costs a whole spell.")


def write_rows(rows):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "gaps_classified.csv")
    keys = ["half", "start_frame", "end_frame", "seconds", "cls", "team_before", "team_after",
            "flip", "set_piece", "set_piece_team", "phase_team", "near_line", "vanish_line",
            "vanish_line_m", "reappear_line", "reappear_line_m", "player_speed_ms"]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print("\nwrote {}".format(os.path.relpath(path, ROOT)))


def main():
    global MIN_GAP_S, DEAD_SPEED_MS
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gaps", default=GAPS_CSV)
    ap.add_argument("--min-gap", type=float, default=MIN_GAP_S, help="seconds")
    ap.add_argument("--near-line", type=float, default=NEAR_LINE_M, help="metres")
    ap.add_argument("--dead-speed", type=float, default=DEAD_SPEED_MS, help="m/s")
    args = ap.parse_args()
    MIN_GAP_S, DEAD_SPEED_MS = args.min_gap, args.dead_speed

    meta_path, raw_path = require(["atd_meta", "atd_raw"])
    team_of_slot = team_of_each_track(meta_path)
    from pass_detection import team_names
    names = team_names(meta_path)
    blocks = half_video_blocks(GAME)
    lo, hi = blocks[0][1], blocks[-1][2]
    graded = sum(b - a + 1 for _, a, b in blocks)

    gaps = load_gaps(args.gaps)
    print("{} gaps from {}".format(len(gaps), os.path.relpath(args.gaps, ROOT)))
    print("loading tracking ...")
    px, py, ball = load_tracking(raw_path, len(team_of_slot), lo, hi)
    in_match = np.zeros(hi - lo + 1, dtype=bool)
    for _, a, b in blocks:
        in_match[a - lo:b - lo + 1] = True
    px[~in_match], py[~in_match], ball[~in_match] = np.nan, np.nan, np.nan
    ball = clean_ball(ball)
    speed = mean_player_speed(px, py)
    ctrl = control_team(px, py, ball, team_of_slot)
    pieces, phases = load_tags()
    print("Smart Tagging: {} set pieces, {} phases with a team tag".format(
        len(pieces), sum(1 for p in phases if tag_team(p["tags"].get("Team"), names) is not None)))
    rows = classify(gaps, px, py, ball, lo, ctrl, speed, pieces, phases, names, args)
    short = [g for g in gaps if g[2] < args.min_gap * FPS]
    report(rows, graded, len(short), sum(g[2] for g in short))
    write_rows(rows)


if __name__ == "__main__":
    main()
