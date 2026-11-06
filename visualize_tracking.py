"""Render a clip of the tactical video side by side with the tracking data AND the detected events.

Left panel : the tactical video, with the tracked players/ball projected onto it through the
             homography, plus the active events (actor ring, pass / carry / shot line) projected too.
Right panel: a top-down pitch with every tracked slot (home red, away blue), the ball, a short ball
             trail, the line to the player nearest the ball, and the events of event_detection.py
             drawn where they happen: pass arrows (green complete / orange intercepted), carry paths,
             holds, duels, tackles, interceptions, possession changes, shots, out-of-play points,
             restarts, foul candidates, ball gaps, and a grey veil while the ball is dead.
             A banner flashes at the top whenever a NEW event starts or the possession implied by
             the events changes team, so every change of event is visible as it happens.
Info bar   : frame / time / half, players and ball tracked; LIVE/DEAD and the possession implied by
             the events next to the public ground truth (agree / disagree); the current event and its
             outcome (OK = success, FAIL = failure, -- = neutral); Smart Tagging phases; a log of the
             last events with their outcome.
Timeline   : the event-implied possession band, the public GT band under it, one tick per event
             (coloured by type) with an outcome dot, and a cursor.

Besides the mp4 it writes, next to it:
  <out>_events.csv    every event of the clip (id, frames, team, slot, outcome, confidence, places)
  <out>_report.json   the progress report also printed on the console: counts and outcome rates per
                      team, confidence, ball visibility, consistency checks on the detector output,
                      and the event-implied possession scored against the public GT (macro F1).

Nothing is loaded that the clip does not need (see the lazy readers below), so the cost scales with
the clip, not the match. Only the events JSON is read whole (it is small).

Usage (repository root; data in ./data or $MTAH_DATA_DIR; events from event_detection.py):

    pip install numpy pandas opencv-python imageio-ffmpeg    # imageio-ffmpeg only if no system ffmpeg
    python visualize_tracking.py --start 07:27 --duration 60            # 1 min of play + events
    python visualize_tracking.py --events reports/events/events_full.json --start-frame 11185
    python visualize_tracking.py --event-id E00412                       # jump to one event (+-5 s)
    python visualize_tracking.py --event-type shot --nth 3 --pad 8       # the 3rd shot, 8 s around it
    python visualize_tracking.py --types pass,tackle,interception        # draw only those types
    python visualize_tracking.py --hold-on-event 12                      # freeze ~0.5 s on each new event
    python visualize_tracking.py --list-events --event-type pass         # print a table and exit
    python visualize_tracking.py --report-only --start-frame 11185 --duration 300   # no video at all
    python visualize_tracking.py --no-events                             # plain tracking view

--start is VIDEO time (hh:mm:ss, mm:ss or plain seconds) from the first frame of the tactical
video. Video frame = round(seconds * 25); ATD frame_id = video frame + 1 (see DATA.md).

The events, the video and the sidecar files are derived from the match data: keep them out of the
public repository and do not upload them to any cloud AI tool (RULES.md). The public GT is shown for
checking what the events mean, never for tuning thresholds on this game.
"""

import argparse
import bisect
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
ROOT = PARENT if os.path.isfile(os.path.join(PARENT, "games.py")) else HERE
for _p in (ROOT, os.path.join(ROOT, "starter"), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from games import GAME, GAMES, require, half_of_video_frame, half_video_blocks  # noqa: E402
from starter.baseline import team_of_each_track, team_names  # noqa: E402

try:
    import cv2
except ImportError:
    sys.exit("OpenCV is required:  pip install opencv-python")

# Params
FPS = GAMES[GAME]["fps"]
N_FRAMES = GAMES[GAME]["n_video_frames"]
PITCH_L, PITCH_W = GAMES[GAME]["pitch"]            # metres
HOME_BGR, AWAY_BGR = (40, 39, 214), (180, 119, 31)  # #d62728 / #1f77b4, in BGR
GRASS_BGR, LINE_BGR = (79, 143, 63), (255, 255, 255)
PANEL_W, PANEL_H = 960, 540
BAR_H = 70                  # kept for scripts that import it; the info bar is now INFO_H tall
INFO_H, TL_H = 124, 70      # info bar and timeline under the two panels
FONT = cv2.FONT_HERSHEY_SIMPLEX
DEFAULT_GT = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
DEFAULT_EVENTS = os.path.join(ROOT, "reports", "events", "events_full.json")
SEARCH_BLOCK = 1 << 20      # binary search narrows the ATD to ~1 MB, then scans lines linearly

# Event styling (BGR). Short codes are used on the timeline legend.
EVENT_STYLE = {
    "pass":              ((60, 200, 60), "P"),
    "carry":             ((0, 215, 255), "C"),
    "hold":              ((230, 230, 0), "Ho"),
    "shot":              ((0, 0, 255), "S"),
    "goal":              ((0, 200, 255), "G"),
    "duel":              ((255, 0, 255), "Du"),
    "tackle":            ((0, 110, 255), "T"),
    "interception":      ((147, 20, 255), "I"),
    "possession_change": ((255, 255, 255), "PC"),
    "ball_out_of_play":  ((140, 140, 255), "O"),
    "dead_ball":         ((150, 150, 150), "DB"),
    "restart":           ((255, 200, 120), "R"),
    "foul_candidate":    ((40, 40, 200), "F"),
    "tracking_gap":      ((100, 100, 100), "Gap"),
}
INTERCEPTED_BGR = (0, 140, 255)
STATUS_BGR = {"success": (80, 220, 80), "fail": (70, 70, 255), "neutral": (200, 200, 200)}
STATUS_TAG = {"success": "OK", "fail": "FAIL", "neutral": "--"}
LABEL_BGR = {"H": HOME_BGR, "A": AWAY_BGR, "D": (120, 120, 120), "X": (45, 45, 45)}
LABEL_NAME = {"H": "H (home)", "A": "A (away)", "D": "dead ball", "X": "unknown"}
# Which active event the info bar calls "NOW" when several overlap (first wins).
NOW_PRIORITY = ["goal", "shot", "foul_candidate", "possession_change", "tackle", "interception", "pass",
                "duel", "ball_out_of_play", "restart", "carry", "hold", "dead_ball", "tracking_gap"]
# Events that set the team in possession from their frame_start (the passer still has the ball at release).
POSSESSION_ANCHORS = {"carry", "hold", "pass", "tackle", "interception", "possession_change", "restart"}
PLAY_TYPES = {"pass", "carry", "hold", "tackle", "interception", "duel", "shot", "possession_change"}


# -- helpers -----------------------------------------------------------------------------------

def parse_time(text):
    """'1:15:49' / '07:27' / '447.4' -> seconds."""
    seconds = 0.0
    for part in str(text).split(":"):
        seconds = seconds * 60.0 + float(part)
    return seconds


def fmt_time(frame):
    s = frame / float(FPS)
    return "{:02d}:{:04.1f}".format(int(s // 60), s % 60)


def put(img, text, org, scale=0.55, color=(255, 255, 255), thick=1):
    """Outlined text, readable on both grass and video."""
    cv2.putText(img, text, org, FONT, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, color, thick, cv2.LINE_AA)


# -- lazy tracking -----------------------------------------------------------------------------

def _floats(group, k):
    out = []
    for value in group.strip().split(",")[:k]:
        try:
            out.append(float(value))          # float("NaN") -> nan
        except ValueError:
            out.append(np.nan)
    return out + [np.nan] * (k - len(out))


def _frame_id(line):
    return int(line[:line.index(b":")])


def seek_to_frame_id(handle, target_id):
    """Position a binary handle at (or shortly before) the first line whose frame_id >= target.

    The raw ATD is sorted by frame_id, so bisect on byte offsets down to ~1 MB, then let the
    caller skip the few remaining lines with a smaller id. Reads ~log2(filesize/1MB) lines
    instead of the whole prefix of the file.
    """
    handle.seek(0, os.SEEK_END)
    lo, hi = 0, handle.tell()
    while hi - lo > SEARCH_BLOCK:
        mid = (lo + hi) // 2
        handle.seek(mid - 1)
        handle.readline()                      # finish the line containing mid-1 -> line start
        line = handle.readline()
        if not line:                           # mid lies inside the last line
            hi = mid
        elif _frame_id(line) < target_id:
            lo = mid                           # everything up to here is before the target
        else:
            hi = mid
    if lo == 0:
        handle.seek(0)
    else:
        handle.seek(lo - 1)
        handle.readline()                      # realign to a line start at or after lo


def read_tracking(raw_path, first_frame, n, n_slots):
    """Keep only video frames [first_frame, first_frame + n) of the raw ATD.

    Returns players (n, n_slots, 3) = x, y (normalised 0-1), speed (m/s); ball (n, 2). NaN = missing.
    Raw line: "<frame_id>:<x,y,s>;<x,y,s>;...:<ball>", frame_id = video frame + 1.
    """
    players = np.full((n, n_slots, 3), np.nan, dtype=np.float32)
    ball = np.full((n, 2), np.nan, dtype=np.float32)
    lo, hi = first_frame + 1, first_frame + n          # inclusive ATD frame_id range
    with open(raw_path, "rb") as handle:
        seek_to_frame_id(handle, lo)
        for raw in handle:
            frame_id = _frame_id(raw)
            if frame_id < lo:
                continue
            if frame_id > hi:
                break
            parts = raw.decode("ascii", "replace").split(":")
            if len(parts) != 3:
                continue
            i = frame_id - lo
            for slot, group in enumerate(parts[1].split(";")[:n_slots]):
                players[i, slot] = _floats(group, 3)
            ball[i] = _floats(parts[2], 2)
    return players, ball


def load_homography_window(first_frame, n):
    """(matrices (n, 3, 3), valid (n,)) for the clip only; the file has one row per video frame."""
    import pandas as pd
    from starter.load_homography import MATRIX_COLUMNS
    path = require(["homography"])[0]
    table = pd.read_csv(path, skiprows=range(1, first_frame + 1), nrows=n)
    if len(table) == 0 or int(table["frame"].iloc[0]) != first_frame:
        raise ValueError("homography rows do not start at frame {}".format(first_frame))
    matrices = table[MATRIX_COLUMNS].values.astype(np.float64).reshape(-1, 3, 3)
    return matrices, table["valid"].values.astype(bool)


def load_gt_window(path, first_frame, n):
    """Optional public ground truth for the clip -> array of 'H','A','D','X' (or None)."""
    if not path or not os.path.isfile(path):
        return None
    import pandas as pd
    table = pd.read_csv(path, usecols=["frame", "label"], dtype={"label": "object"},
                        skiprows=range(1, first_frame + 1), nrows=n)
    if len(table) == 0 or int(table["frame"].iloc[0]) != first_frame:
        return None
    return table["label"].astype(str).str.strip().str.upper().values


def load_tags_window(enabled, first_frame, n):
    """Smart Tagging events overlapping the clip (the file is small; filter to keep lookups cheap)."""
    if not enabled:
        return []
    try:
        from starter import load_smart_tagging
        events = load_smart_tagging.load()
    except (IOError, ImportError) as error:
        print("note: Smart Tagging disabled ({})".format(str(error).splitlines()[0]))
        return []
    return [e for e in events if e["end_frame"] > first_frame and e["start_frame"] < first_frame + n]


def active_tags(events, frame, limit=3):
    short = {"barcelona": "H", "atletic club": "A", "athletic club": "A"}
    out = []
    for event in events:
        if event["start_frame"] <= frame < event["end_frame"]:
            team = short.get((event["tags"].get("Team") or "").strip().lower(), "N/A")
            kind = event["tags"].get("Type")
            out.append("{}{} [{}]".format((event["code"] or "").title(),
                                          " - " + kind if kind else "", team))
    return out[:limit]


# -- detected events (event_detection.py output) --------------------------------------------------

def load_events(path):
    """(events sorted by frame, meta) from events_*.json ({"meta", "events"}), a bare list, or .jsonl.

    Every event gets two extra keys used for display: _status (success / fail / neutral) and
    _outcome (short text). They are never written back to the detector's file.
    """
    if not path:
        return None, {}
    if not os.path.isfile(path):
        print("note: events file not found ({}); run event_detection.py first, or pass --events".format(path))
        return None, {}
    with open(path) as handle:
        if path.endswith(".jsonl"):
            events, meta = [json.loads(line) for line in handle if line.strip()], {}
        else:
            doc = json.load(handle)
            events, meta = (doc, {}) if isinstance(doc, list) else (doc.get("events", []), doc.get("meta", {}))
    for e in events:
        e["frame_start"], e["frame_end"] = int(e["frame_start"]), int(e["frame_end"])
        e["details"] = e.get("details") or {}
    events.sort(key=lambda e: (e["frame_start"], e["frame_end"], e.get("id", "")))
    annotate_outcomes(events)
    return events, meta


def annotate_outcomes(events):
    """Decide success / failure for every event.

    Direct outcomes come from the detector's own fields (pass outcome, shot outcome...). Carries and
    holds have none, so they are judged by what happens next: kept (the same team acts next) or lost
    (the next relevant event is a tackle / interception / possession change by the other team, or the
    ball goes out).
    """
    nxt_types = {"pass", "carry", "hold", "tackle", "interception", "possession_change", "ball_out_of_play", "shot"}
    starts = [e["frame_start"] for e in events]
    for idx, e in enumerate(events):
        t, d = e["type"], e["details"]
        status, text = "neutral", ""
        if t == "pass":
            ok = d.get("outcome") == "complete"
            status, text = ("success" if ok else "fail"), d.get("outcome") or "?"
        elif t == "shot":
            if d.get("outcome") == "goal":
                status, text = "success", "goal"
            else:
                status, text = "fail", "no goal" + (", reached line" if d.get("reached_goal_line") else "")
        elif t == "goal":
            status, text = "success", "goal"
        elif t in ("tackle", "interception"):
            status, text = "success", "ball won"
        elif t == "possession_change":
            status, text = "success", "{} won it ({})".format(d.get("to_team"), d.get("kind"))
        elif t == "duel":
            w = d.get("winner_team")
            text = "won by {}".format(w) if w else "unresolved"
        elif t in ("carry", "hold"):
            status, text = "success", "kept"
            j = bisect.bisect_left(starts, e["frame_end"] - 2)
            while j < len(events) and events[j]["frame_start"] <= e["frame_end"] + 5 * FPS:
                o = events[j]
                if o is not e and o["type"] in nxt_types and o["frame_start"] >= e["frame_start"]:
                    if o["type"] == "ball_out_of_play":
                        status, text = "fail", "ball out"
                    elif o["type"] in ("tackle", "interception", "possession_change") and o.get("team") \
                            and o.get("team") != e.get("team"):
                        status, text = "fail", "lost ball"
                    elif o.get("team") == e.get("team"):
                        status, text = "success", "kept"
                    else:
                        j += 1
                        continue
                    break
                j += 1
            if t == "carry":
                text = "{} m{}{}, {}".format(d.get("distance_m"), " dribble" if d.get("is_dribble") else "",
                                             " pressured" if d.get("under_pressure") else "", text)
        elif t == "ball_out_of_play":
            status, text = "fail", "out " + (d.get("exit_side") or "").replace("_", " ")
        elif t == "foul_candidate":
            status, text = "fail", "by {}".format(d.get("offending_team"))
        elif t == "restart":
            text = "{} for {}".format(e.get("subtype") or "?", d.get("restart_team") or "?")
        elif t == "dead_ball":
            text = d.get("reason") or ""
        elif t == "tracking_gap":
            text = "{} frames".format(d.get("frames"))
        e["_status"], e["_outcome"] = status, text


def describe(e):
    """One line: id, TYPE/subtype, team#slot and the most useful detail."""
    t, d = e["type"], e["details"]
    who = e.get("team") or "-"
    if e.get("player_slot"):
        who += "#{}".format(e["player_slot"])
    name = t.replace("_", " ").upper() + ("/" + e["subtype"] if e.get("subtype") else "")
    extra = ""
    if t == "pass":
        extra = "-> {}#{} {}m {}".format(d.get("recipient_team") or "-", d.get("recipient_slot") or "-",
                                         d.get("length_m"), d.get("direction") or "")
    elif t == "duel":
        extra = "H#{} vs A#{}".format(d.get("home_slot"), d.get("away_slot"))
    elif t == "shot":
        extra = "{}m out, {}m/s".format(d.get("distance_to_goal_m"), d.get("release_ball_speed_ms"))
    elif t == "tackle":
        extra = "from {}#{}".format(d.get("dispossessed_team"), d.get("dispossessed_slot"))
    elif t == "interception":
        extra = "from {}#{}".format(d.get("passer_team"), d.get("passer_slot"))
    elif t == "possession_change":
        extra = "{} -> {}".format(d.get("from_team"), d.get("to_team"))
    return "{} {} {} {}".format(e.get("id", "?"), name, who, extra).strip()


def event_color(e):
    if e["type"] == "pass" and e["details"].get("outcome") == "intercepted":
        return INTERCEPTED_BGR
    return EVENT_STYLE.get(e["type"], ((200, 200, 200), "?"))[0]


def filter_events(events, types, min_conf):
    keep = set(t.strip() for t in types.split(",") if t.strip()) if types else None
    if keep:
        unknown = keep - set(EVENT_STYLE)
        if unknown:
            print("note: unknown event types ignored: {}".format(", ".join(sorted(unknown))))
    return [e for e in events if (keep is None or e["type"] in keep)
            and float(e.get("confidence", 1.0)) >= min_conf]


def in_window(events, f0, n, linger=0):
    """Events whose (lingering) span overlaps video frames [f0, f0 + n)."""
    return [e for e in events if e["frame_end"] + linger >= f0 and e["frame_start"] < f0 + n]


def event_possession(events, f0, n):
    """Per-frame label implied by the events: 'H' / 'A' (team in possession), 'D' (dead ball), 'X'.

    The team is set at the start of each carry, hold, pass, tackle, interception, possession change
    and restart, and carried forward until the next one; dead_ball spans become 'D'; frames outside
    both halves are 'X'. This is the same quantity as the challenge's possession label, so it can be
    scored against the public GT directly.
    """
    lab = np.full(n, "X", dtype="<U1")
    frames = np.arange(f0, f0 + n)
    anchors = [(e["frame_start"], e["team"]) for e in events
               if e["type"] in POSSESSION_ANCHORS and e.get("team") in ("H", "A")]
    if anchors:
        af = np.array([a[0] for a in anchors])
        at = np.array([a[1] for a in anchors])
        idx = np.searchsorted(af, frames, side="right") - 1
        lab[idx >= 0] = at[idx[idx >= 0]]
    for e in events:
        if e["type"] == "dead_ball":
            a, b = max(e["frame_start"], f0), min(e["frame_end"], f0 + n - 1)
            if a <= b:
                lab[a - f0:b - f0 + 1] = "D"
    inside = np.zeros(n, dtype=bool)
    for _, first, last in half_video_blocks(GAME):
        inside |= (frames >= first) & (frames <= last)
    lab[~inside] = "X"
    return lab


def _runs(labels):
    """Number of maximal runs of 'H' or 'A' (possessions), dead ball and 'X' excluded."""
    count, prev = 0, None
    for v in labels:
        if v in ("H", "A") and v != prev:
            count += 1
        prev = v if v in ("H", "A") else None if v == "D" else prev
    return count


def compare_to_gt(ours, gt):
    """Frame-exact agreement of the event-implied possession with the public GT, or None."""
    if gt is None:
        return None
    graded = np.isin(gt, ["H", "A", "D"])
    if not graded.any():
        return None
    o, g = ours[graded], gt[graded]
    f1 = {}
    for c in "HAD":
        tp = int(np.sum((o == c) & (g == c)))
        fp = int(np.sum((o == c) & (g != c)))
        fn = int(np.sum((o != c) & (g == c)))
        p = tp / float(tp + fp) if tp + fp else 0.0
        r = tp / float(tp + fn) if tp + fn else 0.0
        f1[c] = round(2 * p * r / (p + r), 4) if p + r else 0.0
    confusion = {gl: {ol: int(np.sum((g == gl) & (o == ol))) for ol in "HADX"} for gl in "HAD"}
    return {"graded_frames": int(graded.sum()),
            "coverage": round(float(np.mean(o != "X")), 4),
            "accuracy": round(float(np.mean(o == g)), 4),
            "macro_f1": round(float(np.mean(list(f1.values()))), 4),
            "f1": f1, "confusion_gt_rows_ours_cols": confusion,
            "possessions_ours": _runs(o), "possessions_gt": _runs(g)}


# -- progress report ---------------------------------------------------------------------------

def _mean(values):
    vals = [float(v) for v in values if v is not None]
    return round(float(np.mean(vals)), 3) if vals else None


def summarize(window_events, all_ids, f0, n, ours, gt, source):
    """Counts, outcome rates, confidence, visibility, consistency checks, GT agreement for one window."""
    ev = [e for e in window_events if f0 <= e["frame_start"] < f0 + n]
    counts = Counter(e["type"] for e in ev)
    teams = {}
    for t in ("H", "A"):
        mine = [e for e in ev if e.get("team") == t]
        by = lambda k: [e for e in mine if e["type"] == k]  # noqa: E731
        passes = by("pass")
        complete = [p for p in passes if p["details"].get("outcome") == "complete"]
        carries = by("carry")
        shots = by("shot")
        duels = [e for e in ev if e["type"] == "duel" and e["details"].get("winner_team") == t]
        teams[t] = {
            "passes": len(passes), "passes_complete": len(complete),
            "pass_completion_pct": round(100.0 * len(complete) / len(passes), 1) if passes else None,
            "pass_length_mean_m": _mean(p["details"].get("length_m") for p in passes),
            "progressive_passes_10m": sum(1 for p in complete if (p["details"].get("progressive_m") or 0) >= 10),
            "forward_pass_share": round(sum(1 for p in passes if p["details"].get("direction") == "forward")
                                        / float(len(passes)), 3) if passes else None,
            "crosses": sum(1 for p in passes if p["details"].get("is_cross")),
            "carries": len(carries), "carry_distance_m": round(sum(c["details"].get("distance_m") or 0 for c in carries), 1),
            "carries_lost": sum(1 for c in carries if c["_status"] == "fail"),
            "dribbles": sum(1 for c in carries if c["details"].get("is_dribble")),
            "holds": len(by("hold")),
            "shots": len(shots), "shots_reaching_goal_line": sum(1 for s in shots if s["details"].get("reached_goal_line")),
            "goals": len(by("goal")), "tackles": len(by("tackle")), "interceptions": len(by("interception")),
            "possessions_won": len(by("possession_change")), "duels_won": len(duels),
            "foul_candidates_committed": len(by("foul_candidate")),
            "ball_out_last_touch": len(by("ball_out_of_play")),
            "restarts_taken": len(by("restart")),
        }
    status = Counter((e["type"], e["_status"]) for e in ev)
    outcome_rates = {}
    for t in sorted(counts):
        s, f = status[(t, "success")], status[(t, "fail")]
        if s + f:
            outcome_rates[t] = {"success": s, "fail": f, "success_pct": round(100.0 * s / (s + f), 1)}
    conf = {t: _mean(e.get("confidence") for e in ev if e["type"] == t) for t in sorted(counts)}

    # Consistency checks on the detector output: each one is a likely bug or a weak spot.
    def rel(e):
        return e - f0

    dead = ours == "D"
    checks = {
        "frame_end_before_start": [e for e in ev if e["frame_end"] < e["frame_start"]],
        "pass_to_same_slot": [e for e in ev if e["type"] == "pass" and e["details"].get("recipient_slot") == e.get("player_slot")],
        "complete_pass_to_other_team": [e for e in ev if e["type"] == "pass" and e["details"].get("outcome") == "complete"
                                        and e["details"].get("recipient_team") not in (None, e.get("team"))],
        "intercepted_pass_to_same_team": [e for e in ev if e["type"] == "pass" and e["details"].get("outcome") == "intercepted"
                                          and e["details"].get("recipient_team") == e.get("team")],
        "possession_change_same_team": [e for e in ev if e["type"] == "possession_change"
                                        and e["details"].get("from_team") == e["details"].get("to_team")],
        "dangling_related_ids": [e for e in ev if any(r not in all_ids for r in e.get("related_event_ids", []))],
        "play_event_inside_dead_ball": [e for e in ev if e["type"] in PLAY_TYPES and 0 <= rel(e["frame_start"]) < n
                                        and dead[rel(e["frame_start"])]],
        "long_pass_without_flight(>20m,<=1fr)": [e for e in ev if e["type"] == "pass" and (e["details"].get("length_m") or 0) > 20
                                                 and (e["details"].get("flight_frames") or 0) <= 1],
        "ball_speed_over_40ms": [e for e in ev if (e["details"].get("max_ball_speed_ms") or 0) > 40
                                 or (e["details"].get("release_ball_speed_ms") or 0) > 40],
        "ball_mostly_interpolated(seen<0.5)": [e for e in ev if e["details"].get("ball_seen_share") is not None
                                               and e["details"]["ball_seen_share"] < 0.5],
        "no_start_location": [e for e in ev if e["type"] in PLAY_TYPES and not e.get("start_location")],
        "low_confidence(<0.5)": [e for e in ev if float(e.get("confidence", 1)) < 0.5],
    }
    spells = sorted((e for e in ev if e["type"] in ("carry", "hold")), key=lambda e: e["frame_start"])
    overlap = []
    for a, b in zip(spells, spells[1:]):
        if b["frame_start"] <= a["frame_end"] and a.get("team") != b.get("team"):
            overlap.append(b)
    checks["overlapping_control_of_both_teams"] = overlap

    live_frames = int(np.sum(np.isin(ours, ["H", "A"])))
    report = {
        "events_file": source, "window": {"frame_start": f0, "frame_end": f0 + n - 1,
                                          "time_start": fmt_time(f0), "time_end": fmt_time(f0 + n - 1)},
        "n_events": len(ev), "counts": dict(sorted(counts.items())),
        "restart_types": dict(Counter(e.get("subtype") for e in ev if e["type"] == "restart")),
        "events_per_live_minute": round(len([e for e in ev if e["type"] in PLAY_TYPES]) / (live_frames / FPS / 60.0), 2)
        if live_frames else None,
        "live_share": round(live_frames / float(n), 3), "dead_share": round(float(np.mean(dead)), 3),
        "teams": teams, "outcome_rates": outcome_rates, "mean_confidence": conf,
        "ball_seen_share_mean": {t: _mean(e["details"].get("ball_seen_share") for e in ev if e["type"] == t)
                                 for t in ("pass", "carry", "hold")},
        "checks": {k: {"count": len(v), "examples": [e.get("id") for e in v[:5]]} for k, v in checks.items()},
        "possession_vs_public_gt": compare_to_gt(ours, gt),
    }
    return report


def print_report(r):
    w = r["window"]
    print("\n=== event detection report: {} .. {} (frames {}..{}) ===".format(
        w["time_start"], w["time_end"], w["frame_start"], w["frame_end"]))
    print("events: {}   live {:.0%} / dead {:.0%}   play events per live minute: {}".format(
        r["n_events"], r["live_share"], r["dead_share"], r["events_per_live_minute"]))
    print("counts: " + ", ".join("{} {}".format(v, k) for k, v in r["counts"].items()))
    if r["restart_types"]:
        print("restarts: " + ", ".join("{} {}".format(v, k) for k, v in r["restart_types"].items()))
    keys = ["passes", "passes_complete", "pass_completion_pct", "pass_length_mean_m", "progressive_passes_10m",
            "forward_pass_share", "carries", "carries_lost", "dribbles", "shots", "goals", "tackles",
            "interceptions", "possessions_won", "duels_won", "foul_candidates_committed", "ball_out_last_touch"]
    print("\n{:<28}{:>10}{:>10}".format("per team", "H", "A"))
    for k in keys:
        print("{:<28}{:>10}{:>10}".format(k, str(r["teams"]["H"][k]), str(r["teams"]["A"][k])))
    if r["outcome_rates"]:
        print("\noutcomes (success / fail):  " + "   ".join(
            "{} {}/{} ({}%)".format(k, v["success"], v["fail"], v["success_pct"]) for k, v in r["outcome_rates"].items()))
    print("mean confidence: " + ", ".join("{} {}".format(k, v) for k, v in r["mean_confidence"].items()))
    print("ball seen share: " + ", ".join("{} {}".format(k, v) for k, v in r["ball_seen_share_mean"].items()))
    print("\nconsistency checks (count, first ids):")
    for k, v in r["checks"].items():
        flag = "  " if v["count"] == 0 else "! "
        print("  {}{:<42}{:>5}  {}".format(flag, k, v["count"], " ".join(i or "?" for i in v["examples"])))
    g = r["possession_vs_public_gt"]
    if g:
        print("\nevent-implied possession vs public GT ({} graded frames, coverage {:.0%}):".format(
            g["graded_frames"], g["coverage"]))
        print("  macro F1 {:.4f}   accuracy {:.4f}   F1 H {} / A {} / D {}   possessions ours {} vs GT {}".format(
            g["macro_f1"], g["accuracy"], g["f1"]["H"], g["f1"]["A"], g["f1"]["D"],
            g["possessions_ours"], g["possessions_gt"]))
        print("  confusion (rows GT, cols ours H A D X): " + "  ".join(
            "{}: {}".format(gl, " ".join(str(v) for v in row.values())) for gl, row in g["confusion_gt_rows_ours_cols"].items()))
        print("  (for checking what the events mean only - never tune on this game, RULES.md)")
    else:
        print("\n(no public GT in this window: possession agreement not computed)")


def write_sidecars(base, events, report):
    csv_path, json_path = base + "_events.csv", base + "_report.json"
    cols = ["id", "type", "subtype", "period", "frame_start", "frame_end", "time_start", "duration_s", "team",
            "player_slot", "status", "outcome", "confidence", "start_x_m", "start_y_m", "end_x_m", "end_y_m",
            "related_event_ids", "description"]
    with open(csv_path, "w", newline="") as handle:
        w = csv.writer(handle)
        w.writerow(cols)
        for e in events:
            s, en = e.get("start_location") or {}, e.get("end_location") or {}
            w.writerow([e.get("id"), e["type"], e.get("subtype") or "", e.get("period"), e["frame_start"],
                        e["frame_end"], e.get("time_start"), e.get("duration_s"), e.get("team") or "",
                        e.get("player_slot") or "", e["_status"], e["_outcome"], e.get("confidence"),
                        s.get("x_m"), s.get("y_m"), en.get("x_m"), en.get("y_m"),
                        " ".join(e.get("related_event_ids", [])), describe(e)])
    with open(json_path, "w") as handle:
        json.dump(report, handle, indent=1)
    print("wrote {}\nwrote {}".format(csv_path, json_path))


# -- lazy video --------------------------------------------------------------------------------

def find_ffmpeg():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


class FfmpegFrames:
    """Stream n frames starting at start_frame, scaled to w x h, from an ffmpeg pipe.

    -ss before -i seeks through the mp4 index (no full read) and, because the video is
    re-encoded to rawvideo, ffmpeg decodes from the previous keyframe and discards up to the
    requested timestamp, so the seek is frame-accurate. Half a frame is subtracted so rounding
    in the timestamp can never skip the first frame. Decoding runs in the ffmpeg process,
    in parallel with the drawing done here.
    """

    def __init__(self, exe, path, start_frame, n, w, h, fps):
        self.w, self.h, self.size = w, h, w * h * 3
        ss = max(0.0, (start_frame - 0.5) / float(fps))
        cmd = [exe, "-v", "error", "-nostdin", "-ss", "{:.4f}".format(ss), "-i", path,
               "-map", "0:v:0", "-an", "-sn", "-frames:v", str(n),
               "-vf", "scale={}:{}:flags=area".format(w, h),
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
        self._err = tempfile.TemporaryFile()
        self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=self._err,
                                      bufsize=self.size * 4)

    def read(self):
        buf = bytearray(self.size)
        view, got = memoryview(buf), 0
        while got < self.size:
            k = self._proc.stdout.readinto(view[got:])
            if not k:
                return None
            got += k
        return np.frombuffer(buf, dtype=np.uint8).reshape(self.h, self.w, 3)   # writable

    def error(self):
        self._err.seek(0)
        return self._err.read().decode("utf-8", "replace").strip()

    def close(self):
        if self._proc.poll() is None:
            self._proc.kill()
        self._proc.wait()
        self._proc.stdout.close()
        self._err.close()


class OpenCVFrames:
    """Fallback: seek with OpenCV. Cheap to open, but seeking can be a few frames off."""

    def __init__(self, path, start_frame, w, h, exact):
        self.w, self.h = w, h
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            sys.exit("could not open {}".format(path))
        if exact:
            for _ in range(start_frame):
                if not self.cap.grab():
                    break
        elif start_frame:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            if int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) != start_frame:
                print("WARNING: seek landed elsewhere; use --exact-seek (slow) or install ffmpeg")

    def read(self):
        ok, img = self.cap.read()
        if not ok:
            return None
        return cv2.resize(img, (self.w, self.h), interpolation=cv2.INTER_AREA)

    def error(self):
        return ""

    def close(self):
        self.cap.release()


def open_video(args, start_frame, n):
    """(reader, source_width) -- opening only reads the mp4 header, never the 5 GB of frames."""
    path = require(["video"])[0]
    probe = cv2.VideoCapture(path)
    if not probe.isOpened():
        sys.exit("could not open {}".format(path))
    src_w = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
    total = int(probe.get(cv2.CAP_PROP_FRAME_COUNT))
    probe.release()
    if total and total != N_FRAMES:
        print("WARNING: video reports {} frames, expected {} - decode may differ from the "
              "reference".format(total, N_FRAMES))

    start_frame = max(0, start_frame + args.video_offset)
    exe = find_ffmpeg() if args.backend in ("auto", "ffmpeg") else None
    if args.backend == "ffmpeg" and not exe:
        sys.exit("--backend ffmpeg requested but no ffmpeg found (install it, or: pip install imageio-ffmpeg)")
    if exe:
        print("video backend: ffmpeg ({})".format(exe))
        return FfmpegFrames(exe, path, start_frame, n, PANEL_W, PANEL_H, FPS), src_w
    print("video backend: OpenCV seek (no ffmpeg found; frame accuracy not guaranteed)")
    return OpenCVFrames(path, start_frame, PANEL_W, PANEL_H, args.exact_seek), src_w


# -- pitch panel -------------------------------------------------------------------------------

class PitchPanel:
    """Top-down pitch drawn with OpenCV (matplotlib is ~10x too slow for 7,500 frames)."""

    def __init__(self, w, h, flip_y=True, margin=30):
        self.flip_y = flip_y
        self.scale = min((w - 2 * margin) / PITCH_L, (h - 2 * margin) / PITCH_W)
        self.ox = (w - PITCH_L * self.scale) / 2.0
        self.oy = (h - PITCH_W * self.scale) / 2.0
        self.bg = self._background(w, h)

    def metres(self, xm, ym):
        """Pitch metres -> pixel (ints)."""
        y = (PITCH_W - ym) if self.flip_y else ym
        return int(round(self.ox + xm * self.scale)), int(round(self.oy + y * self.scale))

    def norm(self, x, y):
        """Normalised 0-1 -> pixel (float arrays)."""
        yy = (1.0 - y) if self.flip_y else y
        return self.ox + x * PITCH_L * self.scale, self.oy + yy * PITCH_W * self.scale

    def _background(self, w, h):
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:] = GRASS_BGR
        pt = self.metres
        rect = lambda a, b: cv2.rectangle(img, pt(*a), pt(*b), LINE_BGR, 1, cv2.LINE_AA)  # noqa: E731
        rect((0, 0), (PITCH_L, PITCH_W))
        cv2.line(img, pt(PITCH_L / 2, 0), pt(PITCH_L / 2, PITCH_W), LINE_BGR, 1, cv2.LINE_AA)
        cv2.circle(img, pt(PITCH_L / 2, PITCH_W / 2), int(9.15 * self.scale), LINE_BGR, 1, cv2.LINE_AA)
        cv2.circle(img, pt(PITCH_L / 2, PITCH_W / 2), 2, LINE_BGR, -1, cv2.LINE_AA)
        mid = PITCH_W / 2
        for left in (True, False):
            sx = (lambda d: d) if left else (lambda d: PITCH_L - d)
            rect((sx(0), mid - 20.15), (sx(16.5), mid + 20.15))      # penalty area
            rect((sx(0), mid - 9.16), (sx(5.5), mid + 9.16))         # six-yard box
            cv2.circle(img, pt(sx(11), mid), 2, LINE_BGR, -1, cv2.LINE_AA)  # penalty spot
            rect((sx(0), mid - 3.66), (sx(-1.5), mid + 3.66))        # goal
        return img

    def render(self, players, ball, team, trail, home_name, away_name, show_ids):
        """players (S, 3), ball (2,), trail (k, 2) of earlier ball positions.

        Returns (image, n_home, n_away, nearest_team) with nearest_team in {0, 1, -1}.
        """
        img = self.bg.copy()
        ok = ~np.isnan(players[:, 0]) & ~np.isnan(players[:, 1])
        px, py = self.norm(players[:, 0], players[:, 1])

        # Ball trail: connect consecutive valid points only, older = darker.
        for k in range(1, len(trail)):
            a, b = trail[k - 1], trail[k]
            if np.isnan(a).any() or np.isnan(b).any():
                continue
            shade = int(80 + 175 * k / len(trail))
            cv2.line(img, tuple(int(v) for v in np.round(self.norm(*a))),
                     tuple(int(v) for v in np.round(self.norm(*b))),
                     (shade, shade, 255), 2, cv2.LINE_AA)

        for slot in np.flatnonzero(ok):
            centre = (int(round(px[slot])), int(round(py[slot])))
            cv2.circle(img, centre, 7, HOME_BGR if team[slot] == 0 else AWAY_BGR, -1, cv2.LINE_AA)
            cv2.circle(img, centre, 7, (0, 0, 0), 1, cv2.LINE_AA)
            if show_ids:
                put(img, str(slot + 1), (centre[0] + 8, centre[1] - 6), 0.4)

        nearest = -1
        if not np.isnan(ball).any():
            bx, by = self.norm(ball[0], ball[1])
            if ok.any():
                d = ((players[ok, 0] - ball[0]) * PITCH_L) ** 2 + ((players[ok, 1] - ball[1]) * PITCH_W) ** 2
                j = np.flatnonzero(ok)[int(np.argmin(d))]
                nearest = int(team[j])
                cv2.line(img, (int(round(bx)), int(round(by))), (int(round(px[j])), int(round(py[j]))),
                         (0, 255, 255), 1, cv2.LINE_AA)
            cv2.circle(img, (int(round(bx)), int(round(by))), 6, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(img, (int(round(bx)), int(round(by))), 6, (0, 0, 0), 2, cv2.LINE_AA)
        else:
            put(img, "BALL NOT TRACKED", (int(self.ox) + 10, 24), 0.6, (80, 80, 255), 2)

        put(img, "H {} (red)".format(home_name), (12, PANEL_H - 34), 0.5, (200, 200, 255))
        put(img, "A {} (blue)".format(away_name), (12, PANEL_H - 14), 0.5, (255, 220, 200))
        n_home = int(np.sum(ok & (team == 0)))
        n_away = int(np.sum(ok & (team == 1)))
        return img, n_home, n_away, nearest


# -- event drawing -------------------------------------------------------------------------------

def _loc_px(panel, loc):
    if not loc or loc.get("x_m") is None or loc.get("y_m") is None:
        return None
    return panel.metres(loc["x_m"], loc["y_m"])


def _slot_px(panel, players_i, slot):
    """Pixel of a 1-based track slot on the pitch panel, or None if not tracked on this frame."""
    if not slot or slot < 1 or slot > len(players_i):
        return None
    x, y = players_i[slot - 1, 0], players_i[slot - 1, 1]
    if np.isnan(x) or np.isnan(y):
        return None
    u, v = panel.norm(x, y)
    return int(round(u)), int(round(v))


def _ring(img, p, color, r=12, thick=2):
    if p is not None:
        cv2.circle(img, p, r, color, thick, cv2.LINE_AA)


def _dashed(img, a, b, color, thick=2, dash=8):
    length = float(np.hypot(b[0] - a[0], b[1] - a[1]))
    steps = max(1, int(length // dash))
    for k in range(0, steps, 2):
        p = (int(a[0] + (b[0] - a[0]) * k / steps), int(a[1] + (b[1] - a[1]) * k / steps))
        q = (int(a[0] + (b[0] - a[0]) * min(steps, k + 1) / steps), int(a[1] + (b[1] - a[1]) * min(steps, k + 1) / steps))
        cv2.line(img, p, q, color, thick, cv2.LINE_AA)


def _cross(img, p, color, s=8, thick=2):
    cv2.line(img, (p[0] - s, p[1] - s), (p[0] + s, p[1] + s), color, thick, cv2.LINE_AA)
    cv2.line(img, (p[0] - s, p[1] + s), (p[0] + s, p[1] - s), color, thick, cv2.LINE_AA)


def draw_event_pitch(img, panel, e, players_i, label=True):
    """Draw one event on the pitch panel where it happened."""
    t, d = e["type"], e["details"]
    col = event_color(e)
    s, en = _loc_px(panel, e.get("start_location")), _loc_px(panel, e.get("end_location"))
    team_col = HOME_BGR if e.get("team") == "H" else AWAY_BGR if e.get("team") == "A" else (220, 220, 220)
    text = None
    if t == "pass" and s and en:
        cv2.arrowedLine(img, s, en, col, 2, cv2.LINE_AA, tipLength=0.08)
        _ring(img, _slot_px(panel, players_i, d.get("recipient_slot")), col, 11, 2)
        text = "PASS {}m".format(d.get("length_m"))
    elif t == "carry" and s and en:
        _dashed(img, s, en, col, 2)
        cv2.circle(img, s, 4, col, -1, cv2.LINE_AA)
        text = "CARRY"
    elif t == "hold" and s:
        cv2.circle(img, s, 15, col, 2, cv2.LINE_AA)
        text = "HOLD"
    elif t == "shot" and s:
        gx = PITCH_L if d.get("target_goal") == "x_max" else 0.0
        gy = d.get("projected_goal_line_y_m")
        gy = PITCH_W / 2 if gy is None else float(np.clip(gy, -5, PITCH_W + 5))
        g = panel.metres(gx, gy)
        cv2.arrowedLine(img, s, g, (0, 200, 255) if d.get("outcome") == "goal" else col, 3, cv2.LINE_AA, tipLength=0.06)
        text = "SHOT"
    elif t == "goal":
        put(img, "GOAL {}".format(e.get("team") or ""), (PANEL_W // 2 - 80, PANEL_H // 2), 1.6, col, 4)
    elif t == "duel" and s:
        cv2.circle(img, s, 20, col, 2, cv2.LINE_AA)
        _ring(img, _slot_px(panel, players_i, d.get("home_slot")), col, 10, 2)
        _ring(img, _slot_px(panel, players_i, d.get("away_slot")), col, 10, 2)
        text = "DUEL"
    elif t == "tackle" and s:
        _cross(img, s, col, 10, 3)
        text = "TACKLE"
    elif t == "interception" and s:
        pts = np.array([(s[0], s[1] - 11), (s[0] + 11, s[1]), (s[0], s[1] + 11), (s[0] - 11, s[1])], np.int32)
        cv2.polylines(img, [pts], True, col, 2, cv2.LINE_AA)
        text = "INTERCEPT"
    elif t == "possession_change" and s:
        cv2.circle(img, s, 18, team_col, 3, cv2.LINE_AA)
        cv2.circle(img, s, 23, col, 1, cv2.LINE_AA)
        text = "-> {}".format(d.get("to_team"))
    elif t == "ball_out_of_play" and s:
        _cross(img, s, col, 9, 2)
        text = "OUT"
    elif t == "restart" and s:
        cv2.rectangle(img, (s[0] - 9, s[1] - 9), (s[0] + 9, s[1] + 9), col, 2, cv2.LINE_AA)
        text = (e.get("subtype") or "restart").upper()
    elif t == "foul_candidate" and s:
        pts = np.array([(s[0], s[1] - 13), (s[0] + 12, s[1] + 9), (s[0] - 12, s[1] + 9)], np.int32)
        cv2.polylines(img, [pts], True, col, 2, cv2.LINE_AA)
        text = "FOUL?"
    elif t == "tracking_gap" and s and en:
        _dashed(img, s, en, col, 1, 5)
        text = "BALL GAP"
    if t in PLAY_TYPES or t in ("restart", "foul_candidate"):
        _ring(img, _slot_px(panel, players_i, e.get("player_slot")), (0, 255, 255), 12, 2)
    if label and text and s:
        put(img, text, (s[0] + 12, s[1] + 18), 0.42, col)


def draw_event_video(img, e, H_i, scale, players_i):
    """Project the actor ring and the pass / carry / shot line onto the video frame."""
    from starter.load_homography import project

    def px(x_norm, y_norm):
        u, v = project(H_i, x_norm, y_norm)
        if not (np.isfinite(u) and np.isfinite(v)):
            return None
        return int(u * scale), int(v * scale)

    def loc(l):
        if not l or l.get("x_m") is None:
            return None
        return px(l["x_m"] / PITCH_L, l["y_m"] / PITCH_W)

    col = event_color(e)
    slot = e.get("player_slot")
    if slot and 1 <= slot <= len(players_i) and not np.isnan(players_i[slot - 1, :2]).any():
        p = px(players_i[slot - 1, 0], players_i[slot - 1, 1])
        if p:
            cv2.ellipse(img, p, (14, 6), 0, 0, 360, (0, 255, 255), 2, cv2.LINE_AA)
    if e["type"] in ("pass", "carry", "tracking_gap", "shot"):
        a, b = loc(e.get("start_location")), loc(e.get("end_location"))
        if a and b and max(abs(a[0]), abs(a[1]), abs(b[0]), abs(b[1])) < 10000:
            if e["type"] == "carry":
                _dashed(img, a, b, col, 2)
            else:
                cv2.arrowedLine(img, a, b, col, 2, cv2.LINE_AA, tipLength=0.05)


def build_timeline(f0, n, ours, gt, shown):
    """Static timeline image (the cursor is added per frame) and the frame -> x mapping."""
    W, L, R = 2 * PANEL_W, 70, 12
    tl = np.zeros((TL_H, W, 3), dtype=np.uint8)
    tl[:] = (25, 25, 25)

    def x_of(f):
        return int(L + (f - f0) / float(max(1, n - 1)) * (W - L - R))

    cols = np.arange(L, W - R)
    idx = np.clip(((cols - L) / float(W - L - R - 1) * (n - 1)).astype(int), 0, n - 1)

    def band(labels, y0, y1):
        for c, lab in LABEL_BGR.items():
            sel = cols[labels[idx] == c]
            tl[y0:y1, sel] = lab

    put(tl, "EVENTS", (6, 16), 0.38)
    band(ours, 5, 19)
    put(tl, "GT", (6, 34), 0.38)
    if gt is not None:
        band(np.where(np.isin(gt, ["H", "A", "D"]), gt, "X"), 22, 34)
    put(tl, "TICKS", (6, 56), 0.38)
    for e in shown:
        if e["type"] in ("dead_ball",):
            continue
        x = x_of(min(max(e["frame_start"], f0), f0 + n - 1))
        cv2.line(tl, (x, 38), (x, 56), event_color(e), 1)
        if e["_status"] != "neutral":
            tl[59:64, max(0, x - 1):x + 2] = STATUS_BGR[e["_status"]]
    return tl, x_of


# -- main --------------------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default=None, help="start, video time: hh:mm:ss, mm:ss or seconds (default 07:27)")
    ap.add_argument("--start-frame", type=int, help="start as a video frame (overrides --start)")
    ap.add_argument("--duration", type=float, default=None, help="seconds to render (default 300, or the event +- --pad)")
    ap.add_argument("--out", help="output mp4 (default viz_<startframe>.mp4)")
    ap.add_argument("--trail", type=int, default=50, help="ball trail length in frames (0 = off)")
    ap.add_argument("--ids", action="store_true", help="label each point with its track slot (1-46)")
    ap.add_argument("--no-overlay", action="store_true", help="do not project tracking onto the video")
    ap.add_argument("--no-video", action="store_true", help="skip the video panel (no mp4 decode)")
    ap.add_argument("--no-tags", action="store_true", help="do not show Smart Tagging phases")
    ap.add_argument("--gt", default=DEFAULT_GT, help="ground-truth CSV to display ('' to disable)")
    ap.add_argument("--no-flip-y", dest="flip_y", action="store_false",
                    help="draw normalised y downwards instead of upwards (default matches utils.py)")
    ap.add_argument("--backend", choices=("auto", "ffmpeg", "opencv"), default="auto",
                    help="video reader: ffmpeg pipe (fast, frame-accurate seek) or OpenCV seek")
    ap.add_argument("--exact-seek", action="store_true",
                    help="OpenCV backend only: decode forward to the start frame (slow, frame-exact)")
    ap.add_argument("--video-offset", type=int, default=0,
                    help="shift the video by this many frames relative to the tracking, if the "
                         "overlay is consistently off (positive = read later video frames)")
    g = ap.add_argument_group("events (output of event_detection.py)")
    g.add_argument("--events", default=None, help="events JSON / JSONL (default reports/events/events_full.json)")
    g.add_argument("--no-events", action="store_true", help="plain tracking view, no event layer")
    g.add_argument("--types", default="", help="comma list of event types to draw, e.g. pass,shot,tackle")
    g.add_argument("--min-confidence", type=float, default=0.0, help="hide events below this confidence")
    g.add_argument("--event-id", help="centre the clip on this event id (e.g. E00412)")
    g.add_argument("--event-type", help="centre the clip on the --nth event of this type")
    g.add_argument("--nth", type=int, default=1, help="with --event-type: 1 = first, -1 = last (default 1)")
    g.add_argument("--pad", type=float, default=5.0, help="seconds shown before/after a chosen event (default 5)")
    g.add_argument("--linger", type=float, default=1.0, help="seconds an event stays (faded) after it ends")
    g.add_argument("--banner", type=float, default=1.5, help="seconds the 'NEW EVENT' banner stays up")
    g.add_argument("--hold-on-event", type=int, default=0,
                   help="repeat the frame this many times when a new event starts (freeze to inspect)")
    g.add_argument("--list-events", action="store_true", help="print the (filtered) events as a table and exit")
    g.add_argument("--report-only", action="store_true", help="write the progress report for the window, no video")
    g.add_argument("--no-report", action="store_true", help="do not write the _events.csv / _report.json sidecars")
    return ap.parse_args()


def resolve_window(args, events):
    """(f0, n, focus_event) from --event-id / --event-type or --start / --start-frame / --duration."""
    focus = None
    if (args.event_id or args.event_type) and not events:
        sys.exit("--event-id / --event-type need an events file (see --events)")
    if args.event_id:
        focus = next((e for e in events if e.get("id") == args.event_id), None)
        if focus is None:
            sys.exit("event {} not found in the events file".format(args.event_id))
    elif args.event_type:
        cands = [e for e in events if e["type"] == args.event_type]
        if not cands:
            sys.exit("no event of type '{}' (types present: {})".format(
                args.event_type, ", ".join(sorted(set(e["type"] for e in events)))))
        k = args.nth - 1 if args.nth > 0 else args.nth
        if not -len(cands) <= k < len(cands):
            sys.exit("--nth {} out of range: {} events of type {}".format(args.nth, len(cands), args.event_type))
        focus = cands[k]
    if focus is not None:
        pad = int(round(args.pad * FPS))
        f0 = max(0, focus["frame_start"] - pad)
        if args.duration is not None:
            n = int(round(args.duration * FPS))
        else:
            n = focus["frame_end"] + pad - f0 + 1
        print("focus: {}  [{} {}]  at {}".format(describe(focus), STATUS_TAG[focus["_status"]],
                                                 focus["_outcome"], fmt_time(focus["frame_start"])))
    else:
        f0 = args.start_frame if args.start_frame is not None else int(round(parse_time(args.start or "07:27") * FPS))
        n = int(round((args.duration if args.duration is not None else 300.0) * FPS))
    if not 0 <= f0 < N_FRAMES:
        sys.exit("start frame {} outside 0..{}".format(f0, N_FRAMES - 1))
    return f0, max(1, min(n, N_FRAMES - f0)), focus


def list_events(args, events, shown):
    window_given = args.start is not None or args.start_frame is not None or args.duration is not None
    rows = shown
    if window_given:
        f0, n, _ = resolve_window(args, events)
        rows = [e for e in shown if f0 <= e["frame_start"] < f0 + n]
    elif args.event_type:
        rows = [e for e in shown if e["type"] == args.event_type]
    print("{:<8}{:>9}{:>9}  {:<10}{:<20}{:<6}{:<6}{:>5}  {}".format(
        "id", "frame", "time", "status", "type", "team", "slot", "conf", "outcome / description"))
    for e in rows:
        print("{:<8}{:>9}{:>9}  {:<10}{:<20}{:<6}{:<6}{:>5}  {} | {}".format(
            e.get("id", "?"), e["frame_start"], fmt_time(e["frame_start"]), STATUS_TAG[e["_status"]],
            (e["type"] + ("/" + e["subtype"] if e.get("subtype") else ""))[:19], e.get("team") or "-",
            e.get("player_slot") or "-", e.get("confidence"), e["_outcome"], describe(e)))
    print("{} events".format(len(rows)))


def main():
    args = parse_args()

    # Events: everything is read once; `shown` is what gets drawn, `events` drives possession / report.
    events, meta = (None, {})
    if not args.no_events:
        path = args.events or (DEFAULT_EVENTS if os.path.isfile(DEFAULT_EVENTS) else None)
        if args.events is None and path is None:
            print("note: no events file at {} - run event_detection.py --full, or pass --events".format(DEFAULT_EVENTS))
        events, meta = load_events(path)
        if events is not None:
            print("events: {} loaded from {}".format(len(events), path))
    shown_all = filter_events(events, args.types, args.min_confidence) if events else []
    if args.list_events:
        if not events:
            sys.exit("--list-events needs an events file")
        list_events(args, events, shown_all)
        return

    f0, n, focus = resolve_window(args, events)
    linger = int(round(args.linger * FPS))
    out_path = args.out or "viz_{}.mp4".format(f0)
    base = os.path.splitext(out_path)[0]
    gt = load_gt_window(args.gt, f0, n)
    ours = event_possession(events, f0, n) if events else np.full(n, "X", dtype="<U1")
    shown = in_window(shown_all, f0, n, linger)

    if events:
        report = summarize(in_window(events, f0, n), set(e.get("id") for e in events), f0, n, ours, gt,
                           args.events or DEFAULT_EVENTS)
        print_report(report)
        if not args.no_report:
            write_sidecars(base, [e for e in shown if f0 <= e["frame_start"] < f0 + n], report)
    if args.report_only:
        if not events:
            sys.exit("--report-only needs an events file")
        return

    print("rendering video frames {}..{} ({} frames, {} to {}) -> {}".format(
        f0, f0 + n - 1, n, fmt_time(f0), fmt_time(f0 + n), out_path))

    meta_path, raw_path = require(["atd_meta", "atd_raw"])
    team = team_of_each_track(meta_path)
    home_name, away_name = team_names(meta_path)

    t = time.time()
    players, ball = read_tracking(raw_path, f0, n, len(team))
    print("tracking: {} frames read in {:.1f}s, ball present on {:.1f}%".format(
        n, time.time() - t, 100.0 * np.mean(~np.isnan(ball[:, 0]))))

    tags = load_tags_window(not args.no_tags, f0, n)

    H = H_valid = project = None
    if not args.no_overlay and not args.no_video:
        try:
            from starter.load_homography import project
            H, H_valid = load_homography_window(f0, n)
        except (IOError, ImportError, ValueError) as error:
            print("note: homography overlay disabled ({})".format(str(error).splitlines()[0]))
            H = None

    reader, src_w = (None, 1920)
    if not args.no_video:
        reader, src_w = open_video(args, f0, n)
    scale = PANEL_W / float(src_w)           # panel pixels per source pixel (homography is in source px)

    # Per-frame index of the events to draw, events starting on each frame, and the static timeline.
    active = [[] for _ in range(n)]
    starts = {}
    for e in shown:
        for f in range(max(e["frame_start"], f0), min(e["frame_end"] + linger, f0 + n - 1) + 1):
            active[f - f0].append(e)
        if f0 <= e["frame_start"] < f0 + n:
            starts.setdefault(e["frame_start"], []).append(e)
    started = [e for e in shown if f0 <= e["frame_start"] < f0 + n]
    start_frames = [e["frame_start"] for e in started]
    timeline, x_of = build_timeline(f0, n, ours, gt, shown) if events else (None, None)
    banner_frames = max(1, int(round(args.banner * FPS)))

    panel = PitchPanel(PANEL_W, PANEL_H, flip_y=args.flip_y)
    size = (2 * PANEL_W, PANEL_H + INFO_H + TL_H)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, size)
    if not writer.isOpened():
        if reader is not None:
            reader.close()
        sys.exit("could not open the video writer for {}".format(out_path))

    rendered, written, banners, prev_label, n_changes = 0, 0, [], None, 0
    t = time.time()
    try:
        for i in range(n):
            f = f0 + i
            canvas = np.zeros((size[1], size[0], 3), dtype=np.uint8)
            live_now = [e for e in active[i] if e["frame_start"] <= f <= e["frame_end"]]
            ghosts = [e for e in active[i] if f > e["frame_end"]]

            # Left: video (+ projected tracking and events).
            if reader is not None:
                img = reader.read()
                if img is None:
                    msg = reader.error()
                    print("video ended at frame {}{}".format(f, ("\n  " + msg) if msg else ""))
                    break
                if H is not None and i < len(H_valid) and H_valid[i]:
                    ok_p = ~np.isnan(players[i, :, 0]) & ~np.isnan(players[i, :, 1])
                    u, v = project(H[i], players[i, ok_p, 0], players[i, ok_p, 1])
                    for uu, vv, tm in zip(u, v, team[ok_p]):
                        if np.isfinite(uu) and np.isfinite(vv):
                            cv2.circle(img, (int(uu * scale), int(vv * scale)), 4,
                                       HOME_BGR if tm == 0 else AWAY_BGR, -1, cv2.LINE_AA)
                    if not np.isnan(ball[i]).any():
                        u, v = project(H[i], ball[i, 0], ball[i, 1])
                        if np.isfinite(u) and np.isfinite(v):
                            cv2.circle(img, (int(u * scale), int(v * scale)), 6, (255, 255, 255), 2, cv2.LINE_AA)
                    for e in live_now:
                        if e["type"] != "dead_ball":
                            draw_event_video(img, e, H[i], scale, players[i])
                canvas[:PANEL_H, :PANEL_W] = img
            else:
                put(canvas, "video disabled (--no-video)", (20, 40))

            # Right: pitch + events.
            trail = ball[max(0, i - args.trail):i + 1] if args.trail > 0 else ball[i:i + 1]
            pitch_img, n_home, n_away, nearest = panel.render(
                players[i], ball[i], team, trail, home_name, away_name, args.ids)
            if any(e["type"] == "dead_ball" for e in live_now):
                pitch_img = cv2.addWeighted(pitch_img, 0.65, np.full_like(pitch_img, 110), 0.35, 0)
                reason = next(e for e in live_now if e["type"] == "dead_ball")["details"].get("reason")
                put(pitch_img, "DEAD BALL ({})".format(reason), (PANEL_W - 330, PANEL_H - 14), 0.55, (230, 230, 230))
            if ghosts:
                layer = pitch_img.copy()
                for e in ghosts:
                    draw_event_pitch(layer, panel, e, players[i], label=False)
                pitch_img = cv2.addWeighted(layer, 0.45, pitch_img, 0.55, 0)
            for e in live_now:
                draw_event_pitch(pitch_img, panel, e, players[i])

            # Event-change detection: banner on every new event and every possession change.
            label = ours[i]
            new_events = starts.get(f, [])
            for e in new_events:
                banners.append((f + banner_frames, "NEW  {}   [{} {}]".format(
                    describe(e), STATUS_TAG[e["_status"]], e["_outcome"]), event_color(e), e["_status"]))
            if events and prev_label is not None and label != prev_label:
                n_changes += 1
                banners.append((f + banner_frames, "POSSESSION  {}  ->  {}".format(
                    LABEL_NAME[prev_label], LABEL_NAME[label]), LABEL_BGR[label], "neutral"))
            prev_label = label
            banners = [b for b in banners if b[0] > f][-4:]
            for k, (until, text, col, st) in enumerate(reversed(banners)):
                y = 4 + 26 * k
                fade = min(1.0, (until - f) / float(banner_frames) * 2.0)
                box = pitch_img[y:y + 24, 4:PANEL_W - 4]
                overlay = np.full_like(box, 0)
                overlay[:] = col
                pitch_img[y:y + 24, 4:PANEL_W - 4] = cv2.addWeighted(box, 1 - 0.55 * fade, overlay, 0.55 * fade, 0)
                cv2.rectangle(pitch_img, (4, y), (PANEL_W - 5, y + 23), STATUS_BGR[st], 2)
                put(pitch_img, text[:110], (12, y + 17), 0.5)
            canvas[:PANEL_H, PANEL_W:] = pitch_img
            if new_events:                    # flash a border around both panels
                cv2.rectangle(canvas, (0, 0), (2 * PANEL_W - 1, PANEL_H - 1), event_color(new_events[0]), 3)

            # Info bar.
            y0 = PANEL_H
            half = half_of_video_frame(GAME, f)
            has_ball = not np.isnan(ball[i, 0])
            put(canvas, "frame {}   t {}   half {}".format(f, fmt_time(f), half or "-"), (12, y0 + 20), 0.58)
            put(canvas, "tracked  H {}  A {}   ball {}   nearest-to-ball: {}".format(
                n_home, n_away, "tracked" if has_ball else "MISSING", {0: "H", 1: "A", -1: "-"}[nearest]),
                (12, y0 + 43), 0.52, (255, 255, 255) if has_ball else (120, 120, 255))
            if events:
                g = gt[i] if gt is not None and i < len(gt) else "X"
                state = "DEAD" if label == "D" else "LIVE" if label in ("H", "A") else "-"
                gt_txt = "GT {}".format(g) if g != "X" else "GT -"
                agree = "" if g == "X" else ("  agree" if g == label else "  DISAGREE")
                put(canvas, "state {}   possession(events) {}   {}{}   changes {}".format(
                    state, label, gt_txt, agree, n_changes), (12, y0 + 66), 0.55,
                    (255, 255, 255) if not agree else (120, 255, 120) if g == label else (110, 110, 255))
                now = sorted((e for e in live_now if e["type"] != "dead_ball"),
                             key=lambda e: NOW_PRIORITY.index(e["type"]) if e["type"] in NOW_PRIORITY else 99)
                if now:
                    e = now[0]
                    put(canvas, "NOW  {}".format(describe(e))[:88], (12, y0 + 90), 0.52, event_color(e))
                    put(canvas, "[{}] {}   conf {}".format(STATUS_TAG[e["_status"]], e["_outcome"], e.get("confidence"))[:88],
                        (12, y0 + 113), 0.52, STATUS_BGR[e["_status"]])
                else:
                    put(canvas, "NOW  no event", (12, y0 + 90), 0.52, (160, 160, 160))
                k = bisect.bisect_right(start_frames, f)
                put(canvas, "event log  ({} / {} in clip)".format(k, len(started)), (PANEL_W + 12, y0 + 43), 0.5,
                    (200, 200, 200))
                for r, e in enumerate(reversed(started[max(0, k - 3):k])):
                    put(canvas, "{} [{}] {}".format(fmt_time(e["frame_start"]), STATUS_TAG[e["_status"]],
                                                   describe(e))[:92],
                        (PANEL_W + 12, y0 + 66 + 23 * r), 0.48, STATUS_BGR[e["_status"]])
            elif gt is not None and i < len(gt):
                lab = gt[i]
                put(canvas, "public GT: {}".format(lab if lab != "X" else "X (not in public window)"),
                    (12, y0 + 66), 0.58)
            tg = active_tags(tags, f)
            if tg:
                put(canvas, " | ".join(tg)[:95], (PANEL_W + 12, y0 + 20), 0.5, (200, 255, 200))

            # Timeline.
            if timeline is not None:
                tl = timeline.copy()
                x = x_of(f)
                cv2.line(tl, (x, 0), (x, TL_H - 1), (255, 255, 255), 2)
                canvas[PANEL_H + INFO_H:, :] = tl
            else:
                put(canvas, "no events loaded - timeline off", (12, PANEL_H + INFO_H + 30), 0.5, (160, 160, 160))

            writer.write(canvas)
            written += 1
            if new_events and args.hold_on_event > 0:
                put(canvas, "PAUSED ON NEW EVENT", (PANEL_W + PANEL_W // 2 - 120, PANEL_H - 40), 0.7, (0, 255, 255), 2)
                for _ in range(args.hold_on_event):
                    writer.write(canvas)
                    written += 1
            rendered += 1
            if i % 500 == 0:
                print("  {}/{} frames  ({:.1f} fps)".format(i, n, rendered / max(1e-6, time.time() - t)))
    finally:
        writer.release()
        if reader is not None:
            reader.close()
    print("done: {} frames ({} written) -> {}".format(rendered, written, os.path.abspath(out_path)))


if __name__ == "__main__":
    main()
