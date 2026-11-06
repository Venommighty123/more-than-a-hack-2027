"""Detect football events (pass, carry, tackle, shot, restart, foul candidate, ...) from the provided
tracking + Smart Tagging, and emit them as JSON together with a JSON Schema describing them.

    python starter/event_detection.py --schema-only                  # just event_schema.json (no data needed)
    python starter/event_detection.py                                # first 5 min of play
    python starter/event_detection.py --start 11185 --seconds 120    # any window (clipped to the halves)
    python starter/event_detection.py --full --validate              # whole match, check against the schema
    python starter/event_detection.py --full --jsonl --include-gaps

Writes to reports/events/ (or --out-dir):
    event_schema.json          JSON Schema (draft 2020-12). Contains NO match data: safe to share.
    events_<lo>_<hi>.json      {"meta": ..., "events": [...]}   (events_full.json with --full)
    events_<lo>_<hi>.jsonl     one event per line, with --jsonl

Inputs used: ATD tracking (players + ball), the ATD metadata (team of each slot) and Smart Tagging
(set-piece type / team, goals). Nothing else: no ground truth, no hand labels, no video, and not
ball_flight.py. Ideas reused from the repo: the zone-of-influence control rule and the touch-spell /
transfer logic of possession_test_1.py, the ball-gap handling of ball_gap_analysis.py, and the
byte-offset seek of visualize_tracking.py.

What is solid and what is not (read before trusting an event)
-------------------------------------------------------------
  solid-ish   carry, hold, pass (complete / intercepted), tackle, interception, duel, possession
              change, ball out of play, dead ball, restart (type + team, using Smart Tagging).
  inferred    shot (ball leaves a holder fast, towards the goal mouth), goal (Smart Tagging GOALS).
  weak        foul_candidate. The data has no contact, no referee and no ball height. A "Free Kick"
              restart tag preceded by a duel/tackle is the best evidence available, and it cannot be
              told from offside or handball. Confidence is capped at 0.6 and the name says candidate.
  not possible  aerial duels, headers, clearances vs passes, ball height, offside lines.

Every threshold is an UNTUNED starting point. Calibrate them on other games, never on the public 5
minutes of this one (RULES.md). Track slots are not people: a slot can switch player, so
`player_slot` is a tracklet id. The events file contains positions derived from the match data: keep
it out of the public repository (reports/ is git-ignored) and do not upload it to any cloud AI tool.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
ROOT = PARENT if os.path.isfile(os.path.join(PARENT, "games.py")) else HERE
for _p in (ROOT, os.path.join(ROOT, "starter"), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from games import GAME, GAMES, half_video_blocks, require  # noqa: E402

SCHEMA_VERSION = "1.0.0"
FPS = GAMES[GAME]["fps"]
PITCH_L, PITCH_W = GAMES[GAME]["pitch"]
HOME, AWAY, NONE, BOTH = 0, 1, -1, 2
LETTER = {HOME: "H", AWAY: "A"}
HALF_KICKOFF = {1: 11185, 2: 113742}          # published in DATA.md / FAQ.md
BOX_DEPTH, BOX_HALF_W, GOAL_HALF_W = 16.5, 20.15, 3.66
DEFAULT_OUT = os.path.join(ROOT, "reports", "events")

# Fixed engineering constants (not exposed on the command line)
ZOI_SPEED_MAX = 8.0
ZOI_FRONT_GAIN, ZOI_SIDE_SHRINK, ZOI_REAR_SHRINK = 0.8, 0.35, 0.5
SMOOTH_HALF = 2
MAX_BALL_SPEED = 40.0
OUT_MARGIN_M = 0.5
TOUCH_GAP_FRAMES = 3
SLOT_SWITCH_M = 1.0
FLIGHT_MIN_FRAMES = 3
CHUNK = 20000
SEARCH_BLOCK = 1 << 20
CONTEST_WINNER_LOOKAHEAD = 50


# ==================================================================================================
# 1. JSON Schema
# ==================================================================================================

def _t(base, nullable):
    return [base, "null"] if nullable else base


def num(desc, nullable=False, minimum=None, maximum=None):
    s = {"type": _t("number", nullable), "description": desc}
    if minimum is not None:
        s["minimum"] = minimum
    if maximum is not None:
        s["maximum"] = maximum
    return s


def integer(desc, nullable=False, minimum=None):
    s = {"type": _t("integer", nullable), "description": desc}
    if minimum is not None:
        s["minimum"] = minimum
    return s


def boolean(desc, nullable=False):
    return {"type": _t("boolean", nullable), "description": desc}


def string(desc, enum=None, nullable=False):
    s = {"type": _t("string", nullable), "description": desc}
    if enum is not None:
        s["enum"] = list(enum) + ([None] if nullable else [])
    return s


TEAM_NN = string("Team letter: H = home (first team in the ATD metadata), A = away.", ["H", "A"], True)
SLOT_NN = integer("Track slot 1..46. A tracklet, NOT a person: slots swap players.", True, 1)
ID_NN = string("Event id of a related event (same file).", None, True)
DIRECTIONS = ["forward", "lateral", "backward"]
SET_PIECES = ["throw_in", "corner_kick", "goal_kick", "free_kick", "kick_off", "kick_off_after_goal",
              "penalty_kick", "unknown"]

EVENT_TYPES = {
    "pass": {
        "description": "Ball goes from one player's control to another player's control. "
                       "Outcome is complete (teammate) or intercepted (opponent, free flight >= 3 frames).",
        "subtypes": None,
        "details": {
            "outcome": string("complete = received by a teammate; intercepted = received by an opponent.",
                              ["complete", "intercepted"]),
            "recipient_slot": SLOT_NN,
            "recipient_team": TEAM_NN,
            "length_m": num("Straight-line ball displacement, metres.", minimum=0),
            "length_class": string("short / medium / long by length.", ["short", "medium", "long"]),
            "direction": string("Relative to the passer's attacking direction (45 degree sectors).", DIRECTIONS),
            "angle_deg": num("0 = straight at the opponent goal, +-180 = straight back.", True),
            "progressive_m": num("Metres gained towards the opponent goal (negative = backwards).", True),
            "flight_frames": integer("Frames with nobody in control between release and reception.", minimum=0),
            "max_ball_speed_ms": num("Peak smoothed ball speed during the pass.", True, 0),
            "ball_seen_share": num("Share of flight frames with a REAL ball detection (not interpolated).",
                                   False, 0, 1),
            "is_cross": boolean("Released from a wide area of the final third, ending in the opponent box."),
            "is_switch_of_play": boolean("Mostly lateral and >= switch_min_m across the pitch."),
            "into_final_third": boolean("Starts outside, ends inside the attacking third."),
            "into_box": boolean("Starts outside, ends inside the opponent penalty area."),
            "nearest_opponent_m": num("Distance from the passer to the nearest tracked opponent at release.", True, 0),
            "under_pressure": boolean("nearest_opponent_m below the pressure threshold."),
        },
        "required": ["outcome", "length_m", "length_class", "direction", "flight_frames"],
    },
    "carry": {
        "description": "One player (slot) keeps the ball inside his zone of influence while it moves >= carry_min_m.",
        "subtypes": None,
        "details": {
            "distance_m": num("Ball displacement during the spell.", minimum=0),
            "progressive_m": num("Metres gained towards the opponent goal.", True),
            "mean_speed_ms": num("distance / duration.", True, 0),
            "nearest_opponent_min_m": num("Closest tracked opponent during the carry.", True, 0),
            "under_pressure": boolean("An opponent came within the pressure distance."),
            "is_dribble": boolean("Carry with an opponent within dribble_m at some point."),
            "into_final_third": boolean("Starts outside, ends inside the attacking third."),
            "into_box": boolean("Starts outside, ends inside the opponent penalty area."),
            "ball_seen_share": num("Share of frames with a real ball detection.", False, 0, 1),
        },
        "required": ["distance_m", "under_pressure", "is_dribble"],
    },
    "hold": {
        "description": "A player keeps the ball for >= min_hold_s without moving it >= carry_min_m.",
        "subtypes": None,
        "details": {
            "nearest_opponent_min_m": num("Closest tracked opponent during the hold.", True, 0),
            "under_pressure": boolean("An opponent came within the pressure distance."),
            "ball_seen_share": num("Share of frames with a real ball detection.", False, 0, 1),
        },
        "required": ["under_pressure"],
    },
    "interception": {
        "description": "An opponent takes control of a ball in free flight from a pass.",
        "subtypes": None,
        "details": {
            "intercepted_pass_id": ID_NN,
            "passer_slot": SLOT_NN,
            "passer_team": TEAM_NN,
            "flight_frames": integer("Free-flight frames before the interception.", minimum=0),
            "distance_from_passer_m": num("Ball displacement release -> interception.", True, 0),
        },
        "required": ["flight_frames"],
    },
    "tackle": {
        "description": "Ball goes straight from one team's control to the other's (< 3 frames of free flight): "
                       "a tackle or a steal. No contact information exists, so fouls are not separated here.",
        "subtypes": None,
        "details": {
            "dispossessed_slot": SLOT_NN,
            "dispossessed_team": TEAM_NN,
            "duel_event_id": ID_NN,
        },
        "required": [],
    },
    "duel": {
        "description": "Players of both teams have the ball inside their zones of influence at the same time.",
        "subtypes": None,
        "details": {
            "home_slot": SLOT_NN,
            "away_slot": SLOT_NN,
            "winner_team": TEAM_NN,
            "min_player_distance_m": num("Closest the two nearest players got.", True, 0),
        },
        "required": [],
    },
    "possession_change": {
        "description": "Controlled possession passes to the other team (new team holds the ball >= min_change_hold_s). "
                       "A touch without control does not count.",
        "subtypes": None,
        "details": {
            "from_team": TEAM_NN,
            "to_team": TEAM_NN,
            "kind": string("How it happened.", ["tackle", "interception", "contest", "loose_ball_recovery"]),
            "trigger_event_id": ID_NN,
            "flight_frames": integer("Free-flight frames between the two controls.", minimum=0),
            "new_team_hold_s": num("How long the new team kept the ball in the first spell.", minimum=0),
        },
        "required": ["from_team", "to_team", "kind"],
    },
    "shot": {
        "description": "Ball released fast towards the goal mouth from within shot_max_dist_m. "
                       "Inferred from kinematics only; blocks, saves and misses are not distinguished.",
        "subtypes": None,
        "details": {
            "target_goal": string("x_min = goal at x = 0, x_max = goal at x = pitch length.", ["x_min", "x_max"]),
            "distance_to_goal_m": num("Release point to goal centre.", minimum=0),
            "projected_goal_line_y_m": num("Where the release velocity would cross the goal line.", True),
            "release_ball_speed_ms": num("Mean ball speed over the 6 frames after release.", minimum=0),
            "outcome": string("goal = a Smart Tagging GOALS phase for this team within 10 s.",
                              ["goal", "no_goal_detected"]),
            "reached_goal_line": boolean("Ball seen beyond the goal line in the next 3 s."),
        },
        "required": ["target_goal", "distance_to_goal_m", "release_ball_speed_ms", "outcome"],
    },
    "goal": {
        "description": "Smart Tagging GOALS phase (automatic, not verified). Frames are the tagged interval.",
        "subtypes": None,
        "details": {
            "tag_frame_start": integer("First frame of the tagged interval."),
            "tag_frame_end": integer("Last frame of the tagged interval."),
            "shot_event_id": ID_NN,
        },
        "required": ["tag_frame_start", "tag_frame_end"],
    },
    "ball_out_of_play": {
        "description": "Ball centre crosses a pitch line.",
        "subtypes": None,
        "details": {
            "exit_side": string("Which line was crossed (axes of the normalised ATD coordinates).",
                                ["touchline_y_min", "touchline_y_max", "goal_line_x_min", "goal_line_x_max"]),
            "last_touch_team": TEAM_NN,
            "last_touch_slot": SLOT_NN,
        },
        "required": ["exit_side"],
    },
    "dead_ball": {
        "description": "A stretch where the ball is treated as dead, ending at the next control (the restart).",
        "subtypes": None,
        "details": {
            "reason": string("Why the stretch is dead.",
                             ["out_of_pitch", "long_ball_gap", "before_kickoff", "stoppage"]),
        },
        "required": ["reason"],
    },
    "restart": {
        "description": "Play restarts after a dead ball. Type/team from Smart Tagging when present, else inferred "
                       "from where the ball left the pitch and who touched it last.",
        "subtypes": SET_PIECES,
        "details": {
            "restart_team": TEAM_NN,
            "team_source": string("Where restart_team comes from.",
                                  ["smart_tagging", "inferred_opposite_of_last_touch", "first_control", "unknown"]),
            "type_source": string("Where the subtype comes from.", ["smart_tagging", "inferred", "unknown"]),
            "taker_slot": SLOT_NN,
            "set_piece_tag_frames": {"type": ["array", "null"], "items": {"type": "integer"},
                                     "minItems": 2, "maxItems": 2,
                                     "description": "[start_frame, end_frame] of the matched tag, if any."},
            "dead_ball_event_id": ID_NN,
        },
        "required": ["team_source", "type_source"],
    },
    "foul_candidate": {
        "description": "A free-kick restart that follows a duel/tackle. LOW CONFIDENCE: offside and handball give the "
                       "same signature, and there is no contact data.",
        "subtypes": ["free_kick_stoppage", "penalty_candidate"],
        "details": {
            "offending_team": TEAM_NN,
            "fouled_team": TEAM_NN,
            "evidence": {"type": "array", "items": {"type": "string"},
                         "description": "Reasons for the call, e.g. free_kick_tag, preceding_duel."},
            "in_penalty_area": boolean("Ball inside the offending team's own penalty area at the stoppage.", True),
            "players_within_5m": integer("Tracked players within 5 m of the ball at the stoppage.", True, 0),
            "preceding_event_id": ID_NN,
        },
        "required": ["evidence"],
    },
    "tracking_gap": {
        "description": "Ball not detected for a long stretch while play is live (only with --include-gaps).",
        "subtypes": None,
        "details": {
            "frames": integer("Length of the gap.", minimum=1),
            "bracket_distance_m": num("Ball displacement between last and next detection.", True, 0),
            "implied_speed_ms": num("bracket_distance / gap duration.", True, 0),
        },
        "required": ["frames"],
    },
}
TYPE_ORDER = {name: i for i, name in enumerate(
    ["restart", "ball_out_of_play", "dead_ball", "duel", "pass", "carry", "hold", "tackle", "interception",
     "possession_change", "shot", "goal", "foul_candidate", "tracking_gap"])}
ID_KEYS = ("intercepted_pass_id", "trigger_event_id", "shot_event_id", "duel_event_id",
           "dead_ball_event_id", "preceding_event_id")

LOCATION = {
    "type": ["object", "null"],
    "description": "Pitch position in metres (x along the pitch 0..105.3, y across 0..68). x_att_m is x "
                   "mirrored so the acting team always attacks towards larger values.",
    "properties": {
        "x_m": num("Metres along the pitch.", True),
        "y_m": num("Metres across the pitch.", True),
        "x_att_m": num("x in the acting team's attacking direction (null if no team).", True),
        "zone": string("Third of the pitch relative to the acting team.",
                       ["defensive_third", "middle_third", "final_third"], True),
        "lane": string("central = within a third of the width of the centre line.", ["central", "wide"], True),
        "in_opponent_box": boolean("Inside the penalty area the acting team attacks.", True),
        "in_own_box": boolean("Inside the acting team's own penalty area.", True),
    },
    "required": ["x_m", "y_m"],
    "additionalProperties": False,
}


def build_schema():
    base = {
        "id": string("Unique id within the file, E00001... in time order."),
        "type": string("Event type.", sorted(EVENT_TYPES)),
        "subtype": string("Type-specific refinement (see per-type constraints).", None, True),
        "period": integer("1 or 2 (half).", False, 1),
        "frame_start": integer("First video frame (0-based, tactical video).", False, 0),
        "frame_end": integer("Last video frame, inclusive.", False, 0),
        "time_start": string("Video time mm:ss.ss of frame_start (from the first frame of the video)."),
        "duration_s": num("(frame_end - frame_start + 1) / 25.", minimum=0),
        "team": TEAM_NN,
        "player_slot": SLOT_NN,
        "start_location": {"$ref": "#/$defs/location"},
        "end_location": {"$ref": "#/$defs/location"},
        "confidence": num("Heuristic 0..1 (ball visibility, rule strength). NOT a calibrated probability.",
                          False, 0, 1),
        "sources": {"type": "array", "minItems": 1, "uniqueItems": True,
                    "items": {"type": "string", "enum": ["tracking", "smart_tagging"]}},
        "details": {"type": "object", "description": "Type-specific fields, see the per-type constraints."},
        "related_event_ids": {"type": "array", "items": {"type": "string"}, "uniqueItems": True},
    }
    required = ["id", "type", "period", "frame_start", "frame_end", "time_start", "duration_s", "team",
                "player_slot", "start_location", "end_location", "confidence", "sources", "details"]
    branches = []
    for name, spec in sorted(EVENT_TYPES.items()):
        then = {"properties": {
            "details": {"type": "object", "properties": spec["details"], "required": spec["required"],
                        "additionalProperties": False}}}
        if spec["subtypes"]:
            then["properties"]["subtype"] = {"enum": list(spec["subtypes"]) + [None]}
        else:
            then["properties"]["subtype"] = {"type": "null"}
        branches.append({"if": {"properties": {"type": {"const": name}}, "required": ["type"]}, "then": then})
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://example.org/mtah2027/event-stream.schema.json",
        "title": "Football event stream (derived from automatic tracking)",
        "description": "Events detected from tracking + Smart Tagging. Automatic and uncorrected: treat as evidence.",
        "type": "object",
        "required": ["meta", "events"],
        "properties": {
            "meta": {
                "type": "object",
                "required": ["schema_version", "game", "fps", "pitch_m", "windows", "event_counts"],
                "properties": {
                    "schema_version": {"type": "string", "const": SCHEMA_VERSION},
                    "game": {"type": "string"},
                    "fps": {"type": "integer"},
                    "pitch_m": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
                    "teams": {"type": "object"},
                    "windows": {"type": "array", "items": {"type": "object"}},
                    "attack_direction": {"type": "object"},
                    "thresholds": {"type": "object"},
                    "event_counts": {"type": "object", "additionalProperties": {"type": "integer"}},
                    "warnings": {"type": "array", "items": {"type": "string"}},
                },
                "additionalProperties": True,
            },
            "events": {"type": "array", "items": {"$ref": "#/$defs/event"}},
        },
        "$defs": {
            "location": LOCATION,
            "event": {"type": "object", "required": required, "properties": base,
                      "additionalProperties": False, "allOf": branches},
        },
    }


# ==================================================================================================
# 2. Loading and signal processing
# ==================================================================================================

def clock(frame):
    s = frame / float(FPS)
    return "{:02d}:{:05.2f}".format(int(s // 60), s % 60)


def r2(v):
    if v is None:
        return None
    v = float(v)
    return round(v, 2) if np.isfinite(v) else None


def runs_of(mask):
    edges = np.diff(np.concatenate(([0], np.asarray(mask).astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def _frame_id(line):
    return int(line[:line.index(b":")])


def seek_to_frame_id(handle, target_id):
    """Byte-offset bisection on the frame-id-sorted ATD (same idea as visualize_tracking.py)."""
    handle.seek(0, os.SEEK_END)
    lo, hi = 0, handle.tell()
    while hi - lo > SEARCH_BLOCK:
        mid = (lo + hi) // 2
        handle.seek(mid - 1)
        handle.readline()
        line = handle.readline()
        if not line:
            hi = mid
        elif _frame_id(line) < target_id:
            lo = mid
        else:
            hi = mid
    if lo == 0:
        handle.seek(0)
    else:
        handle.seek(lo - 1)
        handle.readline()


def _slow_groups(text, n_slots):
    out = np.full((n_slots, 3), np.nan, dtype=np.float32)
    for slot, group in enumerate(text.split(";")[:n_slots]):
        for i, v in enumerate(group.split(",")[:3]):
            try:
                out[slot, i] = float(v)
            except ValueError:
                pass
    return out


def load_tracking(raw_path, n_slots, lo, hi):
    """Metres for video frames lo..hi: px, py (n, slots) and bx, by (n,). NaN = not seen."""
    n = hi - lo + 1
    px = np.full((n, n_slots), np.nan, dtype=np.float32)
    py = np.full((n, n_slots), np.nan, dtype=np.float32)
    bx = np.full(n, np.nan, dtype=np.float32)
    by = np.full(n, np.nan, dtype=np.float32)
    with open(raw_path, "rb") as handle:
        seek_to_frame_id(handle, lo + 1)
        for raw in handle:
            try:
                frame = _frame_id(raw) - 1
            except ValueError:
                continue
            if frame < lo:
                continue
            if frame > hi:
                break
            parts = raw.decode("ascii", "replace").strip().split(":")
            if len(parts) != 3:
                continue
            k = frame - lo
            try:
                vals = np.array(parts[1].replace(";", ",").split(","), dtype=np.float32)
                if len(vals) % n_slots:
                    raise ValueError
                vals = vals.reshape(n_slots, -1)
            except ValueError:
                vals = _slow_groups(parts[1], n_slots)
            px[k] = vals[:, 0] * PITCH_L
            py[k] = vals[:, 1] * PITCH_W
            ball = parts[2].strip().split(",")
            if len(ball) >= 2 and ball[0][:1] not in ("N", "n"):
                try:
                    bx[k] = float(ball[0]) * PITCH_L
                    by[k] = float(ball[1]) * PITCH_W
                except ValueError:
                    pass
    return px, py, bx, by


def nan_smooth(values, half):
    """Centred moving average ignoring NaN along axis 0. NaN stays NaN."""
    if half <= 0:
        return values
    ok = ~np.isnan(values)
    filled = np.where(ok, values, 0.0).astype(np.float64)
    pad = [(half + 1, half)] + [(0, 0)] * (values.ndim - 1)
    csum = np.cumsum(np.pad(filled, pad), axis=0)
    ccnt = np.cumsum(np.pad(ok.astype(np.float64), pad), axis=0)
    width = 2 * half + 1
    total, count = csum[width:] - csum[:-width], ccnt[width:] - ccnt[:-width]
    out = np.where(count > 0, total / np.maximum(count, 1), np.nan)
    return np.where(ok, out, np.nan).astype(np.float32)


def velocity(pos):
    vel = np.full_like(pos, np.nan)
    vel[1:-1] = (pos[2:] - pos[:-2]) * (FPS / 2.0)
    return vel


def remove_teleports(bx, by):
    with np.errstate(invalid="ignore"):
        step = np.hypot(np.diff(bx, prepend=np.nan), np.diff(by, prepend=np.nan)) * FPS
        jump = step > MAX_BALL_SPEED
    bx, by = bx.copy(), by.copy()
    bx[jump] = np.nan
    by[jump] = np.nan
    return bx, by


def fill_short_gaps(bx, by, max_gap):
    """Linear interpolation of ball gaps <= max_gap frames (ball_gap_analysis.py shows most are short)."""
    fx, fy = bx.copy(), by.copy()
    idx = np.flatnonzero(~np.isnan(bx) & ~np.isnan(by))
    if len(idx) < 2:
        return fx, fy
    for g in np.flatnonzero(np.diff(idx) > 1):
        a, b = idx[g], idx[g + 1]
        if b - a - 1 <= max_gap:
            t = np.arange(a + 1, b)
            fx[t] = np.interp(t, [a, b], [bx[a], bx[b]])
            fy[t] = np.interp(t, [a, b], [by[a], by[b]])
    return fx, fy


def zoi_q(px, py, vx, vy, bx, by, r0):
    """Normalised zone distance of the ball per frame and slot (<= 1 means inside the player's zone)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        speed = np.nan_to_num(np.hypot(vx, vy))
        ratio = np.clip(speed / ZOI_SPEED_MAX, 0.0, 1.0)
        moving = speed > 0.2
        ux = np.where(moving, np.nan_to_num(vx) / np.maximum(speed, 1e-6), 1.0)
        uy = np.where(moving, np.nan_to_num(vy) / np.maximum(speed, 1e-6), 0.0)
        front = r0 * (1.0 + ZOI_FRONT_GAIN * ratio)
        rear = r0 * (1.0 - ZOI_REAR_SHRINK * ratio)
        side = r0 * (1.0 - ZOI_SIDE_SHRINK * ratio)
        dx, dy = bx[:, None] - px, by[:, None] - py
        along = dx * ux + dy * uy
        lateral = -dx * uy + dy * ux
        q = (along / np.where(along >= 0, front, rear)) ** 2 + (lateral / side) ** 2
    return np.nan_to_num(q, nan=np.inf)


def compute_control(px, py, bx, by, team, args):
    spx, spy = nan_smooth(px, SMOOTH_HALF), nan_smooth(py, SMOOTH_HALF)
    vx, vy = velocity(spx), velocity(spy)
    sbx, sby = nan_smooth(bx, SMOOTH_HALF), nan_smooth(by, SMOOTH_HALF)
    bvx, bvy = velocity(sbx), velocity(sby)
    home = team == HOME
    n = len(bx)
    q = np.full(px.shape, np.inf, dtype=np.float32)
    slow = np.zeros(px.shape, dtype=bool)
    for a in range(0, n, CHUNK):
        b = min(n, a + CHUNK)
        q[a:b] = zoi_q(spx[a:b], spy[a:b], vx[a:b], vy[a:b], sbx[a:b], sby[a:b], args.zoi_r0)
        with np.errstate(invalid="ignore"):
            rel = np.hypot(bvx[a:b, None] - np.nan_to_num(vx[a:b]), bvy[a:b, None] - np.nan_to_num(vy[a:b]))
        slow[a:b] = np.where(np.isnan(rel), True, rel <= args.rel_speed_max)
    inside = q <= 1.0
    controls = inside & slow          # in the zone AND moving about as fast as the player
    h_ok, a_ok = controls[:, home].any(axis=1), controls[:, ~home].any(axis=1)
    contest = inside[:, home].any(axis=1) & inside[:, ~home].any(axis=1)
    ctrl = np.full(n, NONE, dtype=np.int8)
    ctrl[h_ok & ~a_ok] = HOME
    ctrl[a_ok & ~h_ok] = AWAY
    ctrl[contest] = BOTH
    holder = np.where(np.isin(ctrl, (HOME, AWAY)), np.argmin(np.where(controls, q, np.inf), axis=1), -1)

    def nearest(mask):
        return np.where(contest, np.argmin(np.where(mask[None, :], q, np.inf), axis=1), -1)

    return {"ctrl": ctrl, "holder": holder.astype(np.int16), "contest_h": nearest(home),
            "contest_a": nearest(~home), "spx": spx, "spy": spy, "sbx": sbx, "sby": sby,
            "bvx": bvx, "bvy": bvy}


def dead_mask(sbx, sby, ctrl, n, args, kick_k):
    """Dead = ball outside the pitch, long ball gap, or before the half's kick-off; each lasts until the
    next frame some team controls the ball. Returns (dead, reason) with reason 0 live, 1 out_of_pitch,
    2 long_ball_gap, 3 before_kickoff, 4 stoppage (waiting for the restart)."""
    missing = np.isnan(sbx)
    with np.errstate(invalid="ignore"):
        out = (~missing) & ((sbx < -OUT_MARGIN_M) | (sbx > PITCH_L + OUT_MARGIN_M)
                            | (sby < -OUT_MARGIN_M) | (sby > PITCH_W + OUT_MARGIN_M))
    dead, reason = out.copy(), np.where(out, 1, 0).astype(np.int8)
    for s, e in runs_of(missing):
        if e - s >= args.dead_gap * FPS:
            dead[s:e], reason[s:e] = True, 2
    if kick_k > 0:
        dead[:kick_k], reason[:kick_k] = True, 3
    controlled = (ctrl == HOME) | (ctrl == AWAY)
    i = 0
    while i < n:
        if dead[i]:
            j = i
            while j < n and (dead[j] or not controlled[j]):
                if not dead[j]:
                    reason[j] = 4
                j += 1
            dead[i:j] = True
            i = j
        else:
            i += 1
    return dead, reason


def touch_spells(holder, dead, gap):
    """[(start, stop, slot)]: runs where one slot holds the ball, bridging <= gap frames without a holder."""
    spells, k, n = [], 0, len(holder)
    while k < n:
        slot = holder[k]
        if slot < 0 or dead[k]:
            k += 1
            continue
        end, j = k + 1, k + 1
        while j < n and not dead[j]:
            if holder[j] == slot:
                end = j + 1
            elif holder[j] >= 0 or j - end >= gap:
                break
            j += 1
        spells.append((k, end, int(slot)))
        k = end
    return spells


# ==================================================================================================
# 3. Context shared across segments
# ==================================================================================================

class Context(object):
    def __init__(self, raw_path, team, names, tags, args):
        self.raw_path, self.team, self.names, self.args = raw_path, team, names, args
        self.n_slots = len(team)
        self.counter = 0
        self.sign = {}            # half -> {HOME: +1/-1, AWAY: +1/-1}  (+1 = attacks towards larger x)
        self.direction_info = {}
        self.set_pieces, self.goals = [], []
        for t in tags:
            code = (t.get("code") or "").strip().lower()
            if code == "set pieces":
                self.set_pieces.append(t)
            elif code == "goals":
                self.goals.append(t)

    def new_id(self):
        self.counter += 1
        return "T{:07d}".format(self.counter)

    def tag_team(self, value):
        t = (value or "").strip().lower()
        if not t or t in ("n/a", "na"):
            return None
        for code, name in zip((HOME, AWAY), self.names):
            n = (name or "").strip().lower()
            if n and (n == t or n in t or t in n):
                return code
        if "barcelona" in t:
            return HOME
        if "atletic" in t or "athletic" in t:
            return AWAY
        return None

    def attack_sign(self, half):
        """Which way each team attacks in a half, from the kick-off formation (each team starts in its own half)."""
        if half in self.sign:
            return self.sign[half]
        kick = HALF_KICKOFF[half]
        px, _, _, _ = load_tracking(self.raw_path, self.n_slots, kick, kick + 24)
        mh = np.nanmean(px[:, self.team == HOME]) if np.isfinite(px[:, self.team == HOME]).any() else np.nan
        ma = np.nanmean(px[:, self.team == AWAY]) if np.isfinite(px[:, self.team == AWAY]).any() else np.nan
        reliable = np.isfinite(mh) and np.isfinite(ma) and abs(mh - ma) >= 3.0
        home_sign = 1 if (not reliable and half == 1) or (reliable and mh < ma) else -1
        if not reliable and half == 2 and 1 in self.sign:
            home_sign = -self.sign[1][HOME]
        self.sign[half] = {HOME: home_sign, AWAY: -home_sign}
        self.direction_info[str(half)] = {
            "H": "+x" if home_sign > 0 else "-x", "A": "-x" if home_sign > 0 else "+x",
            "source": "kickoff_formation" if reliable else "fallback_assumption",
            "home_mean_x_m": r2(mh), "away_mean_x_m": r2(ma)}
        return self.sign[half]


SET_PIECE_TYPES = {"throw-in": "throw_in", "corner kick": "corner_kick", "goal kick": "goal_kick",
                   "free kick": "free_kick", "kick off (start)": "kick_off",
                   "kick off (after goal)": "kick_off_after_goal", "penalty kick": "penalty_kick",
                   "penalty": "penalty_kick"}


# ==================================================================================================
# 4. Per-segment analysis (one half or a window inside one)
# ==================================================================================================

class SegmentAnalyser(object):
    def __init__(self, ctx, half, lo, hi):
        self.ctx, self.args, self.half, self.lo, self.hi = ctx, ctx.args, half, lo, hi
        self.team = ctx.team
        self.events = []

    # ---- small helpers ----------------------------------------------------------------------

    def ball_pos(self, k, slot=None):
        if 0 <= k < self.n and np.isfinite(self.sbx[k]) and np.isfinite(self.sby[k]):
            return float(self.sbx[k]), float(self.sby[k])
        if slot is not None and 0 <= k < self.n and np.isfinite(self.spx[k, slot]):
            return float(self.spx[k, slot]), float(self.spy[k, slot])
        return float("nan"), float("nan")

    def ball_near(self, k, back=50):
        for kk in range(min(k, self.n - 1), max(-1, k - back - 1), -1):
            x, y = self.ball_pos(kk)
            if np.isfinite(x):
                return x, y
        return float("nan"), float("nan")

    def att(self, team, x):
        return x if self.sign[team] > 0 else PITCH_L - x

    def loc(self, xy, team=None):
        x, y = xy
        if not (np.isfinite(x) and np.isfinite(y)):
            return None
        out = {"x_m": r2(x), "y_m": r2(y), "x_att_m": None, "zone": None,
               "lane": "central" if abs(y - PITCH_W / 2) <= PITCH_W / 6 else "wide",
               "in_opponent_box": None, "in_own_box": None}
        if team in (HOME, AWAY):
            ax = self.att(team, x)
            out["x_att_m"] = r2(ax)
            out["zone"] = ("defensive_third" if ax < PITCH_L / 3 else
                           "middle_third" if ax < 2 * PITCH_L / 3 else "final_third")
            near_axis = abs(y - PITCH_W / 2) <= BOX_HALF_W
            out["in_opponent_box"] = bool(near_axis and ax >= PITCH_L - BOX_DEPTH)
            out["in_own_box"] = bool(near_axis and ax <= BOX_DEPTH)
        return out

    def seen_share(self, k0, k1):
        k1 = max(k1, k0 + 1)
        return float(np.mean(self.observed[k0:k1])) if k1 > k0 else 1.0

    def opp_dist(self, k, slot):
        opp = self.team != self.team[slot]
        x0, y0 = self.spx[k, slot], self.spy[k, slot]
        if not np.isfinite(x0):
            return float("nan")
        d = np.hypot(self.spx[k, opp] - x0, self.spy[k, opp] - y0)
        d = d[np.isfinite(d)]
        return float(d.min()) if len(d) else float("nan")

    def opp_dist_min(self, k0, k1, slot):
        opp = self.team != self.team[slot]
        d = np.hypot(self.spx[k0:k1][:, opp] - self.spx[k0:k1, slot][:, None],
                     self.spy[k0:k1][:, opp] - self.spy[k0:k1, slot][:, None])
        d = d[np.isfinite(d)]
        return float(d.min()) if len(d) else float("nan")

    def emit(self, etype, k0, k1, team, slot, start_xy, end_xy, details, conf, sources=("tracking",),
             subtype=None, related=None):
        fs, fe = self.lo + k0, self.lo + max(k0, k1 - 1)
        ev = {
            "id": self.ctx.new_id(), "type": etype, "subtype": subtype, "period": self.half,
            "frame_start": int(fs), "frame_end": int(fe), "time_start": clock(fs),
            "duration_s": r2((fe - fs + 1) / float(FPS)),
            "team": LETTER.get(team), "player_slot": (int(slot) + 1) if slot is not None and slot >= 0 else None,
            "start_location": self.loc(start_xy, team) if start_xy is not None else None,
            "end_location": self.loc(end_xy, team) if end_xy is not None else None,
            "confidence": round(float(min(0.95, max(0.05, conf))), 2),
            "sources": list(sources), "details": details, "related_event_ids": list(related or []),
        }
        self.events.append(ev)
        return ev

    # ---- pipeline ---------------------------------------------------------------------------

    def run(self):
        a, ctx = self.args, self.ctx
        self.sign = ctx.attack_sign(self.half)
        px, py, bx, by = load_tracking(ctx.raw_path, ctx.n_slots, self.lo, self.hi)
        self.n = n = len(bx)
        bx, by = remove_teleports(bx, by)
        self.observed = ~np.isnan(bx)
        fx, fy = fill_short_gaps(bx, by, a.max_gap)
        c = compute_control(px, py, fx, fy, self.team, a)
        self.spx, self.spy, self.sbx, self.sby = c["spx"], c["spy"], c["sbx"], c["sby"]
        self.bvx, self.bvy = c["bvx"], c["bvy"]
        self.ctrl = c["ctrl"]
        self.bspeed = np.hypot(self.bvx, self.bvy)
        kick_k = max(0, HALF_KICKOFF[self.half] - self.lo)
        self.dead, self.reason = dead_mask(self.sbx, self.sby, self.ctrl, n, a, kick_k)
        holder = np.where(self.dead, -1, c["holder"])
        spells = [s for s in touch_spells(holder, self.dead, TOUCH_GAP_FRAMES) if s[1] - s[0] >= a.min_spell]
        self.spells = self.merge_slot_switches(spells)
        self.holder_raw = c["holder"]

        self.detect_duels(c)
        self.detect_touch_events()
        self.detect_shots()
        self.detect_dead_events()
        self.detect_goals()
        if a.include_gaps:
            self.detect_gaps()
        return self.events

    def merge_slot_switches(self, spells):
        """The tracker hands the ball's owner a new slot: same team, almost the same place, no flight."""
        merged = []
        for s in spells:
            if merged:
                a0, a1, aslot = merged[-1]
                b0, b1, bslot = s
                close = (b0 - a1 < FLIGHT_MIN_FRAMES and
                         np.hypot(self.spx[a1 - 1, aslot] - self.spx[b0, bslot],
                                  self.spy[a1 - 1, aslot] - self.spy[b0, bslot]) < SLOT_SWITCH_M)
                if aslot != bslot and self.team[aslot] == self.team[bslot] and close:
                    merged[-1] = (a0, b1, bslot)
                    continue
                if aslot == bslot and b0 - a1 <= 2 * TOUCH_GAP_FRAMES and not self.dead[a1:b0].any():
                    merged[-1] = (a0, b1, bslot)
                    continue
            merged.append(s)
        return merged

    # ---- duels ------------------------------------------------------------------------------

    def detect_duels(self, c):
        self.duels = []
        live_both = (self.ctrl == BOTH) & ~self.dead
        for k0, k1 in runs_of(live_both):
            if k1 - k0 < self.args.duel_min_frames:
                continue
            hs, as_ = int(c["contest_h"][k0]), int(c["contest_a"][k0])
            if hs < 0 or as_ < 0:
                continue
            after = self.ctrl[k1:k1 + CONTEST_WINNER_LOOKAHEAD]
            sole = after[np.isin(after, (HOME, AWAY))]
            winner = LETTER[int(sole[0])] if len(sole) else None
            d = np.hypot(self.spx[k0:k1, hs] - self.spx[k0:k1, as_], self.spy[k0:k1, hs] - self.spy[k0:k1, as_])
            d = d[np.isfinite(d)]
            ev = self.emit("duel", k0, k1, None, None, self.ball_pos(k0), self.ball_pos(k1 - 1), {
                "home_slot": hs + 1, "away_slot": as_ + 1, "winner_team": winner,
                "min_player_distance_m": r2(d.min()) if len(d) else None},
                0.4 + 0.4 * self.seen_share(k0, k1))
            self.duels.append((k0, k1, ev["id"]))

    def duel_near(self, k0, k1, slack=25):
        for d0, d1, did in self.duels:
            if d0 <= k1 + slack and d1 >= k0 - slack:
                return did
        return None

    # ---- carries, holds, passes, tackles, interceptions, possession changes -----------------------

    def detect_touch_events(self):
        a = self.args
        L = PITCH_L
        for k0, k1, slot in self.spells:
            t = int(self.team[slot])
            sxy, exy = self.ball_pos(k0, slot), self.ball_pos(k1 - 1, slot)
            dist = float(np.hypot(exy[0] - sxy[0], exy[1] - sxy[1])) if np.isfinite(sxy[0]) and np.isfinite(exy[0]) else float("nan")
            dur = (k1 - k0) / float(FPS)
            opp_min = self.opp_dist_min(k0, k1, slot)
            pressure = bool(np.isfinite(opp_min) and opp_min < a.pressure_m)
            share = self.seen_share(k0, k1)
            if np.isfinite(dist) and dist >= a.carry_min_m:
                prog = self.att(t, exy[0]) - self.att(t, sxy[0])
                self.emit("carry", k0, k1, t, slot, sxy, exy, {
                    "distance_m": r2(dist), "progressive_m": r2(prog), "mean_speed_ms": r2(dist / dur),
                    "nearest_opponent_min_m": r2(opp_min), "under_pressure": pressure,
                    "is_dribble": bool(np.isfinite(opp_min) and opp_min < a.dribble_m),
                    "into_final_third": bool(self.att(t, sxy[0]) < 2 * L / 3 <= self.att(t, exy[0])),
                    "into_box": bool(self.in_box(t, exy) and not self.in_box(t, sxy)),
                    "ball_seen_share": r2(share)}, 0.5 + 0.4 * share)
            elif dur >= a.min_hold_s:
                self.emit("hold", k0, k1, t, slot, sxy, exy, {
                    "nearest_opponent_min_m": r2(opp_min), "under_pressure": pressure,
                    "ball_seen_share": r2(share)}, 0.5 + 0.4 * share)

        for (a0, a1, aslot), (b0, b1, bslot) in zip(self.spells, self.spells[1:]):
            if aslot == bslot or self.dead[a1:b0].any():
                continue
            flight = b0 - a1
            if flight > a.max_flight_frames:
                continue
            ta, tb = int(self.team[aslot]), int(self.team[bslot])
            sxy, exy = self.ball_pos(a1 - 1, aslot), self.ball_pos(b0, bslot)
            if not (np.isfinite(sxy[0]) and np.isfinite(exy[0])):
                continue
            dx, dy = exy[0] - sxy[0], exy[1] - sxy[1]
            length = float(np.hypot(dx, dy))
            share = self.seen_share(a1, b0) if flight > 0 else 1.0
            win = self.bspeed[max(0, a1 - 1):b0 + 2]
            top = float(np.nanmax(win)) if np.isfinite(win).any() else float("nan")

            if ta == tb:
                if flight < FLIGHT_MIN_FRAMES and length < a.min_pass_m:
                    continue
                dxa = dx if self.sign[ta] > 0 else -dx
                angle = float(np.degrees(np.arctan2(dy, dxa))) if length > 0 else float("nan")
                direction = ("forward" if abs(angle) < 45 else "backward" if abs(angle) > 135 else "lateral") \
                    if np.isfinite(angle) else "lateral"
                nopp = self.opp_dist(a1 - 1, aslot)
                start_wide = abs(sxy[1] - PITCH_W / 2) > BOX_HALF_W
                self.emit("pass", a1 - 1, b0 + 1, ta, aslot, sxy, exy, {
                    "outcome": "complete", "recipient_slot": bslot + 1, "recipient_team": LETTER[tb],
                    "length_m": r2(length),
                    "length_class": "short" if length < a.short_pass_m else "long" if length >= a.long_pass_m else "medium",
                    "direction": direction, "angle_deg": r2(angle), "progressive_m": r2(dxa),
                    "flight_frames": int(flight), "max_ball_speed_ms": r2(top), "ball_seen_share": r2(share),
                    "is_cross": bool(start_wide and self.att(ta, sxy[0]) >= 2 * PITCH_L / 3 and self.in_box(ta, exy)),
                    "is_switch_of_play": bool(abs(dy) >= a.switch_min_m and abs(dy) > abs(dx)),
                    "into_final_third": bool(self.att(ta, sxy[0]) < 2 * PITCH_L / 3 <= self.att(ta, exy[0])),
                    "into_box": bool(self.in_box(ta, exy) and not self.in_box(ta, sxy)),
                    "nearest_opponent_m": r2(nopp),
                    "under_pressure": bool(np.isfinite(nopp) and nopp < a.pressure_m)},
                    0.45 + 0.4 * share + (0.1 if flight >= FLIGHT_MIN_FRAMES else 0.0))
                continue

            # Ball changes team.
            new_hold = (b1 - b0) / float(FPS)
            duel_id = self.duel_near(a1, b0)
            if flight >= FLIGHT_MIN_FRAMES:
                dxa = dx if self.sign[ta] > 0 else -dx
                p = self.emit("pass", a1 - 1, b0 + 1, ta, aslot, sxy, exy, {
                    "outcome": "intercepted", "recipient_slot": bslot + 1, "recipient_team": LETTER[tb],
                    "length_m": r2(length),
                    "length_class": "short" if length < a.short_pass_m else "long" if length >= a.long_pass_m else "medium",
                    "direction": "forward" if dxa > 0.7 * length else "backward" if dxa < -0.7 * length else "lateral",
                    "angle_deg": None, "progressive_m": r2(dxa), "flight_frames": int(flight),
                    "max_ball_speed_ms": r2(top), "ball_seen_share": r2(share), "is_cross": False,
                    "is_switch_of_play": False, "into_final_third": False, "into_box": False,
                    "nearest_opponent_m": r2(self.opp_dist(a1 - 1, aslot)), "under_pressure": False},
                    0.4 + 0.4 * share)
                i = self.emit("interception", b0, b0 + 2, tb, bslot, exy, exy, {
                    "intercepted_pass_id": p["id"], "passer_slot": aslot + 1, "passer_team": LETTER[ta],
                    "flight_frames": int(flight), "distance_from_passer_m": r2(length)},
                    0.4 + 0.4 * share, related=[p["id"]])
                trigger, kind = i["id"], ("contest" if duel_id else "interception" if flight <= a.max_flight_frames // 2 else "loose_ball_recovery")
            else:
                tk = self.emit("tackle", b0, b0 + 2, tb, bslot, exy, exy, {
                    "dispossessed_slot": aslot + 1, "dispossessed_team": LETTER[ta], "duel_event_id": duel_id},
                    0.5 + 0.3 * share, related=[duel_id] if duel_id else [])
                trigger, kind = tk["id"], ("contest" if duel_id else "tackle")
            if new_hold >= a.min_change_hold_s:
                self.emit("possession_change", b0, b0 + 2, tb, bslot, exy, exy, {
                    "from_team": LETTER[ta], "to_team": LETTER[tb], "kind": kind, "trigger_event_id": trigger,
                    "flight_frames": int(flight), "new_team_hold_s": r2(new_hold)},
                    0.5 + 0.3 * share, related=[trigger])

    def in_box(self, team, xy):
        x, y = xy
        return bool(np.isfinite(x) and abs(y - PITCH_W / 2) <= BOX_HALF_W and self.att(team, x) >= PITCH_L - BOX_DEPTH)

    # ---- shots ------------------------------------------------------------------------------

    def detect_shots(self):
        a = self.args
        for idx, (k0, k1, slot) in enumerate(self.spells):
            t = int(self.team[slot])
            s = self.sign[t]
            xg = PITCH_L if s > 0 else 0.0
            win = slice(k1, min(self.n, k1 + 6))
            vx, vy = self.bvx[win], self.bvy[win]
            ok = np.isfinite(vx) & np.isfinite(vy)
            if ok.sum() < 2:
                continue
            vx, vy = float(vx[ok].mean()), float(vy[ok].mean())
            speed = float(np.hypot(vx, vy))
            if speed < a.shot_min_speed or vx * s <= 0:
                continue
            x0, y0 = self.ball_pos(k1 - 1, slot)
            if not np.isfinite(x0):
                continue
            dist = float(np.hypot(xg - x0, PITCH_W / 2 - y0))
            tt = (xg - x0) / vx
            if dist > a.shot_max_dist_m or tt <= 0 or tt > 4.0:
                continue
            y_line = y0 + vy * tt
            if abs(y_line - PITCH_W / 2) > GOAL_HALF_W + a.shot_goal_margin_m:
                continue
            end = min(self.n, k1 + 75)
            seg = self.sbx[k1:end]
            with np.errstate(invalid="ignore"):
                crossed = bool(np.any(seg >= PITCH_L - 0.3) if s > 0 else np.any(seg <= 0.3))
            nxt = self.spells[idx + 1] if idx + 1 < len(self.spells) else None
            received_by_mate = (nxt is not None and nxt[0] - k1 <= 75 and self.team[nxt[2]] == t and nxt[2] != slot)
            if received_by_mate and not crossed:
                continue
            f_rel = self.lo + k1
            goal = any(self.ctx.tag_team(g["tags"].get("Team")) == t and g["end_frame"] >= f_rel - 25
                       and g["start_frame"] <= f_rel + 10 * FPS for g in self.ctx.goals)
            src = ("tracking", "smart_tagging") if goal else ("tracking",)
            self.emit("shot", k1 - 1, k1 + 1, t, slot, (x0, y0), self.ball_near(k1 + int(tt * FPS)), {
                "target_goal": "x_max" if s > 0 else "x_min", "distance_to_goal_m": r2(dist),
                "projected_goal_line_y_m": r2(y_line), "release_ball_speed_ms": r2(speed),
                "outcome": "goal" if goal else "no_goal_detected", "reached_goal_line": crossed},
                0.35 + (0.25 if crossed else 0.0) + (0.2 if goal else 0.0), sources=src)

    # ---- dead balls, out of play, restarts, fouls -------------------------------------------

    def last_touch(self, k, max_back=125):
        for a0, a1, slot in reversed(self.spells):
            if a0 > k:
                continue
            return slot if k - a1 <= max_back else None
        return None

    def match_set_piece(self, f_end):
        best = None
        for sp in self.ctx.set_pieces:
            if sp["end_frame"] >= f_end - 75 and sp["start_frame"] <= f_end + 75:
                d = abs((sp["start_frame"] + sp["end_frame"]) / 2.0 - f_end)
                if best is None or d < best[0]:
                    best = (d, sp)
        return best[1] if best else None

    def out_mask(self, margin):
        with np.errstate(invalid="ignore"):
            return (np.isfinite(self.sbx) & ((self.sbx < -margin) | (self.sbx > PITCH_L + margin)
                                              | (self.sby < -margin) | (self.sby > PITCH_W + margin)))

    def detect_dead_events(self):
        a = self.args
        out_loose, out_strict = self.out_mask(OUT_MARGIN_M), self.out_mask(0.0)
        for k0, k1 in runs_of(self.dead):
            codes = self.reason[k0:k1]
            reason = ("before_kickoff" if (codes == 3).any() else "out_of_pitch" if (codes == 1).any()
                      else "long_ball_gap" if (codes == 2).any() else "stoppage")
            start_xy = self.ball_near(k0)
            dead_ev = self.emit("dead_ball", k0, k1, None, None, start_xy, self.ball_near(k1 - 1) if k1 > k0 else None,
                                {"reason": reason}, 0.6 if reason == "out_of_pitch" else 0.45)
            last_slot = self.last_touch(k0)
            last_team = int(self.team[last_slot]) if last_slot is not None else None
            exit_side = None
            if reason == "out_of_pitch":
                idx = np.flatnonzero(out_loose[k0:k1])
                kk = k0 + int(idx[0]) if len(idx) else k0
                while kk > 0 and out_strict[kk - 1]:
                    kk -= 1
                x, y = self.ball_pos(kk)
                if np.isfinite(x):
                    exit_side = ("goal_line_x_min" if x < 0 else "goal_line_x_max" if x > PITCH_L else
                                 "touchline_y_min" if y < 0 else "touchline_y_max")
                    self.emit("ball_out_of_play", kk, kk + 1, last_team, last_slot, (x, y), (x, y), {
                        "exit_side": exit_side, "last_touch_team": LETTER.get(last_team),
                        "last_touch_slot": (last_slot + 1) if last_slot is not None else None},
                        0.55, related=[dead_ev["id"]])
            if k1 >= self.n:
                continue                      # still dead at the end of the window: no restart seen
            self.emit_restart(k0, k1, reason, exit_side, last_team, start_xy, dead_ev)

    def emit_restart(self, k0, k1, reason, exit_side, last_team, start_xy, dead_ev):
        f_end = self.lo + k1
        sp = self.match_set_piece(f_end)
        subtype, type_src = "unknown", "unknown"
        tag_team = None
        if sp is not None:
            raw = (sp["tags"].get("Type") or "").strip().lower()
            if raw in SET_PIECE_TYPES:
                subtype, type_src = SET_PIECE_TYPES[raw], "smart_tagging"
            tag_team = self.ctx.tag_team(sp["tags"].get("Team"))
        if subtype == "unknown" and reason == "out_of_pitch" and exit_side:
            if exit_side.startswith("touchline"):
                subtype, type_src = "throw_in", "inferred"
            elif last_team is not None:
                end_x = 0.0 if exit_side.endswith("x_min") else PITCH_L
                defends_end = (self.sign[last_team] > 0) == (end_x == 0.0)
                subtype, type_src = ("corner_kick" if defends_end else "goal_kick"), "inferred"
        first = int(self.ctrl[k1]) if self.ctrl[k1] in (HOME, AWAY) else None
        if tag_team is not None:
            team, team_src = tag_team, "smart_tagging"
        elif last_team is not None and reason == "out_of_pitch":
            team, team_src = 1 - last_team, "inferred_opposite_of_last_touch"
        elif first is not None:
            team, team_src = first, "first_control"
        else:
            team, team_src = None, "unknown"
        taker = int(self.holder_raw[k1]) if self.holder_raw[k1] >= 0 else None
        xy = self.ball_pos(k1, taker)
        conf = 0.5 + (0.3 if type_src == "smart_tagging" else 0.1 if type_src == "inferred" else 0.0)
        ev = self.emit("restart", k1, k1 + 1, team, taker, xy, xy, {
            "restart_team": LETTER.get(team), "team_source": team_src, "type_source": type_src,
            "taker_slot": (taker + 1) if taker is not None else None,
            "set_piece_tag_frames": [int(sp["start_frame"]), int(sp["end_frame"])] if sp is not None else None,
            "dead_ball_event_id": dead_ev["id"]}, conf,
            sources=("tracking", "smart_tagging") if sp is not None else ("tracking",),
            subtype=subtype, related=[dead_ev["id"]])
        if subtype == "free_kick" and team is not None:
            self.emit_foul_candidate(k0, team, start_xy, ev, dead_ev)

    def emit_foul_candidate(self, k0, fouled_team, start_xy, restart_ev, dead_ev):
        offender = 1 - fouled_team
        f0 = self.lo + k0
        pre = [e for e in self.events if e["type"] in ("duel", "tackle", "possession_change")
               and f0 - 8 * FPS <= e["frame_start"] <= f0 + 10]
        evidence = ["free_kick_tag"]
        if any(e["type"] == "duel" for e in pre):
            evidence.append("preceding_duel")
        if any(e["type"] == "tackle" for e in pre):
            evidence.append("preceding_tackle")
        if any(e["type"] == "possession_change" for e in pre):
            evidence.append("preceding_possession_change")
        near = None
        in_area = None
        x, y = start_xy
        if np.isfinite(x):
            goal_x = 0.0 if self.sign[offender] > 0 else PITCH_L
            in_area = bool(abs(x - goal_x) <= BOX_DEPTH and abs(y - PITCH_W / 2) <= BOX_HALF_W)
            kk = max(0, min(self.n - 1, k0))
            d = np.hypot(self.spx[kk] - x, self.spy[kk] - y)
            near = int(np.sum(d < 5.0))
        conf = min(0.6, 0.3 + 0.1 * (len(evidence) - 1) + (0.1 if in_area else 0.0))
        self.emit("foul_candidate", k0, k0 + 1, offender, None, start_xy, start_xy, {
            "offending_team": LETTER[offender], "fouled_team": LETTER[fouled_team], "evidence": evidence,
            "in_penalty_area": in_area, "players_within_5m": near,
            "preceding_event_id": pre[-1]["id"] if pre else None}, conf,
            sources=("tracking", "smart_tagging"),
            subtype="penalty_candidate" if in_area else "free_kick_stoppage",
            related=[restart_ev["id"], dead_ev["id"]] + [e["id"] for e in pre[-2:]])

    # ---- goals (Smart Tagging) --------------------------------------------------------------

    def detect_goals(self):
        for g in self.ctx.goals:
            if g["end_frame"] < self.lo or g["start_frame"] > self.hi:
                continue
            k0 = max(0, g["start_frame"] - self.lo)
            k1 = min(self.n, g["end_frame"] - self.lo + 1)
            team = self.ctx.tag_team(g["tags"].get("Team"))
            shots = [e for e in self.events if e["type"] == "shot" and e["team"] == LETTER.get(team)
                     and g["start_frame"] - 10 * FPS <= e["frame_start"] <= g["end_frame"] + 25]
            sid = shots[-1]["id"] if shots else None
            xy = self.ball_near((k0 + k1) // 2)
            self.emit("goal", k0, max(k1, k0 + 1), team, None, xy, xy, {
                "tag_frame_start": int(g["start_frame"]), "tag_frame_end": int(g["end_frame"]),
                "shot_event_id": sid}, 0.7, sources=("smart_tagging",), related=[sid] if sid else [])

    # ---- tracking gaps ----------------------------------------------------------------------

    def detect_gaps(self):
        for k0, k1 in runs_of(~self.observed & ~self.dead):
            if k1 - k0 < 25:
                continue
            b, f = (self.ball_pos(k0 - 1), self.ball_pos(k1)) if k0 > 0 and k1 < self.n else (None, None)
            dist = float(np.hypot(f[0] - b[0], f[1] - b[1])) if b is not None and np.isfinite(b[0]) and np.isfinite(f[0]) else float("nan")
            self.emit("tracking_gap", k0, k1, None, None, b, f, {
                "frames": int(k1 - k0), "bracket_distance_m": r2(dist),
                "implied_speed_ms": r2(dist / ((k1 - k0 + 1) / float(FPS))) if np.isfinite(dist) else None}, 0.9)


# ==================================================================================================
# 5. Output
# ==================================================================================================

def renumber(events):
    events.sort(key=lambda e: (e["frame_start"], TYPE_ORDER.get(e["type"], 99), e["id"]))
    mapping = {e["id"]: "E{:05d}".format(i + 1) for i, e in enumerate(events)}
    for e in events:
        e["id"] = mapping[e["id"]]
        e["related_event_ids"] = sorted({mapping[r] for r in e["related_event_ids"] if r in mapping})
        for key in ID_KEYS:
            if key in e["details"] and e["details"][key] is not None:
                e["details"][key] = mapping.get(e["details"][key])
    return events


def load_tags():
    try:
        from starter import load_smart_tagging
        return load_smart_tagging.load()
    except (IOError, ImportError) as error:
        print("note: Smart Tagging disabled ({})".format(str(error).splitlines()[0]))
        return []


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--schema-only", action="store_true", help="write event_schema.json and stop (no data needed)")
    ap.add_argument("--full", action="store_true", help="both halves (several minutes)")
    ap.add_argument("--start", type=int, default=HALF_KICKOFF[1], help="first video frame (default 11185)")
    ap.add_argument("--seconds", type=float, default=300.0, help="window length (default 300)")
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--jsonl", action="store_true", help="also write one event per line")
    ap.add_argument("--validate", action="store_true", help="check the output against the schema (needs jsonschema)")
    ap.add_argument("--include-gaps", action="store_true", help="emit tracking_gap events")
    g = ap.add_argument_group("thresholds (untuned starting points; calibrate on OTHER games)")
    g.add_argument("--zoi-r0", type=float, default=1.5, help="zone radius at standstill, m")
    g.add_argument("--rel-speed-max", type=float, default=4.0, help="ball speed relative to player still 'control', m/s")
    g.add_argument("--max-gap", type=int, default=12, help="interpolate ball gaps up to this many frames")
    g.add_argument("--dead-gap", type=float, default=10.0, help="ball missing this long (s) = dead ball")
    g.add_argument("--min-spell", type=int, default=2, help="shortest touch spell, frames")
    g.add_argument("--max-flight-frames", type=int, default=75, help="longest free flight linked into a pass")
    g.add_argument("--min-pass-m", type=float, default=2.0, help="shortest pass without free flight")
    g.add_argument("--short-pass-m", type=float, default=12.0)
    g.add_argument("--long-pass-m", type=float, default=30.0)
    g.add_argument("--switch-min-m", type=float, default=25.0, help="lateral distance of a switch of play")
    g.add_argument("--carry-min-m", type=float, default=2.0)
    g.add_argument("--min-hold-s", type=float, default=0.5)
    g.add_argument("--pressure-m", type=float, default=2.5)
    g.add_argument("--dribble-m", type=float, default=2.0)
    g.add_argument("--min-change-hold-s", type=float, default=0.3, help="new team must keep the ball this long")
    g.add_argument("--duel-min-frames", type=int, default=5)
    g.add_argument("--shot-min-speed", type=float, default=14.0, help="m/s")
    g.add_argument("--shot-max-dist-m", type=float, default=40.0)
    g.add_argument("--shot-goal-margin-m", type=float, default=2.0, help="slack either side of the posts")
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    schema = build_schema()
    schema_path = os.path.join(args.out_dir, "event_schema.json")
    with open(schema_path, "w") as handle:
        json.dump(schema, handle, indent=2)
    print("wrote {}  ({} event types)".format(schema_path, len(EVENT_TYPES)))
    if args.schema_only:
        return

    from starter import baseline
    meta_path, raw_path = require(["atd_meta", "atd_raw"])
    team = baseline.team_of_each_track(meta_path)
    names = list(baseline.team_names(meta_path))
    ctx = Context(raw_path, team, names, load_tags(), args)

    blocks = half_video_blocks(GAME)
    lo_req = blocks[0][1] if args.full else args.start
    hi_req = blocks[-1][2] if args.full else args.start + int(round(args.seconds * FPS)) - 1
    segments = [(h, max(lo_req, f), min(hi_req, l)) for h, f, l in blocks if max(lo_req, f) <= min(hi_req, l)]
    if not segments:
        sys.exit("window {}..{} is outside both halves {}".format(lo_req, hi_req, [(f, l) for _, f, l in blocks]))

    events, windows, t0 = [], [], time.time()
    for half, lo, hi in segments:
        t1 = time.time()
        found = SegmentAnalyser(ctx, half, lo, hi).run()
        events.extend(found)
        windows.append({"period": half, "frame_start": lo, "frame_end": hi, "time_start": clock(lo),
                        "time_end": clock(hi), "events": len(found)})
        print("half {}: frames {}..{} -> {} events ({:.0f}s)".format(half, lo, hi, len(found), time.time() - t1))
    events = renumber(events)

    counts = {}
    for e in events:
        counts[e["type"]] = counts.get(e["type"], 0) + 1
    warnings = ["Automatic, uncorrected tracking: events are evidence, not ground truth.",
                "foul_candidate is a Free Kick tag + preceding duel/tackle; offside and handball look the same.",
                "Derived from the match data: do not publish or upload to any cloud AI tool (RULES.md)."]
    if len(ctx.sign) == 2 and ctx.sign[1][HOME] == ctx.sign[2][HOME]:
        warnings.append("Home attacks the same way in both halves: the kick-off formation check looks wrong.")
    if not ctx.set_pieces:
        warnings.append("No Smart Tagging loaded: restart type/team are inferred only and goals are missing.")
    doc = {"meta": {
        "schema_version": SCHEMA_VERSION, "game": GAME, "fps": FPS, "pitch_m": [PITCH_L, PITCH_W],
        "teams": {"H": names[0] if names else "home", "A": names[1] if len(names) > 1 else "away"},
        "windows": windows, "attack_direction": ctx.direction_info,
        "thresholds": {k: v for k, v in vars(args).items() if k not in ("out_dir",) and not isinstance(v, bool)},
        "event_counts": counts, "warnings": warnings}, "events": events}

    tag = "full" if args.full else "{}_{}".format(lo_req, hi_req)
    out_path = os.path.join(args.out_dir, "events_{}.json".format(tag))
    with open(out_path, "w") as handle:
        json.dump(doc, handle, indent=1)
    print("wrote {}  ({} events, {:.0f}s total)".format(out_path, len(events), time.time() - t0))
    if args.jsonl:
        with open(out_path.replace(".json", ".jsonl"), "w") as handle:
            handle.writelines(json.dumps(e) + "\n" for e in events)
    print("  " + ", ".join("{} {}".format(v, k) for k, v in sorted(counts.items())))

    if args.validate:
        try:
            import jsonschema
        except ImportError:
            print("note: --validate needs `pip install jsonschema`")
            return
        validator = jsonschema.Draft202012Validator(schema)
        errors = list(validator.iter_errors(doc))
        print("schema validation: {}".format("OK" if not errors else "{} errors".format(len(errors))))
        for e in errors[:5]:
            print("  {}: {}".format("/".join(str(p) for p in e.absolute_path), e.message[:160]))
        if errors:
            sys.exit(1)


if __name__ == "__main__":
    main()