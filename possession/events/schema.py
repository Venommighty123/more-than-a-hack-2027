"""JSON Schema (draft 2020-12) of the football events written by possession/events/detect.py.

From event_detection.py, section 1 (unchanged). Contains NO match data: the schema file it writes
(outputs/events/event_schema.json) is safe to share.

    from possession.events.schema import build_schema, EVENT_TYPES, SCHEMA_VERSION
"""

SCHEMA_VERSION = "1.0.0"


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
