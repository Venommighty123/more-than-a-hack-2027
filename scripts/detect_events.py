"""Detect football events (pass, carry, tackle, shot, restart, foul candidate, ...) from the provided
tracking + Smart Tagging, and emit them as JSON together with a JSON Schema describing them.

    python scripts/detect_events.py --schema-only                  # just event_schema.json (no data needed)
    python scripts/detect_events.py                                # first 5 min of play
    python scripts/detect_events.py --start 11185 --seconds 120    # any window (clipped to the halves)
    python scripts/detect_events.py --full --validate              # whole match, check against the schema
    python scripts/detect_events.py --full --jsonl --include-gaps

Writes to outputs/events/ (or --out-dir):
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
it out of the public repository (outputs/ is git-ignored) and do not upload it to any cloud AI tool.
"""
import argparse
import json
import os
import sys
import time

try:
    import _bootstrap  # noqa: F401  (python scripts/x.py)
except ImportError:
    from scripts import _bootstrap  # noqa: F401  (python -m scripts.x)

from games import GAME, half_video_blocks, require
from possession.events.detect import (FPS, HALF_KICKOFF, HOME, PITCH_L, PITCH_W, Context,
                                      SegmentAnalyser, clock, load_tags, renumber)
from possession.events.schema import EVENT_TYPES, SCHEMA_VERSION, build_schema
from possession.io.teams import load_teams

DEFAULT_OUT = os.path.join(_bootstrap.ROOT, "outputs", "events")


def parse_args(argv=None):
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
    return ap.parse_args(argv)


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

    meta_path, raw_path = require(["atd_meta", "atd_raw"])
    teams = load_teams(meta_path)
    team, names = teams.team, list(teams.names)
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
