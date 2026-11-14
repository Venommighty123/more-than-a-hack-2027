"""Detect football events (pass, carry, tackle, shot, restart, foul candidate, ...) from OUR YOLO player
tracks + the fixed ball + Smart Tagging, and emit them as JSON together with a JSON Schema.

    python scripts/detect_events.py --schema-only                  # just event_schema.json (no data needed)
    python scripts/detect_events.py                                # first 5 min of play
    python scripts/detect_events.py --start 11185 --seconds 120    # any window (clipped to the halves)
    python scripts/detect_events.py --full --validate              # whole match, check against the schema
    python scripts/detect_events.py --full --jsonl --include-gaps

Writes to outputs/events/ (or --out-dir):
    event_schema.json          JSON Schema (draft 2020-12). Contains NO match data: safe to share.
    events_<lo>_<hi>.json      {"meta": ..., "events": [...]}   (events_full.json with --full)
    events_<lo>_<hi>.jsonl     one event per line, with --jsonl

Inputs used: possession.io.load() = our YOLO player tracks (packed and cleaned by
possession/io/detections.py) + the fixed ball (outputs/ball/ball_reconstructed.npz), the club names
from the ATD metadata, and Smart Tagging (set-piece type / team, goals). The provided ATD player
tracks are not used. Merged from scripts/fix_events.py (now in backup/), so two of its choices are
the defaults here:
  * the ball gap bridge (--max-gap) is set to 0 when the fixed ball is used: the reconstruction already
    filled what is safe, re-bridging would draw back the fake 300 km/h kicks (--keep-max-gap to keep it);
  * "seen" (event confidence, tracking_gap events) counts only frames the tracker really saw the ball.
--compare-events OLD.json writes events_diff.csv (kept / moved / added / removed) against older events. Nothing else: no ground truth, no hand labels, no video, and not
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

import csv

import numpy as np

from games import GAME, half_video_blocks
from possession.events.detect import (FPS, HALF_KICKOFF, HOME, PITCH_L, PITCH_W, Context,
                                      SegmentAnalyser, clock, load_tags, renumber)
from possession.events.schema import EVENT_TYPES, SCHEMA_VERSION, build_schema
from possession.io import load
from possession.io.detections import DEFAULT_FILE as PLAYERS_FILE
from possession.preprocess.reconstruct import DEFAULT_FILE as BALL_FILE, load_ball_file

DEFAULT_OUT = os.path.join(_bootstrap.ROOT, "outputs", "events")


MATCH_FRAMES = 12          # an old and a new event of the same type within 0.5 s are the same event


class ObservedAnalyser(SegmentAnalyser):
    """The detector, but "observed" is what the tracker saw, not what reconstruction filled in."""

    def __init__(self, ctx, half, lo, hi, observed_full):
        super().__init__(ctx, half, lo, hi)
        n = hi - lo + 1
        obs = np.zeros(n, dtype=bool)
        a, b = max(lo, 0), min(hi + 1, len(observed_full))
        obs[a - lo:b - lo] = observed_full[a:b]
        self._obs = obs

    @property
    def observed(self):
        return self._obs

    @observed.setter
    def observed(self, value):          # SegmentAnalyser.run assigns ~isnan(ball): ignore it
        pass


def ball_observed(path, n, bx):
    """Frames the tracker really saw the ball (not removed as a spike, not reconstructed)."""
    if not os.path.isfile(path):
        return np.isfinite(bx)
    z = load_ball_file(path)
    obs = np.zeros(n, dtype=bool)
    lo = z["lo"]
    a, b = max(lo, 0), min(lo + len(z["x"]), n)
    obs[a:b] = z["observed"][a - lo:b - lo].astype(bool) & ~z["removed"][a - lo:b - lo].astype(bool)
    return obs


# -- comparison ------------------------------------------------------------------------------------

def diff_events(old, new, tol=MATCH_FRAMES):
    """Greedy one-to-one match per type (and team) by start frame. Rows for a CSV."""
    rows, used = [], set()
    by_type = {}
    for j, e in enumerate(new):
        by_type.setdefault(e["type"], []).append(j)
    for e in old:
        best, best_d = None, tol + 1
        for j in by_type.get(e["type"], []):
            if j in used or new[j].get("team") != e.get("team"):
                continue
            d = abs(new[j]["frame_start"] - e["frame_start"])
            if d < best_d:
                best, best_d = j, d
        if best is None:
            rows.append(("removed", e, None))
        else:
            used.add(best)
            rows.append(("kept" if best_d == 0 and new[best]["frame_end"] == e["frame_end"] else "moved",
                         e, new[best]))
    rows += [("added", None, new[j]) for j in range(len(new)) if j not in used]
    rows.sort(key=lambda r: (r[2] or r[1])["frame_start"])
    return rows


def write_diff(path, rows):
    with open(path, "w", newline="") as h:
        w = csv.writer(h)
        w.writerow(["status", "type", "team", "old_id", "new_id", "old_frame_start", "new_frame_start",
                    "old_frame_end", "new_frame_end", "old_confidence", "new_confidence", "clock"])
        for st, o, n in rows:
            e = n or o
            w.writerow([st, e["type"], e.get("team"), o and o["id"], n and n["id"], o and o["frame_start"],
                        n and n["frame_start"], o and o["frame_end"], n and n["frame_end"],
                        o and o.get("confidence"), n and n.get("confidence"), clock(e["frame_start"])])


def event_counts(events):
    c = {}
    for e in events:
        c[e["type"]] = c.get(e["type"], 0) + 1
    return c


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
    ap.add_argument("--players-file", default=os.path.join(_bootstrap.ROOT, PLAYERS_FILE),
                    help="our YOLO tracks (kaggle/kaggle_player_tracking.ipynb)")
    ap.add_argument("--team-mode", choices=("local", "track"), default="local")
    ap.add_argument("--ball-file", default=os.path.join(_bootstrap.ROOT, BALL_FILE),
                    help="fixed ball of scripts/reconstruct_ball.py")
    ap.add_argument("--keep-max-gap", action="store_true",
                    help="keep bridging ball gaps (--max-gap) even on the fixed ball")
    ap.add_argument("--compare-events", default=None,
                    help="older events JSON to diff against (writes events_diff.csv)")
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

    tr = load(players_file=args.players_file, team_mode=args.team_mode, ball_file=args.ball_file)
    team, names = tr.teams.team, list(tr.teams.names)
    print("players: {}\nball: {}".format(tr.stats["players"], tr.stats["ball"]))
    fixed_ball = tr.stats["ball"].startswith("fixed ball")
    if fixed_ball and not args.keep_max_gap and args.max_gap:
        print("ball gap bridge off (--max-gap {} -> 0): the fixed ball is already filled".format(args.max_gap))
        args.max_gap = 0
    observed = ball_observed(args.ball_file, tr.n, tr.bx)
    ctx = Context(None, team, names, load_tags(), args, arrays=(tr.px, tr.py, tr.bx, tr.by),
                  kickoff=HALF_KICKOFF)

    blocks = half_video_blocks(GAME)
    lo_req = blocks[0][1] if args.full else args.start
    hi_req = blocks[-1][2] if args.full else args.start + int(round(args.seconds * FPS)) - 1
    segments = [(h, max(lo_req, f), min(hi_req, l)) for h, f, l in blocks if max(lo_req, f) <= min(hi_req, l)]
    if not segments:
        sys.exit("window {}..{} is outside both halves {}".format(lo_req, hi_req, [(f, l) for _, f, l in blocks]))

    events, windows, t0 = [], [], time.time()
    for half, lo, hi in segments:
        t1 = time.time()
        found = ObservedAnalyser(ctx, half, lo, hi, observed).run()
        events.extend(found)
        windows.append({"period": half, "frame_start": lo, "frame_end": hi, "time_start": clock(lo),
                        "time_end": clock(hi), "events": len(found)})
        print("half {}: frames {}..{} -> {} events ({:.0f}s)".format(half, lo, hi, len(found), time.time() - t1))
    events = renumber(events)

    counts = {}
    for e in events:
        counts[e["type"]] = counts.get(e["type"], 0) + 1
    warnings = ["Automatic tracking (our YOLO players, fixed ball): events are evidence, not ground truth.",
                "Player slots are tracklets (one slot holds several YOLO tracks in turn), not people.",
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
        "inputs": {"players": tr.stats["players"], "team_mode": tr.stats["team_mode"],
                   "players_clean": tr.stats.get("clean"), "ball": tr.stats["ball"],
                   "detector_max_gap": args.max_gap, "seen_share": "tracker-observed ball frames only"},
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

    if args.compare_events:
        with open(args.compare_events) as handle:
            old = [e for e in json.load(handle)["events"] if lo_req <= e["frame_start"] <= hi_req]
        rows = diff_events(old, events)
        diff_path = os.path.join(args.out_dir, "events_diff.csv")
        write_diff(diff_path, rows)
        st = {}
        for k, _, _ in rows:
            st[k] = st.get(k, 0) + 1
        oc = event_counts(old)
        print("\nchanges vs {}: {}".format(args.compare_events, "  ".join(
            "{} {}".format(k, v) for k, v in sorted(st.items()))))
        for k in sorted(set(oc) | set(counts)):
            print("  {:<22} {:>6} {:>6} {:>+7}".format(k, oc.get(k, 0), counts.get(k, 0),
                                                     counts.get(k, 0) - oc.get(k, 0)))
        print("-> {}".format(diff_path))

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
