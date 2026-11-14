p = "scripts/detect_events.py"
s = open(p, encoding="utf-8").read()
fe = open("scripts/fix_events.py", encoding="utf-8").read()

# docstring
s = s.replace('''"""Detect football events (pass, carry, tackle, shot, restart, foul candidate, ...) from the provided
tracking + Smart Tagging, and emit them as JSON together with a JSON Schema describing them.''',
'''"""Detect football events (pass, carry, tackle, shot, restart, foul candidate, ...) from OUR YOLO player
tracks + the fixed ball + Smart Tagging, and emit them as JSON together with a JSON Schema.''', 1)
s = s.replace('''Inputs used: ATD tracking (players + ball), the ATD metadata (team of each slot) and Smart Tagging
(set-piece type / team, goals).''',
'''Inputs used: possession.io.load() = our YOLO player tracks (packed and cleaned by
possession/io/detections.py) + the fixed ball (outputs/ball/ball_reconstructed.npz), the club names
from the ATD metadata, and Smart Tagging (set-piece type / team, goals). The provided ATD player
tracks are not used. Merged from scripts/fix_events.py (now in backup/), so two of its choices are
the defaults here:
  * the ball gap bridge (--max-gap) is set to 0 when the fixed ball is used: the reconstruction already
    filled what is safe, re-bridging would draw back the fake 300 km/h kicks (--keep-max-gap to keep it);
  * "seen" (event confidence, tracking_gap events) counts only frames the tracker really saw the ball.
--compare-events OLD.json writes events_diff.csv (kept / moved / added / removed) against older events.''', 1)

# imports
s = s.replace('''from games import GAME, half_video_blocks, require
from possession.events.detect import (FPS, HALF_KICKOFF, HOME, PITCH_L, PITCH_W, Context,
                                      SegmentAnalyser, clock, load_tags, renumber)
from possession.events.schema import EVENT_TYPES, SCHEMA_VERSION, build_schema
from possession.io.teams import load_teams
''', '''import csv

import numpy as np

from games import GAME, half_video_blocks
from possession.events.detect import (FPS, HALF_KICKOFF, HOME, PITCH_L, PITCH_W, Context,
                                      SegmentAnalyser, clock, load_tags, renumber)
from possession.events.schema import EVENT_TYPES, SCHEMA_VERSION, build_schema
from possession.io import load
from possession.io.detections import DEFAULT_FILE as PLAYERS_FILE
from possession.preprocess.reconstruct import DEFAULT_FILE as BALL_FILE, load_ball_file
''', 1)

# helpers moved from fix_events
a = fe.index("MATCH_FRAMES = 12")
b = fe.index("# -- inputs ---")
analyser = fe[a:b]
a = fe.index("# -- comparison ---")
b = fe.index("# -- main ---")
comparison = fe[a:b]
helpers = analyser + '''def ball_observed(path, n, bx):
    """Frames the tracker really saw the ball (not removed as a spike, not reconstructed)."""
    if not os.path.isfile(path):
        return np.isfinite(bx)
    z = load_ball_file(path)
    obs = np.zeros(n, dtype=bool)
    lo = z["lo"]
    a, b = max(lo, 0), min(lo + len(z["x"]), n)
    obs[a:b] = z["observed"][a - lo:b - lo].astype(bool) & ~z["removed"][a - lo:b - lo].astype(bool)
    return obs


''' + comparison
anchor = "def parse_args(argv=None):"
s = s.replace(anchor, helpers + anchor, 1)

# new args
old = '''    ap.add_argument("--include-gaps", action="store_true", help="emit tracking_gap events")'''
new = '''    ap.add_argument("--include-gaps", action="store_true", help="emit tracking_gap events")
    ap.add_argument("--players-file", default=os.path.join(_bootstrap.ROOT, PLAYERS_FILE),
                    help="our YOLO tracks (kaggle/kaggle_player_tracking.ipynb)")
    ap.add_argument("--team-mode", choices=("local", "track"), default="local")
    ap.add_argument("--ball-file", default=os.path.join(_bootstrap.ROOT, BALL_FILE),
                    help="fixed ball of scripts/reconstruct_ball.py")
    ap.add_argument("--keep-max-gap", action="store_true",
                    help="keep bridging ball gaps (--max-gap) even on the fixed ball")
    ap.add_argument("--compare-events", default=None,
                    help="older events JSON to diff against (writes events_diff.csv)")'''
assert s.count(old) == 1; s = s.replace(old, new)

# main body
old = '''    meta_path, raw_path = require(["atd_meta", "atd_raw"])
    teams = load_teams(meta_path)
    team, names = teams.team, list(teams.names)
    ctx = Context(raw_path, team, names, load_tags(), args)
'''
new = '''    tr = load(players_file=args.players_file, team_mode=args.team_mode, ball_file=args.ball_file)
    team, names = tr.teams.team, list(tr.teams.names)
    print("players: {}\\nball: {}".format(tr.stats["players"], tr.stats["ball"]))
    fixed_ball = tr.stats["ball"].startswith("fixed ball")
    if fixed_ball and not args.keep_max_gap and args.max_gap:
        print("ball gap bridge off (--max-gap {} -> 0): the fixed ball is already filled".format(args.max_gap))
        args.max_gap = 0
    observed = ball_observed(args.ball_file, tr.n, tr.bx)
    ctx = Context(None, team, names, load_tags(), args, arrays=(tr.px, tr.py, tr.bx, tr.by),
                  kickoff=HALF_KICKOFF)
'''
assert s.count(old) == 1; s = s.replace(old, new)
old = "        found = SegmentAnalyser(ctx, half, lo, hi).run()"
new = "        found = ObservedAnalyser(ctx, half, lo, hi, observed).run()"
assert s.count(old) == 1; s = s.replace(old, new)

old = '''    warnings = ["Automatic, uncorrected tracking: events are evidence, not ground truth.",'''
new = '''    warnings = ["Automatic tracking (our YOLO players, fixed ball): events are evidence, not ground truth.",
                "Player slots are tracklets (one slot holds several YOLO tracks in turn), not people.",'''
assert s.count(old) == 1; s = s.replace(old, new)
old = '''        "windows": windows, "attack_direction": ctx.direction_info,'''
new = '''        "windows": windows, "attack_direction": ctx.direction_info,
        "inputs": {"players": tr.stats["players"], "team_mode": tr.stats["team_mode"],
                   "players_clean": tr.stats.get("clean"), "ball": tr.stats["ball"],
                   "detector_max_gap": args.max_gap, "seen_share": "tracker-observed ball frames only"},'''
assert s.count(old) == 1; s = s.replace(old, new)

old = '''    print("  " + ", ".join("{} {}".format(v, k) for k, v in sorted(counts.items())))
'''
new = '''    print("  " + ", ".join("{} {}".format(v, k) for k, v in sorted(counts.items())))

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
        print("\\nchanges vs {}: {}".format(args.compare_events, "  ".join(
            "{} {}".format(k, v) for k, v in sorted(st.items()))))
        for k in sorted(set(oc) | set(counts)):
            print("  {:<22} {:>6} {:>6} {:>+7}".format(k, oc.get(k, 0), counts.get(k, 0),
                                                     counts.get(k, 0) - oc.get(k, 0)))
        print("-> {}".format(diff_path))
'''
assert s.count(old) == 1; s = s.replace(old, new)
# counts() helper from fix_events clashes with the local variable `counts`: rename it
s = s.replace("def counts(events):", "def event_counts(events):")
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("ok")
