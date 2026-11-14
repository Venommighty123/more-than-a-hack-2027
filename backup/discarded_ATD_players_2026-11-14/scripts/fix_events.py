"""Re-detect the match events on the RECONSTRUCTED ball and FIXED player tracks, and report what
changed against the events detected on the raw ATD.

    python scripts/reconstruct_ball.py                  # once: outputs/ball/ball_reconstructed.npz
    python scripts/fix_events.py                        # whole match -> outputs/events/events_full_fixed.json
    python scripts/fix_events.py --no-yolo              # fix players from ATD alone (no gap bridging)
    python scripts/fix_events.py --start 11185 --seconds 300 --validate

INPUTS
  ball     outputs/ball/ball_reconstructed.npz (scripts/reconstruct_ball.py): spikes removed, gaps up
           to 1 s filled with Hermite, implausible gaps (tracker re-detections) left empty.
  players  ATD, cleaned by possession/preprocess/fix_players.py: every slot cut at its jumps
           (> 12 m/s over 0.2 s, as viz/track_jumps.py), gaps up to 2 s bridged from our YOLO
           tracks (data/kaggle_outputs/final_tracks_full.parquet) when the same YOLO id is the slot's
           person on both sides, gaps up to 0.5 s interpolated when physically plausible.
  tags     Smart Tagging, as scripts/detect_events.py.
  The fixed players are cached in outputs/cache/atd_players_fixed.npz for other steps (labellers).

SAME DETECTOR, two deliberate differences from scripts/detect_events.py
  * --max-gap is 0: the detector would otherwise linearly bridge ball gaps of <= 12 frames again,
    including the re-detection gaps the reconstruction refused to fill (the fake 300 km/h kicks).
  * "seen" (event confidence, tracking_gap events) counts only frames the tracker really saw the
    ball, never reconstructed ones, so a pass across a filled gap is not more confident than before.
All other thresholds are the detector's defaults (untuned; RULES.md: never tune on BAR-ATH).

OUTPUT (outputs/events/, git-ignored; derived match data: never upload to a cloud AI tool)
  events_full_fixed.json   same format and schema as events_full.json, with meta.inputs and
                           meta.player_fix describing the fix
  events_fix_diff.csv      every event, old vs new: kept / moved / added / removed
  printed                  counts per type before -> after
"""
import argparse
import csv
import json
import os
import sys
import time

try:
    import _bootstrap  # noqa: F401  (python scripts/x.py)
except ImportError:
    from scripts import _bootstrap  # noqa: F401  (python -m scripts.x)

import numpy as np

from games import GAME, half_video_blocks
from possession.events.detect import (FPS, HALF_KICKOFF, HOME, PITCH_L, PITCH_W, Context,
                                      SegmentAnalyser, clock, load_tags, renumber)
from possession.events.schema import SCHEMA_VERSION, build_schema
from possession.io import load
from possession.preprocess.fix_players import FixConfig, fix_players, save_fixed
from possession.preprocess.reconstruct import DEFAULT_FILE, load_ball_file
from scripts.detect_events import parse_args as detector_args

ROOT = _bootstrap.ROOT
OUT_DIR = os.path.join(ROOT, "outputs", "events")
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


# -- inputs ----------------------------------------------------------------------------------------

def load_ball(path, n):
    """Reconstructed ball on rows 0..n-1 (= video frames) plus the tracker-observed mask."""
    if not os.path.isfile(path):
        sys.exit("{} not found: run python scripts/reconstruct_ball.py first".format(path))
    z = load_ball_file(path)
    bx, by, obs = np.full(n, np.nan), np.full(n, np.nan), np.zeros(n, dtype=bool)
    lo = z["lo"]
    a, b = max(lo, 0), min(lo + len(z["x"]), n)
    bx[a:b], by[a:b] = z["x"][a - lo:b - lo], z["y"][a - lo:b - lo]
    obs[a:b] = z["observed"][a - lo:b - lo].astype(bool) & ~z["removed"][a - lo:b - lo].astype(bool)
    return bx, by, obs, z


def load_yolo(path, lo, hi):
    import pandas as pd
    df = pd.read_parquet(path, filters=[("frame", ">=", lo), ("frame", "<=", hi)],
                         columns=["frame", "track_id", "role", "team", "x_m", "y_m"])
    df = df[(df.role != "referee") & np.isfinite(df.x_m) & np.isfinite(df.y_m)]
    return pd.DataFrame({"frame": df.frame.to_numpy(np.int64), "id": df.track_id.to_numpy(np.int64),
                         "team": df.team.map({"H": 0, "A": 1}).fillna(-1).to_numpy(np.int64),
                         "x_m": df.x_m.to_numpy(float), "y_m": df.y_m.to_numpy(float)})


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


def counts(events):
    c = {}
    for e in events:
        c[e["type"]] = c.get(e["type"], 0) + 1
    return c


# -- main ------------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", type=int, help="first video frame of a window (instead of the full match)")
    ap.add_argument("--seconds", type=float, default=300.0)
    ap.add_argument("--ball-file", default=os.path.join(ROOT, DEFAULT_FILE))
    ap.add_argument("--players-file", default=os.path.join(ROOT, "data", "kaggle_outputs", "final_tracks_full.parquet"))
    ap.add_argument("--no-yolo", action="store_true", help="do not bridge ATD gaps with our YOLO tracks")
    ap.add_argument("--old-events", default=os.path.join(OUT_DIR, "events_full.json"),
                    help="events on the raw ATD, to compare with (scripts/detect_events.py --full)")
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--include-gaps", action="store_true")
    ap.add_argument("--validate", action="store_true")
    for k, v in FixConfig().to_dict().items():
        if k != "fps":
            ap.add_argument("--fix-" + k.replace("_", "-"), type=type(v), default=v, dest="fix_" + k)
    args = ap.parse_args()
    t0 = time.time()

    # Players: whole recording so that row k = video frame k (what Context expects).
    print("loading ATD ...", flush=True)
    tr = load(cache=True)
    if tr.lo != 0:
        sys.exit("expected the ATD from frame 0, got lo={}".format(tr.lo))
    n = tr.n
    team, names = tr.teams.team, list(tr.teams.names)
    blocks = half_video_blocks(GAME)
    if args.start is not None:
        lo_req, hi_req = args.start, args.start + int(round(args.seconds * FPS)) - 1
    else:
        lo_req, hi_req = blocks[0][1], blocks[-1][2]
    segments = [(h, max(lo_req, f), min(hi_req, l)) for h, f, l in blocks if max(lo_req, f) <= min(hi_req, l)]
    if not segments:
        sys.exit("window {}..{} is outside both halves".format(lo_req, hi_req))

    yolo = None
    if not args.no_yolo:
        if os.path.isfile(args.players_file):
            print("loading our YOLO tracks ...", flush=True)
            yolo = load_yolo(args.players_file, 0, n - 1)
        else:
            print("note: {} not found, fixing players from ATD alone".format(args.players_file))
    cfg = FixConfig(**{k[4:]: v for k, v in vars(args).items() if k.startswith("fix_")}, fps=FPS)
    print("fixing players ...", flush=True)
    px, py, source, fix_report = fix_players(tr.px, tr.py, team, yolo=yolo, cfg=cfg, lo=0)
    fix_report["yolo_file"] = os.path.relpath(args.players_file, ROOT) if yolo is not None else None
    save_fixed(os.path.join(ROOT, "outputs", "cache", "atd_players_fixed.npz"), px, py, source, 0, fix_report)
    print("  jumps cut {jumps_cut}, gaps bridged from YOLO {gaps_bridged_from_yolo} ({frames_bridged} frames), "
          "interpolated {gaps_interpolated} ({frames_interpolated} frames)\n"
          "  jumps_per_1000 {jumps_per_1000_before:.2f} -> {jumps_per_1000_after:.2f}".format(**fix_report))

    bx, by, observed, ball_meta = load_ball(args.ball_file, n)
    print("ball: {} (method {}, max gap {}), present on {:.1f}% of frames, observed {:.1f}%".format(
        os.path.relpath(args.ball_file, ROOT), ball_meta["method"], ball_meta["max_gap"],
        100 * np.isfinite(bx).mean(), 100 * observed.mean()))

    dargs = detector_args([])
    dargs.max_gap = 0                                       # the reconstruction already filled what is safe
    dargs.include_gaps = args.include_gaps
    ctx = Context(None, team, names, load_tags(), dargs, arrays=(px, py, bx, by), kickoff=HALF_KICKOFF)
    events, windows = [], []
    for half, lo, hi in segments:
        t1 = time.time()
        found = ObservedAnalyser(ctx, half, lo, hi, observed).run()
        events.extend(found)
        windows.append({"period": half, "frame_start": lo, "frame_end": hi, "time_start": clock(lo),
                        "time_end": clock(hi), "events": len(found)})
        print("half {}: frames {}..{} -> {} events ({:.0f}s)".format(half, lo, hi, len(found), time.time() - t1))
    events = renumber(events)

    new_counts = counts(events)
    warnings = ["Automatic tracking, reconstructed ball and algorithmically fixed players: events are "
                "evidence, not ground truth.",
                "Player slots are tracklets: after a cut jump the slot continues on another person.",
                "foul_candidate is a Free Kick tag + preceding duel/tackle; offside and handball look the same.",
                "Derived from the match data: do not publish or upload to any cloud AI tool (RULES.md)."]
    if len(ctx.sign) == 2 and ctx.sign[1][HOME] == ctx.sign[2][HOME]:
        warnings.append("Home attacks the same way in both halves: the kick-off formation check looks wrong.")
    doc = {"meta": {
        "schema_version": SCHEMA_VERSION, "game": GAME, "fps": FPS, "pitch_m": [PITCH_L, PITCH_W],
        "teams": {"H": names[0], "A": names[1]}, "windows": windows, "attack_direction": ctx.direction_info,
        "inputs": {"ball": {"file": os.path.relpath(args.ball_file, ROOT), "method": ball_meta["method"],
                            "max_gap": ball_meta["max_gap"], "max_speed_ms": ball_meta["max_speed_ms"]},
                   "players": "ATD fixed by possession/preprocess/fix_players.py",
                   "detector_max_gap": 0, "seen_share": "tracker-observed ball frames only"},
        "player_fix": fix_report,
        "thresholds": {k: v for k, v in vars(dargs).items() if k != "out_dir" and not isinstance(v, bool)},
        "event_counts": new_counts, "warnings": warnings}, "events": events}

    os.makedirs(args.out_dir, exist_ok=True)
    tag = "full" if args.start is None else "{}_{}".format(lo_req, hi_req)
    out_path = os.path.join(args.out_dir, "events_{}_fixed.json".format(tag))
    with open(out_path, "w") as h:
        json.dump(doc, h, indent=1)
    print("wrote {}  ({} events, {:.0f}s total)".format(os.path.relpath(out_path, ROOT), len(events), time.time() - t0))

    if os.path.isfile(args.old_events):
        with open(args.old_events) as h:
            old = json.load(h)["events"]
        old = [e for e in old if lo_req <= e["frame_start"] <= hi_req]
        rows = diff_events(old, events)
        diff_path = os.path.join(args.out_dir, "events_fix_diff.csv")
        write_diff(diff_path, rows)
        oc = counts(old)
        st = {}
        for s, _, _ in rows:
            st[s] = st.get(s, 0) + 1
        print("\nchanges vs {}: {}".format(os.path.relpath(args.old_events, ROOT),
                                          "  ".join("{} {}".format(k, v) for k, v in sorted(st.items()))))
        print("  {:<22} {:>6} {:>6} {:>7}".format("type", "old", "new", "change"))
        for k in sorted(set(oc) | set(new_counts)):
            a, b = oc.get(k, 0), new_counts.get(k, 0)
            print("  {:<22} {:>6} {:>6} {:>+7}".format(k, a, b, b - a))
        print("-> {}".format(os.path.relpath(diff_path, ROOT)))
    else:
        print("note: {} not found, no comparison (run scripts/detect_events.py --full first)".format(args.old_events))

    if args.validate:
        try:
            import jsonschema
        except ImportError:
            print("note: --validate needs `pip install jsonschema`")
            return
        errors = list(jsonschema.Draft202012Validator(build_schema()).iter_errors(doc))
        print("schema validation: {}".format("OK" if not errors else "{} errors".format(len(errors))))
        for e in errors[:5]:
            print("  {}: {}".format("/".join(str(p) for p in e.absolute_path), e.message[:160]))
        if errors:
            sys.exit(1)


if __name__ == "__main__":
    main()
