"""Event test clips: one folder per event type, each clip centred on one detected event.

    python scripts/detect_events.py --full               # once: outputs/events/events_full.json
    python viz/test_event_clips.py --dry-run             # list what would be rendered, write index.csv
    python viz/test_event_clips.py                       # up to 8 clips per type, spread over the match
    python viz/test_event_clips.py --types pass,interception --per-type 20
    python viz/test_event_clips.py --types restart --per-type 0          # every restart
    python viz/test_event_clips.py --pick low_conf       # the least confident detections of each type
    python viz/test_event_clips.py --only E00412         # one event by id
    python viz/test_event_clips.py --focus-only          # draw only the clip's own event type

WHAT YOU GET. outputs/test_events/ (git-ignored: clips and events are derived match data, never
commit or upload them, RULES.md):

    <type>/                    pass, carry, hold, duel, tackle, interception, possession_change,
                               ball_out_of_play, dead_ball, goal, shot, ...
    <type>/<subtype>/          for restart (throw_in, corner_kick, goal_kick, free_kick, kick_off,
                               unknown, ...) and foul_candidate
        <frame>_<id>_<team>[_<outcome>].mp4
    index.csv                  one row per selected event: type, subtype, id, frames, time, team,
                               slot, confidence, outcome, clip file
    summary.json               events per type in the file and clips selected per type

Every clip is viz/render_match.py --event-id <id>: the tactical video with the tracking and events
projected (left), the top-down pitch with the events drawn (right), the event banner, the info bar
(event-implied possession vs the public ground truth where it exists) and the event timeline. The
clip runs from --pad s before the event starts to --pad s after it ends (default 3 s = 75 frames,
like viz/test_ball_clips.py), capped at --max-seconds.

SELECTION (--pick, per type or subtype, --per-type clips each, 0 = all):
    spread     evenly spaced over the match, so both halves and different phases show (default)
    first      the first ones in time
    low_conf   lowest detector confidence first: where the detector is least sure
    high_conf  highest confidence first: what a "clean" detection looks like
    random     random, reproducible with --seed

RULES.md: watching clips to understand and debug the event detector is fine. Do not turn what you
see into per-event verdicts, labels, or thresholds tuned for this match; change a threshold only
with a bench score on other games.
"""
import argparse
import csv
import json
import os
import random
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from games import GAME, GAMES  # noqa: E402

FPS = GAMES[GAME]["fps"]
EVENTS = os.path.join(ROOT, "outputs", "events", "events_full.json")
OUT_DIR = os.path.join(ROOT, "outputs", "test_events")
RENDER = os.path.join(HERE, "render_match.py")
SPLIT_SUBTYPES = ("restart", "foul_candidate")
PICKS = ("spread", "first", "low_conf", "high_conf", "random")


def clock(frame):
    s = frame / float(FPS)
    return "{:02d}:{:05.2f}".format(int(s // 60), s % 60)


def group_of(e):
    """Folder for an event: its type, plus its subtype for restarts and foul candidates."""
    if e["type"] in SPLIT_SUBTYPES:
        return os.path.join(e["type"], e.get("subtype") or "unknown")
    return e["type"]


def outcome_of(e):
    d = e.get("details") or {}
    for key in ("outcome", "winner_team", "reason", "exit_side", "kind"):
        if d.get(key) not in (None, ""):
            return "{}".format(d[key])
    return ""


def select(events, per_type=8, pick="spread", seed=0):
    """{group: [events]} with at most per_type events per group (0 = all), in time order."""
    groups = {}
    for e in events:
        groups.setdefault(group_of(e), []).append(e)
    rng = random.Random(seed)
    out = {}
    for g, evs in sorted(groups.items()):
        evs = sorted(evs, key=lambda e: e["frame_start"])
        k = len(evs) if per_type <= 0 else min(per_type, len(evs))
        if pick == "spread":
            idx = sorted({int(round(i * (len(evs) - 1) / max(1, k - 1))) for i in range(k)}) if k > 1 else [0]
            chosen = [evs[i] for i in idx]
        elif pick == "first":
            chosen = evs[:k]
        elif pick in ("low_conf", "high_conf"):
            ranked = sorted(evs, key=lambda e: (e.get("confidence") or 0.0), reverse=pick == "high_conf")
            chosen = sorted(ranked[:k], key=lambda e: e["frame_start"])
        elif pick == "random":
            chosen = sorted(rng.sample(evs, k), key=lambda e: e["frame_start"])
        else:
            raise ValueError(pick)
        out[g] = chosen
    return out


def clip_name(e):
    parts = ["{:07d}".format(e["frame_start"]), e.get("id", "E?"), e.get("team") or "-"]
    oc = outcome_of(e).replace(" ", "-").replace("/", "-")
    if oc:
        parts.append(oc)
    return "_".join(parts) + ".mp4"


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--events", default=EVENTS, help="events JSON (default outputs/events/events_full.json)")
    ap.add_argument("--types", default="", help="comma list of event types (default: every type in the file)")
    ap.add_argument("--per-type", type=int, default=8, help="clips per type / subtype (0 = all)")
    ap.add_argument("--pick", choices=PICKS, default="spread", help="which events to clip (default spread)")
    ap.add_argument("--seed", type=int, default=0, help="for --pick random")
    ap.add_argument("--min-confidence", type=float, default=0.0, help="ignore events below this confidence")
    ap.add_argument("--half", type=int, choices=(1, 2), help="only this half")
    ap.add_argument("--only", default="", help="only this event id (e.g. E00412)")
    ap.add_argument("--pad", type=float, default=3.0, help="seconds before and after the event (default 3)")
    ap.add_argument("--max-seconds", type=float, default=20.0, help="longest clip; longer events are cut")
    ap.add_argument("--focus-only", action="store_true", help="draw only the clip's own event type")
    ap.add_argument("--sidecars", action="store_true", help="keep render_match's _events.csv / _report.json")
    ap.add_argument("--no-video", action="store_true", help="pitch panel only (fast)")
    ap.add_argument("--extra-args", default="", help="more options forwarded to viz/render_match.py")
    ap.add_argument("--out-dir", default=OUT_DIR, help="default outputs/test_events")
    ap.add_argument("--overwrite", action="store_true", help="re-render clips that already exist")
    ap.add_argument("--dry-run", action="store_true", help="list the clips and write index.csv only")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not os.path.isfile(args.events):
        sys.exit("no events file at {}\nrun first:  python scripts/detect_events.py --full".format(
            os.path.relpath(args.events, ROOT)))
    with open(args.events) as handle:
        doc = json.load(handle)
    events = doc["events"] if isinstance(doc, dict) else doc
    all_counts = {}
    for e in events:
        all_counts[group_of(e)] = all_counts.get(group_of(e), 0) + 1

    types = [t.strip() for t in args.types.split(",") if t.strip()]
    present = sorted(set(e["type"] for e in events))
    unknown = [t for t in types if t not in present]
    if unknown:
        print("note: no events of type {} in the file (types present: {})".format(
            ", ".join(unknown), ", ".join(present)))
    pool = [e for e in events
            if (not types or e["type"] in types)
            and (e.get("confidence") or 0.0) >= args.min_confidence
            and (args.half is None or e.get("period") == args.half)
            and (not args.only or e.get("id") == args.only)]
    chosen = select(pool, 0 if args.only else args.per_type, args.pick, args.seed)
    n_clips = sum(len(v) for v in chosen.values())
    print("{} events in {} ({} types); {} clips selected ({} per type, pick {})".format(
        len(events), os.path.relpath(args.events, ROOT), len(present), n_clips,
        args.per_type or "all", args.pick))
    for g in sorted(all_counts):
        print("  {:<34} {:>5} events  {:>3} clips".format(g, all_counts[g], len(chosen.get(g, []))))

    os.makedirs(args.out_dir, exist_ok=True)
    rows = []
    for g, evs in sorted(chosen.items()):
        for e in evs:
            rows.append(dict(group=g.replace(os.sep, "/"), type=e["type"], subtype=e.get("subtype") or "",
                             id=e.get("id"), frame_start=e["frame_start"], frame_end=e["frame_end"],
                             time=clock(e["frame_start"]), half=e.get("period"), team=e.get("team") or "",
                             slot=e.get("player_slot") or "", confidence=e.get("confidence"),
                             outcome=outcome_of(e), file=os.path.join(g, clip_name(e)).replace(os.sep, "/")))
    index = os.path.join(args.out_dir, "index.csv")
    if not args.only:                          # one-event runs leave the full index alone
        with open(index, "w", newline="") as handle:
            w = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["file"])
            w.writeheader()
            w.writerows(rows)
        with open(os.path.join(args.out_dir, "summary.json"), "w") as handle:
            json.dump({"events_file": os.path.relpath(args.events, ROOT), "events_per_group": all_counts,
                       "clips_per_group": {g: len(v) for g, v in chosen.items()},
                       "pick": args.pick, "per_type": args.per_type, "pad_s": args.pad,
                       "missing_types_in_schema": sorted(set(_schema_types()) - set(present))},
                      handle, indent=2)
        print("wrote {} and summary.json".format(os.path.relpath(index, ROOT)))
    if args.dry_run:
        for r in rows:
            print("  {:<30} {:<8} frame {:>7}  {}  {:<2} conf {}  {}".format(
                r["group"], r["id"], r["frame_start"], r["time"], r["team"], r["confidence"], r["outcome"]))
        return

    extra = args.extra_args.split() if args.extra_args else []
    done = failed = skipped = 0
    t_all = time.time()
    for k, r in enumerate(rows, 1):
        out = os.path.join(args.out_dir, r["file"])
        if os.path.isfile(out) and not args.overwrite:
            skipped += 1
            continue
        os.makedirs(os.path.dirname(out), exist_ok=True)
        length_s = (r["frame_end"] - r["frame_start"]) / float(FPS) + 2 * args.pad
        cmd = [sys.executable, RENDER, "--events", args.events, "--event-id", r["id"],
               "--pad", "{:.2f}".format(args.pad), "--out", out]
        if length_s > args.max_seconds:
            cmd += ["--duration", "{:.2f}".format(args.max_seconds)]
        if args.focus_only:
            cmd += ["--types", r["type"]]
        if not args.sidecars:
            cmd += ["--no-report"]
        if args.no_video:
            cmd += ["--no-video"]
        cmd += extra
        t = time.time()
        print("[{}/{}] {}  ({:.1f} s of video)".format(k, len(rows), r["file"], min(length_s, args.max_seconds)),
              flush=True)
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if proc.returncode != 0 or not os.path.isfile(out):
            failed += 1
            tail = "\n".join(proc.stdout.strip().splitlines()[-3:])
            print("    FAILED:\n    " + tail.replace("\n", "\n    "), flush=True)
        else:
            done += 1
            print("    {:.1f} s".format(time.time() - t), flush=True)
    print("rendered {}, skipped (already there) {}, failed {} in {:.0f} s -> {}".format(
        done, skipped, failed, time.time() - t_all, os.path.relpath(args.out_dir, ROOT)))


def _schema_types():
    try:
        from possession.events.schema import EVENT_TYPES
        return list(EVENT_TYPES)
    except Exception:
        return []


if __name__ == "__main__":
    main()
