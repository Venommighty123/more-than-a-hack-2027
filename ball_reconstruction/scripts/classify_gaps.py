"""Classify every ball gap of the match (kick-off / tagged set piece / untagged dead / live) and
write one row per gap, with who had the ball either side.

    python scripts/classify_gaps.py                         # -> outputs/gaps/gaps_classified.csv
    python scripts/classify_gaps.py --no-tags               # without Smart Tagging (like the bench)
    python scripts/classify_gaps.py --set near_line_m=3 --set dead_speed_ms=2.0   # sensitivity

Replaces Ashwyn's test_scripts/gap_classification.py: possession/gaps/{find,classify}.py do the
work, with labeller v1's control and player speed as the signals. The CSV holds frames and teams,
no positions; it is still derived match data (outputs/ is git-ignored). The point of the table is
the SIZE of each class -- watch how it moves with --set before trusting one number.
"""
import argparse
import csv
import os

try:
    import _bootstrap  # noqa: F401  (python scripts/x.py)
except ImportError:
    from scripts import _bootstrap  # noqa: F401  (python -m scripts.x)

import numpy as np

from possession.gaps.classify import GapClassConfig, class_table, classify_gaps
from possession.gaps.find import bracket, find_gaps
from possession.io import graded_blocks, load, load_tags
from possession.labellers import v1

try:
    from scripts.run import play_blocks
except ImportError:
    from run import play_blocks

ROOT = _bootstrap.ROOT
OUT = os.path.join(ROOT, "outputs", "gaps", "gaps_classified.csv")
KEYS = ("half", "start_frame", "end_frame", "length_frames", "seconds", "cls", "near_line",
        "vanish_line_m", "reappear_line_m", "player_speed_ms", "team_before", "team_after", "flip",
        "set_piece", "set_piece_team", "goal", "phase_team", "dist_m", "implied_speed_ms")


def _value(text):
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--no-tags", action="store_true")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a GapClassConfig field")
    args = ap.parse_args()
    cfg = GapClassConfig.from_dict({k: _value(v) for k, v in (s.split("=", 1) for s in args.set)})

    tr = load(cache=True)
    blocks, kickoff2 = play_blocks()
    out = v1.run(tr.px, tr.py, tr.ball, tr.teams.team, blocks, fps=tr.fps, pitch=tr.pitch)
    track = out["ball"]
    tags = None if args.no_tags else load_tags(tr.teams)
    gaps = bracket(find_gaps(track.present, graded_blocks()), track.x, track.y, tr.fps)
    gaps = classify_gaps(gaps, track.x, track.y, out["control"].ctrl, out["mean_speed"], tags=tags,
                         kickoffs={1: blocks[0][1], 2: kickoff2}, cfg=cfg, fps=tr.fps,
                         pitch=tr.pitch)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=KEYS, extrasaction="ignore")
        w.writeheader()
        for g in gaps:
            w.writerow(g.to_row(tr.lo, tr.fps))
    graded = int(sum(b - a + 1 for _, a, b in graded_blocks()))
    print("{} gaps of at least {} s -> {}".format(len(gaps), cfg.min_gap_s,
                                                  os.path.relpath(args.out, ROOT)))
    print("{:<14} {:>5} {:>9} {:>9} {:>7}".format("class", "gaps", "seconds", "graded%", "flip%"))
    for r in class_table(gaps, graded, tr.fps):
        print("{cls:<14} {gaps:>5} {seconds:>9} {graded_pct:>9} {flip}".format(
            flip="{:>7}".format(r["flip_pct"]) if r["flip_pct"] is not None else "      -", **r))
    live = [g for g in gaps if g.info["cls"] == "live"]
    if live:
        print("live gaps: median {:.1f} s, longest {:.1f} s".format(
            np.median([g.seconds(tr.fps) for g in live]), max(g.seconds(tr.fps) for g in live)))


if __name__ == "__main__":
    main()
