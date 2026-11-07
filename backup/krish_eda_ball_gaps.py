"""EDA: where the ball is missing in the tactical ATD, and how long those gaps last.

The raw ATD line is "<frame_id>:<slot>;<slot>;...:<ball>" -- the ball is the segment after
the LAST colon. A slot is "x,y,speed"; the ball is present when its segment is not "NaN,NaN,NaN".

This script streams the raw file, so it needs no kloppy and no per-frame parsing of 46 slots.
It answers two questions:

  1. Where are the largest gaps?  -- the longest runs of consecutive frames with no ball.
  2. How often does the ball go missing for longer than T?  -- a threshold sweep, so you can
     pick a window that separates "occlusion" from "the ball is genuinely not tracked".

Run:
    python eda/eda_ball_gaps.py
    python eda/eda_ball_gaps.py --thresholds 5,15,30,60,125,250,500
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAME, DEFAULT_SOURCE, require  # noqa: E402

# A slot is missing when every component is NaN. The ball is the segment after the last colon.
BALL_MISSING = "NaN,NaN,NaN"


def iter_ball_present(raw_path):
    """Yield (frame_id, ball_present) for every record in the raw ATD, one line at a time."""
    with open(raw_path) as handle:
        for line in handle:
            frame_id = int(line[: line.index(":")])
            ball = line.rsplit(":", 1)[-1].strip()
            yield frame_id, ball != BALL_MISSING


def find_gaps(present):
    """From an iterable of (frame_id, present), return the missing runs as
    [(start_frame_id, end_frame_id, length_in_frames), ...] sorted longest first."""
    gaps = []
    in_gap = False
    start = None
    for frame_id, present in present:
        if not present and not in_gap:
            in_gap = True
            start = frame_id
        elif present and in_gap:
            in_gap = False
            gaps.append((start, frame_id - 1, frame_id - start))
    if in_gap:
        gaps.append((start, frame_id, frame_id - start + 1))
    gaps.sort(key=lambda g: (-g[2], g[0]))
    return gaps


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--game", default=GAME)
    parser.add_argument("--source", default=DEFAULT_SOURCE,
                        help="tactical (default) or broadcast")
    parser.add_argument("--thresholds", default="5,15,30,60,125,250,500",
                        help="comma-separated gap lengths to count")
    parser.add_argument("--top", type=int, default=25,
                        help="how many of the longest gaps to print")
    args = parser.parse_args()

    raw_path = require(["atd_raw"], args.game, None, args.source)[0]
    thresholds = [int(t) for t in args.thresholds.split(",")]

    present = list(iter_ball_present(raw_path))
    total = len(present)
    n_ball = sum(1 for _, p in present if p)
    gaps = find_gaps(present)

    print("source: {}   game: {}".format(args.source, args.game))
    print("raw ATD: {}".format(raw_path))
    print("frames: {} total, ball present on {} ({:.1f}%), missing on {} ({:.1f}%)".format(
        total, n_ball, 100.0 * n_ball / total, total - n_ball, 100.0 * (total - n_ball) / total))
    print("missing runs: {} gaps".format(len(gaps)))
    print()

    print("longest {} gaps (start, end, length in frames = length/25 seconds):".format(args.top))
    print("  {:>8}  {:>8}  {:>8}  {:>8}".format("start", "end", "frames", "seconds"))
    for start, end, length in gaps[:args.top]:
        print("  {:>8}  {:>8}  {:>8}  {:>8.1f}".format(start, end, length, length / 25.0))
    print()

    print("how many gaps exceed each threshold:")
    print("  {:>8}  {:>8}  {:>10}".format("threshold", "n gaps", "frames lost"))
    for t in thresholds:
        n = sum(1 for _, _, length in gaps if length >= t)
        lost = sum(length for _, _, length in gaps if length >= t)
        print("  {:>8}  {:>8}  {:>10}".format(t, n, lost))
    print()

    # Where do the big gaps sit relative to the halves?  A gap that starts before the first
    # graded frame or ends after the last one is not something a system has to recover from.
    from games import half_video_blocks
    blocks = half_video_blocks(args.game)
    graded = []
    for half, first, last in blocks:
        for start, end, length in gaps:
            if end >= first and start <= last:
                graded.append((half, start, end, length))
    print("gaps overlapping graded play (half, start, end, length):")
    for half, start, end, length in sorted(graded, key=lambda g: (-g[3], g[1])):
        print("  half {}  {:>8}..{:>8}  {:>6} frames  {:>7.1f}s".format(
            half, start, end, length, length / 25.0))
    if not graded:
        print("  none")


if __name__ == "__main__":
    main()