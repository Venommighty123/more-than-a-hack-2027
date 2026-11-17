"""Render one clip per long ball-tracking gap, with viz/render_match.py.

    python viz/render_gaps.py                              # gaps >= 60 frames in the graded halves
    python viz/render_gaps.py --min-length 125 --max-clips 10
    python viz/render_gaps.py --pad 5 --extra-args "--no-tags --trail 80"

For understanding what happens while the tracker is blind (a replay, the ball behind the goal, a
crowded box). From Krish's render_gap_clips.py, on possession.gaps.find (video frames, graded
halves only, so no frame_id conversion is needed). Clips are named gap_<start>_<length>.mp4.

RULES.md: watching clips to understand the data is fine; turning what you see into labels, rules
or constants for this match is not (no gap-by-gap verdicts).
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import possession.io  # noqa: E402,F401  (repository root on sys.path)
from games import GAME, GAMES  # noqa: E402
from possession.gaps.find import find_gaps  # noqa: E402
from possession.io import graded_blocks, load_ball  # noqa: E402

FPS = GAMES[GAME]["fps"]
VISUALIZE = os.path.join(HERE, "render_match.py")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-length", type=int, default=60, help="frames (default 60 = 2.4 s)")
    ap.add_argument("--pad", type=float, default=2.0, help="seconds before and after each gap")
    ap.add_argument("--max-clips", type=int, default=0, help="stop after this many (0 = all)")
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "outputs", "videos", "gaps"))
    ap.add_argument("--extra-args", default="", help="forwarded to viz/render_match.py")
    ap.add_argument("--dry-run", action="store_true", help="print the commands only")
    args = ap.parse_args()

    bx, _, _ = load_ball()
    import numpy as np  # noqa: E402
    gaps = [g for g in find_gaps(np.isfinite(bx), graded_blocks()) if g.length >= args.min_length]
    if args.max_clips:
        gaps = gaps[:args.max_clips]
    if not gaps:
        print("no gaps of at least {} frames".format(args.min_length))
        return
    os.makedirs(args.out_dir, exist_ok=True)
    pad = int(round(args.pad * FPS))
    extra = args.extra_args.split() if args.extra_args else []
    print("{} gap(s) to render (min {} frames)\n".format(len(gaps), args.min_length))
    for g in gaps:
        v0 = max(0, g.start - pad)
        duration = (g.length + 2 * pad) / float(FPS)
        out = os.path.join(args.out_dir, "gap_{:07d}_{:05d}.mp4".format(g.start, g.length))
        if os.path.isfile(out):
            continue
        cmd = [sys.executable, VISUALIZE, "--start-frame", str(v0), "--duration",
               "{:.2f}".format(duration), "--out", out] + extra
        print("  half {} gap {}..{} ({:.1f} s)  $ {}".format(g.half, g.start, g.end - 1,
                                                          g.seconds(FPS), " ".join(cmd)))
        if not args.dry_run and subprocess.call(cmd) != 0:
            print("    FAILED")


if __name__ == "__main__":
    main()
