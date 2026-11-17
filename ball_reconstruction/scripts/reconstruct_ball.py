"""Clean and reconstruct the BAR-ATH ball once, and store it for every later step.

    python scripts/reconstruct_ball.py                          # -> outputs/ball/ball_reconstructed.npz
    python scripts/reconstruct_ball.py --max-gap 25 --method hermite
    python scripts/reconstruct_ball.py --csv                    # also a frame,x,y,observed,filled CSV

What it does (possession/preprocess/reconstruct.prepare_ball):
  1. spikes out: a detection that jumps more than 2 m per frame away AND back is a false blob
  2. every gap of at most --max-gap frames is filled with --method (default Hermite up to 25 frames
     = 1 s at 25 fps; the old pipeline bridged only 6 frames = 0.24 s, linearly)
  3. a gap whose two ends are further apart than a 35 m/s ball could travel is NOT filled: on
     BAR-ATH most short gaps are the tracker jumping to another detection, not a lost ball
  4. longer gaps stay NaN; the dead-ball and gap rules decide what happens inside them

Who reads the file: scripts/run.py (labeller v5 uses it as its ball source when the settings match)
and scripts/detect_flights.py. Both fall back to computing the same thing in memory when the file
is missing, so the bench (which has no file) runs the identical code path.
The file is derived match data: outputs/ is git-ignored, keep it that way.
"""
import argparse
import os

try:
    import _bootstrap  # noqa: F401  (python scripts/x.py)
except ImportError:
    from scripts import _bootstrap  # noqa: F401  (python -m scripts.x)

import numpy as np

from possession.io import graded_mask, load
from possession.preprocess.reconstruct import DEFAULT_FILE, METHODS, prepare_ball, save_ball_file
from possession.utils.frames import runs

ROOT = _bootstrap.ROOT


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=os.path.join(ROOT, DEFAULT_FILE))
    ap.add_argument("--method", default="hermite", choices=METHODS)
    ap.add_argument("--max-gap", type=int, default=25, help="frames (25 = 1 s at 25 fps)")
    ap.add_argument("--max-step", type=float, default=2.0, help="spike threshold, m per frame")
    ap.add_argument("--max-speed", type=float, default=35.0,
                    help="m/s; a gap whose ends imply a faster ball is a re-detection, left NaN (0 = off)")
    ap.add_argument("--csv", action="store_true", help="also write a CSV next to the .npz")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    tr = load(cache=not args.no_cache)
    x, y, observed, removed, filled = prepare_ball(tr.bx, tr.by, args.method, args.max_gap,
                                                   args.max_step, tr.fps, args.max_speed or None)
    save_ball_file(args.out, x, y, observed, removed, filled, method=args.method,
                   max_gap=args.max_gap, max_step_m=args.max_step, lo=tr.lo,
                   source="ATD {} frames".format(tr.n), max_speed_ms=args.max_speed)
    graded = graded_mask(tr.n, tr.lo)
    present = ~np.isnan(x)
    lengths = np.array([e - s for s, e in zip(*runs(filled))])
    left = np.array([e - s for s, e in zip(*runs(~present & graded))])
    print("wrote {} ({} frames)".format(os.path.relpath(args.out, ROOT), tr.n))
    print("graded frames: observed {:.1f}%, spikes removed {}, reconstructed {} frames in {} gaps "
          "(longest {} frames), ball present after: {:.1f}%".format(
              100 * observed[graded].mean(), int(removed.sum()), int(filled.sum()), len(lengths),
              int(lengths.max()) if len(lengths) else 0, 100 * present[graded].mean()))
    if len(left):
        print("gaps left (graded): {}  median {:.1f} s, shortest {:.2f} s".format(
            len(left), np.median(left) / tr.fps, left.min() / tr.fps))
    if args.csv:
        path = os.path.splitext(args.out)[0] + ".csv"
        with open(path, "w", newline="") as h:
            h.write("frame,x_m,y_m,observed,filled\n")
            for i in range(tr.n):
                h.write("{},{},{},{},{}\n".format(tr.lo + i, "" if np.isnan(x[i]) else round(x[i], 3),
                                                  "" if np.isnan(y[i]) else round(y[i], 3),
                                                  int(observed[i]), int(filled[i])))
        print("wrote {}".format(os.path.relpath(path, ROOT)))


if __name__ == "__main__":
    main()
