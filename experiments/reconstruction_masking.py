"""EXPERIMENT: how well can a missing ball position be filled, by gap length and method?

From Krish's ball_reconstruction.py (masking suite). Known, fully tracked stretches of the ball are
hidden, refilled with possession/preprocess/reconstruct.py (linear / Hermite / quadratic) and the
error is measured in metres. Tracking only, no labels, so it may run on BAR-ATH; use the bench
(--source bench) when the numbers are used to choose a method or a cap.

    python experiments/reconstruction_masking.py                       # BAR-ATH graded halves
    python experiments/reconstruction_masking.py --source bench --game g1 --variant clean
    python experiments/reconstruction_masking.py --gaps 5 25 50 125 --trials 300

Krish's long-gap fallback evaluation (team centroid vs the public ground truth) is left out: it
scored rules on the public 5 minutes, which RULES.md reserves for sanity checks.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

import possession.io  # noqa: E402,F401
from possession.preprocess.ball import clean_ball  # noqa: E402
from possession.preprocess.reconstruct import METHODS, mask_and_score  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", choices=("barath", "bench"), default="barath")
    ap.add_argument("--game", default="g1")
    ap.add_argument("--variant", default="clean")
    ap.add_argument("--gaps", type=int, nargs="+", default=[5, 15, 30, 60, 125, 250])
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.source == "bench":
        from possession.io.bench import load_bench_game
        g = load_bench_game(os.path.join(ROOT, "outputs", "bench"), args.game, args.variant)
        bx, by, fps, pitch = g.bx, g.by, g.fps, g.pitch
        where = "bench {} {}".format(args.game, args.variant)
    else:
        from possession.io import graded_mask, load
        tr = load(cache=True)
        keep = graded_mask(tr.n, tr.lo)
        bx, by = np.where(keep, tr.bx, np.nan), np.where(keep, tr.by, np.nan)
        fps, pitch, where = tr.fps, tr.pitch, "BAR-ATH graded halves"
    track = clean_ball(bx, by, fps=fps, pitch=pitch)
    rows = mask_and_score(track.x, track.y, tuple(args.gaps), args.trials, METHODS, args.seed)
    print("masking experiment on {} ({} trials per gap length)".format(where, args.trials))
    print("{:>6} {:>7}  {:<10} {:>8} {:>8} {:>8} {:>9}".format(
        "frames", "seconds", "method", "rmse_m", "mae_m", "max_m", "<=5m %"))
    for r in rows:
        print("{gap:>6} {s:>7.1f}  {method:<10} {rmse_m:>8.2f} {mae_m:>8.2f} {max_m:>8.1f} "
              "{within_pct:>9.1f}".format(s=r["gap"] / fps, **r))


if __name__ == "__main__":
    main()
