"""Render one clip per long ball-tracking gap, using visualize_tracking.py.

Finds every run of consecutive frames where the ball is missing in the tactical ATD, then asks
visualize_tracking.py to render a clip covering each gap (plus a little before and after, so you
can see the ball coming back). Useful for answering the obvious next question after the EDA: what
is actually happening during those gaps -- a replay, the ball out of frame behind the goal, a
crowded box?

The clips are video frame ranges. The gap analyser works in ATD frame_id (1-based, = video frame
+ 1), so every start is converted before it reaches visualize_tracking.py.

  python eda/render_gap_clips.py                       # gaps >= 60 frames, in play
  python eda/render_gap_clips.py --min-length 125      # only the 2s+ gaps
  python eda/render_gap_clips.py --pad 5.0 --only-graded --max-clips 10
  python eda/render_gap_clips.py --extra-args "--no-tags --trail 80"
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from eda.eda_ball_gaps import find_gaps, iter_ball_present  # noqa: E402
from games import GAME, DEFAULT_SOURCE, half_video_blocks, require  # noqa: E402

VISUALIZE = os.path.join(HERE, "visualize_tracking.py")
FPS = 25.0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--game", default=GAME)
    ap.add_argument("--source", default=DEFAULT_SOURCE)
    ap.add_argument("--min-length", type=int, default=60,
                    help="only render gaps at least this many frames long (default 60 = 2.4s)")
    ap.add_argument("--pad", type=float, default=2.0,
                    help="seconds of context to render before and after each gap (default 2)")
    ap.add_argument("--only-graded", action="store_true",
                    help="skip gaps that fall entirely outside graded play")
    ap.add_argument("--max-clips", type=int, default=0,
                    help="stop after this many clips (0 = all)")
    ap.add_argument("--out-dir", default="gap_clips",
                    help="where to write the mp4s (default: gap_clips/)")
    ap.add_argument("--extra-args", default="",
                    help="arguments forwarded to visualize_tracking.py, e.g. --no-tags --trail 80")
    args = ap.parse_args()

    raw_path = require(["atd_raw"], args.game, None, args.source)[0]
    present = list(iter_ball_present(raw_path))
    gaps = find_gaps(present)

    # Graded blocks, for the --only-graded filter. A gap counts as in-play if it overlaps any block.
    blocks = half_video_blocks(args.game) if args.only_graded else None

    pad_frames = int(round(args.pad * FPS))
    clips = []
    for start_id, end_id, length in gaps:
        if length < args.min_length:
            continue
        if blocks and not any(end_id >= first and start_id <= last
                              for _, first, last in blocks):
            continue
        v0 = start_id - 1 - pad_frames          # ATD frame_id -> video frame, minus pad
        v0 = max(0, v0)
        duration = length + 2 * pad_frames
        clips.append((v0, duration, start_id, end_id, length))
        if args.max_clips and len(clips) >= args.max_clips:
            break

    if not clips:
        print("no gaps match --min-length {}{}".format(
            args.min_length, " / --only-graded" if args.only_graded else ""))
        return

    os.makedirs(args.out_dir, exist_ok=True)
    extra = args.extra_args.split() if args.extra_args else []
    print("{} gap{} to render (min {} frames{}):\n".format(
        len(clips), "" if len(clips) == 1 else "s", args.min_length,
        ", graded play only" if args.only_graded else ""))

    for v0, duration, start_id, end_id, length in clips:
        name = "gap_{:07d}_{:05d}.mp4".format(v0, length)
        out = os.path.abspath(os.path.join(args.out_dir, name))
        cmd = [sys.executable, VISUALIZE, "--start-frame", str(v0),
               "--duration", str(duration / FPS), "--out", out] + extra
        print("  gap ATD {}..{} ({} frames, {:.1f}s) -> video {}..{}  {:.1f}s".format(
            start_id, end_id, length, length / FPS, v0, v0 + int(duration) - 1, duration / FPS))
        print("    $ {}".format(" ".join(cmd)))
        proc = subprocess.run(cmd)
        if proc.returncode != 0:
            print("    FAILED (exit {})".format(proc.returncode))
        print()


if __name__ == "__main__":
    main()