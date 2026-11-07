"""Render one clip per row of flights.csv, using visualize_tracking.py unchanged.

Every candidate flight (is_airborne True or False) gets a clip from `before` frames ahead of its
start_frame to `after` frames after it (default 100 + 100, plus the start frame itself = 201
frames). visualize_tracking.py is called as a subprocess, so it must sit in the same directory as
this file. Anything this script does not recognise is forwarded to it, e.g. --no-video, --ids,
--backend opencv.

    python render_flight_clips.py                          # flights.csv -> videos/
    python render_flight_clips.py --csv flights.csv --out-dir videos --limit 3
    python render_flight_clips.py --no-video               # tracking-only clips (no mp4 decode)
    python render_flight_clips.py --dry-run                # print the commands only

Clips that already exist are skipped, so an interrupted run can simply be restarted.
"""

import argparse
import os
import subprocess
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
VIZ = os.path.join(HERE, "visualize_tracking.py")


def clip_window(start_frame, before, after):
    """(first_frame, n_frames) covering start-before .. start+after inclusive, clamped at 0."""
    first = max(0, int(start_frame) - before)
    return first, int(start_frame) + after - first + 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--csv", default="flights.csv", help="output of ball_flight.py")
    ap.add_argument("--out-dir", default="videos")
    ap.add_argument("--before", type=int, default=100, help="frames before start_frame")
    ap.add_argument("--after", type=int, default=100, help="frames after start_frame")
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--limit", type=int, help="render only the first N flights")
    ap.add_argument("--dry-run", action="store_true")
    args, passthrough = ap.parse_known_args()

    if not os.path.isfile(VIZ):
        sys.exit("visualize_tracking.py must be in the same directory as this script ({})".format(HERE))
    flights = pd.read_csv(args.csv)
    if flights.empty:
        sys.exit("{} has no flights".format(args.csv))
    os.makedirs(args.out_dir, exist_ok=True)
    if args.limit:
        flights = flights.head(args.limit)

    failed = []
    for index, row in flights.iterrows():
        first, n = clip_window(row["start_frame"], args.before, args.after)
        kind = "air" if bool(row["is_airborne"]) else "ground"
        out = os.path.join(args.out_dir, "flight_{:04d}_f{}_{}.mp4".format(
            index, int(row["start_frame"]), kind))
        if os.path.isfile(out):
            print("skip (exists): {}".format(out))
            continue
        cmd = [sys.executable, VIZ, "--start-frame", str(first),
               "--duration", "{:.4f}".format(n / args.fps), "--out", out] + passthrough
        print("[{}/{}] frames {}..{} -> {}".format(index + 1, len(flights), first, first + n - 1, out))
        if args.dry_run:
            print("   " + " ".join(cmd))
            continue
        if subprocess.call(cmd) != 0 or not os.path.isfile(out):
            failed.append(index)
            print("   FAILED")
    if failed:
        sys.exit("{} clip(s) failed: rows {}".format(len(failed), failed))


if __name__ == "__main__":
    main()