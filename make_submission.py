"""Build the submission CSV: possession labeller v6 on the video-based player tracks (T5).

    python make_submission.py                                   # -> submission/<name>.csv
    python make_submission.py --name <team_name>
    python make_submission.py --tracks results_new/final_tracks_full_t5.parquet --out my.csv

INPUTS
  data/tactical/BAR-ATH_tactical_atd.txt / .xml     provided tracking: the BALL and the team names
  data/tactical/BAR-ATH_tactical_smart_tagging.xml  provided Smart Tagging: "Set Pieces" start frames
  results_new/final_tracks_full_t5.parquet           players + goalkeepers with team H/A, from
                                                     test_scripts/kaggle-player-tracking_new.ipynb
STEPS (all deterministic, CPU, about one minute)
  1. read the ATD ball (progress bar)
  2. read the player tracks, keep players and goalkeepers with a team, pack them into slots (bar)
  3. read the set-piece tags
  4. labeller v6 (test_scripts/labeller_v6.py, default settings) on the two graded halves
  5. second half from the start of grading to the 2nd-half kickoff = D (FAQ.md)
  6. write frame,label for all 199,319 frames, validate with the official checks, print the md5

This is the same code path as test_scripts/run_labeller_v6_t5.py, without its comparisons, so it
runs on a fresh clone. Expected md5 for the submitted file: see README.md.
"""

import argparse
import hashlib
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(ROOT, "test_scripts"), os.path.join(ROOT, "evaluation"), ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from games import GAME, GAMES, anchor_video_frame, half_video_blocks, require  # noqa: E402
from atd_io import team_of_each_track, team_names  # noqa: E402
import labeller_v6  # noqa: E402
import possession_metrics as pm  # noqa: E402
from run_labeller_v6 import describe, load_set_piece_frames  # noqa: E402
import run_labeller_v6_newtracks as nt  # noqa: E402  (ATD and track readers)

N_FRAMES = GAMES[GAME]["n_video_frames"]
TRACKS = os.path.join(ROOT, "results_new", "final_tracks_full_t5.parquet")


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tracks", default=TRACKS, help="player tracks parquet (default results_new/final_tracks_full_t5.parquet)")
    ap.add_argument("--name", default="submission", help="file name without .csv (use your team name)")
    ap.add_argument("--out", help="full output path (overrides --name)")
    args = ap.parse_args()
    out = args.out or os.path.join(ROOT, "submission", args.name + ".csv")
    t0 = time.time()
    if not os.path.isfile(args.tracks):
        sys.exit("player tracks not found: {}\n  -> run test_scripts/kaggle-player-tracking_new.ipynb "
                 "(README.md, step 2) and put final_tracks_full_t5.parquet there".format(args.tracks))

    meta_path, raw_path = require(["atd_meta", "atd_raw"])
    atd_team = team_of_each_track(meta_path)
    print("H = {}, A = {}".format(*team_names(meta_path)))
    graded_blocks = half_video_blocks(GAME)
    kickoff2 = anchor_video_frame(GAMES[GAME]["kickoff_2nd"])
    play_blocks = [(h, max(a, kickoff2) if h == 2 else a, b) for h, a, b in graded_blocks]

    print("[1/4] ATD ball")
    _, _, ball = nt.load_atd(raw_path, len(atd_team))
    print("[2/4] player tracks")
    px, py, team = nt.load_new_tracks(args.tracks, True)
    print("[3/4] Smart Tagging set pieces")
    tags = load_set_piece_frames()
    print("  {} set-piece tags".format(len(tags)))

    print("[4/4] labeller v6 ...", flush=True)
    t = time.time()
    labels = labeller_v6.label(px, py, ball, np.asarray(team), play_blocks, params={"USE_TAGS": True}, tags=tags)
    for h, a, b in graded_blocks:
        if h == 2:
            labels[a:kickoff2] = "D"        # graded but before the 2nd-half kickoff: dead (FAQ.md)
    print("  done in {:.0f} s".format(time.time() - t))

    os.makedirs(os.path.dirname(out), exist_ok=True)
    nt.write_csv(out, labels)
    problem = pm.validate_submission(pm.read_label_csv(out), N_FRAMES)
    print("\nwrote {}".format(os.path.relpath(out, ROOT)))
    print("validation (official checks): {}".format("OK" if problem is None else "FAILED - " + problem))
    describe(labels, graded_blocks)
    print("md5 {}".format(md5(out)))
    print("done in {:.0f} s".format(time.time() - t0))
    if problem is not None:
        sys.exit(1)


if __name__ == "__main__":
    main()
