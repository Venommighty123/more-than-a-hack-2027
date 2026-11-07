"""Run possession labeller v1 on BAR-ATH and write a full-match submission CSV.

    python test_scripts/run_labeller_v1.py
    python test_scripts/run_labeller_v1.py --out evaluation/results/labeller_v1.csv

Everything that decides a label lives in test_scripts/labeller_v1.py and was tuned ONLY on the
Metrica public sample games (test_scripts/bench/). Nothing here is tuned on this match.

What it prints (all of it is safe to share, none of it is raw data):
  1. the official validation of the file (evaluation/validate_submission.py rules)
  2. what the file says: label shares, number of possessions, mean possession, number of stoppages
  3. a SANITY score on the public 5 minutes. RULE: this number is a format / sanity check only and
     must never be used to choose between versions or settings (EVALUATION.md shows that tuning on
     it picks the wrong option). Versions are compared on the bench.
"""

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

from games import GAME, GAMES, anchor_video_frame, half_video_blocks, require  # noqa: E402
from pass_detection import load_tracking, team_of_each_track, team_names  # noqa: E402
import labeller_v1  # noqa: E402
import possession_metrics as pm  # noqa: E402

N_FRAMES = GAMES[GAME]["n_video_frames"]
FPS = GAMES[GAME]["fps"]
PUBLIC_GT = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
OUT = os.path.join(ROOT, "evaluation", "results", "labeller_v1.csv")


def describe(labels, blocks):
    graded = np.zeros(len(labels), dtype=bool)
    for _, a, b in blocks:
        graded[a:b + 1] = True
    g = labels[graded]
    shares = {k: 100.0 * np.mean(g == k) for k in "HADX"}
    spells, stops = [], 0
    for _, a, b in blocks:
        seg = labels[a:b + 1]
        edges = np.r_[0, np.flatnonzero(seg[1:] != seg[:-1]) + 1, len(seg)]
        for s, e in zip(edges[:-1], edges[1:]):
            if seg[s] in "HA":
                spells.append(e - s)
            elif seg[s] == "D":
                stops += 1
    spells = np.array(spells) / FPS
    print("graded frames: H {H:.1f}%  A {A:.1f}%  D {D:.1f}%  X {X:.1f}%".format(**shares))
    print("possessions: {}  (mean {:.1f} s, median {:.1f} s, shorter than 2 s: {})".format(
        len(spells), spells.mean(), np.median(spells), int((spells < 2).sum())))
    print("stoppages (D stretches): {}".format(stops))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    t0 = time.time()
    meta_path, raw_path = require(["atd_meta", "atd_raw"])
    team = team_of_each_track(meta_path)
    home, away = team_names(meta_path)
    print("H = {}, A = {}, {} tracking slots".format(home, away, len(team)))
    graded_blocks = half_video_blocks(GAME)                       # what the scorer grades
    kickoff2 = anchor_video_frame(GAMES[GAME]["kickoff_2nd"])     # 113742, FAQ.md
    print("loading tracking ...")
    px, py, ball = load_tracking(raw_path, len(team), 0, N_FRAMES - 1)
    print("  loaded in {:.0f} s".format(time.time() - t0))

    # The labeller sees play from each kickoff. Half 2 is graded from 110808 but kicks off at
    # 113742; FAQ.md says those 2934 frames are all D, so they are written as D directly.
    play_blocks = []
    for half, a, b in graded_blocks:
        play_blocks.append((half, max(a, kickoff2) if half == 2 else a, b))
    labels = labeller_v1.label(px, py, ball, team, play_blocks)
    for half, a, b in graded_blocks:
        if half == 2:
            labels[a:kickoff2] = "D"

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, v) for i, v in enumerate(labels))
    print("wrote {} ({} rows)".format(os.path.relpath(args.out, ROOT), len(labels)))

    frame = pm.read_label_csv(args.out)
    problem = pm.validate_submission(frame, N_FRAMES)
    print("validation: {}".format("OK" if problem is None else "FAILED - " + problem))
    describe(labels, graded_blocks)

    if os.path.exists(PUBLIC_GT):
        gt = pm.label_codes(pm.read_label_csv(PUBLIC_GT))
        r = pm.evaluate(gt, pm.label_codes(frame))
        print("SANITY ONLY (public 5 min, never tune on it): spellF1 {:.4f}  macroF1 {:.4f}  "
              "possessions {}/{}".format(r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"],
                                         r["n_gt_spells"]))
    print("done in {:.0f} s".format(time.time() - t0))


if __name__ == "__main__":
    main()
