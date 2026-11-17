"""Labeller v6, unchanged, on Kushagra's NEW player tracks (T5 team model), compared with the old tracks.

    python test_scripts/run_labeller_v6_t5.py
    python test_scripts/run_labeller_v6_t5.py --tracks results_new/final_tracks_full.parquet   # new run, no T5
    python test_scripts/run_labeller_v6_t5.py --skip-check

WHAT CHANGES AND WHAT DOES NOT
  players  NEW: results_new/final_tracks_full_t5.parquet (Kushagra, Nov 17). Same columns as before;
           every track segment re-labelled home / away / home GK / away GK / official by the T5 model
           (SigLIP2 on whole-body crops, trained on ATD team membership; his report: 97.0% agreement
           with the ATD teams vs 75.2% before). Read with exactly the same function as
           run_labeller_v6_newtracks.py (players + goalkeepers with team H/A; officials left out).
  ball     ATD, as before (the YOLO ball and referee files are NOT used).
  tags     Smart Tagging set-piece tags, as before.
  labeller test_scripts/labeller_v6.py, defaults, unchanged. Nothing tuned.

STEPS
  1. read the ATD (ball only is used for the labeller)
  2. reproduction check: v6 on the OLD tracks (results/final_tracks_full.parquet) must give the md5 of
     evaluation/results/labeller_v6_newtracks.csv, so the only thing that differs is the track file
  3. v6 on the NEW tracks -> evaluation/results/labeller_v6_newtracks_t5.csv (validated)
  4. players per team per frame (old vs new tracks): how often a team shows more than 11
  5. all three files side by side (ATD SAFE, old tracks, new tracks) and two comparisons:
       new vs old tracks  -> compare_v6_t5_vs_newtracks_minutes.csv / _stretches.csv
       new vs ATD (SAFE)  -> compare_v6_t5_vs_safe_minutes.csv / _stretches.csv
  6. SANITY lines on the public 5 minutes (format check only, never a reason to choose)

No truth exists for the full match: this shows how much and where the labels change, not which is right.
Run time about 1 minute; reading the ATD and packing the tracks show progress bars.
"""

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _p in (os.path.join(ROOT, "evaluation"), HERE, ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from games import GAME, GAMES, anchor_video_frame, half_video_blocks, require  # noqa: E402
from atd_io import team_of_each_track  # noqa: E402
import labeller_v6  # noqa: E402
import possession_metrics as pm  # noqa: E402
from run_labeller_v6 import describe, load_set_piece_frames  # noqa: E402
import run_labeller_v6_newtracks as nt  # noqa: E402  (same readers, same comparison)

N_FRAMES = GAMES[GAME]["n_video_frames"]
RESULTS = os.path.join(ROOT, "evaluation", "results")
OLD_TRACKS = os.path.join(ROOT, "results", "final_tracks_full.parquet")
OLD_CSV = os.path.join(RESULTS, "labeller_v6_newtracks.csv")
NEW_TRACKS = os.path.join(ROOT, "results_new", "final_tracks_full_t5.parquet")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tracks", default=NEW_TRACKS)
    ap.add_argument("--out", default=os.path.join(RESULTS, "labeller_v6_newtracks_t5.csv"))
    ap.add_argument("--skip-check", action="store_true", help="skip the old-tracks reproduction check")
    args = ap.parse_args()
    t0 = time.time()
    for p in (args.tracks, OLD_CSV, nt.SAFE) + (() if args.skip_check else (OLD_TRACKS,)):
        if not os.path.isfile(p):
            sys.exit("not found: {}".format(p))

    meta_path, raw_path = require(["atd_meta", "atd_raw"])
    atd_team = team_of_each_track(meta_path)
    graded_blocks = half_video_blocks(GAME)
    kickoff2 = anchor_video_frame(GAMES[GAME]["kickoff_2nd"])
    play_blocks = [(h, max(a, kickoff2) if h == 2 else a, b) for h, a, b in graded_blocks]
    tags = load_set_piece_frames()
    graded = np.zeros(N_FRAMES, dtype=bool)
    for _, a, b in graded_blocks:
        graded[a:b + 1] = True

    def run_v6(px, py, ball, team, name):
        t = time.time()
        print("running labeller v6 on {} ...".format(name), flush=True)
        lab = labeller_v6.label(px, py, ball, np.asarray(team), play_blocks, params={"USE_TAGS": True},
                                tags=tags)
        for h, a, b in graded_blocks:
            if h == 2:
                lab[a:kickoff2] = "D"
        print("  done in {:.0f} s".format(time.time() - t))
        return lab

    print("[1/6] ATD (ball only)")
    apx, apy, ball = nt.load_atd(raw_path, len(atd_team))
    del apx, apy

    old_lab = pm.read_label_csv(OLD_CSV)["label"].to_numpy().astype(str)
    opx = None
    if args.skip_check:
        print("\n[2/6] reproduction check skipped")
    else:
        print("\n[2/6] reproduction check: old tracks ({})".format(os.path.relpath(OLD_TRACKS, ROOT)))
        opx, opy, oteam = nt.load_new_tracks(OLD_TRACKS, True)
        lab = run_v6(opx, opy, ball, oteam, "the OLD tracks")
        tmp = args.out + ".check.tmp"
        nt.write_csv(tmp, lab)
        same = nt.md5(tmp) == nt.md5(OLD_CSV)
        os.remove(tmp)
        print("REPRODUCTION CHECK: old tracks {} labeller_v6_newtracks.csv{}".format(
            "== (md5 identical)" if same else "!=",
            "" if same else "  -> STOP: something besides the track file changed"))
        if not same:
            sys.exit(1)

    print("\n[3/6] NEW tracks ({})".format(os.path.relpath(args.tracks, ROOT)))
    npx, npy, nteam = nt.load_new_tracks(args.tracks, True)
    new_lab = run_v6(npx, npy, ball, nteam, "the NEW tracks")
    nt.write_csv(args.out, new_lab)
    problem = pm.validate_submission(pm.read_label_csv(args.out), N_FRAMES)
    print("wrote {}  validation: {}".format(os.path.relpath(args.out, ROOT),
                                           "OK" if problem is None else "FAILED - " + problem))

    print("\n[4/6] players per team per frame (graded frames)")
    rows = [("new", npx, nteam)] if opx is None else [("old", opx, oteam), ("new", npx, nteam)]
    for name, px_, tm in rows:
        h, a = nt.team_counts(px_, tm, graded)
        print("  {}: home {:.1f} / away {:.1f} per frame; a team shows more than 11 on {:.1f}% of frames; "
              "under 9 on {:.1f}%".format(name, h.mean(), a.mean(), 100 * np.mean((h > 11) | (a > 11)),
                                          100 * np.mean((h < 9) | (a < 9))))

    safe = pm.read_label_csv(nt.SAFE)["label"].to_numpy().astype(str)
    print("\n[5/6] the three files")
    for title, lab in (("SAFE_SUBMISSION_labeller_v6.csv (ATD players)", safe),
                       ("labeller_v6_newtracks.csv (OLD Kushagra tracks, current choice)", old_lab),
                       ("labeller_v6_newtracks_t5.csv (NEW Kushagra tracks, T5)", new_lab)):
        print("-- " + title)
        describe(lab, graded_blocks)
    print("\n### NEW tracks vs OLD tracks  (rows 'ATD' = OLD tracks run, columns 'new' = NEW tracks run)")
    nt.compare(old_lab, new_lab, graded_blocks,
               os.path.join(RESULTS, "compare_v6_t5_vs_newtracks_minutes.csv"),
               os.path.join(RESULTS, "compare_v6_t5_vs_newtracks_stretches.csv"))
    print("\n### NEW tracks vs ATD players (SAFE file)")
    nt.compare(safe, new_lab, graded_blocks,
               os.path.join(RESULTS, "compare_v6_t5_vs_safe_minutes.csv"),
               os.path.join(RESULTS, "compare_v6_t5_vs_safe_stretches.csv"))
    print("  (the printed visualize commands name the old files; watch the new run with "
          "--labels evaluation/results/labeller_v6_newtracks_t5.csv)")

    print("\n[6/6]")
    nt.sanity("ATD (SAFE)", safe)
    nt.sanity("old tracks", old_lab)
    nt.sanity("new tracks", new_lab)
    print("\nREMINDER: no truth exists for the full match; this shows how much and where the labels change. "
          "labeller_v6_newtracks.csv is untouched.")
    print("done in {:.0f} s".format(time.time() - t0))


if __name__ == "__main__":
    main()
