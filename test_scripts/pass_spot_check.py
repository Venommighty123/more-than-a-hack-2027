"""Spot-check detected passes against the video: a random sample with video timestamps.

Reads reports/pass_detection/passes.json (run test_scripts/pass_detection.py first) and picks a
random, reproducible sample of passes per outcome. For each one it prints the time to seek to in
the TACTICAL video, the match clock, the team, and who the script thinks passed and received.

Watch each clip (a few seconds from the given time) and fill in the `verdict` column of the CSV
it writes:
    ok             the script got it right
    not_a_pass     no pass happened here (dribble, tackle, noise ...)
    wrong_outcome  a real pass, but complete/intercepted/out is wrong
    wrong_team     a real pass, but by the other team
Then paste only the verdict counts per outcome back into the thread (not the rows).

    python test_scripts/pass_spot_check.py                    # 20 intercepted + 10 complete
    python test_scripts/pass_spot_check.py --n 30 --outcome intercepted --seed 2

Writes reports/pass_detection/spot_check.csv (git-ignored folder: it holds event positions).
"""

import argparse
import csv
import json
import os
import random

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR = os.path.join(ROOT, "reports", "pass_detection")


def hms(seconds):
    seconds = int(seconds)
    return "{:d}:{:02d}:{:02d}".format(seconds // 3600, seconds // 60 % 60, seconds % 60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20, help="sample size for --outcome")
    parser.add_argument("--outcome", default="intercepted")
    parser.add_argument("--n-complete", type=int, default=10,
                        help="also sample this many complete passes as a control")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--lead", type=float, default=2.0, help="seconds before the pass to seek")
    parser.add_argument("--touches", action="store_true",
                        help="sample rows the script did NOT count as passes (type touch / duel) "
                             "instead: were any of them real passes?")
    args = parser.parse_args()

    with open(os.path.join(DIR, "passes.json")) as handle:
        everything = json.load(handle)["passes"]
    rows = [r for r in everything if r["is_pass"]]
    rng = random.Random(args.seed)
    if args.touches:
        pool = [r for r in everything if r["type"] in ("touch", "duel")]
        rows = []
        for r in rng.sample(pool, min(args.n, len(pool))):
            r = dict(r, outcome=r["type"])
            r["receiver_team"] = r["receiver_team"] or r["team"]
            rows.append(r)
        args.outcome, args.n_complete = rows[0]["outcome"] if rows else "touch", 0
        picks = sorted(rows, key=lambda r: r["frame_start"])
        return report(picks, args)
    picks = []
    for outcome, n in ((args.outcome, args.n), ("complete", args.n_complete)):
        pool = [r for r in rows if r["outcome"] == outcome]
        if outcome == args.outcome and outcome == "complete":
            continue
        picks += sorted(rng.sample(pool, min(n, len(pool))), key=lambda r: r["frame_start"])
    if args.outcome == "complete":
        pool = [r for r in rows if r["outcome"] == "complete"]
        picks = sorted(rng.sample(pool, min(args.n, len(pool))), key=lambda r: r["frame_start"])

    report(picks, args)


def report(picks, args):
    out = []
    print("{:>4} {:<12} {:>9} {:>6} {:<4} {:>6} {:>8} {:>6} {:<12}".format(
        "#", "outcome", "seek to", "clock", "team", "from", "to", "len m", "speed km/h"))
    for i, r in enumerate(picks, 1):
        seek = hms(max(0.0, r["video_time_s"] - args.lead))
        to = "{}{}".format(r["receiver_team"] or "", r["receiver_slot"] or "-")
        print("{:>4} {:<12} {:>9} {:>6} {:<4} {:>6} {:>8} {:>6} {:<12}".format(
            i, r["outcome"], seek, r["match_clock"], r["team"],
            "{}{}".format(r["team"], r["passer_slot"]), to, r["length_m"],
            r["max_ball_speed_kmh"]))
        out.append(dict(n=i, id=r["id"], outcome=r["outcome"], seek_to=seek,
                        match_clock=r["match_clock"], team=r["team"],
                        passer_slot=r["passer_slot"], receiver=to, length_m=r["length_m"],
                        duration_s=r["duration_s"], verdict=""))
    path = os.path.join(DIR, "spot_check.csv")
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(out[0]) if out else ["n"])
        writer.writeheader()
        writer.writerows(out)
    print("\nwrote {}  (fill in the verdict column: ok / not_a_pass / wrong_outcome / "
          "wrong_team)".format(os.path.relpath(path, ROOT)))
    print("Slots are tracking slots, not shirt numbers. 'seek to' is the tactical video time, "
          "{:.0f} s before the kick.".format(args.lead))


if __name__ == "__main__":
    main()
