"""Run labeller v6, unchanged, on Kushagra's video-based player tracks and compare with the ATD run.

    python test_scripts/run_labeller_v6_newtracks.py
    python test_scripts/run_labeller_v6_newtracks.py --tracks results/final_tracks_full.parquet
    python test_scripts/run_labeller_v6_newtracks.py --no-goalkeepers      # outfield players only
    python test_scripts/run_labeller_v6_newtracks.py --skip-atd-check       # faster, no reproduction check

WHAT CHANGES AND WHAT DOES NOT
  players  from Kushagra's tracks (results/final_tracks_full.parquet: frame, track_id, role, team,
           x_m, y_m). Players and goalkeepers with a team H/A are used; referees and rows with no team
           are left out (counts printed). x_m / y_m are metres on the same 105.3 x 68 pitch and with
           the same origin as the ATD (Kushagra's head-to-head: median 0.14 m from the ATD).
           Each track keeps its own team; tracks are packed into "slots" per team (a slot holds one
           track at a time), because the labeller takes (frames x slots) arrays plus a team per slot.
  ball     from the ATD, exactly as in run_labeller_v6.py (the new tracks have no ball).
  tags     Smart Tagging set-piece tags, exactly as in run_labeller_v6.py.
  labeller test_scripts/labeller_v6.py with its defaults. NOTHING is tuned here.

WHAT IT WRITES
  evaluation/results/labeller_v6_newtracks.csv                the new full-match CSV (validated)
  evaluation/results/compare_v6_newtracks_minutes.csv         disagreement per minute
  evaluation/results/compare_v6_newtracks_stretches.csv       every stretch where the two CSVs differ
WHAT IT PRINTS (summary numbers only, safe to share)
  1. reproduction check: v6 on the ATD read by this script must equal SAFE_SUBMISSION_labeller_v6.csv
     (md5), so the only difference between the two runs is the player source
  2. the player tracks: players per team per frame, how often a team shows more than 11
  3. both CSVs side by side: label shares, possessions, stoppages, changes
  4. the comparison: same label on X% of graded frames, a 3 x 3 table (ATD run vs new run), H<->A
     swaps vs dead-ball differences, the 10 minutes where they differ most (with commands to watch
     them in visualize_labeller.py)
  5. SANITY lines on the public 5 minutes for both. RULE: format check only, never a reason to
     choose one over the other (EVALUATION.md). There is no truth for the full match, so this script
     can say HOW MUCH and WHERE the two differ, not which one is right.

The CSVs are derived from the match data: evaluation/results/ stays out of the public repo.
Needs numpy, pandas and pyarrow (pip install pyarrow) to read the parquet.
"""

import argparse
import hashlib
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
from atd_io import team_of_each_track, team_names, PITCH_L, PITCH_W  # noqa: E402
import labeller_v6  # noqa: E402
import possession_metrics as pm  # noqa: E402
from run_labeller_v6 import describe, load_set_piece_frames  # noqa: E402

N_FRAMES = GAMES[GAME]["n_video_frames"]
FPS = GAMES[GAME]["fps"]
RESULTS = os.path.join(ROOT, "evaluation", "results")
SAFE = os.path.join(RESULTS, "SAFE_SUBMISSION_labeller_v6.csv")
PUBLIC_GT = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
TRACKS = os.path.join(ROOT, "results", "final_tracks_full.parquet")
TRACK_PITCH = (105.3, 68.0)       # Kushagra's x_m / y_m pitch (kaggle_player_tracking.ipynb, PITCH)


def progress(label, i, n, t0, width=30):
    """One line, updated in place (Ashwyn's rule: anything over a minute shows a progress bar)."""
    f = min(1.0, (i + 1) / float(max(1, n)))
    el = time.time() - t0
    eta = el / f - el if f > 0 else 0.0
    bar = "#" * int(width * f) + "-" * (width - int(width * f))
    print("\r  {} [{}] {:5.1f}%  {:5.0f}s elapsed  ETA {:5.0f}s".format(label, bar, 100 * f, el, eta),
          end="", flush=True)
    if f >= 1.0:
        print(flush=True)


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# -- ATD (same parse as pass_detection.load_tracking, plus a progress bar) -------------------------

def load_atd(raw_path, n_slots):
    """Players (px, py) and ball from the raw ATD for every video frame, exactly as
    pass_detection.load_tracking(raw_path, n_slots, 0, N_FRAMES - 1) builds them (same dtypes)."""
    n = N_FRAMES
    px = np.full((n, n_slots), np.nan, dtype=np.float32)
    py = np.full((n, n_slots), np.nan, dtype=np.float32)
    ball = np.full((n, 2), np.nan)
    size = os.path.getsize(raw_path)
    per_slot, done, t0, k = None, 0, time.time(), 0
    with open(raw_path) as handle:
        for line in handle:
            done += len(line) + 1
            k += 1
            if k % 5000 == 0:
                progress("reading ATD   ", done, size, t0)
            head = line[:line.index(":")] if ":" in line else ""
            if not head.isdigit():
                continue
            frame = int(head) - 1
            if frame < 0:
                continue
            if frame > n - 1:
                break
            parts = line.strip().split(":")
            if len(parts) != 3:
                continue
            values = np.array(parts[1].replace(";", ",").split(","), dtype=np.float32)
            if per_slot is None:
                per_slot = len(values) // n_slots
                if per_slot < 2 or per_slot * n_slots != len(values):
                    raise ValueError("cannot split {} values into {} slots".format(len(values), n_slots))
            values = values.reshape(n_slots, per_slot)
            px[frame] = values[:, 0] * PITCH_L
            py[frame] = values[:, 1] * PITCH_W
            b = parts[2].split(",")
            if len(b) >= 2 and b[0][:1] != "N":
                ball[frame] = (float(b[0]) * PITCH_L, float(b[1]) * PITCH_W)
    progress("reading ATD   ", size, size, t0)
    return px, py, ball


# -- Kushagra's tracks -> (frames x slots) ---------------------------------------------------------

def load_new_tracks(path, keep_gk):
    try:
        import pandas as pd
        table = pd.read_parquet(path, columns=["frame", "track_id", "role", "team", "x_m", "y_m"])
    except ImportError as error:
        sys.exit("reading the parquet needs pyarrow:  pip install pyarrow   ({})".format(error))
    n0 = len(table)
    role = table["role"].astype(str).str.lower()
    team = table["team"].astype(str).str.upper().str.strip()
    roles = ["player", "goalkeeper"] if keep_gk else ["player"]
    print("tracks file: {:,} rows; roles {}; team {}".format(
        n0, role.value_counts().to_dict(), team.replace("", "none").value_counts().to_dict()))
    keep = role.isin(roles) & team.isin(["H", "A"]) & np.isfinite(table["x_m"]) & np.isfinite(table["y_m"])
    keep &= (table["frame"] >= 0) & (table["frame"] < N_FRAMES)
    t = table[keep].copy()
    t["team"] = team[keep].map({"H": 0, "A": 1}).astype(np.int8)
    print("  used {:,} rows ({:.1f}%): {}{}; left out referees, rows with no team, off-range rows".format(
        len(t), 100.0 * len(t) / max(1, n0), " and ".join(roles), "" if keep_gk else " (no goalkeepers)"))
    t = t.sort_values(["frame", "x_m"]).drop_duplicates(["frame", "track_id", "team"])
    # a track keeps one team in Kushagra's output; split defensively if it ever changes
    t["key"] = t["track_id"].astype(np.int64) * 2 + t["team"].astype(np.int64)
    span = t.groupby("key").agg(start=("frame", "min"), end=("frame", "max"), team=("team", "first"))
    span = span.sort_values(["start", "end"])
    # interval packing per team: a slot carries one track at a time
    slot_of, slot_team, free_at = {}, [], []
    t0 = time.time()
    keys, starts, ends, teams = span.index.values, span["start"].values, span["end"].values, span["team"].values
    for i in range(len(keys)):
        if i % 2000 == 0:
            progress("packing tracks", i, len(keys), t0)
        best = None
        for s, (tm, end) in enumerate(zip(slot_team, free_at)):
            if tm == teams[i] and end < starts[i]:
                best = s
                break
        if best is None:
            best = len(slot_team)
            slot_team.append(int(teams[i]))
            free_at.append(-1)
        slot_of[keys[i]] = best
        free_at[best] = ends[i]
    progress("packing tracks", len(keys), len(keys), t0)
    m = len(slot_team)
    sx, sy = TRACK_PITCH
    px = np.full((N_FRAMES, m), np.nan, dtype=np.float32)
    py = np.full((N_FRAMES, m), np.nan, dtype=np.float32)
    slots = t["key"].map(slot_of).values
    frames = t["frame"].values.astype(np.int64)
    px[frames, slots] = (t["x_m"].values * (PITCH_L / sx)).astype(np.float32)
    py[frames, slots] = (t["y_m"].values * (PITCH_W / sy)).astype(np.float32)
    team_arr = np.array(slot_team, dtype=np.int64)
    print("  {:,} tracks packed into {} slots (home {}, away {})".format(
        len(keys), m, int((team_arr == 0).sum()), int((team_arr == 1).sum())))
    return px, py, team_arr


def team_counts(px, team, graded):
    ok = ~np.isnan(px[graded])
    h = ok[:, team == 0].sum(1)
    a = ok[:, team == 1].sum(1)
    return h, a


# -- comparison ------------------------------------------------------------------------------------

def runs(values, lo, hi):
    seg = values[lo:hi + 1]
    cut = np.flatnonzero(seg[1:] != seg[:-1]) + 1
    st = np.r_[0, cut]
    en = np.r_[cut, len(seg)] - 1
    return [(lo + int(a), lo + int(b)) for a, b in zip(st, en)]


def clock(f):
    s = f / float(FPS)
    return "{:02d}:{:04.1f}".format(int(s // 60), s % 60)


def compare(old, new, blocks, out_min, out_str):
    import csv
    graded = np.zeros(N_FRAMES, dtype=bool)
    for _, a, b in blocks:
        graded[a:b + 1] = True
    o, n = old[graded], new[graded]
    print("\n== comparison: ATD run (SAFE file) vs new-tracks run, graded frames ==")
    print("same label on {:.2f}% of graded frames ({:,} of {:,} differ)".format(
        100.0 * np.mean(o == n), int((o != n).sum()), int(graded.sum())))
    print("            new H     new A     new D      (% of graded frames)")
    for r in "HAD":
        print("  ATD {}  ".format(r) + "".join("{:9.2f} ".format(100.0 * np.mean((o == r) & (n == c))) for c in "HAD"))
    swap = np.mean(((o == "H") & (n == "A")) | ((o == "A") & (n == "H")))
    dead = np.mean((o == "D") != (n == "D"))
    print("H<->A swaps (team in possession differs): {:.2f}% of graded frames; "
          "dead-ball differences (D in one only): {:.2f}%".format(100 * swap, 100 * dead))
    r = pm.evaluate(_codes(old), _codes(new))
    print("similarity, spellF1 of the new run measured against the ATD run (NOT a score, neither is "
          "truth): {:.3f}".format(r["spell_f1_mean"]))

    # per minute
    rows = []
    for _, a, b in blocks:
        for s in range(a, b + 1, 60 * FPS):
            e = min(b, s + 60 * FPS - 1)
            seg_o, seg_n = old[s:e + 1], new[s:e + 1]
            rows.append((s, e, clock(s), 100.0 * np.mean(seg_o != seg_n),
                         100.0 * np.mean(((seg_o == "H") & (seg_n == "A")) | ((seg_o == "A") & (seg_n == "H")))))
    with open(out_min, "w", newline="") as h:
        w = csv.writer(h)
        w.writerow(["first_frame", "last_frame", "video_clock", "differ_pct", "team_swap_pct"])
        w.writerows([(a, b, c, "{:.1f}".format(d), "{:.1f}".format(t)) for a, b, c, d, t in rows])
    top = sorted(rows, key=lambda x: -x[3])[:10]
    print("\nthe 10 minutes where they differ most (watch both with visualize_labeller.py):")
    print("  {:>9} {:>8} {:>8} {:>8} {:>10}".format("clock", "first", "last", "differ", "team swap"))
    for a, b, c, d, t in top:
        print("  {:>9} {:>8} {:>8} {:>7.1f}% {:>9.1f}%".format(c, a, b, d, t))
    a, b = top[0][0], top[0][1]
    print("  e.g.  python test_scripts/visualize_labeller.py --range {} {} "
          "--labels evaluation/results/labeller_v6_newtracks.csv".format(a, b))
    print("        python test_scripts/visualize_labeller.py --range {} {}            (the SAFE file)".format(a, b))

    # stretches
    diff = (old != new) & graded
    n_str = 0
    with open(out_str, "w", newline="") as h:
        w = csv.writer(h)
        w.writerow(["first_frame", "last_frame", "video_clock", "seconds", "atd_run", "new_run"])
        for s, e in runs(diff.astype(np.int8), 0, N_FRAMES - 1):
            if diff[s]:
                n_str += 1
                w.writerow([s, e, clock(s), "{:.2f}".format((e - s + 1) / FPS),
                            "".join(sorted(set(old[s:e + 1]))), "".join(sorted(set(new[s:e + 1])))])
    print("\n{} stretches where the two files differ -> {}".format(n_str, os.path.relpath(out_str, ROOT)))
    print("per-minute table -> {}".format(os.path.relpath(out_min, ROOT)))


def _codes(labels):
    return pm.label_codes(labels)


def sanity(name, labels):
    if not os.path.exists(PUBLIC_GT):
        return
    gt = pm.label_codes(pm.read_label_csv(PUBLIC_GT))
    r = pm.evaluate(gt, _codes(labels))
    print("SANITY ONLY (public 5 min, never choose on it) {}: spellF1 {:.4f}  macroF1 {:.4f}".format(
        name, r["spell_f1_mean"], r["macro_f1"]))


def write_csv(path, labels):
    with open(path, "w", newline="") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, v) for i, v in enumerate(labels))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tracks", default=TRACKS)
    ap.add_argument("--out", default=os.path.join(RESULTS, "labeller_v6_newtracks.csv"))
    ap.add_argument("--no-goalkeepers", action="store_true", help="outfield players only")
    ap.add_argument("--skip-atd-check", action="store_true",
                    help="do not re-run v6 on the ATD (the reproduction check)")
    args = ap.parse_args()
    t0 = time.time()
    if not os.path.isfile(args.tracks):
        sys.exit("tracks file not found: {}".format(args.tracks))
    if not os.path.isfile(SAFE):
        sys.exit("needs {} to compare against".format(SAFE))

    meta_path, raw_path = require(["atd_meta", "atd_raw"])
    atd_team = team_of_each_track(meta_path)
    home, away = team_names(meta_path)
    print("H = {}, A = {}".format(home, away))
    graded_blocks = half_video_blocks(GAME)
    kickoff2 = anchor_video_frame(GAMES[GAME]["kickoff_2nd"])
    play_blocks = [(h, max(a, kickoff2) if h == 2 else a, b) for h, a, b in graded_blocks]
    tags = load_set_piece_frames()
    graded = np.zeros(N_FRAMES, dtype=bool)
    for _, a, b in graded_blocks:
        graded[a:b + 1] = True

    def run_v6(px, py, ball, team, name):
        t = time.time()
        print("running labeller v6 on {} (about a minute, no progress inside the labeller) ...".format(name),
              flush=True)
        lab = labeller_v6.label(px, py, ball, np.asarray(team), play_blocks, params={"USE_TAGS": True},
                                tags=tags)
        for h, a, b in graded_blocks:
            if h == 2:
                lab[a:kickoff2] = "D"
        print("  done in {:.0f} s".format(time.time() - t))
        return lab

    print("\n[1/4] ATD (ball for both runs; players for the check)")
    apx, apy, ball = load_atd(raw_path, len(atd_team))
    safe = pm.read_label_csv(SAFE)["label"].to_numpy().astype(str)
    if not args.skip_atd_check:
        lab_atd = run_v6(apx, apy, ball, atd_team, "the ATD players")
        tmp = args.out + ".atdcheck.tmp"
        write_csv(tmp, lab_atd)
        same = md5(tmp) == md5(SAFE)
        os.remove(tmp)
        print("REPRODUCTION CHECK: v6 on the ATD {} SAFE_SUBMISSION_labeller_v6.csv{}".format(
            "== (md5 identical)" if same else "!= ",
            "" if same else "  -> STOP: this script does not read the ATD like run_labeller_v6.py; "
                            "the comparison below would not isolate the player source"))
        if not same:
            sys.exit(1)

    print("\n[2/4] Kushagra's tracks")
    npx, npy, nteam = load_new_tracks(args.tracks, not args.no_goalkeepers)
    for name, px_, tm in (("ATD", apx, np.asarray(atd_team)), ("new", npx, nteam)):
        h, a = team_counts(px_, tm, graded)
        print("  {:>4}: players per frame home {:.1f} / away {:.1f}; a team shows more than 11 on {:.1f}% "
              "of graded frames; under 9 on {:.1f}%".format(
                  name, h.mean(), a.mean(), 100 * np.mean((h > 11) | (a > 11)), 100 * np.mean((h < 9) | (a < 9))))
    del apx, apy

    print("\n[3/4] labeller v6 on the new tracks (ATD ball, same tags, same settings)")
    lab_new = run_v6(npx, npy, ball, nteam, "the new tracks")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    write_csv(args.out, lab_new)
    problem = pm.validate_submission(pm.read_label_csv(args.out), N_FRAMES)
    print("wrote {}  validation: {}".format(os.path.relpath(args.out, ROOT),
                                           "OK" if problem is None else "FAILED - " + problem))

    print("\n[4/4] both files")
    print("-- SAFE_SUBMISSION_labeller_v6.csv (ATD players)")
    describe(safe, graded_blocks)
    print("-- labeller_v6_newtracks.csv (Kushagra's players)")
    describe(lab_new, graded_blocks)
    compare(safe, lab_new, graded_blocks,
            os.path.join(RESULTS, "compare_v6_newtracks_minutes.csv"),
            os.path.join(RESULTS, "compare_v6_newtracks_stretches.csv"))
    print()
    sanity("ATD run", safe)
    sanity("new run", lab_new)
    print("\nREMINDER: no truth exists for the full match. This shows how much and where the two differ, "
          "not which is right.")
    print("done in {:.0f} s".format(time.time() - t0))


if __name__ == "__main__":
    main()
