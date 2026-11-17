"""List every label transition in a labeller CSV: H<->A, D<->A and H<->D.

    python test_scripts/list_transitions.py                                   # labeller_v6_newtracks.csv
    python test_scripts/list_transitions.py --labels evaluation/results/SAFE_SUBMISSION_labeller_v6.csv

WHAT A TRANSITION IS. Frame f where the label changes: label[f-1] != label[f], both frames inside the
same graded half (frames outside the halves are X and are never counted). "frame" in the list is the
FIRST frame of the new label (= CSV row = video frame), so the change is seen between frame-1 and frame.
  type       H<->A  possession change between the teams (H->A or A->H)
             D<->A  Athletic restart after a stoppage (D->A) or a stoppage while Athletic had it (A->D)
             H<->D  Barcelona restart (D->H) or a stoppage while Barcelona had it (H->D)
  before_s   how long the old label had run;   after_s  how long the new label runs
  in_compare whether the --compare file (default: the SAFE file, ATD players) has the same change
             (same from -> to) within --tolerance seconds (default 1.0 s), and how far off it is.
             Nothing is scored: it only says whether both files put this change at about the same time.

OUTPUT  evaluation/results/transitions_<labels>.csv (one row per transition) and a summary:
        counts per type and direction, per half, how many are short (new label < 2 s), how many the
        compare file shares. visualize_transitions.py reads this list (or rebuilds it the same way).

Runs in a few seconds (no progress bar needed). Reads only the two CSVs, no match data.
RULES. This lists what the labeller wrote; it is not truth. Keep the output out of the public repo
and out of cloud AI tools (derived from match data).
"""

import argparse
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, os.path.join(ROOT, "evaluation"), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from games import GAME, GAMES, half_video_blocks  # noqa: E402

FPS = GAMES[GAME]["fps"]
N_FRAMES = GAMES[GAME]["n_video_frames"]
RES = os.path.join(ROOT, "evaluation", "results")
DEFAULT_LABELS = os.path.join(RES, "labeller_v6_newtracks.csv")
DEFAULT_COMPARE = os.path.join(RES, "SAFE_SUBMISSION_labeller_v6.csv")
TYPES = {frozenset("HA"): "H<->A", frozenset("DA"): "D<->A", frozenset("HD"): "H<->D"}
FIELDS = ["n", "frame", "video_clock", "half", "from", "to", "type", "direction",
          "before_s", "after_s", "short", "in_compare", "compare_offset_s"]


def clock(f):
    s = f / float(FPS)
    return "{:d}:{:02d}.{:d}".format(int(s // 60), int(s % 60), int((s * 10) % 10))


def load_labels(path):
    """The CSV as N_FRAMES labels (same checks as visualize_labeller.load_labels)."""
    if not os.path.isfile(path):
        sys.exit("labels file not found: {}".format(path))
    with open(path, newline="") as handle:
        reader = csv.reader(handle)
        header = [h.strip().lower() for h in next(reader)]
        if header[:2] != ["frame", "label"]:
            sys.exit("{}: header must be 'frame,label', found {}".format(path, header))
        rows = [(int(r[0]), r[1].strip().upper()) for r in reader if r]
    if len(rows) != N_FRAMES or [r[0] for r in rows] != list(range(N_FRAMES)):
        sys.exit("{}: need frames 0..{} in order, one row each".format(path, N_FRAMES - 1))
    labels = np.array([r[1] for r in rows], dtype="<U1")
    bad = sorted(set(labels) - set("HADX"))
    if bad:
        sys.exit("{}: unexpected labels {}".format(path, bad))
    return labels


def find_transitions(labels):
    """[dict] one per change inside a graded half, in match order. Every change between H, A and D
    has exactly one of the three types; changes to or from X are not transitions."""
    out = []
    for half, a, b in half_video_blocks(GAME):
        seg = labels[a:b + 1]
        cut = np.flatnonzero(seg[1:] != seg[:-1]) + 1            # first frame of each new run
        starts = np.r_[0, cut]
        ends = np.r_[cut, len(seg)]                               # exclusive
        for j in range(1, len(starts)):
            old, new = str(seg[starts[j - 1]]), str(seg[starts[j]])
            kind = TYPES.get(frozenset(old + new))
            if kind is None:                                      # X involved
                continue
            f = a + int(starts[j])
            out.append({"frame": f, "video_clock": clock(f), "half": half, "from": old, "to": new,
                        "type": kind, "direction": "{}->{}".format(old, new),
                        "before_s": round((starts[j] - starts[j - 1]) / float(FPS), 2),
                        "after_s": round((ends[j] - starts[j]) / float(FPS), 2)})
    for k, t in enumerate(out, 1):
        t["n"] = k
        t["short"] = int(t["after_s"] < 2.0)
    return out


def mark_compare(trans, other, tol_frames):
    """in_compare / compare_offset_s: the nearest same-direction change in the other file."""
    by_dir = {}
    for t in other:
        by_dir.setdefault(t["direction"], []).append(t["frame"])
    by_dir = {k: np.array(v) for k, v in by_dir.items()}
    for t in trans:
        cand = by_dir.get(t["direction"])
        if cand is None or not len(cand):
            t["in_compare"], t["compare_offset_s"] = 0, ""
            continue
        d = cand - t["frame"]
        j = int(np.argmin(np.abs(d)))
        t["in_compare"] = int(abs(d[j]) <= tol_frames)
        t["compare_offset_s"] = round(d[j] / float(FPS), 2) if t["in_compare"] else ""


def summary(trans, name, has_compare):
    print("\n== transitions in {} ==".format(name))
    print("total {}  (inside the graded halves; changes to/from X not counted)".format(len(trans)))
    head = "  {:<7} {:<6} {:>6} {:>8} {:>8} {:>13} {:>12}".format(
        "type", "dir", "count", "half 1", "half 2", "new < 2 s", "in compare" if has_compare else "")
    print(head)
    for kind in ("H<->A", "D<->A", "H<->D"):
        for d in sorted({t["direction"] for t in trans if t["type"] == kind}):
            rows = [t for t in trans if t["direction"] == d]
            print("  {:<7} {:<6} {:>6} {:>8} {:>8} {:>13} {:>12}".format(
                kind, d, len(rows), sum(t["half"] == 1 for t in rows), sum(t["half"] == 2 for t in rows),
                sum(t["short"] for t in rows),
                "{} ({:.0f}%)".format(sum(t["in_compare"] for t in rows),
                                      100.0 * np.mean([t["in_compare"] for t in rows])) if has_compare else ""))
        rows = [t for t in trans if t["type"] == kind]
        if rows:
            ab = np.array([t["after_s"] for t in rows])
            print("  {:<7} all    {:>6}   new label runs median {:.1f} s (p10 {:.1f}, p90 {:.1f})".format(
                kind, len(rows), np.median(ab), np.percentile(ab, 10), np.percentile(ab, 90)))


def write(path, trans):
    with open(path, "w", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=FIELDS)
        w.writeheader()
        for t in trans:
            w.writerow({k: t.get(k, "") for k in FIELDS})


def read(path):
    """The list as written by write() (used by visualize_transitions.py)."""
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    for r in rows:
        for k in ("n", "frame", "half", "short", "in_compare"):
            r[k] = int(r[k]) if r.get(k, "") != "" else 0
        for k in ("before_s", "after_s"):
            r[k] = float(r[k])
    return rows


def default_out(labels_path):
    stem = os.path.splitext(os.path.basename(labels_path))[0]
    return os.path.join(RES, "transitions_{}.csv".format(stem))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--labels", default=DEFAULT_LABELS, help="labeller CSV (default: labeller_v6_newtracks.csv)")
    ap.add_argument("--compare", default=DEFAULT_COMPARE,
                    help="second CSV to check each change against (default: SAFE file); 'none' to skip")
    ap.add_argument("--tolerance", type=float, default=1.0, help="seconds for in_compare (default 1.0)")
    ap.add_argument("--out", help="default evaluation/results/transitions_<labels>.csv")
    args = ap.parse_args()

    labels = load_labels(args.labels)
    trans = find_transitions(labels)
    has_compare = args.compare.lower() != "none" and os.path.isfile(args.compare)
    if has_compare:
        other = find_transitions(load_labels(args.compare))
        mark_compare(trans, other, int(round(args.tolerance * FPS)))
        print("compare file {}: {} transitions".format(args.compare, len(other)))
    else:
        for t in trans:
            t["in_compare"], t["compare_offset_s"] = 0, ""
        if args.compare.lower() != "none":
            print("note: compare file not found ({}), in_compare left empty".format(args.compare))
    summary(trans, args.labels, has_compare)

    out = args.out or default_out(args.labels)
    write(out, trans)
    print("\nwrote {} ({} rows)".format(out, len(trans)))
    print("first rows:")
    print("  {:>4} {:>8} {:>10} {:>4} {:<6} {:>8} {:>8} {:>6}".format(
        "n", "frame", "clock", "half", "dir", "before", "after", "compare"))
    for t in trans[:12]:
        print("  {:>4} {:>8} {:>10} {:>4} {:<6} {:>7.1f}s {:>7.1f}s {:>6}".format(
            t["n"], t["frame"], t["video_clock"], t["half"], t["direction"], t["before_s"], t["after_s"],
            ("yes" if t["in_compare"] else "no") if has_compare else "-"))
    print("watch them:  python test_scripts/visualize_transitions.py --type HA --max 10   (HA, DA, HD; or --dir H2A)")


if __name__ == "__main__":
    main()
