"""Frame ranges of every dead-ball stretch (D) and of every A<->D / H<->D transition in a labeller CSV.

    python test_scripts/list_dead_transitions.py                       # labeller_v6_newtracks.csv
    python test_scripts/list_dead_transitions.py --type AD             # only A<->D in the transition file
    python test_scripts/list_dead_transitions.py --labels evaluation/results/SAFE_SUBMISSION_labeller_v6.csv

No video. Reads only the labels CSV (and the --compare CSV), runs in a few seconds.

OUTPUT 1  evaluation/results/dead_stretches_<labels>.csv   one row per D stretch (graded halves only)
  d               stretch number in match order
  first_frame, last_frame   inclusive frames (= CSV rows = video frames) labelled D
  frames, seconds, start_clock, end_clock, half
  before / after  label just before / after the stretch: H, A, or "half_start" / "half_end" when the
                  stretch touches the start or end of a graded half
  entry / exit    the transition into / out of it (e.g. A->D, D->H), empty at a half boundary
  in_compare      1 if the --compare file (default: SAFE file) has a D stretch overlapping this one
  compare_first, compare_last   that overlapping D stretch in the compare file (the longest overlap)

OUTPUT 2  evaluation/results/dead_transitions_<labels>.csv   one row per A<->D / H<->D change
  n               number in list_transitions.py's list (same numbering: all H/A/D changes)
  frame           first frame of the NEW label; the change happens between frame-1 and frame
  last_old_frame  = frame-1 (last frame of the old label)
  type, direction (A<->D: A->D, D->A;  H<->D: H->D, D->H), kind (stoppage = into D, restart = out of D)
  old_first, old_last, old_seconds    the run of the old label that ends at the change
  new_first, new_last, new_seconds    the run of the new label that starts at the change
  d               the D stretch (OUTPUT 1) this change enters or leaves
  in_compare, compare_offset_s        same change (same direction) in the compare file within 1 s
  clock, half

RULES. This is what the labeller wrote, not truth. Derived from match data: keep it out of the public
repository and out of cloud AI tools.
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

import list_transitions as lt  # noqa: E402  (labels reader, transition finder, same numbering)
from games import GAME, half_video_blocks  # noqa: E402

FPS = lt.FPS
RES = lt.RES
TYPE_CODES = {"AD": "D<->A", "HD": "H<->D"}
S_FIELDS = ["d", "first_frame", "last_frame", "frames", "seconds", "start_clock", "end_clock", "half",
            "before", "after", "entry", "exit", "in_compare", "compare_first", "compare_last"]
T_FIELDS = ["n", "frame", "last_old_frame", "clock", "half", "type", "direction", "kind",
            "old_first", "old_last", "old_seconds", "new_first", "new_last", "new_seconds", "d",
            "in_compare", "compare_offset_s"]


def secs(frames):
    return round(frames / float(FPS), 2)


def dead_stretches(labels):
    """[dict] one per maximal D run inside a graded half."""
    out = []
    for half, a, b in half_video_blocks(GAME):
        seg = labels[a:b + 1]
        cut = np.flatnonzero(seg[1:] != seg[:-1]) + 1
        starts, ends = np.r_[0, cut], np.r_[cut, len(seg)] - 1
        for j, (s, e) in enumerate(zip(starts, ends)):
            if seg[s] != "D":
                continue
            before = str(seg[starts[j - 1]]) if j > 0 else "half_start"
            after = str(seg[starts[j + 1]]) if j + 1 < len(starts) else "half_end"
            f0, f1 = a + int(s), a + int(e)
            out.append({"first_frame": f0, "last_frame": f1, "frames": f1 - f0 + 1, "seconds": secs(f1 - f0 + 1),
                        "start_clock": lt.clock(f0), "end_clock": lt.clock(f1 + 1), "half": half,
                        "before": before, "after": after,
                        "entry": "{}->D".format(before) if before in ("H", "A") else "",
                        "exit": "D->{}".format(after) if after in ("H", "A") else ""})
    for k, d in enumerate(out, 1):
        d["d"] = k
    return out


def mark_stretch_compare(mine, other):
    """in_compare: an overlapping D stretch exists in the other file (longest overlap reported)."""
    if not other:
        for d in mine:
            d["in_compare"], d["compare_first"], d["compare_last"] = 0, "", ""
        return
    o0 = np.array([o["first_frame"] for o in other])
    o1 = np.array([o["last_frame"] for o in other])
    for d in mine:
        ov = np.minimum(o1, d["last_frame"]) - np.maximum(o0, d["first_frame"]) + 1
        j = int(np.argmax(ov))
        if ov[j] > 0:
            d["in_compare"], d["compare_first"], d["compare_last"] = 1, int(o0[j]), int(o1[j])
        else:
            d["in_compare"], d["compare_first"], d["compare_last"] = 0, "", ""


def dead_transitions(labels, trans, stretches, kinds):
    """The A<->D / H<->D rows of the full transition list, with the old and new runs as frame ranges."""
    first_of = {d["first_frame"]: d["d"] for d in stretches}
    last_of = {d["last_frame"]: d["d"] for d in stretches}
    out = []
    for t in trans:
        if t["type"] not in kinds:
            continue
        f = t["frame"]
        old_n = int(round(t["before_s"] * FPS))
        new_n = int(round(t["after_s"] * FPS))
        into_d = t["to"] == "D"
        out.append({"n": t["n"], "frame": f, "last_old_frame": f - 1, "clock": t["video_clock"], "half": t["half"],
                    "type": t["type"], "direction": t["direction"], "kind": "stoppage" if into_d else "restart",
                    "old_first": f - old_n, "old_last": f - 1, "old_seconds": t["before_s"],
                    "new_first": f, "new_last": f + new_n - 1, "new_seconds": t["after_s"],
                    "d": first_of.get(f, "") if into_d else last_of.get(f - 1, ""),
                    "in_compare": t.get("in_compare", 0), "compare_offset_s": t.get("compare_offset_s", "")})
    return out


def write(path, rows, fields):
    with open(path, "w", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--labels", default=lt.DEFAULT_LABELS, help="labeller CSV (default: labeller_v6_newtracks.csv)")
    ap.add_argument("--compare", default=lt.DEFAULT_COMPARE, help="second CSV (default: SAFE file); 'none' to skip")
    ap.add_argument("--tolerance", type=float, default=1.0, help="seconds for in_compare on transitions (default 1)")
    ap.add_argument("--type", nargs="+", choices=sorted(TYPE_CODES), default=sorted(TYPE_CODES),
                    help="AD and/or HD (default both)")
    ap.add_argument("--show", type=int, default=15, help="rows of each table to print (default 15, 0 = all)")
    args = ap.parse_args()

    labels = lt.load_labels(args.labels)
    trans = lt.find_transitions(labels)
    stretches = dead_stretches(labels)
    has_compare = args.compare.lower() != "none" and os.path.isfile(args.compare)
    if has_compare:
        other_labels = lt.load_labels(args.compare)
        lt.mark_compare(trans, lt.find_transitions(other_labels), int(round(args.tolerance * FPS)))
        mark_stretch_compare(stretches, dead_stretches(other_labels))
    else:
        mark_stretch_compare(stretches, [])
        for t in trans:
            t["in_compare"], t["compare_offset_s"] = 0, ""
    kinds = {TYPE_CODES[c] for c in args.type}
    dts = dead_transitions(labels, trans, stretches, kinds)

    stem = os.path.splitext(os.path.basename(args.labels))[0]
    p1 = os.path.join(RES, "dead_stretches_{}.csv".format(stem))
    p2 = os.path.join(RES, "dead_transitions_{}.csv".format(stem))
    write(p1, stretches, S_FIELDS)
    write(p2, dts, T_FIELDS)

    # -- summary -------------------------------------------------------------------------------
    sec = np.array([d["seconds"] for d in stretches]) if stretches else np.zeros(1)
    print("labels: {}".format(args.labels))
    print("\n== 1. D stretches ==")
    print("{} stretches, {:.1f} min of D in total; length median {:.1f} s (p10 {:.1f}, p90 {:.1f}, longest {:.1f})".format(
        len(stretches), sec.sum() / 60.0, np.median(sec), np.percentile(sec, 10), np.percentile(sec, 90), sec.max()))
    for half in (1, 2):
        print("  half {}: {} stretches".format(half, sum(d["half"] == half for d in stretches)))
    pairs = {}
    for d in stretches:
        key = "{} -> D -> {}".format(d["before"], d["after"])
        pairs[key] = pairs.get(key, 0) + 1
    print("  what surrounds them (before -> D -> after):")
    for key, c in sorted(pairs.items(), key=lambda kv: -kv[1]):
        print("    {:<28} {:>4}".format(key, c))
    if has_compare:
        print("  overlapping a D stretch in the compare file: {} of {}".format(
            sum(d["in_compare"] for d in stretches), len(stretches)))
    show = args.show if args.show > 0 else 10 ** 9
    print("\n  {:>4} {:>8} {:>8} {:>8} {:>10} {:>4}  {:<11} {:<11} {:>8}".format(
        "d", "first", "last", "seconds", "clock", "half", "before", "after", "compare"))
    for d in stretches[:show]:
        print("  {:>4} {:>8} {:>8} {:>8.1f} {:>10} {:>4}  {:<11} {:<11} {:>8}".format(
            d["d"], d["first_frame"], d["last_frame"], d["seconds"], d["start_clock"], d["half"],
            d["before"], d["after"], ("yes" if d["in_compare"] else "no") if has_compare else "-"))
    if len(stretches) > show:
        print("  ... {} more in {}".format(len(stretches) - show, p1))

    print("\n== 2. transitions: {} ==".format(" and ".join(sorted(kinds))))
    for kind in sorted(kinds):
        for dr in sorted({t["direction"] for t in dts if t["type"] == kind}):
            rows = [t for t in dts if t["direction"] == dr]
            print("  {:<6} {:<5} {:>4}  ({}; in compare {})".format(
                kind, dr, len(rows), "stoppage" if dr.endswith("D") else "restart",
                sum(t["in_compare"] for t in rows) if has_compare else "-"))
    print("\n  {:>4} {:>8} {:<5} {:>19} {:>19} {:>10} {:>4} {:>8}".format(
        "n", "frame", "dir", "old run (frames)", "new run (frames)", "clock", "d", "compare"))
    for t in dts[:show]:
        print("  {:>4} {:>8} {:<5} {:>19} {:>19} {:>10} {:>4} {:>8}".format(
            t["n"], t["frame"], t["direction"], "{}..{}".format(t["old_first"], t["old_last"]),
            "{}..{}".format(t["new_first"], t["new_last"]), t["clock"], t["d"],
            ("yes" if t["in_compare"] else "no") if has_compare else "-"))
    if len(dts) > show:
        print("  ... {} more in {}".format(len(dts) - show, p2))
    print("\nwrote {}\nwrote {}".format(p1, p2))


if __name__ == "__main__":
    main()
