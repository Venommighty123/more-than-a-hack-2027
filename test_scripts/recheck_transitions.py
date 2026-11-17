"""Re-check the transitions we already watched (old tracks) on the NEW labels (Kushagra's T5 tracks).

    python test_scripts/recheck_transitions.py --list-only      # the table only, no video (seconds)
    python test_scripts/recheck_transitions.py                  # clips of the same windows, new labels
    python test_scripts/recheck_transitions.py --reel           # + one reel mp4
    python test_scripts/recheck_transitions.py --frames-from reports/labeller_viz/transitions/next20/selection.csv

WHICH FRAMES. The "anchors" are the transitions in the selection.csv files that visualize_transitions.py
wrote when we watched labeller_v6_newtracks.csv (default: every selection.csv under
reports/labeller_viz/transitions/, except this script's own folder; the files used are printed).
Each anchor is looked up in labeller_v6_newtracks.csv (--old-labels) by its frame, so its number n,
direction and half are the old ones.

GROUPING. Exactly visualize_transitions.py's rules (its own window_of / group functions are called):
each anchor gets --before / --after seconds (default 4 / 4), widened to at least --min-length (default
15 s); windows that overlap or are less than --join-gap apart (default 0) become one clip, never across
a half, and a group stops growing at --max-length (default 60 s). Same options, same defaults.

WHAT EACH CLIP SHOWS. The NEW labels (labeller_v6_newtracks_t5.csv) drawn with visualize_labeller.py's
renderer, players = Kushagra's T5 tracks (results_new/final_tracks_full_t5.parquet), ball = ATD.
Two lines on the info bar:
  NEW  the new run's change nearest the current frame ("no change in the new labels in this clip"
       when there is none), with t = time to / since it
  OLD  the old change nearest the current frame and what happened to it in the new run:
       SAME     same direction within --tolerance s (default 1 s), with the offset
       MOVED    same direction elsewhere in the clip, with the offset
       CHANGED  no same-direction change in the clip, but other changes there
       GONE     the new labels have no change at all in the clip
Drawn labels are checked against the CSV for every clip, as in visualize_labeller.py.

OUTPUT  reports/labeller_viz/transitions/recheck_t5/  (--out-dir)
  <labels>_R<clip>_<first>_<last>.mp4 (+ _spells.csv, _drawn.csv), anchors.csv (one row per old
  change: status, offset, clip), clips.csv (one row per clip: old and new changes in it), with --reel
  <labels>_recheck_reel_<k>clips.mp4.
Run time: like visualize_transitions.py for the same frames (a few minutes for ~40 changes); progress
bars for loading and every clip. Every other visualize_transitions.py option (--every, --gt,
--no-video, --backend, ...) works the same here; --type/--dir/--n/--max select nothing in this mode.

RULES. Watching to UNDERSTAND is allowed; turning what you see into labels, corrections, constants,
rules or a choice of version is not (RULES.md rule 1). Outputs are derived from match data: keep them
out of the public repository and out of any cloud AI tool.
"""

import argparse
import csv
import glob
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, os.path.join(ROOT, "evaluation"), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import visualize_transitions as vtr   # noqa: E402  (grouping, Kushagra's tracks, CLI options)

vl, lt, vt, cv2 = vtr.vl, vtr.lt, vtr.vt, vtr.cv2
FPS = vtr.FPS
RES = os.path.join(ROOT, "evaluation", "results")
NEW_LABELS = os.path.join(RES, "labeller_v6_newtracks_t5.csv")
NEW_TRACKS = os.path.join(ROOT, "results_new", "final_tracks_full_t5.parquet")
OLD_LABELS = os.path.join(RES, "labeller_v6_newtracks.csv")
TRANS_DIR = os.path.join(vl.DEFAULT_OUT, "transitions")
OUT_DIR = os.path.join(TRANS_DIR, "recheck_t5")
STATUS_BGR = {"SAME": (80, 220, 80), "MOVED": (0, 200, 255), "CHANGED": (0, 140, 255), "GONE": (60, 60, 255)}


def parse():
    """This script's own options; everything else goes to visualize_transitions.py's parser, with the
    new labels / tracks / output folder as defaults."""
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--frames-from", nargs="+", help="selection.csv file(s) with the old changes to re-check")
    ap.add_argument("--old-labels", default=OLD_LABELS)
    mine, rest = ap.parse_known_args()
    if "-h" in rest or "--help" in rest:
        print(__doc__)
        print("own options: --frames-from CSV [CSV ...]   --old-labels CSV (default labeller_v6_newtracks.csv)\n"
              "all visualize_transitions.py options also apply (shown below)\n")
    for flag, val in (("--labels", NEW_LABELS), ("--tracks", NEW_TRACKS), ("--out-dir", OUT_DIR),
                      ("--compare", "none")):
        if flag not in rest:
            rest += [flag, val]
    sys.argv = [sys.argv[0]] + rest
    return mine, vtr.parse_args()


def anchor_files(mine):
    if mine.frames_from:
        return mine.frames_from
    files = sorted(glob.glob(os.path.join(TRANS_DIR, "**", "selection.csv"), recursive=True))
    return [f for f in files if os.path.normpath(OUT_DIR) not in os.path.normpath(f)]


def read_anchors(files, old_trans):
    by_frame = {t["frame"]: t for t in old_trans}
    seen, out, missing = set(), [], 0
    for path in files:
        with open(path, newline="") as handle:
            rows = list(csv.DictReader(handle))
        k = 0
        for r in rows:
            f = int(r["frame"])
            if f in seen:
                continue
            t = by_frame.get(f)
            if t is None:
                missing += 1
                continue
            seen.add(f)
            out.append(dict(t))
            k += 1
        print("  {:<70} {} change(s)".format(os.path.relpath(path, ROOT), k))
    if missing:
        print("  note: {} row(s) not found as a change in the old labels (skipped)".format(missing))
    out.sort(key=lambda t: t["frame"])
    for k, t in enumerate(out, 1):
        t["k"] = k
    return out


def mark(anchors, clips, new_trans, tol):
    for c in clips:
        c["changes"] = [t for t in new_trans if c["first"] <= t["frame"] <= c["last"]]
        for a in c["sel"]:
            same = [t for t in c["changes"] if t["direction"] == a["direction"]]
            if same:
                t = min(same, key=lambda x: abs(x["frame"] - a["frame"]))
                off = (t["frame"] - a["frame"]) / float(FPS)
                a["status"] = "SAME" if abs(t["frame"] - a["frame"]) <= tol else "MOVED"
                a["offset_s"] = round(off, 2)
                a["new_n"] = t["n"]
            else:
                a["status"] = "CHANGED" if c["changes"] else "GONE"
                a["offset_s"], a["new_n"] = "", ""
    for c in clips:
        for t in c["changes"]:
            t["extra"] = not any(a["direction"] == t["direction"] and abs(a["frame"] - t["frame"]) <= tol
                                 for a in c["sel"])


class RecheckWriter:
    """Writes the NEW and OLD lines on each canvas (replaces visualize_transitions' line)."""

    def __init__(self, inner, clip, names, first, every):
        self.inner, self.c, self.names = inner, clip, names
        self.f, self.every = first, every

    def isOpened(self):
        return self.inner.isOpened()

    def release(self):
        self.inner.release()

    def write(self, canvas):
        f, c, x = self.f, self.c, vl.PANEL_W + 12
        if c["changes"]:
            t = min(c["changes"], key=lambda z: abs(z["frame"] - f))
            on = abs(f - t["frame"]) < max(1, self.every)
            txt = "NEW #{} {}  {} -> {}  at {}  t = {:+.1f} s{}{}".format(
                t["n"], t["direction"], vl.team_word(t["from"], self.names), vl.team_word(t["to"], self.names),
                t["video_clock"], (f - t["frame"]) / float(FPS), "  (not in old)" if t.get("extra") else "",
                "  << CHANGE" if on else "")
            vt.put(canvas, txt[:100], (x, vl.PANEL_H + 80), 0.48, (0, 255, 255) if on else (255, 255, 255),
                   2 if on else 1)
        else:
            vt.put(canvas, "NEW: no change in the new labels in this clip", (x, vl.PANEL_H + 80), 0.48,
                   (200, 200, 200), 1)
        a = min(c["sel"], key=lambda z: abs(z["frame"] - f))
        on = abs(f - a["frame"]) < max(1, self.every)
        st = a["status"] + ("" if a["offset_s"] == "" else " {:+.1f} s".format(a["offset_s"]))
        txt = "OLD #{} {}  at {}  t = {:+.1f} s  ->  {}{}".format(
            a["n"], a["direction"], a["video_clock"], (f - a["frame"]) / float(FPS), st,
            "  << OLD CHANGE" if on else "")
        vt.put(canvas, txt[:100], (x, vl.PANEL_H + 98), 0.44, STATUS_BGR[a["status"]], 2 if on else 1)
        self.inner.write(canvas)
        self.f += self.every


def make_reel(clips, names, out_path):
    cap = cv2.VideoCapture(clips[0]["mp4"])
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
    t0 = time.time()
    for i, c in enumerate(clips):
        card = np.zeros((h, w, 3), dtype=np.uint8)
        card[:] = (30, 30, 30)
        vt.put(card, "CLIP {} of {}   frames {}..{}   {:.1f} s   half {}".format(
            c["c"], len(clips), c["first"], c["last"], (c["last"] - c["first"] + 1) / float(FPS), c["half"]),
            (60, 90), 1.1, (0, 255, 255), 2)
        vt.put(card, "old changes re-checked: {}   changes in the new labels: {}".format(
            len(c["sel"]), len(c["changes"])), (60, 150), 0.8, (255, 255, 255), 2)
        for j, a in enumerate(c["sel"][:14]):
            vt.put(card, "OLD #{:<4} {}  {:<5} -> {}{}".format(
                a["n"], a["video_clock"], a["direction"], a["status"],
                "" if a["offset_s"] == "" else " {:+.1f} s".format(a["offset_s"])),
                (80, 195 + 34 * j), 0.65, STATUS_BGR[a["status"]], 2)
        for _ in range(int(1.5 * FPS)):
            writer.write(card)
        cap = cv2.VideoCapture(c["mp4"])
        while True:
            ok, img = cap.read()
            if not ok:
                break
            if img.shape[1] != w or img.shape[0] != h:
                img = cv2.resize(img, (w, h))
            writer.write(img)
        cap.release()
        vl.progress("joining reel", i, len(clips), t0)
    writer.release()


def main():
    mine, args = parse()
    labels = vl.load_labels(args.labels)
    old = vl.load_labels(mine.old_labels)
    new_trans, old_trans = lt.find_transitions(labels), lt.find_transitions(old)
    meta_path, raw_path = vtr.require(["atd_meta", "atd_raw"])
    names = vtr.team_names(meta_path)
    print("new labels {}: {} transitions | old labels {}: {} transitions".format(
        os.path.relpath(args.labels, ROOT), len(new_trans), os.path.relpath(mine.old_labels, ROOT), len(old_trans)))
    print("old changes to re-check, from:")
    anchors = read_anchors(anchor_files(mine), old_trans)
    if not anchors:
        sys.exit("no old changes found; pass --frames-from <selection.csv>")
    clips = vtr.group(anchors, args)
    tol = int(round(args.tolerance * FPS))
    mark(anchors, clips, new_trans, tol)

    print("\n{} old change(s) -> {} clip(s) (min {:.0f} s, max {:.0f} s{})".format(
        len(anchors), len(clips), args.min_length, args.max_length, ", no grouping" if args.no_group else ", grouped"))
    print("  {:>4} {:>8} {:>10} {:<5} {:<34} {:>8} {:>8} {:>5}".format(
        "old n", "frame", "clock", "dir", "teams", "status", "offset", "clip"))
    for a in anchors:
        print("  {:>4} {:>8} {:>10} {:<5} {:<34} {:>8} {:>8} {:>5}".format(
            a["n"], a["frame"], a["video_clock"], a["direction"],
            "{} -> {}".format(vl.team_word(a["from"], names), vl.team_word(a["to"], names))[:34], a["status"],
            "" if a["offset_s"] == "" else "{:+.1f}s".format(a["offset_s"]), a["clip"]))
    cnt = {s: sum(a["status"] == s for a in anchors) for s in ("SAME", "MOVED", "CHANGED", "GONE")}
    extra = sum(t["extra"] for c in clips for t in c["changes"])
    print("\nSUMMARY: of {} old changes, SAME {} | MOVED {} | CHANGED {} | GONE {};  "
          "new changes in these clips with no old one within {:.0f} s: {}".format(
              len(anchors), cnt["SAME"], cnt["MOVED"], cnt["CHANGED"], cnt["GONE"], args.tolerance, extra))
    print("\n  {:>4} {:>8} {:>8} {:>8}  {:<28} {}".format("clip", "first", "last", "seconds", "old n", "new n (* = not in old)"))
    for c in clips:
        print("  {:>4} {:>8} {:>8} {:>8.1f}  {:<28} {}".format(
            c["c"], c["first"], c["last"], (c["last"] - c["first"] + 1) / float(FPS),
            " ".join(str(a["n"]) for a in c["sel"])[:28],
            " ".join(str(t["n"]) + ("*" if t["extra"] else "") for t in c["changes"]) or "-"))

    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.labels))[0]
    for c in clips:
        c["mp4"] = os.path.join(args.out_dir, "{}_R{}_{}_{}.mp4".format(stem, c["c"], c["first"], c["last"]))
    with open(os.path.join(args.out_dir, "anchors.csv"), "w", newline="") as handle:
        w = csv.writer(handle)
        w.writerow(["old_n", "frame", "video_clock", "half", "direction", "status", "offset_s", "new_n", "clip", "mp4"])
        for a in anchors:
            w.writerow([a["n"], a["frame"], a["video_clock"], a["half"], a["direction"], a["status"], a["offset_s"],
                        a["new_n"], a["clip"], clips[a["clip"] - 1]["mp4"]])
    with open(os.path.join(args.out_dir, "clips.csv"), "w", newline="") as handle:
        w = csv.writer(handle)
        w.writerow(["clip", "first_frame", "last_frame", "start_clock", "seconds", "half", "old_changes",
                    "new_changes", "new_not_in_old", "mp4"])
        for c in clips:
            w.writerow([c["c"], c["first"], c["last"], vt.fmt_time(c["first"]),
                        "{:.2f}".format((c["last"] - c["first"] + 1) / float(FPS)), c["half"],
                        " ".join(str(a["n"]) for a in c["sel"]), " ".join(str(t["n"]) for t in c["changes"]),
                        " ".join(str(t["n"]) for t in c["changes"] if t["extra"]), c["mp4"]])
    print("\nanchors.csv / clips.csv -> {}".format(os.path.abspath(args.out_dir)))
    if args.list_only:
        return

    new = vtr.NewTracks(args.tracks, keep_gk=not args.no_goalkeepers) if args.players == "kushagra" else None
    atd_team = vtr.team_of_each_track(meta_path)
    spells = vl.MatchSpells(labels)
    real_writer = cv2.VideoWriter
    done, all_ok = [], True
    try:
        for c in clips:
            a, b, mp4 = c["first"], c["last"], c["mp4"]
            n = b - a + 1
            print("\n=== clip {} of {}: frames {}..{} ({:.1f} s), old {} | new {} ===".format(
                c["c"], len(clips), a, b, n / float(FPS),
                ", ".join("#{} {} {}".format(x["n"], x["direction"], x["status"]) for x in c["sel"]),
                ", ".join("#{} {}".format(x["n"], x["direction"]) for x in c["changes"]) or "none"))
            if new is not None:
                players, team = new.window(a, n)
                _, ball = vtr._orig_read_tracking(raw_path, a, n, 1)
                vt.read_tracking = lambda *_a, _p=players, _b=ball: (_p, _b)
            else:
                team = atd_team
                vt.read_tracking = vtr._orig_read_tracking
            cv2.VideoWriter = (lambda *wa, _c=c, _a=a: RecheckWriter(real_writer(*wa), _c, names, _a,
                                                                     max(1, args.every)))
            base = os.path.splitext(mp4)[0]
            vl.write_spells(base + "_spells.csv", labels, a, b, names)
            drawn = vl.render(args, labels, spells, a, b, names, team, raw_path, mp4)
            cv2.VideoWriter = real_writer
            all_ok &= vl.check_drawn(drawn, labels, base + "_drawn.csv")
            print("wrote {}".format(os.path.abspath(mp4)))
            done.append(c)
    finally:
        cv2.VideoWriter = real_writer
        vt.read_tracking = vtr._orig_read_tracking

    if args.reel and done and not args.no_video:
        reel = os.path.join(args.out_dir, "{}_recheck_reel_{}clips.mp4".format(stem, len(done)))
        make_reel(done, names, reel)
        print("wrote reel {}".format(os.path.abspath(reel)))
    print("\n{} clip(s) for {} old change(s) in {}".format(len(done), len(anchors), os.path.abspath(args.out_dir)))
    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
