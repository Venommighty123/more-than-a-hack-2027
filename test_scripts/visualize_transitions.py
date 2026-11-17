"""Watch the label transitions (H<->A, D<->A, H<->D) of a labeller CSV on the tactical video.

    python test_scripts/visualize_transitions.py --list-only                    # the selection, no video
    python test_scripts/visualize_transitions.py --type HA --max 10              # first 10 possession changes
    python test_scripts/visualize_transitions.py --dir D2A H2D --max 20 --reel   # 20 clips + one reel mp4
    python test_scripts/visualize_transitions.py --n 17 42 108                   # these transitions (n in the list)
    python test_scripts/visualize_transitions.py --type HA --only-new            # changes the SAFE file lacks
    python test_scripts/visualize_transitions.py --range 74185 77184             # every transition in a window

WHAT IT DOES. It takes the transitions exactly as list_transitions.py finds them (same code, rebuilt
from --labels so the list can never be out of date), keeps those you select, and GROUPS them into
clips (below). Each clip is rendered with visualize_labeller.py's renderer: label border and badge, change banner, info bar, timeline. Nothing is recomputed; the label
drawn on frame f is row f of the CSV, and "CHECK: drawn labels match CSV" is printed for every clip.
Added on top of visualize_labeller.py:
  * Players drawn are Kushagra's tracks (results/final_tracks_full.parquet), the players the labeller
    run used, coloured by HIS team label. The ball is the ATD ball (as in the labeller run).
    --players atd draws the ATD players instead (same as visualize_labeller.py).
  * A transition line on the right of the info bar: "TRANSITION #n  H->A  Barcelona -> Athletic at
    <clock>   t = -1.2 s" (time to / since the change), turning yellow on the change frame.
  * --reel joins the clips into one mp4 with a one-second title card before each.

GROUPING (one clip per group, not per transition). Each selected change gets a window from --before
seconds before to --after seconds after it, widened evenly to at least --min-length seconds (default
15). Windows that overlap, or are less than --join-gap seconds apart (default 0), become ONE clip, so 4
changes inside 10 s give one clip of at least 15 s that shows all four. Windows never cross a half
boundary (a window near the start or end of a half is shifted inside it; a clip is only shorter than
--min-length if the half itself is). A group stops growing at --max-length seconds (default 60), so a
long run of quick changes is split into several clips of up to a minute that may overlap a little. The overlay line follows the change nearest the current frame and
says which of the clip's changes it is ("change 2 of 4 in this clip"). Changes that fall inside a clip
but were not selected (e.g. a shared change in an --only-new run) are shown too and marked
"(not selected)". --no-group gives the old behaviour: one clip per change.

  select   --type HA DA HD (both directions)   --dir H2A A2H D2A A2D H2D D2H   --n ...   --from-n 21   --range FIRST LAST
           --only-new (no same change in --compare within 1 s)   --only-shared   --only-short (new label < 2 s)
           --max N transitions (default 10; 0 = all); clips are fewer when changes are grouped
  clip     --before 4 --after 4 (seconds), --min-length 15, --join-gap 0, --max-length 60, --no-group,
           --every N, --gt, --ids, --no-overlay, --no-video ... as in
           visualize_labeller.py
OUTPUT  reports/labeller_viz/transitions/<labels>_C<clip>_<first>_<last>_<n>chg.mp4 (+ _spells.csv,
        _drawn.csv), selection.csv (one row per selected change, with its clip), clips.csv (one row per
        clip: frames, seconds, changes in it), and with --reel <labels>_reel_<k>clips_<n>chg.mp4.
Run time: about as long as visualize_labeller.py for the same number of frames (15 s clip = 375
frames, 20 clips ~ 7,500 frames, a few minutes); progress bars for loading and for every clip.

RULES. Watching to UNDERSTAND errors is allowed. Turning what you see on BAR-ATH into labels,
corrections, constants, rules or a choice of version is not (RULES.md rule 1). Outputs are derived
from match data: keep them out of the public repository and out of any cloud AI tool.

Needs what visualize_labeller.py needs, plus pyarrow for the parquet (pip install pyarrow).
"""

import argparse
import csv
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, os.path.join(ROOT, "evaluation"), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import visualize_labeller as vl   # noqa: E402  (renderer, label checks, helpers)
import list_transitions as lt     # noqa: E402  (the transition list, same code)

vt, cv2 = vl.vt, vl.cv2
from games import GAME, require, half_video_blocks  # noqa: E402
from starter.baseline import team_of_each_track, team_names  # noqa: E402

FPS, N_FRAMES = vl.FPS, vl.N_FRAMES
DEFAULT_TRACKS = os.path.join(ROOT, "results", "final_tracks_full.parquet")
TRACK_PITCH = (105.3, 68.0)        # Kushagra's x_m / y_m pitch (same as run_labeller_v6_newtracks.py)
DIR_CODES = {"H2A": "H->A", "A2H": "A->H", "D2A": "D->A", "A2D": "A->D", "H2D": "H->D", "D2H": "D->H"}
TYPE_CODES = {"HA": "H<->A", "DA": "D<->A", "HD": "H<->D"}
_orig_read_tracking = vt.read_tracking


# -- Kushagra's players ---------------------------------------------------------------------------

class NewTracks:
    """Kushagra's player/goalkeeper rows with team H/A (as run_labeller_v6_newtracks.py uses them),
    cut per clip into (n, slots, 3) normalised x, y, NaN speed, plus each slot's team (0 H, 1 A)."""

    def __init__(self, path, keep_gk=True):
        try:
            import pandas as pd
            t = pd.read_parquet(path, columns=["frame", "track_id", "role", "team", "x_m", "y_m"])
        except ImportError as error:
            sys.exit("reading the parquet needs pyarrow:  pip install pyarrow   ({})".format(error))
        role = t["role"].astype(str).str.lower()
        team = t["team"].astype(str).str.upper().str.strip()
        keep = role.isin(["player", "goalkeeper"] if keep_gk else ["player"]) & team.isin(["H", "A"])
        keep &= np.isfinite(t["x_m"]) & np.isfinite(t["y_m"]) & (t["frame"] >= 0) & (t["frame"] < N_FRAMES)
        t = t[keep]
        self.frame = t["frame"].values.astype(np.int64)
        order = np.argsort(self.frame, kind="stable")
        self.frame = self.frame[order]
        self.tid = t["track_id"].values.astype(np.int64)[order]
        self.team = team[keep].map({"H": 0, "A": 1}).values.astype(np.int64)[order]
        self.x = (t["x_m"].values / TRACK_PITCH[0]).astype(np.float32)[order]
        self.y = (t["y_m"].values / TRACK_PITCH[1]).astype(np.float32)[order]
        print("Kushagra's tracks: {:,} player/goalkeeper rows with a team ({})".format(len(self.frame), path))

    def window(self, a, n):
        lo, hi = np.searchsorted(self.frame, [a, a + n])
        f, tid, tm = self.frame[lo:hi] - a, self.tid[lo:hi], self.team[lo:hi]
        key = tid * 2 + tm
        uniq, slot = np.unique(key, return_inverse=True)
        players = np.full((n, max(1, len(uniq)), 3), np.nan, dtype=np.float32)
        players[f, slot, 0] = self.x[lo:hi]
        players[f, slot, 1] = self.y[lo:hi]
        team = (uniq % 2).astype(np.int64) if len(uniq) else np.zeros(1, dtype=np.int64)
        return players, team


# -- overlay: one transition line on every written frame -------------------------------------------

class TransitionWriter:
    """Wraps cv2.VideoWriter: writes the transition line (the change nearest the current frame among
    the clip's changes) onto each canvas before writing it."""

    def __init__(self, inner, changes, total, names, first, every):
        self.inner, self.changes, self.total, self.names = inner, changes, total, names
        self.frames = np.array([t["frame"] for t in changes])
        self.f, self.every = first, every

    def isOpened(self):
        return self.inner.isOpened()

    def release(self):
        self.inner.release()

    def write(self, canvas):
        f = self.f
        j = int(np.argmin(np.abs(self.frames - f)))
        tr = self.changes[j]
        dt = (f - tr["frame"]) / float(FPS)
        on = abs(f - tr["frame"]) < max(1, self.every)
        col = (0, 255, 255) if on else (255, 255, 255) if tr.get("k") else (170, 170, 170)
        who = "{} -> {}".format(vl.team_word(tr["from"], self.names), vl.team_word(tr["to"], self.names))
        sel = "({} of {} selected)".format(tr["k"], self.total) if tr.get("k") else "(not selected)"
        text = "TRANSITION #{} {}  {}  {}  t = {:+.1f} s{}".format(
            tr["n"], sel, tr["direction"], who, dt, "  << CHANGE" if on else "")
        vt.put(canvas, text[:100], (vl.PANEL_W + 12, vl.PANEL_H + 80), 0.48, col, 2 if on else 1)
        vt.put(canvas, "change {} of {} in this clip, at {}".format(j + 1, len(self.changes), tr["video_clock"]),
               (vl.PANEL_W + 12, vl.PANEL_H + 98), 0.42, (200, 200, 200))
        self.inner.write(canvas)
        self.f += self.every


# -- selection ------------------------------------------------------------------------------------

def select(trans, args):
    keep = trans
    if args.type:
        want = {TYPE_CODES[c] for c in args.type}
        keep = [t for t in keep if t["type"] in want]
    if args.dir:
        want = {DIR_CODES[c] for c in args.dir}
        keep = [t for t in keep if t["direction"] in want]
    if args.n:
        want = set(args.n)
        keep = [t for t in keep if t["n"] in want]
    if args.from_n:
        keep = [t for t in keep if t["n"] >= args.from_n]
    if args.range:
        a, b = args.range
        keep = [t for t in keep if a <= t["frame"] <= b]
    if args.only_new:
        keep = [t for t in keep if not t["in_compare"]]
    if args.only_shared:
        keep = [t for t in keep if t["in_compare"]]
    if args.only_short:
        keep = [t for t in keep if t["short"]]
    if args.max > 0:
        keep = keep[:args.max]
    for k, t in enumerate(keep, 1):
        t["k"] = k
    return keep


def window_of(tr, args):
    """[a, b] for one change: --before / --after, widened evenly to --min-length, kept inside its half
    (shifted, not cut, when it would cross the half's start or end)."""
    lo, hi = [(blk[1], blk[2]) for blk in half_video_blocks(GAME) if blk[0] == tr["half"]][0]
    a = tr["frame"] - int(round(args.before * FPS))
    b = tr["frame"] + int(round(args.after * FPS)) - 1
    need = int(round(args.min_length * FPS)) - (b - a + 1)
    if need > 0:
        a -= need // 2
        b += need - need // 2
    if a < lo:
        b, a = min(hi, b + (lo - a)), lo
    if b > hi:
        a, b = max(lo, a - (b - hi)), hi
    return a, b


def group(sel, args):
    """[{first, last, half, changes}] clips: windows of the selected changes, merged when they overlap
    or are under --join-gap apart (same half only). --no-group: one clip per change."""
    gap = int(round(args.join_gap * FPS))
    cap = int(round(args.max_length * FPS)) if args.max_length > 0 else 10 ** 9
    clips = []
    for t in sel:                                            # sel is in match order
        a, b = window_of(t, args)
        c = clips[-1] if clips else None
        if (not args.no_group and c is not None and c["half"] == t["half"] and a <= c["last"] + 1 + gap
                and max(c["last"], b) - c["first"] + 1 <= cap):
            c["last"] = max(c["last"], b)
            c["sel"].append(t)
        else:
            clips.append({"first": a, "last": b, "half": t["half"], "sel": [t]})
    for i, c in enumerate(clips, 1):
        c["c"] = i
        for t in c["sel"]:
            t["clip"] = i
    return clips


def print_selection(sel, names, has_compare):
    print("\n  {:>4} {:>4} {:>8} {:>10} {:<5} {:<34} {:>7} {:>7} {:>8}".format(
        "k", "n", "frame", "clock", "dir", "teams", "before", "after", "compare"))
    for t in sel:
        print("  {:>4} {:>4} {:>8} {:>10} {:<5} {:<34} {:>6.1f}s {:>6.1f}s {:>8}".format(
            t["k"], t["n"], t["frame"], t["video_clock"], t["direction"],
            "{} -> {}".format(vl.team_word(t["from"], names), vl.team_word(t["to"], names))[:34],
            t["before_s"], t["after_s"],
            ("yes {:+.1f}s".format(t["compare_offset_s"]) if t["in_compare"] else "no") if has_compare else "-"))


# -- reel -----------------------------------------------------------------------------------------

def make_reel(clips, names, out_path):
    """Join the clip mp4s, a 1.5 s title card before each listing the clip's changes."""
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
        vt.put(card, "{} change(s) in this clip:".format(len(c["changes"])), (60, 150), 0.8, (255, 255, 255), 2)
        for j, tr in enumerate(c["changes"][:14]):
            vt.put(card, "#{:<4} {}  {:<5} {} -> {}   old ran {:.1f} s, new runs {:.1f} s{}".format(
                tr["n"], tr["video_clock"], tr["direction"], vl.team_word(tr["from"], names),
                vl.team_word(tr["to"], names), tr["before_s"], tr["after_s"],
                "" if tr.get("k") else "   (not selected)"),
                (80, 195 + 34 * j), 0.65, vl.LABEL_BGR[tr["to"]] if tr["to"] != "D" else (200, 200, 200), 2)
        if len(c["changes"]) > 14:
            vt.put(card, "... {} more".format(len(c["changes"]) - 14), (80, 195 + 34 * 14), 0.65)
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


# -- main -----------------------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--labels", default=lt.DEFAULT_LABELS, help="labeller CSV (default: labeller_v6_newtracks.csv)")
    ap.add_argument("--compare", default=lt.DEFAULT_COMPARE, help="for in_compare / --only-new (default SAFE); 'none'")
    ap.add_argument("--tolerance", type=float, default=1.0, help="seconds for in_compare (default 1.0)")
    s = ap.add_argument_group("which transitions")
    s.add_argument("--type", nargs="+", choices=sorted(TYPE_CODES), help="HA, DA, HD (both directions each)")
    s.add_argument("--dir", nargs="+", choices=sorted(DIR_CODES), help="H2A A2H D2A A2D H2D D2H")
    s.add_argument("--n", type=int, nargs="+", help="transition numbers (column n of the list)")
    s.add_argument("--from-n", type=int, default=0,
                   help="skip transitions numbered below this (e.g. 21 = drop #1-#20 already watched)")
    s.add_argument("--range", type=int, nargs=2, metavar=("FIRST", "LAST"), help="changes inside these frames")
    s.add_argument("--only-new", action="store_true", help="only changes the --compare file does not have")
    s.add_argument("--only-shared", action="store_true", help="only changes the --compare file also has")
    s.add_argument("--only-short", action="store_true", help="only changes whose new label lasts < 2 s")
    s.add_argument("--max", type=int, default=10, help="at most N clips (default 10, 0 = all)")
    s.add_argument("--list-only", action="store_true", help="print the selection, no video")
    c = ap.add_argument_group("clips")
    c.add_argument("--before", type=float, default=4.0, help="seconds before the change (default 4)")
    c.add_argument("--after", type=float, default=4.0, help="seconds after the change (default 4)")
    c.add_argument("--min-length", type=float, default=15.0, help="shortest clip in seconds (default 15)")
    c.add_argument("--join-gap", type=float, default=0.0,
                   help="also join clips less than this many seconds apart (default 0 = only overlapping)")
    c.add_argument("--max-length", type=float, default=60.0,
                   help="a group stops growing at this many seconds and the next change starts a new clip "
                        "(default 60, 0 = no limit); clips may then overlap a little")
    c.add_argument("--no-group", action="store_true", help="one clip per change (no grouping)")
    c.add_argument("--players", choices=("kushagra", "atd"), default="kushagra", help="players drawn (default kushagra)")
    c.add_argument("--tracks", default=DEFAULT_TRACKS, help="Kushagra's parquet")
    c.add_argument("--no-goalkeepers", action="store_true", help="draw outfield players only (Kushagra's tracks)")
    c.add_argument("--reel", action="store_true", help="also join all clips into one mp4")
    c.add_argument("--out-dir", default=os.path.join(vl.DEFAULT_OUT, "transitions"))
    # passed to visualize_labeller's renderer, same meaning there
    c.add_argument("--every", type=int, default=1)
    c.add_argument("--gt", action="store_true", help="public 5-minute GT band (view only)")
    c.add_argument("--gt-path", default=vt.DEFAULT_GT)
    c.add_argument("--banner", type=float, default=1.5)
    c.add_argument("--trail", type=int, default=50)
    c.add_argument("--ids", action="store_true")
    c.add_argument("--no-overlay", action="store_true")
    c.add_argument("--no-video", action="store_true")
    c.add_argument("--no-tags", action="store_true")
    c.add_argument("--no-flip-y", dest="flip_y", action="store_false")
    c.add_argument("--backend", choices=("auto", "ffmpeg", "opencv"), default="auto")
    c.add_argument("--exact-seek", action="store_true")
    c.add_argument("--video-offset", type=int, default=0)
    args = ap.parse_args()
    if args.only_new and args.only_shared:
        ap.error("--only-new and --only-shared exclude each other")
    return args


def main():
    args = parse_args()
    labels = vl.load_labels(args.labels)
    trans = lt.find_transitions(labels)
    has_compare = args.compare.lower() != "none" and os.path.isfile(args.compare)
    if has_compare:
        lt.mark_compare(trans, lt.find_transitions(vl.load_labels(args.compare)), int(round(args.tolerance * FPS)))
    else:
        for t in trans:
            t["in_compare"], t["compare_offset_s"] = 0, ""
        if args.only_new or args.only_shared:
            sys.exit("--only-new / --only-shared need the --compare file ({})".format(args.compare))
    listed = lt.default_out(args.labels)
    if os.path.isfile(listed):
        same = [int(r["frame"]) for r in lt.read(listed)] == [t["frame"] for t in trans]
        print("list file {}: {}".format(listed, "matches the labels" if same else
                                        "OUT OF DATE (rerun list_transitions.py); using the labels"))
    meta_path, raw_path = require(["atd_meta", "atd_raw"])
    names = team_names(meta_path)
    print("labels {}: {} transitions in the graded match;  H = {}, A = {}".format(
        args.labels, len(trans), *names))

    sel = select(trans, args)
    clips = group(sel, args) if sel else []
    for c in clips:                      # every change inside the clip, selected or not, in order
        c["changes"] = [t for t in trans if c["first"] <= t["frame"] <= c["last"]]
    print("selected {} transition(s) -> {} clip(s) (min {:.0f} s each{})".format(
        len(sel), len(clips), args.min_length, ", no grouping" if args.no_group else ", grouped"))
    print_selection(sel, names, has_compare)
    print("\n  {:>4} {:>8} {:>8} {:>8} {:>9}  {}".format("clip", "first", "last", "seconds", "selected", "changes in clip (n)"))
    for c in clips:
        print("  {:>4} {:>8} {:>8} {:>8.1f} {:>9}  {}".format(
            c["c"], c["first"], c["last"], (c["last"] - c["first"] + 1) / float(FPS), len(c["sel"]),
            " ".join(str(t["n"]) + ("" if t.get("k") else "*") for t in c["changes"])))
    if clips:
        print("  (* = inside the clip but not selected)")
    if not sel or args.list_only:
        return

    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.labels))[0]
    for c in clips:
        c["mp4"] = os.path.join(args.out_dir, "{}_C{}_{}_{}_{}chg.mp4".format(
            stem, c["c"], c["first"], c["last"], len(c["changes"])))
    with open(os.path.join(args.out_dir, "selection.csv"), "w", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=["k"] + lt.FIELDS + ["clip", "mp4"])
        w.writeheader()
        for t in sel:
            w.writerow(dict({k: t.get(k, "") for k in ["k"] + lt.FIELDS}, clip=t["clip"], mp4=clips[t["clip"] - 1]["mp4"]))
    with open(os.path.join(args.out_dir, "clips.csv"), "w", newline="") as handle:
        w = csv.writer(handle)
        w.writerow(["clip", "first_frame", "last_frame", "start_clock", "seconds", "half", "n_selected",
                    "n_changes", "changes", "mp4"])
        for c in clips:
            w.writerow([c["c"], c["first"], c["last"], vt.fmt_time(c["first"]),
                        "{:.2f}".format((c["last"] - c["first"] + 1) / float(FPS)), c["half"], len(c["sel"]),
                        len(c["changes"]), " ".join(str(t["n"]) for t in c["changes"]), c["mp4"]])

    atd_team = team_of_each_track(meta_path)
    new = NewTracks(args.tracks, keep_gk=not args.no_goalkeepers) if args.players == "kushagra" else None
    spells = vl.MatchSpells(labels)
    real_writer = cv2.VideoWriter
    done, all_ok = [], True
    try:
        for c in clips:
            a, b, mp4 = c["first"], c["last"], c["mp4"]
            n = b - a + 1
            print("\n=== clip {} of {}: frames {}..{} ({:.1f} s), {} change(s): {} ===".format(
                c["c"], len(clips), a, b, n / float(FPS), len(c["changes"]),
                ", ".join("#{} {}".format(t["n"], t["direction"]) for t in c["changes"])))
            if new is not None:
                players, team = new.window(a, n)
                _, ball = _orig_read_tracking(raw_path, a, n, 1)
                vt.read_tracking = lambda *_a, _p=players, _b=ball: (_p, _b)
                print("  players: Kushagra's tracks, {} tracks in the clip (home {}, away {})".format(
                    len(team), int((team == 0).sum()), int((team == 1).sum())))
            else:
                team = atd_team
                vt.read_tracking = _orig_read_tracking
            cv2.VideoWriter = (lambda *wa, _c=c["changes"], _a=a: TransitionWriter(
                real_writer(*wa), _c, len(sel), names, _a, max(1, args.every)))
            base = os.path.splitext(mp4)[0]
            vl.write_spells(base + "_spells.csv", labels, a, b, names)
            drawn = vl.render(args, labels, spells, a, b, names, team, raw_path, mp4)
            cv2.VideoWriter = real_writer
            all_ok &= vl.check_drawn(drawn, labels, base + "_drawn.csv")
            print("wrote {}".format(os.path.abspath(mp4)))
            done.append(c)
    finally:
        cv2.VideoWriter = real_writer
        vt.read_tracking = _orig_read_tracking

    if args.reel and done and not args.no_video:
        reel = os.path.join(args.out_dir, "{}_reel_{}clips_{}chg.mp4".format(
            stem, len(done), sum(len(c["sel"]) for c in done)))
        make_reel(done, names, reel)
        print("wrote reel {}".format(os.path.abspath(reel)))
    print("\n{} clip(s) for {} selected change(s) in {}   clips.csv / selection.csv list them".format(
        len(done), len(sel), os.path.abspath(args.out_dir)))
    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
