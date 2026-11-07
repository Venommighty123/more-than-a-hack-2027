"""Visualise what the public ground truth means by "possession" -- without touching viz/render_match.py.

Three outputs, all about the 5 labelled minutes (video frames 11185..18684):

  1. A table of every ground-truth segment (H / A / D): start, end, duration, how much of it the
     ATD ball is tracked, and how often the nearest-player rule agrees with the label.
  2. A timeline PNG: GT bands, the nearest-player baseline (raw and smoothed), ball tracked or
     missing, and the Smart Tagging phases coloured by team -- all on one time axis, with the GT
     segment boundaries drawn through every row.
  3. (opt-in) Video clips rendered by viz/render_match.py, which is called as a subprocess:
       --clips   one short clip around the start of every GT segment (the hand-overs)
       --full    the whole labelled window as one video

    python viz/possession_definition.py                       # table + timeline PNG
    python viz/possession_definition.py --clips --before 75 --after 75
    python viz/possession_definition.py --clips --limit 3 --no-video
    python viz/possession_definition.py --full --out-dir outputs/possession_definition

Anything this script does not recognise is forwarded to viz/render_match.py, e.g. --ids,
--no-video, --backend opencv.


The table and PNG need numpy, pandas and matplotlib. The nearest-player rows additionally need the
match data (ATD); without it they are skipped and the GT-only view is still produced. The ATD pass
(baseline.build) takes a minute or two, so its result is cached as an .npy in --out-dir. That cache
is derived tracking data: do not commit or publish it (RULES.md).

This is for UNDERSTANDING the definition. Do not tune or train on the public window, and do not turn
what you see into constants tied to specific frames of this match (RULES.md).
"""

import argparse
import os
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import possession.io  # noqa: E402,F401  (repository root on sys.path)
from games import GAME, GAMES, source_frames  # noqa: E402

# Params
FPS = GAMES[GAME]["fps"]
N_FRAMES = source_frames("tactical", GAME)
GT_PATH = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
DEFAULT_OUT = os.path.join(ROOT, "outputs", "possession_definition")
SMOOTH_WINDOW = 50      # the published floor's window; shown only as a "what a smoothed rule says" row
COLORS = {
    "H": "#d62728", "A": "#1f77b4", "D": "#9aa5b1",
    "?": "#e5e7eb",     # nearest-player rule had no ball, so no answer
    "T": "#4b5563",     # ball tracked
    "M": "#fca5a5",     # ball missing
    "NA": "#c4b5a0",    # Smart Tagging Team = N/A
}


def find_viz():
    """viz/render_match.py (the match renderer). Never re-implemented here."""
    path = os.path.join(HERE, "render_match.py")
    if not os.path.isfile(path):
        sys.exit("viz/render_match.py not found")
    return path


def clock(frame):
    seconds = frame / float(FPS)
    return "{:02d}:{:04.1f}".format(int(seconds // 60), seconds % 60)


def runs(values):
    """[(start_index, stop_index, value)] maximal constant runs of a 1-D array (stop exclusive)."""
    if len(values) == 0:
        return []
    breaks = np.flatnonzero(values[1:] != values[:-1]) + 1
    bounds = np.concatenate(([0], breaks, [len(values)]))
    return [(int(bounds[i]), int(bounds[i + 1]), values[bounds[i]]) for i in range(len(bounds) - 1)]


# -- ground truth ------------------------------------------------------------------------------

def load_gt(path):
    """(first_frame, labels) for the labelled window. labels[i] is the label of video frame first+i."""
    table = pd.read_csv(path, dtype={"label": "object"})
    table.columns = [str(c).strip().lower() for c in table.columns]
    labels = table["label"].astype(str).str.strip().str.upper().values
    graded = np.flatnonzero(labels != "X")
    if not len(graded):
        sys.exit("{} has no graded frames".format(path))
    first, last = int(graded[0]), int(graded[-1])
    window = labels[first:last + 1]
    if (window == "X").any():
        print("note: {} X frames inside the labelled window".format(int((window == "X").sum())))
    return first, window


# -- nearest-player baseline (reuses starter/baseline.py) -------------------------------------

def load_codes(out_dir, refresh):
    """Per-frame nearest-player team code for the whole video (0 home, 1 away, -1 none), or None.

    Cached, because streaming the 199k-line ATD costs a minute or two. Returns None when the match
    data is not installed, so the GT-only outputs still work.
    """
    cache = os.path.join(out_dir, "atd_nearest_player_codes.npy")
    if os.path.isfile(cache) and not refresh:
        codes = np.load(cache)
        if len(codes) == N_FRAMES:
            print("nearest-player codes: loaded from {}".format(cache))
            return codes
    try:
        from starter import baseline
        codes = baseline.build()
    except (IOError, ImportError) as error:
        print("note: nearest-player rows skipped ({})".format(str(error).splitlines()[0]))
        return None
    np.save(cache, codes)
    return codes


def baseline_rows(codes, first, n):
    """(raw, smooth, ball_ok) as single-letter arrays over the labelled window."""
    from starter import baseline
    window = codes[first:first + n]
    raw = np.where(window == baseline.HOME, "H", np.where(window == baseline.AWAY, "A", "?"))
    smooth_full = baseline.smooth(codes, SMOOTH_WINDOW)[first:first + n]
    # baseline.smooth says D where nobody had the ball in the window; that is "no answer", not dead.
    smooth = np.where(smooth_full == "D", "?", smooth_full)
    ball_ok = np.where(window != baseline.NOBODY, "T", "M")
    return raw, smooth, ball_ok


# -- segment table -----------------------------------------------------------------------------

def segment_table(first, gt, raw, ball_ok):
    rows = []
    for k, (a, b, label) in enumerate(runs(gt), start=1):
        row = {"seg": k, "label": label, "start_frame": first + a, "end_frame": first + b - 1,
               "frames": b - a, "seconds": (b - a) / float(FPS), "video_clock": clock(first + a)}
        if raw is not None:
            tracked = ball_ok[a:b] == "T"
            row["ball_tracked_pct"] = 100.0 * tracked.mean()
            known = raw[a:b] != "?"
            if label in ("H", "A") and known.any():
                row["nearest_agree_pct"] = 100.0 * (raw[a:b][known] == label).mean()
            else:
                row["nearest_agree_pct"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def print_summary(table, first, gt, raw):
    alive = np.isin(gt, ("H", "A"))
    print("\nlabelled window: video frames {}..{}  ({} frames, {:.1f} s)".format(
        first, first + len(gt) - 1, len(gt), len(gt) / float(FPS)))
    spells = table[table["label"].isin(["H", "A"])]
    print("possessions (maximal H or A runs): {}   mean {:.1f} s   D stretches: {}".format(
        len(spells), spells["seconds"].mean() if len(spells) else 0.0,
        int((table["label"] == "D").sum())))
    if raw is not None:
        both = alive & (raw != "?")
        print("nearest-player rule vs GT on alive frames with a tracked ball: {:.1f}% agree "
              "({} frames)".format(100.0 * (raw[both] == gt[both]).mean(), int(both.sum())))
        print("  alive frames with no ball tracked: {:.1f}%".format(
            100.0 * (alive & (raw == "?")).sum() / max(1, alive.sum())))
    print()
    pd.set_option("display.width", 140)
    print(table.to_string(index=False, float_format=lambda v: "{:.1f}".format(v)))


# -- timeline ----------------------------------------------------------------------------------

def team_color(tags):
    team = (tags.get("Team") or "").strip().lower()
    if "barcelona" in team:
        return COLORS["H"]
    if "atletic" in team or "athletic" in team:
        return COLORS["A"]
    return COLORS["NA"]


def load_events(first, n):
    """Smart Tagging events overlapping the window, or []."""
    try:
        from starter import load_smart_tagging
        events = load_smart_tagging.load()
    except (IOError, ImportError) as error:
        print("note: Smart Tagging skipped ({})".format(str(error).splitlines()[0]))
        return []
    return [e for e in events if e["end_frame"] > first and e["start_frame"] < first + n]


def plot_timeline(path, first, gt, raw, smooth, ball_ok, events, table):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    n = len(gt)
    seconds = lambda i: i / float(FPS)  # noqa: E731

    rows = [("GT (public)", gt)]
    if raw is not None:
        rows += [("nearest player, raw", raw),
                 ("nearest player, +/-{}".format(SMOOTH_WINDOW), smooth),
                 ("ball tracked", ball_ok)]
    lanes = sorted({e["code"] for e in events})
    ratios = [max(1.2, 0.6 * len(rows)), max(1.0, 0.35 * len(lanes))] if lanes else [1]
    fig, axes = plt.subplots(len(ratios), 1, sharex=True, figsize=(16, 2.2 + 0.55 * len(rows) + 0.3 * len(lanes)),
                             gridspec_kw={"height_ratios": ratios})
    axes = np.atleast_1d(axes)
    top = axes[0]

    for r, (name, values) in enumerate(rows):
        y = len(rows) - 1 - r
        for a, b, value in runs(values):
            top.broken_barh([(seconds(a), seconds(b - a))], (y + 0.08, 0.84),
                            facecolors=COLORS.get(value, "#000000"), linewidth=0)
    top.set_yticks([len(rows) - 1 - r + 0.5 for r in range(len(rows))])
    top.set_yticklabels([name for name, _ in rows])
    top.set_ylim(0, len(rows))

    # Segment numbers on the GT row, matching the printed table.
    for _, seg in table.iterrows():
        a = seg["start_frame"] - first
        mid = seconds(a + seg["frames"] / 2.0)
        if seg["seconds"] >= 2.0:
            top.text(mid, len(rows) - 0.5, "#{}".format(int(seg["seg"])), ha="center", va="center",
                     color="white", fontsize=8, fontweight="bold")

    if lanes:
        tags = axes[1]
        for j, code in enumerate(lanes):
            for e in events:
                if e["code"] != code:
                    continue
                a = max(e["start_frame"], first) - first
                b = min(e["end_frame"], first + n) - first
                if b > a:
                    tags.broken_barh([(seconds(a), seconds(b - a))], (j + 0.1, 0.8),
                                     facecolors=team_color(e["tags"]), edgecolors="white", linewidth=0.4)
        tags.set_yticks([j + 0.5 for j in range(len(lanes))])
        tags.set_yticklabels([c.title() for c in lanes], fontsize=8)
        tags.set_ylim(0, len(lanes))
        tags.invert_yaxis()

    for ax in axes:
        for a in table["start_frame"].values[1:]:
            ax.axvline(seconds(a - first), color="black", linewidth=0.5, linestyle=(0, (3, 3)), alpha=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_xlim(0, seconds(n))
    axes[-1].set_xlabel("seconds from video frame {} (first graded frame)".format(first))

    legend = [Patch(color=COLORS["H"], label="H / Barcelona"), Patch(color=COLORS["A"], label="A / Athletic"),
              Patch(color=COLORS["D"], label="D dead ball")]
    if raw is not None:
        legend += [Patch(color=COLORS["?"], label="no answer (ball not tracked)"),
                   Patch(color=COLORS["M"], label="ball missing")]
    if lanes:
        legend.append(Patch(color=COLORS["NA"], label="tag Team = N/A"))
    top.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, 1.35), ncol=len(legend),
               frameon=False, fontsize=8)
    top.set_title("Public ground truth vs nearest-player rule, ball tracking and Smart Tagging "
                  "(dashed lines = GT segment starts)", loc="left", fontsize=11, pad=34)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# -- clips: delegated to viz/render_match.py --------------------------------------------------

def run_viz(viz, first_frame, duration_s, out, passthrough, dry_run):
    cmd = [sys.executable, viz, "--start-frame", str(max(0, first_frame)),
           "--duration", "{:.4f}".format(duration_s), "--out", out] + passthrough
    if dry_run:
        print("   " + " ".join(cmd))
        return True
    return subprocess.call(cmd) == 0 and os.path.isfile(out)


def render_clips(viz, table, args, passthrough, clip_dir):
    os.makedirs(clip_dir, exist_ok=True)
    segments = table if not args.limit else table.head(args.limit)
    failed = []
    for _, seg in segments.iterrows():
        start = int(seg["start_frame"])
        out = os.path.join(clip_dir, "seg{:02d}_{}_f{}.mp4".format(int(seg["seg"]), seg["label"], start))
        if os.path.isfile(out):
            print("skip (exists): {}".format(out))
            continue
        first = start - args.before
        n = args.before + args.after
        print("[seg {}/{}] {} starts at frame {}  ->  frames {}..{}  {}".format(
            int(seg["seg"]), len(table), seg["label"], start, max(0, first), max(0, first) + n - 1, out))
        if not run_viz(viz, first, n / float(FPS), out, passthrough, args.dry_run):
            failed.append(int(seg["seg"]))
            print("   FAILED")
    if failed:
        print("{} clip(s) failed: segments {}".format(len(failed), failed))


# -- main --------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gt", default=GT_PATH, help="public ground-truth CSV")
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--refresh", action="store_true", help="recompute the cached nearest-player codes")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--clips", action="store_true", help="render a clip around every GT segment start")
    ap.add_argument("--before", type=int, default=75, help="clip frames before the segment start")
    ap.add_argument("--after", type=int, default=75, help="clip frames after the segment start")
    ap.add_argument("--limit", type=int, help="only the first N segments' clips")
    ap.add_argument("--full", action="store_true", help="render the whole labelled window as one video")
    ap.add_argument("--dry-run", action="store_true", help="print the visualize_tracking commands only")
    args, passthrough = ap.parse_known_args()
    os.makedirs(args.out_dir, exist_ok=True)

    first, gt = load_gt(args.gt)
    n = len(gt)

    codes = load_codes(args.out_dir, args.refresh)
    raw = smooth = ball_ok = None
    if codes is not None:
        raw, smooth, ball_ok = baseline_rows(codes, first, n)

    table = segment_table(first, gt, raw, ball_ok)
    print_summary(table, first, gt, raw)
    csv_path = os.path.join(args.out_dir, "gt_segments.csv")
    table.to_csv(csv_path, index=False)
    print("\nwrote {}".format(csv_path))

    if not args.no_plot:
        png_path = os.path.join(args.out_dir, "possession_timeline.png")
        plot_timeline(png_path, first, gt, raw, smooth, ball_ok, load_events(first, n), table)
        print("wrote {}".format(png_path))

    if args.clips or args.full:
        viz = find_viz()
        if args.full:
            out = os.path.join(args.out_dir, "gt_window_{}.mp4".format(first))
            print("\nrendering the whole window with visualize_tracking.py -> {}".format(out))
            run_viz(viz, first, n / float(FPS), out, passthrough, args.dry_run)
        if args.clips:
            print()
            render_clips(viz, table, args, passthrough, os.path.join(args.out_dir, "clips"))


if __name__ == "__main__":
    main()