"""Audit the tactical data for missing pieces -- above all, how long the ball goes missing.

The question this answers: is "hold the last / interpolate to the next detection" enough, or does
the ball need its own detector? That depends on the *length* of the gaps, not on the 43% headline,
so the main output is a histogram of consecutive ball-missing frames (a "gap"), counted two ways:

  by gaps    how many gaps of each length            (most gaps are short)
  by frames  how much of the missing time they hold  (a few long gaps usually hold most of it)

Both matter, and they usually disagree. For each gap length it also reports how far the ball moved
between the last frame before the gap and the first frame after it, the speed that implies, and
whether the nearest-player team (the baseline's rule) differs across the gap. Those three numbers
say what "previous and next detection" would cost you.

Gaps are measured INSIDE the two playing halves only (games.half_video_blocks, the 158,857 graded
frames). Pre-match, halftime and post-match have no ball and would otherwise show up as a few
enormous gaps that swamp the histogram. A gap never crosses a half boundary.

Everything else that can be missing is audited too: ATD alignment and malformed records, tracked
players per frame, slots that never fire, the homography, and Smart Tagging coverage.

Reuses, rather than re-implements: games (paths, half blocks), load_atd.check_alignment_raw,
baseline.team_of_each_track / nearest_player_labels, load_homography.load / check_alignment and
load_smart_tagging.load / check_time_base. Put this file in starter/ or the repository root.

    pip install numpy pandas matplotlib          # matplotlib only for the PNG
    python ball_gap_analysis.py                  # ~1-3 min: it streams the raw ATD twice
    python ball_gap_analysis.py --skip-team-flip # ~1 min: skip the nearest-player pass
    python ball_gap_analysis.py --out my_reports --top 20

Writes to reports/data_audit/ (or --out):
    ball_gap_histogram.png   the histogram (exact lengths, log bins, cumulative share of frames)
    ball_gap_histogram.csv   exact histogram: gap length in frames -> number of gaps
    ball_gaps.csv            every gap: half, start frame, length, bracket distance/speed, flip
    data_audit.json          every number printed here, machine-readable
"""

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
ROOT = PARENT if os.path.isfile(os.path.join(PARENT, "games.py")) else HERE
for _p in (ROOT, os.path.join(ROOT, "starter"), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from games import GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402
from starter import baseline, load_atd

# Params
FPS = GAMES[GAME]["fps"]
PITCH_L, PITCH_W = GAMES[GAME]["pitch"]
FIELD_SEP, GROUP_SEP, MISSING = ":", ";", "N"      # same conventions as baseline.py
DEFAULT_OUT = os.path.join(ROOT, "reports", "data_audit")
# A kicked football rarely exceeds ~35-40 m/s. Faster implied speed across a gap means the two
# bracketing detections are not the same ball path (a mis-detection, or the ball left and re-entered).
MAX_PLAUSIBLE_SPEED = 40.0                         # m/s
PENALTY_AREA_DEPTH = 16.5                          # m
# Gap-length bins, lower bound in frames. Edges sit on 0.5 s / 1 s / 2 s / 5 s / 10 s / 20 s.
BIN_LOWER = (1, 2, 3, 6, 13, 26, 51, 126, 251, 501)
BIN_LABEL = ("1", "2", "3-5", "6-12", "13-25", "26-50", "51-125", "126-250", "251-500", ">500")
BIN_SECONDS = ("0.04 s", "0.08 s", "<0.24 s", "<0.5 s", "<1 s", "<2 s", "<5 s", "<10 s", "<20 s", ">20 s")
CUMULATIVE_AT = (1, 5, 12, 25, 50, 125, 250, 500)  # frames

INK, INK_MUTED, GRID, SERIES, ACCENT = "#1f2933", "#52606d", "#d9dee3", "#2a6fdb", "#c2410c"


def clock(frame):
    s = frame / float(FPS)
    return "{:02d}:{:04.1f}".format(int(s // 60), s % 60)


def heading(title):
    print("\n" + "=" * 78 + "\n" + title + "\n" + "=" * 78)


def pct(part, whole):
    return 100.0 * part / whole if whole else 0.0


# -- pass 1: one streaming read of the raw ATD -------------------------------------------------

def scan_atd(raw_path, n_frames, n_slots):
    """Per video frame: which track slots have a position, and the ball position.

    Also counts every malformed thing it meets. Raw line: "<frame_id>:<x,y,s>;...:<ball>", with
    frame_id = video frame + 1, and NaN for a missing observation.
    """
    present = np.zeros((n_frames, n_slots), dtype=bool)
    ball = np.full((n_frames, 2), np.nan, dtype=np.float32)
    issues = {"bad_lines": 0, "id_not_sequential": 0, "id_out_of_range": 0,
              "wrong_slot_count": 0, "player_position_partial": 0, "player_speed_only_nan": 0,
              "ball_partial": 0, "ball_outside_pitch": 0}
    expected_id = 1

    with open(raw_path) as handle:
        for line in handle:
            parts = line.split(FIELD_SEP)
            if len(parts) != 3:
                issues["bad_lines"] += 1
                continue
            frame_id = int(parts[0])
            if frame_id != expected_id:
                issues["id_not_sequential"] += 1
            expected_id = frame_id + 1
            i = frame_id - 1
            if not 0 <= i < n_frames:
                issues["id_out_of_range"] += 1
                continue

            groups = parts[1].split(GROUP_SEP)
            if len(groups) != n_slots:
                issues["wrong_slot_count"] += 1
            for slot, group in enumerate(groups[:n_slots]):
                if group.count(MISSING) == 0:
                    present[i, slot] = True
                    continue
                fields = group.split(",")
                gone = [MISSING in f for f in fields[:2]]
                if all(gone):
                    continue                               # a clean missing slot
                if any(gone):
                    issues["player_position_partial"] += 1  # x or y missing, not both
                else:
                    present[i, slot] = True                 # position fine, speed NaN
                    issues["player_speed_only_nan"] += 1

            fields = parts[2].strip().split(",")
            if len(fields) < 2:
                issues["bad_lines"] += 1
                continue
            try:
                bx, by = float(fields[0]), float(fields[1])
            except ValueError:
                issues["bad_lines"] += 1
                continue
            nx, ny = np.isnan(bx), np.isnan(by)
            if nx != ny:
                issues["ball_partial"] += 1                 # half a position is no position
            elif not nx:
                ball[i] = (bx, by)
                if not (0.0 <= bx <= 1.0 and 0.0 <= by <= 1.0):
                    issues["ball_outside_pitch"] += 1
    return present, ball, issues


# -- gaps --------------------------------------------------------------------------------------

def find_gaps(ball_ok, blocks):
    """Maximal runs of missing ball inside each half. Returns dict of equal-length arrays.

    has_before / has_after say whether a detection exists on that side *within the half*. A gap
    touching the whiste-to-whistle edge is censored: its true length is unknown, and it has no
    bracketing detection on one side, so it is left out of the bracket statistics.
    """
    half, start, length, before, after = [], [], [], [], []
    for h, first, last in blocks:
        window = ~ball_ok[first:last + 1]
        edges = np.diff(np.concatenate(([0], window.astype(np.int8), [0])))
        starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
        for s, e in zip(starts.tolist(), stops.tolist()):
            half.append(h)
            start.append(first + s)
            length.append(e - s)
            before.append(s > 0)
            after.append(e < len(window))
    return {"half": np.array(half, dtype=np.int8), "start": np.array(start, dtype=np.int64),
            "length": np.array(length, dtype=np.int64),
            "has_before": np.array(before, dtype=bool), "has_after": np.array(after, dtype=bool)}


def add_bracket_stats(gaps, ball, codes):
    """Distance, implied speed and team flip between the detections either side of each gap."""
    n = len(gaps["length"])
    dist = np.full(n, np.nan)
    speed = np.full(n, np.nan)
    flip = np.full(n, -1, dtype=np.int8)                    # -1 unknown, 0 same team, 1 differs
    both = np.flatnonzero(gaps["has_before"] & gaps["has_after"])
    b = gaps["start"][both] - 1
    a = gaps["start"][both] + gaps["length"][both]
    dist[both] = np.hypot((ball[a, 0] - ball[b, 0]) * PITCH_L, (ball[a, 1] - ball[b, 1]) * PITCH_W)
    speed[both] = dist[both] / ((gaps["length"][both] + 1) / float(FPS))
    if codes is not None:
        known = (codes[b] != baseline.NOBODY) & (codes[a] != baseline.NOBODY)
        flip[both[known]] = (codes[b[known]] != codes[a[known]]).astype(np.int8)
    gaps["dist_m"], gaps["speed_ms"], gaps["team_flip"] = dist, speed, flip
    return gaps


def histogram_rows(gaps):
    """One row per length bin: counts by gap and by frame, plus the bracket statistics."""
    length = gaps["length"]
    total_gaps, total_frames = len(length), int(length.sum())
    bin_of = np.searchsorted(BIN_LOWER, length, side="right") - 1
    rows, running = [], 0
    for k, lower in enumerate(BIN_LOWER):
        pick = bin_of == k
        n, frames = int(pick.sum()), int(length[pick].sum())
        running += frames
        d, s, f = gaps["dist_m"][pick], gaps["speed_ms"][pick], gaps["team_flip"][pick]
        d, s, f = d[~np.isnan(d)], s[~np.isnan(s)], f[f >= 0]
        rows.append({
            "bin": BIN_LABEL[k], "approx_time": BIN_SECONDS[k], "lower_frames": lower,
            "n_gaps": n, "pct_of_gaps": pct(n, total_gaps),
            "frames": frames, "pct_of_missing_frames": pct(frames, total_frames),
            "cum_pct_of_missing_frames": pct(running, total_frames),
            "median_bracket_m": float(np.median(d)) if len(d) else None,
            "p90_bracket_m": float(np.percentile(d, 90)) if len(d) else None,
            "pct_implausible_speed": pct(int((s > MAX_PLAUSIBLE_SPEED).sum()), len(s)) if len(s) else None,
            "pct_team_flip": pct(int(f.sum()), len(f)) if len(f) else None,
            "n_bracketed": int(len(d)),
        })
    return rows


# -- plot --------------------------------------------------------------------------------------

def plot_histogram(path, gaps, in_play_frames):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("note: matplotlib not installed, skipping the PNG (pip install matplotlib)")
        return False

    length = gaps["length"]
    total = float(length.sum())
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    fig.patch.set_facecolor("white")

    def style(ax, title, xlabel, ylabel):
        ax.set_title(title, loc="left", fontsize=11, color=INK)
        ax.set_xlabel(xlabel, color=INK_MUTED)
        ax.set_ylabel(ylabel, color=INK_MUTED)
        ax.tick_params(colors=INK_MUTED, labelsize=9)
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)

    # 1. exact lengths, 1..50 frames: where most gaps live.
    cap = 50
    counts = np.bincount(np.minimum(length, cap + 1), minlength=cap + 2)[1:cap + 1]
    axes[0].bar(np.arange(1, cap + 1), counts, width=0.8, color=SERIES)
    style(axes[0], "Gap length, exact (1-{} frames = up to {} s)".format(cap, cap // FPS),
          "consecutive frames without a ball", "number of gaps")
    axes[0].set_xlim(0, cap + 1)
    for frames in (12.5, 25):
        axes[0].axvline(frames, color=INK_MUTED, linewidth=0.8, linestyle=(0, (4, 3)))
    axes[0].text(12.7, axes[0].get_ylim()[1] * 0.94, "0.5 s", color=INK_MUTED, fontsize=8)
    axes[0].text(25.3, axes[0].get_ylim()[1] * 0.94, "1 s", color=INK_MUTED, fontsize=8)

    # 2. the full range, log-spaced bins, log y: the long tail.
    edges = np.unique(np.round(np.logspace(0, np.log10(length.max() + 1), 36)).astype(int))
    axes[1].hist(length, bins=edges, color=SERIES, rwidth=0.9)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    style(axes[1], "Gap length, full range (log bins)", "consecutive frames without a ball (log)",
          "number of gaps (log)")
    for frames, label in ((25, "1 s"), (125, "5 s"), (500, "20 s")):
        axes[1].axvline(frames, color=INK_MUTED, linewidth=0.8, linestyle=(0, (4, 3)))
        axes[1].text(frames * 1.05, axes[1].get_ylim()[1] * 0.55, label, color=INK_MUTED, fontsize=8)

    # 3. cumulative share of MISSING FRAMES by gap length: the number that decides it.
    order = np.sort(length)
    cumulative = 100.0 * np.cumsum(order) / total
    axes[2].step(order, cumulative, where="post", color=SERIES, linewidth=2)
    axes[2].set_xscale("log")
    axes[2].set_ylim(0, 102)
    style(axes[2], "Share of missing frames held by gaps up to this long",
          "gap length in frames (log)", "% of all missing in-play frames")
    for frames in (25, 125):
        share = 100.0 * order[order <= frames].sum() / total
        axes[2].plot([frames], [share], "o", color=ACCENT, markersize=6,
                     markeredgecolor="white", markeredgewidth=1.5)
        axes[2].annotate("{:.0f}% in gaps <= {} frames ({} s)".format(share, frames, frames // FPS),
                         (frames, share), textcoords="offset points", xytext=(8, -14),
                         color=INK, fontsize=8.5)

    fig.suptitle("Ball-missing gaps inside the two halves: {:,} gaps, {:,} of {:,} frames "
                 "({:.1f}%) without a ball".format(len(length), int(total), in_play_frames,
                                                   pct(total, in_play_frames)),
                 x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return True


# -- main --------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=DEFAULT_OUT, help="output directory (default reports/data_audit)")
    ap.add_argument("--top", type=int, default=10, help="how many of the longest gaps to list")
    ap.add_argument("--skip-team-flip", action="store_true",
                    help="skip the nearest-player pass (saves about half the run time)")
    ap.add_argument("--no-plot", action="store_true", help="do not write the PNG")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    n_frames = source_frames("tactical", GAME)
    meta_path, raw_path = require(["atd_meta", "atd_raw"])
    blocks = half_video_blocks(GAME)
    in_play = np.zeros(n_frames, dtype=bool)
    for _, first, last in blocks:
        in_play[first:last + 1] = True
    n_play = int(in_play.sum())
    team = baseline.team_of_each_track(meta_path)
    n_slots = len(team)
    report = {"n_frames": n_frames, "n_in_play_frames": n_play,
              "half_blocks": [list(b) for b in blocks]}

    # -- 1. alignment ---------------------------------------------------------------------------
    heading("1. ATD records and alignment")
    align = load_atd.check_alignment_raw()
    report["alignment"] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in align.items()}
    print("  records {} (video frames {}), frame_id {}..{}, contiguous {}, agrees {}".format(
        align["n_records"], align["n_video_frames"], align["frame_id_range"][0],
        align["frame_id_range"][1], align["contiguous"], align["agrees"]))
    print("  in-play blocks (video frames): {}".format(
        ", ".join("half {} {}..{}".format(*b) for b in blocks)))

    t0 = time.time()
    print("\nreading the raw ATD ...")
    present, ball, issues = scan_atd(raw_path, n_frames, n_slots)
    print("  done in {:.0f}s".format(time.time() - t0))
    ball_ok = ~np.isnan(ball[:, 0])

    # -- 2. the ball ----------------------------------------------------------------------------
    heading("2. Ball presence")
    rows = [("whole file", np.ones(n_frames, dtype=bool))] + [
        ("half {} in play".format(h), np.arange(n_frames).__ge__(a) & np.arange(n_frames).__le__(b))
        for h, a, b in blocks] + [("in play (both halves)", in_play), ("outside play", ~in_play)]
    report["ball_presence"] = {}
    for name, mask in rows:
        seen, total = int(ball_ok[mask].sum()), int(mask.sum())
        report["ball_presence"][name] = {"present": seen, "frames": total, "pct": pct(seen, total)}
        print("  {:<24} {:>8,} / {:>8,}  {:5.1f}% present".format(name, seen, total, pct(seen, total)))
    print("  (DATA.md quotes 56.9% over all frames and 59.3% over graded ones; "
          "these should match.)")

    # -- 3. the histogram -----------------------------------------------------------------------
    codes = None
    if not args.skip_team_flip:
        t0 = time.time()
        print("\nnearest-player pass (baseline.nearest_player_labels) ...")
        codes = baseline.nearest_player_labels(raw_path, team, n_frames, GAMES[GAME]["pitch"])
        print("  done in {:.0f}s".format(time.time() - t0))

    gaps = add_bracket_stats(find_gaps(ball_ok, blocks), ball, codes)
    length = gaps["length"]
    if not len(length):
        print("no ball-missing gaps inside the halves")
        return
    n_gaps, missing_total = len(length), int(length.sum())
    censored = int((~(gaps["has_before"] & gaps["has_after"])).sum())

    heading("3. Histogram of consecutive ball-missing frames (inside the halves only)")
    print("  {:,} gaps holding {:,} missing frames ({:.1f}% of {:,} in-play frames)".format(
        n_gaps, missing_total, pct(missing_total, n_play), n_play))
    print("  length: median {:.0f}  mean {:.1f}  p90 {:.0f}  p99 {:.0f}  max {} frames ({:.1f} s)".format(
        np.median(length), length.mean(), np.percentile(length, 90), np.percentile(length, 99),
        length.max(), length.max() / float(FPS)))
    print("  {} gap(s) touch a half boundary (censored: length unknown, no bracket on one side)\n"
          .format(censored))

    hist = histogram_rows(gaps)
    print("  {:>8} {:>8} | {:>7} {:>6} | {:>9} {:>6} {:>6} | {:>7} {:>7} {:>7} {:>6}".format(
        "frames", "~time", "gaps", "%gaps", "frames", "%miss", "cum%", "med m", "p90 m", ">40m/s", "flip%"))
    for r in hist:
        fmt = lambda v, f: "-" if v is None else f.format(v)  # noqa: E731
        print("  {:>8} {:>8} | {:>7,} {:>5.1f}% | {:>9,} {:>5.1f}% {:>5.1f}% | {:>7} {:>7} {:>6}% {:>5}%".format(
            r["bin"], r["approx_time"], r["n_gaps"], r["pct_of_gaps"], r["frames"],
            r["pct_of_missing_frames"], r["cum_pct_of_missing_frames"],
            fmt(r["median_bracket_m"], "{:.1f}"), fmt(r["p90_bracket_m"], "{:.1f}"),
            fmt(r["pct_implausible_speed"], "{:.0f}"), fmt(r["pct_team_flip"], "{:.0f}")))
    print("\n  med m / p90 m  distance the ball moved between the last frame before the gap and the first"
          "\n                 after it: the worst-case error of holding the last position to the gap's end"
          "\n  >40m/s         share of gaps whose implied speed is not physically a kicked ball"
          "\n  flip%          share of gaps where the nearest-player team differs across the gap"
          "{}".format("" if codes is not None else "  (skipped)"))

    print("\n  share of missing frames inside gaps no longer than:")
    shares = {}
    for frames in CUMULATIVE_AT:
        share = pct(int(length[length <= frames].sum()), missing_total)
        shares[frames] = share
        print("    {:>4} frames ({:>5.2f} s)  {:5.1f}%   [{:5.1f}% of gaps]".format(
            frames, frames / float(FPS), share, pct(int((length <= frames).sum()), n_gaps)))
    report["gap_summary"] = {
        "n_gaps": n_gaps, "missing_frames": missing_total, "censored_gaps": censored,
        "median": float(np.median(length)), "p90": float(np.percentile(length, 90)),
        "p99": float(np.percentile(length, 99)), "max": int(length.max()),
        "pct_missing_frames_in_gaps_up_to": {str(k): v for k, v in shares.items()},
        "bins": hist}

    # Where is the ball last seen before it is lost? DATA.md says boxes and crowds.
    zone = PENALTY_AREA_DEPTH / PITCH_L
    seen_x = ball[in_play & ball_ok, 0]
    base_rate = pct(int(((seen_x < zone) | (seen_x > 1 - zone)).sum()), len(seen_x))
    has_before = gaps["has_before"]
    last_x = ball[gaps["start"][has_before] - 1, 0]
    near_goal = (last_x < zone) | (last_x > 1 - zone)
    weight = length[has_before]
    print("\n  ball last seen within {:.1f} m of a goal line: {:.1f}% of missing frames "
          "(vs {:.1f}% of frames where the ball is seen)".format(
              PENALTY_AREA_DEPTH, pct(int(weight[near_goal].sum()), int(weight.sum())), base_rate))
    report["near_goal_line"] = {"pct_missing_frames": pct(int(weight[near_goal].sum()), int(weight.sum())),
                                "pct_seen_frames": base_rate}

    top = np.argsort(-length)[:args.top]
    print("\n  longest gaps (open one with:  python visualize_tracking.py --start-frame <start-100> "
          "--duration 20)")
    print("  {:>5} {:>9} {:>9} {:>8} {:>7} {:>8} {:>6}".format(
        "half", "start", "clock", "frames", "secs", "bracket", "flip"))
    for i in top:
        d = gaps["dist_m"][i]
        print("  {:>5} {:>9} {:>9} {:>8,} {:>7.1f} {:>8} {:>6}".format(
            gaps["half"][i], gaps["start"][i], clock(int(gaps["start"][i])), length[i],
            length[i] / float(FPS), "-" if np.isnan(d) else "{:.1f} m".format(d),
            {-1: "-", 0: "no", 1: "YES"}[int(gaps["team_flip"][i])]))

    # exact histogram + every gap, for your own plotting
    exact = np.bincount(length)
    with open(os.path.join(args.out, "ball_gap_histogram.csv"), "w") as handle:
        handle.write("length_frames,length_seconds,n_gaps,missing_frames\n")
        for value in np.flatnonzero(exact):
            handle.write("{},{:.2f},{},{}\n".format(value, value / float(FPS), exact[value],
                                                    int(value) * int(exact[value])))
    with open(os.path.join(args.out, "ball_gaps.csv"), "w") as handle:
        handle.write("half,start_frame,length_frames,bracketed,bracket_distance_m,"
                     "implied_speed_ms,team_flip\n")
        for i in range(n_gaps):
            handle.write("{},{},{},{},{},{},{}\n".format(
                gaps["half"][i], gaps["start"][i], length[i],
                int(gaps["has_before"][i] and gaps["has_after"][i]),
                "" if np.isnan(gaps["dist_m"][i]) else "{:.2f}".format(gaps["dist_m"][i]),
                "" if np.isnan(gaps["speed_ms"][i]) else "{:.2f}".format(gaps["speed_ms"][i]),
                "" if gaps["team_flip"][i] < 0 else gaps["team_flip"][i]))
    if not args.no_plot and plot_histogram(os.path.join(args.out, "ball_gap_histogram.png"),
                                           gaps, n_play):
        print("\n  plot -> {}".format(os.path.join(args.out, "ball_gap_histogram.png")))

    # -- 4. players -----------------------------------------------------------------------------
    heading("4. Tracked players (46 slots, in play)")
    total = present[in_play].sum(axis=1)
    home = present[in_play][:, team == baseline.HOME].sum(axis=1)
    away = present[in_play][:, team == baseline.AWAY].sum(axis=1)
    print("  tracked per frame: mean {:.1f}  median {:.0f}  min {}  max {}  "
          "(DATA.md: mean 20.5 over graded frames)".format(total.mean(), np.median(total),
                                                           total.min(), total.max()))
    print("  home mean {:.1f}   away mean {:.1f}".format(home.mean(), away.mean()))
    for name, mask in (("no player at all", total == 0), ("fewer than 18", total < 18),
                       ("fewer than 20", total < 20), ("more than 22 (duplicates)", total > 22)):
        print("  frames with {:<26} {:>8,}  {:5.1f}%".format(name, int(mask.sum()), pct(int(mask.sum()), n_play)))
    active = present[in_play].mean(axis=0)
    print("  slots never tracked in play: {} of {}   median slot active {:.1f}% of in-play frames".format(
        int((active == 0).sum()), n_slots, 100 * float(np.median(active))))
    report["players"] = {"mean": float(total.mean()), "median": float(np.median(total)),
                         "min": int(total.min()), "max": int(total.max()),
                         "pct_lt_18": pct(int((total < 18).sum()), n_play),
                         "pct_lt_20": pct(int((total < 20).sum()), n_play),
                         "pct_gt_22": pct(int((total > 22).sum()), n_play),
                         "slots_never_active": int((active == 0).sum())}

    # -- 5. malformed records -------------------------------------------------------------------
    heading("5. Malformed or odd records (whole file)")
    for key, value in issues.items():
        print("  {:<26} {:>9,}".format(key, value))
    report["parse_issues"] = issues

    # -- 6. homography --------------------------------------------------------------------------
    heading("6. Homography")
    try:
        from starter import load_homography
        _, error, valid = load_homography.load()
        check = load_homography.check_alignment()
        print("  rows {} (expected {}), contiguous {}".format(
            check["n_rows"], check["n_video_frames"], check["contiguous"]))
        print("  valid: {:.1f}% overall, {:.1f}% in play".format(
            100 * valid.mean(), 100 * valid[in_play].mean()))
        usable = error[in_play & valid]
        if len(usable):
            print("  reprojection error in play: median {:.2f}  p99 {:.2f}  max {:.2f}".format(
                np.median(usable), np.percentile(usable, 99), usable.max()))
        report["homography"] = {"valid_pct_in_play": 100 * float(valid[in_play].mean()),
                                "error_median": float(np.median(usable)) if len(usable) else None}
    except (IOError, ImportError, ValueError) as exc:
        print("  skipped: {}".format(str(exc).splitlines()[0]))

    # -- 7. smart tagging -----------------------------------------------------------------------
    heading("7. Smart Tagging coverage")
    try:
        from starter import load_smart_tagging
        events = load_smart_tagging.load()
        cover = np.zeros(n_frames + 1, dtype=np.int32)
        for e in events:
            s, t = max(0, e["start_frame"]), min(n_frames, e["end_frame"])
            if t > s:
                cover[s] += 1
                cover[t] -= 1
        cover = np.cumsum(cover)[:n_frames]
        none = in_play & (cover == 0)
        no_team = sum(1 for e in events if (e["tags"].get("Team") or "").strip().upper() == "N/A")
        print("  {} instances, {} tagged Team = N/A".format(len(events), no_team))
        print("  in-play frames covered by no phase at all: {:,}  ({:.1f}%)".format(
            int(none.sum()), pct(int(none.sum()), n_play)))
        print("  ...of which the ball is also missing:     {:,}  ({:.1f}% of those)".format(
            int((none & ~ball_ok).sum()), pct(int((none & ~ball_ok).sum()), int(none.sum()))))
        print("  time base agrees with the tactical kickoffs: {}".format(
            load_smart_tagging.check_time_base(events)["agrees"]))
        report["smart_tagging"] = {"n": len(events), "team_na": no_team,
                                   "pct_in_play_uncovered": pct(int(none.sum()), n_play)}
    except (IOError, ImportError) as exc:
        print("  skipped: {}".format(str(exc).splitlines()[0]))

    with open(os.path.join(args.out, "data_audit.json"), "w") as handle:
        json.dump(report, handle, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print("\nwrote {}/ ball_gap_histogram.csv, ball_gaps.csv, data_audit.json".format(args.out))


if __name__ == "__main__":
    main()