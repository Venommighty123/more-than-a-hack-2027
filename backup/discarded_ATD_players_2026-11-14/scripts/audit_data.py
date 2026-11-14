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

Section 5 audits the team mapping, because a possession call is only as good as "whose player is
this?". baseline.team_of_each_track() maps every slot whose teamId is not the first team to AWAY,
so a slot with no team (or a third id) silently becomes an away player. Nothing in the repository
counts them, so this does: slots without a team, frames where a team is barely tracked, how often
the nearest-player rule cannot give a clear answer, and how often a slot's track breaks (a proxy
for the identity swaps DATA.md warns about).

Reuses, rather than re-implements: games (paths, half blocks), load_atd.check_alignment_raw,
baseline.team_of_each_track / nearest_player_labels, load_homography.load / check_alignment and
load_smart_tagging.load / check_time_base. Team labels are baseline.team_of_each_track, the mapping
under test. Moved from ball_gap_analysis.py; library versions of its gap logic live in possession/gaps/.

    pip install numpy pandas matplotlib          # matplotlib only for the PNG
    python scripts/audit_data.py                  # ~1-3 min: it streams the raw ATD twice
    python scripts/audit_data.py --skip-team-flip # ~1 min: skip the nearest-player pass
    python scripts/audit_data.py --out my_reports --top 20

Writes to outputs/audit/ (or --out):
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

try:
    import _bootstrap  # noqa: F401  (python scripts/x.py)
except ImportError:
    from scripts import _bootstrap  # noqa: F401  (python -m scripts.x)
ROOT = _bootstrap.ROOT
for _p in (ROOT, os.path.join(ROOT, "starter")):        # starter modules import each other flat
    if _p not in sys.path:
        sys.path.insert(0, _p)

from games import GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402
import baseline  # noqa: E402  (starter/baseline.py: the team mapping under audit)
import load_atd  # noqa: E402

# Params
FPS = GAMES[GAME]["fps"]
PITCH_L, PITCH_W = GAMES[GAME]["pitch"]
FIELD_SEP, GROUP_SEP, MISSING = ":", ";", "N"      # same conventions as baseline.py
DEFAULT_OUT = os.path.join(ROOT, "outputs", "audit")
# A kicked football rarely exceeds ~35-40 m/s. Faster implied speed across a gap means the two
# bracketing detections are not the same ball path (a mis-detection, or the ball left and re-entered).
MAX_PLAUSIBLE_SPEED = 40.0                         # m/s
PENALTY_AREA_DEPTH = 16.5                          # m
# Gap-length bins, lower bound in frames. Edges sit on 0.5 s / 1 s / 2 s / 5 s / 10 s / 20 s.
BIN_LOWER = (1, 2, 3, 6, 13, 26, 51, 126, 251, 501)
BIN_LABEL = ("1", "2", "3-5", "6-12", "13-25", "26-50", "51-125", "126-250", "251-500", ">500")
BIN_SECONDS = ("0.04 s", "0.08 s", "<0.24 s", "<0.5 s", "<1 s", "<2 s", "<5 s", "<10 s", "<20 s", ">20 s")
CUMULATIVE_AT = (1, 5, 12, 25, 50, 125, 250, 500)  # frames
# Team-mapping thresholds. Judgement calls, so they are named and easy to change.
MIN_TEAM_TRACKED = 8        # a team with fewer tracked players than this is too sparse to trust
AMBIGUOUS_M = 1.0           # nearest home and nearest away player within this many metres: a coin flip
UNATTENDED_M = 5.0          # nearest player farther than this: nobody is plausibly on the ball
MAX_PLAYER_SPEED = 12.0     # m/s, a sprint. Faster than this between detections = a different person
JUMP_SLACK_M = 2.0          # tracking noise allowed on top of MAX_PLAYER_SPEED

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
    """Per video frame: every track slot's position (NaN = absent) and the ball position.

    Also counts every malformed thing it meets. Raw line: "<frame_id>:<x,y,s>;...:<ball>", with
    frame_id = video frame + 1, and NaN for a missing observation.
    """
    pos = np.full((n_frames, n_slots, 2), np.nan, dtype=np.float32)    # normalised x, y per slot
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
                fields = group.split(",")
                if len(fields) < 2:
                    issues["player_position_partial"] += 1
                    continue
                if MISSING in group:
                    gone = [MISSING in f for f in fields[:2]]
                    if all(gone):
                        continue                               # a clean missing slot
                    if any(gone):
                        issues["player_position_partial"] += 1  # x or y missing, not both
                        continue
                    issues["player_speed_only_nan"] += 1        # position fine, speed NaN
                pos[i, slot] = (float(fields[0]), float(fields[1]))

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
    return pos, ball, issues


# -- gaps --------------------------------------------------------------------------------------

def find_gaps(ball_ok, blocks):
    """Maximal runs of missing ball inside each half. Returns dict of equal-length arrays.

    has_before / has_after say whether a detection exists on that side *within the half*. A gap
    touching the edge of a half is censored: its true length is unknown, and it has no
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


# -- team mapping ------------------------------------------------------------------------------

CALL_CLASSES = ("no_player", "nearest_is_unmapped", "one_team_only", "unattended", "ambiguous", "clear")
CALL_MEANING = {
    "no_player": "ball seen, but no player tracked at all",
    "nearest_is_unmapped": "nearest tracked slot has no team (baseline calls it AWAY)",
    "one_team_only": "ball seen, but one team has nobody tracked",
    "unattended": "nearest player more than {:.0f} m from the ball".format(UNATTENDED_M),
    "ambiguous": "nearest home and away player within {:.1f} m of each other".format(AMBIGUOUS_M),
    "clear": "a usable nearest-player team call",
}


def audit_slot_teams(meta_path):
    """Which track slots does the metadata give a team? Same XML and slot order as baseline.

    Returns (status, info): status is an array over slots of "home" / "away" / "unmapped", where
    unmapped means the teamId is missing or is neither of the two teams. baseline.team_of_each_track
    labels those AWAY, which is exactly what this exists to expose.
    """
    import xml.etree.ElementTree as ET
    root = ET.parse(meta_path).getroot()
    team_ids = [t.get("id") for t in root.iter("Team")]
    players = list(root.iter("Player"))
    ids = [p.get("id") for p in players]
    home_id = team_ids[0] if team_ids else None
    away_id = team_ids[1] if len(team_ids) > 1 else None
    status = np.array(["home" if p.get("teamId") == home_id else
                       "away" if p.get("teamId") == away_id else "unmapped" for p in players])

    # Does the raw column order follow the Player list? baseline assumes so. Where the metadata
    # names the channel order it can be checked; where it does not, say so rather than guess.
    channel_order = None
    channels = {c.get("id"): c.get("playerId") for c in root.iter("PlayerChannel")}
    refs = [r.get("playerChannelId") for r in root.iter("PlayerChannelRef")]
    if channels and refs:
        order = []
        for ref in refs:
            player_id = channels.get(ref)
            if player_id is not None and player_id not in order:
                order.append(player_id)
        if len(order) == len(ids):
            channel_order = {"verifiable": True, "same_order": order == ids}
        else:
            channel_order = {"verifiable": False,
                             "reason": "{} players in channel refs vs {} Player entries".format(
                                 len(order), len(ids))}
    return status, {"n_players": len(players), "team_ids": team_ids,
                    "duplicate_ids": len(ids) - len(set(ids)), "channel_order": channel_order}


def team_call_quality(pos, ball, in_play, status, team, codes):
    """How often the nearest-player rule gives a usable team call, and why not when it does not.

    Over in-play frames where the ball is seen. Each frame gets the FIRST class that applies:
    no_player > nearest_is_unmapped > one_team_only > unattended > ambiguous > clear.
    Also checks this vectorised rule against baseline.nearest_player_labels (codes) as a
    cross-check of both: they must agree wherever a player is tracked.
    """
    is_home, is_away = status == "home", status == "away"
    is_other = ~(is_home | is_away)
    frames = np.flatnonzero(in_play & ~np.isnan(ball[:, 0]))
    counts = np.zeros(len(CALL_CLASSES), dtype=np.int64)
    compared = mismatch = 0
    for chunk in np.array_split(frames, max(1, len(frames) // 20000)):
        p, b = pos[chunk], ball[chunk]
        d = np.hypot((p[:, :, 0] - b[:, None, 0]) * PITCH_L, (p[:, :, 1] - b[:, None, 1]) * PITCH_W)
        d = np.where(np.isnan(d), np.inf, d)                  # absent slot = infinitely far
        d_home = np.where(is_home, d, np.inf).min(axis=1)
        d_away = np.where(is_away, d, np.inf).min(axis=1)
        d_other = np.where(is_other, d, np.inf).min(axis=1)
        d_any = d.min(axis=1)
        nearest = np.minimum(d_home, d_away)
        with np.errstate(invalid="ignore"):                   # inf - inf
            close = np.abs(d_home - d_away) < AMBIGUOUS_M

        cls = np.full(len(chunk), 5, dtype=np.int8)           # clear, then overridden by priority
        cls[close] = 4
        cls[nearest > UNATTENDED_M] = 3
        cls[np.isinf(d_home) | np.isinf(d_away)] = 2
        cls[d_other < nearest] = 1
        cls[np.isinf(d_any)] = 0
        counts += np.bincount(cls, minlength=len(CALL_CLASSES))

        if codes is not None:
            has = ~np.isinf(d_any)
            mine = team[d.argmin(axis=1)]
            compared += int(has.sum())
            mismatch += int((codes[chunk][has] != mine[has]).sum())
    return {"frames": int(len(frames)), "counts": dict(zip(CALL_CLASSES, counts.tolist())),
            "baseline_compared": compared, "baseline_mismatch": mismatch}


def identity_breaks(pos, in_play, status):
    """Proxy for identity swaps: a slot moving farther between detections than a person can.

    Slots appear, vanish and swap identity (DATA.md). A swap that hands a slot to a player of the
    other team would silently flip its team, and no file says when it happens. What can be seen is
    the footprint: between two detections of one slot, dt frames apart, the position changes by
    more than MAX_PLAYER_SPEED * dt + JUMP_SLACK_M. That is a lower bound on swaps (a swap between
    two neighbours is invisible) and an upper bound on them too (a tracker glitch also trips it),
    so read it as an indicator, not a count.

    A detection is "suspect" for one second after a break.
    """
    n_frames, n_slots = pos.shape[:2]
    jumps, reappear = np.zeros(n_slots, dtype=np.int64), np.zeros(n_slots, dtype=np.int64)
    suspect_detections = np.zeros(n_slots, dtype=np.int64)
    detections = np.zeros(n_slots, dtype=np.int64)
    for slot in range(n_slots):
        x, y = pos[:, slot, 0] * PITCH_L, pos[:, slot, 1] * PITCH_W
        idx = np.flatnonzero(in_play & ~np.isnan(x))
        detections[slot] = len(idx)
        if len(idx) < 2:
            continue
        dt = np.diff(idx)
        moved = np.hypot(np.diff(x[idx]), np.diff(y[idx]))
        broke = moved > MAX_PLAYER_SPEED * dt / FPS + JUMP_SLACK_M
        jumps[slot] = int((broke & (dt == 1)).sum())          # teleported between adjacent frames
        reappear[slot] = int((broke & (dt > 1)).sum())        # came back somewhere it could not be
        starts = idx[1:][broke]
        mark = np.zeros(n_frames + 1, dtype=np.int32)
        np.add.at(mark, starts, 1)
        np.add.at(mark, np.minimum(starts + FPS, n_frames), -1)
        suspect_detections[slot] = int((in_play & ~np.isnan(x) & (np.cumsum(mark)[:n_frames] > 0)).sum())
    return {"jumps": jumps, "reappear": reappear, "suspect": suspect_detections,
            "detections": detections}


# -- main --------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=DEFAULT_OUT, help="output directory (default outputs/audit)")
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
    pos, ball, issues = scan_atd(raw_path, n_frames, n_slots)
    print("  done in {:.0f}s".format(time.time() - t0))
    present = ~np.isnan(pos[:, :, 0])
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

    # -- 5. team mapping ------------------------------------------------------------------------
    heading("5. Team mapping: how much of the tracking cannot say whose it is")
    status, slot_info = audit_slot_teams(meta_path)
    is_home, is_away, unmapped = status == "home", status == "away", status == "unmapped"
    print("  metadata: {} Player entries -> {} home, {} away, {} with no usable team".format(
        slot_info["n_players"], int(is_home.sum()), int(is_away.sum()), int(unmapped.sum())))
    if len(status) != n_slots:
        print("  WARNING: {} Player entries but the team array has {} slots".format(len(status), n_slots))
    if unmapped.any():
        print("  slots with no team (1-based): {}".format((np.flatnonzero(unmapped) + 1).tolist()))
        print("  baseline.team_of_each_track() labels all of them AWAY (labelled away: {})".format(
            bool((team[unmapped] == baseline.AWAY).all())))
    if slot_info["duplicate_ids"]:
        print("  WARNING: {} duplicate Player ids in the metadata".format(slot_info["duplicate_ids"]))
    order = slot_info["channel_order"]
    if order is None:
        print("  raw column order vs Player list: not verifiable (metadata names no player channels)")
    elif not order["verifiable"]:
        print("  raw column order vs Player list: not verifiable ({})".format(order["reason"]))
    else:
        print("  raw column order vs Player list: {}".format(
            "same" if order["same_order"] else "DIFFERENT -- baseline's slot -> team mapping is wrong"))

    seen = present[in_play]
    detections, from_unmapped = int(seen.sum()), int(seen[:, unmapped].sum())
    print("\n  in-play detections from slots with no team: {:,} of {:,} ({:.2f}%), on {:,} frames ({:.1f}%)".format(
        from_unmapped, detections, pct(from_unmapped, detections),
        int(seen[:, unmapped].any(axis=1).sum()), pct(int(seen[:, unmapped].any(axis=1).sum()), n_play)))

    n_home_seen, n_away_seen = seen[:, is_home].sum(axis=1), seen[:, is_away].sum(axis=1)
    print("\n  per-team coverage of in-play frames (mapped slots only):")
    coverage = {}
    for name, mask in (("home team has nobody tracked", n_home_seen == 0),
                       ("away team has nobody tracked", n_away_seen == 0),
                       ("home below {} tracked".format(MIN_TEAM_TRACKED), n_home_seen < MIN_TEAM_TRACKED),
                       ("away below {} tracked".format(MIN_TEAM_TRACKED), n_away_seen < MIN_TEAM_TRACKED),
                       ("either team below {}".format(MIN_TEAM_TRACKED),
                        (n_home_seen < MIN_TEAM_TRACKED) | (n_away_seen < MIN_TEAM_TRACKED))):
        coverage[name] = pct(int(mask.sum()), n_play)
        print("    {:<34} {:>8,}  {:5.1f}%".format(name, int(mask.sum()), coverage[name]))

    quality = team_call_quality(pos, ball, in_play, status, team, codes)
    n_ball_play = quality["frames"]
    print("\n  nearest-player team call, in-play frames where the ball is seen ({:,}):".format(n_ball_play))
    print("    {:<22} {:>8} {:>8} {:>8}   {}".format("outcome", "frames", "% ball", "% play", "meaning"))
    for name in CALL_CLASSES:
        c = quality["counts"][name]
        print("    {:<22} {:>8,} {:>7.1f}% {:>7.1f}%   {}".format(
            name, c, pct(c, n_ball_play), pct(c, n_play), CALL_MEANING[name]))
    ball_gone = n_play - n_ball_play
    unusable = n_ball_play - quality["counts"]["clear"]
    print("    {:<22} {:>8,} {:>8} {:>7.1f}%   no ball, so no call at all".format(
        "ball_missing", ball_gone, "", pct(ball_gone, n_play)))
    print("\n  => a clear team call exists on {:.1f}% of in-play frames; {:.1f}% have the ball but "
          "no clear call, {:.1f}% have no ball".format(
              pct(quality["counts"]["clear"], n_play), pct(unusable, n_play), pct(ball_gone, n_play)))
    if quality["baseline_compared"]:
        print("  cross-check vs baseline.nearest_player_labels: {:,} frames compared, {:,} differ".format(
            quality["baseline_compared"], quality["baseline_mismatch"]))
    elif codes is None:
        print("  cross-check vs baseline skipped (--skip-team-flip)")

    breaks = identity_breaks(pos, in_play, status)
    tot_det = int(breaks["detections"].sum())
    print("\n  identity-continuity breaks (a slot moves farther than a sprint allows between detections):")
    for name, mask in (("home slots", is_home), ("away slots", is_away), ("no-team slots", unmapped)):
        if not mask.any():
            continue
        print("    {:<14} {:>6,} teleports, {:>6,} implausible re-appearances, "
              "{:.2f}% of detections within 1 s of a break".format(
                  name, int(breaks["jumps"][mask].sum()), int(breaks["reappear"][mask].sum()),
                  pct(int(breaks["suspect"][mask].sum()), int(breaks["detections"][mask].sum()))))
    worst = np.argsort(-(breaks["jumps"] + breaks["reappear"]))[:5]
    print("    most breaks (slot, 1-based): {}".format(
        ", ".join("{} ({:,})".format(s + 1, int(breaks["jumps"][s] + breaks["reappear"][s])) for s in worst)))
    print("    all slots: {:.2f}% of detections are within 1 s of a break (an indicator of identity "
          "swaps, not a count)".format(pct(int(breaks["suspect"].sum()), tot_det)))

    report["team_mapping"] = {
        "slots": {"home": int(is_home.sum()), "away": int(is_away.sum()),
                  "unmapped": int(unmapped.sum()),
                  "unmapped_slots_1_based": (np.flatnonzero(unmapped) + 1).tolist()},
        "channel_order": order, "duplicate_player_ids": slot_info["duplicate_ids"],
        "pct_detections_from_unmapped_slots": pct(from_unmapped, detections),
        "team_coverage_pct_of_in_play_frames": coverage,
        "team_call": {**quality, "pct_clear_of_in_play": pct(quality["counts"]["clear"], n_play),
                      "pct_ball_missing_of_in_play": pct(ball_gone, n_play)},
        "identity_breaks": {"teleports": int(breaks["jumps"].sum()),
                            "reappearances": int(breaks["reappear"].sum()),
                            "pct_detections_suspect": pct(int(breaks["suspect"].sum()), tot_det)}}

    # -- 6. malformed records -------------------------------------------------------------------
    heading("6. Malformed or odd records (whole file)")
    for key, value in issues.items():
        print("  {:<26} {:>9,}".format(key, value))
    report["parse_issues"] = issues

    # -- 6. homography --------------------------------------------------------------------------
    heading("7. Homography")
    try:
        import load_homography
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
    heading("8. Smart Tagging coverage")
    try:
        import load_smart_tagging
        events = load_smart_tagging.load()

        def coverage_of(selected):
            """Per-frame count of the selected events covering that frame."""
            diff = np.zeros(n_frames + 1, dtype=np.int32)
            for e in selected:
                s, t = max(0, e["start_frame"]), min(n_frames, e["end_frame"])
                if t > s:
                    diff[s] += 1
                    diff[t] -= 1
            return np.cumsum(diff)[:n_frames]

        cover = coverage_of(events)
        none = in_play & (cover == 0)
        team_na = [e for e in events if (e["tags"].get("Team") or "").strip().upper() == "N/A"]
        no_team = len(team_na)
        in_na = in_play & (coverage_of(team_na) > 0)
        print("  {} instances, {} tagged Team = N/A (throw-ins and the like: no side to take from the tag)"
              .format(len(events), no_team))
        print("  in-play frames inside a Team = N/A phase:  {:,}  ({:.1f}%)".format(
            int(in_na.sum()), pct(int(in_na.sum()), n_play)))
        print("  in-play frames covered by no phase at all: {:,}  ({:.1f}%)".format(
            int(none.sum()), pct(int(none.sum()), n_play)))
        print("  ...of which the ball is also missing:     {:,}  ({:.1f}% of those)".format(
            int((none & ~ball_ok).sum()), pct(int((none & ~ball_ok).sum()), int(none.sum()))))
        print("  time base agrees with the tactical kickoffs: {}".format(
            load_smart_tagging.check_time_base(events)["agrees"]))
        report["smart_tagging"] = {"n": len(events), "team_na": no_team,
                                   "pct_in_play_in_team_na": pct(int(in_na.sum()), n_play),
                                   "pct_in_play_uncovered": pct(int(none.sum()), n_play)}
    except (IOError, ImportError) as exc:
        print("  skipped: {}".format(str(exc).splitlines()[0]))

    with open(os.path.join(args.out, "data_audit.json"), "w") as handle:
        json.dump(report, handle, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print("\nwrote {}/ ball_gap_histogram.csv, ball_gaps.csv, data_audit.json".format(args.out))


if __name__ == "__main__":
    main()