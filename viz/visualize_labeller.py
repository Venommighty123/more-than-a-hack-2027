"""Watch a labeller's CSV on the tactical video: what label it wrote on every frame, as a team.

    python viz/visualize_labeller.py --range 11185 12684             # frames 11185..12684
    python viz/visualize_labeller.py --start 23:40 --duration 60     # 1 min from video time 23:40
    python viz/visualize_labeller.py --full                          # whole graded match, 1 mp4 per half
    python viz/visualize_labeller.py --full --every 5                # same, time-lapse (5x faster)
    python viz/visualize_labeller.py --range 11185 18684 --spells-only   # list, no video
    python viz/visualize_labeller.py --labels evaluation/results/labeller_v4.csv --range ...
    python viz/visualize_labeller.py --compare --range 73997 75081 \
        --labels outputs/submissions/v6_atd.csv --labels2 outputs/submissions/v6_det_local.csv
        # five panels: original | video + ATD | video + YOLO / top ATD | top YOLO | info
    python viz/visualize_labeller.py --compare --range 74010 74110 --ids \
        --jumps outputs/track_audit/jumps_atd.csv
        # same, with the jumps found by viz/track_jumps.py highlighted in red (ATD and/or YOLO CSV)

WHAT IT SHOWS. Nothing is recomputed: for video frame f it draws exactly the label on row f of the
CSV (default: outputs/submissions/v6.csv, labeller v6 on our YOLO players). H is shown as the home club, A as the away club
(names from the ATD metadata), D as dead ball, X as not graded.
  Left    tactical video, tracked players / ball projected through the homography, a thick border in
          the label's colour (red H, blue A, grey D) and a badge such as "A  Athletic Club".
  Right   top-down pitch from viz/common.py (players, ball, ball trail, nearest player);
          greyed out while the label is D.
  Banner  on every label change: POSSESSION <team> -> <team>, STOPPAGE, RESTART -> <team>.
  Info    frame (= CSV row), clock, half, label, which possession of the match it is, when it started,
          how long it has run, time to the next change, ball tracked / missing, Smart Tagging phase.
  Timeline  the labels of the window with a tick at each change and a cursor; under it the whole
          match, with the window boxed.
  --gt    adds the public 5-minute ground truth as a band and in the info bar (where it exists).

OUTPUT  outputs/labeller_viz/<labels>_<first>_<last>.mp4, and next to it
          _spells.csv  every possession / stoppage in the window (start, end, seconds, label, team)
          _drawn.csv   the label drawn on every rendered frame; at the end it is compared with the CSV
                       and "CHECK: drawn labels match CSV" (or the mismatches) is printed.
CONSISTENCY  video frame f = CSV row f = ATD frame_id f + 1 (DATA.md), the same mapping the labeller
          run scripts use. The video is read with ffmpeg (frame-accurate seek, as in
          viz/common.py); without ffmpeg OpenCV is used and --exact-seek is advised.
          With --every N only every Nth frame is drawn: change banners and timeline ticks still come
          from every frame, but a spell shorter than N frames may not be seen.

RULES. Watching to UNDERSTAND errors is allowed. Turning what you see on BAR-ATH into labels,
corrections, constants, rules or a choice of version is not (RULES.md rule 1). The public GT band is
for viewing only, never for choosing settings (rule 3). The mp4 and sidecars are derived from match
data: keep them out of the public repository and out of any cloud AI tool.

Needs what viz/common.py needs: numpy, pandas, opencv-python (+ ffmpeg or imageio-ffmpeg).
"""

import argparse
import csv
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from viz import common as vt  # noqa: E402  (readers, video, pitch panel, drawing helpers)
from games import GAME, GAMES, require, half_video_blocks, half_of_video_frame  # noqa: E402
from starter.baseline import team_of_each_track, team_names  # noqa: E402

cv2 = vt.cv2
FPS = GAMES[GAME]["fps"]
N_FRAMES = GAMES[GAME]["n_video_frames"]
PANEL_W, PANEL_H, INFO_H = vt.PANEL_W, vt.PANEL_H, vt.INFO_H
TL_H = 86                                     # a little taller than viz/common.py's: 3 bands
DEFAULT_LABELS = os.path.join(ROOT, "outputs", "submissions", "v6.csv")
DEFAULT_OUT = os.path.join(ROOT, "outputs", "labeller_viz")
LABEL_BGR = {"H": vt.HOME_BGR, "A": vt.AWAY_BGR, "D": (140, 140, 140), "X": (45, 45, 45)}
BORDER = 10


# -- small helpers -------------------------------------------------------------------------------

def progress(label, i, n, t0, width=30):
    """One line, updated in place: percent, count, speed, elapsed, ETA."""
    f = (i + 1) / float(max(1, n))
    el = time.time() - t0
    eta = el / f - el if f > 0 else 0.0
    bar = "#" * int(width * f) + "-" * (width - int(width * f))
    print("\r  {} [{}] {:5.1f}%  {}/{}  {:5.1f} fps  {:5.0f}s elapsed  ETA {:5.0f}s".format(
        label, bar, 100 * f, i + 1, n, (i + 1) / max(el, 1e-6), el, eta), end="", flush=True)
    if i + 1 >= n:
        print(flush=True)


def runs(values, lo, hi):
    """[(first, last, value)] maximal constant runs of values[lo..hi], inclusive frames."""
    seg = values[lo:hi + 1]
    if not len(seg):
        return []
    cut = np.flatnonzero(seg[1:] != seg[:-1]) + 1
    starts = np.r_[0, cut]
    ends = np.r_[cut, len(seg)] - 1
    return [(lo + int(a), lo + int(b), str(seg[a])) for a, b in zip(starts, ends)]


def secs(frames):
    return frames / float(FPS)


# -- the labels ----------------------------------------------------------------------------------

def load_labels(path):
    """The CSV as an array of N_FRAMES labels, after the same checks the scorer makes on shape."""
    if not os.path.isfile(path):
        sys.exit("labels file not found: {}".format(path))
    with open(path, newline="") as handle:
        reader = csv.reader(handle)
        header = [h.strip().lower() for h in next(reader)]
        if header[:2] != ["frame", "label"]:
            sys.exit("{}: header must be 'frame,label', found {}".format(path, header))
        rows = [(int(r[0]), r[1].strip().upper()) for r in reader if r]
    if len(rows) != N_FRAMES:
        sys.exit("{}: {} rows, expected {}".format(path, len(rows), N_FRAMES))
    frames = np.array([r[0] for r in rows])
    if not np.array_equal(frames, np.arange(N_FRAMES)):
        sys.exit("{}: frames must be 0..{} in order".format(path, N_FRAMES - 1))
    labels = np.array([r[1] for r in rows], dtype="<U1")
    bad = sorted(set(labels) - set("HADX"))
    if bad:
        sys.exit("{}: unexpected labels {}".format(path, bad))
    return labels


class MatchSpells:
    """Every possession / stoppage of the graded match, numbered in match order (runs never cross
    a half boundary), and a frame -> run lookup used by the info bar."""

    def __init__(self, labels):
        self.items = []
        for half, a, b in half_video_blocks(GAME):
            for s, e, v in runs(labels, a, b):
                self.items.append({"half": half, "start": s, "end": e, "label": v})
        k_poss = k_stop = 0
        for it in self.items:
            if it["label"] in "HA":
                k_poss += 1
                it["n"] = k_poss
            elif it["label"] == "D":
                k_stop += 1
                it["n"] = k_stop
        self.n_poss, self.n_stop = k_poss, k_stop
        self.starts = np.array([it["start"] for it in self.items])

    def at(self, frame):
        """The run containing frame, or None outside the graded halves."""
        j = int(np.searchsorted(self.starts, frame, side="right")) - 1
        if j < 0:
            return None
        it = self.items[j]
        return it if it["start"] <= frame <= it["end"] else None


# -- windows -------------------------------------------------------------------------------------

def windows(args):
    """[(first, last)] inclusive video frames to render."""
    if args.full:
        return [(a, b) for _, a, b in half_video_blocks(GAME)]
    if args.range:
        a, b = args.range
    else:
        if args.start is None:
            sys.exit("give a window: --range FIRST LAST, --start mm:ss --duration s, or --full")
        a = int(round(vt.parse_time(args.start) * FPS))
        b = a + int(round((args.duration or 60.0) * FPS)) - 1
    a, b = max(0, int(a)), min(N_FRAMES - 1, int(b))
    if a > b:
        sys.exit("empty window {}..{}".format(a, b))
    return [(a, b)]


def team_word(label, names):
    return {"H": names[0], "A": names[1], "D": "dead ball", "X": "not graded"}[label]


def describe_change(prev, cur, names):
    if prev in "HA" and cur in "HA":
        return "POSSESSION  {}  ->  {}".format(names[0 if prev == "H" else 1], names[0 if cur == "H" else 1])
    if cur == "D":
        return "STOPPAGE  (dead ball){}".format("" if prev not in "HA" else
                                                 "   {} had it".format(names[0 if prev == "H" else 1]))
    if prev == "D" and cur in "HA":
        return "RESTART  ->  {}".format(names[0 if cur == "H" else 1])
    return "{}  ->  {}".format(team_word(prev, names), team_word(cur, names))


def write_spells(path, labels, a, b, names):
    rows = runs(labels, a, b)
    with open(path, "w", newline="") as handle:
        w = csv.writer(handle)
        w.writerow(["start_frame", "end_frame", "start_clock", "seconds", "label", "team"])
        for s, e, v in rows:
            w.writerow([s, e, vt.fmt_time(s), "{:.2f}".format(secs(e - s + 1)), v, team_word(v, names)])
    return rows


def print_spells(rows, names, limit=60):
    print("  {:>8} {:>8} {:>9} {:>7}  {}".format("start", "end", "clock", "secs", "label"))
    for s, e, v in rows[:limit]:
        print("  {:>8} {:>8} {:>9} {:>7.1f}  {} ({})".format(s, e, vt.fmt_time(s), secs(e - s + 1), v,
                                                            team_word(v, names)))
    if len(rows) > limit:
        print("  ... {} more rows in the _spells.csv".format(len(rows) - limit))


# -- timeline ------------------------------------------------------------------------------------

def build_timeline(labels, gt, a, b, match_lo, match_hi):
    """Static timeline image and the frame -> x mapping for the window band."""
    W, L, R = 2 * PANEL_W, 92, 12
    tl = np.zeros((TL_H, W, 3), dtype=np.uint8)
    tl[:] = (25, 25, 25)
    n = b - a + 1
    cols = np.arange(L, W - R)

    def x_of(f):
        return int(L + (f - a) / float(max(1, n - 1)) * (W - L - R - 1))

    def band(vals, lo, hi, y0, y1):
        idx = np.clip(((cols - L) / float(W - L - R - 1) * (hi - lo)).astype(int) + lo, lo, hi)
        lab = vals[idx]
        for c, col in LABEL_BGR.items():
            tl[y0:y1, cols[lab == c]] = col

    vt.put(tl, "LABELS", (6, 17), 0.4)
    band(labels, a, b, 5, 22)
    for s, _, _ in runs(labels, a, b)[1:]:                 # one tick per change
        x = x_of(s)
        tl[22:28, x] = (255, 255, 255)
    y = 32
    if gt is not None:
        vt.put(tl, "PUBLIC GT", (6, y + 12), 0.4)
        full_gt = np.full(N_FRAMES, "X", dtype="<U1")
        full_gt[a:b + 1] = np.where(np.isin(gt, ["H", "A", "D"]), gt, "X")
        band(full_gt, a, b, y, y + 14)
        y += 20
    vt.put(tl, "MATCH", (6, TL_H - 8), 0.4)
    band(labels, match_lo, match_hi, TL_H - 20, TL_H - 6)

    def mx(f):
        return int(L + (f - match_lo) / float(max(1, match_hi - match_lo)) * (W - L - R - 1))

    cv2.rectangle(tl, (mx(a), TL_H - 22), (max(mx(b), mx(a) + 1), TL_H - 4), (0, 255, 255), 1)
    return tl, x_of, mx


# -- rendering -----------------------------------------------------------------------------------

def render(args, labels, spells, a, b, names, team, raw_path, out_path):
    n = b - a + 1
    every = max(1, args.every)
    todo = list(range(a, b + 1, every))
    print("\nwindow {}..{} ({} frames, {} to {}), drawing {} frames{} -> {}".format(
        a, b, n, vt.fmt_time(a), vt.fmt_time(b + 1), len(todo),
        " (every {}th)".format(every) if every > 1 else "", out_path))

    t = time.time()
    print("  reading tracking for the window ...", flush=True)
    players, ball, team, _ = vt.read_window(a, n)       # our YOLO players + the fixed ball
    print("  tracking read in {:.1f} s, ball present on {:.1f}% of frames".format(
        time.time() - t, 100.0 * np.mean(~np.isnan(ball[:, 0]))))
    tags = vt.load_tags_window(not args.no_tags, a, n)
    gt = vt.load_gt_window(args.gt_path, a, n) if args.gt else None
    if args.gt and gt is None:
        print("  note: no public GT for this window")

    H = H_valid = project = None
    if not args.no_overlay and not args.no_video:
        try:
            from starter.load_homography import project
            H, H_valid = vt.load_homography_window(a, n)
        except (IOError, ImportError, ValueError) as error:
            print("  note: homography overlay disabled ({})".format(str(error).splitlines()[0]))
            H = None
    reader, src_w = (None, 1920)
    if not args.no_video:
        reader, src_w = vt.open_video(args, a, n)
    scale = PANEL_W / float(src_w)

    blocks = half_video_blocks(GAME)
    timeline, x_of, mx = build_timeline(labels, gt, a, b, blocks[0][1], blocks[-1][2])
    panel = vt.PitchPanel(PANEL_W, PANEL_H, flip_y=args.flip_y)
    size = (2 * PANEL_W, PANEL_H + INFO_H + TL_H)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, size)
    if not writer.isOpened():
        sys.exit("could not open the video writer for {}".format(out_path))

    change_at = {s: (labels[s - 1], v) for s, _, v in runs(labels, a, b) if s > 0 and labels[s - 1] != v}
    if a > 0 and labels[a - 1] != labels[a]:
        change_at[a] = (labels[a - 1], labels[a])
    banner_frames = max(1, int(round(args.banner * FPS)))
    banners, drawn, prev_drawn = [], [], a - 1
    t = time.time()
    try:
        for k in range(n):
            f = a + k
            img = reader.read() if reader is not None else None
            if reader is not None and img is None:
                msg = reader.error()
                print("\n  video ended at frame {}{}".format(f, ("\n  " + msg) if msg else ""))
                break
            if (f - a) % every:
                continue                                        # time-lapse: decoded, not drawn
            lab = str(labels[f])                                 # THE label: row f of the CSV
            # changes since the previous drawn frame (all of them, also the skipped ones)
            for g in range(prev_drawn + 1, f + 1):
                if g in change_at:
                    p, c = change_at[g]
                    banners.append((g + banner_frames, describe_change(p, c, names), LABEL_BGR[c], vt.fmt_time(g)))
            prev_drawn = f
            banners = [x for x in banners if x[0] > f][-3:]

            canvas = np.zeros((size[1], size[0], 3), dtype=np.uint8)

            # Left: video + projected tracking + label border and badge.
            if img is None:
                img = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
                vt.put(img, "video disabled (--no-video)", (20, 80))
            if H is not None and k < len(H_valid) and H_valid[k]:
                ok_p = ~np.isnan(players[k, :, 0]) & ~np.isnan(players[k, :, 1])
                u, v = project(H[k], players[k, ok_p, 0], players[k, ok_p, 1])
                for uu, vv, tm in zip(u, v, team[ok_p]):
                    if np.isfinite(uu) and np.isfinite(vv):
                        cv2.circle(img, (int(uu * scale), int(vv * scale)), 4,
                                   vt.HOME_BGR if tm == 0 else vt.AWAY_BGR, -1, cv2.LINE_AA)
                if not np.isnan(ball[k]).any():
                    u, v = project(H[k], ball[k, 0], ball[k, 1])
                    if np.isfinite(u) and np.isfinite(v):
                        cv2.circle(img, (int(u * scale), int(v * scale)), 7, (255, 255, 255), 2, cv2.LINE_AA)
            col = LABEL_BGR[lab]
            cv2.rectangle(img, (BORDER // 2, BORDER // 2), (PANEL_W - 1 - BORDER // 2, PANEL_H - 1 - BORDER // 2),
                          col, BORDER)
            badge = "{}  {}".format(lab, team_word(lab, names))
            (tw, th), _ = cv2.getTextSize(badge, vt.FONT, 0.9, 2)
            cv2.rectangle(img, (BORDER + 6, BORDER + 6), (BORDER + 26 + tw, BORDER + 22 + th), col, -1)
            vt.put(img, badge, (BORDER + 16, BORDER + 14 + th), 0.9, (255, 255, 255), 2)
            canvas[:PANEL_H, :PANEL_W] = img

            # Right: pitch.
            trail = ball[max(0, k - args.trail):k + 1] if args.trail > 0 else ball[k:k + 1]
            pitch, n_home, n_away, nearest = panel.render(players[k], ball[k], team, trail, names[0], names[1],
                                                          args.ids)
            if lab == "D":
                pitch = cv2.addWeighted(pitch, 0.6, np.full_like(pitch, 110), 0.4, 0)
                vt.put(pitch, "DEAD BALL (label D)", (PANEL_W - 250, PANEL_H - 14), 0.6, (235, 235, 235), 2)
            for j, (until, text, bcol, clock) in enumerate(reversed(banners)):
                y = 30 + 28 * j
                fade = min(1.0, (until - f) / float(banner_frames) * 2.0)
                box = pitch[y:y + 25, 4:PANEL_W - 4]
                over = np.full_like(box, 0)
                over[:] = bcol
                pitch[y:y + 25, 4:PANEL_W - 4] = cv2.addWeighted(box, 1 - 0.6 * fade, over, 0.6 * fade, 0)
                vt.put(pitch, "{}   at {}".format(text, clock)[:100], (12, y + 18), 0.55)
            canvas[:PANEL_H, PANEL_W:] = pitch

            # Info bar.
            y0 = PANEL_H
            half = half_of_video_frame(GAME, f)
            vt.put(canvas, "frame {} (CSV row {})   t {}   half {}".format(f, f, vt.fmt_time(f), half or "-"),
                   (12, y0 + 22), 0.58)
            vt.put(canvas, "LABEL  {}  =  {}".format(lab, team_word(lab, names)), (12, y0 + 52), 0.8, col, 2)
            sp = spells.at(f)
            if sp is not None:
                kind = "possession" if sp["label"] in "HA" else "stoppage" if sp["label"] == "D" else "run"
                total = spells.n_poss if kind == "possession" else spells.n_stop
                vt.put(canvas, "{} #{} of {}   started {}   running {:.1f} s of {:.1f} s   next change in {:.1f} s".format(
                    kind, sp.get("n", "-"), total, vt.fmt_time(sp["start"]), secs(f - sp["start"] + 1),
                    secs(sp["end"] - sp["start"] + 1), secs(sp["end"] - f + 1)), (12, y0 + 80), 0.52)
            else:
                vt.put(canvas, "outside the graded halves", (12, y0 + 80), 0.52, (170, 170, 170))
            has_ball = not np.isnan(ball[k, 0])
            vt.put(canvas, "tracked H {}  A {}   ball {}   nearest-to-ball {}".format(
                n_home, n_away, "tracked" if has_ball else "MISSING", {0: "H", 1: "A", -1: "-"}[nearest]),
                (12, y0 + 108), 0.52, (255, 255, 255) if has_ball else (120, 120, 255))
            tg = vt.active_tags(tags, f)
            if tg:
                vt.put(canvas, "Smart Tagging: " + " | ".join(tg)[:80], (PANEL_W + 12, y0 + 22), 0.5, (200, 255, 200))
            if gt is not None:
                g = gt[k] if k < len(gt) else "X"
                if g != "X":
                    same = g == lab
                    vt.put(canvas, "public GT {}  ({})   view only".format(g, "agree" if same else "DIFFERENT"),
                           (PANEL_W + 12, y0 + 52), 0.55, (120, 255, 120) if same else (110, 110, 255))
            vt.put(canvas, "labels: {}".format(os.path.basename(args.labels)), (PANEL_W + 12, y0 + 108), 0.48,
                   (170, 170, 170))

            # Timeline.
            tl = timeline.copy()
            x = x_of(f)
            cv2.line(tl, (x, 0), (x, TL_H - 26), (255, 255, 255), 2)
            xm = mx(f)
            cv2.line(tl, (xm, TL_H - 22), (xm, TL_H - 4), (255, 255, 255), 1)
            canvas[PANEL_H + INFO_H:, :] = tl

            writer.write(canvas)
            drawn.append((f, lab))
            j = len(drawn) - 1
            if j % 25 == 0 or j == len(todo) - 1:
                progress("rendering", j, len(todo), t)
    finally:
        writer.release()
        if reader is not None:
            reader.close()
    print()
    return drawn


# -- five-panel comparison: ATD vs our YOLO detections -------------------------------------------
# DIAGNOSTICS ONLY. The pipeline's players are YOLO (the normal render above draws them); --compare
# is the one place that still draws the provided ATD player slots, to see where the two disagree.

CELL_W, CELL_H = (2 * PANEL_W) // 3, (2 * PANEL_H) // 3      # 640 x 360: three cells = timeline width
REF_BGR, GK_BGR = (0, 230, 255), (255, 255, 255)


def load_yolo_window(path, a, b, team_mode, window=12):
    """Our detections for frames a..b: the slot layout v6 reads (top view) and the raw boxes (video).

    Slots come from possession/io/detections.py (same packing and team rule as
    `scripts/run.py --players detected`), so the top view shows exactly what the labeller was given.
    Boxes are read from the parquet with the same team rule computed on the window (+/- window).
    """
    import pandas as pd
    from possession.io.detections import _local_team, load_detected

    det = load_detected(path, n_frames=N_FRAMES, team_mode=team_mode, window=window)
    L, W = vt.PITCH_L, vt.PITCH_W
    slots = np.stack([det["px"][a:b + 1] / L, det["py"][a:b + 1] / W], axis=2)    # normalised
    rows = pd.read_parquet(path, filters=[("frame", ">=", a - window), ("frame", "<=", b + window)],
                           columns=["frame", "track_id", "role", "team", "x1", "y1", "x2", "y2",
                                    "x_m", "y_m", "team_p_home"])
    rows["vote"] = rows["team"].map({"H": 0, "A": 1})
    if team_mode == "local":
        rows, p = _local_team(rows, window)
        rows = rows.assign(tm=np.where(np.isfinite(p), (p < 0.5).astype(float), rows["vote"].to_numpy()))
    else:
        rows = rows.assign(tm=rows["vote"])
    rows = rows[(rows.frame >= a) & (rows.frame <= b)].sort_values("frame")
    starts = np.searchsorted(rows.frame.to_numpy(), np.arange(a, b + 2))
    return slots, det["team"], rows.reset_index(drop=True), starts, det["stats"]


def draw_atd_video(img, Hk, players_k, team, ball_k, scale, project):
    ok = ~np.isnan(players_k[:, 0]) & ~np.isnan(players_k[:, 1])
    if ok.any():
        u, v = project(Hk, players_k[ok, 0], players_k[ok, 1])
        for uu, vv, tm in zip(np.atleast_1d(u), np.atleast_1d(v), team[ok]):
            if np.isfinite(uu) and np.isfinite(vv):
                c = (int(uu * scale), int(vv * scale))
                cv2.circle(img, c, 6, vt.HOME_BGR if tm == 0 else vt.AWAY_BGR, -1, cv2.LINE_AA)
                cv2.circle(img, c, 6, (0, 0, 0), 1, cv2.LINE_AA)
    if not np.isnan(ball_k).any():
        u, v = project(Hk, ball_k[0], ball_k[1])
        if np.isfinite(u) and np.isfinite(v):
            cv2.circle(img, (int(u * scale), int(v * scale)), 8, (255, 255, 255), 2, cv2.LINE_AA)


def draw_yolo_video(img, rows, scale, show_ids):
    for r in rows.itertuples():
        if r.role == "referee":
            col = REF_BGR
        elif np.isfinite(r.tm):
            col = vt.HOME_BGR if r.tm == 0 else vt.AWAY_BGR
        else:
            col = (160, 160, 160)
        x1, y1, x2, y2 = r.x1 * scale, r.y1 * scale, r.x2 * scale, r.y2 * scale
        cx, w = int((x1 + x2) / 2), max(int((x2 - x1) / 2), 5)
        cv2.ellipse(img, (cx, int(y2)), (w, max(w // 3, 2)), 0, -45, 235, col, 2, cv2.LINE_AA)
        if r.role == "goalkeeper":
            cv2.ellipse(img, (cx, int(y2)), (w + 4, max(w // 3, 2) + 3), 0, -45, 235, GK_BGR, 1, cv2.LINE_AA)
        if show_ids:
            vt.put(img, str(int(r.track_id) % 1_000_000), (int(x1), int(y1) - 3), 0.35, col)


JUMP_BGR = (0, 0, 255)
JUMP_HOLD = 25                                   # frames a jump stays highlighted before / after


def load_jumps(paths, a, b, hold=JUMP_HOLD):
    """Events of viz/track_jumps.py CSVs that touch frames a..b (with the hold), as a list of dicts."""
    import pandas as pd
    out = []
    for path in paths or []:
        if not os.path.isfile(path):
            sys.exit("jumps file not found: {} (run viz/track_jumps.py first)".format(path))
        E = pd.read_csv(path)
        E = E[(E.end + hold >= a) & (E.start - hold <= b)]
        out += E.to_dict("records")
    return out


def draw_jumps(f, k, events, panel, top_atd, top_yolo, atd, yolo, players, rows, H, hv, scale, project):
    """Red overlay for every jump active at frame f: on the top view, a cross at the position
    before, a dashed arrow to the position after, a ring and a caption; for ATD also the slot's own
    trail over the last second; on the matching video panel, a ring on the track right now."""
    shown = 0
    for e in events:
        if not (e["start"] - JUMP_HOLD <= f <= e["end"] + JUMP_HOLD):
            continue
        atd_src = e["source"] == "atd"
        top, vid = (top_atd, atd) if atd_src else (top_yolo, yolo)
        live = e["start"] <= f <= e["end"]
        pre = panel.metres(e["pre_x"], e["pre_y"])
        post = panel.metres(e["post_x"], e["post_y"])
        vt._cross(top, pre, JUMP_BGR, 6, 2)
        vt._dashed(top, pre, post, JUMP_BGR, 2)
        vt._ring(top, post, JUMP_BGR, 14 if live else 11, 3 if live else 2)
        if atd_src:                                              # the slot's trail, last second
            tr = players[max(0, k - FPS):k + 1, int(e["id"]) - 1, :2]
            for p0, p1 in zip(tr[:-1], tr[1:]):
                if np.isfinite(p0).all() and np.isfinite(p1).all():
                    cv2.line(top, tuple(int(v) for v in panel.norm(*p0)), tuple(int(v) for v in panel.norm(*p1)),
                             JUMP_BGR, 2, cv2.LINE_AA)
        txt = "{} JUMP id {} {} {:.0f} m/s {:.1f} m  {}".format(
            e["source"].upper(), int(e["id"]), e["kind"], e["peak_speed"], e["jump_m"], e.get("verdict", ""))
        if isinstance(e.get("partner"), (int, float)) and e["partner"] > 0:
            txt += "  (with {})".format(int(e["partner"]))
        y = 60 + 24 * shown                                      # below the panel caption strip
        cv2.rectangle(top, (6, y - 18), (min(top.shape[1] - 6, 14 + 9 * len(txt)), y + 6), (0, 0, 0), -1)
        vt.put(top, txt[:90], (12, y), 0.55, (90, 90, 255) if not live else (255, 255, 255), 1)
        shown += 1
        # Ring on the video: where the track is NOW (ATD slot or YOLO box), else the event's end.
        if hv:
            now = None
            if atd_src:
                xy = players[k, int(e["id"]) - 1, :2]
                if np.isfinite(xy).all():
                    now = xy
            else:
                r = rows[rows.track_id == e["id"]]
                if len(r):
                    r = r.iloc[0]
                    cv2.circle(vid, (int((r.x1 + r.x2) / 2 * scale), int(r.y2 * scale)), 22, JUMP_BGR, 3, cv2.LINE_AA)
                    continue
            if now is None:
                now = (e["post_x"] / vt.PITCH_L, e["post_y"] / vt.PITCH_W)
            u, v = project(H[k], now[0], now[1])
            if np.isfinite(u) and np.isfinite(v):
                cv2.circle(vid, (int(u * scale), int(v * scale)), 18, JUMP_BGR, 3, cv2.LINE_AA)
    return shown


def label_frame(img, lab, names, title):
    col = LABEL_BGR[lab]
    cv2.rectangle(img, (BORDER // 2, BORDER // 2), (img.shape[1] - 1 - BORDER // 2, img.shape[0] - 1 - BORDER // 2),
                  col, BORDER)
    badge = "{}   {}  {}".format(title, lab, team_word(lab, names))
    (tw, th), _ = cv2.getTextSize(badge, vt.FONT, 0.8, 2)
    cv2.rectangle(img, (BORDER + 6, BORDER + 6), (BORDER + 24 + tw, BORDER + 20 + th), col, -1)
    vt.put(img, badge, (BORDER + 14, BORDER + 13 + th), 0.8, (255, 255, 255), 2)


def caption(img, text, top=False):
    """Panel title on a black strip: bottom for videos, top-right for the pitch views (their own
    legend sits at the bottom)."""
    (tw, _), _ = cv2.getTextSize(text, vt.FONT, 0.75, 2)
    if top:
        x0 = img.shape[1] - tw - 28
        cv2.rectangle(img, (x0, 0), (img.shape[1], 34), (0, 0, 0), -1)
        cv2.putText(img, text, (x0 + 14, 25), vt.FONT, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    else:
        cv2.rectangle(img, (0, img.shape[0] - 34), (img.shape[1], img.shape[0]), (0, 0, 0), -1)
        cv2.putText(img, text, (14, img.shape[0] - 10), vt.FONT, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def cell(img):
    return cv2.resize(img, (CELL_W, CELL_H), interpolation=cv2.INTER_AREA)


def render_compare(args, labels, labels2, a, b, names, team, raw_path, out_path):
    """One mp4, five synced panels + an info cell + timeline:

        original video | video + ATD players | video + YOLO players
        top view ATD   | top view YOLO        | info (labels, counts, nearest-to-ball)
    """
    n = b - a + 1
    print("\nCOMPARE window {}..{} ({} frames, {} to {}) -> {}".format(
        a, b, n, vt.fmt_time(a), vt.fmt_time(b + 1), out_path))
    players, ball = vt.read_tracking(raw_path, a, n, len(team))
    slots, yteam, rows, starts, st = load_yolo_window(args.players_file, a, b, args.team_mode)
    print("  YOLO: {} slots (H {}, A {}), team mode {}, {} boxes in the window".format(
        st["slots"], st["slots_home"], st["slots_away"], st["team_mode"], len(rows)))
    from starter.load_homography import project
    H, H_valid = vt.load_homography_window(a, n)
    reader, src_w = vt.open_video(args, a, n)
    scale = PANEL_W / float(src_w)
    jumps = load_jumps(args.jumps, a, b)
    if args.jumps:
        print("  jumps: {} events from {} touch this window".format(len(jumps), ", ".join(args.jumps)))

    blocks = half_video_blocks(GAME)
    second = labels2[a:b + 1] if labels2 is not None else None
    timeline, x_of, mx = build_timeline(labels, second, a, b, blocks[0][1], blocks[-1][2])
    if second is not None:                                   # the GT band holds labels2 here
        timeline[32:46, :90] = (25, 25, 25)
        vt.put(timeline, "LABELS 2", (6, 44), 0.4)
    panel = vt.PitchPanel(PANEL_W, PANEL_H, flip_y=args.flip_y)
    size = (3 * CELL_W, 2 * CELL_H + TL_H)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, size)
    if not writer.isOpened():
        sys.exit("could not open the video writer for {}".format(out_path))
    name1 = os.path.splitext(os.path.basename(args.labels))[0]
    name2 = os.path.splitext(os.path.basename(args.labels2))[0] if labels2 is not None else None
    drawn, t = [], time.time()
    try:
        for k in range(n):
            f = a + k
            img = reader.read()
            if img is None:
                print("\n  video ended at frame {}".format(f))
                break
            lab = str(labels[f])
            lab2 = str(labels2[f]) if labels2 is not None else None
            hv = k < len(H_valid) and H_valid[k]

            raw = img.copy()
            atd = img.copy()
            if hv:
                draw_atd_video(atd, H[k], players[k], team, ball[k], scale, project)
            yolo = img.copy()
            yr = rows.iloc[starts[k]:starts[k + 1]]
            draw_yolo_video(yolo, yr, scale, args.ids)
            label_frame(atd, lab, names, "ATD run" if labels2 is None else name1)
            if lab2 is not None:
                label_frame(yolo, lab2, names, name2)

            trail = ball[max(0, k - args.trail):k + 1] if args.trail > 0 else ball[k:k + 1]
            top_atd, ah, aa, anear = panel.render(players[k], ball[k], team, trail, names[0], names[1], args.ids)
            top_yolo, yh, ya, ynear = panel.render(slots[k], ball[k], yteam, trail, names[0], names[1], args.ids)
            for r in yr.itertuples():                           # referees: not in the slots v6 reads
                if r.role == "referee" and np.isfinite(r.x_m):
                    px_, py_ = panel.norm(r.x_m / vt.PITCH_L, r.y_m / vt.PITCH_W)
                    cv2.rectangle(top_yolo, (int(px_) - 5, int(py_) - 5), (int(px_) + 5, int(py_) + 5), REF_BGR, -1)
            if jumps:
                draw_jumps(f, k, jumps, panel, top_atd, top_yolo, atd, yolo, players, yr, H, hv, scale, project)
            for top, lb in ((top_atd, lab), (top_yolo, lab2 if lab2 is not None else lab)):
                if lb == "D":
                    top[:] = cv2.addWeighted(top, 0.6, np.full_like(top, 110), 0.4, 0)

            n_ref = int((yr.role == "referee").sum())
            n_gk = int((yr.role == "goalkeeper").sum())
            info = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
            info[:] = (30, 30, 30)
            half = half_of_video_frame(GAME, f)
            lines = [
                ("frame {}   t {}   half {}".format(f, vt.fmt_time(f), half or "-"), 0.8, (255, 255, 255)),
                ("{}: {}  {}".format(name1, lab, team_word(lab, names)), 0.85, LABEL_BGR[lab]),
            ]
            if lab2 is not None:
                lines.append(("{}: {}  {}".format(name2, lab2, team_word(lab2, names)), 0.85, LABEL_BGR[lab2]))
                lines.append(("labels AGREE" if lab == lab2 else "labels DIFFER", 0.85,
                              (120, 255, 120) if lab == lab2 else (110, 110, 255)))
            lines += [
                ("ATD   players H {}  A {}   nearest-to-ball {}".format(ah, aa, {0: "H", 1: "A", -1: "-"}[anear]),
                 0.7, (230, 230, 230)),
                ("YOLO  players H {}  A {}   nearest-to-ball {}".format(yh, ya, {0: "H", 1: "A", -1: "-"}[ynear]),
                 0.7, (230, 230, 230)),
                ("YOLO  referees {}   goalkeepers {}".format(n_ref, n_gk), 0.7, (230, 230, 230)),
                ("ball {}  (ATD ball on both top views)".format(
                    "tracked" if not np.isnan(ball[k, 0]) else "MISSING"), 0.7,
                 (255, 255, 255) if not np.isnan(ball[k, 0]) else (120, 120, 255)),
                ("over 11 on a team:  ATD {}   YOLO {}".format(
                    "YES" if max(ah, aa) > 11 else "no", "YES" if max(yh, ya) > 11 else "no"), 0.7, (230, 230, 230)),
                ("ellipse = YOLO box, yellow = referee, white ring = GK", 0.6, (170, 170, 170)),
            ]
            for j, (text, sc, col) in enumerate(lines):
                cv2.putText(info, text[:70], (24, 50 + 52 * j), vt.FONT, sc, col, 2 if sc >= 0.8 else 1,
                            cv2.LINE_AA)                 # plain text: the outline ghosts on the dark cell

            canvas = np.zeros((size[1], size[0], 3), dtype=np.uint8)
            row1 = [caption(raw, "1  original"), caption(atd, "2  ATD players"), caption(yolo, "3  YOLO players")]
            row2 = [caption(top_atd, "4  top view ATD", top=True),
                    caption(top_yolo, "5  top view YOLO (what v6 reads)", top=True), info]
            for i, im in enumerate(row1):
                canvas[:CELL_H, i * CELL_W:(i + 1) * CELL_W] = cell(im)
            for i, im in enumerate(row2):
                canvas[CELL_H:2 * CELL_H, i * CELL_W:(i + 1) * CELL_W] = cell(im)
            tl = timeline.copy()
            x = x_of(f)
            cv2.line(tl, (x, 0), (x, TL_H - 26), (255, 255, 255), 2)
            xm = mx(f)
            cv2.line(tl, (xm, TL_H - 22), (xm, TL_H - 4), (255, 255, 255), 1)
            canvas[2 * CELL_H:, :] = tl[:, :size[0]]
            writer.write(canvas)
            drawn.append((f, lab))
            if len(drawn) % 25 == 0 or k == n - 1:
                progress("rendering", len(drawn) - 1, n, t)
    finally:
        writer.release()
        reader.close()
    print()
    return drawn


def check_drawn(drawn, labels, path):
    """Write what was drawn and prove it is the CSV."""
    with open(path, "w", newline="") as handle:
        w = csv.writer(handle)
        w.writerow(["frame", "label_drawn", "label_csv"])
        bad = []
        for f, lab in drawn:
            w.writerow([f, lab, labels[f]])
            if lab != labels[f]:
                bad.append(f)
    if bad:
        print("CHECK FAILED: {} of {} drawn frames differ from the CSV, first: {}".format(len(bad), len(drawn), bad[:10]))
    else:
        print("CHECK: drawn labels match CSV ({}/{} frames)".format(len(drawn), len(drawn)))
    return not bad


# -- main ----------------------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--labels", default=DEFAULT_LABELS, help="labeller CSV (default: the v6 safe submission)")
    w = ap.add_argument_group("window (pick one)")
    w.add_argument("--range", type=int, nargs=2, metavar=("FIRST", "LAST"), help="video frames, inclusive")
    w.add_argument("--start", help="video time hh:mm:ss / mm:ss / seconds")
    w.add_argument("--duration", type=float, help="seconds, with --start (default 60)")
    w.add_argument("--full", action="store_true", help="the whole graded match, one mp4 per half")
    ap.add_argument("--every", type=int, default=1, help="draw every Nth frame (time-lapse); default 1 = all")
    ap.add_argument("--spells-only", action="store_true", help="list possessions / stoppages, no video")
    ap.add_argument("--gt", action="store_true", help="show the public 5-minute GT (view only)")
    ap.add_argument("--gt-path", default=vt.DEFAULT_GT)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--banner", type=float, default=1.5, help="seconds a change banner stays up")
    ap.add_argument("--trail", type=int, default=50, help="ball trail in frames (0 = off)")
    ap.add_argument("--ids", action="store_true", help="label each player with its track slot")
    ap.add_argument("--no-overlay", action="store_true", help="do not project tracking onto the video")
    ap.add_argument("--no-video", action="store_true", help="pitch only, no mp4 decode")
    ap.add_argument("--no-tags", action="store_true", help="do not show Smart Tagging phases")
    ap.add_argument("--no-flip-y", dest="flip_y", action="store_false")
    ap.add_argument("--backend", choices=("auto", "ffmpeg", "opencv"), default="auto")
    ap.add_argument("--exact-seek", action="store_true", help="OpenCV backend only: frame-accurate, slow start")
    ap.add_argument("--video-offset", type=int, default=0, help="shift the video by N frames (as viz/render_match.py)")
    c = ap.add_argument_group("five-panel comparison with our YOLO detections")
    c.add_argument("--compare", action="store_true",
                   help="original | video+ATD | video+YOLO / top ATD | top YOLO | info, one mp4")
    c.add_argument("--labels2", default=None,
                   help="a second labeller CSV (e.g. the YOLO run), shown on the YOLO panel and timeline")
    c.add_argument("--players-file", default=os.path.join(ROOT, "data", "kaggle_outputs", "final_tracks_full.parquet"))
    c.add_argument("--team-mode", choices=("local", "track"), default="local",
                   help="team of a YOLO player, as scripts/run.py --team-mode")
    c.add_argument("--jumps", nargs="+", default=None, metavar="CSV",
                   help="jump lists from viz/track_jumps.py (jumps_atd.csv and/or jumps_yolo.csv): drawn in red")
    return ap.parse_args()


def main():
    args = parse_args()
    labels = load_labels(args.labels)
    spells = MatchSpells(labels)
    print("labels: {}   {} possessions, {} stoppages in the graded match".format(
        args.labels, spells.n_poss, spells.n_stop))
    meta_path, raw_path = require(["atd_meta", "atd_raw"])
    team = team_of_each_track(meta_path)
    names = team_names(meta_path)
    print("H = {}, A = {}".format(*names))

    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.labels))[0]
    all_ok = True
    for a, b in windows(args):
        base = os.path.join(args.out_dir, "{}_{}_{}{}".format(stem, a, b, "_x{}".format(args.every) if args.every > 1 else ""))
        rows = write_spells(base + "_spells.csv", labels, a, b, names)
        n_poss = sum(v in "HA" for _, _, v in rows)
        print("\nwindow {}..{}: {} runs ({} possessions, {} stoppages) -> {}".format(
            a, b, len(rows), n_poss, sum(v == "D" for _, _, v in rows), base + "_spells.csv"))
        if args.spells_only:
            print_spells(rows, names)
            continue
        if args.compare:
            base += "_compare"
            labels2 = load_labels(args.labels2) if args.labels2 else None
            drawn = render_compare(args, labels, labels2, a, b, names, team, raw_path, base + ".mp4")
        else:
            drawn = render(args, labels, spells, a, b, names, team, raw_path, base + ".mp4")
        all_ok &= check_drawn(drawn, labels, base + "_drawn.csv")
        print("wrote {}".format(os.path.abspath(base + ".mp4")))
    if not args.spells_only and not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
