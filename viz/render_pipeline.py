"""Render what the sync / referee / ball-fill pipeline sees: tactical and broadcast side by side, in sync.

Left panel : the TACTICAL video. Ball markers projected through the tactical homography:
             white dot     = ATD ball (provided tracking)
             green dot     = fixed ball (outputs/ball/ball_reconstructed.npz) where the ATD has none
             orange ring   = cleaned YOLO detection (ball_yolo_fixed.npz, source yolo: the fill candidate)
             grey cross    = YOLO bridged / flight-fit frame (not used by the fill)
Right panel: the BROADCAST frame showing the same instant (sync notebook, outputs/sync/**/sync_full.json),
             with the referee box + skeleton from the P3 notebook (outputs/referee/**/referee_frames.parquet):
             yellow = referee seen, red = arm pointing. A banner shows referee pointing events.
Info bar   : tactical frame, video time, half, match clock | labeller v6 state (DEAD d=.. entry->exit, or
             LIVE) and the stretch's referee signal | ball status (ATD / fixed / YOLO) | the ball gap the
             frame is in and the fill decision (ball_filled_gaps.csv) | broadcast frame and sync label.
Timeline   : v6 dead (grey), ATD ball missing (red), YOLO detection (orange), broadcast live (green),
             referee pointing (yellow), and a cursor.

Also writes outputs/dead/dead_frames_v6_old.csv: one row per tactical frame of the labeller v6 dead
stretches (the "old" dead-ball frames, before any ball filling), with the same per-frame facts.

Usage (repository root; every input is optional except the tactical video and the v6 stretches):

    python viz/render_pipeline.py                         # overview: gesture stretches + 3 gaps + 3 dead stretches
    python viz/render_pipeline.py --d 12 25               # the given v6 dead stretches
    python viz/render_pipeline.py --gap 31 40             # the given ball gaps (ball_filled_gaps.csv)
    python viz/render_pipeline.py --start-frame 17000 --seconds 30
    python viz/render_pipeline.py --csv-only              # only the dead-frames CSV
    python viz/render_pipeline.py --d 12 --pad 4 --max-seconds 60 --out-dir outputs/visualization

Outputs go to outputs/visualization/ (clips + index.csv) and outputs/dead/. Derived match data:
never commit or upload them (RULES.md). Clips are for looking at the pipeline, not for labelling.
"""
import argparse
import csv
import glob
import json
import os
import subprocess
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402

from viz.common import (FONT, FPS, INFO_H, PANEL_H, PANEL_W, PITCH_L, PITCH_W, ROOT, TL_H, cv2, find_ffmpeg,  # noqa: E402
                        load_homography_window, require)
from possession.io.tracking import load_ball  # noqa: E402
from starter.load_homography import project  # noqa: E402

OUT_DIR = os.path.join(ROOT, "outputs", "visualization")
DEAD_CSV = os.path.join(ROOT, "outputs", "dead", "dead_stretches_labeller_v6_newtracks.csv")
DEAD_FRAMES_CSV = os.path.join(ROOT, "outputs", "dead", "dead_frames_v6_old.csv")
BASE = os.path.join(ROOT, "outputs", "ball", "ball_reconstructed.npz")
YOLO = os.path.join(ROOT, "outputs", "ball", "ball_yolo_fixed.npz")
GAPS = os.path.join(ROOT, "outputs", "ball", "ball_filled_gaps.csv")
GLOBS = {"sync": "outputs/sync/**/sync_full.json", "ref_frames": "outputs/referee/**/referee_frames.parquet",
         "ref_events": "outputs/referee/**/referee_events.csv", "ref_stretches": "outputs/referee/**/referee_stretches.csv"}
LABELS = {0: "OFF_MATCH", 1: "LIVE_WIDE", 2: "LIVE_CLOSE", 3: "REPLAY", 4: "UNKNOWN"}
SKEL = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12)]
WHITE, GREEN, ORANGE, GREY, RED, YELLOW = (255, 255, 255), (60, 220, 60), (0, 140, 255), (150, 150, 150), (40, 40, 230), (0, 230, 255)
W_OUT, H_OUT = 2 * PANEL_W, PANEL_H + INFO_H + TL_H


def put(img, s, org, scale=0.55, color=(255, 255, 255), thick=1):
    """Text on a solid dark box: readable on grass and video, no outline ghosting."""
    (w, h), base = cv2.getTextSize(s, FONT, scale, thick)
    x, y = org
    cv2.rectangle(img, (x - 3, y - h - 4), (x + w + 3, y + base + 2), (0, 0, 0), -1)
    cv2.putText(img, s, (x, y), FONT, scale, color, thick, cv2.LINE_AA)


def _find(pattern, explicit=None):
    if explicit:
        return explicit if os.path.exists(explicit) else None
    hits = sorted(glob.glob(os.path.join(ROOT, pattern), recursive=True))
    return hits[0] if hits else None


def _read_csv(path):
    if not path or not os.path.exists(path):
        return []
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def mmss(seconds):
    if seconds is None or not np.isfinite(seconds):
        return "--:--.-"
    sign = "-" if seconds < 0 else ""
    m, s = divmod(abs(float(seconds)), 60.0)
    return "{}{:02d}:{:04.1f}".format(sign, int(m), s)


# --------------------------------------------------------------------------------------------- data
class Pipeline:
    """Everything the pipeline produced, loaded once; each part optional except the v6 stretches."""

    def __init__(self, args):
        self.stretches = _read_csv(args.dead)
        if not self.stretches:
            sys.exit("missing v6 dead stretches: {}".format(args.dead))
        base = np.load(args.base, allow_pickle=False) if os.path.exists(args.base) else None
        yolo = np.load(args.yolo, allow_pickle=False) if os.path.exists(args.yolo) else None
        self.n = len(base["x"]) if base is not None else int(max(int(s["last_frame"]) for s in self.stretches) + 1)
        self.base_x = base["x"] if base is not None else np.full(self.n, np.nan)
        self.base_y = base["y"] if base is not None else np.full(self.n, np.nan)
        self.yolo_x = yolo["x"] if yolo is not None else np.full(self.n, np.nan)
        self.yolo_y = yolo["y"] if yolo is not None else np.full(self.n, np.nan)
        self.yolo_src = yolo["src"] if yolo is not None else np.zeros(self.n, np.int8)
        self.dead_d = np.zeros(self.n, dtype=np.int32)
        self.by_d = {}
        for s in self.stretches:
            a, b = int(s["first_frame"]), int(s["last_frame"])
            self.dead_d[a:min(b, self.n - 1) + 1] = int(s["d"])
            self.by_d[int(s["d"])] = s
        self.gaps = _read_csv(args.gaps)
        self.gap_id = np.full(self.n, -1, dtype=np.int32)
        for r in self.gaps:
            a, b = int(r["first_frame"]), int(r["last_frame"])
            self.gap_id[a:min(b, self.n - 1) + 1] = int(r["gap"])
        self.gap_by_id = {int(r["gap"]): r for r in self.gaps}
        self.ref_signal = {int(r["d"]): r["referee_signal"] for r in _read_csv(_find(GLOBS["ref_stretches"], args.ref_stretches))}
        self.ref_events = [r for r in _read_csv(_find(GLOBS["ref_events"], args.ref_events))
                           if str(r.get("replay_flag", "False")) != "True"]
        self.pm = self.labels = None
        sp = _find(GLOBS["sync"], args.sync)
        if sp:
            from possession.referee.windows import PairMap
            with open(sp) as handle:
                sync = json.load(handle)
            self.pm = PairMap(sync)
            bfr = sync.get("broadcast_frames")
            self.labels = np.full(self.pm.n_b, 4, dtype=np.int8)
            if bfr:
                lab = np.asarray(bfr["columns"]["label"], dtype=np.int8)
                self.labels[bfr["start"]:bfr["start"] + len(lab)] = lab
        self.ref = None
        rp = _find(GLOBS["ref_frames"], args.ref_frames)
        if rp:
            try:
                import pandas as pd
                self.ref = pd.read_parquet(rp, columns=["frame", "x0", "y0", "x1", "y1", "kp", "pointing", "direction"])
            except Exception as exc:                       # pyarrow missing etc.
                print("note: referee frames not loaded ({}): no skeletons".format(exc))
        self.ref_point_b = set(int(f) for f in self.ref.loc[self.ref["pointing"], "frame"]) if self.ref is not None else set()
        self.ref_dir_b = ({int(f): d for f, d in zip(self.ref.loc[self.ref["pointing"], "frame"], self.ref.loc[self.ref["pointing"], "direction"])}
                          if self.ref is not None else {})
        print("loaded: v6 stretches {} | ball gaps {} | sync {} | referee frames {} | referee events {}".format(
            len(self.stretches), len(self.gaps), "yes" if self.pm else "no",
            0 if self.ref is None else len(self.ref), len(self.ref_events)))

    def broadcast(self, frames):
        """(half, match clock, broadcast frame (-1 none), label code) for tactical frames."""
        frames = np.asarray(frames)
        if self.pm is None:
            z = np.full(len(frames), -1)
            return z, np.full(len(frames), np.nan), z, np.full(len(frames), 4)
        half, clock, bf = self.pm.map_frames(frames)
        lab = np.where(bf >= 0, self.labels[np.clip(bf, 0, len(self.labels) - 1)], 4)
        return half, clock, bf, lab

    def referee_pointing_tactical(self, frames):
        """Tactical frames inside a (non-replay) referee pointing event."""
        out = np.zeros(len(frames), dtype=bool)
        f0 = int(frames[0]) if len(frames) else 0
        for e in self.ref_events:
            a, b = int(e.get("tactical_first", -1)), int(e.get("tactical_last", -1))
            if a < 0:
                continue
            lo, hi = max(a - f0, 0), min(b - f0 + 1, len(frames))
            if hi > lo:
                out[lo:hi] = True
        return out


# ------------------------------------------------------------------------------------------ the CSV
def write_dead_frames_csv(p, path):
    """One row per tactical frame of the v6 dead stretches, with what every stage says about it."""
    frames = np.concatenate([np.arange(int(s["first_frame"]), int(s["last_frame"]) + 1) for s in p.stretches])
    frames = frames[frames < p.n]
    bx, _, _ = load_ball(metres=False)
    atd = np.isfinite(bx[frames])
    half, clock, bf, lab = p.broadcast(frames)
    pointing = np.zeros(len(frames), dtype=bool)
    for e in p.ref_events:
        a, b = int(e.get("tactical_first", -1)), int(e.get("tactical_last", -1))
        if a >= 0:
            pointing |= (frames >= a) & (frames <= b)
    cols = ["tactical_frame", "d", "half", "video_time", "match_clock", "entry", "exit", "atd_ball", "fixed_ball",
            "yolo_detection", "ball_gap", "fill_decision", "broadcast_frame", "broadcast_label", "referee_signal",
            "referee_pointing"]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as handle:
        w = csv.writer(handle)
        w.writerow(cols)
        for i, f in enumerate(frames):
            d = int(p.dead_d[f]); s = p.by_d.get(d, {})
            g = int(p.gap_id[f])
            w.writerow([int(f), d, int(half[i]) if half[i] > 0 else s.get("half", ""), mmss(f / float(FPS)), mmss(clock[i]),
                        s.get("entry", ""), s.get("exit", ""), int(atd[i]), int(np.isfinite(p.base_x[f])),
                        int(p.yolo_src[f] == 1), g if g >= 0 else "", p.gap_by_id[g]["decision"] if g >= 0 else "",
                        int(bf[i]), LABELS.get(int(lab[i]), "") if bf[i] >= 0 else "", p.ref_signal.get(d, ""),
                        int(pointing[i])])
    print("dead frames: {} rows -> {} | ATD ball present on {:.1%} | YOLO detection on {:.1%} | referee pointing on {}".format(
        len(frames), os.path.relpath(path, ROOT), atd.mean(), np.mean(p.yolo_src[frames] == 1), int(pointing.sum())))
    return len(frames)


# ------------------------------------------------------------------------------------------ video io
class Reader:
    """Sequential reader with exact positioning: seek, check, else read forward from earlier."""

    def __init__(self, path):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise FileNotFoundError(path)
        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
        self.pos = 0

    def _goto(self, f):
        if f == self.pos:
            return
        if self.pos < f <= self.pos + 50:
            while self.pos < f and self.cap.grab():
                self.pos += 1
            return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        self.pos = int(round(self.cap.get(cv2.CAP_PROP_POS_FRAMES)))
        if self.pos != f:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, f - 250))
            self.pos = int(round(self.cap.get(cv2.CAP_PROP_POS_FRAMES)))
            while self.pos < f and self.cap.grab():
                self.pos += 1

    def read(self, f):
        self._goto(int(f))
        ok, img = self.cap.read()
        if ok:
            self.pos += 1
            return img
        return None

    def close(self):
        self.cap.release()


class Writer:
    """H.264 through ffmpeg when available (plays everywhere), else OpenCV mp4v."""

    def __init__(self, path):
        exe = find_ffmpeg()
        self.proc = self.vw = None
        if exe:
            self.proc = subprocess.Popen(
                [exe, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "{}x{}".format(W_OUT, H_OUT),
                 "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", path],
                stdin=subprocess.PIPE)
        else:
            self.vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W_OUT, H_OUT))

    def write(self, img):
        if self.proc:
            self.proc.stdin.write(img.tobytes())
        else:
            self.vw.write(img)

    def close(self):
        if self.proc:
            self.proc.stdin.close(); self.proc.wait()
        else:
            self.vw.release()


# ----------------------------------------------------------------------------------------- one clip
def _pt(H, x_norm, y_norm, sx, sy):
    if H is None or not (np.isfinite(x_norm) and np.isfinite(y_norm)) or not np.isfinite(H).all():
        return None
    px, py = project(H, x_norm, y_norm)
    px, py = float(px) * sx, float(py) * sy
    if not (np.isfinite(px) and np.isfinite(py)) or abs(px) > 5 * PANEL_W or abs(py) > 5 * PANEL_H:
        return None
    return int(round(px)), int(round(py))


def render_clip(p, first, last, path, title, tactical_video, broadcast_video):
    first, last = max(0, int(first)), min(p.n - 1, int(last))
    frames = np.arange(first, last + 1)
    n = len(frames)
    atd_x, atd_y, _ = load_ball(first, last, metres=False)
    try:
        Hs, hvalid = load_homography_window(first, n)
    except Exception as exc:
        print("note: no tactical homography for the clip ({}): ball markers off".format(exc))
        Hs, hvalid = None, np.zeros(n, bool)
    half, clock, bf, lab = p.broadcast(frames)
    tac = Reader(tactical_video)
    brd = Reader(broadcast_video) if broadcast_video and os.path.exists(broadcast_video) else None
    sx, sy = PANEL_W / float(tac.w), PANEL_H / float(tac.h)
    bsx, bsy = (PANEL_W / float(brd.w), PANEL_H / float(brd.h)) if brd else (1, 1)
    # timeline rows
    rows = [("v6 dead", p.dead_d[frames] > 0, GREY), ("ATD ball missing", ~np.isfinite(atd_x), RED),
            ("YOLO detection", p.yolo_src[frames] == 1, ORANGE), ("broadcast live", (bf >= 0) & np.isin(lab, (1, 2)), GREEN),
            ("referee pointing", p.referee_pointing_tactical(frames) | np.array([int(b) in p.ref_point_b for b in bf]), YELLOW)]
    tl = np.zeros((TL_H, W_OUT, 3), np.uint8)
    x0t, rh = 150, (TL_H - 8) // len(rows)
    for r, (name, mask, col) in enumerate(rows):
        y0 = 4 + r * rh
        put(tl, name, (6, y0 + rh - 3), 0.38, col)
        xs = x0t + (np.arange(n) * (W_OUT - x0t - 6) / max(1, n - 1)).astype(int)
        for i in np.flatnonzero(mask):
            cv2.line(tl, (xs[i], y0 + 1), (xs[i], y0 + rh - 2), col, 1)
    ref_b = p.ref.groupby("frame") if p.ref is not None else None
    ref_frames = set(p.ref["frame"].astype(int)) if p.ref is not None else set()
    out = Writer(path)
    for i, f in enumerate(frames):
        canvas = np.zeros((H_OUT, W_OUT, 3), np.uint8)
        img = tac.read(f)
        left = cv2.resize(img, (PANEL_W, PANEL_H), interpolation=cv2.INTER_AREA) if img is not None else np.zeros((PANEL_H, PANEL_W, 3), np.uint8)
        H = Hs[i] if Hs is not None and hvalid[i] else None
        src = int(p.yolo_src[f])
        if src in (2, 3):
            q = _pt(H, p.yolo_x[f] / PITCH_L, p.yolo_y[f] / PITCH_W, sx, sy)
            if q:
                cv2.drawMarker(left, q, GREY, cv2.MARKER_TILTED_CROSS, 14, 2)
        if src == 1:
            q = _pt(H, p.yolo_x[f] / PITCH_L, p.yolo_y[f] / PITCH_W, sx, sy)
            if q:
                cv2.circle(left, q, 11, ORANGE, 2, cv2.LINE_AA)
        q = _pt(H, atd_x[i], atd_y[i], sx, sy)
        if q:
            cv2.circle(left, q, 6, WHITE, -1, cv2.LINE_AA); cv2.circle(left, q, 6, (0, 0, 0), 1, cv2.LINE_AA)
        elif np.isfinite(p.base_x[f]):
            q = _pt(H, p.base_x[f] / PITCH_L, p.base_y[f] / PITCH_W, sx, sy)
            if q:
                cv2.circle(left, q, 6, GREEN, -1, cv2.LINE_AA)
        put(left, "TACTICAL  {}".format(title), (10, 24), 0.6)
        if p.dead_d[f]:
            put(left, "DEAD d={}".format(p.dead_d[f]), (10, 52), 0.8, (120, 120, 255), 2)
        # right: broadcast
        b = int(bf[i])
        right = None
        if brd and b >= 0:
            bimg = brd.read(b)
            if bimg is not None:
                right = cv2.resize(bimg, (PANEL_W, PANEL_H), interpolation=cv2.INTER_AREA)
                if b in ref_frames:
                    for r in ref_b.get_group(b).itertuples():
                        col = RED if r.pointing else YELLOW
                        cv2.rectangle(right, (int(r.x0 * bsx), int(r.y0 * bsy)), (int(r.x1 * bsx), int(r.y1 * bsy)), col, 2)
                        if r.kp is not None and len(r.kp) == 34:
                            kp = np.asarray(r.kp, dtype=float).reshape(17, 2) * [bsx, bsy]
                            for a_, c_ in SKEL:
                                if np.isfinite(kp[[a_, c_]]).all() and (kp[[a_, c_]] > 0).all():
                                    cv2.line(right, tuple(kp[a_].astype(int)), tuple(kp[c_].astype(int)), col, 2, cv2.LINE_AA)
        if right is None:
            right = np.full((PANEL_H, PANEL_W, 3), 30, np.uint8)
            put(right, "no broadcast frame for this instant" if b < 0 else "broadcast frame {} unreadable".format(b),
                (20, PANEL_H // 2), 0.7, GREY)
        put(right, "BROADCAST  frame {}  {}".format(b if b >= 0 else "-", LABELS.get(int(lab[i]), "") if b >= 0 else ""), (10, 24), 0.6)
        if b in p.ref_point_b or rows[4][1][i]:
            put(right, "REFEREE POINTING {}".format(p.ref_dir_b.get(b, "")).strip(), (10, 56), 0.9, RED, 2)
        canvas[:PANEL_H, :PANEL_W] = left
        canvas[:PANEL_H, PANEL_W:] = right
        # info bar
        d = int(p.dead_d[f]); s = p.by_d.get(d)
        g = int(p.gap_id[f]); gr = p.gap_by_id.get(g)
        y = PANEL_H + 26
        put(canvas, "tactical frame {}   video {}   half {}   match clock {}".format(
            f, mmss(f / float(FPS)), int(half[i]) if half[i] > 0 else "-", mmss(clock[i])), (12, y), 0.6)
        put(canvas, ("v6: DEAD d={} ({} .. {})   referee signal: {}".format(d, s.get("entry", ""), s.get("exit", ""),
                                                                           p.ref_signal.get(d, "n/a")) if s else "v6: LIVE"),
            (12, y + 30), 0.6, (150, 150, 255) if s else (150, 255, 150))
        put(canvas, "ball: ATD {}   fixed {}   YOLO {}".format(
            "yes" if np.isfinite(atd_x[i]) else "MISSING", "yes" if np.isfinite(p.base_x[f]) else "no",
            {0: "-", 1: "detection", 2: "bridged", 3: "flight fit", 4: "provided"}.get(src, "-")), (12, y + 60), 0.6)
        if gr:
            put(canvas, "ball gap #{} ({}, {} s): {} - {}".format(g, gr["scope"], round(int(gr["frames"]) / FPS, 1),
                                                              gr["decision"].upper(), gr["reason"]), (12, y + 90), 0.55, (0, 200, 255))
        put(canvas, "white = ATD ball   green = fixed ball   orange ring = YOLO detection   grey x = YOLO bridged/flight   "
                    "broadcast: yellow = referee, red = pointing", (PANEL_W + 12, y + 90), 0.42, GREY)
        t = tl.copy()
        cx = x0t + int(i * (W_OUT - x0t - 6) / max(1, n - 1))
        cv2.line(t, (cx, 0), (cx, TL_H), WHITE, 1)
        canvas[PANEL_H + INFO_H:, :] = t
        out.write(canvas)
    out.close(); tac.close()
    if brd:
        brd.close()
    return n


# --------------------------------------------------------------------------------------- selection
def select_clips(p, args):
    """[(name, first, last, title)] from the CLI, or the overview preset."""
    pad, cap = int(args.pad * FPS), int(args.max_seconds * FPS)
    clips = []

    def add(name, a, b, title):
        a, b = a - pad, b + pad
        clips.append((name, a, min(b, a + cap), title))

    if args.start_frame is not None:
        add("frames_{}".format(args.start_frame), args.start_frame + pad, args.start_frame + pad + int(args.seconds * FPS) - 1,
            "frames {}".format(args.start_frame))
    for d in args.d or []:
        s = p.by_d.get(d)
        if s:
            add("d{:03d}".format(d), int(s["first_frame"]), int(s["last_frame"]), "v6 stretch d={}".format(d))
        else:
            print("no v6 stretch d={}".format(d))
    for g in args.gap or []:
        r = p.gap_by_id.get(g)
        if r:
            add("gap{:03d}".format(g), int(r["first_frame"]), int(r["last_frame"]), "ball gap #{} ({})".format(g, r["scope"]))
        else:
            print("no ball gap #{}".format(g))
    if clips:
        return clips
    # overview preset
    for d, sig in sorted(p.ref_signal.items()):
        if sig == "yes" and d in p.by_d:
            s = p.by_d[d]
            add("d{:03d}_gesture".format(d), int(s["first_frame"]), int(s["last_frame"]), "v6 stretch d={} (referee gesture)".format(d))
    gaps = sorted((r for r in p.gaps if r["scope"] == "open_play" and r["decision"] in ("filled", "partial")),
                  key=lambda r: -int(r["kept_points"]))[:3]
    for r in gaps:
        add("gap{:03d}_open_play".format(int(r["gap"])), int(r["first_frame"]), int(r["last_frame"]),
            "ball gap #{} (open play, {})".format(r["gap"], r["decision"]))
    rng = np.random.default_rng(args.seed)
    others = [d for d in sorted(p.by_d) if p.ref_signal.get(d) != "yes"]
    for d in rng.choice(others, size=min(3, len(others)), replace=False) if others else []:
        s = p.by_d[int(d)]
        add("d{:03d}_dead".format(int(d)), int(s["first_frame"]), int(s["last_frame"]), "v6 stretch d={}".format(int(d)))
    return clips


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--d", type=int, nargs="*", help="v6 dead stretch ids")
    ap.add_argument("--gap", type=int, nargs="*", help="ball gap ids (ball_filled_gaps.csv)")
    ap.add_argument("--start-frame", type=int, default=None, help="tactical frame to start a free clip at")
    ap.add_argument("--seconds", type=float, default=30.0, help="length of the --start-frame clip")
    ap.add_argument("--pad", type=float, default=2.0, help="seconds shown before and after a stretch / gap")
    ap.add_argument("--max-seconds", type=float, default=40.0, help="cap per clip (long stretches show their start)")
    ap.add_argument("--csv-only", action="store_true")
    ap.add_argument("--no-csv", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--dead", default=DEAD_CSV)
    ap.add_argument("--dead-frames-csv", default=DEAD_FRAMES_CSV)
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--yolo", default=YOLO)
    ap.add_argument("--gaps", default=GAPS)
    for k in ("sync", "ref_frames", "ref_events", "ref_stretches"):
        ap.add_argument("--" + k.replace("_", "-"), dest=k, default=None)
    args = ap.parse_args()

    p = Pipeline(args)
    if not args.no_csv:
        write_dead_frames_csv(p, args.dead_frames_csv)
    if args.csv_only:
        return
    tactical_video = require(["video"])[0]
    try:
        broadcast_video = require(["video"], source="broadcast")[0]
    except (KeyError, SystemExit, FileNotFoundError, ValueError):
        broadcast_video = None
    os.makedirs(args.out_dir, exist_ok=True)
    index = []
    for name, a, b, title in select_clips(p, args):
        path = os.path.join(args.out_dir, "clip_{}.mp4".format(name))
        n = render_clip(p, a, b, path, title, tactical_video, broadcast_video)
        index.append({"clip": os.path.basename(path), "title": title, "tactical_first": a, "tactical_last": a + n - 1,
                      "seconds": round(n / float(FPS), 1)})
        print("  {}  {:5.1f} s  {}".format(os.path.relpath(path, ROOT), n / float(FPS), title))
    with open(os.path.join(args.out_dir, "index.csv"), "w", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=["clip", "title", "tactical_first", "tactical_last", "seconds"])
        w.writeheader(); w.writerows(index)
    print("{} clips -> {}".format(len(index), os.path.relpath(args.out_dir, ROOT)))


if __name__ == "__main__":
    main()
