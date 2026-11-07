"""Shared building blocks of every renderer in viz/: lazy readers (tracking, homography, public GT,
Smart Tagging) for one clip, the video reader (ffmpeg pipe, OpenCV fallback), the top-down pitch
panel and drawing helpers.

Split out of visualize_tracking.py unchanged, so render_match.py, render_possession.py and the
clip scripts draw the same way. Needs numpy and opencv-python (+ ffmpeg or imageio-ffmpeg).
"""
import bisect  # noqa: F401
import csv  # noqa: F401
import json  # noqa: F401
import os
import shutil  # noqa: F401
import subprocess  # noqa: F401
import sys
import tempfile  # noqa: F401
import time  # noqa: F401

import numpy as np  # noqa: F401

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import possession.io  # noqa: F401  (repository root on sys.path)
from possession.io import ROOT
from games import GAME, GAMES, require, half_of_video_frame, half_video_blocks  # noqa: F401
from starter.baseline import team_of_each_track, team_names  # noqa: F401

try:
    import cv2
except ImportError:
    sys.exit("OpenCV is required:  pip install opencv-python")

FPS = GAMES[GAME]["fps"]

N_FRAMES = GAMES[GAME]["n_video_frames"]

PITCH_L, PITCH_W = GAMES[GAME]["pitch"]            # metres

HOME_BGR, AWAY_BGR = (40, 39, 214), (180, 119, 31)  # #d62728 / #1f77b4, in BGR

GRASS_BGR, LINE_BGR = (79, 143, 63), (255, 255, 255)

PANEL_W, PANEL_H = 960, 540

BAR_H = 70                  # kept for scripts that import it; the info bar is now INFO_H tall

INFO_H, TL_H = 124, 70      # info bar and timeline under the two panels

FONT = cv2.FONT_HERSHEY_SIMPLEX

DEFAULT_GT = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")

SEARCH_BLOCK = 1 << 20      # binary search narrows the ATD to ~1 MB, then scans lines linearly

LABEL_BGR = {"H": HOME_BGR, "A": AWAY_BGR, "D": (120, 120, 120), "X": (45, 45, 45)}

LABEL_NAME = {"H": "H (home)", "A": "A (away)", "D": "dead ball", "X": "unknown"}

def parse_time(text):
    """'1:15:49' / '07:27' / '447.4' -> seconds."""
    seconds = 0.0
    for part in str(text).split(":"):
        seconds = seconds * 60.0 + float(part)
    return seconds

def fmt_time(frame):
    s = frame / float(FPS)
    return "{:02d}:{:04.1f}".format(int(s // 60), s % 60)

def put(img, text, org, scale=0.55, color=(255, 255, 255), thick=1):
    """Outlined text, readable on both grass and video."""
    cv2.putText(img, text, org, FONT, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, color, thick, cv2.LINE_AA)

def _floats(group, k):
    out = []
    for value in group.strip().split(",")[:k]:
        try:
            out.append(float(value))          # float("NaN") -> nan
        except ValueError:
            out.append(np.nan)
    return out + [np.nan] * (k - len(out))

def _frame_id(line):
    return int(line[:line.index(b":")])

def seek_to_frame_id(handle, target_id):
    """Position a binary handle at (or shortly before) the first line whose frame_id >= target.

    The raw ATD is sorted by frame_id, so bisect on byte offsets down to ~1 MB, then let the
    caller skip the few remaining lines with a smaller id. Reads ~log2(filesize/1MB) lines
    instead of the whole prefix of the file.
    """
    handle.seek(0, os.SEEK_END)
    lo, hi = 0, handle.tell()
    while hi - lo > SEARCH_BLOCK:
        mid = (lo + hi) // 2
        handle.seek(mid - 1)
        handle.readline()                      # finish the line containing mid-1 -> line start
        line = handle.readline()
        if not line:                           # mid lies inside the last line
            hi = mid
        elif _frame_id(line) < target_id:
            lo = mid                           # everything up to here is before the target
        else:
            hi = mid
    if lo == 0:
        handle.seek(0)
    else:
        handle.seek(lo - 1)
        handle.readline()                      # realign to a line start at or after lo

def read_tracking(raw_path, first_frame, n, n_slots):
    """Keep only video frames [first_frame, first_frame + n) of the raw ATD.

    Returns players (n, n_slots, 3) = x, y (normalised 0-1), speed (m/s); ball (n, 2). NaN = missing.
    Raw line: "<frame_id>:<x,y,s>;<x,y,s>;...:<ball>", frame_id = video frame + 1.
    """
    players = np.full((n, n_slots, 3), np.nan, dtype=np.float32)
    ball = np.full((n, 2), np.nan, dtype=np.float32)
    lo, hi = first_frame + 1, first_frame + n          # inclusive ATD frame_id range
    with open(raw_path, "rb") as handle:
        seek_to_frame_id(handle, lo)
        for raw in handle:
            frame_id = _frame_id(raw)
            if frame_id < lo:
                continue
            if frame_id > hi:
                break
            parts = raw.decode("ascii", "replace").split(":")
            if len(parts) != 3:
                continue
            i = frame_id - lo
            for slot, group in enumerate(parts[1].split(";")[:n_slots]):
                players[i, slot] = _floats(group, 3)
            ball[i] = _floats(parts[2], 2)
    return players, ball

def load_homography_window(first_frame, n):
    """(matrices (n, 3, 3), valid (n,)) for the clip only; the file has one row per video frame."""
    import pandas as pd
    from starter.load_homography import MATRIX_COLUMNS
    path = require(["homography"])[0]
    table = pd.read_csv(path, skiprows=range(1, first_frame + 1), nrows=n)
    if len(table) == 0 or int(table["frame"].iloc[0]) != first_frame:
        raise ValueError("homography rows do not start at frame {}".format(first_frame))
    matrices = table[MATRIX_COLUMNS].values.astype(np.float64).reshape(-1, 3, 3)
    return matrices, table["valid"].values.astype(bool)

def load_gt_window(path, first_frame, n):
    """Optional public ground truth for the clip -> array of 'H','A','D','X' (or None)."""
    if not path or not os.path.isfile(path):
        return None
    import pandas as pd
    table = pd.read_csv(path, usecols=["frame", "label"], dtype={"label": "object"},
                        skiprows=range(1, first_frame + 1), nrows=n)
    if len(table) == 0 or int(table["frame"].iloc[0]) != first_frame:
        return None
    return table["label"].astype(str).str.strip().str.upper().values

def load_tags_window(enabled, first_frame, n):
    """Smart Tagging events overlapping the clip (the file is small; filter to keep lookups cheap)."""
    if not enabled:
        return []
    try:
        from starter import load_smart_tagging
        events = load_smart_tagging.load()
    except (IOError, ImportError) as error:
        print("note: Smart Tagging disabled ({})".format(str(error).splitlines()[0]))
        return []
    return [e for e in events if e["end_frame"] > first_frame and e["start_frame"] < first_frame + n]

def active_tags(events, frame, limit=3):
    short = {"barcelona": "H", "atletic club": "A", "athletic club": "A"}
    out = []
    for event in events:
        if event["start_frame"] <= frame < event["end_frame"]:
            team = short.get((event["tags"].get("Team") or "").strip().lower(), "N/A")
            kind = event["tags"].get("Type")
            out.append("{}{} [{}]".format((event["code"] or "").title(),
                                          " - " + kind if kind else "", team))
    return out[:limit]

def _runs(labels):
    """Number of maximal runs of 'H' or 'A' (possessions), dead ball and 'X' excluded."""
    count, prev = 0, None
    for v in labels:
        if v in ("H", "A") and v != prev:
            count += 1
        prev = v if v in ("H", "A") else None if v == "D" else prev
    return count

def find_ffmpeg():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None

class FfmpegFrames:
    """Stream n frames starting at start_frame, scaled to w x h, from an ffmpeg pipe.

    -ss before -i seeks through the mp4 index (no full read) and, because the video is
    re-encoded to rawvideo, ffmpeg decodes from the previous keyframe and discards up to the
    requested timestamp, so the seek is frame-accurate. Half a frame is subtracted so rounding
    in the timestamp can never skip the first frame. Decoding runs in the ffmpeg process,
    in parallel with the drawing done here.
    """

    def __init__(self, exe, path, start_frame, n, w, h, fps):
        self.w, self.h, self.size = w, h, w * h * 3
        ss = max(0.0, (start_frame - 0.5) / float(fps))
        cmd = [exe, "-v", "error", "-nostdin", "-ss", "{:.4f}".format(ss), "-i", path,
               "-map", "0:v:0", "-an", "-sn", "-frames:v", str(n),
               "-vf", "scale={}:{}:flags=area".format(w, h),
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
        self._err = tempfile.TemporaryFile()
        self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=self._err,
                                      bufsize=self.size * 4)

    def read(self):
        buf = bytearray(self.size)
        view, got = memoryview(buf), 0
        while got < self.size:
            k = self._proc.stdout.readinto(view[got:])
            if not k:
                return None
            got += k
        return np.frombuffer(buf, dtype=np.uint8).reshape(self.h, self.w, 3)   # writable

    def error(self):
        self._err.seek(0)
        return self._err.read().decode("utf-8", "replace").strip()

    def close(self):
        if self._proc.poll() is None:
            self._proc.kill()
        self._proc.wait()
        self._proc.stdout.close()
        self._err.close()

class OpenCVFrames:
    """Fallback: seek with OpenCV. Cheap to open, but seeking can be a few frames off."""

    def __init__(self, path, start_frame, w, h, exact):
        self.w, self.h = w, h
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            sys.exit("could not open {}".format(path))
        if exact:
            for _ in range(start_frame):
                if not self.cap.grab():
                    break
        elif start_frame:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            if int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) != start_frame:
                print("WARNING: seek landed elsewhere; use --exact-seek (slow) or install ffmpeg")

    def read(self):
        ok, img = self.cap.read()
        if not ok:
            return None
        return cv2.resize(img, (self.w, self.h), interpolation=cv2.INTER_AREA)

    def error(self):
        return ""

    def close(self):
        self.cap.release()

def open_video(args, start_frame, n):
    """(reader, source_width) -- opening only reads the mp4 header, never the 5 GB of frames."""
    path = require(["video"])[0]
    probe = cv2.VideoCapture(path)
    if not probe.isOpened():
        sys.exit("could not open {}".format(path))
    src_w = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
    total = int(probe.get(cv2.CAP_PROP_FRAME_COUNT))
    probe.release()
    if total and total != N_FRAMES:
        print("WARNING: video reports {} frames, expected {} - decode may differ from the "
              "reference".format(total, N_FRAMES))

    start_frame = max(0, start_frame + args.video_offset)
    exe = find_ffmpeg() if args.backend in ("auto", "ffmpeg") else None
    if args.backend == "ffmpeg" and not exe:
        sys.exit("--backend ffmpeg requested but no ffmpeg found (install it, or: pip install imageio-ffmpeg)")
    if exe:
        print("video backend: ffmpeg ({})".format(exe))
        return FfmpegFrames(exe, path, start_frame, n, PANEL_W, PANEL_H, FPS), src_w
    print("video backend: OpenCV seek (no ffmpeg found; frame accuracy not guaranteed)")
    return OpenCVFrames(path, start_frame, PANEL_W, PANEL_H, args.exact_seek), src_w

class PitchPanel:
    """Top-down pitch drawn with OpenCV (matplotlib is ~10x too slow for 7,500 frames)."""

    def __init__(self, w, h, flip_y=True, margin=30):
        self.flip_y = flip_y
        self.scale = min((w - 2 * margin) / PITCH_L, (h - 2 * margin) / PITCH_W)
        self.ox = (w - PITCH_L * self.scale) / 2.0
        self.oy = (h - PITCH_W * self.scale) / 2.0
        self.bg = self._background(w, h)

    def metres(self, xm, ym):
        """Pitch metres -> pixel (ints)."""
        y = (PITCH_W - ym) if self.flip_y else ym
        return int(round(self.ox + xm * self.scale)), int(round(self.oy + y * self.scale))

    def norm(self, x, y):
        """Normalised 0-1 -> pixel (float arrays)."""
        yy = (1.0 - y) if self.flip_y else y
        return self.ox + x * PITCH_L * self.scale, self.oy + yy * PITCH_W * self.scale

    def _background(self, w, h):
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:] = GRASS_BGR
        pt = self.metres
        rect = lambda a, b: cv2.rectangle(img, pt(*a), pt(*b), LINE_BGR, 1, cv2.LINE_AA)  # noqa: E731
        rect((0, 0), (PITCH_L, PITCH_W))
        cv2.line(img, pt(PITCH_L / 2, 0), pt(PITCH_L / 2, PITCH_W), LINE_BGR, 1, cv2.LINE_AA)
        cv2.circle(img, pt(PITCH_L / 2, PITCH_W / 2), int(9.15 * self.scale), LINE_BGR, 1, cv2.LINE_AA)
        cv2.circle(img, pt(PITCH_L / 2, PITCH_W / 2), 2, LINE_BGR, -1, cv2.LINE_AA)
        mid = PITCH_W / 2
        for left in (True, False):
            sx = (lambda d: d) if left else (lambda d: PITCH_L - d)
            rect((sx(0), mid - 20.15), (sx(16.5), mid + 20.15))      # penalty area
            rect((sx(0), mid - 9.16), (sx(5.5), mid + 9.16))         # six-yard box
            cv2.circle(img, pt(sx(11), mid), 2, LINE_BGR, -1, cv2.LINE_AA)  # penalty spot
            rect((sx(0), mid - 3.66), (sx(-1.5), mid + 3.66))        # goal
        return img

    def render(self, players, ball, team, trail, home_name, away_name, show_ids):
        """players (S, 3), ball (2,), trail (k, 2) of earlier ball positions.

        Returns (image, n_home, n_away, nearest_team) with nearest_team in {0, 1, -1}.
        """
        img = self.bg.copy()
        ok = ~np.isnan(players[:, 0]) & ~np.isnan(players[:, 1])
        px, py = self.norm(players[:, 0], players[:, 1])

        # Ball trail: connect consecutive valid points only, older = darker.
        for k in range(1, len(trail)):
            a, b = trail[k - 1], trail[k]
            if np.isnan(a).any() or np.isnan(b).any():
                continue
            shade = int(80 + 175 * k / len(trail))
            cv2.line(img, tuple(int(v) for v in np.round(self.norm(*a))),
                     tuple(int(v) for v in np.round(self.norm(*b))),
                     (shade, shade, 255), 2, cv2.LINE_AA)

        for slot in np.flatnonzero(ok):
            centre = (int(round(px[slot])), int(round(py[slot])))
            cv2.circle(img, centre, 7, HOME_BGR if team[slot] == 0 else AWAY_BGR, -1, cv2.LINE_AA)
            cv2.circle(img, centre, 7, (0, 0, 0), 1, cv2.LINE_AA)
            if show_ids:
                put(img, str(slot + 1), (centre[0] + 8, centre[1] - 6), 0.4)

        nearest = -1
        if not np.isnan(ball).any():
            bx, by = self.norm(ball[0], ball[1])
            if ok.any():
                d = ((players[ok, 0] - ball[0]) * PITCH_L) ** 2 + ((players[ok, 1] - ball[1]) * PITCH_W) ** 2
                j = np.flatnonzero(ok)[int(np.argmin(d))]
                nearest = int(team[j])
                cv2.line(img, (int(round(bx)), int(round(by))), (int(round(px[j])), int(round(py[j]))),
                         (0, 255, 255), 1, cv2.LINE_AA)
            cv2.circle(img, (int(round(bx)), int(round(by))), 6, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(img, (int(round(bx)), int(round(by))), 6, (0, 0, 0), 2, cv2.LINE_AA)
        else:
            put(img, "BALL NOT TRACKED", (int(self.ox) + 10, 24), 0.6, (80, 80, 255), 2)

        put(img, "H {} (red)".format(home_name), (12, PANEL_H - 34), 0.5, (200, 200, 255))
        put(img, "A {} (blue)".format(away_name), (12, PANEL_H - 14), 0.5, (255, 220, 200))
        n_home = int(np.sum(ok & (team == 0)))
        n_away = int(np.sum(ok & (team == 1)))
        return img, n_home, n_away, nearest

def _ring(img, p, color, r=12, thick=2):
    if p is not None:
        cv2.circle(img, p, r, color, thick, cv2.LINE_AA)

def _dashed(img, a, b, color, thick=2, dash=8):
    length = float(np.hypot(b[0] - a[0], b[1] - a[1]))
    steps = max(1, int(length // dash))
    for k in range(0, steps, 2):
        p = (int(a[0] + (b[0] - a[0]) * k / steps), int(a[1] + (b[1] - a[1]) * k / steps))
        q = (int(a[0] + (b[0] - a[0]) * min(steps, k + 1) / steps), int(a[1] + (b[1] - a[1]) * min(steps, k + 1) / steps))
        cv2.line(img, p, q, color, thick, cv2.LINE_AA)

def _cross(img, p, color, s=8, thick=2):
    cv2.line(img, (p[0] - s, p[1] - s), (p[0] + s, p[1] + s), color, thick, cv2.LINE_AA)
    cv2.line(img, (p[0] - s, p[1] + s), (p[0] + s, p[1] - s), color, thick, cv2.LINE_AA)
