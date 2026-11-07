"""Watch the possession test: tracking, zones of influence, events and possession on one video.

Runs the same analysis as test_scripts/possession_test_1.py on a short window (it imports it, so
the two can never disagree), then renders an mp4:

  Left panel   the tactical video with the tracking projected onto it through the homography:
               players, ball, each player's zone of influence, the ball holder and contests.
               With --no-video (or --left zoom) a zoomed top-down view that follows the ball.
  Right panel  the whole pitch from above: players with their zone of influence and a velocity
               arrow, the ball and its trail, the holder (yellow ring), contests (magenta ring),
               the current pass (green complete, orange intercepted) or carry, a possession
               badge and a log of the last events.
  Info bar     frame, video time, half; our label next to the public ground truth when the frame
               is inside the public 5 minutes; control state and holder; the latest event; the
               Smart Tagging phases active on the frame.
  Timeline     our possession band for the whole window, the public ground truth under it, ticks
               for passes, interceptions, contests and possession changes, and a cursor.

Needs numpy, pandas, opencv-python and ffmpeg (or imageio-ffmpeg) - the same as
visualize_tracking.py, whose video reader and homography loader it reuses. Run from the repo root:

    python test_scripts/visualize_events.py                          # 60 s from kickoff
    python test_scripts/visualize_events.py --start 07:40 --seconds 20
    python test_scripts/visualize_events.py --start-frame 13000 --seconds 30 --ids
    python test_scripts/visualize_events.py --no-video               # no mp4 needed
    python test_scripts/visualize_events.py --contest-hold 2         # same flags as the test

Writes reports/possession_test_1/events_<start>.mp4, and the same events_<start>.json and
frames_<start>.json as possession_test_1.py (skip them with --no-json).

The video and the JSON are derived from the match data: keep them out of the public repository
and do not upload them to any cloud AI tool (RULES.md). reports/possession_test_1/ is git-ignored.
"""

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _p in (ROOT, os.path.join(ROOT, "starter"), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import possession_test_1 as pt  # noqa: E402
import visualize_tracking as vt  # noqa: E402  (exits with a message if OpenCV is missing)
from games import GAME, half_of_video_frame  # noqa: E402

cv2 = vt.cv2

# Params
FPS = pt.FPS
PITCH_L, PITCH_W = pt.PITCH_L, pt.PITCH_W
PANEL_W, PANEL_H = vt.PANEL_W, vt.PANEL_H          # 960 x 540 each
INFO_H, TIMELINE_H = 92, 64
HOME_BGR, AWAY_BGR = vt.HOME_BGR, vt.AWAY_BGR
DEAD_BGR = (177, 165, 154)
HOLDER_BGR = (0, 230, 255)           # yellow
CONTEST_BGR = (220, 60, 220)         # magenta
PASS_OK_BGR = (90, 220, 90)          # green
PASS_LOST_BGR = (0, 140, 255)        # orange
CHANGE_BGR = (0, 255, 255)
ZOOM_SPAN_M = 36.0                   # width of the zoomed view, metres
EVENT_LINGER_S = 1.5                 # how long a pass / change stays drawn after it ends
LOG_LINES = 5
ZOI_POINTS = 28
SHOW_IN_LOG = ("pass", "dispossessed", "possession_change", "change_cancelled", "contest",
               "dead_ball", "carry")
TEAM_BGR = {0: HOME_BGR, 1: AWAY_BGR}
LABEL_BGR = {"H": HOME_BGR, "A": AWAY_BGR, "D": DEAD_BGR, "X": (90, 90, 90)}

def put(img, text, org, scale=0.55, color=(255, 255, 255), thick=1):
    """Text with a 1-pixel black outline drawn by offsets, so glyphs keep the same width in every
    OpenCV version (a thicker outline stroke changes glyph widths in OpenCV 5)."""
    x, y = org
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)):
        cv2.putText(img, text, (x + dx, y + dy), vt.FONT, scale, (0, 0, 0), thick, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), vt.FONT, scale, color, thick, cv2.LINE_AA)


# -- geometry ----------------------------------------------------------------------------------

def zoi_polygon(x, y, ux, uy, front, rear, side):
    """Zone of influence outline in metres: front radius ahead, rear behind, side across."""
    t = np.linspace(0.0, 2.0 * np.pi, ZOI_POINTS, endpoint=False)
    c, s = np.cos(t), np.sin(t)
    along = np.where(c >= 0, front, rear) * c
    lateral = side * s
    return x + along * ux - lateral * uy, y + along * uy + lateral * ux


class View:
    """Metres -> pixels for a top-down view. scale px/m, (x0, y0) = metres at the left/top edge."""

    def __init__(self, w, h, scale, x0, y0, flip_y=True):
        self.w, self.h, self.scale, self.x0, self.y0, self.flip_y = w, h, scale, x0, y0, flip_y

    @classmethod
    def whole_pitch(cls, w, h, flip_y=True, margin=30):
        scale = min((w - 2 * margin) / PITCH_L, (h - 2 * margin) / PITCH_W)
        x0 = -(w / scale - PITCH_L) / 2.0
        y0 = -(h / scale - PITCH_W) / 2.0
        return cls(w, h, scale, x0, y0, flip_y)

    @classmethod
    def zoom(cls, w, h, cx, cy, span, flip_y=True):
        scale = w / span
        return cls(w, h, scale, cx - span / 2.0, cy - (h / scale) / 2.0, flip_y)

    def px(self, xm, ym):
        xm, ym = np.asarray(xm, dtype=float), np.asarray(ym, dtype=float)
        u = (xm - self.x0) * self.scale
        if self.flip_y:
            v = self.h - (ym - self.y0) * self.scale
        else:
            v = (ym - self.y0) * self.scale
        return u, v

    def pt(self, xm, ym):
        u, v = self.px(xm, ym)
        return int(round(float(u))), int(round(float(v)))

    def poly(self, xs, ys):
        u, v = self.px(xs, ys)
        return np.stack([u, v], axis=1).round().astype(np.int32)


def draw_pitch(img, view):
    """Grass and markings, in any view."""
    img[:] = vt.GRASS_BGR
    line = vt.LINE_BGR

    def rect(a, b):
        cv2.rectangle(img, view.pt(*a), view.pt(*b), line, 1, cv2.LINE_AA)

    rect((0, 0), (PITCH_L, PITCH_W))
    cv2.line(img, view.pt(PITCH_L / 2, 0), view.pt(PITCH_L / 2, PITCH_W), line, 1, cv2.LINE_AA)
    cv2.circle(img, view.pt(PITCH_L / 2, PITCH_W / 2), int(9.15 * view.scale), line, 1,
               cv2.LINE_AA)
    mid = PITCH_W / 2
    for left in (True, False):
        sx = (lambda d: d) if left else (lambda d: PITCH_L - d)
        rect((sx(0), mid - 20.15), (sx(16.5), mid + 20.15))
        rect((sx(0), mid - 9.16), (sx(5.5), mid + 9.16))
        rect((sx(0), mid - 3.66), (sx(-1.5), mid + 3.66))
        cv2.circle(img, view.pt(sx(11), mid), 2, line, -1, cv2.LINE_AA)
    return img


def arrow(img, a, b, color, thick=2):
    cv2.arrowedLine(img, a, b, color, thick, cv2.LINE_AA, tipLength=0.12)


# -- event text --------------------------------------------------------------------------------

def event_text(e, names):
    team = lambda L: "{} ({})".format(L, names.get(L, L)) if L else "?"  # noqa: E731
    t = e["type"]
    if t == "pass":
        return "PASS {} #{} -> {}#{}  {} m  {}".format(
            e["team"], e["player"], "" if e["outcome"] == "complete" else e["recipient_team"],
            e["recipient"], e["length_m"], e["outcome"].upper())
    if t == "carry":
        return "CARRY {} #{}  {} m in {} s".format(e["team"], e["player"], e["distance_m"],
                                                   e["duration_s"])
    if t == "hold":
        return "HOLD {} #{}  {} s".format(e["team"], e["player"], e["duration_s"])
    if t == "dispossessed":
        return "DISPOSSESSED {} #{} by {} #{}".format(e["team"], e["player"],
                                                      e["recipient_team"], e["recipient"])
    if t == "contest":
        return "CONTEST H#{} v A#{}  {} s  won by {}".format(
            e["home_player"], e["away_player"], e["duration_s"], e["won_by"] or "-")
    if t == "possession_change":
        return "POSSESSION {} -> {}  ({}, confirmed after {} s)".format(
            e["team_from"], team(e["team_to"]), e["kind"], e["decision_delay_s"])
    if t == "change_cancelled":
        return "CHANGE CANCELLED {} kept it ({})".format(e["team_from"], e["kind"])
    if t == "dead_ball":
        return "DEAD BALL ({})  {} s  restart {}".format(e["reason"], e["duration_s"],
                                                         e["restart_team"] or "-")
    if t == "ball_not_tracked":
        return "BALL NOT TRACKED {} s".format(e["duration_s"])
    return t


def event_frames(e):
    start = e.get("frame_start", e.get("frame"))
    end = e.get("frame_end", e.get("decided_frame", start))
    return start, end


# -- timeline ----------------------------------------------------------------------------------

def build_timeline(labels_pred, gt, events, w, lo, n):
    """Static strip: our labels, the public truth, event ticks. A cursor is added per frame."""
    img = np.full((TIMELINE_H, w, 3), 28, dtype=np.uint8)
    x0, x1 = 110, w - 20
    xs = lambda k: int(x0 + (x1 - x0) * k / max(1, n - 1))  # noqa: E731

    def band(letters, y, h):
        k = 0
        while k < n:
            j = k
            while j < n and letters[j] == letters[k]:
                j += 1
            cv2.rectangle(img, (xs(k), y), (max(xs(k) + 1, xs(j - 1)), y + h),
                          LABEL_BGR.get(letters[k], (90, 90, 90)), -1)
            k = j

    put(img, "ours", (10, 20), 0.45)
    band(labels_pred, 8, 16)
    if gt is not None:
        put(img, "public GT", (10, 42), 0.45)
        band(gt, 30, 16)
    for e in events:
        f, _ = event_frames(e)
        k = f - lo
        if not 0 <= k < n:
            continue
        x = xs(k)
        if e["type"] == "possession_change":
            cv2.drawMarker(img, (x, 56), CHANGE_BGR, cv2.MARKER_TRIANGLE_UP, 9, 2)
        elif e["type"] == "pass":
            color = PASS_OK_BGR if e["outcome"] == "complete" else PASS_LOST_BGR
            cv2.line(img, (x, 50), (x, 60), color, 1)
        elif e["type"] == "contest":
            cv2.circle(img, (x, 56), 3, CONTEST_BGR, -1)
    return img, xs


# -- main --------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--start", default=None, help="start, video time hh:mm:ss / mm:ss / seconds")
    ap.add_argument("--start-frame", type=int, default=None,
                    help="start as a video frame (default 11185, the 1st-half kickoff)")
    ap.add_argument("--seconds", type=float, default=60.0, help="length (default 60, max 300)")
    ap.add_argument("--out", help="output mp4 (default reports/possession_test_1/events_<f>.mp4)")
    ap.add_argument("--left", choices=("video", "zoom"), default="video",
                    help="left panel: the tactical video with overlay, or a zoomed pitch view")
    ap.add_argument("--no-video", action="store_true", help="no mp4 decode (implies --left zoom)")
    ap.add_argument("--no-zones", action="store_true", help="do not draw zones of influence")
    ap.add_argument("--ids", action="store_true", help="label players with their track slot")
    ap.add_argument("--trail", type=int, default=40, help="ball trail in frames (0 = off)")
    ap.add_argument("--no-tags", action="store_true", help="do not show Smart Tagging phases")
    ap.add_argument("--no-json", action="store_true", help="do not write the JSON files")
    ap.add_argument("--gt", default=vt.DEFAULT_GT, help="public ground truth CSV ('' = off)")
    ap.add_argument("--no-flip-y", dest="flip_y", action="store_false")
    ap.add_argument("--backend", choices=("auto", "ffmpeg", "opencv"), default="auto")
    ap.add_argument("--exact-seek", action="store_true")
    ap.add_argument("--video-offset", type=int, default=0)
    pt.add_settings_args(ap)
    args = ap.parse_args()

    if args.start_frame is not None:
        lo = args.start_frame
    elif args.start is not None:
        lo = int(round(vt.parse_time(args.start) * FPS))
    else:
        lo = pt.WINDOW_START
    seconds = min(args.seconds, 300.0)
    hi = lo + int(round(seconds * FPS)) - 1
    if args.no_video:
        args.left = "zoom"

    t0 = time.time()
    try:
        res = pt.analyse_window(lo, hi, args)
    except ValueError as error:
        sys.exit(str(error))
    hi, n = res["hi"], res["n"]
    events = res["events"]
    names = {"H": res["team_names"][0] if res["team_names"] else "home",
             "A": res["team_names"][1] if len(res["team_names"]) > 1 else "away"}
    print("window {}..{} ({:.0f} s, {} to {}): analysed in {:.1f} s, {} events".format(
        lo, hi, n / float(FPS), pt.clock(lo), pt.clock(hi), time.time() - t0, len(events)))

    out_dir = pt.REPORT_DIR
    os.makedirs(out_dir, exist_ok=True)
    if not args.no_json:
        json_path = os.path.join(out_dir, "events_{}.json".format(lo))
        pt.write_json(res, args, json_path, detail=n / float(FPS) <= pt.DETAIL_MAX_S)
        print("wrote {}".format(json_path))

    labels = np.array(list("HAD"))[res["labels"]]
    gt = vt.load_gt_window(args.gt, lo, n) if args.gt else None
    tags = vt.load_tags_window(not args.no_tags, lo, n)
    team = res["team_of_slot"]
    ux, uy, front, rear, side = pt.zoi_geometry(res["vx"], res["vy"], args)
    spx, spy, sbx, sby = res["spx"], res["spy"], res["sbx"], res["sby"]
    holder, ctrl = res["holder"], res["ctrl"]

    # Video and homography.
    reader, src_w, H, H_valid, project = None, 1920, None, None, None
    if args.left == "video":
        try:
            from starter.load_homography import project
            H, H_valid = vt.load_homography_window(lo, n)
        except (IOError, ImportError, ValueError) as error:
            print("note: no homography overlay ({})".format(str(error).splitlines()[0]))
        reader, src_w = vt.open_video(args, lo, n)
    vscale = PANEL_W / float(src_w)

    whole = View.whole_pitch(PANEL_W, PANEL_H, args.flip_y)
    pitch_bg = draw_pitch(np.zeros((PANEL_H, PANEL_W, 3), np.uint8), whole)
    width = 2 * PANEL_W
    height = PANEL_H + INFO_H + TIMELINE_H
    timeline, xs = build_timeline(labels, gt, events, width, lo, n)

    out_path = args.out or os.path.join(out_dir, "events_{}.mp4".format(lo))
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (width, height))
    if not writer.isOpened():
        sys.exit("could not open the video writer for {}".format(out_path))

    def project_m(i, xm, ym):
        """Pitch metres -> video panel pixels for window frame i (None if no homography)."""
        if H is None or i >= len(H_valid) or not H_valid[i]:
            return None
        u, v = project(H[i], np.asarray(xm) / PITCH_L, np.asarray(ym) / PITCH_W)
        return np.asarray(u) * vscale, np.asarray(v) * vscale

    def active(f, kinds):
        out = []
        linger = int(EVENT_LINGER_S * FPS)
        for e in events:
            a, b = event_frames(e)
            if e["type"] in kinds and a <= f <= b + linger:
                out.append(e)
        return out

    def draw_players(img, i, to_px, zone_scale_px, show_zone, with_ids):
        """Zones, velocity arrows, players, holder and contest markers in one view."""
        seen = np.flatnonzero(~np.isnan(spx[i]))
        layer = img.copy()
        for s in seen:
            if show_zone:
                xs_, ys_ = zoi_polygon(spx[i, s], spy[i, s], ux[i, s], uy[i, s],
                                       front[i, s], rear[i, s], side[i, s])
                pts = to_px(xs_, ys_)
                if pts is None:
                    continue
                pts = np.stack(pts, axis=1).round().astype(np.int32)
                inside = res["q"][i, s] <= 1.0
                color = TEAM_BGR[int(team[s])]
                if inside:
                    cv2.fillPoly(layer, [pts], color)
                cv2.polylines(img, [pts], True, color, 1, cv2.LINE_AA)
        cv2.addWeighted(layer, 0.35, img, 0.65, 0, dst=img)
        for s in seen:
            p = to_px(spx[i, s], spy[i, s])
            if p is None:
                continue
            c = (int(round(float(p[0]))), int(round(float(p[1]))))
            color = TEAM_BGR[int(team[s])]
            if np.isfinite(res["vx"][i, s]):
                q = to_px(spx[i, s] + 0.5 * res["vx"][i, s], spy[i, s] + 0.5 * res["vy"][i, s])
                if q is not None:
                    cv2.line(img, c, (int(round(float(q[0]))), int(round(float(q[1])))),
                             (240, 240, 240), 1, cv2.LINE_AA)
            r = max(4, int(zone_scale_px * 0.45))
            cv2.circle(img, c, r, color, -1, cv2.LINE_AA)
            cv2.circle(img, c, r, (0, 0, 0), 1, cv2.LINE_AA)
            if s == holder[i]:
                cv2.circle(img, c, r + 5, HOLDER_BGR, 2, cv2.LINE_AA)
            if ctrl[i] == pt.BOTH and s in (res["contest_h"][i], res["contest_a"][i]):
                cv2.circle(img, c, r + 5, CONTEST_BGR, 2, cv2.LINE_AA)
            if with_ids:
                put(img, str(s + 1), (c[0] + r + 2, c[1] - r), 0.4)

    def draw_ball(img, i, to_px, trail):
        if trail > 0:
            prev = None
            for k in range(max(0, i - trail), i + 1):
                p = to_px(sbx[k], sby[k]) if np.isfinite(sbx[k]) else None
                p = (int(round(float(p[0]))), int(round(float(p[1])))) if p is not None else None
                if p is not None and prev is not None:
                    shade = int(80 + 175 * (k - i + trail) / float(trail))
                    cv2.line(img, prev, p, (shade, shade, 255), 2, cv2.LINE_AA)
                prev = p
        if np.isfinite(sbx[i]):
            p = to_px(sbx[i], sby[i])
            if p is not None:
                c = (int(round(float(p[0]))), int(round(float(p[1]))))
                cv2.circle(img, c, 6, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(img, c, 6, (0, 0, 0), 2, cv2.LINE_AA)
                if ctrl[i] == pt.BOTH:
                    cv2.circle(img, c, 14, CONTEST_BGR, 2, cv2.LINE_AA)
            return True
        return False

    def draw_actions(img, f, i, to_px):
        """The current pass (arrow) or carry (path so far)."""
        for e in active(f, ("pass", "dispossessed")):
            a = to_px(e["start"]["x"], e["start"]["y"]) if e["start"]["x"] is not None else None
            b = to_px(e["end"]["x"], e["end"]["y"]) if e["end"]["x"] is not None else None
            if a is None or b is None:
                continue
            color = PASS_OK_BGR if e.get("outcome") == "complete" else PASS_LOST_BGR
            arrow(img, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), color, 2)
        for e in active(f, ("carry",)):
            k0 = e["frame_start"] - lo
            k1 = min(i, e["frame_end"] - lo)
            pts = [to_px(sbx[k], sby[k]) for k in range(k0, k1 + 1) if np.isfinite(sbx[k])]
            pts = [(int(p[0]), int(p[1])) for p in pts if p is not None]
            if len(pts) > 1:
                color = TEAM_BGR[0 if e["team"] == "H" else 1]
                cv2.polylines(img, [np.array(pts, np.int32)], False, color, 3, cv2.LINE_AA)

    rendered = 0
    last_center = (PITCH_L / 2, PITCH_W / 2)
    t1 = time.time()
    try:
        for i in range(n):
            f = lo + i
            canvas = np.zeros((height, width, 3), dtype=np.uint8)

            # Left panel.
            if args.left == "video":
                img = reader.read()
                if img is None:
                    print("video ended at frame {} {}".format(f, reader.error()))
                    break
                to_video = lambda xm, ym: project_m(i, xm, ym)  # noqa: E731
                if H is not None:
                    draw_players(img, i, to_video, 10, not args.no_zones, args.ids)
                    draw_ball(img, i, to_video, args.trail)
                    draw_actions(img, f, i, to_video)
                else:
                    put(img, "no homography for this frame", (20, 30))
            else:
                if np.isfinite(sbx[i]):
                    last_center = (float(sbx[i]), float(sby[i]))
                view = View.zoom(PANEL_W, PANEL_H, last_center[0], last_center[1],
                                 ZOOM_SPAN_M, args.flip_y)
                img = draw_pitch(np.zeros((PANEL_H, PANEL_W, 3), np.uint8), view)
                draw_players(img, i, view.px, view.scale, not args.no_zones, True)
                if not draw_ball(img, i, view.px, args.trail):
                    put(img, "BALL NOT TRACKED - view held at last position", (20, 30), 0.6,
                        (80, 80, 255), 2)
                draw_actions(img, f, i, view.px)
                put(img, "zoom {:.0f} m".format(ZOOM_SPAN_M), (PANEL_W - 110, 24), 0.5)
            canvas[:PANEL_H, :PANEL_W] = img

            # Right panel: whole pitch.
            img = pitch_bg.copy()
            draw_players(img, i, whole.px, whole.scale * 1.5, not args.no_zones, args.ids)
            tracked = draw_ball(img, i, whole.px, args.trail)
            draw_actions(img, f, i, whole.px)
            if not tracked:
                put(img, "BALL NOT TRACKED", (int(-whole.x0 * whole.scale) + 10, 24), 0.6,
                    (80, 80, 255), 2)
            ours = labels[i]
            cv2.rectangle(img, (PANEL_W - 250, 8), (PANEL_W - 10, 40), LABEL_BGR[ours], -1)
            put(img, "POSSESSION {}".format(
                "D (dead ball)" if ours == "D" else "{} {}".format(ours, names[ours])[:22]),
                (PANEL_W - 242, 30), 0.55)
            shown = [e for e in events if e["type"] in SHOW_IN_LOG
                     and event_frames(e)[0] <= f][-LOG_LINES:]
            for j, e in enumerate(shown):
                age = (f - event_frames(e)[0]) / float(FPS)
                color = (255, 255, 255) if age < EVENT_LINGER_S else (190, 190, 190)
                put(img, "{} {}".format(e.get("time_start", e.get("time", "")),
                                        event_text(e, names))[:70],
                    (12, PANEL_H - 16 - 18 * (len(shown) - 1 - j)), 0.45, color)
            canvas[:PANEL_H, PANEL_W:] = img

            # Info bar.
            y = PANEL_H
            half = half_of_video_frame(GAME, f)
            c = int(ctrl[i])
            hold = int(holder[i])
            put(canvas, "frame {}   video {}   half {}".format(f, pt.clock(f), half or "-"),
                (12, y + 22), 0.6)
            put(canvas, "ours: {}".format(ours), (420, y + 22), 0.6, (255, 255, 255))
            cv2.rectangle(canvas, (500, y + 8), (520, y + 26), LABEL_BGR[ours], -1)
            if gt is not None and i < len(gt) and gt[i] != "X":
                put(canvas, "public GT: {}".format(gt[i]), (540, y + 22), 0.6)
                cv2.rectangle(canvas, (680, y + 8), (700, y + 26), LABEL_BGR[gt[i]], -1)
            put(canvas, "control: {}{}{}".format(
                pt.CTRL_NAME[c], "  holder #{} ({})".format(hold + 1, "HA"[team[hold]])
                if hold >= 0 else "", "  DEAD ({})".format(
                    pt.DEAD_REASONS[int(res["dead_reason"][i])] or "-") if res["dead"][i] else ""),
                (PANEL_W + 12, y + 22), 0.6, HOLDER_BGR if hold >= 0 else (255, 255, 255))
            now = [e for e in active(f, SHOW_IN_LOG)]
            if now:
                put(canvas, event_text(now[-1], names)[:110], (12, y + 52), 0.6, CHANGE_BGR
                    if now[-1]["type"] == "possession_change" else (255, 255, 255))
            phases = vt.active_tags(tags, f)
            if phases:
                put(canvas, "Smart Tagging: " + " | ".join(phases)[:120], (12, y + 80), 0.5,
                    (200, 255, 200))

            # Timeline with cursor.
            strip = timeline.copy()
            cv2.line(strip, (xs(i), 2), (xs(i), TIMELINE_H - 2), (255, 255, 255), 2)
            canvas[PANEL_H + INFO_H:] = strip

            writer.write(canvas)
            rendered += 1
            if i % 250 == 0:
                print("  {}/{} frames ({:.1f} fps)".format(i, n,
                                                           rendered / max(1e-6, time.time() - t1)))
    finally:
        writer.release()
        if reader is not None:
            reader.close()
    print("done: {} frames -> {}".format(rendered, os.path.abspath(out_path)))


if __name__ == "__main__":
    main()
