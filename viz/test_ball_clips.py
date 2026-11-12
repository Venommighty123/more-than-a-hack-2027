"""Ball-tracking test clips: where the ball is still missing after cleaning, and where the cleaning changed it.

    python viz/test_ball_clips.py --dry-run              # list the clips, write index.csv, render nothing
    python viz/test_ball_clips.py                        # render every clip (+-75 frames)
    python viz/test_ball_clips.py --types corrected      # only the frames the pipeline corrected
    python viz/test_ball_clips.py --types not_tracked --min-gap 125 --max-clips 10
    python viz/test_ball_clips.py --pipeline prepared    # test the v5 prepared ball instead
    python viz/test_ball_clips.py --no-video             # pitch panel only (fast, no mp4 decode)
    python viz/test_ball_clips.py --only fix_0061740     # one clip by name (see index.csv)

WHAT IS TESTED. The ball goes through the same clean-up as the labeller (``--pipeline``):

  v6        (default, the current submission) positions smoothed over +/- 3 frames, then
            preprocess/ball.clean_ball with v4's settings: spikes removed (jump in AND out
            > 2 m/frame), interior gaps <= 6 frames bridged linearly
  v4        clean_ball with v4's settings, no smoothing (v1-v4)
  prepared  preprocess/reconstruct.prepare_ball (v5): spikes out, gaps <= 25 frames Hermite-filled
            unless the ends imply > 35 m/s

Two kinds of clip, graded halves only (the rest of the video is never scored):

  not_tracked/  the ball is still missing AFTER cleaning. A gap of up to 2 x pad frames is one clip
                (gap_<start>_<len>.mp4, gap +- pad). A longer gap gives two clips: the moment the
                ball is lost (lost_<start>_<len>.mp4, start +- pad) and the moment it is found
                again (found_<end>_<len>.mp4, end +- pad); the middle is the same empty pitch.
  corrected/    frames the cleaning changed: a raw detection REMOVED (spike / teleport) or a
                position FILLED in (bridged / reconstructed). Corrections closer than 2 x pad
                frames share one clip (fix_<first>_<last>_r<removed>_f<filled>.mp4, +- pad).

Each clip: left the tactical video with the tracking projected (raw ball: white dot; removed
detection: red x; filled position: green ring), right the top-down pitch with the cleaned ball,
an info bar with the ball state of the frame, and a strip with the state of every frame of the clip
(white tracked, green filled, red removed, dark grey missing, yellow cursor, the event outlined).

Output (default outputs/test_ball/, git-ignored: clips are derived match data and must never be
committed or uploaded, RULES.md): not_tracked/, corrected/, index.csv (one row per clip),
summary.json (counts). Existing clips are skipped unless --overwrite.

RULES.md: watching the clips to understand and debug the ball pipeline is fine. Turning what you see
into labels, a frame list, or a constant tuned for this match is not: change a threshold only with
a bench score (docs/changelog/labeller.md).
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import possession.io  # noqa: E402,F401  (repository root on sys.path)
from games import GAME, GAMES, half_of_video_frame  # noqa: E402
from possession.io import graded_blocks, load_ball  # noqa: E402
from possession.preprocess.ball import runs  # noqa: E402

FPS = GAMES[GAME]["fps"]
PITCH = tuple(GAMES[GAME]["pitch"])          # load_ball scales to this; divide by it to normalise
OUT_DIR = os.path.join(ROOT, "outputs", "test_ball")

# per-frame ball state after cleaning
MISSING, TRACKED, FILLED, REMOVED = 0, 1, 2, 3
STATE_NAME = {MISSING: "NOT TRACKED", TRACKED: "tracked", FILLED: "FILLED (pipeline position)",
              REMOVED: "REMOVED (raw detection dropped)"}
STATE_BGR = {MISSING: (60, 60, 60), TRACKED: (235, 235, 235), FILLED: (80, 220, 80),
             REMOVED: (60, 60, 255)}


# ------------------------------------------------------------------------------------------------
# 1. The ball pipeline under test (no video, no OpenCV: importable by tests)
# ------------------------------------------------------------------------------------------------

def run_pipeline(bx, by, pipeline="v6"):
    """Clean the full-match ball the way the labeller does.

    Returns dict(x, y, observed, removed, filled) over all frames, metres; x/y NaN = missing after
    cleaning, observed = the tracker gave a position, removed = that position was dropped,
    filled = the pipeline wrote a position where there was none (or a dropped one).
    """
    bx = np.asarray(bx, dtype=np.float64)
    by = np.asarray(by, dtype=np.float64)
    observed = ~(np.isnan(bx) | np.isnan(by))
    if pipeline in ("v4", "v6"):
        from possession.labellers import v4, v6
        from possession.preprocess.ball import clean_ball
        p = dict(v6.DEFAULTS if pipeline == "v6" else v4.DEFAULTS)
        ball = np.stack([bx, by], axis=1)
        if pipeline == "v6" and p["SMOOTH_K"]:
            ball = v6.smooth(ball, int(p["SMOOTH_K"]))
        t = clean_ball(ball[:, 0], ball[:, 1], v4.ball_config(p), fps=FPS, pitch=PITCH)
        return dict(x=t.x, y=t.y, observed=observed, removed=t.removed & observed,
                    filled=t.bridged)
    if pipeline == "prepared":
        from possession.preprocess.reconstruct import prepare_ball
        x, y, _, removed, filled = prepare_ball(bx, by, fps=FPS)
        return dict(x=x, y=y, observed=observed, removed=removed & observed, filled=filled)
    raise ValueError("unknown pipeline {!r}".format(pipeline))


def frame_states(res):
    """(n,) int8 state per frame: REMOVED > FILLED > TRACKED > MISSING."""
    present = ~(np.isnan(res["x"]) | np.isnan(res["y"]))
    st = np.where(present, TRACKED, MISSING).astype(np.int8)
    st[res["filled"]] = FILLED
    st[res["removed"]] = REMOVED
    return st


# ------------------------------------------------------------------------------------------------
# 2. Which clips to make
# ------------------------------------------------------------------------------------------------

def plan_clips(res, blocks, pad=75, min_gap=1, types=("not_tracked", "corrected")):
    """List of clip dicts (kind, name, event_lo, event_hi (inclusive), clip_lo, clip_hi, half, ...)."""
    n = len(res["x"])
    graded = np.zeros(n, dtype=bool)
    half_of = np.zeros(n, dtype=np.int8)
    for half, a, b in blocks:
        graded[a:b + 1] = True
        half_of[a:b + 1] = half
    present = ~(np.isnan(res["x"]) | np.isnan(res["y"]))
    span = {int(h): (int(a), int(b)) for h, a, b in blocks}
    clips = []

    def clip(kind, name, ev_lo, ev_hi, lo, hi, **extra):
        h = int(half_of[ev_lo])
        blo, bhi = span[h]
        d = dict(kind=kind, name=name, half=h, event_lo=int(ev_lo), event_hi=int(ev_hi),
                 clip_lo=int(max(blo, lo)), clip_hi=int(min(bhi, hi)))
        d.update(extra)
        clips.append(d)

    if "not_tracked" in types:
        for s, e in zip(*runs(~present & graded)):
            length = int(e - s)
            if length < min_gap:
                continue
            blo, bhi = span[int(half_of[s])]
            censored = "start" if s == blo else ("end" if e - 1 == bhi else "")
            extra = dict(gap_len=length, gap_s=round(length / FPS, 2), censored=censored)
            if length <= 2 * pad:
                clip("not_tracked", "gap_{:07d}_{:05d}".format(s, length), s, e - 1,
                     s - pad, e - 1 + pad, part="whole", **extra)
            else:
                if censored != "start":
                    clip("not_tracked", "lost_{:07d}_{:05d}".format(s, length), s, s,
                         s - pad, s + pad, part="lost", **extra)
                if censored != "end":
                    clip("not_tracked", "found_{:07d}_{:05d}".format(e, length), e - 1, e - 1,
                         e - pad, e + pad, part="found", **extra)

    if "corrected" in types:
        changed = (res["removed"] | res["filled"]) & graded
        events = [(int(s), int(e)) for s, e in zip(*runs(changed))]
        groups = []
        for s, e in events:                          # merge corrections whose windows overlap
            if groups and s - pad <= groups[-1][1] - 1 + pad and half_of[s] == half_of[groups[-1][0]]:
                groups[-1][1] = e
            else:
                groups.append([s, e])
        for s, e in groups:
            r = int(res["removed"][s:e].sum())
            f = int(res["filled"][s:e].sum())
            clip("corrected", "fix_{:07d}_{:07d}_r{}_f{}".format(s, e - 1, r, f), s, e - 1,
                 s - pad, e - 1 + pad, removed=r, filled=f,
                 n_corrections=sum(1 for a, b in events if s <= a < e))
    clips.sort(key=lambda c: (c["kind"], c["clip_lo"]))
    return clips


def summarize(res, blocks, clips, pipeline):
    n = len(res["x"])
    graded = np.zeros(n, dtype=bool)
    for _, a, b in blocks:
        graded[a:b + 1] = True
    present = ~(np.isnan(res["x"]) | np.isnan(res["y"]))
    s, e = runs(~present & graded)
    lens = (e - s) / FPS
    return {
        "pipeline": pipeline, "graded_frames": int(graded.sum()),
        "raw_tracked_pct": round(100 * float((res["observed"] & graded).sum()) / graded.sum(), 2),
        "after_cleaning_pct": round(100 * float((present & graded).sum()) / graded.sum(), 2),
        "removed_frames": int((res["removed"] & graded).sum()),
        "filled_frames": int((res["filled"] & graded).sum()),
        "missing_runs": int(len(s)),
        "missing_runs_by_length_s": {"<1": int((lens < 1).sum()), "1-5": int(((lens >= 1) & (lens < 5)).sum()),
                                     "5-20": int(((lens >= 5) & (lens < 20)).sum()),
                                     ">=20": int((lens >= 20).sum())},
        "clips": {k: sum(1 for c in clips if c["kind"] == k) for k in ("not_tracked", "corrected")},
    }


# ------------------------------------------------------------------------------------------------
# 3. Rendering
# ------------------------------------------------------------------------------------------------

def render_clip(c, res, states, args, ctx, out_path):
    """One mp4: video + projected tracking | pitch, info bar, state strip."""
    from viz.common import (INFO_H, PANEL_H, PANEL_W, TL_H, PitchPanel, _cross, cv2, fmt_time,
                            load_homography_window, open_video, put, read_tracking)
    f0, n = c["clip_lo"], c["clip_hi"] - c["clip_lo"] + 1
    team, home_name, away_name, raw_path = ctx["team"], ctx["home"], ctx["away"], ctx["raw"]
    players, raw_ball = read_tracking(raw_path, f0, n, len(team))      # normalised 0-1
    clean = np.stack([res["x"][f0:f0 + n] / PITCH[0], res["y"][f0:f0 + n] / PITCH[1]], axis=1)
    st = states[f0:f0 + n]
    removed = res["removed"][f0:f0 + n]

    H = H_valid = project = None
    reader, src_w = None, 1920
    if not args.no_video:
        try:
            from starter.load_homography import project
            H, H_valid = load_homography_window(f0, n)
        except (IOError, ImportError, ValueError) as error:
            print("    note: no homography overlay ({})".format(str(error).splitlines()[0]))
        reader, src_w = open_video(args, f0, n)
    scale = PANEL_W / float(src_w)
    panel = PitchPanel(PANEL_W, PANEL_H)
    size = (2 * PANEL_W, PANEL_H + INFO_H + TL_H)

    # static state strip
    strip = np.zeros((TL_H, size[0], 3), dtype=np.uint8)
    x_of = lambda i: int(10 + (size[0] - 20) * i / max(1, n - 1))  # noqa: E731
    for i in range(n):
        cv2.rectangle(strip, (x_of(i), 18), (x_of(i + 1) if i + 1 < n else x_of(i) + 1, 44),
                      STATE_BGR[int(st[i])], -1)
    a, b = c["event_lo"] - f0, c["event_hi"] - f0
    if 0 <= a < n:
        cv2.rectangle(strip, (x_of(a) - 1, 14), (x_of(min(b, n - 1)) + 1, 48), (0, 215, 255), 2)
    put(strip, "white tracked   green filled   red removed   grey NOT TRACKED   yellow box = event",
        (10, 64), 0.45, (210, 210, 210))
    put(strip, "frame {}".format(f0), (10, 12), 0.4, (180, 180, 180))
    put(strip, "frame {}".format(f0 + n - 1), (size[0] - 110, 12), 0.4, (180, 180, 180))

    title = {"not_tracked": "BALL NOT TRACKED AFTER CLEANING", "corrected": "CORRECTED BY THE PIPELINE"}[c["kind"]]
    if c["kind"] == "not_tracked":
        detail = "gap {} frames ({:.1f} s), {}{}".format(
            c["gap_len"], c["gap_s"], {"whole": "whole gap", "lost": "ball LOST at the yellow box",
                                       "found": "ball FOUND after the yellow box"}[c["part"]],
            ", gap touches the half boundary" if c["censored"] else "")
    else:
        detail = "{} removed, {} filled frame(s) in {} correction(s)".format(
            c["removed"], c["filled"], c["n_corrections"])

    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, size)
    if not writer.isOpened():
        if reader is not None:
            reader.close()
        raise IOError("could not open the video writer for {}".format(out_path))
    try:
        for i in range(n):
            f = f0 + i
            canvas = np.zeros((size[1], size[0], 3), dtype=np.uint8)
            raw_ok = not np.isnan(raw_ball[i]).any()
            cl_ok = not np.isnan(clean[i]).any()

            # left: video with the tracking projected
            if reader is not None:
                img = reader.read()
                if img is None:
                    print("    video ended at frame {}".format(f))
                    break
                if H is not None and i < len(H_valid) and H_valid[i]:
                    ok_p = ~np.isnan(players[i, :, 0]) & ~np.isnan(players[i, :, 1])
                    u, v = project(H[i], players[i, ok_p, 0], players[i, ok_p, 1])
                    for uu, vv, tm in zip(u, v, team[ok_p]):
                        if np.isfinite(uu) and np.isfinite(vv):
                            cv2.circle(img, (int(uu * scale), int(vv * scale)), 4,
                                       (40, 39, 214) if tm == 0 else (180, 119, 31), -1, cv2.LINE_AA)
                    if raw_ok:
                        u, v = project(H[i], raw_ball[i, 0], raw_ball[i, 1])
                        if np.isfinite(u) and np.isfinite(v):
                            p = (int(u * scale), int(v * scale))
                            if removed[i]:
                                _cross(img, p, STATE_BGR[REMOVED], 8, 2)
                            else:
                                cv2.circle(img, p, 4, (255, 255, 255), -1, cv2.LINE_AA)
                    if cl_ok and st[i] in (FILLED, REMOVED):
                        u, v = project(H[i], clean[i, 0], clean[i, 1])
                        if np.isfinite(u) and np.isfinite(v):
                            cv2.circle(img, (int(u * scale), int(v * scale)), 9, STATE_BGR[FILLED], 2,
                                       cv2.LINE_AA)
                canvas[:PANEL_H, :PANEL_W] = img
            else:
                put(canvas, "video disabled (--no-video)", (20, 40))

            # right: pitch with the cleaned ball, plus raw / removed markers
            trail = clean[max(0, i - args.trail):i + 1]
            pimg, _, _, _ = panel.render(players[i], clean[i], team, trail, home_name, away_name, False)
            if removed[i] and raw_ok:
                p = tuple(int(round(v)) for v in panel.norm(raw_ball[i, 0], raw_ball[i, 1]))
                _cross(pimg, p, STATE_BGR[REMOVED], 9, 2)
                put(pimg, "removed", (p[0] + 10, p[1] + 18), 0.45, STATE_BGR[REMOVED])
            if cl_ok and st[i] in (FILLED, REMOVED) and res["filled"][f]:
                p = tuple(int(round(v)) for v in panel.norm(clean[i, 0], clean[i, 1]))
                cv2.circle(pimg, p, 11, STATE_BGR[FILLED], 2, cv2.LINE_AA)
                put(pimg, "filled", (p[0] + 12, p[1] - 10), 0.45, STATE_BGR[FILLED])
            if raw_ok and cl_ok and not removed[i]:
                off = float(np.hypot((raw_ball[i, 0] - clean[i, 0]) * PITCH[0],
                                     (raw_ball[i, 1] - clean[i, 1]) * PITCH[1]))
            else:
                off = float("nan")
            canvas[:PANEL_H, PANEL_W:] = pimg

            # info bar
            y0 = PANEL_H
            put(canvas, "{}   [{}]   pipeline {}".format(title, c["name"], args.pipeline), (12, y0 + 22),
                0.6, (0, 215, 255), 2)
            put(canvas, detail, (12, y0 + 46), 0.52)
            col = STATE_BGR[int(st[i])] if st[i] != MISSING else (110, 110, 255)
            put(canvas, "frame {}   t {}   half {}   ball: {}   raw {}   cleaned {}{}".format(
                f, fmt_time(f), half_of_video_frame(GAME, f) or "-", STATE_NAME[int(st[i])],
                "seen" if raw_ok else "-", "yes" if cl_ok else "NO",
                "   raw->cleaned {:.2f} m".format(off) if np.isfinite(off) else ""),
                (12, y0 + 72), 0.52, col)
            put(canvas, "event frames {}..{}   clip {}..{} (+-{} frames)".format(
                c["event_lo"], c["event_hi"], c["clip_lo"], c["clip_hi"], args.pad),
                (12, y0 + 96), 0.48, (190, 190, 190))
            bar = strip.copy()
            cv2.line(bar, (x_of(i), 14), (x_of(i), 48), (0, 255, 255), 2)
            canvas[PANEL_H + INFO_H:, :] = bar
            writer.write(canvas)
    finally:
        writer.release()
        if reader is not None:
            reader.close()


# ------------------------------------------------------------------------------------------------
# 4. Command line
# ------------------------------------------------------------------------------------------------

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pipeline", choices=("v6", "v4", "prepared"), default="v6",
                    help="ball clean-up to test (default v6 = the current submission)")
    ap.add_argument("--types", default="not_tracked,corrected",
                    help="comma list: not_tracked, corrected (default both)")
    ap.add_argument("--pad", type=int, default=75, help="frames before and after each event (default 75)")
    ap.add_argument("--min-gap", type=int, default=1, help="shortest missing stretch to clip, frames")
    ap.add_argument("--half", type=int, choices=(1, 2), help="only this half")
    ap.add_argument("--max-clips", type=int, default=0, help="stop after this many per type (0 = all)")
    ap.add_argument("--only", default="", help="only clips whose name contains this, e.g. fix_0061740")
    ap.add_argument("--out-dir", default=OUT_DIR, help="default outputs/test_ball")
    ap.add_argument("--overwrite", action="store_true", help="re-render clips that already exist")
    ap.add_argument("--dry-run", action="store_true", help="list the clips and write index.csv only")
    ap.add_argument("--no-video", action="store_true", help="pitch panel only (no mp4 decode)")
    ap.add_argument("--trail", type=int, default=25, help="ball trail on the pitch, frames")
    ap.add_argument("--backend", choices=("auto", "ffmpeg", "opencv"), default="auto")
    ap.add_argument("--exact-seek", action="store_true", help="OpenCV backend: grab frame by frame")
    ap.add_argument("--video-offset", type=int, default=0, help="video frames to shift the video by")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    types = tuple(t.strip() for t in args.types.split(",") if t.strip())
    bad = set(types) - {"not_tracked", "corrected"}
    if bad:
        sys.exit("unknown --types {}".format(sorted(bad)))

    t0 = time.time()
    bx, by, _ = load_ball()
    res = run_pipeline(bx, by, args.pipeline)
    states = frame_states(res)
    blocks = [b for b in graded_blocks() if args.half in (None, b[0])]
    clips = plan_clips(res, blocks, args.pad, args.min_gap, types)
    summary = summarize(res, blocks, clips, args.pipeline)
    print("ball pipeline {}: tracked {raw_tracked_pct}% of graded frames raw, {after_cleaning_pct}% "
          "after cleaning; {removed_frames} frames removed, {filled_frames} filled; {missing_runs} "
          "missing stretches {missing_runs_by_length_s}".format(args.pipeline, **summary))
    print("clips: {}  ({:.1f} s)".format(summary["clips"], time.time() - t0))

    os.makedirs(args.out_dir, exist_ok=True)
    fields = ["kind", "name", "file", "half", "event_lo", "event_hi", "clip_lo", "clip_hi", "part",
              "gap_len", "gap_s", "censored", "removed", "filled", "n_corrections"]
    with open(os.path.join(args.out_dir, "index.csv"), "w", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for c in clips:
            c["file"] = os.path.join(c["kind"], c["name"] + ".mp4")
            w.writerow(c)
    with open(os.path.join(args.out_dir, "summary.json"), "w") as handle:
        json.dump(dict(summary, pad=args.pad, min_gap=args.min_gap, half=args.half), handle, indent=2)
    print("wrote {} and summary.json".format(os.path.relpath(os.path.join(args.out_dir, "index.csv"), ROOT)))
    if args.only:                        # filters choose what to render; index.csv lists all
        clips = [c for c in clips if args.only in c["name"]]
    if args.max_clips:
        clips = [c for k in types for c in [x for x in clips if x["kind"] == k][:args.max_clips]]
    if args.dry_run:
        for c in clips:
            print("  {:<12} {:<44} clip {}..{}".format(c["kind"], c["name"], c["clip_lo"], c["clip_hi"]))
        return

    from games import require
    from starter.baseline import team_names, team_of_each_track
    meta_path, raw_path = require(["atd_meta", "atd_raw"])
    ctx = dict(team=np.asarray(team_of_each_track(meta_path)), raw=raw_path)
    ctx["home"], ctx["away"] = team_names(meta_path)
    done = failed = skipped = 0
    for k, c in enumerate(clips, 1):
        out = os.path.join(args.out_dir, c["file"])
        os.makedirs(os.path.dirname(out), exist_ok=True)
        if os.path.isfile(out) and not args.overwrite:
            skipped += 1
            continue
        t = time.time()
        print("[{}/{}] {} ({} frames)".format(k, len(clips), c["file"], c["clip_hi"] - c["clip_lo"] + 1),
              flush=True)
        try:
            render_clip(c, res, states, args, ctx, out)
            done += 1
            print("    {:.1f} s".format(time.time() - t), flush=True)
        except Exception as error:          # one bad clip must not stop the batch
            failed += 1
            print("    FAILED: {}".format(error), flush=True)
    print("rendered {}, skipped (already there) {}, failed {} -> {}".format(
        done, skipped, failed, os.path.relpath(args.out_dir, ROOT)))


if __name__ == "__main__":
    main()
