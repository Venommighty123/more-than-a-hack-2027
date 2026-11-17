"""Clean the YOLO ball and fix its flights; write a ball file the whole pipeline can use.

    python scripts/fix_yolo_ball.py                                   # -> outputs/ball/ball_yolo_fixed.npz
    python scripts/fix_yolo_ball.py --fallback none                   # YOLO only, no provided ball
    python scripts/fix_yolo_ball.py --yolo data/kaggle_outputs/ball_yolo_smoke.parquet --csv
    python scripts/run.py --labeller v6 --ball-file outputs/ball/ball_yolo_fixed.npz --out outputs/submissions/v6_yoloball.csv

INPUT   data/kaggle_outputs/ball_yolo_<mode>.parquet (+ ball_frames_run_<mode>.npy) from
        kaggle/kaggle_ball_detection.ipynb, the homography, and (default) the fixed provided ball
        outputs/ball/ball_reconstructed.npz as a fallback where YOLO has no ball.
WHAT    possession/preprocess/ball_yolo.py: spikes / isolated picks out, gaps <= 5 frames bridged in
        pixels, flight candidates on the naive ground track, a 3D parabola per candidate through
        each frame's camera; airborne flights get their true ground position (X, Y) and height Z.
OUTPUT  (git-ignored, derived match data: never commit or upload, RULES.md)
  <out>.npz              same format as ball_reconstructed.npz (x, y, observed, removed, filled, ...)
                         plus z (height, m), src (per-frame source code), in_air: so --ball-file works
                         in run.py / detect_events.py / possession.io.load()
  <out>_flights.csv      every flight candidate: frames, fitted peak height, fit errors, verdict
  <out>_summary.json     counts, frames per source, settings
  <out>.csv              per-frame table, with --csv
"""
import argparse
import glob
import json
import os

try:
    import _bootstrap  # noqa: F401  (python scripts/x.py)
except ImportError:
    from scripts import _bootstrap  # noqa: F401  (python -m scripts.x)

import numpy as np
import pandas as pd

from games import GAME, GAMES
from possession.features.flight import FlightConfig
from possession.preprocess.ball_yolo import SOURCE_NAMES, SRC_FALLBACK, BallFixConfig, fix_ball, load_yolo_ball
from possession.preprocess.reconstruct import DEFAULT_FILE as PROVIDED_BALL, load_ball_file
from starter.load_homography import load as load_homography

ROOT = _bootstrap.ROOT


def default_yolo():
    for mode in ("full", "smoke"):
        hits = sorted(glob.glob(os.path.join(ROOT, "data", "kaggle_outputs", "**", "ball_yolo_{}.parquet".format(mode)),
                           recursive=True))
        if hits:
            return hits[0]
    return os.path.join(ROOT, "data", "kaggle_outputs", "ball_yolo_full.parquet")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--yolo", default=None, help="ball_yolo_<mode>.parquet (default: data/kaggle_outputs/)")
    ap.add_argument("--frames-run", default=None, help="ball_frames_run_<mode>.npy (default: next to --yolo)")
    ap.add_argument("--fallback", choices=("provided", "none"), default="provided",
                    help="fill frames without a YOLO ball from the fixed provided ball")
    ap.add_argument("--provided-file", default=os.path.join(ROOT, PROVIDED_BALL))
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "ball", "ball_yolo_fixed.npz"))
    ap.add_argument("--csv", action="store_true", help="also write a per-frame CSV")
    ap.add_argument("--min-conf", type=float, default=BallFixConfig.min_conf)
    ap.add_argument("--min-height", type=float, default=FlightConfig.min_height_m,
                    help="m; a fitted flight must peak this high to count as airborne")
    args = ap.parse_args()

    yolo = args.yolo or default_yolo()
    if not os.path.isfile(yolo):
        raise SystemExit("{} not found: download the ball notebook's output into data/kaggle_outputs/".format(yolo))
    run_file = args.frames_run or yolo.replace("ball_yolo_", "ball_frames_run_").replace(".parquet", ".npy")
    n = GAMES[GAME]["n_video_frames"]
    raw = load_yolo_ball(yolo, n, run_file if os.path.isfile(run_file) else None, args.min_conf)
    Hn, _, valid = load_homography()
    fcfg = FlightConfig(fps=GAMES[GAME]["fps"], pitch=tuple(GAMES[GAME]["pitch"]), min_height_m=args.min_height)
    cfg = BallFixConfig(min_conf=args.min_conf)

    fallback = None
    if args.fallback == "provided":
        if os.path.isfile(args.provided_file):
            pb = load_ball_file(args.provided_file)
            fallback = np.stack([pb["x"], pb["y"]], 1)
        else:
            print("note: {} not found, no fallback (run scripts/reconstruct_ball.py)".format(args.provided_file))
    print("YOLO ball: {}".format(os.path.relpath(yolo, ROOT)))
    from possession.io import graded_mask
    out = fix_ball(raw, Hn, valid, fcfg, cfg, fallback=fallback, in_play=graded_mask(n))

    src = out["source"]
    observed = np.isin(src, [1, SRC_FALLBACK]) & np.isfinite(out["x"])
    filled = np.isin(src, [2, 3])
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, x=out["x"], y=out["y"], observed=observed, removed=out["removed"],
                        filled=filled, z=out["z"], src=src, in_air=out["in_air"],
                        method=np.array("yolo+flightfix" + ("+provided" if fallback is not None else "")),
                        max_gap=np.array(int(cfg.bridge)), max_step_m=np.array(float(cfg.spike_step_m)),
                        lo=np.array(0), max_speed_ms=np.array(float(fcfg.max_step_speed)),
                        source=np.array("YOLO {}".format(os.path.relpath(yolo, ROOT))))
    stem = os.path.splitext(args.out)[0]
    pd.DataFrame(out["flights"]).to_csv(stem + "_flights.csv", index=False, float_format="%.3f")
    rep = dict(out["report"], yolo_file=os.path.relpath(yolo, ROOT), fallback=args.fallback,
               graded_frames_with_ball=None)
    g = graded_mask(n)
    rep["graded_frames_with_ball_pct"] = round(100 * float(np.isfinite(out["x"][g]).mean()), 2)
    with open(stem + "_summary.json", "w") as h:
        json.dump(rep, h, indent=2, default=str)
    if args.csv:
        pd.DataFrame({"frame": np.arange(n), "x_m": out["x"], "y_m": out["y"], "z_m": out["z"],
                      "source": [SOURCE_NAMES[int(k)] for k in src], "in_air": out["in_air"]}
                     ).to_csv(stem + ".csv", index=False, float_format="%.3f")
    print("frames with a ball: {:.1f}% of graded frames; by source {}".format(
        rep["graded_frames_with_ball_pct"], rep["by_source"]))
    print("wrote {}\n      {}_flights.csv, {}_summary.json".format(
        os.path.relpath(args.out, ROOT), os.path.relpath(stem, ROOT), os.path.relpath(stem, ROOT)))


if __name__ == "__main__":
    main()
