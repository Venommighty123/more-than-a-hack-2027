"""Choose the ball among the YOLO candidates (Level 1): pitch / static / size filters + a Viterbi path.

    python scripts/select_ball_candidates.py                     # -> outputs/ball/ball_yolo_selected.parquet
    python scripts/select_ball_candidates.py --no-anchor --out outputs/ball/ball_yolo_selected_noanchor.parquet
    python scripts/fix_yolo_ball.py --yolo outputs/ball/ball_yolo_selected.parquet \
        --frames-run outputs/ball/ball_frames_run_full.npy --out outputs/ball/ball_yolo_selected_fixed.npz
    python scripts/fill_ball_gaps.py --yolo outputs/ball/ball_yolo_selected_fixed.npz --out outputs/ball/ball_filled_v2.npz

INPUT   outputs/ball/ball_candidates_full.parquet (top-3 candidates per frame, kaggle/kaggle_ball_detection.ipynb),
        ball_frames_run_full.npy (frames the model ran on), the tactical homography, the ATD ball.
WHAT    possession/preprocess/ball_select.py: reject candidates far off the pitch, static objects and
        wrong-sized boxes, then pick the most plausible path (confidence, movement, pitch, ATD anchors).
        --no-anchor drops the ATD anchors: use that version to MEASURE (the fill gate hides the ATD ball,
        so an anchored path would partly see what it is scored against).
OUTPUT  a parquet in the ball_yolo_<mode>.parquet format (frame, half, u, v, conf, x_m, y_m, n_cand, jump),
        so scripts/fix_yolo_ball.py takes it unchanged; <out>_summary.json with the filter counts and the
        comparison with the old pick. Derived match data (outputs/ is git-ignored). Aggregates only printed.
"""
import argparse
import json
import os
import time

try:
    import _bootstrap  # noqa: F401  (python scripts/x.py)
except ImportError:
    from scripts import _bootstrap  # noqa: F401  (python -m scripts.x)

import numpy as np
import pandas as pd

from games import GAME, GAMES
from possession.io import graded_blocks
from possession.io.tracking import load_ball
from possession.preprocess.ball_select import SelectConfig, expected_px, outside_m, static_mask, viterbi_run
from starter.load_homography import load as load_homography

ROOT = _bootstrap.ROOT
CANDS = os.path.join(ROOT, "outputs", "ball", "ball_candidates_full.parquet")
RUN = os.path.join(ROOT, "outputs", "ball", "ball_frames_run_full.npy")
OLD = os.path.join(ROOT, "outputs", "ball", "ball_yolo_full.parquet")
OUT = os.path.join(ROOT, "outputs", "ball", "ball_yolo_selected.parquet")


def _value(text):
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text


def accuracy(frames, x, y, ax, ay):
    """Distance to the ATD ball on frames where both exist."""
    m = np.isfinite(ax[frames]) & np.isfinite(x)
    d = np.hypot(x[m] - ax[frames][m], y[m] - ay[frames][m])
    if not len(d):
        return {"frames": 0}
    return {"frames": int(len(d)), "median_m": round(float(np.median(d)), 2), "within_2m": round(float((d <= 2).mean()), 3),
            "over_5m": round(float((d > 5).mean()), 3), "over_10m": round(float((d > 10).mean()), 3)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidates", default=CANDS)
    ap.add_argument("--frames-run", default=RUN)
    ap.add_argument("--old", default=OLD, help="the notebook's own pick, for the comparison")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--no-anchor", action="store_true", help="no ATD anchors (the version to measure with)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a SelectConfig field")
    args = ap.parse_args()
    cfg = SelectConfig.from_dict({**{"pitch": tuple(GAMES[GAME]["pitch"]), "fps": GAMES[GAME]["fps"]},
                                  **{k: _value(v) for k, v in (s.split("=", 1) for s in args.set)}})
    if args.no_anchor:
        cfg.anchor_w = 0.0
    t0 = time.time()

    C = pd.read_parquet(args.candidates).reset_index(drop=True)
    ran = np.load(args.frames_run).astype(bool)
    n = len(ran)
    H, _, valid = load_homography()
    ax, ay, _ = load_ball(metres=True)

    # ---- filters ----------------------------------------------------------------------------------
    out_m = outside_m(C.x_m.to_numpy(float), C.y_m.to_numpy(float), cfg.pitch)
    off = ~np.isfinite(out_m) | (out_m > cfg.hard_out_m)
    stat = static_mask(C.frame.to_numpy(), C.u.to_numpy(float), C.v.to_numpy(float), cfg)
    box = ((C.x2 - C.x1) + (C.y2 - C.y1)).to_numpy(float) / 2.0
    exp = np.full(len(C), np.nan)
    for i, (f, xm, ym) in enumerate(zip(C.frame.to_numpy(), C.x_m.to_numpy(float), C.y_m.to_numpy(float))):
        if valid[f] and np.isfinite(xm) and not off[i]:
            exp[i] = expected_px(H[f], xm, ym, cfg.pitch)
    ratio = box / exp
    size_bad = np.isfinite(ratio) & ((ratio < cfg.size_min) | (ratio > cfg.size_max))
    ok = ~off & ~stat & ~size_bad
    C["ok"] = ok

    # ---- path per run of processed frames inside the halves ---------------------------------------
    cands = {}
    for f, g in C.groupby("frame"):
        cands[int(f)] = (g.x_m.to_numpy(float), g.y_m.to_numpy(float), g.conf.to_numpy(float), g.ok.to_numpy(bool))
    atd = {} if args.no_anchor else {int(f): (ax[f], ay[f]) for f in np.flatnonzero(np.isfinite(ax))}
    chosen = {}
    half_of = np.zeros(n, dtype=np.int8)
    for h, a, b in graded_blocks():
        half_of[a:b + 1] = h
        idx = np.flatnonzero(ran[a:b + 1]) + a
        if not len(idx):
            continue
        cuts = np.flatnonzero(np.diff(idx) > 1) + 1
        for run in np.split(idx, cuts):
            chosen.update(viterbi_run(run, cands, atd, cfg))

    groups = C.groupby("frame").indices                     # frame -> row positions, in the same order as cands
    sel_idx = [int(groups[f][j]) for f, j in chosen.items() if j >= 0]
    S = C.loc[sel_idx].copy()
    S["half"] = half_of[S.frame.to_numpy()]
    S["n_cand"] = S.frame.map(C[C.ok].groupby("frame").size()).fillna(0).astype(int)
    S["jump"] = False
    S = S.sort_values("frame")[["frame", "half", "u", "v", "conf", "x_m", "y_m", "n_cand", "jump"]]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    S.to_parquet(args.out, index=False)

    # ---- comparison with the notebook's own pick (aggregates) ---------------------------------------
    O = pd.read_parquet(args.old) if os.path.exists(args.old) else None
    def per_frame(df):
        x, y = np.full(n, np.nan), np.full(n, np.nan)
        x[df.frame.to_numpy()] = df.x_m.to_numpy(float); y[df.frame.to_numpy()] = df.y_m.to_numpy(float)
        return x, y
    graded = half_of > 0
    frames_eval = np.flatnonzero(graded & ran)
    nx, ny = per_frame(S)
    summary = {"config": cfg.to_dict(), "anchored": not args.no_anchor,
               "candidates": int(len(C)), "frames_with_candidates": int(C.frame.nunique()),
               "rejected": {"off_pitch": int(off.sum()), "static": int((stat & ~off).sum()),
                            "size": int((size_bad & ~off & ~stat).sum()), "kept": int(ok.sum())},
               "chosen_frames": int(len(S)),
               "new": {"off_pitch_1m_share": round(float((outside_m(S.x_m.to_numpy(float), S.y_m.to_numpy(float), cfg.pitch) > 1).mean()), 4) if len(S) else None,
                       "vs_atd": accuracy(frames_eval, nx[frames_eval], ny[frames_eval], ax, ay),
                       "on_atd_missing_frames": int(np.isfinite(nx[frames_eval][~np.isfinite(ax[frames_eval])]).sum())}}
    if O is not None:
        ox, oy = per_frame(O)
        summary["old"] = {"chosen_frames": int(len(O)),
                          "off_pitch_1m_share": round(float((outside_m(O.x_m.to_numpy(float), O.y_m.to_numpy(float), cfg.pitch) > 1).mean()), 4),
                          "vs_atd": accuracy(frames_eval, ox[frames_eval], oy[frames_eval], ax, ay),
                          "on_atd_missing_frames": int(np.isfinite(ox[frames_eval][~np.isfinite(ax[frames_eval])]).sum())}
    summary["runtime_sec"] = round(time.time() - t0, 1)
    with open(os.path.splitext(args.out)[0] + "_summary.json", "w") as handle:
        json.dump(summary, handle, indent=1)

    print("candidates {} on {} frames | rejected: off pitch {}, static {}, size {} | kept {}".format(
        summary["candidates"], summary["frames_with_candidates"], *summary["rejected"].values()))
    print("{} path ({}): ball chosen on {} frames -> {}".format(
        "anchored" if not args.no_anchor else "no-anchor", "ATD anchors on" if not args.no_anchor else "measure with this",
        len(S), os.path.relpath(args.out, ROOT)))
    for k in ("old", "new"):
        if k in summary:
            s = summary[k]
            print("  {:<4} off pitch (>1 m) {:>6.1%} | vs ATD {} | frames where the ATD has no ball: {}".format(
                k, s["off_pitch_1m_share"] or 0, s["vs_atd"], s["on_atd_missing_frames"]))
    print("done in {:.0f}s".format(summary["runtime_sec"]))


if __name__ == "__main__":
    main()
