"""Fill ball gaps with the cleaned YOLO ball, conservatively, behind an accuracy gate (P4 + P5).

    python scripts/fill_ball_gaps.py                     # -> outputs/ball/ball_filled.npz (+ _gaps.csv, _summary.json)
    python scripts/fill_ball_gaps.py --set anchor_m=2 --set max_hole=3
    python scripts/run.py --labeller v6 --ball-file outputs/ball/ball_filled.npz --out outputs/submissions/v6_filled.csv

INPUTS
    outputs/ball/ball_reconstructed.npz                 base ball (the pipeline default), metres
    outputs/ball/ball_yolo_fixed.npz                    cleaned YOLO ball (scripts/fix_yolo_ball.py); only src == yolo is used
    outputs/dead/dead_stretches_labeller_v6_newtracks.csv   v6 dead stretches (scope)
    outputs/referee/**/referee_stretches.csv            referee gestures per stretch (gesture = stays dead)

WHAT (possession/preprocess/ball_fill.py)
    1. scope   open-play gaps are eligible; gaps touching a v6 dead stretch only with live evidence;
               a referee gesture keeps a stretch dead
    2. rule    continuous, anchor-attached chains of YOLO detections (holes <= max_hole, speed <= max_speed)
    3. gate    hide the base ball on random open-play windows, refill with the rule, measure the error.
               The fill is written only if >= --gate-within-2m of refilled frames are within 2 m;
               otherwise the output is the base ball unchanged (use --force to override, not advised).

OUTPUT (git-ignored, derived match data): <out>.npz in ball_reconstructed.npz format (+ fill_code:
0 base/none, 1 YOLO point, 2 interpolated), <out>_gaps.csv (one row per gap and its decision),
<out>_summary.json. Only aggregates are printed.
"""
import argparse
import csv
import glob
import json
import os

try:
    import _bootstrap  # noqa: F401  (python scripts/x.py)
except ImportError:
    from scripts import _bootstrap  # noqa: F401  (python -m scripts.x)

import numpy as np

from games import GAME, GAMES
from possession.io import graded_blocks
from possession.preprocess.ball_fill import FillConfig, fill_gaps, find_gaps, hide_and_refill

ROOT = _bootstrap.ROOT
BASE = os.path.join(ROOT, "outputs", "ball", "ball_reconstructed.npz")
YOLO = os.path.join(ROOT, "outputs", "ball", "ball_yolo_fixed.npz")
DEAD = os.path.join(ROOT, "outputs", "dead", "dead_stretches_labeller_v6_newtracks.csv")
REF_GLOB = os.path.join(ROOT, "outputs", "referee", "**", "referee_stretches.csv")
OUT = os.path.join(ROOT, "outputs", "ball", "ball_filled.npz")
SRC_YOLO = 1          # possession/preprocess/ball_yolo.py SRC_OBSERVED


def _value(text):
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text


def _read_csv(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--yolo", default=YOLO)
    ap.add_argument("--dead", default=DEAD)
    ap.add_argument("--referee", default=None, help="referee_stretches.csv (default: search outputs/referee)")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--gate-within-2m", type=float, default=0.80)
    ap.add_argument("--test-windows", type=int, default=400)
    ap.add_argument("--force", action="store_true", help="write the fill even if the gate fails")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a FillConfig field")
    args = ap.parse_args()
    cfg = FillConfig.from_dict({**{"pitch": tuple(GAMES[GAME]["pitch"]), "fps": GAMES[GAME]["fps"]},
                                **{k: _value(v) for k, v in (s.split("=", 1) for s in args.set)}})
    ref_path = args.referee or next(iter(sorted(glob.glob(REF_GLOB, recursive=True))), None)
    for p in (args.base, args.yolo, args.dead):
        if not os.path.exists(p):
            raise SystemExit("missing input: {}".format(os.path.relpath(p, ROOT)))

    base = np.load(args.base, allow_pickle=False)
    bx, by = base["x"].astype(np.float64), base["y"].astype(np.float64)
    yolo = np.load(args.yolo, allow_pickle=False)
    if len(yolo["x"]) != len(bx):
        raise SystemExit("ball files differ in length: {} vs {}".format(len(yolo["x"]), len(bx)))
    cx, cy = yolo["x"].astype(np.float64), yolo["y"].astype(np.float64)
    cand = (yolo["src"] == SRC_YOLO) & np.isfinite(cx) & np.isfinite(cy)

    n = len(bx)
    blocks = graded_blocks()
    graded = np.zeros(n, dtype=bool)
    for _, a, b in blocks:
        graded[a:b + 1] = True
    dead_rows = _read_csv(args.dead)
    dead_id = np.zeros(n, dtype=np.int32)
    for r in dead_rows:
        dead_id[int(r["first_frame"]):int(r["last_frame"]) + 1] = int(r["d"])
    gesture_d = set()
    if ref_path and os.path.exists(ref_path):
        gesture_d = {int(r["d"]) for r in _read_csv(ref_path) if r.get("referee_signal") == "yes"}

    present = np.isfinite(bx) & np.isfinite(by)
    gaps = find_gaps(present, blocks)
    dead_of_gap, gesture_of_gap = {}, {}
    for gi, (_, a, b) in enumerate(gaps):
        ids = sorted(set(int(v) for v in dead_id[a:b] if v))
        if ids:
            dead_of_gap[gi] = ids
            gesture_of_gap[gi] = bool(set(ids) & gesture_d)

    # 3. gate first: hide-and-refill on open-play windows where the base ball was observed
    observed = base["observed"].astype(bool) if "observed" in base.files else present
    eligible = graded & present & observed & (dead_id == 0)
    open_lengths = [b - a for gi, (_, a, b) in enumerate(gaps) if gi not in dead_of_gap]
    test = hide_and_refill(bx, by, cx, cy, cand, eligible, open_lengths or [25, 50, 100], cfg, n=args.test_windows)
    passed = bool(test.get("within_2m") is not None and test["within_2m"] >= args.gate_within_2m)

    X, Y, code, rows = fill_gaps(bx, by, cx, cy, cand, gaps, dead_of_gap, gesture_of_gap, cfg)
    write_fill = passed or args.force
    if not write_fill:
        X, Y, code = bx.copy(), by.copy(), np.zeros(n, dtype=np.int8)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    new = code > 0
    np.savez_compressed(
        args.out, x=X, y=Y, observed=base["observed"], removed=base["removed"],
        filled=base["filled"].astype(bool) | new, fill_code=code,
        method=np.str_("{}+yolo_gapfill".format(str(base["method"])) if write_fill else str(base["method"])),
        max_gap=base["max_gap"], max_step_m=base["max_step_m"], lo=base["lo"],
        max_speed_ms=base["max_speed_ms"] if "max_speed_ms" in base.files else np.float64(0.0),
        source=np.str_(str(base["source"])))
    stem = os.path.splitext(args.out)[0]
    keys = list(rows[0].keys()) if rows else ["gap"]
    with open(stem + "_gaps.csv", "w", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=keys); w.writeheader(); w.writerows(rows)

    def agg(scope):
        rs = [r for r in rows if r["scope"] == scope]
        out = {"gaps": len(rs), "frames": int(sum(r["frames"] for r in rs)),
               "filled_frames": int(sum(r["filled_frames"] for r in rs)) if write_fill else 0, "by_decision": {}}
        for r in rs:
            out["by_decision"][r["decision"]] = out["by_decision"].get(r["decision"], 0) + 1
        reasons = {}
        for r in rs:
            if r["decision"] == "skipped":
                key = r["reason"].split(" (")[0]
                reasons[key] = reasons.get(key, 0) + 1
        out["skip_reasons"] = reasons
        return out

    missing_graded = int((graded & ~present).sum())
    summary = {
        "config": cfg.to_dict(), "gate_within_2m": args.gate_within_2m, "gate_passed": passed,
        "fill_written": write_fill, "forced": bool(args.force and not passed),
        "inputs": {k: os.path.relpath(v, ROOT) if v else None for k, v in
                   (("base", args.base), ("yolo", args.yolo), ("dead", args.dead), ("referee", ref_path))},
        "referee_gesture_stretches": len(gesture_d),
        "hide_and_refill": test,
        "graded_missing_before": missing_graded,
        "graded_missing_after": int((graded & ~(np.isfinite(X) & np.isfinite(Y))).sum()),
        "open_play": agg("open_play"), "dead": agg("dead"),
        "fill_code_frames": {"yolo_point": int((code == 1).sum()), "interpolated": int((code == 2).sum())},
    }
    with open(stem + "_summary.json", "w") as handle:
        json.dump(summary, handle, indent=1)

    print("gate (hide-and-refill on {} open-play windows): {} of refilled frames within 2 m "
          "(median {} m, >5 m {}, fill rate {}) -> {}".format(
              test.get("windows"), test.get("within_2m"), test.get("median_err_m"), test.get("over_5m"),
              test.get("fill_rate"), "PASS" if passed else "FAIL (threshold {})".format(args.gate_within_2m)))
    for scope in ("open_play", "dead"):
        s = summary[scope]
        print("{:<10} gaps {:4d}  frames {:6d}  filled {:6d}  decisions {}  skipped because {}".format(
            scope, s["gaps"], s["frames"], s["filled_frames"], s["by_decision"], s["skip_reasons"]))
    print("graded frames without a ball: {} -> {}".format(summary["graded_missing_before"], summary["graded_missing_after"]))
    print("{} -> {}".format("fill written" if write_fill else "gate failed: base ball written unchanged",
                            os.path.relpath(args.out, ROOT)))


if __name__ == "__main__":
    main()
