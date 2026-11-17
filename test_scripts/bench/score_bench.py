"""Score a possession labeller on the Metrica bench (games built by build_metrica_bench.py).

A labeller is any function  label(px, py, ball, team, blocks) -> array of "H"/"A"/"D"/"X"  with one
entry per frame. It never sees the ground truth. Scores use the official
evaluation/possession_metrics.evaluate(), so a number here means the same as a number on the board.

    python test_scripts/bench/score_bench.py --bench bench/ --labeller baseline50
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # test_scripts/
import possession_metrics as pm  # noqa: E402

CODES = {"H": pm.H, "A": pm.A, "D": pm.D, "X": pm.X}


def to_codes(labels):
    out = np.full(len(labels), pm.X, dtype=np.int8)
    for k, v in CODES.items():
        out[labels == k] = v
    return out


def baseline(px, py, ball, team, blocks, window=50):
    """Copy of starter/baseline.py: nearest player to the ball, majority over +/- window frames."""
    n = len(ball)
    codes = np.full(n, -1, dtype=np.int8)
    ok = ~np.isnan(ball[:, 0])
    d = np.hypot(px[ok] - ball[ok, 0:1], py[ok] - ball[ok, 1:2])
    d = np.where(np.isnan(d), np.inf, d)
    has = np.isfinite(d).any(1)
    near = np.full(ok.sum(), -1, dtype=np.int8)
    near[has] = team[d[has].argmin(1)]
    codes[ok] = near
    hc = np.concatenate(([0], np.cumsum(codes == 0)))
    ac = np.concatenate(([0], np.cumsum(codes == 1)))
    i = np.arange(n)
    lo, hi = np.maximum(0, i - window), np.minimum(n, i + window + 1)
    h, a = hc[hi] - hc[lo], ac[hi] - ac[lo]
    lab = np.full(n, "D", dtype="<U1")
    seen = (h + a) > 0
    lab[seen] = np.where(h[seen] >= a[seen], "H", "A")
    return lab


LABELLERS = {"baseline0": lambda *a: baseline(*a, window=0),
             "baseline50": baseline,
             "baseline125": lambda *a: baseline(*a, window=125)}


def get_labeller(name):
    if name in LABELLERS:
        return LABELLERS[name]
    import importlib
    return importlib.import_module(name).label


def score(bench, name, variant="degraded", games=("g1", "g2", "g3"), verbose=True, use_tags=True):
    fn = get_labeller(name)
    rows = []
    for g in games:
        d = np.load(os.path.join(bench, "{}_{}.npz".format(g, variant)))
        args = (d["px"], d["py"], d["ball"], d["team"], [tuple(b) for b in d["blocks"]])
        try:
            pred = fn(*args, tags=d["tags"] if use_tags else None)
        except TypeError:          # labellers that take no tags (baselines, v1)
            pred = fn(*args)
        r = pm.evaluate(to_codes(d["gt"]), to_codes(np.asarray(pred)))
        rows.append(r)
        if verbose:
            pc = r["per_class"]
            print("  {} {:<9} spellF1 {:.4f}  macroF1 {:.4f}  spells pred/gt {}/{}  "
                  "F1 H {:.3f} A {:.3f} D {:.3f}  alive/dead F1 {:.3f}".format(
                      g, variant, r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"],
                      r["n_gt_spells"], pc["H"]["f1"], pc["A"]["f1"], pc["D"]["f1"],
                      r["alive_dead_f1"]))
    s = np.mean([r["spell_f1_mean"] for r in rows])
    m = np.mean([r["macro_f1"] for r in rows])
    print("{:<14} {:<9} MEAN spellF1 {:.4f}  macroF1 {:.4f}".format(name, variant, s, m))
    return s, m, rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--labeller", default="baseline50")
    ap.add_argument("--variant", default="degraded", choices=("clean", "degraded", "hard", "tracked", "tracked_all", "real", "real_occ", "all"))
    ap.add_argument("--no-tags", action="store_true", help="do not give the labeller set-piece tags")
    args = ap.parse_args()
    for v in (("clean", "degraded", "hard", "tracked", "tracked_all", "real", "real_occ") if args.variant == "all" else (args.variant,)):
        score(args.bench, args.labeller, v, use_tags=not args.no_tags)
