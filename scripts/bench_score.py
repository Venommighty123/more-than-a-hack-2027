"""Score labellers on the bench with the official metric (evaluation/possession_metrics.evaluate).

    python scripts/bench_score.py                                   # v1 and baseline50, degraded
    python scripts/bench_score.py --labeller v1 baseline50 --variant all
    python scripts/bench_score.py --labeller v1 --set CONFIRM=8 --set MIN_SPELL=1.5

A labeller never sees the ground truth. This is the ONLY place versions and settings are compared
(RULES.md, EVALUATION.md). Log every accepted change with its score in docs/changelog/labeller.md.
The bench reads ~0.07 higher than the board on spellF1: compare bench numbers with each other only.
"""
import argparse
import os

try:
    import _bootstrap  # noqa: F401  (python scripts/x.py)
except ImportError:
    from scripts import _bootstrap  # noqa: F401  (python -m scripts.x)

import numpy as np

import possession_metrics as pm
from possession.io.bench import GAMES, iter_bench
from possession.labellers import get_labeller
from possession.utils.frames import to_codes

ROOT = _bootstrap.ROOT
VARIANTS = ("clean", "degraded", "hard")


def _value(text):
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text


def score(bench, name, variant="degraded", games=GAMES, params=None, verbose=True):
    fn = get_labeller(name)
    rows = []
    for g in iter_bench(bench, variant, games):
        kw = {"fps": g.fps, "pitch": g.pitch}
        pred = fn(g.px, g.py, g.ball, g.team, g.blocks, params, **kw) if params else \
            fn(g.px, g.py, g.ball, g.team, g.blocks, **kw)
        r = pm.evaluate(to_codes(g.gt), to_codes(np.asarray(pred)))
        rows.append(r)
        if verbose:
            pc = r["per_class"]
            print("  {} {:<9} spellF1 {:.4f}  macroF1 {:.4f}  spells pred/gt {}/{}  "
                  "F1 H {:.3f} A {:.3f} D {:.3f}  alive/dead F1 {:.3f}".format(
                      g.name, variant, r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"],
                      r["n_gt_spells"], pc["H"]["f1"], pc["A"]["f1"], pc["D"]["f1"],
                      r["alive_dead_f1"]))
    s = float(np.mean([r["spell_f1_mean"] for r in rows]))
    m = float(np.mean([r["macro_f1"] for r in rows]))
    print("{:<14} {:<9} MEAN spellF1 {:.4f}  macroF1 {:.4f}".format(name, variant, s, m))
    return s, m, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bench", default=os.path.join(ROOT, "outputs", "bench"))
    ap.add_argument("--labeller", nargs="+", default=["v1", "baseline50"])
    ap.add_argument("--variant", default="degraded", choices=VARIANTS + ("all",))
    ap.add_argument("--games", nargs="+", default=list(GAMES))
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a labeller parameter (v1 uses its DEFAULTS keys)")
    args = ap.parse_args()
    params = dict((k, _value(v)) for k, v in (s.split("=", 1) for s in args.set)) or None
    variants = VARIANTS if args.variant == "all" else (args.variant,)
    for name in args.labeller:
        for v in variants:
            score(args.bench, name, v, args.games, params)


if __name__ == "__main__":
    main()
