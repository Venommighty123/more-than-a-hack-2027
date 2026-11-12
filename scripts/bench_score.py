"""Score labellers on the bench with the official metric (evaluation/possession_metrics.evaluate).

    python scripts/bench_score.py                                   # v1 and baseline50, degraded
    python scripts/bench_score.py --labeller v1 baseline50 --variant all
    python scripts/bench_score.py --labeller v1 --set CONFIRM=8 --set MIN_SPELL=1.5
    python scripts/bench_score.py --labeller v4 v5 --variant decision    # real, real_occ, hard, degraded
    python scripts/bench_score.py --labeller v5 --no-tags                # hide the simulated tags

Labellers that take ``tags`` get the bench's simulated set-piece tags (as Ashwyn's score_bench.py).
"R+O" = mean of real and real_occ, the number Ashwyn's v4 decisions were taken on.

A labeller never sees the ground truth. This is the ONLY place versions and settings are compared
(RULES.md, EVALUATION.md). Log every accepted change with its score in docs/changelog/labeller.md.
The bench reads ~0.07 higher than the board on spellF1: compare bench numbers with each other only.
"""
import argparse
import inspect
import os

try:
    import _bootstrap  # noqa: F401  (python scripts/x.py)
except ImportError:
    from scripts import _bootstrap  # noqa: F401  (python -m scripts.x)

import numpy as np

import possession_metrics as pm
from possession.io.bench import ALL_VARIANTS, GAMES, iter_bench
from possession.labellers import get_labeller
from possession.utils.config import load_config, section
from possession.utils.frames import to_codes

ROOT = _bootstrap.ROOT
VARIANTS = ALL_VARIANTS
DECISION = ("real", "real_occ", "hard", "degraded")   # the copies decisions are taken on
CFG = section(load_config(os.path.join(ROOT, "config", "bench.yaml")), "bench")


def _value(text):
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text


def score(bench, name, variant="degraded", games=GAMES, params=None, verbose=True, use_tags=True):
    fn = get_labeller(name)
    takes_tags = "tags" in inspect.signature(fn).parameters
    rows = []
    for g in iter_bench(bench, variant, games):
        kw = {"fps": g.fps, "pitch": g.pitch}
        if takes_tags and use_tags and g.tags is not None:
            kw["tags"] = g.tags
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
    ap.add_argument("--bench", default=os.path.join(ROOT, CFG.get("dir", "outputs/bench")))
    ap.add_argument("--labeller", nargs="+", default=list(CFG.get("labellers", ["v1", "baseline50"])))
    ap.add_argument("--variant", default=CFG.get("variant", "degraded"),
                    choices=VARIANTS + ("all", "decision"))
    ap.add_argument("--no-tags", action="store_true", help="do not give labellers the simulated tags")
    ap.add_argument("--quiet", action="store_true", help="only the mean lines")
    ap.add_argument("--games", nargs="+", default=list(CFG.get("games", GAMES)))
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a labeller parameter (v1 uses its DEFAULTS keys)")
    args = ap.parse_args()
    params = dict((k, _value(v)) for k, v in (s.split("=", 1) for s in args.set)) or None
    variants = VARIANTS if args.variant == "all" else DECISION if args.variant == "decision" \
        else (args.variant,)
    for name in args.labeller:
        means = {}
        for v in variants:
            means[v] = score(args.bench, name, v, args.games, params, verbose=not args.quiet,
                             use_tags=not args.no_tags)[0]
        if "real" in means and "real_occ" in means:
            print("{:<14} R+O (mean of real, real_occ) spellF1 {:.4f}".format(
                name, (means["real"] + means["real_occ"]) / 2))


if __name__ == "__main__":
    main()
