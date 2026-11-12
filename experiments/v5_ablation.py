"""EXPERIMENT: what each v5 change does on the bench, alone and together (spellF1, games 1-3).

    python experiments/v5_ablation.py                       # all configurations, decision copies
    python experiments/v5_ablation.py --only v4 v5          # a subset

The copies are those the v4 decisions were taken on: real, real_occ (R+O = their mean), hard and
degraded. Labellers get the simulated set-piece tags. Results: docs/changelog/labeller.md.
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

import numpy as np  # noqa: E402

import possession.io  # noqa: E402,F401
import possession_metrics as pm  # noqa: E402
from possession.io.bench import iter_bench  # noqa: E402
from possession.labellers import v5  # noqa: E402
from possession.utils.frames import to_codes  # noqa: E402

OFF = dict(FILL_METHOD="linear", FILL_MAX=6, FILL_MAX_SPEED=None, ZOI_MODE="off", CONFIRM=6,
           USE_FLIGHTS=False)
CONFIGS = {
    "v4": OFF,
    "+fill": dict(OFF, FILL_METHOD="hermite", FILL_MAX=25, FILL_MAX_SPEED=35.0),
    "+fill_nogate": dict(OFF, FILL_METHOD="hermite", FILL_MAX=25, FILL_MAX_SPEED=None),
    "+flights_all": dict(OFF, USE_FLIGHTS=True, FLIGHT_MODE="all"),
    "+flights_opp": dict(OFF, USE_FLIGHTS=True, FLIGHT_MODE="opponent"),
    "+zoi_contest": dict(OFF, ZOI_MODE="contest"),
    "+zoi_contest_r15": dict(OFF, ZOI_MODE="contest", ZOI_R0=1.5),
    "+zoi_contest_v6": dict(OFF, ZOI_MODE="contest", ZOI_REL_V=6.0),
    "+zoi_veto": dict(OFF, ZOI_MODE="veto"),
    "+zoi_veto_r15": dict(OFF, ZOI_MODE="veto", ZOI_R0=1.5),
    "+zoi_confirm": dict(OFF, ZOI_MODE="confirm"),
    "+confirm15": dict(OFF, CONFIRM=15),
    "+confirm15_hole6": dict(OFF, CONFIRM=15, HOLE=6),
    "+confirm8": dict(OFF, CONFIRM=8),
    "+confirm10": dict(OFF, CONFIRM=10),
    "+zoi_veto_r12": dict(OFF, ZOI_MODE="veto", ZOI_R0=1.2),
    "+zoi_veto_r10": dict(OFF, ZOI_MODE="veto", ZOI_R0=1.0),
    "v5": {},
    "v5_zoi_confirm": dict(ZOI_MODE="confirm"),
    "v5_confirm6": dict(CONFIRM=6),
    "v5_confirm8": dict(CONFIRM=8),
    "v5_no_flights": dict(USE_FLIGHTS=False),
    "v5_no_zoi": dict(ZOI_MODE="off"),
}
VARIANTS = ("real", "real_occ", "hard", "degraded")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bench", default=os.path.join(ROOT, "outputs", "bench"))
    ap.add_argument("--only", nargs="+", default=list(CONFIGS))
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    args = ap.parse_args()
    games = {v: list(iter_bench(args.bench, v)) for v in args.variants}
    print("{:<18}".format("config") + "".join("{:>10}".format(v) for v in args.variants) +
          "{:>8}  spells(real)".format("R+O"))
    for name in args.only:
        t0 = time.time()
        means, spells = {}, 0
        for v in args.variants:
            s = []
            for g in games[v]:
                pred = v5.label(g.px, g.py, g.ball, g.team, g.blocks, CONFIGS[name], fps=g.fps,
                                pitch=g.pitch, tags=g.tags)
                r = pm.evaluate(to_codes(g.gt), to_codes(pred))
                s.append(r["spell_f1_mean"])
                if v == "real":
                    spells += r["n_pred_spells"]
            means[v] = float(np.mean(s))
        ro = (means.get("real", np.nan) + means.get("real_occ", np.nan)) / 2
        print("{:<18}".format(name) + "".join("{:>10.4f}".format(means[v]) for v in args.variants)
              + "{:>8.4f}  {}  ({:.0f} s)".format(ro, spells, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
