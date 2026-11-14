"""Labeller v7 on bench v2: every change alone on top of v6, every change left out of v7, both truths.

    python experiments/v7_ablation.py                 # real + real_occ, games g1-g3 (~8 min)
    python experiments/v7_ablation.py --configs v6 v7 --variants real

Truths: EV (bench event truth) and TRK (tracking-based, possession/io/trk_truth.py). Each cell is the
mean spellF1 over the games; R+O = mean of real and real_occ; BOTH = mean of EV R+O and TRK R+O
(Ashwyn's headline number). Acceptance rule (team): a change is kept only if it loses on neither
truth and neither copy. Events: the same detector as BAR-ATH, run on each bench copy
(possession/events/from_arrays.py, cached in outputs/bench_events/).
Writes outputs/bench_results/v7_ablation.json.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

import possession_metrics as pm  # noqa: E402
from possession.events.from_arrays import cached_events  # noqa: E402
from possession.io.bench import load_bench_game  # noqa: E402
from possession.io.trk_truth import trk_truth  # noqa: E402
from possession.labellers import v6, v7  # noqa: E402

BENCH = os.path.join(ROOT, "outputs", "bench")
SWITCHES = ("BALL", "FLIGHTS", "PASS_SHIELD", "ANCHOR", "DUEL_GUARD", "HOLD_FILL", "RESTART")


def configs():
    c = {"v6": dict(v7.AS_V6), "v7": {}}
    for s in SWITCHES:
        on = dict(v7.AS_V6)
        on[s] = v7.DEFAULTS[s]
        c["v6+" + s] = on
        c["v7-" + s] = {s: v7.AS_V6[s]}
    return c


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--configs", nargs="*", default=None)
    ap.add_argument("--variants", nargs="*", default=["real", "real_occ"])
    ap.add_argument("--games", nargs="*", default=["g1", "g2", "g3"])
    ap.add_argument("--fresh", action="store_true", help="ignore cached scores (after changing v7)")
    args = ap.parse_args()
    cfgs = configs()
    names = args.configs or list(cfgs)
    # per (config, game, copy) scores are cached, so an interrupted run resumes where it stopped
    cache_path = os.path.join(ROOT, "outputs", "bench_results", "v7_ablation_cache.json")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    cache = json.load(open(cache_path)) if os.path.isfile(cache_path) and not args.fresh else {}
    res = {k: {} for k in names}
    t0 = time.time()
    for v in args.variants:
        for g in args.games:
            todo = [k for k in names if "{}|{}|{}".format(k, g, v) not in cache]
            if todo:
                G = load_bench_game(BENCH, g, v)
                ev = cached_events(G)
                truths = {"EV": pm.label_codes(G.gt), "TRK": pm.label_codes(trk_truth(BENCH, g))}
                for k in todo:
                    lab = v7.label(G.px, G.py, G.ball, G.team, G.blocks, params=cfgs[k], tags=G.tags,
                                   events=ev)
                    pred = pm.label_codes(lab)
                    cache["{}|{}|{}".format(k, g, v)] = {
                        tn: pm.evaluate(gt, pred, bootstrap_n=0)["spell_f1_mean"] for tn, gt in truths.items()}
                    with open(cache_path, "w") as handle:
                        json.dump(cache, handle)
                print("{} {} done ({:.0f} s)".format(g, v, time.time() - t0), flush=True)
            for k in names:
                for tn, val in cache["{}|{}|{}".format(k, g, v)].items():
                    res[k].setdefault("{}|{}".format(tn, v), []).append(val)
    rows = {}
    for k in names:
        m = {c: float(np.mean(x)) for c, x in res[k].items()}
        for tn in ("EV", "TRK"):
            vals = [m.get("{}|{}".format(tn, v)) for v in ("real", "real_occ")]
            if None not in vals:
                m["{}|R+O".format(tn)] = float(np.mean(vals))
        if "EV|R+O" in m and "TRK|R+O" in m:
            m["BOTH"] = (m["EV|R+O"] + m["TRK|R+O"]) / 2
        m["games"] = res[k]
        rows[k] = m
    base = rows.get("v6")
    cols = [c for c in ("EV|real", "EV|real_occ", "TRK|real", "TRK|real_occ", "BOTH") if c in rows[names[0]]]
    print("\n{:<16}".format("config") + "".join("{:>13}".format(c) for c in cols) + "   worst cell vs v6")
    for k in names:
        line = "{:<16}".format(k) + "".join("{:>13.4f}".format(rows[k][c]) for c in cols)
        if base and k != "v6":
            diffs = [a - b for c in rows[k]["games"] for a, b in zip(rows[k]["games"][c], base["games"][c])]
            line += "   {:+.4f}".format(min(diffs))
        print(line)
    os.makedirs(os.path.join(ROOT, "outputs", "bench_results"), exist_ok=True)
    with open(os.path.join(ROOT, "outputs", "bench_results", "v7_ablation.json"), "w") as handle:
        json.dump(rows, handle, indent=1)


if __name__ == "__main__":
    main()
