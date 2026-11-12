"""Tracking-based bench truth (TRK), Ashwyn's test_scripts/bench/build_trk_gt.py (backup/ashwyn_build_trk_gt.py).

Like the organisers' (corrected tracking): on live frames, the team of the player the ball is on in
the CLEAN Metrica tracking (ball within TRK_R m of a player, the other team at least TRK_M m further,
for TRK_RUN frames in a row); flights, loose and contested balls keep the last team; after a
stoppage the first team in control owns the frames back to the restart. D / X come from the event
truth. Ashwyn checked that TRK reproduces the board's baseline window ratios best (0.81 / 0.93).
Logic unchanged; ``trk_truth`` caches it as outputs/bench/<game>_gttrk.npy.
"""
from __future__ import annotations

import os

import numpy as np

TRK_R, TRK_M, TRK_RUN = 0.5, 0.5, 3


def build(ball, px, py, team, gt):
    b = ball
    team = np.asarray(team)
    n = len(b)
    ok = ~np.isnan(b[:, 0])
    c = np.full(n, -1, np.int8)
    dd = np.hypot(px[ok] - b[ok, 0:1], py[ok] - b[ok, 1:2])
    dd = np.where(np.isnan(dd), np.inf, dd)
    dh = dd[:, team == 0].min(1)
    da = dd[:, team == 1].min(1)
    cc = np.full(ok.sum(), -1, np.int8)
    cc[(dh < TRK_R) & (da > dh + TRK_M)] = 0
    cc[(da < TRK_R) & (dh > da + TRK_M)] = 1
    c[ok] = cc
    out = gt.copy()
    live = np.isin(gt, ["H", "A"])
    e = np.diff(np.r_[0, live.astype(int), 0])
    for s, t in zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)):
        cur, run, prev, pend = -1, 0, -1, s
        seg = np.full(t - s, -1, np.int8)
        for i in range(s, t):
            x = c[i]
            run = run + 1 if (x >= 0 and x == prev) else (1 if x >= 0 else 0)
            prev = x
            if x >= 0 and run >= TRK_RUN and x != cur:
                start = i - TRK_RUN + 1
                if cur == -1:
                    seg[pend - s:start - s] = x
                cur = x
                seg[start - s:i - s + 1] = x
                continue
            seg[i - s] = cur
        if cur == -1:
            seg[:] = np.where(gt[s:t] == "H", 0, 1)
        seg[seg == -1] = np.where(gt[s:t][seg == -1] == "H", 0, 1)
        out[s:t] = np.array(["H", "A"])[seg]
    return out


def trk_truth(bench_dir, name):
    """TRK truth of bench game ``name`` (from its clean copy), cached next to the bench files."""
    path = os.path.join(bench_dir, "{}_gttrk.npy".format(name))
    if os.path.isfile(path):
        return np.load(path)
    from possession.io.bench import load_bench_game
    g = load_bench_game(bench_dir, name, "clean")
    t = build(np.asarray(g.ball, dtype=float), g.px, g.py, g.team, g.gt)
    np.save(path, t)
    return t
