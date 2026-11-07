"""The reference baseline: whoever is closest to the ball has it, majority over +/- window frames.

Same rule as starter/baseline.py (the published floor, spellF1 0.2433 with window 50), in the
labeller signature so it runs on the bench. From Ashwyn's score_bench.baseline.
Where nobody is seen within the window the frame is D, as in the starter.
"""
from __future__ import annotations

import numpy as np


def label(px, py, ball, team, blocks=None, params=None, *, window=50, fps=25, pitch=None):
    team = np.asarray(team)
    n = len(ball)
    codes = np.full(n, -1, dtype=np.int8)
    ok = ~np.isnan(ball[:, 0])
    with np.errstate(invalid="ignore"):
        d = np.hypot(px[ok] - ball[ok, 0:1], py[ok] - ball[ok, 1:2])
    d = np.where(np.isnan(d) | (team[None, :] < 0), np.inf, d)
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
    if blocks:
        out = np.full(n, "X", dtype="<U1")
        for _, first, last in blocks:
            out[int(first):int(last) + 1] = lab[int(first):int(last) + 1]
        return out
    return lab
