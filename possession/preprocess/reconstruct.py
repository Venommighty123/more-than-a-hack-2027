"""Fill missing ball positions inside a gap: linear, velocity-aware Hermite, quadratic.

From Krish's ``ball_reconstruction.py`` (methods unchanged, rewritten over gap runs and in metres).
His experiments showed what DATA.md and the gap audit suggest: positions are trustworthy for short
gaps and drift fast after a second or two, so ``max_gap`` caps what gets filled. Reconstructed
frames are flagged, never mixed silently with observed ones.

Masking evaluation (``mask_and_score``) hides known stretches of a tracked ball and measures the
error. It uses tracking only -- no labels -- so it may run on any game; prefer the bench games
when the number is used to choose a method or a cap.
"""
from __future__ import annotations

import os

import numpy as np

from .ball import runs

METHODS = ("linear", "hermite", "quadratic")


def _linear(x, y, s, e):
    t = np.arange(s, e)
    return np.interp(t, [s - 1, e], [x[s - 1], x[e]]), np.interp(t, [s - 1, e], [y[s - 1], y[e]])


def _hermite(x, y, s, e, present):
    """Cubic Hermite matching position and per-frame velocity at both ends of the gap."""
    n = len(x)
    dt = e - s + 1
    x0, y0, x1, y1 = x[s - 1], y[s - 1], x[e], y[e]
    lin_vx, lin_vy = (x1 - x0) / dt, (y1 - y0) / dt
    vx0 = x[s - 1] - x[s - 2] if s >= 2 and present[s - 2] else lin_vx
    vy0 = y[s - 1] - y[s - 2] if s >= 2 and present[s - 2] else lin_vy
    vx1 = x[e + 1] - x[e] if e + 1 < n and present[e + 1] else lin_vx
    vy1 = y[e + 1] - y[e] if e + 1 < n and present[e + 1] else lin_vy
    t = (np.arange(s, e) - s + 1) / dt
    h00, h10 = 2 * t ** 3 - 3 * t ** 2 + 1, t ** 3 - 2 * t ** 2 + t
    h01, h11 = -2 * t ** 3 + 3 * t ** 2, t ** 3 - t ** 2
    return (h00 * x0 + h10 * dt * vx0 + h01 * x1 + h11 * dt * vx1,
            h00 * y0 + h10 * dt * vy0 + h01 * y1 + h11 * dt * vy1)


def _quadratic(x, y, s, e, present, context=4):
    """Quadratic fit through up to ``context`` observed points on each side; linear fallback."""
    n = len(x)
    left = [i for i in range(max(0, s - context), s) if present[i]]
    right = [i for i in range(e, min(n, e + context)) if present[i]]
    if len(left) + len(right) < 4 or not left or not right:
        return _linear(x, y, s, e)
    times = np.array(left + right, dtype=float)
    try:
        cx = np.polyfit(times, x[left + right], 2)
        cy = np.polyfit(times, y[left + right], 2)
    except (np.linalg.LinAlgError, ValueError):
        return _linear(x, y, s, e)
    t = np.arange(s, e, dtype=float)
    return np.polyval(cx, t), np.polyval(cy, t)


def reconstruct(x, y, method="hermite", max_gap=None, mask=None, max_speed_ms=None, fps=25):
    """Fill interior gaps of the ball track.

    x, y      (n,) metres, NaN where missing
    method    "linear" | "hermite" | "quadratic"
    max_gap   only fill gaps of at most this many frames (None = all interior gaps)
    mask      optional bool (n,): only fill these frames (default: every NaN frame)
    max_speed_ms  skip a gap whose two ends are further apart than a ball can travel at this speed
              in the time (a re-detection elsewhere, not a dropout); None = no check
    Returns (x, y, filled) where ``filled`` flags the reconstructed frames.
    """
    if method not in METHODS:
        raise ValueError("method must be one of {}".format(METHODS))
    x, y = np.array(x, dtype=np.float64), np.array(y, dtype=np.float64)
    missing = np.isnan(x) | np.isnan(y)
    target = missing if mask is None else (np.asarray(mask, dtype=bool) & missing)
    present = ~missing
    filled = np.zeros(len(x), dtype=bool)
    for s, e in zip(*runs(target)):
        if s == 0 or e >= len(x) or not (present[s - 1] and present[e]):
            continue
        if max_gap is not None and e - s > max_gap:
            continue
        if max_speed_ms is not None and \
                np.hypot(x[e] - x[s - 1], y[e] - y[s - 1]) * fps / (e - s + 1) > max_speed_ms:
            continue
        if method == "linear":
            fx, fy = _linear(x, y, s, e)
        elif method == "hermite":
            fx, fy = _hermite(x, y, s, e, present)
        else:
            fx, fy = _quadratic(x, y, s, e, present)
        x[s:e], y[s:e] = fx, fy
        filled[s:e] = True
    return x, y, filled


def mask_and_score(x, y, gap_lengths=(5, 15, 30, 60, 125, 250), trials=200, methods=METHODS,
                   seed=0, within_m=5.0):
    """Hide ``trials`` random fully-tracked stretches per length and score each method.

    Returns a list of dicts: gap, method, n, rmse_m, mae_m, max_m, within_pct.
    """
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    present = ~(np.isnan(x) | np.isnan(y))
    rows = []
    for g in gap_lengths:
        need = g + 8          # a little context on both sides must be tracked too
        ok = np.convolve(present.astype(int), np.ones(need, dtype=int), "valid") == need
        starts = np.flatnonzero(ok)
        if not len(starts):
            continue
        picks = rng.choice(starts, size=min(trials, len(starts)), replace=False) + 4
        for m in methods:
            err = []
            for s in picks:
                hx, hy = x.copy(), y.copy()
                hx[s:s + g] = hy[s:s + g] = np.nan
                lo, hi = max(0, s - 8), min(len(x), s + g + 8)
                rx, ry, _ = reconstruct(hx[lo:hi], hy[lo:hi], m)
                err.append(np.hypot(rx[s - lo:s - lo + g] - x[s:s + g],
                                    ry[s - lo:s - lo + g] - y[s:s + g]))
            e = np.concatenate(err)
            rows.append({"gap": g, "method": m, "n": int(len(picks)),
                         "rmse_m": float(np.sqrt(np.mean(e ** 2))), "mae_m": float(e.mean()),
                         "max_m": float(e.max()), "within_pct": float(100 * (e <= within_m).mean())})
    return rows


# ------------------------------------------------------------------------------------------------
# The prepared ball: clean-up + reconstruction, shared by the labellers, flights and the file
# written by scripts/reconstruct_ball.py
# ------------------------------------------------------------------------------------------------

DEFAULT_FILE = "outputs/ball/ball_reconstructed.npz"


def prepare_ball(bx, by, method="hermite", max_gap=25, max_step_m=2.0, fps=25, max_speed_ms=35.0):
    """Spikes out (as clean_ball), then gaps of at most ``max_gap`` frames filled with ``method``.

    At 25 fps the old 6-frame bridge is 0.24 s. Krish's masking test (experiments/
    reconstruction_masking.py) found Hermite best up to ~1 s (0.13 m vs 1.11 m linear) and every
    method drifting by 5 s, so the default fills up to 25 frames (1 s) with Hermite. Longer gaps
    stay NaN: they are a modelling decision (dead-ball rules, gap logic), not a position.

    PLAUSIBILITY (``max_speed_ms``, default 35 m/s = 126 km/h): on BAR-ATH 122 of the 129 gaps of
    <= 25 frames join two points the ball could not link in the time (median implied speed 58 m/s,
    up to 650): the tracker re-detected the ball somewhere else, it did not lose it. Filling those
    draws fake 50-1000 m/s kicks (the cause of the 300+ km/h "flights"), so such a gap stays NaN.
    Returns (x, y, observed, removed, filled), observed = what the tracker gave.
    """
    from .ball import remove_spikes
    bx, by = np.asarray(bx, dtype=np.float64), np.asarray(by, dtype=np.float64)
    observed = ~(np.isnan(bx) | np.isnan(by))
    x, y, removed = remove_spikes(bx, by, max_step_m)
    if max_gap and max_gap > 0:
        x, y, filled = reconstruct(x, y, method, max_gap=int(max_gap), max_speed_ms=max_speed_ms,
                                   fps=fps)
    else:
        filled = np.zeros(len(x), dtype=bool)
    return x, y, observed, removed, filled


def save_ball_file(path, x, y, observed, removed, filled, *, method, max_gap, max_step_m, lo=0,
                   source="", max_speed_ms=35.0):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(path, x=x, y=y, observed=observed, removed=removed, filled=filled,
                        method=np.array(method), max_gap=np.array(int(max_gap)),
                        max_step_m=np.array(float(max_step_m)), lo=np.array(int(lo)),
                        max_speed_ms=np.array(float(max_speed_ms if max_speed_ms else 0.0)),
                        source=np.array(source))


def load_ball_file(path):
    """The file written by scripts/reconstruct_ball.py as a dict (x, y, flags, settings)."""
    z = np.load(path, allow_pickle=False)
    out = {k: z[k] for k in ("x", "y", "observed", "removed", "filled")}
    out.update(method=str(z["method"]), max_gap=int(z["max_gap"]), max_step_m=float(z["max_step_m"]),
               max_speed_ms=float(z["max_speed_ms"]) if "max_speed_ms" in z.files else 0.0,
               lo=int(z["lo"]), source=str(z["source"]))
    return out
