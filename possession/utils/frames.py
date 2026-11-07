"""Frame, run and label helpers used everywhere.

Merges the copies of ``clock``, ``runs_of``/``_runs``, ``r2`` and ``half_of`` that lived in
event_detection.py, possession_test_1.py, labeller_v1.py, pass_detection.py, visualize_tracking.py
and the gap scripts.

Label codes follow evaluation/possession_metrics.py: the letters are the submission format,
the integer codes are what numpy works with.
"""
from __future__ import annotations

import numpy as np

FPS = 25

# Integer codes, identical to evaluation/possession_metrics.py so arrays go straight to the scorer.
# H and A equal the team sides HOME = 0 / AWAY = 1 used everywhere in possession/.
H, A, D, X = 0, 1, 2, 3
LABELS = {H: "H", A: "A", D: "D", X: "X"}
CODE = {v: k for k, v in LABELS.items()}


# ------------------------------------------------------------------------------------------------
# Runs
# ------------------------------------------------------------------------------------------------

def runs(mask):
    """(starts, ends) arrays of the True runs of ``mask``; ends exclusive."""
    e = np.diff(np.concatenate(([0], np.asarray(mask, dtype=np.int8), [0])))
    return np.flatnonzero(e == 1), np.flatnonzero(e == -1)


def runs_of(mask):
    """Same as ``runs`` as a list of (start, end) pairs."""
    s, e = runs(mask)
    return list(zip(s.tolist(), e.tolist()))


def label_runs(values, lo=0):
    """Constant runs of any 1-D array as (value, start, end) with end exclusive, offset by ``lo``.

    Works on letter arrays ("H", "A", ...) and integer codes alike; one row per possession spell.
    """
    v = np.asarray(values)
    if not len(v):
        return []
    cut = np.flatnonzero(v[1:] != v[:-1]) + 1
    starts = np.r_[0, cut]
    ends = np.r_[cut, len(v)]
    return [(v[s].item(), int(s + lo), int(e + lo)) for s, e in zip(starts, ends)]


# ------------------------------------------------------------------------------------------------
# Labels
# ------------------------------------------------------------------------------------------------

def to_letters(codes) -> np.ndarray:
    """Integer codes -> "H" / "A" / "D" / "X" (anything unknown becomes "X")."""
    codes = np.asarray(codes)
    out = np.full(codes.shape, "X", dtype="<U1")
    for k, letter in LABELS.items():
        out[codes == k] = letter
    return out


def to_codes(letters) -> np.ndarray:
    """"H" / "A" / "D" / "X" -> integer codes (anything unknown becomes X)."""
    letters = np.asarray(letters)
    out = np.full(letters.shape, X, dtype=np.int8)
    for letter, k in CODE.items():
        out[letters == letter] = k
    return out


# ------------------------------------------------------------------------------------------------
# Time
# ------------------------------------------------------------------------------------------------

def seconds_to_frames(seconds, fps=FPS) -> int:
    return int(round(float(seconds) * fps))


def clock(frame, fps=FPS) -> str:
    """Video time "mm:ss.ss" of a video frame (the tactical video's own clock, not match time)."""
    s = frame / float(fps)
    return "{:02d}:{:05.2f}".format(int(s // 60), s % 60)


def half_of(frame, blocks):
    """Half (1 / 2) that a video frame falls in, or None. ``blocks`` = [(half, first, last)]."""
    for half, first, last in blocks:
        if first <= frame <= last:
            return half
    return None


def r2(v):
    """Round to 2 decimals for JSON; None for None / NaN / inf."""
    if v is None:
        return None
    v = float(v)
    return round(v, 2) if np.isfinite(v) else None
