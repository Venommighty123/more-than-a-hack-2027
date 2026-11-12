"""Labellers: complete recipes from tracking to one H / A / D / X letter per frame.

Every labeller has the same signature, so the bench scores any of them and scripts/run.py runs any
of them on BAR-ATH:

    label(px, py, ball, team, blocks, params=None, *, fps=25, pitch=(105.0, 68.0)) -> (n,) "<U1"

    px, py   (frames, slots) players in metres, NaN when absent
    ball     (frames, 2) ball in metres, NaN when not tracked
    team     (slots,) 0 home / 1 away (-1 unknown slots are ignored)
    blocks   [(period, first_row, last_row)] graded stretches; everything outside is "X"

A labeller never sees ground truth.
"""
from __future__ import annotations

import importlib

from . import baseline, v1, v2, v3, v4, v5, v6, v7

LABELLERS = {
    "baseline0": lambda *a, **k: baseline.label(*a, window=0, **k),
    "baseline25": lambda *a, **k: baseline.label(*a, window=25, **k),
    "baseline50": lambda *a, **k: baseline.label(*a, window=50, **k),
    "baseline125": lambda *a, **k: baseline.label(*a, window=125, **k),
    "v1": v1.label,
    "v2": v2.label,
    "v3": v3.label,
    "v3c": v3.label_classifier,
    "v4": v4.label,
    "v5": v5.label,
    "v6": v6.label,
    "v7": v7.label,
}


def get_labeller(name: str):
    """A registered name above, or 'package.module' exposing ``label``."""
    if name in LABELLERS:
        return LABELLERS[name]
    return importlib.import_module(name).label
