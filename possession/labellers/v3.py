"""Possession labeller v3: labeller v1's signals decoded by the HMM (model/hmm.py) instead of the
state machine -- and, optionally, a bench-trained classifier as the evidence.

  MODE="rules"       emissions from v1's per-frame control + dead-ball rule (no training)
  MODE="classifier"  emissions from model/classifier.py probabilities (features/build.py), with the
                     rule emissions added at RULE_WEIGHT; the model file comes from
                     possession.model.classifier.train_on_bench (scripts/train_classifier.py) (outputs/models/classifier.npz)

A spell shorter than MIN_SPELL s between two spells of the other team is merged afterwards, as in
v1. Every knob is tuned on the bench only.
"""
from __future__ import annotations

import os

import numpy as np

from possession.labellers import v1
from possession.model.hmm import (HmmConfig, decode_block, emissions_from_proba,
                                  emissions_from_rules)
from possession.model.resolve import merge_short

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(ROOT, "outputs", "models", "classifier.npz")
LETTERS = np.array(["H", "A", "D"])

DEFAULTS = dict(v1.DEFAULTS, MODE="rules", SWITCH=12.0, TO_DEAD=6.0, FROM_DEAD=6.0,
                P_OWN=0.25, P_OTHER=0.03, P_DEAD_CTRL=0.02, Q_DEAD=0.9, Q_ALIVE=0.02,
                RULE_WEIGHT=1.0, PROBA_WEIGHT=0.5, MODEL=MODEL_PATH)


def hmm_config(p) -> HmmConfig:
    return HmmConfig(switch=p["SWITCH"], to_dead=p["TO_DEAD"], from_dead=p["FROM_DEAD"],
                     p_own=p["P_OWN"], p_other=p["P_OTHER"], p_dead_ctrl=p["P_DEAD_CTRL"],
                     q_dead=p["Q_DEAD"], q_alive=p["Q_ALIVE"])


def run(px, py, ball, team, blocks, params=None, *, fps=25, pitch=(105.0, 68.0), proba=None):
    """``proba`` (n, 3) may be passed directly (e.g. out-of-fold predictions on the bench)."""
    p = dict(DEFAULTS, **(params or {}))
    v1_params = {k: v for k, v in p.items() if k in v1.DEFAULTS}
    base = v1.run(px, py, ball, team, blocks, v1_params, fps=fps, pitch=pitch)
    cfg = hmm_config(p)
    ctrl = base["control"].ctrl
    em = p["RULE_WEIGHT"] * emissions_from_rules(ctrl, base["dead"], cfg)
    if p["MODE"] == "classifier":
        if proba is None:
            from possession.features.build import build_features
            from possession.model.classifier import Classifier, design
            clf = Classifier.load(p["MODEL"])
            table = build_features(px, py, ball, team, blocks, fps=fps, pitch=pitch)
            proba = clf.predict_proba(design(table, clf.cfg, fps))
            prior = clf.prior
        else:
            prior = None
        em = em + emissions_from_proba(proba, prior, p["PROBA_WEIGHT"])
    labels = np.full(len(ball), "X", dtype="<U1")
    for _, lo, hi in blocks:
        lo, hi = int(lo), int(hi)
        codes = decode_block(em, lo, hi, cfg)
        codes = merge_short(codes, int(p["MIN_SPELL"] * fps))
        labels[lo:hi + 1] = LETTERS[codes]
    base.update(labels=labels, emissions=em, proba=proba)
    return base


def label(px, py, ball, team, blocks, params=None, *, fps=25, pitch=(105.0, 68.0)):
    return run(px, py, ball, team, blocks, params, fps=fps, pitch=pitch)["labels"]


def label_classifier(px, py, ball, team, blocks, params=None, *, fps=25, pitch=(105.0, 68.0)):
    """v3 with MODE="classifier" (registered as "v3c")."""
    return label(px, py, ball, team, blocks, dict({"MODE": "classifier"}, **(params or {})),
                 fps=fps, pitch=pitch)
