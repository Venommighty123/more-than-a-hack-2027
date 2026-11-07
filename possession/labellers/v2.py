"""Possession labeller v2 = v1 + hand-overs moved into the ball gaps (detectors/gap_fill.py).

Same signature as every labeller, plus an optional ``tags`` argument (BAR-ATH only):

  1-4  exactly labeller v1 (ball clean-up, control, dead ball, resolve + merge_short)
  5    gap fill: for each live gap after which possession switches, move the switch from the end
       of the gap to the frame chosen by GAP_SPLIT ("end" = v1, "fixed", "evidence")
  6    (USE_TAGS, off by default) throw-in sides inferred (detectors/throw_in_side.py) and the
       Smart Tagging prior (detectors/tag_prior.py) added to the gap evidence

BENCH RESULT (2026-11-07, docs/changelog/labeller.md): moving the switch into the gap did not
help -- spellF1 degraded 0.5226 (v1 / "end") vs 0.5208 ("evidence"), 0.5198 ("fixed" 0.5); hard
0.4577 vs 0.4520; clean 0.6174 vs 0.6167. Half the true hand-overs near a moved switch happen
AFTER the gap ends. So GAP_SPLIT defaults to "end" and v2 == v1 until a variant wins on the bench.

Steps 1-5 are scored on the bench. Step 6 cannot be (the bench has no Smart Tagging), so it stays
off until there is a reason to turn it on that does not come from the public 5 minutes (RULES.md).
"""
from __future__ import annotations

import numpy as np

from possession.detectors.gap_fill import GapFillConfig, fill_gaps, shape_evidence
from possession.detectors.tag_prior import build_prior
from possession.detectors.throw_in_side import fill_tag_teams, infer_throw_ins
from possession.labellers import v1
from possession.preprocess.players import attack_signs, velocity
from possession.utils.frames import to_codes

DEFAULTS = dict(v1.DEFAULTS, GAP_SPLIT="end", GAP_FRAC=0.5, GAP_MIN_S=1.0,
                GAP_CONTEXT_S=5.0, GAP_SMOOTH_S=1.0, GAP_MIN_EVIDENCE=0.5,
                USE_TAGS=False, TAG_WEIGHT=1.0)
LETTERS = np.array(["H", "A", "D"])


def gap_config(p) -> GapFillConfig:
    return GapFillConfig(split=str(p["GAP_SPLIT"]), default_frac=float(p["GAP_FRAC"]),
                         min_gap_s=float(p["GAP_MIN_S"]), context_s=float(p["GAP_CONTEXT_S"]),
                         smooth_s=float(p["GAP_SMOOTH_S"]),
                         min_evidence=float(p["GAP_MIN_EVIDENCE"]),
                         tag_weight=float(p["TAG_WEIGHT"]))


def run(px, py, ball, team, blocks, params=None, *, fps=25, pitch=(105.0, 68.0), tags=None):
    p = dict(DEFAULTS, **(params or {}))
    v1_params = {k: v for k, v in p.items() if k in v1.DEFAULTS}
    base = v1.run(px, py, ball, team, blocks, v1_params, fps=fps, pitch=pitch)
    team = np.asarray(team)
    track = base["ball"]
    cfg = gap_config(p)

    evidence = None
    if cfg.split == "evidence":
        vx, _, _ = velocity(px, py, fps, half=12, max_speed_ms=12.0)
        signs = attack_signs(px, team, blocks)
        evidence = shape_evidence(vx, team, signs, blocks, fps=fps, smooth_s=cfg.smooth_s)

    tag_score, throw_ins = None, []
    if p["USE_TAGS"] and tags:
        signs = attack_signs(px, team, blocks)
        throw_ins = infer_throw_ins(tags, px, py, track.x, track.y, team, signs,
                                    dead=base["dead"], pitch=pitch)
        tag_score = build_prior(fill_tag_teams(tags, throw_ins), len(ball)).score

    labels = base["labels"].copy()
    moved = []
    for _, lo, hi in blocks:
        lo, hi = int(lo), int(hi)
        codes = to_codes(labels[lo:hi + 1])
        codes, m = fill_gaps(codes, track.present, lo, hi, evidence=evidence,
                             tag_score=tag_score, cfg=cfg, fps=fps)
        labels[lo:hi + 1] = LETTERS[codes]
        moved += m
    base.update(labels=labels, moved=moved, evidence=evidence, tag_score=tag_score,
                throw_ins=throw_ins)
    return base


def label(px, py, ball, team, blocks, params=None, *, fps=25, pitch=(105.0, 68.0), tags=None):
    return run(px, py, ball, team, blocks, params, fps=fps, pitch=pitch, tags=tags)["labels"]
