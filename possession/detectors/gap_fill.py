"""WHERE inside a live ball gap possession changes hands -- the 41 % blind spot.

The resolver (model/resolve.py) holds the last team through a gap and only switches once the
other team is seen controlling the ball, so every hand-over that happens in the dark is written
LATE, at the end of the gap. spellF1 punishes lateness hard (1 s late ~ a third of a spell). The
team either side of the gap is reliable; only the change frame is unknown. This module moves it.

For each live gap of at least ``min_gap_s`` (no dead frame inside) after which the resolved
label switches from team P to team Q within ``context_s``, the change frame k is chosen inside the
gap by one of:

  split="end"       keep the resolver's choice (= labeller v1)
  split="fixed"     k = start + default_frac * length
  split="evidence"  k minimises the cumulative evidence for P before k / Q after k; the evidence
                    is the team-shape signal below (+ the Smart Tagging prior when given). Falls
                    back to ``fixed`` when the evidence is weaker than ``min_evidence``.

Shape signal ("who is pushing"): the team in possession moves towards the opponent's goal. Per
frame, adv_side = mean player velocity along that team's attacking direction; the evidence for a
HOME possession is adv_home - adv_away, smoothed over ``smooth_s``. The game plan warns shape lags a
turnover by ~1 s: it places the change roughly, it never decides the teams.

Everything here except the tag prior runs on the bench, so ``split``, ``default_frac`` and the
evidence knobs are tuned there (scripts/bench_score.py --labeller v2 --set GAP_SPLIT=...).
"""
from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass

import numpy as np

from possession.preprocess.players import attack_sign_per_frame
from possession.utils.frames import A, D, H

HOME, AWAY = 0, 1


@dataclass
class GapFillConfig:
    split: str = "end"             # "end" | "fixed" | "evidence" (end won on the bench, see v2)
    default_frac: float = 0.5
    min_gap_s: float = 1.0
    context_s: float = 5.0         # the switch must happen within this after the gap
    smooth_s: float = 1.0
    min_evidence: float = 0.5      # m/s * s of separation needed to trust the evidence
    tag_weight: float = 1.0        # m/s-equivalent per unit of tag score (BAR-ATH only)

    @classmethod
    def from_dict(cls, d: dict | None) -> "GapFillConfig":
        d = d or {}
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict:
        return asdict(self)


def shape_evidence(vx, team, signs, blocks, lo=0, fps=25, smooth_s=1.0):
    """(n,) m/s: + when HOME is the team pushing towards the opponent goal, - for AWAY."""
    n = len(vx)
    team = np.asarray(team)
    adv = {}
    for side in (HOME, AWAY):
        sign = attack_sign_per_frame(signs, blocks, n, side, lo).astype(np.float64)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            v = np.nanmean(vx[:, team == side], axis=1)
        adv[side] = np.nan_to_num(v) * sign
    e = adv[HOME] - adv[AWAY]
    w = max(1, int(round(smooth_s * fps)))
    return np.convolve(e, np.ones(w) / w, mode="same")


def _best_split(e_sign, frac, min_evidence, fps):
    """Index k in [0, L] minimising sum(e[:k]) - sum(e[k:]) relative to the fixed split."""
    L = len(e_sign)
    c = np.concatenate(([0.0], np.cumsum(e_sign)))
    # score(k) = (sum after k) - (sum before k) = total - 2 c[k]; maximise
    score = c[-1] - 2.0 * c
    k = int(np.argmax(score))
    k_fixed = int(round(frac * L))
    gain = (score[k] - score[k_fixed]) / float(fps)
    return k if gain >= min_evidence else k_fixed


def fill_gaps(codes, present, block_lo, block_hi, *, evidence=None, tag_score=None,
              cfg: GapFillConfig | None = None, fps: int = 25):
    """Move hand-overs into the gaps of one block. ``codes`` covers rows block_lo..block_hi
    (H / A / D, as returned by model/resolve.py); ``present``, ``evidence`` and ``tag_score`` are
    full-length row arrays. Returns (new codes, list of moved changes)."""
    cfg = cfg or GapFillConfig()
    codes = np.array(codes, copy=True)
    if cfg.split == "end":
        return codes, []
    seg_present = np.asarray(present[block_lo:block_hi + 1], dtype=bool)
    n = len(codes)
    min_len, ctx = int(cfg.min_gap_s * fps), int(cfg.context_s * fps)
    e = np.zeros(len(present)) if evidence is None else np.asarray(evidence, dtype=np.float64)
    if tag_score is not None:
        e = e + cfg.tag_weight * np.asarray(tag_score, dtype=np.float64)
    e = e[block_lo:block_hi + 1]
    edges = np.diff(np.concatenate(([0], (~seg_present).astype(np.int8), [0])))
    moved = []
    for s, t in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if t - s < min_len or s == 0 or t >= n:
            continue
        before = codes[s - 1]
        if before not in (H, A) or (codes[s:t] == D).any():
            continue
        window = codes[t:min(n, t + ctx)]
        diff = np.flatnonzero(window != before)
        if not len(diff):
            continue
        c = t + int(diff[0])
        after = codes[c]
        if after not in (H, A) or (codes[s:c] == D).any():
            continue
        if cfg.split == "fixed":
            k = s + int(round(cfg.default_frac * (t - s)))
        else:
            sign = 1.0 if after == H else -1.0          # evidence oriented towards "after"
            k = s + _best_split(sign * e[s:t], cfg.default_frac, cfg.min_evidence, fps)
        if k < c:
            codes[k:c] = after
            moved.append({"gap_start": s + block_lo, "gap_end": t + block_lo,
                          "old_change": c + block_lo, "new_change": k + block_lo,
                          "team": "HA"[after]})
    return codes, moved
