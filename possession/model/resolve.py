"""Turn per-frame control and dead-ball masks into possessions (H / A / D codes).

Ported from Ashwyn's labeller_v1.resolve and merge_short (logic unchanged). It answers the rules of
CHALLENGE.md:

  * possession changes only on CONTROLLED possession: the other team must control the ball for
    ``confirm`` frames (gaps of nobody up to ``hole`` frames allowed). A touch, a deflection or a
    won header without control never reaches ``confirm``, so possession stays with the last team;
  * the switch is BACKDATED to the first frame of that control, because spellF1 punishes lateness
    (1 s late costs about a third of a spell's score);
  * after a stoppage the possessor is the first team to control the ball ("whenever play restarts
    there is a clear possessor"); the frames before it are given to that team;
  * a live stretch nobody ever controls copies the nearest known neighbour;
  * ``merge_short``: a team spell shorter than ``min_spell`` frames sitting between two spells of
    the other team is merged into them (a short spurious hand-over costs two split possessions).

Codes are those of evaluation/possession_metrics.py: H = 0, A = 1, D = 2.
``model/hmm.py`` is the planned alternative; both are compared on the bench.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from possession.utils.frames import A, D, H  # noqa: F401  (0, 1, 2)

NONE = -1


@dataclass
class ResolveConfig:
    confirm: int = 6          # frames of the new team's control needed to switch
    hole: int = 3             # frames of nobody tolerated inside that control
    min_spell_s: float = 2.0  # merge_short threshold

    @classmethod
    def from_dict(cls, d: dict | None) -> "ResolveConfig":
        d = d or {}
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict:
        return asdict(self)


def resolve(ctrl, dead, lo, hi, cfg: ResolveConfig | None = None) -> np.ndarray:
    """State machine over rows lo..hi (inclusive). Returns int8 codes H/A/D of length hi-lo+1.

    ctrl  (n,) HOME 0 / AWAY 1 / anything else = nobody (contests should be passed as nobody)
    dead  (n,) bool
    """
    cfg = cfg or ResolveConfig()
    out = np.full(hi - lo + 1, NONE, dtype=np.int8)
    cur = NONE                      # team in possession, NONE = unknown (start / after stoppage)
    cand, cand_start, cand_len, holes = NONE, 0, 0, 0
    pending = None                  # start of the current possession whose team is not known yet
    for k in range(lo, hi + 1):
        i = k - lo
        if dead[k]:
            out[i] = D
            cur, cand, cand_len, holes = NONE, NONE, 0, 0
            continue
        c = ctrl[k]
        if c in (H, A):
            if c == cand:
                cand_len += 1 + holes
                holes = 0
            else:
                cand, cand_start, cand_len, holes = c, i, 1, 0
        elif cand >= 0:
            holes += 1
            if holes > cfg.hole:
                cand, cand_len, holes = NONE, 0, 0
        if cur == NONE:
            if pending is None:
                pending = i
            if cand >= 0 and cand_len >= 1:
                cur = cand
                out[pending:i + 1] = cur
                pending = None
                continue
            out[i] = NONE
            continue
        pending = None
        if cand >= 0 and cand != cur and cand_len >= cfg.confirm:
            cur = cand
            out[cand_start:i + 1] = cur          # backdate to the first frame of control
            continue
        out[i] = cur
    unk = out == NONE
    if unk.any() and (~unk).any():
        idx = np.flatnonzero(~unk)
        j = np.clip(np.searchsorted(idx, np.flatnonzero(unk)), 0, len(idx) - 1)
        out[unk] = out[idx[j]]
    out[out == NONE] = D
    return out


def merge_short(codes, min_len: int) -> np.ndarray:
    """Merge a team spell shorter than ``min_len`` frames between two spells of the other team."""
    c = np.array(codes, copy=True)
    changed = True
    while changed:
        changed = False
        edges = np.flatnonzero(np.diff(c)) + 1
        starts = np.r_[0, edges]
        ends = np.r_[edges, len(c)]
        for j in range(1, len(starts) - 1):
            s, e = starts[j], ends[j]
            prev, nxt = c[starts[j - 1]], c[starts[j + 1]]
            if c[s] < D and e - s < min_len and prev == nxt and prev < D and prev != c[s]:
                c[s:e] = prev
                changed = True
                break
    return c


def resolve_block(ctrl, dead, lo, hi, cfg: ResolveConfig | None = None, fps: int = 25):
    """``resolve`` then ``merge_short`` -- what a labeller calls per graded block."""
    cfg = cfg or ResolveConfig()
    return merge_short(resolve(ctrl, dead, lo, hi, cfg), int(cfg.min_spell_s * fps))
