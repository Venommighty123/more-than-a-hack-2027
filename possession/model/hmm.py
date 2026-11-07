"""Hidden-state model over three states (H, A, D) decoded with Viterbi -- the alternative to the
hand-written state machine in model/resolve.py (game plan idea 7).

Each frame's evidence votes for a state (log-emission); every change of state costs a penalty;
Viterbi picks the single most likely sequence over the whole block. Because it looks at the whole
block at once, a hand-over lands at the most likely frame without any explicit backdating, and one
penalty knob (``switch``) sets how many possessions come out.

    emissions = emissions_from_rules(ctrl, dead, cfg)          # or from a classifier
    codes = viterbi(emissions, transition_matrix(cfg))         # H = 0, A = 1, D = 2

Emission model for the rule signals (labeller v1's control and dead-ball masks):
  ctrl   P(ctrl = own team | state) = p_own, P(ctrl = other team | state) = p_other, rest = nobody;
         in D: P(ctrl = a team) = p_dead_ctrl each
  dead   P(dead rule fires | D) = q_dead, P(fires | alive) = q_alive
Penalties are in nats. Untuned defaults; tune on the bench.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

H, A, D = 0, 1, 2


@dataclass
class HmmConfig:
    switch: float = 12.0          # cost of an H <-> A change
    to_dead: float = 6.0          # cost of entering D
    from_dead: float = 6.0        # cost of leaving D
    p_own: float = 0.25           # P(control by the possessing team) per frame while alive
    p_other: float = 0.03         # P(control by the other team) per frame while alive
    p_dead_ctrl: float = 0.02     # P(control by a given team) per frame while dead
    q_dead: float = 0.9           # P(dead rule fires | D)
    q_alive: float = 0.02         # P(dead rule fires | H or A)
    evidence_weight: float = 1.0  # multiplier on any extra soft evidence

    @classmethod
    def from_dict(cls, d: dict | None) -> "HmmConfig":
        d = d or {}
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict:
        return asdict(self)


def transition_matrix(cfg: HmmConfig | None = None) -> np.ndarray:
    """(3, 3) log-transition scores, rows = from, columns = to (0 = stay)."""
    cfg = cfg or HmmConfig()
    t = np.zeros((3, 3))
    t[H, A] = t[A, H] = -cfg.switch
    t[H, D] = t[A, D] = -cfg.to_dead
    t[D, H] = t[D, A] = -cfg.from_dead
    return t


def emissions_from_rules(ctrl, dead, cfg: HmmConfig | None = None, extra=None) -> np.ndarray:
    """(n, 3) log-emissions from per-frame control (H / A / other) and the dead-ball rule.

    extra: optional (n,) soft score, + favours H and - favours A (e.g. a tag prior), added as
    +/- evidence_weight * extra / 2 to the H / A columns.
    """
    cfg = cfg or HmmConfig()
    ctrl = np.asarray(ctrl)
    dead = np.asarray(dead, dtype=bool)
    n = len(ctrl)
    e = np.zeros((n, 3))
    none_alive = np.log(max(1e-9, 1.0 - cfg.p_own - cfg.p_other))
    none_dead = np.log(max(1e-9, 1.0 - 2 * cfg.p_dead_ctrl))
    is_h, is_a = ctrl == H, ctrl == A
    nobody = ~(is_h | is_a)
    e[:, H] = np.where(is_h, np.log(cfg.p_own), np.where(is_a, np.log(cfg.p_other), none_alive))
    e[:, A] = np.where(is_a, np.log(cfg.p_own), np.where(is_h, np.log(cfg.p_other), none_alive))
    e[:, D] = np.where(nobody, none_dead, np.log(cfg.p_dead_ctrl))
    e[:, H] += np.where(dead, np.log(cfg.q_alive), np.log(1 - cfg.q_alive))
    e[:, A] += np.where(dead, np.log(cfg.q_alive), np.log(1 - cfg.q_alive))
    e[:, D] += np.where(dead, np.log(cfg.q_dead), np.log(1 - cfg.q_dead))
    if extra is not None:
        x = cfg.evidence_weight * np.nan_to_num(np.asarray(extra, dtype=np.float64)) / 2.0
        e[:, H] += x
        e[:, A] -= x
    return e


def emissions_from_proba(proba, prior=None, weight: float = 1.0, floor: float = 1e-4):
    """(n, 3) log-emissions from classifier probabilities P(state | x): log P - log prior."""
    p = np.clip(np.asarray(proba, dtype=np.float64), floor, 1.0)
    prior = np.full(3, 1.0 / 3) if prior is None else np.asarray(prior, dtype=np.float64)
    return weight * (np.log(p) - np.log(prior)[None, :])


def viterbi(emissions, trans, start=None) -> np.ndarray:
    """Most likely state sequence. emissions (n, S) log scores, trans (S, S). Returns int8 (n,)."""
    e = np.asarray(emissions, dtype=np.float64)
    n, S = e.shape
    if n == 0:
        return np.zeros(0, dtype=np.int8)
    back = np.zeros((n, S), dtype=np.int8)
    score = (np.zeros(S) if start is None else np.asarray(start, dtype=np.float64)) + e[0]
    for t in range(1, n):
        cand = score[:, None] + trans            # from x to
        back[t] = np.argmax(cand, axis=0)
        score = cand[back[t], np.arange(S)] + e[t]
    path = np.empty(n, dtype=np.int8)
    path[-1] = int(np.argmax(score))
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def decode_block(emissions, lo, hi, cfg: HmmConfig | None = None) -> np.ndarray:
    """Viterbi over rows lo..hi (inclusive) of a full-length emission array."""
    return viterbi(emissions[lo:hi + 1], transition_matrix(cfg))
