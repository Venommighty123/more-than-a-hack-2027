"""Smart Tagging as soft evidence about WHO has the ball, per frame. BAR-ATH only.

Phases are not possessions (DATA.md): a Build Up can stop short of a possession or span two. So
nothing here copies an interval into the labels; it produces a signed score per frame
(+ favours HOME, - favours AWAY) for detectors/gap_fill.py and the labeller to weigh.

Direction of each code (does the TAGGED team have the ball?) follows the code's meaning:
attacking phases -> yes, "DEFENDING IN ..." and "DEFENSIVE TRANSITION" -> no (the tagged team is
the one defending). A tracking-only check on BAR-ATH agrees with every sign: the tagged team holds
85-87 % of controlled frames in BUILD UP / PROGRESSION and 13-15 % in DEFENDING IN MIDDLE /
ATTACKING THIRD. The weights below are set by meaning (possession phase 1.0, territory 0.5),
NOT fitted on BAR-ATH, and cannot be tuned on the bench (it has no Smart Tagging). Changing them
needs a reason that does not come from the public 5 minutes (RULES.md).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

HOME, AWAY = 0, 1

CODE_WEIGHTS = {
    # the tagged team has the ball
    "build up": 1.0, "progression": 1.0, "creating chances": 1.0, "attacking transition": 1.0,
    "long ball": 0.5, "ball in final third": 0.5, "ball in the box": 0.5,
    "players in the box": 0.5,
    # the tagged team is defending: the OTHER team has the ball
    "defending in middle third": -1.0, "defending in attacking third": -1.0,
    "defensive transition": -1.0, "defending in defensive third": -0.5,
}


@dataclass
class TagPrior:
    score: np.ndarray                      # (n,) float: + HOME, - AWAY, 0 no evidence
    restarts: list = field(default_factory=list)   # set pieces: dict(start, end, team, type)

    def side(self, a: int, b: int, min_abs: float = 0.5):
        """HOME / AWAY if the mean score over rows a..b-1 is clear, else None."""
        if b <= a:
            return None
        m = float(np.mean(self.score[a:b]))
        return HOME if m >= min_abs else AWAY if m <= -min_abs else None


def team_score(tags, n: int, lo: int = 0, weights=None) -> np.ndarray:
    weights = CODE_WEIGHTS if weights is None else weights
    s = np.zeros(n, dtype=np.float32)
    for t in tags:
        w = weights.get(t.code)
        if not w or t.team not in (HOME, AWAY):
            continue
        a, b = max(t.start_frame - lo, 0), min(t.end_frame - lo, n)
        if b > a:
            s[a:b] += w if t.team == HOME else -w
    return s


def build_prior(tags, n: int, lo: int = 0, weights=None) -> TagPrior:
    """``tags`` should already have throw-in teams filled (detectors/throw_in_side.py)."""
    restarts = [{"start": t.start_frame - lo, "end": t.end_frame - lo, "team": t.team,
                 "type": t.type} for t in tags if t.code == "set pieces"]
    return TagPrior(team_score(tags, n, lo, weights), restarts)
