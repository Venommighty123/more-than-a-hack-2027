"""Smart Tagging as typed records, with team names resolved to HOME / AWAY.

Wraps ``starter.load_smart_tagging.load`` (organiser code, unchanged) and takes care of the traps
DATA.md lists:

* tag groups are title case (``Team``), codes upper case (``SET PIECES``), values title case
  (``Kick Off (Start)``) -- everything is matched case-insensitively here;
* ``Team`` is ``N/A`` on 36 instances, including every throw-in -- ``team`` is then ``None``
  (``detectors/throw_in_side.py`` is where a side gets inferred);
* the club is written ``Atletic Club`` in the tags but may differ elsewhere -- names are matched
  fuzzily against the ATD metadata through ``TeamMap.side_of_name``;
* ``Half`` is a string (``1st Half``) -- parsed to 1 / 2.

Frames are video frames, ``round(seconds * 25)``, half-open: ``end_frame`` is the first frame
after the phase. A tag is automatic output, not ground truth: use it as evidence only.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from games import GAME
from starter.load_smart_tagging import load as _load_raw

from .teams import AWAY, HOME, TeamMap, load_teams

SET_PIECES = "set pieces"
GOALS = "goals"


def _norm(text):
    return text.strip().casefold() if isinstance(text, str) else None


@dataclass(frozen=True)
class Tag:
    code: str                    # casefolded, e.g. "set pieces", "build up"
    start_frame: int             # first video frame
    end_frame: int               # first video frame AFTER the phase
    start_s: float
    end_s: float
    team: int | None             # HOME / AWAY, None for N/A or unknown
    team_name: str | None
    half: int | None
    type: str | None             # casefolded "Type" group, e.g. "throw-in", "corner kick"
    side: str | None             # casefolded "Side" group: "central" / "left" / "right"
    groups: dict = field(default_factory=dict)   # every group, keys and values casefolded

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    def contains(self, frame: int) -> bool:
        return self.start_frame <= frame < self.end_frame

    def overlaps(self, first: int, last: int) -> bool:
        """True if the tag shares any frame with first..last (inclusive)."""
        return self.start_frame <= last and first < self.end_frame


def _half(text):
    t = _norm(text) or ""
    return 1 if t.startswith("1") else 2 if t.startswith("2") else None


def load_tags(teams: TeamMap | None = None, game: str = GAME,
              data_dir: str | None = None) -> list[Tag]:
    """All Smart Tagging instances, in file order."""
    teams = teams or load_teams(game=game, data_dir=data_dir)
    out = []
    for r in _load_raw(game, data_dir):
        groups = {_norm(k): _norm(v) for k, v in (r.get("tags") or {}).items() if k}
        raw_team = next((v for k, v in (r.get("tags") or {}).items() if _norm(k) == "team"), None)
        out.append(Tag(code=_norm(r["code"]), start_frame=int(r["start_frame"]),
                       end_frame=int(r["end_frame"]), start_s=float(r["start_s"]),
                       end_s=float(r["end_s"]), team=teams.side_of_name(raw_team),
                       team_name=raw_team, half=_half(groups.get("half")),
                       type=groups.get("type"), side=groups.get("side"), groups=groups))
    return out


# ------------------------------------------------------------------------------------------------
# Selection helpers
# ------------------------------------------------------------------------------------------------

def select(tags, code=None, type=None, team="any", half=None):
    """Filter by code / Type / team (HOME, AWAY, None for N/A, "any") / half. Case-insensitive."""
    code, type = _norm(code), _norm(type)
    return [t for t in tags
            if (code is None or t.code == code)
            and (type is None or t.type == type)
            and (team == "any" or t.team == team)
            and (half is None or t.half == half)]


def set_pieces(tags):
    return select(tags, code=SET_PIECES)


def goals(tags):
    return select(tags, code=GOALS)


def active_at(tags, frame: int):
    return [t for t in tags if t.contains(frame)]


def overlapping(tags, first: int, last: int):
    return [t for t in tags if t.overlaps(first, last)]


def frame_mask(tags, n: int, lo: int = 0) -> np.ndarray:
    """Bool per row (video frame lo + k): covered by at least one of ``tags``."""
    mask = np.zeros(n, dtype=bool)
    for t in tags:
        a, b = max(t.start_frame - lo, 0), min(t.end_frame - lo, n)
        if b > a:
            mask[a:b] = True
    return mask


def team_votes(tags, n: int, lo: int = 0) -> np.ndarray:
    """(n, 2) int16: how many tags credit HOME / AWAY on each frame. Phases are not possessions
    (DATA.md) -- these are counts of evidence, for ``detectors/tag_prior.py`` to weigh."""
    votes = np.zeros((n, 2), dtype=np.int16)
    for t in tags:
        if t.team not in (HOME, AWAY):
            continue
        a, b = max(t.start_frame - lo, 0), min(t.end_frame - lo, n)
        if b > a:
            votes[a:b, t.team] += 1
    return votes
