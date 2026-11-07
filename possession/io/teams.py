"""Which team each tracking slot belongs to, read from the ATD metadata XML.

Replaces the four copies of ``team_of_each_track`` (starter/baseline.py, possession_test_1.py,
pass_detection.py, ball_reconstruction.py) and fixes the bug they share: a slot whose ``teamId``
is missing or matches neither team used to be counted as AWAY. Here it is UNKNOWN (-1) and is in
neither team's mask. ``ball_gap_analysis.audit_slot_teams`` exposed the problem; this fixes it.

A slot is a tracklet, not a person (DATA.md): 46 slots for 22 players, and a slot can switch
player. Use slots for collective features (nearest of a team, centroids), never as identities.

The first ``<Team>`` in the XML is the home side (FIFA/EPTS convention; Barcelona for BAR-ATH).
"""
from __future__ import annotations

import difflib
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np

from games import DEFAULT_SOURCE, GAME, require

HOME, AWAY, UNKNOWN = 0, 1, -1
LETTER = {HOME: "H", AWAY: "A"}
NAME_MATCH = 0.8  # difflib ratio; "Atletic Club" (tags) vs "Athletic Club" must still match


@dataclass(frozen=True)
class TeamMap:
    """Team of every slot, in the column order of the raw ATD file."""

    team: np.ndarray            # int8 per slot: HOME, AWAY or UNKNOWN
    slot_ids: tuple             # Player id attribute of each slot ("1".."46")
    team_ids: tuple             # (home_id, away_id)
    names: tuple                # (home_name, away_name)
    channel_order_ok: bool | None = None   # raw columns follow the Player list? None = not checkable

    @property
    def n_slots(self) -> int:
        return len(self.team)

    @property
    def home(self) -> np.ndarray:
        return self.team == HOME

    @property
    def away(self) -> np.ndarray:
        return self.team == AWAY

    @property
    def unknown(self) -> np.ndarray:
        return self.team == UNKNOWN

    def mask(self, side: int) -> np.ndarray:
        return self.team == side

    def side_of_name(self, name: str | None) -> int | None:
        """HOME / AWAY for a club name as written anywhere (Smart Tagging, reports), else None."""
        if not name:
            return None
        key = name.strip().casefold()
        if key in ("n/a", "na", "none", ""):
            return None
        scores = [difflib.SequenceMatcher(None, key, n.casefold()).ratio() for n in self.names]
        best = int(np.argmax(scores))
        return (HOME, AWAY)[best] if scores[best] >= NAME_MATCH else None

    def summary(self) -> dict:
        return {"slots": self.n_slots, "home": int(self.home.sum()), "away": int(self.away.sum()),
                "unknown": int(self.unknown.sum()), "names": self.names,
                "channel_order_ok": self.channel_order_ok}


def _channel_order_ok(root, slot_ids):
    """True / False if the metadata lets us check that raw columns follow the Player list."""
    channels = {c.get("id"): c.get("playerId") for c in root.iter("PlayerChannel")}
    refs = [r.get("playerChannelId") for r in root.iter("PlayerChannelRef")]
    if not channels or not refs:
        return None
    order = []
    for ref in refs:
        pid = channels.get(ref)
        if pid is not None and pid not in order:
            order.append(pid)
    if len(order) != len(slot_ids):
        return None
    return order == list(slot_ids)


def load_teams(meta_path: str | None = None, game: str = GAME, data_dir: str | None = None,
               source: str = DEFAULT_SOURCE) -> TeamMap:
    """Parse the ATD metadata into a TeamMap. ``meta_path`` overrides the bundle lookup."""
    if meta_path is None:
        meta_path = require(["atd_meta"], game, data_dir, source)[0]
    root = ET.parse(meta_path).getroot()
    teams = list(root.iter("Team"))
    if len(teams) != 2:
        raise ValueError("{}: expected 2 teams, found {}".format(meta_path, len(teams)))
    team_ids = tuple(t.get("id") for t in teams)
    names = tuple((t.findtext("Name") or t.get("id")).strip() for t in teams)
    players = list(root.iter("Player"))
    if not players:
        raise ValueError("{}: no Player entries".format(meta_path))
    side = {team_ids[0]: HOME, team_ids[1]: AWAY}
    team = np.array([side.get(p.get("teamId"), UNKNOWN) for p in players], dtype=np.int8)
    slot_ids = tuple(p.get("id") for p in players)
    return TeamMap(team=team, slot_ids=slot_ids, team_ids=team_ids, names=names,
                   channel_order_ok=_channel_order_ok(root, slot_ids))


def team_of_each_track(meta_path: str | None = None) -> np.ndarray:
    """Drop-in for the old helper, except unmapped slots are UNKNOWN (-1) instead of AWAY."""
    return load_teams(meta_path).team
