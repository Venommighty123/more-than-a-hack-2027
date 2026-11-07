"""Which team takes each throw-in. Smart Tagging writes Team = N/A on every throw-in (DATA.md),
exactly the restart where the side would be worth the most.

From Krish's THROW_IN_SIDE.md / verify_throwin_takers.py (centroid-in-own-half), with three
changes:
  * teams and attacking direction come from the data (io/teams.py, preprocess/players.attack_signs)
    instead of hard-coded slot ranges and half directions;
  * the goalkeeper is the player closest to his OWN GOAL LINE (x), not to y = 0/1 as in the original
    (y runs along the goal line, so that picked a winger on the touchline);
  * a second, more direct cue is tried first: the team of the player nearest the ball when it
    comes back to life (the thrower holds it), if a player is within ``ball_r_m``.

Cues, in order:
  ball      nearest player to the ball at the restart (first alive frame after the stoppage inside
            the tag, or the tag start), within ball_r_m
  centroid  the team whose outfield centroid sits deeper in its own half takes it (takers line up
            inside their half, defenders press up the line)

This is inference, not ground truth. Smart Tagging exists only for BAR-ATH, so it cannot be scored
on the bench. Do not check it against the video frame by frame and change it: that would be hand
labelling (RULES.md rule 1). Untuned defaults.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

HOME, AWAY = 0, 1
PITCH = (105.3, 68.0)


@dataclass
class ThrowInConfig:
    ball_r_m: float = 2.5          # thrower within this of the ball
    ball_search_frames: int = 10   # look this far either side of the restart for a ball fix
    min_players: int = 3


@dataclass
class ThrowIn:
    tag_index: int
    frame: int                 # video frame the decision was taken at
    side: int | None           # HOME / AWAY / None
    cue: str                   # "ball", "centroid" or "none"
    scores: dict


def _own_goal_x(sign, pitch):
    """A team attacking +x (sign +1) defends x = 0."""
    return 0.0 if sign > 0 else pitch[0]


def _outfield_centroid_x(xs, sign, pitch, min_players):
    xs = xs[np.isfinite(xs)]
    if len(xs) < min_players:
        return None
    gk = np.argmin(np.abs(xs - _own_goal_x(sign, pitch)))
    rest = np.delete(xs, gk)
    return float(rest.mean()) if len(rest) >= 2 else None


def infer_throw_ins(tags, px, py, bx, by, team, signs, *, lo=0, dead=None, half_of=None,
                    cfg: ThrowInConfig | None = None, pitch=PITCH):
    """Infer the side of every throw-in tag whose team is unknown.

    tags     io.tags.Tag list; px, py, bx, by rows starting at video frame ``lo``
    signs    {(half, side): +1/-1} from preprocess.players.attack_signs
    dead     optional dead mask (rows): the restart is the first alive frame inside the tag
    half_of  callable video_frame -> half (default: tag.half)
    """
    cfg = cfg or ThrowInConfig()
    team = np.asarray(team)
    n = len(bx)
    out = []
    for i, t in enumerate(tags):
        if t.type != "throw-in" or t.team is not None:
            continue
        half = half_of(t.start_frame) if half_of else t.half
        a, b = t.start_frame - lo, min(t.end_frame - lo, n)
        if not 0 <= a < n:
            continue
        r = a
        if dead is not None:
            alive = np.flatnonzero(~dead[a:b])
            r = a + int(alive[0]) if len(alive) else a
        scores = {}
        # cue 1: the thrower holds the ball
        lo_k, hi_k = max(0, r - cfg.ball_search_frames), min(n, r + cfg.ball_search_frames + 1)
        ks = [k for k in range(lo_k, hi_k) if np.isfinite(bx[k])]
        if ks:
            k = min(ks, key=lambda q: abs(q - r))
            d = np.hypot(px[k] - bx[k], py[k] - by[k])
            d = np.where(np.isfinite(d) & (team >= 0), d, np.inf)
            j = int(np.argmin(d))
            scores["ball_dist_m"] = float(d[j]) if np.isfinite(d[j]) else None
            if np.isfinite(d[j]) and d[j] <= cfg.ball_r_m:
                out.append(ThrowIn(i, r + lo, int(team[j]), "ball", scores))
                continue
        # cue 2: centroid depth in own half
        depth = {}
        for side in (HOME, AWAY):
            sign = signs.get((half, side))
            if sign is None:
                continue
            cx = _outfield_centroid_x(px[a, team == side], sign, pitch, cfg.min_players)
            if cx is not None:
                depth[side] = -sign * (cx - pitch[0] / 2.0)     # > 0: inside own half
        scores.update({"depth_" + "HA"[s]: round(v, 2) for s, v in depth.items()})
        if len(depth) == 2:
            out.append(ThrowIn(i, a + lo, max(depth, key=depth.get), "centroid", scores))
        else:
            out.append(ThrowIn(i, a + lo, None, "none", scores))
    return out


def fill_tag_teams(tags, results):
    """Copy of ``tags`` with each inferred throw-in's team filled in."""
    side = {r.tag_index: r.side for r in results if r.side is not None}
    return [dataclasses.replace(t, team=side[i]) if i in side else t for i, t in enumerate(tags)]
