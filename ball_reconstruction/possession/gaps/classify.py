"""Classify every ball gap: dead (and why) or live, plus who had the ball either side.

From Ashwyn's test_scripts/gap_classification.py (rules unchanged) and Krish's
classify_all_135_gaps.py (set-piece / goal overlap), as a library working on rows.

  dead_kickoff    the gap starts before the half's kick-off (published, FAQ.md)
  dead_tagged     a Smart Tagging SET PIECES tag overlaps the gap or starts within tag_slack_s of
                  its end; the tag gives the restart type and (except throw-ins) the team
  dead_untagged   the ball vanished or reappeared within near_line_m of a line AND the players
                  were slow during the gap (median mean speed < dead_speed_ms)
  live            everything else: the ball is in play but nobody sees it -- the blind part

Per gap it also records team_before / team_after (last / first CONTROLLING team within
context_s, from detectors/control.py), flip, the Smart Tagging phase team covering most of the gap
(a prior only: phases are not possessions) and the median player speed.

Without tags (the bench) dead_tagged never fires and phase_team is "none". Untuned defaults.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from possession.preprocess.ball import distance_to_lines

FPS = 25
PITCH = (105.3, 68.0)
HOME, AWAY = 0, 1
LETTER = {HOME: "H", AWAY: "A"}
CLASSES = ("dead_kickoff", "dead_tagged", "dead_untagged", "live")


@dataclass
class GapClassConfig:
    min_gap_s: float = 1.0
    near_line_m: float = 2.0
    dead_speed_ms: float = 1.8
    tag_slack_s: float = 3.0
    context_s: float = 5.0
    phase_majority: float = 0.6

    @classmethod
    def from_dict(cls, d: dict | None) -> "GapClassConfig":
        d = d or {}
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict:
        return asdict(self)


def _team_near(ctrl, a, b, last):
    """Last (or first) HOME/AWAY value of ctrl[a:b], or None."""
    seg = np.asarray(ctrl[max(a, 0):max(b, 0)])
    seg = seg[(seg == HOME) | (seg == AWAY)]
    if not len(seg):
        return None
    return int(seg[-1] if last else seg[0])


def classify_gaps(gaps, bx, by, ctrl, mean_speed, *, lo: int = 0, tags=None, kickoffs=None,
                  cfg: GapClassConfig | None = None, fps: int = FPS, pitch=PITCH):
    """Fill ``gap.info`` for every gap of at least min_gap_s and return those gaps.

    gaps        from gaps.find.find_gaps (rows)
    bx, by      cleaned ball (metres); ctrl per-frame control (HOME / AWAY / other = nobody)
    mean_speed  (n,) mean player speed
    tags        optional list of io.tags.Tag (BAR-ATH only)
    kickoffs    optional {half: kick-off video frame}
    """
    cfg = cfg or GapClassConfig()
    kickoffs = kickoffs or {}
    pieces = [t for t in (tags or []) if t.code == "set pieces"]
    phases = [t for t in (tags or []) if t.code not in ("set pieces", "goals")]
    goals = [t for t in (tags or []) if t.code == "goals"]
    slack, ctx = int(cfg.tag_slack_s * fps), int(cfg.context_s * fps)
    out = []
    for g in gaps:
        if g.length < cfg.min_gap_s * fps:
            continue
        v0, v1 = g.start + lo, g.end + lo            # video frames: first missing, first after
        info = g.info
        near = []
        for key, k in (("vanish", g.start - 1), ("reappear", g.end)):
            if 0 <= k < len(bx) and np.isfinite(bx[k]) and np.isfinite(by[k]):
                d = float(distance_to_lines(bx[k], by[k], pitch))
                info[key + "_line_m"] = round(d, 2)
                near.append(d <= cfg.near_line_m)
            else:
                info[key + "_line_m"] = None
                near.append(False)
        info["near_line"] = bool(any(near))
        sp = mean_speed[g.start:g.end]
        info["player_speed_ms"] = round(float(np.nanmedian(sp)), 2) if np.isfinite(sp).any() else None
        slow = info["player_speed_ms"] is not None and info["player_speed_ms"] < cfg.dead_speed_ms

        tb = _team_near(ctrl, g.start - ctx, g.start, last=True)
        ta = _team_near(ctrl, g.end, g.end + ctx, last=False)
        info["team_before"], info["team_after"] = LETTER.get(tb), LETTER.get(ta)
        info["flip"] = (tb != ta) if tb is not None and ta is not None else None

        hit = [p for p in pieces if p.start_frame <= v1 + slack and p.end_frame >= v0 - slack]
        best = min(hit, key=lambda p: abs(p.start_frame - v1)) if hit else None
        info["set_piece"] = best.type if best else None
        info["set_piece_team"] = LETTER.get(best.team) if best else None
        info["goal"] = any(t.overlaps(v0 - slack, v1 + slack) for t in goals)

        cover = {HOME: 0, AWAY: 0}
        for p in phases:
            if p.team in cover:
                ov = min(p.end_frame, v1) - max(p.start_frame, v0)
                if ov > 0:
                    cover[p.team] += ov
        share = {t: c / float(g.length) for t, c in cover.items()}
        if share[HOME] >= cfg.phase_majority and share[HOME] > share[AWAY]:
            info["phase_team"] = "H"
        elif share[AWAY] >= cfg.phase_majority and share[AWAY] > share[HOME]:
            info["phase_team"] = "A"
        else:
            info["phase_team"] = "mixed" if share[HOME] + share[AWAY] > 0 else "none"

        if v0 < kickoffs.get(g.half, -1):
            info["cls"] = "dead_kickoff"
        elif best is not None:
            info["cls"] = "dead_tagged"
        elif info["near_line"] and slow:
            info["cls"] = "dead_untagged"
        else:
            info["cls"] = "live"
        out.append(g)
    return out


def class_table(gaps, graded_frames: int, fps: int = FPS):
    """Per class: gaps, seconds, share of graded frames, flip rate (where both sides known)."""
    rows = []
    for c in CLASSES:
        sel = [g for g in gaps if g.info.get("cls") == c]
        frames = sum(g.length for g in sel)
        known = [g for g in sel if g.info.get("flip") is not None]
        rows.append({"cls": c, "gaps": len(sel), "seconds": round(frames / fps, 1),
                     "graded_pct": round(100.0 * frames / max(1, graded_frames), 2),
                     "flip_pct": round(100.0 * sum(g.info["flip"] for g in known) / len(known), 1)
                     if known else None})
    return rows
