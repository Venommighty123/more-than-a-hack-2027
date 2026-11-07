"""Find the stretches where the ball is missing, inside the graded halves only.

Merges ball_gap_analysis.find_gaps / add_bracket_stats (ours) and Krish's eda_ball_gaps.find_gaps.

Why gaps get their own module (data audit, game plan): the ball is missing on ~41% of graded
frames, there are almost no gaps between 1 s and 5 s, and two thirds of the missing time sits in
gaps longer than 20 s. Short gaps are bridged in preprocess/ball.py; the long ones decide the
score, so each is classified (classify.py) and filled (detectors/gap_fill.py).

Rows are array rows; pass ``lo`` (video frame of row 0) to get video frames in the output.
A gap touching the edge of a block is censored: its true length is unknown and it has no
detection on one side (``has_before`` / ``has_after`` False).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from possession.utils.frames import runs

FPS = 25


@dataclass
class Gap:
    half: int
    start: int            # first missing row
    end: int              # first row after the gap (exclusive)
    has_before: bool      # a detection exists just before, inside the block
    has_after: bool
    info: dict = field(default_factory=dict)   # filled by bracket() and classify_gaps()

    @property
    def length(self) -> int:
        return self.end - self.start

    def seconds(self, fps: int = FPS) -> float:
        return self.length / float(fps)

    def to_row(self, lo: int = 0, fps: int = FPS) -> dict:
        row = {"half": self.half, "start_frame": self.start + lo, "end_frame": self.end - 1 + lo,
               "length_frames": self.length, "seconds": round(self.seconds(fps), 2),
               "has_before": self.has_before, "has_after": self.has_after}
        row.update(self.info)
        return row


def find_gaps(present, blocks, lo: int = 0, min_len: int = 1) -> list[Gap]:
    """Maximal runs of ``~present`` inside each block. ``blocks`` = [(half, first, last)] in
    video frames; ``lo`` = video frame of row 0."""
    present = np.asarray(present, dtype=bool)
    out = []
    for half, first, last in blocks:
        a, b = max(int(first) - lo, 0), min(int(last) - lo + 1, len(present))
        if b <= a:
            continue
        s_all, e_all = runs(~present[a:b])
        for s, e in zip(s_all, e_all):
            if e - s < min_len:
                continue
            out.append(Gap(int(half), int(a + s), int(a + e), bool(s > 0), bool(a + e < b)))
    return out


def bracket(gaps, bx, by, fps: int = FPS):
    """Add to each gap's info: where the ball vanished / reappeared, the distance between them
    and the speed that implies. In place; returns ``gaps``."""
    for g in gaps:
        if g.has_before:
            g.info["vanish_xy"] = (float(bx[g.start - 1]), float(by[g.start - 1]))
        if g.has_after:
            g.info["reappear_xy"] = (float(bx[g.end]), float(by[g.end]))
        if g.has_before and g.has_after:
            (x0, y0), (x1, y1) = g.info["vanish_xy"], g.info["reappear_xy"]
            d = float(np.hypot(x1 - x0, y1 - y0))
            g.info["dist_m"] = round(d, 2)
            g.info["implied_speed_ms"] = round(d / ((g.length + 1) / float(fps)), 2)
    return gaps


def gap_mask(gaps, n: int, min_len: int = 1) -> np.ndarray:
    mask = np.zeros(n, dtype=bool)
    for g in gaps:
        if g.length >= min_len:
            mask[g.start:g.end] = True
    return mask


def length_histogram(gaps, edges_s=(0, 0.2, 0.5, 1, 2, 5, 10, 20, 60, 1e9), fps: int = FPS):
    """Gaps and missing frames per length bin: [(lo_s, hi_s, n_gaps, frames)]."""
    rows = []
    for lo_s, hi_s in zip(edges_s[:-1], edges_s[1:]):
        sel = [g for g in gaps if lo_s * fps <= g.length < hi_s * fps]
        rows.append((lo_s, hi_s, len(sel), int(sum(g.length for g in sel))))
    return rows
