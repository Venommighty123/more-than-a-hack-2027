"""Choose the ball among the YOLO candidates (Level 1): filters + a Viterbi path, anchored to the ATD.

The Kaggle ball notebook keeps up to TOP_K = 3 candidates per frame (ball_candidates_<mode>.parquet) but
writes the most confident one as "the ball", which is often a spare ball by the touchline, a logo, a
head in the stands. Here every candidate is checked and the ball is chosen as the most plausible PATH:

Filters (a rejected candidate can never be chosen)
    off_pitch   naive ground point more than hard_out_m outside the lines (stands, boards, the bench).
                0..hard_out_m outside is only penalised: an airborne ball's naive ground point lands
                beyond the line it is flying over.
    static      the same pixel cell (static_cell_px) holds a candidate in >= static_frac of the frames
                of a static_s window: a spare ball, the penalty spot, a logo. A real ball moves.
    size        box size / expected pixel size of a 0.22 m ball at that ground point outside
                [size_min, size_max]: lights, bodies, specks.

Path (Viterbi over each run of processed frames; states = the frame's candidates + "no ball")
    emission    -log(conf) + off_w * metres outside the pitch
                + anchor_w * min(d_atd / anchor_m, 3)^2 where the ATD has a ball (provided automatic
                data: the chosen path stays on the ball the ATD sees, and away from look-alikes)
    "no ball"   none_cost (= -log 0.3: a candidate needs about conf 0.3 to beat it)
    transition  candidate -> candidate: move_w * (d / allowed)^2, impossible beyond allowed =
                max_speed_ms * dt * proj_mag + slack_m; any switch to / from "no ball": switch_cost

Pure numpy; scripts/select_ball_candidates.py does the I/O and the comparison with the old pick.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

BALL_M = 0.22


@dataclass
class SelectConfig:
    fps: int = 25
    pitch: tuple = (105.3, 68.0)
    hard_out_m: float = 8.0
    off_w: float = 0.3
    static_cell_px: int = 6
    static_s: float = 8.0
    static_frac: float = 0.9
    size_min: float = 0.4
    size_max: float = 3.0
    none_cost: float = 1.2
    switch_cost: float = 2.0
    max_speed_ms: float = 40.0
    proj_mag: float = 2.0
    slack_m: float = 2.0
    move_w: float = 1.0
    anchor_w: float = 1.0
    anchor_m: float = 3.0

    @classmethod
    def from_dict(cls, d: dict | None) -> "SelectConfig":
        d = d or {}
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict:
        return asdict(self)


def outside_m(x, y, pitch):
    """Metres outside the pitch rectangle (0 inside)."""
    L, W = pitch
    dx = np.maximum(0.0, np.maximum(-x, x - L))
    dy = np.maximum(0.0, np.maximum(-y, y - W))
    return np.hypot(dx, dy)


def static_mask(frame, u, v, cfg: SelectConfig):
    """Candidates sitting in a pixel cell (or its 8 neighbours) that is occupied on >= static_frac of the
    frames of a static_s window centred on them (occupancy summed over the 3x3 cells, capped at the window)."""
    frame = np.asarray(frame, dtype=np.int64)
    cu = np.floor(np.asarray(u) / cfg.static_cell_px).astype(np.int64)
    cv = np.floor(np.asarray(v) / cfg.static_cell_px).astype(np.int64)
    half = int(cfg.static_s * cfg.fps / 2)
    win = 2 * half + 1
    cells = {}
    for i, key in enumerate(zip(cu.tolist(), cv.tolist())):
        cells.setdefault(key, []).append(int(frame[i]))
    cells = {k: np.unique(np.asarray(v_)) for k, v_ in cells.items()}
    count = np.zeros(len(frame), dtype=np.int64)
    # vectorised per neighbour offset: group candidates by their shifted cell
    for a in (-1, 0, 1):
        for b in (-1, 0, 1):
            keys = list(zip((cu + a).tolist(), (cv + b).tolist()))
            order = {}
            for i, k in enumerate(keys):
                order.setdefault(k, []).append(i)
            for k, idx in order.items():
                occ = cells.get(k)
                if occ is None:
                    continue
                idx = np.asarray(idx)
                f = frame[idx]
                count[idx] += np.searchsorted(occ, f + half, side="right") - np.searchsorted(occ, f - half)
    return np.minimum(count, win) >= cfg.static_frac * win


def expected_px(Hn, x_m, y_m, pitch):
    """Pixel size of a 0.22 m ball lying at (x_m, y_m), from a normalised-pitch -> pixel homography."""
    L, W = pitch
    def px(x, y):
        p = Hn @ np.array([x / L, y / W, 1.0])
        return p[:2] / p[2]
    a = px(x_m - BALL_M / 2, y_m)
    b = px(x_m + BALL_M / 2, y_m)
    c = px(x_m, y_m - BALL_M / 2)
    d = px(x_m, y_m + BALL_M / 2)
    return float((np.linalg.norm(b - a) + np.linalg.norm(d - c)) / 2.0)


def viterbi_run(frames, cands, atd, cfg: SelectConfig):
    """One contiguous run of processed frames.

    frames  sorted int array (consecutive)
    cands   {frame: (x (k,), y (k,), conf (k,), ok (k,) bool)}; rejected candidates have ok False
    atd     {frame: (x, y)} ATD ball in metres where present
    Returns {frame: index of the chosen candidate or -1}.
    """
    NONE = -1
    prev_states, prev_cost, back = None, None, []
    for t, f in enumerate(frames):
        x, y, conf, ok = cands.get(int(f), (np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0, bool)))
        idx = np.flatnonzero(ok)
        states = [NONE] + idx.tolist()
        em = np.zeros(len(states))
        em[0] = cfg.none_cost
        if len(idx):
            cx, cy = x[idx], y[idx]
            e = -np.log(np.clip(conf[idx], 1e-3, 1.0)) + cfg.off_w * outside_m(cx, cy, cfg.pitch)
            a = atd.get(int(f))
            if a is not None:
                d = np.hypot(cx - a[0], cy - a[1])
                e = e + cfg.anchor_w * np.minimum(d / cfg.anchor_m, 3.0) ** 2
            em[1:] = e
        if prev_states is None:
            cost = em
            bp = np.full(len(states), -1)
        else:
            allowed = cfg.max_speed_ms / cfg.fps * cfg.proj_mag + cfg.slack_m
            P = len(prev_states)
            trans = np.full((P, len(states)), cfg.switch_cost)
            trans[0, 0] = 0.0
            px_, py_ = prev_xy
            if P > 1 and len(states) > 1:
                d = np.hypot(px_[1:, None] - x[idx][None, :], py_[1:, None] - y[idx][None, :])
                m = cfg.move_w * (d / allowed) ** 2
                m[d > allowed] = np.inf
                trans[1:, 1:] = m
            tot = prev_cost[:, None] + trans
            bp = np.argmin(tot, axis=0)
            cost = tot[bp, np.arange(len(states))] + em
        back.append((states, bp))
        prev_states, prev_cost = states, cost
        prev_xy = (np.r_[np.nan, x[idx]] if len(idx) else np.array([np.nan]),
                   np.r_[np.nan, y[idx]] if len(idx) else np.array([np.nan]))
    out = {}
    if not back:
        return out
    j = int(np.argmin(prev_cost))
    for t in range(len(frames) - 1, -1, -1):
        states, bp = back[t]
        out[int(frames[t])] = states[j]
        j = int(bp[j]) if t > 0 else j
    return out
