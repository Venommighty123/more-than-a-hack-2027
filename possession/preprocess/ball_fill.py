"""Fill ball gaps with the YOLO ball, conservatively (P4 + P5 of the referee plan).

Base ball: the pipeline's default fixed ball (outputs/ball/ball_reconstructed.npz). A *gap* is a
maximal run of frames without a base ball inside a graded half. Candidates are the cleaned YOLO
detections of outputs/ball/ball_yolo_fixed.npz with source ``yolo`` only (bridged and flight-fit
frames are dropped: on frames where both exist they were 8 m and 15 m from the ATD ball, median).

Which gaps may be filled (scope)
    open_play   the gap does not touch any labeller v6 dead stretch            -> eligible
    dead        it overlaps a v6 dead stretch: eligible only with live evidence, i.e. the accepted
                chain covers >= live_min_coverage of the gap, stays on the pitch and moves at least
                live_min_path_m, and no overlapping stretch has a referee gesture (gesture = dead).

How a gap is filled (strict chain rule)
    Both anchors (base ball just before and just after the gap) must exist. A chain grows forward
    from the anchor before: a candidate links if it is within anchor_m + max_speed_ms * dt of the
    last linked point (others are dropped as outliers), and the chain stops at the first hole longer
    than max_hole frames. If it reaches the anchor after the same way, the gap is "full". Otherwise a
    second chain grows backward from the anchor after, and only those two anchor-attached pieces are
    kept ("partial"): a detection run floating in the middle of a gap is never trusted, because
    that is where the tracker latches onto the wrong object. Holes of <= max_hole frames between
    kept points are interpolated linearly; longer holes stay empty.

Everything is pure numpy on arrays in metres; scripts/fill_ball_gaps.py does the I/O and the
hide-and-refill test that gates the whole step.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

FILL_NONE, FILL_POINT, FILL_INTERP = 0, 1, 2


@dataclass
class FillConfig:
    fps: int = 25
    anchor_m: float = 3.0           # link tolerance on top of the speed allowance
    max_speed_ms: float = 40.0      # a ball never moves faster than this
    max_hole: int = 5               # interpolate holes of at most this many frames
    live_min_coverage: float = 0.8  # dead-stretch gaps: share of the gap the chain must cover
    live_min_path_m: float = 5.0    # ... and the distance the ball must travel
    pitch: tuple = (105.3, 68.0)
    pitch_margin_m: float = 1.0

    @classmethod
    def from_dict(cls, d: dict | None) -> "FillConfig":
        d = d or {}
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict:
        return asdict(self)


def runs(mask):
    """(starts, ends) of True runs, ends exclusive."""
    e = np.diff(np.concatenate(([0], np.asarray(mask, dtype=np.int8), [0])))
    return np.flatnonzero(e == 1), np.flatnonzero(e == -1)


def find_gaps(base_present, blocks):
    """[(half, start, end_exclusive)] of runs without a base ball inside the graded blocks."""
    out = []
    for half, first, last in blocks:
        s, e = runs(~np.asarray(base_present[first:last + 1], dtype=bool))
        out += [(int(half), int(first + a), int(first + b)) for a, b in zip(s, e)]
    return out


def _grow(cx, cy, idx, ax, ay, af, cfg: FillConfig, step):
    """Grow a chain from an anchor through candidate frames ``idx`` (in walking order). A candidate links
    if it is within anchor_m + max_speed_ms * dt of the last linked point; others are skipped as
    outliers. The chain stops at the first hole longer than max_hole frames."""
    kept, lx, ly, lf = [], ax, ay, af
    for f in idx:
        if abs(f - lf) - 1 > cfg.max_hole:
            break
        dt = abs(f - lf) / float(cfg.fps)
        if np.hypot(cx[f] - lx, cy[f] - ly) <= cfg.anchor_m + cfg.max_speed_ms * dt:
            kept.append(int(f)); lx, ly, lf = cx[f], cy[f], f
    return kept, lf, lx, ly


def chain_gap(bx, by, cx, cy, cand, a, b, cfg: FillConfig):
    """Strict chain through gap [a, b): continuous (holes <= max_hole) and attached to an anchor.

    Grows forward from the base ball before the gap and backward from the one after it. Returns
    (accepted, frames_kept, reason) with reason "full" when the forward chain reaches the anchor
    after, "partial" when only anchor-attached pieces exist, else why nothing was kept.
    """
    n = len(bx)
    if a <= 0 or b >= n or not (np.isfinite(bx[a - 1]) and np.isfinite(bx[b])):
        return False, np.array([], dtype=int), "no anchor on both sides"
    idx = np.flatnonzero(cand[a:b] & np.isfinite(cx[a:b])) + a
    if not len(idx):
        return False, idx, "no candidate"
    fwd, lf, lx, ly = _grow(cx, cy, idx, bx[a - 1], by[a - 1], a - 1, cfg, +1)
    dt = (b - lf) / float(cfg.fps)
    if fwd and b - lf - 1 <= cfg.max_hole and np.hypot(bx[b] - lx, by[b] - ly) <= cfg.anchor_m + cfg.max_speed_ms * dt:
        return True, np.asarray(fwd), "full"
    bwd, _, _, _ = _grow(cx, cy, idx[::-1], bx[b], by[b], b, cfg, -1)
    kept = sorted(set(fwd) | set(bwd))
    if not kept:
        return False, np.array([], dtype=int), "no candidate attached to an anchor"
    return True, np.asarray(kept), "partial"


def fill_from_chain(bx, by, cx, cy, kept, a, b, cfg: FillConfig):
    """(x, y, code) for the gap frames [a, b): chain points + short interpolated holes."""
    L = b - a
    x, y, code = np.full(L, np.nan), np.full(L, np.nan), np.zeros(L, dtype=np.int8)
    pts = [(a - 1, bx[a - 1], by[a - 1])] + [(int(f), cx[f], cy[f]) for f in kept] + [(b, bx[b], by[b])]
    for f, px, py in pts[1:-1]:
        x[f - a], y[f - a], code[f - a] = px, py, FILL_POINT
    for (f0, x0, y0), (f1, x1, y1) in zip(pts[:-1], pts[1:]):
        hole = f1 - f0 - 1
        if 0 < hole <= cfg.max_hole:
            for k in range(1, hole + 1):
                t = k / float(f1 - f0)
                x[f0 + k - a], y[f0 + k - a], code[f0 + k - a] = x0 + t * (x1 - x0), y0 + t * (y1 - y0), FILL_INTERP
    return x, y, code


def live_evidence(x, y, code, cfg: FillConfig):
    """(ok, coverage, path_m, on_pitch) for a filled dead-stretch gap."""
    ok = code > 0
    cov = float(ok.mean()) if len(ok) else 0.0
    px, py = x[ok], y[ok]
    path = float(np.hypot(np.diff(px), np.diff(py)).sum()) if len(px) > 1 else 0.0
    m = cfg.pitch_margin_m
    on = bool(len(px)) and bool(np.all((px >= -m) & (px <= cfg.pitch[0] + m) & (py >= -m) & (py <= cfg.pitch[1] + m)))
    return (cov >= cfg.live_min_coverage and path >= cfg.live_min_path_m and on), cov, path, on


def fill_gaps(bx, by, cx, cy, cand, gaps, dead_of_gap, gesture_of_gap, cfg: FillConfig):
    """Fill every eligible gap.

    dead_of_gap     {gap index: [v6 stretch ids]} for gaps overlapping a dead stretch
    gesture_of_gap  {gap index: True} if any overlapping stretch has a referee gesture
    Returns (x, y, fill_code, rows) with rows = one report dict per gap.
    """
    X, Y = np.array(bx, dtype=np.float64), np.array(by, dtype=np.float64)
    code = np.zeros(len(X), dtype=np.int8)
    rows = []
    for gi, (half, a, b) in enumerate(gaps):
        dead = dead_of_gap.get(gi, [])
        row = {"gap": gi, "half": half, "first_frame": a, "last_frame": b - 1, "frames": b - a,
               "scope": "dead" if dead else "open_play", "v6_ids": "|".join(map(str, dead)),
               "referee_gesture": bool(gesture_of_gap.get(gi, False)), "candidates": int(cand[a:b].sum()),
               "kept_points": 0, "filled_frames": 0, "coverage": 0.0, "path_m": 0.0, "decision": "skipped", "reason": ""}
        if row["referee_gesture"]:
            row["reason"] = "referee gesture: dead"
            rows.append(row); continue
        accepted, kept, reason = chain_gap(X, Y, cx, cy, cand, a, b, cfg)
        row["kept_points"] = int(len(kept))
        if not accepted:
            row["reason"] = reason
            rows.append(row); continue
        gx, gy, gc = fill_from_chain(X, Y, cx, cy, kept, a, b, cfg)
        ok, cov, path, on = live_evidence(gx, gy, gc, cfg)
        row.update(coverage=round(cov, 3), path_m=round(path, 1))
        if dead and not ok:
            row["reason"] = "dead stretch without live evidence (coverage {:.0%}, path {:.0f} m{})".format(
                cov, path, "" if on else ", off pitch")
            rows.append(row); continue
        m = gc > 0
        X[a:b][m], Y[a:b][m], code[a:b][m] = gx[m], gy[m], gc[m]
        row.update(filled_frames=int(m.sum()), decision="filled" if reason == "full" else "partial",
                   reason="open play" if not dead else "dead stretch with live evidence")
        rows.append(row)
    return X, Y, code, rows


def hide_and_refill(bx, by, cx, cy, cand, eligible, lengths, cfg: FillConfig, n=400, seed=0, min_cand_share=0.3):
    """Gate: hide the base ball on random windows where it exists (and anchors exist), refill with the
    chain rule, and measure the distance of each filled frame to the hidden base ball.

    eligible  bool per frame: where a test window may lie (graded, base present, open play)
    lengths   window lengths to draw from (the real gap lengths)
    """
    rng = np.random.default_rng(seed)
    lengths = np.asarray(lengths, dtype=int)
    lengths = lengths[(lengths >= 5) & (lengths <= 250)]
    if not len(lengths):
        return {"windows": 0}
    ok_run = np.asarray(eligible, dtype=bool) & np.isfinite(bx)
    starts = np.flatnonzero(ok_run)
    errs, filled, total, used, accepted = [], 0, 0, 0, 0
    for _ in range(n * 20):
        if used >= n:
            break
        L = int(rng.choice(lengths))
        a = int(rng.choice(starts))
        b = a + L
        if a < 1 or b + 1 >= len(bx) or not ok_run[a - 1:b + 1].all():
            continue
        if cand[a:b].mean() < min_cand_share:
            continue
        hx, hy = np.array(bx, dtype=np.float64), np.array(by, dtype=np.float64)
        truth_x, truth_y = hx[a:b].copy(), hy[a:b].copy()
        hx[a:b] = np.nan; hy[a:b] = np.nan
        used += 1
        acc, kept, _ = chain_gap(hx, hy, cx, cy, cand, a, b, cfg)
        total += L
        if not acc:
            continue
        accepted += 1
        gx, gy, gc = fill_from_chain(hx, hy, cx, cy, kept, a, b, cfg)
        m = gc > 0
        filled += int(m.sum())
        errs.append(np.hypot(gx[m] - truth_x[m], gy[m] - truth_y[m]))
    e = np.concatenate(errs) if errs else np.array([])
    return {"windows": used, "accepted_windows": accepted, "frames_hidden": total, "frames_filled": filled,
            "fill_rate": round(filled / total, 3) if total else 0.0,
            "median_err_m": round(float(np.median(e)), 2) if len(e) else None,
            "within_1m": round(float(np.mean(e <= 1)), 3) if len(e) else None,
            "within_2m": round(float(np.mean(e <= 2)), 3) if len(e) else None,
            "over_5m": round(float(np.mean(e > 5)), 3) if len(e) else None}
