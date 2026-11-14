"""Clean slot-shaped player tracks: cut slots at jumps, drop duplicate runs, optionally fill gaps.

USE NOW (YOLO players). possession/io/detections.py calls ``fix_players(px, py, team, yolo=None,
cfg=FixConfig(max_interp=0, max_bridge=0))`` on OUR packed slots: steps 1 and 1b only, because a
slot holds several YOLO tracks one after another and filling between them would join two people.
Steps 2-3 (bridge / interp) were written for the ATD, whose player tracks are no longer used; they
stay for reference. Original description (ATD era) below.

Fix ATD player tracks before they are used for events: cut slots at jumps, then fill gaps.

    from possession.preprocess.fix_players import fix_players, FixConfig
    px2, py2, source, report = fix_players(px, py, team, yolo=yolo_rows)     # metres, (n, slots)

WHY. A slot is a tracklet, not a person (DATA.md): when the tracker hands a slot to another player,
or re-locks it somewhere else, the slot "teleports". Downstream code smooths positions and takes
centred differences, so one teleport becomes a 50-100 m/s player for a few frames: a fake winner of
the ball, a fake duel, a fake pass receiver. viz/track_jumps.py lists these jumps; this module
removes them with the same rule (faster than 12 m/s over 5 frames, as the Kaggle notebook's
JUMP_SPEED and PlayerConfig.max_speed_ms).

STEPS (all untuned defaults from the existing pipeline, never fitted on BAR-ATH, RULES.md rule 1)
  1. cut     for every jump, find the single fastest frame step inside it and blank `margin`
             frames either side (margin = the detector's SMOOTH_HALF = 2), so no smoothing or
             velocity is ever computed across the jump. Repeated until no jump is left (or
             `max_passes`). Gaps the slot reappears from too fast get the same margin.
  1b dedupe a run of a slot (between gaps / cuts) that spends at least `dup_share` of at least
             `dup_min_frames` frames within `dup_dist` m of ANOTHER slot is a second copy of that
             person (typically a slot that jumped onto someone else): the run with the shorter life
             is blanked. Without this a home slot sitting on an away player makes a fake duel.
  2. bridge  a gap of at most `max_bridge` frames is filled from OUR tracks (YOLO parquet) when
             the SAME YOLO id sits within `gate` m of the slot on both sides of the gap, has the
             slot's team (or no team), covers most of the gap and moves at a plausible speed.
             This is the use of our tracks the tracking plan allows: fill where ATD is missing or
             jumps, never replace ATD where it is fine.
  3. interp  a gap of at most `max_interp` frames whose two ends imply at most `max_gap_speed`
             (8 m/s = STITCH_MAX_SPEED) is filled linearly. A cut jump is never re-joined: its ends
             imply far more than 8 m/s.
Identity is NOT repaired: after a cut the slot simply continues on whoever it tracks next. The
event detector already treats slots as tracklets (merge_slot_switches), so that is enough.

`source` (int8, same shape) records where every value came from:
  0 missing   1 as given (GIVEN)   2 bridged from a YOLO track   3 interpolated
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

FPS = 25
MISSING, GIVEN, BRIDGED, INTERP = 0, 1, 2, 3
ATD = GIVEN                      # old name: the value as given by the input tracks


@dataclass
class FixConfig:
    jump_speed: float = 12.0        # m/s over `lag` frames = a jump (notebook JUMP_SPEED)
    lag: int = 5                    # frames (0.2 s)
    margin: int = 2                 # frames blanked either side of a cut (detector SMOOTH_HALF)
    max_passes: int = 4
    dup_dist: float = 0.7           # m; two slots this close are one person tracked twice
    dup_share: float = 0.8          # share of a run spent on another slot to call it a duplicate
    dup_min_frames: int = 25        # runs shorter than this are left alone (1 s)
    reappear_speed: float = 8.0     # m/s implied across a gap = the slot came back somewhere else
    max_gap_speed: float = 8.0      # m/s; a gap is only filled when its ends are this close
    max_interp: int = 12            # frames (0.5 s) filled linearly
    max_bridge: int = 50            # frames (2 s) filled from a YOLO track
    gate: float = 2.0               # m, slot <-> YOLO match (notebook head_to_head)
    bridge_cover: float = 0.8       # share of the gap the YOLO track must cover
    fps: int = FPS

    def to_dict(self):
        return asdict(self)


# -- helpers ---------------------------------------------------------------------------------------

def _runs(mask):
    """(starts, ends) of True runs, ends exclusive."""
    m = np.r_[False, np.asarray(mask, bool), False]
    d = np.diff(m.astype(np.int8))
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def _interior_gaps(ok):
    """[(s, e)] NaN runs of one slot with a valid frame on both sides (e exclusive)."""
    s, e = _runs(~ok)
    return [(a, b) for a, b in zip(s, e) if a > 0 and b < len(ok)]


# -- 1. cut at jumps -------------------------------------------------------------------------------

def cut_jumps(px, py, cfg: FixConfig):
    """Blank frames around every jump (in place on copies). Returns (px, py, n_cut_events, cut_mask)."""
    px, py = px.copy(), py.copy()
    n, S = px.shape
    cut = np.zeros((n, S), dtype=bool)
    lag, m = cfg.lag, cfg.margin
    events = 0
    for _ in range(cfg.max_passes):
        found = 0
        with np.errstate(invalid="ignore"):
            sp = np.full((n, S), np.nan)
            sp[lag:] = np.hypot(px[lag:] - px[:-lag], py[lag:] - py[:-lag]) * cfg.fps / lag
            step = np.full((n, S), np.nan)                   # step[j] = move from j-1 to j
            step[1:] = np.hypot(np.diff(px, axis=0), np.diff(py, axis=0))
            flagged = sp > cfg.jump_speed
        for s in np.flatnonzero(flagged.any(0)):
            ks = np.flatnonzero(flagged[:, s])
            new = np.r_[True, np.diff(ks) > lag]
            for grp in np.split(ks, np.flatnonzero(new)[1:]):
                a, b = grp[0] - lag, grp[-1]                 # the jump happened inside a..b
                seg = step[a + 1:b + 1, s]
                if np.isfinite(seg).any():
                    j = a + 1 + int(np.nanargmax(seg))
                else:                                        # spans a gap: cut where the slot reappears
                    ok = np.isfinite(px[a:b + 1, s])
                    j = a + int(np.flatnonzero(ok)[-1])
                lo_, hi_ = max(0, j - m), min(n, j + m)
                px[lo_:hi_, s] = py[lo_:hi_, s] = np.nan
                cut[lo_:hi_, s] = True
                found += 1
        events += found
        if not found:
            break
    # Reappearing too fast across a gap: widen the gap so smoothing never joins the two sides.
    for s in range(S):
        ok = np.isfinite(px[:, s])
        for a, b in _interior_gaps(ok):
            d = np.hypot(px[b, s] - px[a - 1, s], py[b, s] - py[a - 1, s])
            if d / (b - a + 1) * cfg.fps > cfg.reappear_speed and b - a < 2 * m:
                hi_ = min(n, a + 2 * m)
                px[b:hi_, s] = py[b:hi_, s] = np.nan
                cut[b:hi_, s] = True
                events += 1
    return px, py, events, cut


# -- 1b. duplicates --------------------------------------------------------------------------------

def remove_duplicates(px, py, cfg: FixConfig, chunk=4000):
    """Blank runs that sit on another slot (in place). Returns (n_runs_removed, mask)."""
    n, S = px.shape
    near = np.zeros((n, S), dtype=bool)                  # slot s within dup_dist of another slot
    partner = np.full((n, S), -1, dtype=np.int16)
    for a in range(0, n, chunk):
        x, y = px[a:a + chunk].astype(np.float32), py[a:a + chunk].astype(np.float32)   # ~40 MB / chunk
        with np.errstate(invalid="ignore"):
            d = np.hypot(x[:, :, None] - x[:, None, :], y[:, :, None] - y[:, None, :])
        d[:, np.arange(S), np.arange(S)] = np.inf
        d = np.where(np.isfinite(d), d, np.inf)
        j = np.argmin(d, axis=2)
        dm = np.take_along_axis(d, j[:, :, None], axis=2)[:, :, 0]
        near[a:a + chunk] = dm <= cfg.dup_dist
        partner[a:a + chunk] = np.where(dm <= cfg.dup_dist, j, -1)
    life = np.zeros((n, S), dtype=np.int32)              # length of the run each frame belongs to
    runs_by_slot = []
    for s in range(S):
        st, en = _runs(np.isfinite(px[:, s]))
        runs_by_slot.append(list(zip(st, en)))
        for a, b in zip(st, en):
            life[a:b, s] = b - a
    removed = np.zeros((n, S), dtype=bool)
    count = 0
    for s in range(S):
        for a, b in runs_by_slot[s]:
            if b - a < cfg.dup_min_frames or near[a:b, s].mean() < cfg.dup_share:
                continue
            p = partner[a:b, s][near[a:b, s]]
            q = int(np.bincount(p).argmax())
            if removed[a:b, q].any():
                continue                                 # its twin is already gone: keep this one
            if life[a:b, q].max() > b - a or (life[a:b, q].max() == b - a and q < s):
                px[a:b, s] = py[a:b, s] = np.nan
                removed[a:b, s] = True
                count += 1
    return count, removed


# -- 2. bridge from our YOLO tracks ----------------------------------------------------------------

class YoloIndex:
    """Fast lookups in our tracks. rows: DataFrame-like with frame, id, team (0/1/-1), x_m, y_m
    (referees already removed)."""

    def __init__(self, rows):
        fr = np.asarray(rows["frame"], np.int64)
        o = np.argsort(fr, kind="stable")
        self.fr = fr[o]
        self.id = np.asarray(rows["id"], np.int64)[o]
        self.team = np.asarray(rows["team"], np.int64)[o]
        self.x = np.asarray(rows["x_m"], float)[o]
        self.y = np.asarray(rows["y_m"], float)[o]
        self.lo, self.hi = (int(self.fr[0]), int(self.fr[-1])) if len(self.fr) else (0, -1)
        self.starts = np.searchsorted(self.fr, np.arange(self.lo, self.hi + 2))
        o2 = np.lexsort((self.fr, self.id))                  # by id, then frame
        self.by_id = {}
        ids = self.id[o2]
        cut = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1], True])
        for a, b in zip(cut[:-1], cut[1:]):
            k = o2[a:b]
            self.by_id[int(ids[a])] = (self.fr[k], self.x[k], self.y[k])

    def nearest(self, f, x, y, gate):
        if not (self.lo <= f <= self.hi):
            return None
        a, b = self.starts[f - self.lo], self.starts[f - self.lo + 1]
        if b <= a:
            return None
        d = np.hypot(self.x[a:b] - x, self.y[a:b] - y)
        k = int(np.argmin(d))
        return (int(self.id[a + k]), int(self.team[a + k])) if d[k] <= gate else None

    def path(self, tid, s, e):
        """Positions of track tid on frames s..e-1 (NaN where it has none)."""
        fr, x, y = self.by_id[tid]
        a, b = np.searchsorted(fr, s), np.searchsorted(fr, e)
        out = np.full((e - s, 2), np.nan)
        out[fr[a:b] - s, 0], out[fr[a:b] - s, 1] = x[a:b], y[a:b]
        return out


def bridge_gaps(px, py, team, yolo: YoloIndex, cfg: FixConfig, lo=0):
    """Fill slot gaps from a YOLO track that is the same person on both sides. Returns count and mask."""
    n, S = px.shape
    filled = np.zeros((n, S), dtype=bool)
    count = 0
    for s in range(S):
        ok = np.isfinite(px[:, s])
        for a, b in _interior_gaps(ok):
            if b - a > cfg.max_bridge:
                continue
            m0 = yolo.nearest(lo + a - 1, px[a - 1, s], py[a - 1, s], cfg.gate)
            m1 = yolo.nearest(lo + b, px[b, s], py[b, s], cfg.gate)
            if m0 is None or m1 is None or m0[0] != m1[0]:
                continue                                     # not the same person on both sides
            if m0[1] not in (-1, int(team[s])):
                continue                                     # other team
            p = yolo.path(m0[0], lo + a, lo + b)
            have = np.isfinite(p[:, 0])
            if have.mean() < cfg.bridge_cover:
                continue
            # Offset YOLO onto ATD (the two systems differ by a small bias), then fill YOLO's own holes.
            t = np.arange(a, b)
            off = np.array([px[a - 1, s], py[a - 1, s]]) - yolo.path(m0[0], lo + a - 1, lo + a)[0]
            off1 = np.array([px[b, s], py[b, s]]) - yolo.path(m0[0], lo + b, lo + b + 1)[0]
            w = (t - a + 1) / (b - a + 1)
            q = p + (1 - w)[:, None] * off + w[:, None] * off1
            for c in range(2):
                q[~have, c] = np.interp(t[~have], t[have], q[have, c])
            full = np.vstack([[px[a - 1, s], py[a - 1, s]], q, [px[b, s], py[b, s]]])
            L = min(cfg.lag, len(full) - 1)
            moved = np.hypot(full[L:, 0] - full[:-L, 0], full[L:, 1] - full[:-L, 1])
            if (moved * cfg.fps / L > cfg.jump_speed).any():
                continue                                     # YOLO jumps too: don't copy a jump
            px[a:b, s], py[a:b, s] = q[:, 0], q[:, 1]
            filled[a:b, s] = True
            count += 1
    return count, filled


# -- 3. short linear fill --------------------------------------------------------------------------

def interp_gaps(px, py, cfg: FixConfig):
    n, S = px.shape
    filled = np.zeros((n, S), dtype=bool)
    count = 0
    for s in range(S):
        ok = np.isfinite(px[:, s])
        for a, b in _interior_gaps(ok):
            if b - a > cfg.max_interp:
                continue
            d = np.hypot(px[b, s] - px[a - 1, s], py[b, s] - py[a - 1, s])
            if d / (b - a + 1) * cfg.fps > cfg.max_gap_speed:
                continue
            t = np.arange(a, b)
            px[a:b, s] = np.interp(t, [a - 1, b], [px[a - 1, s], px[b, s]])
            py[a:b, s] = np.interp(t, [a - 1, b], [py[a - 1, s], py[b, s]])
            filled[a:b, s] = True
            count += 1
    return count, filled


# -- the pipeline ----------------------------------------------------------------------------------

def jump_rate(px, py, lag=5, speed=12.0, fps=FPS):
    """Jumps per 1,000 lag-steps (the notebook's jumps_per_1000) on a (n, slots) matrix."""
    with np.errstate(invalid="ignore"):
        sp = np.hypot(px[lag:] - px[:-lag], py[lag:] - py[:-lag]) * fps / lag
    ok = np.isfinite(sp)
    return 1000.0 * float(np.mean(sp[ok] > speed)) if ok.any() else float("nan")


def fix_players(px, py, team, yolo=None, cfg: FixConfig | None = None, lo=0):
    """px, py (n, slots) metres, row k = video frame lo + k. yolo: rows of our tracks or a YoloIndex.

    Returns (px_fixed, py_fixed, source, report)."""
    cfg = cfg or FixConfig()
    px = np.asarray(px, dtype=np.float64)
    py = np.asarray(py, dtype=np.float64)
    given = np.isfinite(px) & np.isfinite(py)
    report = {"config": cfg.to_dict(), "jumps_per_1000_before": jump_rate(px, py, cfg.lag, cfg.jump_speed, cfg.fps)}
    px2, py2, n_cut, cut = cut_jumps(px, py, cfg)
    n_dup, dup = remove_duplicates(px2, py2, cfg)
    cut |= dup
    n_bridge, bridged = 0, np.zeros_like(given)
    if yolo is not None:
        idx = yolo if isinstance(yolo, YoloIndex) else YoloIndex(yolo)
        n_bridge, bridged = bridge_gaps(px2, py2, team, idx, cfg, lo)
    n_interp, interped = interp_gaps(px2, py2, cfg)
    source = np.full(px2.shape, MISSING, dtype=np.int8)
    source[given & ~cut] = GIVEN                           # blanked frames: cut jumps and duplicates
    source[bridged] = BRIDGED
    source[interped] = INTERP
    report.update({
        "jumps_cut": int(n_cut), "duplicate_runs_removed": int(n_dup), "frames_blanked": int(cut.sum()),
        "gaps_bridged_from_yolo": int(n_bridge), "frames_bridged": int(bridged.sum()),
        "gaps_interpolated": int(n_interp), "frames_interpolated": int(interped.sum()),
        "player_frames_before": int(given.sum()), "player_frames_after": int(np.isfinite(px2).sum()),
        "jumps_per_1000_after": jump_rate(px2, py2, cfg.lag, cfg.jump_speed, cfg.fps)})
    return px2, py2, source, report


def save_fixed(path, px, py, source, lo, report):
    import json
    import os
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(path, px=px.astype(np.float32), py=py.astype(np.float32), source=source,
                        lo=np.array(int(lo)), report=np.array(json.dumps(report)))


def load_fixed(path):
    import json
    z = np.load(path, allow_pickle=False)
    return z["px"], z["py"], z["source"], int(z["lo"]), json.loads(str(z["report"]))
