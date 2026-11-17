"""Possession labeller v6: v4 plus switches that target false possession changes and false stoppages.

DEFAULTS (Nov 8 bench verdict, see the changelog): SMOOTH_K=3, SW_MARGIN=0.6, MIN_D=2.0 with
MIN_D_KEEP_OUT=False. Everything else = v4. All switches off reproduces v4 exactly:
    params=dict(SMOOTH_K=0, SW_MARGIN=None, MIN_D=0.0)

    label(px, py, ball, team, blocks, params=None, tags=None) -> "H" / "A" / "D" / "X"

WHY. On the tracking-based bench truth (test_scripts/bench/build_trk_gt.py), v4 predicts 640
changes where there are 585, and 40% of its changes are false. Sorted (6 realistic game copies,
549 false changes):
  56%  right team but more than 2 s off; mostly LATE (median 8.5 s), with the new team
       controlling the ball on 18% of the frames in between and the old team on 6%
  12%  wrong team, then a stoppage
  21%  A-B-A blips: a short spell of the other team between two spells of the same team; in it
       the "new" team controls the ball on only 18% of its frames
   8%  wrong team picked after a stoppage
So the errors are of two kinds: switching on too little evidence (blips) and switching too late
when the evidence is spread out (lateness). Each fix below has its own switch.

 1. SMOOTH_K   centred moving average of ball and player positions over +/- SMOOTH_K frames
               before the control test (0 = off).
 2. SW_MARGIN  hysteresis: to TAKE the ball from the current team, the other team must be at
               least SW_MARGIN m clearer (MARGIN still applies to the current team). None = off.
 3. WEAK_MERGE an A-B-A spell is merged into A when B controls the ball on fewer than WEAK_SHARE
               of its frames, up to WEAK_MAX s long (MIN_SPELL still merges anything shorter).
 5. FOLLOW     the ball is with the player whose velocity it matches: among players within
               FOLLOW_R m, the one with the smallest |v_ball - v_player| (both over +/- VEL_W
               frames) controls it if that is below FOLLOW_DV m/s, the ball moves faster than
               FOLLOW_MINV m/s, and the best opponent is at least FOLLOW_M m/s worse. A ~1 m offset
               that drifts every second hides who is NEAREST, but not who the ball MOVES WITH.
               FOLLOW_ONLY_FILL: only used where the distance rule found nobody.
 4. RESOLVE_MODE "density" (from v5): switch on DENS_N of the last DENS_W frames, which answers
               the lateness.
"""

import numpy as np

import labeller_v1 as v1
import labeller_v2 as v2
import labeller_v4 as v4
import labeller_v5 as v5

FPS = v1.FPS

DEFAULTS = dict(v4.DEFAULTS)
DEFAULTS.update(
    SMOOTH_K=3,        # v6 default (bench: with SW_MARGIN 0.6, +0.011 on both truths)
    SW_MARGIN=0.6,     # v6 default
    WEAK_MERGE=False, WEAK_SHARE=0.3, WEAK_MAX=8.0,
    FOLLOW=False, FOLLOW_R=3.0, FOLLOW_DV=1.5, FOLLOW_M=1.0, FOLLOW_MINV=1.5, FOLLOW_ONLY_FILL=True,
    MIN_D=2.0, MIN_D_KEEP_OUT=False,   # v6 default: +0.013, up in all 12 cells
    GAP_SPLIT_REL=None, GAP_SPLIT_SMOOTH=1.0, TAG_BACK_MAX_V6=None, GAP_START_DELAY=0.0,
    RESOLVE_MODE="run", DENS_W=25, DENS_N=6, DENS_M=3,
)


def smooth(a, k):
    """NaN-aware centred moving average along axis 0."""
    if k <= 0:
        return a
    w = 2 * k + 1
    ok = ~np.isnan(a)
    z = np.where(ok, a, 0.0)
    cs = np.cumsum(np.concatenate([np.zeros((1,) + a.shape[1:]), z]), 0)
    cn = np.cumsum(np.concatenate([np.zeros((1,) + a.shape[1:]), ok.astype(float)]), 0)
    n = len(a)
    lo = np.clip(np.arange(n) - k, 0, n)
    hi = np.clip(np.arange(n) + k + 1, 0, n)
    s = cs[hi] - cs[lo]
    c = cn[hi] - cn[lo]
    out = np.where(ok, s / np.maximum(c, 1), np.nan)   # keep missing frames missing
    return out


def resolve_hyst(ctrl, ctrl_sw, dead, lo, hi, p):
    """v1.resolve, except that evidence AGAINST the current team comes from ctrl_sw (stricter)."""
    out = np.full(hi - lo + 1, -1, dtype=np.int8)
    cur = -1
    cand, cand_start, cand_len, holes = -1, 0, 0, 0
    pending = None
    for k in range(lo, hi + 1):
        i = k - lo
        if dead[k]:
            out[i] = 2
            cur, cand, cand_len, holes = -1, -1, 0, 0
            continue
        c = ctrl[k]
        if cur >= 0 and c == 1 - cur and ctrl_sw[k] != c:
            c = -1                     # not clear enough to count against the current team
        if c >= 0:
            if c == cand:
                cand_len += 1 + holes
                holes = 0
            else:
                cand, cand_start, cand_len, holes = c, i, 1, 0
        elif cand >= 0:
            holes += 1
            if holes > p["HOLE"]:
                cand, cand_len, holes = -1, 0, 0
        if cur == -1:
            if pending is None:
                pending = i
            if cand >= 0 and cand_len >= 1:
                cur = cand
                out[pending:i + 1] = cur
                pending = None
                continue
            out[i] = -1
            continue
        pending = None
        if cand >= 0 and cand != cur and cand_len >= p["CONFIRM"]:
            cur = cand
            out[cand_start:i + 1] = cur
            continue
        out[i] = cur
    unk = out == -1
    if unk.any() and (~unk).any():
        idx = np.flatnonzero(~unk)
        j = np.clip(np.searchsorted(idx, np.flatnonzero(unk)), 0, len(idx) - 1)
        out[unk] = out[idx[j]]
    out[out == -1] = 2
    return out


def merge_weak(codes, ctrl, p):
    """Merge A-B-A spells where B has little control evidence (block-relative arrays)."""
    c = codes.copy()
    mx = int(p["WEAK_MAX"] * FPS)
    changed = True
    while changed:
        changed = False
        edges = np.flatnonzero(np.diff(c)) + 1
        starts = np.r_[0, edges]
        ends = np.r_[edges, len(c)]
        for j in range(1, len(starts) - 1):
            s, e = starts[j], ends[j]
            a0, a1 = c[starts[j - 1]], c[starts[j + 1]]
            if c[s] < 2 and a0 == a1 and a0 < 2 and a0 != c[s] and e - s <= mx \
                    and np.mean(ctrl[s:e] == c[s]) < p["WEAK_SHARE"]:
                c[s:e] = a0
                changed = True
                break
    return c


def follow_control(px, py, b, team, vx, vy, ctrl, ok, p):
    w = p["VEL_W"]
    n = len(b)
    bvx = np.full(n, np.nan)
    bvy = np.full(n, np.nan)
    bvx[w:-w] = (b[2 * w:, 0] - b[:-2 * w, 0]) * FPS / (2 * w)
    bvy[w:-w] = (b[2 * w:, 1] - b[:-2 * w, 1]) * FPS / (2 * w)
    sel = ok & ~np.isnan(bvx) & (np.hypot(bvx, bvy) > p["FOLLOW_MINV"])
    if p["FOLLOW_ONLY_FILL"]:
        sel &= ctrl < 0
    if not sel.any():
        return ctrl
    dist = np.hypot(px[sel] - b[sel, 0:1], py[sel] - b[sel, 1:2])
    dv = np.hypot(vx[sel] - bvx[sel, None], vy[sel] - bvy[sel, None])
    dv = np.where((dist < p["FOLLOW_R"]) & ~np.isnan(dist), dv, np.inf)
    bh = dv[:, team == 0].min(1)
    ba = dv[:, team == 1].min(1)
    c = np.full(sel.sum(), -1, dtype=np.int8)
    c[(bh < p["FOLLOW_DV"]) & (ba > bh + p["FOLLOW_M"])] = 0
    c[(ba < p["FOLLOW_DV"]) & (bh > ba + p["FOLLOW_M"])] = 1
    out = ctrl.copy()
    fill = c >= 0
    idx = np.flatnonzero(sel)[fill]
    out[idx] = c[fill]
    return out


def fix_dead(dead, b, p, lo, hi, rel=None):
    """Stoppage clean-up (v6 dead-ball timing switches).
    MIN_D: a D stretch shorter than MIN_D s is live play (a stoppage takes longer than that), unless
    the ball is off the pitch in it and MIN_D_KEEP_OUT is set. GAP_START_DELAY: a D stretch that
    starts where the ball vanishes starts that many s later (the ball is often still in play for a
    moment after the tracker loses it)."""
    d = dead.copy()
    L, W = v1.PITCH
    if p["GAP_START_DELAY"]:
        k = int(p["GAP_START_DELAY"] * FPS)
        miss = np.isnan(b[:, 0])
        for s, e in zip(*v1._runs(d)):
            if s > lo and miss[s] and not miss[s - 1]:
                d[s:min(e, s + k)] = False
    if p["GAP_SPLIT_REL"] is not None and rel is not None:
        # A stoppage that starts where the ball vanishes often starts LATER, inside the gap: the
        # ball was still in play, just not tracked. Start D where the players slow down
        # (rel below GAP_SPLIT_REL) instead; the frames before keep the last team.
        miss = np.isnan(b[:, 0])
        for s, e in zip(*v1._runs(d)):
            if s > lo and miss[s] and not miss[s - 1]:
                g = s
                while g < e and miss[g]:
                    g += 1
                j = s
                while j < g and rel[j] >= p["GAP_SPLIT_REL"]:
                    j += 1
                if j < g:
                    d[s:j] = False
    if p["MIN_D"]:
        out = ~np.isnan(b[:, 0]) & ((b[:, 0] < -p["OUT_M"]) | (b[:, 0] > L + p["OUT_M"]) |
                                    (b[:, 1] < -p["OUT_M"]) | (b[:, 1] > W + p["OUT_M"]))
        for s, e in zip(*v1._runs(d)):
            if s < lo or e > hi + 1:
                continue
            if e - s < p["MIN_D"] * FPS and not (p["MIN_D_KEEP_OUT"] and out[s:e].any()):
                d[s:e] = False
    return d


def label(px, py, ball, team, blocks, params=None, tags=None):
    p = dict(DEFAULTS, **(params or {}))
    team = np.asarray(team)
    if p["SMOOTH_K"]:
        px, py, ball = smooth(px, p["SMOOTH_K"]), smooth(py, p["SMOOTH_K"]), smooth(ball, p["SMOOTH_K"])
    b = v1.clean_ball(ball, p)
    rel = v2.rel_speed(px, py, blocks, p)
    vx, vy = v4.player_velocity(px, py, p["VEL_W"])
    d, ok = v4.eff_dist(px, py, b, vx, vy, p)
    ctrl = v4.control(px, py, b, team, p, d, ok)
    if p["FOLLOW"]:
        ctrl = follow_control(px, py, b, team, vx, vy, ctrl, ok, p)
    ctrl_sw = None
    if p["SW_MARGIN"] is not None:
        ctrl_sw = v4.control(px, py, b, team, dict(p, MARGIN=p["SW_MARGIN"]), d, ok)
    out = np.full(len(ball), "X", dtype="<U1")
    for _, lo, hi in blocks:
        lo, hi = int(lo), int(hi)
        q = p if p["TAG_BACK_MAX_V6"] is None else dict(p, TAG_BACK_MAX=p["TAG_BACK_MAX_V6"],
                                                          BACK_MAX=p["TAG_BACK_MAX_V6"])
        dead = v4.dead_mask(b, rel, q, lo, hi, tags)
        dead = fix_dead(dead, b, p, lo, hi, rel)
        if p["RESOLVE_MODE"] == "density":
            codes = v5.resolve_density(ctrl, dead, lo, hi, p)
        elif ctrl_sw is not None:
            codes = resolve_hyst(ctrl, ctrl_sw, dead, lo, hi, p)
        else:
            codes = v1.resolve(ctrl, dead, lo, hi, p)
        codes = v1.merge_short(codes, int(p["MIN_SPELL"] * FPS))
        if p["WEAK_MERGE"]:
            codes = merge_weak(codes, ctrl[lo:hi + 1], p)
        out[lo:hi + 1] = np.array(["H", "A", "D"])[codes]
    return out
