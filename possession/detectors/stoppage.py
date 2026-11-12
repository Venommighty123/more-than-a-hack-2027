"""Dead-ball detector v2 (Ashwyn's labeller v2.1 / v4 rules), relative to the match's own pace.

Ported from backup/ashwyn_labeller_v2.py (dead_mask, rel_speed, _restart, _strike) and
backup/ashwyn_labeller_v4.py (RULE_STOP_MODE, weak evidence, LONG_SLOW). Logic unchanged: the
parameters stay in Ashwyn's dict (``DEFAULTS`` below) so his changelog entries map one to one.
``detectors/dead_ball.py`` is the older v1 detector (absolute speeds, gaps only).

    rel(t)  mean player speed, smoothed over SMOOTH_S, divided by its median over the graded frames.
            On the bench dead frames sit at ~0.6x and live frames at ~1.2x.

  A. ball out of the pitch: dead. FIX_OUT: once out, play restarts only with a strike inside the
     pitch (players keep running after the ball goes out).
  B. ball-gap stoppage: a gap of at least GAP_MIN s is dead if it started or ended near a line, or
     the players were slow through it (median rel < GAP_REL).
  C. ball-present stoppage (RULE_STOP_MODE): players collectively slow (rel < STOP_REL) for STOP_MIN
     s and the ball quiet. v4 turns it OFF: on the realistic bench copies it marks slow build-up as
     dead (+0.02 spellF1 without it, all 6 game copies up). "weak" keeps it only with independent
     evidence (tag, ball just out, ball near a line).
  D. set-piece tag (USE_TAGS): dead from when the players slowed before the tag to the first strike.
  E. LONG_SLOW (off, rejected on the bench): rel < LONG_SLOW_REL for LONG_SLOW s is dead.
  Restart (end of D): the ball is struck (> RESTART_V m/s) or the players move as in open play
  again (rel > LIVE_REL). Live slivers shorter than D_JOIN s between stoppages are joined.

Tuned on the Metrica bench only (docs/changelog/labeller.md, backup/ashwyn_labeller_v1_CHANGELOG.md).
"""
from __future__ import annotations

import numpy as np

from possession.preprocess.ball import windowed_velocity
from possession.preprocess.players import mean_speed, velocity
from possession.utils.frames import runs

FPS = 25
PITCH = (105.0, 68.0)

DEFAULTS = dict(
    OUT_M=0.5, D_JOIN=1.0,
    SMOOTH_S=3.0,
    GAP_MIN=1.5, GAP_REL=0.75, DEAD_LINE=2.0,
    STOP_REL=0.7, STOP_MIN=4.0, QUIET_V=2.0, QUIET_FRAC=0.5, EXT_REL=0.95, BACK_MAX=10.0,
    TAG_WIN=8.0, TAG_BACK_MAX=15.0,
    RESTART_V=5.0, RESTART_MAX=30.0, LIVE_REL=1.0,
    USE_TAGS=True,
    RULE_OUT=True, RULE_GAP_LINE=True, RULE_GAP_SLOW=True,
    RULE_STOP_MODE="off",              # v4: "off" | "full" (v2.1) | "weak"
    FIX_CARRY=True, CARRY_V=4.0, LINE_QUIET_M=3.0,
    FIX_OUT=True, OUT_MAX=60.0, OUT_MIN=8.0,
    FIX_DROP=False, DROP_RATIO=0.5, DROP_REL_MAX=1.0, DROP_BEFORE_S=5.0,
    FIX_LINE=False, LINE_REL=1.0,
    TAG_STRIKE=True, TAG_MIN_BACK=0.0, TAG_STRIKE_MAX=8.0,
    WEAK_PRE=3.0, WEAK_LINE_M=3.0, WEAK_LINE_FRAC=0.3,
    LONG_SLOW=None, LONG_SLOW_REL=0.7, LONG_SLOW_EXT=False,
)


def ball_speed(b, fps=FPS, half=4):
    """Ball speed from the displacement over +/- half frames (labeller_v1.ball_speed)."""
    vx, vy = windowed_velocity(b[:, 0], b[:, 1], fps, half)
    return np.hypot(vx, vy)


def rel_speed(px, py, blocks, p, fps=FPS):
    """Mean player speed relative to the match's own median over the graded blocks."""
    _, _, sp = velocity(px, py, fps, half=12, max_speed_ms=12.0)
    ms = mean_speed(sp, None, fps, smooth_s=1.0)
    k = max(1, int(p["SMOOTH_S"] * fps))
    ms = np.convolve(ms, np.ones(k) / k, mode="same")
    graded = np.zeros(len(ms), dtype=bool)
    for _, lo, hi in blocks:
        graded[int(lo):int(hi) + 1] = True
    med = np.median(ms[graded]) if graded.any() else np.median(ms)
    return ms / max(med, 1e-6)


def _restart(k, b, v, rel, p, hi, fps=FPS):
    """First frame at or after k where play restarts (ball struck or players back at pace)."""
    lim = min(hi + 1, k + int(p["RESTART_MAX"] * fps))
    r = k
    while r < lim:
        if not np.isnan(b[r, 0]) and v[r] > p["RESTART_V"]:
            return r
        if rel[r] > p["LIVE_REL"]:
            return r
        r += 1
    return lim


def _strike(k, b, v, p, hi, limit_s, fps=FPS, pitch=PITCH):
    """First frame at or after k where the ball, inside the pitch, is struck; else k + limit."""
    L, W = pitch
    lim = min(hi + 1, k + int(limit_s * fps))
    for r in range(k, lim):
        x, y = b[r]
        if not np.isnan(x) and v[r] > p["RESTART_V"] and -p["OUT_M"] <= x <= L + p["OUT_M"] \
                and -p["OUT_M"] <= y <= W + p["OUT_M"]:
            return r
    return lim


def _out_mask(b, p, pitch):
    L, W = pitch
    present = ~np.isnan(b[:, 0])
    with np.errstate(invalid="ignore"):
        return present & ((b[:, 0] < -p["OUT_M"]) | (b[:, 0] > L + p["OUT_M"]) |
                          (b[:, 1] < -p["OUT_M"]) | (b[:, 1] > W + p["OUT_M"]))


def _quiet(b, v, out, p, pitch):
    L, W = pitch
    present = ~np.isnan(b[:, 0])
    with np.errstate(invalid="ignore"):
        quiet = ~present | out | (present & ~(v > p["QUIET_V"]))
        if p["FIX_CARRY"]:
            m = p["LINE_QUIET_M"]
            nl = present & ((b[:, 0] < m) | (b[:, 0] > L - m) | (b[:, 1] < m) | (b[:, 1] > W - m))
            quiet = quiet | (present & ~(v > p["CARRY_V"])) | nl
    return quiet


def _slow_candidates(b, v, rel, out, p, lo, hi, fps, pitch):
    """Rule C stretches [s0, r): collective slowness + quiet ball, walked back, to the restart."""
    slow = rel < p["STOP_REL"]
    slow[:lo] = False
    slow[hi + 1:] = False
    quiet = _quiet(b, v, out, p, pitch)
    for s, e in zip(*runs(slow)):
        if e - s < p["STOP_MIN"] * fps or quiet[s:e].mean() < p["QUIET_FRAC"]:
            continue
        s0 = s
        stop = max(lo, s - int(p["BACK_MAX"] * fps))
        while s0 > stop and rel[s0 - 1] < p["EXT_REL"] and not (v[s0 - 1] > p["RESTART_V"] * 2):
            s0 -= 1
        yield s0, _restart(e, b, v, rel, p, hi, fps)


def _weak_evidence(b, s0, r, out, tags, p, lo, fps, pitch):
    """Independent dead-ball evidence around a rule-C stretch [s0, r) (v4 'weak' mode)."""
    L, W = pitch
    pre = int(p["WEAK_PRE"] * fps)
    if tags is not None and p["USE_TAGS"]:
        t = np.asarray(tags, dtype=int)
        if ((t >= s0 - pre) & (t <= r + int(p["TAG_WIN"] * fps))).any():
            return True
    if out[max(lo, s0 - pre):s0 + 1].any():
        return True
    seg = b[s0:r]
    pres = ~np.isnan(seg[:, 0])
    if pres.any():
        q = seg[pres]
        m = p["WEAK_LINE_M"]
        nl = (q[:, 0] < m) | (q[:, 0] > L - m) | (q[:, 1] < m) | (q[:, 1] > W - m)
        if nl.mean() >= p["WEAK_LINE_FRAC"]:
            return True
    return False


def dead_mask(b, rel, p, lo, hi, tags=None, fps=FPS, pitch=PITCH):
    """Dead-ball mask (n,) for one graded block lo..hi. ``b`` is the cleaned ball (n, 2)."""
    p = dict(DEFAULTS, **p)
    n = len(b)
    dead = np.zeros(n, dtype=bool)
    L, W = pitch
    v = ball_speed(b, fps)
    present = ~np.isnan(b[:, 0])
    out = _out_mask(b, p, pitch)

    # A. out of the pitch
    if p["RULE_OUT"]:
        dead |= out
        if p["FIX_OUT"]:
            o = out.copy()
            o[:lo] = False
            o[hi + 1:] = False
            for s, e in zip(*runs(o)):
                r = _strike(e, b, v, p, hi, p["OUT_MAX"], fps, pitch)
                k = e + int(p["OUT_MIN"] * fps)
                while k < r and not (np.isnan(b[k, 0]) and rel[k] > p["LIVE_REL"]):
                    k += 1
                dead[s:min(r, k)] = True

    def near_line(q):
        return (not np.isnan(q[0])) and (q[0] < p["DEAD_LINE"] or q[0] > L - p["DEAD_LINE"] or
                                         q[1] < p["DEAD_LINE"] or q[1] > W - p["DEAD_LINE"])

    # B. ball-gap stoppages
    miss = ~present
    miss[:lo] = False
    miss[hi + 1:] = False
    for s, e in zip(*runs(miss)):
        if e - s < p["GAP_MIN"] * fps:
            continue
        before = b[s - 1] if s > 0 else np.array([np.nan, np.nan])
        after = b[e] if e < n else np.array([np.nan, np.nan])
        med = np.median(rel[s:e])
        slow = med < p["GAP_REL"]
        line = near_line(before) or near_line(after)
        pre = rel[max(lo, s - int(p["DROP_BEFORE_S"] * fps)):s]
        drop = (len(pre) > 0 and med / max(np.median(pre), 1e-6) < p["DROP_RATIO"]
                and med < p["DROP_REL_MAX"])
        if p["FIX_LINE"]:
            line = line and (med < p["LINE_REL"] or drop)
        if p["FIX_DROP"] and drop:
            slow = True
        if (p["RULE_GAP_LINE"] and line) or (p["RULE_GAP_SLOW"] and slow):
            dead[s:_restart(e, b, v, rel, p, hi, fps)] = True

    # C. ball-present stoppages (v4: off by default)
    mode = p["RULE_STOP_MODE"]
    if mode in ("full", "weak"):
        add = np.zeros(n, dtype=bool)
        for s0, r in _slow_candidates(b, v, rel, out, p, lo, hi, fps, pitch):
            if mode == "full" or _weak_evidence(b, s0, r, out, tags, p, lo, fps, pitch):
                add[s0:r] = True
        if mode == "full":
            dead |= add
            add = None
    else:
        add = None

    # D. set-piece tags
    if tags is not None and p["USE_TAGS"]:
        win = int(p["TAG_WIN"] * fps)
        for t in np.asarray(tags, dtype=int):
            if t < lo or t > hi:
                continue
            if p["TAG_STRIKE"]:
                j = t - int(p["TAG_MIN_BACK"] * fps)
                stop = max(lo, t - int(p["TAG_BACK_MAX"] * fps))
                while j > stop and (dead[j - 1] or rel[j - 1] < p["EXT_REL"]):
                    j -= 1
                dead[max(lo, j):_strike(t, b, v, p, hi, p["TAG_STRIKE_MAX"], fps, pitch)] = True
                continue
            if dead[max(lo, t - win):t + 1].any():
                j = t
                while j > lo and not dead[j]:
                    j -= 1
                dead[j:_restart(t, b, v, rel, p, hi, fps)] = True
                continue
            j = t
            stop = max(lo, t - int(p["TAG_BACK_MAX"] * fps))
            while j > stop and rel[j] < p["STOP_REL"]:
                j -= 1
            j = max(j, t - int(p["TAG_BACK_MAX"] * fps))
            if t - j < fps:
                j = t - fps
            dead[max(lo, j):_restart(t, b, v, rel, p, hi, fps)] = True

    def join(d):
        for s, e in zip(*runs(~d)):
            if s > 0 and e < n and d[s - 1] and d[e] and e - s < p["D_JOIN"] * fps:
                d[s:e] = True

    join(dead)
    # E. long collective slowdown (rejected on the bench, kept as a switch)
    if p["LONG_SLOW"]:
        sl = rel < p["LONG_SLOW_REL"]
        sl[:lo] = False
        sl[hi + 1:] = False
        for s, e in zip(*runs(sl)):
            if e - s >= p["LONG_SLOW"] * fps:
                dead[s:(_restart(e, b, v, rel, p, hi, fps) if p["LONG_SLOW_EXT"] else e)] = True
    if add is not None:              # "weak": stretches with independent evidence, then re-join
        dead |= add
        join(dead)
    return dead
