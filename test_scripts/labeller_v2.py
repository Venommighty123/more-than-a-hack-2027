"""Possession labeller v2.1. It keeps v1's possession logic and replaces the dead-ball detector.

v2.1 changes are listed in test_scripts/labeller_v1_CHANGELOG.md; the switches FIX_* keep each one
measurable on its own.

    label(px, py, ball, team, blocks, params=None, tags=None) -> array of "H" / "A" / "D" / "X"

    tags   optional video frames of Smart Tagging set pieces (throw-in, corner, goal kick, free kick,
           kickoff). Every one marks a restart, so the ball was dead just before it.

WHY v2. Run on BAR-ATH, v1 labelled D almost exactly where the ball was missing, which is wrong in
both directions:
  - v1 used an absolute player-speed threshold of 1.8 m/s. With 46 fragmented slots, BAR-ATH's mean
    player speed has a median of only 1.9 m/s in play, so the threshold fired in nearly every gap.
  - In BAR-ATH the ball is often still tracked through a stoppage, while it is carried or placed
    for the restart. v1 could only find a stoppage when the ball went missing.

DEAD-BALL EVIDENCE IN v2. Every speed is relative to the match's own median, which is computed from
the tracking with no labels:
  rel(t)  mean player speed, smoothed over SMOOTH_S, divided by the median over the graded frames.
          On the bench, dead frames sit at about 0.6x and live frames at about 1.2x
          (AUC 0.84-0.89 on its own).
  A. Ball out of the pitch: dead.
  B. Ball-gap stoppage. A gap of at least GAP_MIN s is dead if it started or ended near a line, or
     if the players were slow through it (median rel < GAP_REL).
  C. Ball-present stoppage. The players are collectively slow (rel < STOP_REL) for at least STOP_MIN
     s, and the ball is quiet through it: on at least QUIET_FRAC of the frames it is missing, slower
     than QUIET_V m/s, or off the pitch. The quiet test separates a stoppage from a slow build-up,
     where the ball keeps moving between players.
  D. Set-piece tag. If no stoppage was found within TAG_WIN s before a tag, one is created, running
     from the last moment the players were moving (rel above STOP_REL) to the restart.
     The start is walked back while the players are still slower than EXT_REL (at most BACK_MAX
     s), because players decelerate after the whistle. On the bench, rel falls from about 1.0 to
     0.6 through a stoppage.
  Restart, which is the end of D: the first frame after the stoppage where the ball moves faster
  than RESTART_V m/s, or where the players move as in open play again (rel > LIVE_REL). Without
  the second test, a ball that reappears at a dribbler's feet kept D running until the next pass.
"""

import numpy as np

import labeller_v1 as v1

FPS = v1.FPS
PITCH = v1.PITCH

DEFAULTS = dict(v1.DEFAULTS)
DEFAULTS.update(
    SMOOTH_S=3.0,
    GAP_MIN=1.5, GAP_REL=0.75, DEAD_LINE=2.0,
    STOP_REL=0.7, STOP_MIN=4.0, QUIET_V=2.0, QUIET_FRAC=0.5, EXT_REL=0.95, BACK_MAX=10.0,
    TAG_WIN=8.0, TAG_BACK_MAX=15.0,
    RESTART_V=5.0, RESTART_MAX=30.0, LIVE_REL=1.0, OUT_M=0.5, D_JOIN=1.0,
    USE_TAGS=True,           # v2.1: with tags simulated as measured on BAR-ATH they help (+0.003)
    RULE_OUT=True, RULE_GAP_LINE=True, RULE_GAP_SLOW=True, RULE_STOP=True,  # switches for diagnosis
    # v2.1 fixes (each has a switch so the bench can measure it alone)
    FIX_CARRY=True, CARRY_V=4.0, LINE_QUIET_M=3.0,
    FIX_OUT=True, OUT_MAX=60.0, OUT_MIN=8.0,
    FIX_DROP=False, DROP_RATIO=0.5, DROP_REL_MAX=1.0, DROP_BEFORE_S=5.0,
    FIX_LINE=False, LINE_REL=1.0,
    TAG_STRIKE=True, TAG_MIN_BACK=0.0, TAG_STRIKE_MAX=8.0,
)


def rel_speed(px, py, blocks, p):
    ms = v1.player_speed(px, py)
    k = max(1, int(p["SMOOTH_S"] * FPS))
    ms = np.convolve(ms, np.ones(k) / k, mode="same")
    graded = np.zeros(len(ms), dtype=bool)
    for _, lo, hi in blocks:
        graded[int(lo):int(hi) + 1] = True
    med = np.median(ms[graded]) if graded.any() else np.median(ms)
    return ms / max(med, 1e-6)


def _restart(k, b, v, rel, p, hi):
    """First frame at or after k where play restarts."""
    lim = min(hi + 1, k + int(p["RESTART_MAX"] * FPS))
    r = k
    while r < lim:
        if not np.isnan(b[r, 0]) and v[r] > p["RESTART_V"]:
            return r                      # the ball is struck
        if rel[r] > p["LIVE_REL"]:
            return r                      # the players are moving as in open play again
        r += 1
    return lim


def _strike(k, b, v, p, hi, limit_s):
    """First frame at or after k where the ball, inside the pitch, is struck; else k + limit."""
    L, W = PITCH
    lim = min(hi + 1, k + int(limit_s * FPS))
    for r in range(k, lim):
        x, y = b[r]
        if not np.isnan(x) and v[r] > p["RESTART_V"] and -p["OUT_M"] <= x <= L + p["OUT_M"] \
                and -p["OUT_M"] <= y <= W + p["OUT_M"]:
            return r
    return lim


def dead_mask(b, rel, p, lo, hi, tags=None):
    n = len(b)
    dead = np.zeros(n, dtype=bool)
    L, W = PITCH
    v = v1.ball_speed(b)
    present = ~np.isnan(b[:, 0])
    out = present & ((b[:, 0] < -p["OUT_M"]) | (b[:, 0] > L + p["OUT_M"]) |
                     (b[:, 1] < -p["OUT_M"]) | (b[:, 1] > W + p["OUT_M"]))
    # A. out of the pitch
    if p["RULE_OUT"]:
        dead |= out
        if p["FIX_OUT"]:
            # v2.1: once the ball has left the pitch, play only restarts with a strike. The players
            # often keep running after the ball goes out, so their speed must not end the stoppage.
            o = out.copy()
            o[:lo] = False
            o[hi + 1:] = False
            for s, e in zip(*v1._runs(o)):
                r = _strike(e, b, v, p, hi, p["OUT_MAX"])
                # if the ball is never seen again, fall back on the players after OUT_MIN s
                k = e + int(p["OUT_MIN"] * FPS)
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
    for s, e in zip(*v1._runs(miss)):
        if e - s < p["GAP_MIN"] * FPS:
            continue
        before = b[s - 1] if s > 0 else np.array([np.nan, np.nan])
        after = b[e] if e < n else np.array([np.nan, np.nan])
        med = np.median(rel[s:e])
        slow = med < p["GAP_REL"]
        line = near_line(before) or near_line(after)
        pre = rel[max(lo, s - int(p["DROP_BEFORE_S"] * FPS)):s]
        drop = (len(pre) > 0 and med / max(np.median(pre), 1e-6) < p["DROP_RATIO"]
                and med < p["DROP_REL_MAX"])
        if p["FIX_LINE"]:
            # v2.1: a gap that starts near a line is dead only if the players are not in full flow
            line = line and (med < p["LINE_REL"] or drop)
        if p["FIX_DROP"] and drop:
            # v2.1: a sudden collective slowdown (whistle) is dead-ball evidence by itself
            slow = True
        if (p["RULE_GAP_LINE"] and line) or (p["RULE_GAP_SLOW"] and slow):
            dead[s:_restart(e, b, v, rel, p, hi)] = True

    # C. ball-present stoppages: collective slowness + quiet ball
    slow = rel < p["STOP_REL"]
    slow[:lo] = False
    slow[hi + 1:] = False
    quiet = miss | out | (present & ~(v > p["QUIET_V"]))
    if p["FIX_CARRY"]:
        # v2.1: a ball carried to a restart moves at walking / jogging pace, ~3 m/s, and a ball
        # being placed for a throw-in sits on the touchline
        nl = present & ((b[:, 0] < p["LINE_QUIET_M"]) | (b[:, 0] > L - p["LINE_QUIET_M"]) |
                        (b[:, 1] < p["LINE_QUIET_M"]) | (b[:, 1] > W - p["LINE_QUIET_M"]))
        quiet = quiet | (present & ~(v > p["CARRY_V"])) | nl
    for s, e in zip(*v1._runs(slow & bool(p["RULE_STOP"]))):
        if e - s < p["STOP_MIN"] * FPS:
            continue
        if quiet[s:e].mean() < p["QUIET_FRAC"]:
            continue
        # hysteresis: the stoppage began before the players were fully slow (they decelerate
        # after the whistle), so walk back while they are still slower than EXT_REL
        s0 = s
        stop = max(lo, s - int(p["BACK_MAX"] * FPS))
        while s0 > stop and rel[s0 - 1] < p["EXT_REL"] and not (v[s0 - 1] > p["RESTART_V"] * 2):
            s0 -= 1
        dead[s0:_restart(e, b, v, rel, p, hi)] = True

    # D. set-piece tags
    if tags is not None and p["USE_TAGS"]:
        win = int(p["TAG_WIN"] * FPS)
        for t in np.asarray(tags, dtype=int):
            if t < lo or t > hi:
                continue
            if p["TAG_STRIKE"]:
                # v2.1: a real tag starts INSIDE the stoppage, a median 6.3 s before the kick
                # (measured on BAR-ATH from tracking only). Dead from the moment the players slowed
                # (at least TAG_MIN_BACK s before the tag) to the first strike after it.
                j = t - int(p["TAG_MIN_BACK"] * FPS)
                stop = max(lo, t - int(p["TAG_BACK_MAX"] * FPS))
                while j > stop and (dead[j - 1] or rel[j - 1] < p["EXT_REL"]):
                    j -= 1
                dead[max(lo, j):_strike(t, b, v, p, hi, p["TAG_STRIKE_MAX"])] = True
                continue
            if dead[max(lo, t - win):t + 1].any():
                # extend the found stoppage to the tagged restart if it ended before it
                j = t
                while j > lo and not dead[j]:
                    j -= 1
                dead[j:_restart(t, b, v, rel, p, hi)] = True
                continue
            j = t
            stop = max(lo, t - int(p["TAG_BACK_MAX"] * FPS))
            while j > stop and rel[j] < p["STOP_REL"]:
                j -= 1
            j = max(j, t - int(p["TAG_BACK_MAX"] * FPS))
            if t - j < FPS:            # players never slowed down: a short stoppage before the tag
                j = t - FPS
            dead[max(lo, j):_restart(t, b, v, rel, p, hi)] = True

    for s, e in zip(*v1._runs(~dead)):
        if s > 0 and e < n and dead[s - 1] and dead[e] and e - s < p["D_JOIN"] * FPS:
            dead[s:e] = True
    return dead


def label(px, py, ball, team, blocks, params=None, tags=None):
    p = dict(DEFAULTS, **(params or {}))
    team = np.asarray(team)
    b = v1.clean_ball(ball, p)
    ctrl = v1.control(px, py, b, team, p)
    rel = rel_speed(px, py, blocks, p)
    out = np.full(len(ball), "X", dtype="<U1")
    for _, lo, hi in blocks:
        lo, hi = int(lo), int(hi)
        dead = dead_mask(b, rel, p, lo, hi, tags)
        codes = v1.resolve(ctrl, dead, lo, hi, p)
        codes = v1.merge_short(codes, int(p["MIN_SPELL"] * FPS))
        out[lo:hi + 1] = np.array(["H", "A", "D"])[codes]
    return out
