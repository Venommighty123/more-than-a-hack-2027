"""Possession labeller v1: per-frame H / A / D from tracking geometry alone.

Source-agnostic on purpose: the same function runs on the Metrica bench (in the cloud, public data)
and on BAR-ATH (on Ashwyn's PC). It never sees ground truth.

    label(px, py, ball, team, blocks, params=None) -> array of "H" / "A" / "D" / "X"

    px, py   (frames x slots) players in metres, NaN when absent
    ball     (frames x 2) ball in metres, NaN when not tracked
    team     team of each slot, 0 home / 1 away
    blocks   [(period, first_frame, last_frame)] graded stretches; everything outside is "X"

LOGIC (each step answers one rule of CHALLENGE.md)
 1. Clean the ball: drop teleports (> MAX_STEP m in one frame), bridge holes <= BRIDGE frames.
 2. Control per frame: a team controls the ball when its nearest player is within CTRL_R m, the
    other team's nearest is at least MARGIN m further away, and the ball is slower than CTRL_V m/s.
    Otherwise nobody controls it (loose / contested / in flight).
 3. Possession changes only on CONTROLLED possession: the other team must control the ball for
    CONFIRM frames (gaps of nobody up to HOLE frames allowed) before possession switches. A touch,
    a deflection or a won header without control never reaches CONFIRM, so possession stays with
    the last team, as the rules say. The switch is backdated to the first frame of that control,
    because spellF1 punishes lateness (1 s late costs about a third of a spell's score).
 4. Dead ball: the ball outside the pitch is dead. A ball gap of at least DEAD_MIN s is dead when it
    started near a line (DEAD_LINE m) or when the players were slow through it (median mean speed
    < DEAD_SPEED m/s; our BAR-ATH gap audit: dead gaps ~1.1 m/s, live gaps faster). D runs from the
    moment the ball vanished to the restart: the first frame after it reappears where it moves
    faster than RESTART_V m/s (it often sits at the restarter's feet first).
    After a stoppage the possessor is the first team to control the ball (rules: "whenever play
    restarts there is a clear possessor").
 5. A live gap holds the last team; any change is decided when the ball reappears.
 6. Possessions shorter than MIN_SPELL s that sit between two spells of the same team are merged
    into it (a short spurious hand-over costs two split possessions).
"""

import numpy as np

FPS = 25
PITCH = (105.0, 68.0)

DEFAULTS = dict(
    MAX_STEP=2.0, BRIDGE=6,
    CTRL_R=1.5, MARGIN=0.3, CTRL_V=12.0,
    CONFIRM=6, HOLE=3,
    DEAD_MIN=1.5, DEAD_LINE=2.0, DEAD_SPEED=1.8, RESTART_V=5.0, OUT_M=0.5, D_JOIN=1.0,
    MIN_SPELL=2.0,
)


def _runs(mask):
    e = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return np.flatnonzero(e == 1), np.flatnonzero(e == -1)


def clean_ball(ball, p):
    b = ball.astype(float).copy()
    ok = ~np.isnan(b[:, 0])
    idx = np.flatnonzero(ok)
    if len(idx) > 2:
        step_prev = np.r_[np.inf, np.hypot(*(b[idx[1:]] - b[idx[:-1]]).T) / np.maximum(1, np.diff(idx))]
        step_next = np.r_[step_prev[1:], np.inf]
        spike = (step_prev > p["MAX_STEP"]) & (step_next > p["MAX_STEP"])
        b[idx[spike]] = np.nan
    miss = np.isnan(b[:, 0])
    for s, e in zip(*_runs(miss)):
        if 0 < s and e < len(b) and e - s <= p["BRIDGE"]:
            for c in (0, 1):
                b[s:e, c] = np.linspace(b[s - 1, c], b[e, c], e - s + 2)[1:-1]
    return b


def ball_speed(b, w=4):
    """Speed from the displacement over +/- w frames: one-frame differences turn 0.3 m of
    detection jitter into ~5 m/s of fake speed, which would fire every speed threshold."""
    v = np.full(len(b), np.nan)
    v[w:-w] = np.hypot(b[2 * w:, 0] - b[:-2 * w, 0], b[2 * w:, 1] - b[:-2 * w, 1]) * FPS / (2 * w)
    return v


def control(px, py, b, team, p):
    """0 / 1 = that team controls the ball this frame, -1 = nobody."""
    n = len(b)
    ctrl = np.full(n, -1, dtype=np.int8)
    ok = ~np.isnan(b[:, 0])
    d = np.hypot(px[ok] - b[ok, 0:1], py[ok] - b[ok, 1:2])
    d = np.where(np.isnan(d), np.inf, d)
    dh = d[:, team == 0].min(1)
    da = d[:, team == 1].min(1)
    v = ball_speed(b)[ok]
    slow = ~(v > p["CTRL_V"])
    c = np.full(ok.sum(), -1, dtype=np.int8)
    c[(dh < p["CTRL_R"]) & (da > dh + p["MARGIN"]) & slow] = 0
    c[(da < p["CTRL_R"]) & (dh > da + p["MARGIN"]) & slow] = 1
    ctrl[ok] = c
    return ctrl


def player_speed(px, py, w=12):
    """Mean player speed from displacement over +/- w frames (robust to position jitter),
    smoothed over 1 s. Jumps faster than 12 m/s are identity switches and are ignored."""
    n = len(px)
    sp = np.full(px.shape, np.nan, dtype=np.float32)
    sp[w:-w] = np.hypot(px[2 * w:] - px[:-2 * w], py[2 * w:] - py[:-2 * w]) * FPS / (2 * w)
    sp[sp > 12] = np.nan
    with np.errstate(all="ignore"):
        cnt = (~np.isnan(sp)).sum(1)
        m = np.where(cnt > 0, np.nansum(sp, 1) / np.maximum(cnt, 1), np.nan)
    good = ~np.isnan(m)
    if good.any():
        m = np.interp(np.arange(n), np.flatnonzero(good), m[good])
    else:
        m = np.zeros(n)
    k = np.ones(FPS) / FPS
    return np.convolve(m, k, mode="same")


def dead_mask(b, raw_ball, mean_speed, p, lo, hi):
    n = len(b)
    dead = np.zeros(n, dtype=bool)
    L, W = PITCH
    out = ((b[:, 0] < -p["OUT_M"]) | (b[:, 0] > L + p["OUT_M"]) |
           (b[:, 1] < -p["OUT_M"]) | (b[:, 1] > W + p["OUT_M"]))
    dead |= out & ~np.isnan(b[:, 0])
    v = ball_speed(b)

    def near_line(q):
        return (not np.isnan(q[0])) and (q[0] < p["DEAD_LINE"] or q[0] > L - p["DEAD_LINE"] or
                                         q[1] < p["DEAD_LINE"] or q[1] > W - p["DEAD_LINE"])

    miss = np.isnan(b[:, 0])
    miss[:lo] = miss[hi + 1:] = False
    for s, e in zip(*_runs(miss)):
        if e - s < p["DEAD_MIN"] * FPS:
            continue
        before = b[s - 1] if s > 0 else np.array([np.nan, np.nan])
        after = b[e] if e < n else np.array([np.nan, np.nan])
        slow = np.median(mean_speed[s:e]) < p["DEAD_SPEED"]
        if not (near_line(before) or near_line(after) or slow):
            continue
        # restart: first frame after reappearance where the ball moves
        r = e
        lim = min(n, hi + 1, e + 30 * FPS)
        while r < lim and not np.isnan(b[r, 0]) and not (v[r] > p["RESTART_V"]):
            r += 1
        dead[s:r] = True
    # a stoppage is one stretch: alive slivers shorter than D_JOIN s inside it are noise
    for s, e in zip(*_runs(~dead)):
        if s > 0 and e < n and dead[s - 1] and dead[e] and e - s < p["D_JOIN"] * FPS:
            dead[s:e] = True
    return dead


def resolve(ctrl, dead, lo, hi, p):
    """State machine over one graded block. Returns codes 0/1/2(D)."""
    out = np.full(hi - lo + 1, -1, dtype=np.int8)
    cur = -1           # team in possession, -1 = unknown (start / after a stoppage)
    cand, cand_start, cand_len, holes = -1, 0, 0, 0
    pending = None     # start of the current possession whose team is not yet known
    for k in range(lo, hi + 1):
        i = k - lo
        if dead[k]:
            out[i] = 2
            cur, cand, cand_len, holes = -1, -1, 0, 0
            continue
        c = ctrl[k]
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
            out[cand_start:i + 1] = cur   # backdate to the first frame of control
            continue
        out[i] = cur
    # anything still unknown (alive but nobody ever controlled): copy the nearest known neighbour
    unk = out == -1
    if unk.any() and (~unk).any():
        idx = np.flatnonzero(~unk)
        j = np.clip(np.searchsorted(idx, np.flatnonzero(unk)), 0, len(idx) - 1)
        out[unk] = out[idx[j]]
    out[out == -1] = 2
    return out


def merge_short(codes, min_len):
    """Merge a short team spell sandwiched between two spells of the other... same team."""
    c = codes.copy()
    changed = True
    while changed:
        changed = False
        edges = np.flatnonzero(np.diff(c)) + 1
        starts = np.r_[0, edges]
        ends = np.r_[edges, len(c)]
        for j in range(1, len(starts) - 1):
            s, e = starts[j], ends[j]
            if c[s] < 2 and e - s < min_len and c[starts[j - 1]] == c[starts[j + 1]] \
                    and c[starts[j - 1]] < 2 and c[starts[j - 1]] != c[s]:
                c[s:e] = c[starts[j - 1]]
                changed = True
                break
    return c


def label(px, py, ball, team, blocks, params=None):
    p = dict(DEFAULTS, **(params or {}))
    team = np.asarray(team)
    b = clean_ball(ball, p)
    ctrl = control(px, py, b, team, p)
    ms = player_speed(px, py)
    out = np.full(len(ball), "X", dtype="<U1")
    for _, lo, hi in blocks:
        lo, hi = int(lo), int(hi)
        dead = dead_mask(b, ball, ms, p, lo, hi)
        codes = resolve(ctrl, dead, lo, hi, p)
        codes = merge_short(codes, int(p["MIN_SPELL"] * FPS))
        out[lo:hi + 1] = np.array(["H", "A", "D"])[codes]
    return out
