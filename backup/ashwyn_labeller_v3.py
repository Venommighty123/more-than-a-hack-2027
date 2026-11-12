"""Possession labeller v3: v2.1 + a false-ball filter; an optimal segmentation is available but OFF.

RESULT ON THE BENCH (see the changelog): the optimal segmentation (RESOLVE="viterbi") scored below
v2.1's greedy switch on every noisy copy (best 0.489 vs 0.501), so the default is RESOLVE="greedy".
What did help is drop_blobs(): short ball tracks that appear and disappear by teleporting are false
detections, and removing them before the control rule raised the bench mean from 0.501 to 0.511.

    label(px, py, ball, team, blocks, params=None, tags=None) -> array of "H" / "A" / "D" / "X"

WHY. v1 / v2 switch possession greedily: the other team must control the ball for CONFIRM frames in
a row (with small holes), and possessions shorter than MIN_SPELL are merged away afterwards. On the
bench only about half of the true hand-overs were found within 2 s, and loosening those thresholds
added more false changes than true ones.

WHAT v3 DOES INSTEAD. Within each stretch of live play (between two stoppages) it picks the whole
sequence of H / A that best explains the evidence, paying a fixed price for every change of
possession:
    maximise   sum over frames of evidence(team in possession)  -  SWITCH_COST x (number of changes)
solved exactly with dynamic programming (Viterbi, two states). Evidence per frame:
    +1 for the team that controls the ball (v1's control rule), -1 for the other team
    +SOFT_W for the team whose player is nearest within SOFT_R m when nobody controls it
    +PASS_W per frame across a pass: the ball leaves a player of team T and the next control is
       also team T (the team keeps the ball through the flight), so flight frames count for T
Because the solution is global, a change is placed exactly where the evidence turns (no lag from a
confirmation window) and a short real possession survives if its evidence outweighs two switch
costs, while a single touch by the other team does not.
"""

import numpy as np

import labeller_v1 as v1
import labeller_v2 as v2

FPS = v1.FPS

DEFAULTS = dict(v2.DEFAULTS)
DEFAULTS.update(SWITCH_COST=8.0, SOFT_W=0.0, SOFT_R=3.0, PASS_W=0.3, PASS_MAX_S=4.0,
                EVIDENCE="touch", TOUCH_W=1.0, FRAME_W=0.05, FRAME_CAP=50, TOUCH_HOLE=3,
                MIN_TOUCH=1, BLOB_JUMP=5.0, BLOB_MAX_S=1.5, RESOLVE="greedy")


def drop_blobs(b, p):
    """Remove short ball tracklets that start AND end with a jump (> BLOB_JUMP m in one frame, or a
    missing stretch on either side): a false detection somewhere else on the pitch, not the ball.
    A real ball cannot teleport, so this needs no labels."""
    b = b.copy()
    n = len(b)
    ok = ~np.isnan(b[:, 0])
    step = np.full(n, np.inf)
    both = ok[1:] & ok[:-1]
    step[1:][both] = np.hypot(*(b[1:][both] - b[:-1][both]).T)
    brk = ~ok | (step > p["BLOB_JUMP"])          # frame k starts a new tracklet
    starts = np.flatnonzero(ok & brk)
    for i, s in enumerate(starts):
        e = s + 1
        while e < n and ok[e] and step[e] <= p["BLOB_JUMP"]:
            e += 1
        if e - s <= p["BLOB_MAX_S"] * FPS:
            jump_in = s > 0 and ok[s - 1]            # entered by a jump from another position
            jump_out = e < n and ok[e]               # left by a jump
            gap_in = s == 0 or not ok[s - 1]
            gap_out = e >= n or not ok[e]
            if (jump_in and (jump_out or gap_out)) or (gap_in and jump_out):
                b[s:e] = np.nan
    return b


def touch_evidence(ctrl, p):
    """One impulse per control episode (a run of frames controlled by one team, holes of up to
    TOUCH_HOLE frames allowed), placed on its first frame: TOUCH_W + FRAME_W x min(length,
    FRAME_CAP). Counting touches rather than frames treats a one-touch interception-and-pass
    (two short touches) like a long dribble, which is closer to "a team completes passes"."""
    n = len(ctrl)
    e = np.zeros(n)
    idx = np.flatnonzero(ctrl >= 0)
    if len(idx) == 0:
        return e
    start = idx[0]
    last = idx[0]
    for k in list(idx[1:]) + [None]:
        if k is not None and ctrl[k] == ctrl[last] and k - last <= p["TOUCH_HOLE"] + 1:
            last = k
            continue
        if last - start + 1 < p["MIN_TOUCH"]:
            if k is not None:
                start = last = k
            continue
        w = p["TOUCH_W"] + p["FRAME_W"] * min(last - start + 1, p["FRAME_CAP"])
        e[start] += w if ctrl[start] == 0 else -w
        if k is not None:
            start = last = k
    return e


def evidence(px, py, b, team, ctrl, p):
    """Per-frame score for HOME (positive) vs AWAY (negative)."""
    n = len(b)
    e = np.zeros(n)
    e[ctrl == 0] = 1.0
    e[ctrl == 1] = -1.0
    if p["SOFT_W"]:
        ok = (~np.isnan(b[:, 0])) & (ctrl < 0)
        d = np.hypot(px[ok] - b[ok, 0:1], py[ok] - b[ok, 1:2])
        d = np.where(np.isnan(d), np.inf, d)
        dh = d[:, team == 0].min(1)
        da = d[:, team == 1].min(1)
        s = np.zeros(ok.sum())
        s[(dh < p["SOFT_R"]) & (dh < da)] = p["SOFT_W"]
        s[(da < p["SOFT_R"]) & (da < dh)] = -p["SOFT_W"]
        e[ok] += s
    if p["PASS_W"]:
        # flights between two controls by the same team count for that team
        idx = np.flatnonzero(ctrl >= 0)
        if len(idx) > 1:
            gap = np.diff(idx)
            same = ctrl[idx[1:]] == ctrl[idx[:-1]]
            for k in np.flatnonzero(same & (gap > 1) & (gap <= p["PASS_MAX_S"] * FPS)):
                a, c = idx[k] + 1, idx[k + 1]
                e[a:c] += p["PASS_W"] if ctrl[idx[k]] == 0 else -p["PASS_W"]
    return e


def viterbi(e, cost):
    """Best 0/1 sequence for evidence e (positive favours 0) with a price per change.

    Ties are broken towards the latest possible change: across a ball flight with no evidence,
    every switch point scores the same, and the rules (and the truth) put the change at the
    receiver's first touch, i.e. at the end of the flight."""
    n = len(e)
    if n == 0:
        return np.zeros(0, dtype=np.int8)
    back = np.zeros((n, 2), dtype=np.int8)
    s0, s1 = e[0], -e[0]
    for k in range(1, n):
        g0, g1 = e[k], -e[k]
        if s0 > s1 - cost:      # strict: on a tie the change is placed as LATE as possible
            n0, b0 = s0 + g0, 0
        else:
            n0, b0 = s1 - cost + g0, 1
        if s1 > s0 - cost:
            n1, b1 = s1 + g1, 1
        else:
            n1, b1 = s0 - cost + g1, 0
        back[k, 0], back[k, 1] = b0, b1
        s0, s1 = n0, n1
    out = np.zeros(n, dtype=np.int8)
    out[-1] = 0 if s0 >= s1 else 1
    for k in range(n - 1, 0, -1):
        out[k - 1] = back[k, out[k]]
    return out


def resolve(e, dead, lo, hi, p):
    codes = np.full(hi - lo + 1, 2, dtype=np.int8)
    alive = ~dead[lo:hi + 1]
    for s, t in zip(*v1._runs(alive)):
        seg = e[lo + s:lo + t]
        if not np.any(seg):
            continue          # no evidence at all: filled from the neighbours below
        codes[s:t] = viterbi(seg, p["SWITCH_COST"])
    # alive stretches with no evidence: copy the previous team (spellF1 forward-fills anyway)
    for s, t in zip(*v1._runs(alive)):
        if not np.any(e[lo + s:lo + t]):
            prev = codes[:s][codes[:s] < 2]
            codes[s:t] = prev[-1] if len(prev) else 0
    return codes


def label(px, py, ball, team, blocks, params=None, tags=None):
    p = dict(DEFAULTS, **(params or {}))
    team = np.asarray(team)
    b = v1.clean_ball(ball, p)
    bc = drop_blobs(b, p) if p["BLOB_JUMP"] else b
    ctrl = v1.control(px, py, bc, team, p)
    rel = v2.rel_speed(px, py, blocks, p)
    if p["EVIDENCE"] == "touch":
        e = touch_evidence(ctrl, p)
    else:
        e = evidence(px, py, b, team, ctrl, p)
    out = np.full(len(ball), "X", dtype="<U1")
    for _, lo, hi in blocks:
        lo, hi = int(lo), int(hi)
        dead = v2.dead_mask(b, rel, p, lo, hi, tags)
        if p["RESOLVE"] == "greedy":
            codes = v1.merge_short(v1.resolve(ctrl, dead, lo, hi, p), int(p["MIN_SPELL"] * FPS))
        else:
            codes = resolve(e, dead, lo, hi, p)
        out[lo:hi + 1] = np.array(["H", "A", "D"])[codes]
    return out
