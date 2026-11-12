"""Possession labeller v5 (experimental): v4 plus a hand-over detector that reads the ball's path.

    label(px, py, ball, team, blocks, params=None, tags=None, info=None) -> "H" / "A" / "D" / "X"

WHY. On the realistic bench copies v4 finds only 40% of the real H<->A hand-overs within 2 s, and a
third of its own changes are false. With the true team on every live frame (v4's D kept) the score
would be 0.89 instead of 0.52, so who-has-the-ball is where the score is. Of the missed hand-overs
with the ball tracked, 93% have a strike before and a new-team player within 2 m after: the
evidence is there, but v4's control rule (6 frames in a row, 0.3 m clearer than any opponent)
throws it away.

LOGIC (each step has a switch; every speed is measured over windows, never frame to frame)
 1. FLIGHTS  stretches where the ball is faster than FLIGHT_V m/s for at least FLIGHT_MIN frames.
    HO_FLIGHT: nobody controls the ball during a flight (a pass flying past a defender is not his).
 2. RECEPTION  the end of a flight (the ball slows down). Over RECV_W frames from there, count the
    frames where each team's nearest player is within RECV_R m and at least RECV_MARGIN m nearer
    than the other team's. The receiver is the team with at least RECV_SHARE of those frames and
    at least RECV_MIN of them; otherwise the ball is contested or loose and nothing happens
    (CHALLENGE.md: a challenge is not a change of possession).
 3. HO_RECV: a clear receiver is injected as control evidence on CONFIRM frames from the reception,
    so v1's state machine switches there (backdated to the reception) if it is the other team, and
    does nothing if it is the same team (a completed pass).
 4. HO_GAP: when the ball reappears after a gap of at least GAP_S s, the same vote over RECV_W
    frames decides who has it; a clear team is injected the same way (placed at the reappearance).
 5. MIN_SPELL can be re-tuned: event-backed changes are more reliable than control flicker.
"""

import numpy as np

import labeller_v1 as v1
import labeller_v2 as v2
import labeller_v4 as v4

FPS = v1.FPS

DEFAULTS = dict(v4.DEFAULTS)
DEFAULTS.update(
    HO_FLIGHT=True, HO_RECV=True, HO_GAP=True,
    FLIGHT_V=7.0, FLIGHT_MIN=4,
    RECV_W=10, RECV_R=2.5, RECV_MARGIN=0.3, RECV_SHARE=0.7, RECV_MIN=3,
    GAP_S=0.5,
    HO_PROTECT=0.0,   # s: after a reception, the other team's control is ignored this long
    HO_KEEP=False,
    RESOLVE_MODE="run",   # "run" = v1 (CONFIRM frames in a row); "density" = see resolve_density
    DENS_W=25, DENS_N=8, DENS_M=4,    # a spell that starts at a reception is not merged away by MIN_SPELL
)


def resolve_density(ctrl, dead, lo, hi, p):
    """Switch when the other team controls the ball on at least DENS_N of the last DENS_W frames
    and on at least DENS_M more of them than the current team. Backdated to the other team's
    first control frame in that window. A noisy ball (~1 m off the player) breaks control into
    short runs, so a run-based rule sees hand-overs late; a count over a window does not care.
    Dead frames and the start after a stoppage are handled as in v1.resolve."""
    W, N, M = int(p["DENS_W"]), int(p["DENS_N"]), int(p["DENS_M"])
    c = ctrl[lo:hi + 1]
    ch = np.concatenate(([0], np.cumsum(c == 0)))
    ca = np.concatenate(([0], np.cumsum(c == 1)))
    out = np.full(hi - lo + 1, -1, dtype=np.int8)
    cur = -1
    pending = None
    for i in range(hi - lo + 1):
        if dead[lo + i]:
            out[i] = 2
            cur = -1
            pending = None
            continue
        a = max(0, i - W + 1)
        if pending is not None and a < pending:
            a = pending
        nh, na = ch[i + 1] - ch[a], ca[i + 1] - ca[a]
        if cur == -1:
            if pending is None:
                pending = i
            if c[i] >= 0:
                cur = int(c[i])
                out[pending:i + 1] = cur
                pending = None
            continue
        own, oth = (nh, na) if cur == 0 else (na, nh)
        if oth >= N and oth >= own + M:
            other = 1 - cur
            j = a + int(np.argmax(c[a:i + 1] == other))
            cur = other
            out[j:i + 1] = cur
            continue
        out[i] = cur
    unk = out == -1
    if unk.any() and (~unk).any():
        idx = np.flatnonzero(~unk)
        j = np.clip(np.searchsorted(idx, np.flatnonzero(unk)), 0, len(idx) - 1)
        out[unk] = out[idx[j]]
    out[out == -1] = 2
    return out


def merge_short_keep(codes, min_len, keep):
    """v1.merge_short, except spells starting on a `keep` frame (block-relative) stay."""
    c = codes.copy()
    changed = True
    while changed:
        changed = False
        edges = np.flatnonzero(np.diff(c)) + 1
        starts = np.r_[0, edges]
        ends = np.r_[edges, len(c)]
        for j in range(1, len(starts) - 1):
            s, e = starts[j], ends[j]
            if keep[s]:
                continue
            if c[s] < 2 and e - s < min_len and c[starts[j - 1]] == c[starts[j + 1]] \
                    and c[starts[j - 1]] < 2 and c[starts[j - 1]] != c[s]:
                c[s:e] = c[starts[j - 1]]
                changed = True
                break
    return c


def team_nearest(d, team):
    """Per tracked frame: nearest distance of each team (inf when nobody)."""
    return d[:, team == 0].min(1), d[:, team == 1].min(1)


def vote(dh, da, k0, k1, p):
    """Clear team (0/1) on the frames k0..k1 of the tracked-frame arrays, or -1."""
    h, a = dh[k0:k1], da[k0:k1]
    ch = np.sum((h < p["RECV_R"]) & (a > h + p["RECV_MARGIN"]))
    ca = np.sum((a < p["RECV_R"]) & (h > a + p["RECV_MARGIN"]))
    tot = ch + ca
    if tot < p["RECV_MIN"]:
        return -1
    if ch >= p["RECV_SHARE"] * tot:
        return 0
    if ca >= p["RECV_SHARE"] * tot:
        return 1
    return -1


def events(b, d, ok, team, p):
    """Reception and reappearance events: list of (frame, team, kind)."""
    n = len(b)
    v = v1.ball_speed(b)
    pos = np.full(n, -1)
    pos[ok] = np.arange(ok.sum())          # frame -> row of d
    dh, da = team_nearest(d, team)
    ev = []
    fast = ok & (v > p["FLIGHT_V"])
    flights = [(s, e) for s, e in zip(*v1._runs(fast)) if e - s >= p["FLIGHT_MIN"]]
    if p["HO_RECV"]:
        for s, e in flights:
            if e >= n or not ok[e]:
                continue
            r0 = pos[e]
            r1 = r0 + p["RECV_W"]
            if pos[min(n - 1, e + p["RECV_W"])] - r0 != p["RECV_W"]:   # ball lost soon after
                r1 = r0 + max(1, int(np.sum(ok[e:e + p["RECV_W"]])))
            t = vote(dh, da, r0, r1, p)
            if t >= 0:
                ev.append((e, t, "recv"))
    if p["HO_GAP"]:
        miss = ~ok
        for s, e in zip(*v1._runs(miss)):
            if e >= n or s == 0 or e - s < p["GAP_S"] * FPS:
                continue
            r0 = pos[e]
            t = vote(dh, da, r0, r0 + p["RECV_W"], p)
            if t >= 0:
                ev.append((e, t, "gap"))
    return ev, flights


def label(px, py, ball, team, blocks, params=None, tags=None, info=None):
    p = dict(DEFAULTS, **(params or {}))
    team = np.asarray(team)
    b = v1.clean_ball(ball, p)
    rel = v2.rel_speed(px, py, blocks, p)
    vx, vy = v4.player_velocity(px, py, p["VEL_W"])
    d, ok = v4.eff_dist(px, py, b, vx, vy, p)
    ctrl = v4.control(px, py, b, team, p, d, ok)
    ev, flights = events(b, d, ok, team, p)
    if p["HO_FLIGHT"]:
        for s, e in flights:
            ctrl[s:e] = -1
    ev = sorted(ev)
    keep = np.zeros(len(ball), dtype=bool)
    for i, (k, t, _) in enumerate(ev):
        if p["HO_PROTECT"]:
            stop = min(k + int(p["HO_PROTECT"] * FPS), ev[i + 1][0] if i + 1 < len(ev) else len(ball))
            w = ctrl[k:stop]
            w[w == 1 - t] = -1
        ctrl[k:k + p["CONFIRM"]] = t
        keep[max(0, k - 2):k + p["CONFIRM"] + 1] = True
    if info is not None:
        info.update(n_events=len(ev), n_flights=len(flights))
    out = np.full(len(ball), "X", dtype="<U1")
    for _, lo, hi in blocks:
        lo, hi = int(lo), int(hi)
        dead = v4.dead_mask(b, rel, p, lo, hi, tags)
        if p["RESOLVE_MODE"] == "density":
            codes = resolve_density(ctrl, dead, lo, hi, p)
        else:
            codes = v1.resolve(ctrl, dead, lo, hi, p)
        if p["HO_KEEP"]:
            codes = merge_short_keep(codes, int(p["MIN_SPELL"] * FPS), keep[lo:hi + 1])
        else:
            codes = v1.merge_short(codes, int(p["MIN_SPELL"] * FPS))
        out[lo:hi + 1] = np.array(["H", "A", "D"])[codes]
    return out
