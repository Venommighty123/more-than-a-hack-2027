"""Possession labeller v4 (experimental). v2.1 plus three switchable changes, all label-free.

    label(px, py, ball, team, blocks, params=None, tags=None) -> array of "H" / "A" / "D" / "X"

WHY. On the realistic bench copies ("real", "real_occ"), built to match BAR-ATH's measured noise:
  - v2.1's rule C (ball present + players slow = dead) marks slow live build-up as dead and costs
    about 0.02 spellF1. It was designed for stoppages with a tracked ball, which the full match
    mostly does not have.
  - The real ball sits about 1 m from the player on it, about 0.5 m AHEAD along his run and 0.4 m
    to the side, so the 1.5 m circle misses about 40% of on-the-ball frames.

1. RULE_STOP_MODE  "full" = v2.1's rule C, "off" = no rule C, "weak" = rule C only when there is
   independent dead-ball evidence around the slow stretch:
     a set-piece tag from WEAK_PRE s before it to TAG_WIN s after it, or
     the ball off the pitch in the WEAK_PRE s before it, or
     the ball near a line (WEAK_LINE_M) on at least WEAK_LINE_FRAC of its tracked frames.
2. CTRL_MODE  "circle" = v2.1 (CTRL_R around the player), "shift" = the control point moves
   LEAD m forward along the player's velocity (only when he moves faster than LEAD_V m/s),
   "ellipse" = an ellipse around the player, R_FWD m ahead, R_BACK m behind, R_SIDE m sideways.
   Velocity is measured over +/- VEL_W frames (not frame to frame). MARGIN is applied to the same
   effective distance for both teams.
3. CTRL_R_AUTO  None = fixed radius. A number q = the radius is the match's own q-quantile of the
   nearest-player distance (same geometry as CTRL_MODE) on open-play frames: ball tracked, inside
   the pitch, slower than AUTO_V m/s, players moving (rel > AUTO_REL). Clipped to AUTO_CLIP. No labels.
   In "ellipse" mode the quantile sets the scale factor of the ellipse instead.
4. LONG_SLOW  None = off (default). A number = rule E: a collective slowdown (rel < LONG_SLOW_REL)
   lasting at least LONG_SLOW s is dead, even with the ball tracked. REJECTED on the bench
   (10 / 15 / 20 s: R+O 0.475 / 0.480 / 0.482 vs 0.489): it splits more real possessions than it
   fixes stoppages. Kept as a switch so the test can be repeated.

BENCH VERDICT (Nov 7, mean of Metrica games 1-3, spellF1; R+O = mean of real and real_occ):
                                  real    real_occ  hard    R+O
  v2.1 (rule C full, circle 1.5)  0.483   0.438     0.466   0.461
  rule C weak                     0.488   0.443     0.462   0.466
  rule C off                      0.504   0.461     0.450   0.483
  rule C off + circle 2.0         0.516   0.462     0.443   0.489   <- v4 defaults
  rule C off + shift 0.5 m, R 2.0 0.489   0.461     0.413   0.475   rejected
  rule C off + ellipse 2.5/1.5/1.5 0.503  0.461     0.436   0.482   rejected
  rule C off + auto radius q0.8   0.517   0.424     0.448   0.471   rejected (occlusion inflates R)
Rule C off beats full on all 6 realistic game copies. "weak" does not rescue rule C: lines and
tags are also present in live play, and tags are already handled by rule D. A forward-shifted or
stretched zone also reaches forward for the defender chasing the carrier, so hand-overs fragment.
Defaults: RULE_STOP_MODE="off", CTRL_MODE="circle", CTRL_R=2.0, CTRL_R_AUTO=None.
"""

import numpy as np

import labeller_v1 as v1
import labeller_v2 as v2

FPS = v1.FPS
PITCH = v1.PITCH

DEFAULTS = dict(v2.DEFAULTS)
DEFAULTS.update(
    CTRL_R=2.0,              # v4: 1.5 -> 2.0 (bench: +0.012 real, +0.001 real_occ; small, kept)
    RULE_STOP_MODE="off",    # v4: rule C off (bench: +0.021 real, +0.023 real_occ, all 6 games up)
    WEAK_PRE=3.0, WEAK_LINE_M=3.0, WEAK_LINE_FRAC=0.3,
    CTRL_MODE="circle", VEL_W=6, LEAD=0.5, LEAD_V=1.0,
    R_FWD=2.5, R_BACK=1.5, R_SIDE=1.5,
    LONG_SLOW=None, LONG_SLOW_REL=0.7, LONG_SLOW_EXT=False,
    CTRL_R_AUTO=None, AUTO_V=8.0, AUTO_REL=0.8, AUTO_CLIP=(1.0, 3.0),
)


# -- control ------------------------------------------------------------------------------------

def player_velocity(px, py, w):
    vx = np.full(px.shape, np.nan, dtype=np.float32)
    vy = np.full(px.shape, np.nan, dtype=np.float32)
    vx[w:-w] = (px[2 * w:] - px[:-2 * w]) * FPS / (2 * w)
    vy[w:-w] = (py[2 * w:] - py[:-2 * w]) * FPS / (2 * w)
    sp = np.hypot(vx, vy)
    bad = ~(sp < 12)                      # identity switches / missing
    vx[bad] = 0
    vy[bad] = 0
    return vx, vy


def eff_dist(px, py, b, vx, vy, p):
    """(tracked frames x slots) effective distance from each player to the ball, in metres."""
    ok = ~np.isnan(b[:, 0])
    dx = b[ok, 0:1] - px[ok]
    dy = b[ok, 1:2] - py[ok]
    mode = p["CTRL_MODE"]
    if mode == "circle":
        d = np.hypot(dx, dy)
    else:
        ux, uy = vx[ok], vy[ok]
        sp = np.hypot(ux, uy)
        moving = sp > p["LEAD_V"]
        with np.errstate(all="ignore"):
            ux = np.where(moving, ux / sp, 0.0)
            uy = np.where(moving, uy / sp, 0.0)
        if mode == "shift":
            d = np.hypot(dx - p["LEAD"] * ux, dy - p["LEAD"] * uy)
        elif mode == "ellipse":
            along = dx * ux + dy * uy
            side = -dx * uy + dy * ux
            rf = np.where(along > 0, p["R_FWD"], p["R_BACK"])
            # a standing player (no direction) gets a circle of R_SIDE
            rf = np.where(moving, rf, p["R_SIDE"])
            along = np.where(moving, along, np.hypot(dx, dy))
            side = np.where(moving, side, 0.0)
            d = np.hypot(along / rf, side / p["R_SIDE"]) * p["R_SIDE"]
        else:
            raise ValueError(mode)
    return np.where(np.isnan(d), np.inf, d), ok


def control(px, py, b, team, p, d=None, ok=None):
    n = len(b)
    ctrl = np.full(n, -1, dtype=np.int8)
    if d is None:
        vx, vy = player_velocity(px, py, p["VEL_W"])
        d, ok = eff_dist(px, py, b, vx, vy, p)
    dh = d[:, team == 0].min(1)
    da = d[:, team == 1].min(1)
    v = v1.ball_speed(b)[ok]
    slow = ~(v > p["CTRL_V"])
    R = p["CTRL_R"] if p["CTRL_MODE"] != "ellipse" else p["R_SIDE"]
    c = np.full(ok.sum(), -1, dtype=np.int8)
    c[(dh < R) & (da > dh + p["MARGIN"]) & slow] = 0
    c[(da < R) & (dh > da + p["MARGIN"]) & slow] = 1
    ctrl[ok] = c
    return ctrl


def auto_radius(d, ok, b, rel, blocks, p):
    """Label-free: the match's own q-quantile of nearest-player distance in open play."""
    L, W = PITCH
    n = len(b)
    graded = np.zeros(n, dtype=bool)
    for _, lo, hi in blocks:
        graded[int(lo):int(hi) + 1] = True
    v = v1.ball_speed(b)
    inside = (b[:, 0] > 0) & (b[:, 0] < L) & (b[:, 1] > 0) & (b[:, 1] < W)
    sel = (ok & graded & inside & (v < p["AUTO_V"]) & (rel > p["AUTO_REL"]))[ok]
    near = d.min(1)[sel]
    near = near[np.isfinite(near)]
    if len(near) < 100:
        return None
    return float(np.clip(np.quantile(near, p["CTRL_R_AUTO"]), *p["AUTO_CLIP"]))


# -- dead ball ----------------------------------------------------------------------------------

def weak_evidence(b, s0, r, out, tags, p, lo):
    """Independent dead-ball evidence around a rule-C candidate stretch [s0, r)."""
    L, W = PITCH
    pre = int(p["WEAK_PRE"] * FPS)
    if tags is not None and p["USE_TAGS"]:
        t = np.asarray(tags, dtype=int)
        if ((t >= s0 - pre) & (t <= r + int(p["TAG_WIN"] * FPS))).any():
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


def dead_mask(b, rel, p, lo, hi, tags=None):
    """v2.1's dead mask with rule C replaced by RULE_STOP_MODE."""
    mode = p["RULE_STOP_MODE"]
    q = dict(p, RULE_STOP=(mode == "full"))
    dead = v2.dead_mask(b, rel, q, lo, hi, tags)
    if p["LONG_SLOW"]:
        # rule E: a very long collective slowdown (rel < LONG_SLOW_REL for >= LONG_SLOW s) is a
        # stoppage even with the ball tracked and no other evidence. Live play is never that slow
        # for that long; rule C's 4 s threshold caught slow build-up as well.
        sl = rel < p["LONG_SLOW_REL"]
        sl[:lo] = False
        sl[hi + 1:] = False
        v = v1.ball_speed(b)
        for s, e in zip(*v1._runs(sl)):
            if e - s >= p["LONG_SLOW"] * FPS:
                dead[s:(v2._restart(e, b, v, rel, p, hi) if p["LONG_SLOW_EXT"] else e)] = True
    if mode != "weak":
        return dead
    # Re-run rule C exactly as v2.1 does, but keep a stretch only with independent evidence.
    # Tags are applied by v2 already; this only adds the stretches that pass the weak test.
    L, W = PITCH
    v = v1.ball_speed(b)
    present = ~np.isnan(b[:, 0])
    out = present & ((b[:, 0] < -p["OUT_M"]) | (b[:, 0] > L + p["OUT_M"]) |
                     (b[:, 1] < -p["OUT_M"]) | (b[:, 1] > W + p["OUT_M"]))
    miss = ~present
    slow = rel < p["STOP_REL"]
    slow[:lo] = False
    slow[hi + 1:] = False
    quiet = miss | out | (present & ~(v > p["QUIET_V"]))
    if p["FIX_CARRY"]:
        nl = present & ((b[:, 0] < p["LINE_QUIET_M"]) | (b[:, 0] > L - p["LINE_QUIET_M"]) |
                        (b[:, 1] < p["LINE_QUIET_M"]) | (b[:, 1] > W - p["LINE_QUIET_M"]))
        quiet = quiet | (present & ~(v > p["CARRY_V"])) | nl
    add = np.zeros(len(b), dtype=bool)
    for s, e in zip(*v1._runs(slow)):
        if e - s < p["STOP_MIN"] * FPS or quiet[s:e].mean() < p["QUIET_FRAC"]:
            continue
        s0 = s
        stop = max(lo, s - int(p["BACK_MAX"] * FPS))
        while s0 > stop and rel[s0 - 1] < p["EXT_REL"] and not (v[s0 - 1] > p["RESTART_V"] * 2):
            s0 -= 1
        r = v2._restart(e, b, v, rel, p, hi)
        if weak_evidence(b, s0, r, out, tags, p, lo):
            add[s0:r] = True
    dead |= add
    n = len(b)
    for s, e in zip(*v1._runs(~dead)):
        if s > 0 and e < n and dead[s - 1] and dead[e] and e - s < p["D_JOIN"] * FPS:
            dead[s:e] = True
    return dead


# -- label --------------------------------------------------------------------------------------

def label(px, py, ball, team, blocks, params=None, tags=None, info=None):
    p = dict(DEFAULTS, **(params or {}))
    team = np.asarray(team)
    b = v1.clean_ball(ball, p)
    rel = v2.rel_speed(px, py, blocks, p)
    vx, vy = player_velocity(px, py, p["VEL_W"])
    d, ok = eff_dist(px, py, b, vx, vy, p)
    if p["CTRL_R_AUTO"] is not None:
        r = auto_radius(d, ok, b, rel, blocks, p)
        if r is not None:
            if p["CTRL_MODE"] == "ellipse":
                k = r / p["R_SIDE"]
                p.update(R_FWD=p["R_FWD"] * k, R_BACK=p["R_BACK"] * k, R_SIDE=r)
                d, ok = eff_dist(px, py, b, vx, vy, p)
            else:
                p["CTRL_R"] = r
        if info is not None:
            info["radius"] = r
    ctrl = control(px, py, b, team, p, d, ok)
    out = np.full(len(ball), "X", dtype="<U1")
    for _, lo, hi in blocks:
        lo, hi = int(lo), int(hi)
        dead = dead_mask(b, rel, p, lo, hi, tags)
        codes = v1.resolve(ctrl, dead, lo, hi, p)
        codes = v1.merge_short(codes, int(p["MIN_SPELL"] * FPS))
        out[lo:hi + 1] = np.array(["H", "A", "D"])[codes]
    return out
