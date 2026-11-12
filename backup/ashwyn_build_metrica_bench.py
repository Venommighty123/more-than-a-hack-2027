"""Build an honest test bench from Metrica's PUBLIC sample games (other matches, so tuning is allowed).

RULES.md forbids tuning on the provided match and on any human data about it, but allows data about
other games "in any amount". Metrica publishes three anonymised matches with synchronised tracking
AND event data (github.com/metrica-sports/sample-data). This script turns each into the same shape
our labeller sees on BAR-ATH, plus a frame-level possession ground truth derived from the events:

    px, py        (frames x slots) player positions in metres (105 x 68), NaN when not on the pitch
    ball          (frames x 2) ball position in metres, NaN when not tracked
    team_of_slot  0 home / 1 away for every slot
    blocks        [(period, first_frame, last_frame)] the graded stretch of each half (0-based)
    gt            per-frame labels H / A / D / X  (X = not graded)

GROUND TRUTH FROM EVENTS (our derivation, approximates CHALLENGE.md's definition)
  alive frames   the team of the most recent "on the ball" event (PASS, CARRY, SET PIECE, SHOT,
                 RECOVERY, BALL LOST). Contested / loose ball (CHALLENGE) keeps the previous team,
                 as CHALLENGE.md says. A change happens at the start of the new team's first event.
  dead frames    from what stopped play (ball out, a shot going out or in, a foul, an offside) to the
                 restart (any SET PIECE, or a goal kick recorded as PASS / BALL LOST "GOAL KICK").
                 Dead-time events in between (keeper's RECOVERY, CARD) are skipped. Before
                 kick-off / after the end = X.
  The organisers' truth comes from corrected tracking, not events, so this is a proxy. It is the
  same proxy for every version of the labeller, so it ranks versions fairly.

DEGRADED COPIES ("degraded", "hard"; see VARIANTS) make the clean sample data as noisy as ours:
  extra live ball gaps, jitter, false ball blobs, and a ball visible at the restarter's feet late in
  dead stretches. Calibrated so the starter baseline behaves as on BAR-ATH. Fixed seed.

    python test_scripts/bench/build_metrica_bench.py --src <folder with Sample_Game_1..3> --out bench/

Needs numpy + pandas (+ kloppy for game 3's EPTS tracking).
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

FPS = 25
PITCH = (105.0, 68.0)
OWNER_TYPES = {"PASS", "CARRY", "SET PIECE", "SHOT", "RECOVERY", "BALL LOST"}
SEED = 7


# -- events -> ground truth -------------------------------------------------------------------

def gt_from_events(ev, n):
    """ev: DataFrame with team (0/1), type, subtype, period, start, end (0-based frames)."""
    gt = np.full(n, "X", dtype="<U1")
    ev = ev.sort_values(["start", "end"]).reset_index(drop=True)
    blocks = []
    for period, e in ev.groupby("period"):
        e = e.sort_values("start").reset_index(drop=True)
        first = int(e["start"].min())
        last = int(max(e["end"].max(), e["start"].max()))
        blocks.append((int(period), first, last))
        owner = np.full(last - first + 1, -1, dtype=np.int8)
        dead = np.zeros(last - first + 1, dtype=bool)
        # owner timeline
        cur = -1
        marks = [(int(r.start), int(r.team)) for r in e.itertuples() if r.type in OWNER_TYPES
                 and r.team in (0, 1)]
        j = 0
        for k in range(first, last + 1):
            while j < len(marks) and marks[j][0] <= k:
                cur = marks[j][1]
                j += 1
            owner[k - first] = cur
        # dead stretches: from what stopped play to the restart that ends the stoppage.
        # A restart is any SET PIECE, or a goal kick, which Metrica often records as a PASS or
        # BALL LOST with subtype GOAL KICK. What stops play: the ball going out (BALL OUT, a SHOT
        # ending OUT / GOAL), a foul (CHALLENGE ...FAULT..., FAULT RECEIVED) or an offside. Between
        # the stop and the restart only dead-time events may appear (the keeper fetching the ball
        # is a RECOVERY, a booking is a CARD). With no recognisable stop, the stoppage starts at
        # the end of the previous event.
        rows = list(e.itertuples())
        for i, r in enumerate(rows):
            sub = str(r.subtype).upper()
            if not (r.type == "SET PIECE" or "GOAL KICK" in sub):
                continue
            s = int(r.start)
            if s <= first:
                continue
            k0 = None
            for q in rows[:i][::-1]:
                qs = str(q.subtype).upper()
                stop_end = int(max(q.end, q.start))
                if q.type == "BALL OUT" or (q.type == "SHOT" and ("OUT" in qs or "GOAL" in qs)) \
                        or "OFFSIDE" in qs:
                    k0 = stop_end
                    break
                if q.type == "FAULT RECEIVED" or (q.type == "CHALLENGE" and "FAULT" in qs):
                    k0 = int(q.start)
                    break
                if q.type in ("RECOVERY", "CARD", "CHALLENGE", "BALL LOST") and s - q.start < 300 * FPS:
                    continue
                break
            if k0 is None:
                prev = [int(max(q.end, q.start)) for q in rows[:i] if q.type != "CARD"]
                k0 = prev[-1] if prev else s
            if 0 < s - k0 < 300 * FPS:
                dead[k0 - first:s - first] = True
        lab = np.where(owner == 0, "H", np.where(owner == 1, "A", "X"))
        lab[dead] = "D"
        gt[first:last + 1] = lab
    return gt, blocks


# -- loaders ----------------------------------------------------------------------------------

def load_csv_game(folder, g):
    ev = pd.read_csv(os.path.join(folder, "Sample_Game_{}_RawEventsData.csv".format(g)))
    ev = pd.DataFrame({
        "team": ev["Team"].map({"Home": 0, "Away": 1}).fillna(-1).astype(int),
        "type": ev["Type"].astype(str).str.upper(), "subtype": ev["Subtype"].astype(str),
        "period": ev["Period"], "start": ev["Start Frame"] - 1, "end": ev["End Frame"] - 1})
    cols, xs, ys, team = [], [], [], []
    frames = None
    for side, code in (("Home", 0), ("Away", 1)):
        path = os.path.join(folder, "Sample_Game_{}_RawTrackingData_{}_Team.csv".format(g, side))
        t = pd.read_csv(path, skiprows=2)
        names = list(t.columns)
        frames = t["Frame"].values
        players = [c for c in names if c.startswith("Player")]
        for p in players:
            i = names.index(p)
            xs.append(t.iloc[:, i].values * PITCH[0])
            ys.append(t.iloc[:, i + 1].values * PITCH[1])
            team.append(code)
        if side == "Home":
            i = names.index("Ball")
            bx, by = t.iloc[:, i].values * PITCH[0], t.iloc[:, i + 1].values * PITCH[1]
    n = int(frames.max())
    px = np.full((n, len(xs)), np.nan, dtype=np.float32)
    py = np.full((n, len(xs)), np.nan, dtype=np.float32)
    px[frames - 1] = np.stack(xs, axis=1)
    py[frames - 1] = np.stack(ys, axis=1)
    ball = np.full((n, 2), np.nan)
    ball[frames - 1, 0], ball[frames - 1, 1] = bx, by
    return px, py, ball, np.array(team, dtype=np.int8), ev


def load_epts_game(folder):
    from kloppy import metrica
    ds = metrica.load_tracking_epts(
        meta_data=os.path.join(folder, "Sample_Game_3_metadata.xml"),
        raw_data=os.path.join(folder, "Sample_Game_3_tracking.txt"))
    df = ds.to_df(engine="pandas")
    home_id = ds.metadata.teams[0].team_id
    pids = sorted({c[:-2] for c in df.columns if c.endswith("_x") and c != "ball_x"})
    team = []
    for pid in pids:
        t = next(tm for tm in ds.metadata.teams for p in tm.players if p.player_id == pid)
        team.append(0 if t.team_id == home_id else 1)
    n = int(df["frame_id"].max())
    idx = df["frame_id"].values - 1
    px = np.full((n, len(pids)), np.nan, dtype=np.float32)
    py = np.full((n, len(pids)), np.nan, dtype=np.float32)
    # kloppy normalises to its own coordinate system; use the raw 0-1 Metrica values * pitch
    for j, pid in enumerate(pids):
        px[idx, j] = df[pid + "_x"].values * PITCH[0]
        py[idx, j] = df[pid + "_y"].values * PITCH[1]
    ball = np.full((n, 2), np.nan)
    ball[idx, 0] = df["ball_x"].values * PITCH[0]
    ball[idx, 1] = df["ball_y"].values * PITCH[1]
    with open(os.path.join(folder, "Sample_Game_3_events.json")) as handle:
        raw = json.load(handle)["data"]
    home_name = ds.metadata.teams[0].team_id
    ev = pd.DataFrame({
        "team": [0 if (e.get("team") or {}).get("id") == home_name else
                 1 if e.get("team") else -1 for e in raw],
        "type": [e["type"]["name"].upper() for e in raw],
        "subtype": [str(e.get("subtypes")) for e in raw],
        "period": [e["period"] for e in raw],
        "start": [e["start"]["frame"] - 1 for e in raw],
        "end": [(e["end"]["frame"] or e["start"]["frame"]) - 1 for e in raw]})
    return px, py, ball, np.array(team, dtype=np.int8), ev


# -- degradation ------------------------------------------------------------------------------
# Knobs picked so that the starter baseline (nearest player + majority window) reproduces on the
# bench the pattern the organisers published for it on the full BAR-ATH match (EVALUATION.md):
#   window      0      25     50     125
#   BAR-ATH  .653/.049 .675/.200 .672/.243 .636/.224   (macroF1 / spellF1)
#   degraded .649/.040 .684/.253 .685/.316 .635/.275   same order on both metrics
# The bench is ~0.07 easier on spellF1 in absolute terms, so compare versions with each other on the
# bench, never a bench number with a board number. Only published baseline scores were used.
VARIANTS = {
    "degraded": dict(live=0.18, bnoise=0.3, pnoise=0.0, false_rate=10, false_len=0.6, dead_present=0.3),
    "hard": dict(live=0.22, bnoise=0.5, pnoise=0.2, false_rate=6, false_len=1.0, dead_present=0.3),
    # BAR-ATH keeps the ball tracked through much of a stoppage (the tracking follows it while it
    # is carried or placed for the restart). These two copies make that the norm, so a dead-ball
    # detector cannot lean on "ball missing = dead".
    "tracked": dict(live=0.18, bnoise=0.3, pnoise=0.0, false_rate=10, false_len=0.6, dead_present=0.7),
    "tracked_all": dict(live=0.18, bnoise=0.3, pnoise=0.0, false_rate=10, false_len=0.6, dead_present=1.0),
}

# Simulated Smart Tagging set-piece tags. The real feed is automatic and incomplete: BAR-ATH has 59
# set-piece tags for a match with well over 100 restarts. Measured on BAR-ATH from automatic data only
# (run_labeller_v2.py): a tag STARTS before the restart kick, by a median 6.3 s (p10 1.5, p25 3.6,
# p75 8.1, p90 12.2). So each true restart (the end of a D stretch) is kept with TAG_KEEP and tagged
# that many seconds earlier, drawn from those quantiles; TAG_FALSE false tags per kept tag are added
# at random alive frames.
TAG_KEEP = 0.55
TAG_OFFSET_Q = ((0.0, 0.5), (0.10, 1.5), (0.25, 3.6), (0.50, 6.3), (0.75, 8.1), (0.90, 12.2),
                (1.0, 18.0))
TAG_FALSE = 0.05


def simulate_tags(ev, gt, rng):
    dead = gt == "D"
    e = np.diff(np.concatenate(([0], dead.astype(np.int8), [0])))
    restarts = np.flatnonzero(e == -1)
    keep = restarts[rng.random(len(restarts)) < TAG_KEEP]
    q, sec = zip(*TAG_OFFSET_Q)
    offs = np.interp(rng.random(len(keep)), q, sec)
    tags = keep - np.round(offs * FPS).astype(int)
    alive = np.flatnonzero(np.isin(gt, ("H", "A")))
    false = rng.choice(alive, int(TAG_FALSE * len(keep)))
    return np.sort(np.clip(np.r_[tags, false], 0, len(gt) - 1))


def degrade(px, py, ball0, gt, rng, live, bnoise, pnoise, false_rate, false_len, dead_present):
    """live: share of alive frames with no ball (extra gaps 2-30 s, log-uniform)
    bnoise / pnoise: gaussian jitter (m) on ball / players
    false_rate, false_len: false ball blobs at a random player's feet, per minute of play / mean s
    dead_present: last share of each dead stretch shows the ball at the restarting player's feet
    """
    ball = ball0.copy()
    px, py = px.copy(), py.copy()
    n = len(ball)
    dead = gt == "D"
    e = np.diff(np.concatenate(([0], dead.astype(np.int8), [0])))
    starts, ends = np.flatnonzero(e == 1), np.flatnonzero(e == -1)
    for s, t in zip(starts, ends):
        ball[s:t] = np.nan
    alive = np.isin(gt, ("H", "A"))
    idx = np.flatnonzero(alive)
    miss = (alive & np.isnan(ball[:, 0])).sum()
    while miss < live * alive.sum():
        length = int(np.exp(rng.uniform(np.log(2 * FPS), np.log(30 * FPS))))
        s = int(rng.choice(idx))
        seg = slice(s, s + length)
        miss += (alive[seg] & ~np.isnan(ball[seg, 0])).sum()
        ball[seg] = np.nan
    for s, t in zip(starts, ends):
        k = int(t - dead_present * (t - s))
        b = ball0[min(t, n - 1)]
        if k < t and not np.isnan(b[0]):
            # the restarter (nearest player to the restart spot at the restart) carries the ball
            # there: it sits at that one player's feet, so it moves at walking pace
            q = min(t, n - 1)
            dd = np.hypot(px[q] - b[0], py[q] - b[1])
            if np.isnan(dd).all():
                continue
            j = int(np.nanargmin(dd))
            ok = ~np.isnan(px[k:t, j])
            r = np.arange(k, t)[ok]
            ball[r, 0], ball[r, 1] = px[r, j] + 0.3, py[r, j]
    if bnoise:
        ball += rng.normal(0, bnoise, ball.shape)
    if pnoise:
        px += rng.normal(0, pnoise, px.shape).astype(np.float32)
        py += rng.normal(0, pnoise, py.shape).astype(np.float32)
    for s in rng.choice(idx, int(false_rate * alive.sum() / FPS / 60)):
        seg = slice(s, s + max(1, int(rng.exponential(false_len * FPS))))
        j = rng.integers(px.shape[1])
        ball[seg, 0] = px[seg, j] + rng.normal(0, 0.5)
        ball[seg, 1] = py[seg, j] + rng.normal(0, 0.5)
    return px, py, ball


# -- "real" copies (Nov 7): noise measured on BAR-ATH by test_scripts/noise_profile.py ------------
# The old copies were calibrated only on the published baseline scores and used the wrong noise
# (many false balls, 0.3-0.5 m ball jitter; BAR-ATH has almost no false balls and 0.03 m jitter).
# Measured on BAR-ATH from tracking only: few long ball gaps (median 11 s), player tracks broken
# into pieces (median 41 s, 9 of 11 players at p10), and the ball at a player's feet drawn ~1 m off
# him (0.46 m ahead along his run, 0.42 m to the side). The Metrica sample ball sits ~8 cm from
# the player, so the control rule was never tested against that offset.
#   real       reproduces the published baseline pattern (macroF1 .652/.664/.658/.616 vs the board's
#              .653/.675/.672/.636 for windows 0/25/50/125) and BAR-ATH's open-play distances
#              (median 1.20 m vs 1.24, within 1.5 m 62% vs 58%)
#   real_occ   + the player at the ball is lost 25% of contacts: a harder stress copy (baseline
#              lands below the board)
REAL_VARIANTS = {
    "real": dict(live=0.18, carry_lead=0.5, carry_sd=1.0, carry_r=2.0),
    "real_occ": dict(live=0.18, carry_lead=0.5, carry_sd=0.9, carry_r=2.0, occ_p=0.25),
}


def _lognorm(rng, med, sig, size=None):
    return np.exp(np.log(med) + sig * rng.standard_normal(size))

def degrade_real(px, py, ball0, gt, rng, live=0.15, gap_lo=5, gap_hi=60, dead_present=0.3,
                 bjit=0.03, bias_sd=0.0, bias_tau=2.0, on_med=40.0, on_sig=1.2, off_med=3.0,
                 off_sig=1.0, occ_r=1.5, occ_p=0.0, occ_med=2.0, dead_mode="feet", carry_lead=0.0, carry_sd=0.0, carry_tau=1.0, carry_r=1.0):
    ball = ball0.astype(float).copy()
    px, py = px.copy(), py.copy()
    n = len(ball)
    dead = gt == "D"
    e = np.diff(np.concatenate(([0], dead.astype(np.int8), [0])))
    starts, ends = np.flatnonzero(e == 1), np.flatnonzero(e == -1)
    for s, t in zip(starts, ends):
        ball[s:t] = np.nan
    # restarter carries the ball over the last dead_present of each stoppage (as in degrade())
    for s, t in zip(starts, ends):
        k = int(t - dead_present * (t - s))
        q = min(t, n - 1)
        b = ball0[q]
        if k < t and not np.isnan(b[0]) and dead_mode == "spot":
            # the ball lies at the restart spot (placed, waiting for the kick or throw)
            ball[k:t] = b + rng.normal(0, 0.05, (t - k, 2))
            continue
        if k < t and not np.isnan(b[0]):
            dd = np.hypot(px[q] - b[0], py[q] - b[1])
            if np.isnan(dd).all():
                continue
            j = int(np.nanargmin(dd))
            r = np.arange(k, t)[~np.isnan(px[k:t, j])]
            ball[r, 0], ball[r, 1] = px[r, j] + 0.3, py[r, j]
    # few, long live gaps
    alive = np.isin(gt, ("H", "A"))
    idx = np.flatnonzero(alive)
    miss = (alive & np.isnan(ball[:, 0])).sum()
    while miss < live * alive.sum():
        length = int(np.exp(rng.uniform(np.log(gap_lo * FPS), np.log(gap_hi * FPS))))
        s = int(rng.choice(idx))
        seg = slice(s, s + length)
        miss += (alive[seg] & ~np.isnan(ball[seg, 0])).sum()
        ball[seg] = np.nan
    # slowly varying position bias (ball estimate off the player's feet) + light jitter
    if bias_sd:
        # knots every bias_tau s, linear in between: smooth, so the frame-to-frame wobble stays
        # as small as measured (0.03 m) while the ball sits ~1-2 m off the nearest player
        step = max(1, int(bias_tau * FPS))
        kn = np.arange(0, n + step, step)
        kv = rng.normal(0, bias_sd, (len(kn), 2))
        t = np.arange(n)
        ball += np.c_[np.interp(t, kn, kv[:, 0]), np.interp(t, kn, kv[:, 1])]
    if carry_sd or carry_lead:
        # the ball at a player's feet is drawn ~1 m off him: carry_lead m ahead along his run plus
        # a smooth error (knots every carry_tau s) - measured on BAR-ATH: along +0.46, side 0.42 m
        ok0 = ~np.isnan(ball0[:, 0])
        d = np.full(px.shape, np.inf)
        d[ok0] = np.hypot(px[ok0] - ball0[ok0, 0:1], py[ok0] - ball0[ok0, 1:2])
        d = np.where(np.isnan(d), np.inf, d)
        j = d.argmin(1)
        at = ok0 & (d.min(1) < carry_r) & ~np.isnan(ball[:, 0])
        w = 12
        t = np.arange(n)
        a, c = np.clip(t - w, 0, n - 1), np.clip(t + w, 0, n - 1)
        vx = px[c, j] - px[a, j]
        vy = py[c, j] - py[a, j]
        sp = np.hypot(vx, vy)
        mv = np.isfinite(sp) & (sp > 0.5)
        ux = np.where(mv, vx / np.where(mv, sp, 1), 0)
        uy = np.where(mv, vy / np.where(mv, sp, 1), 0)
        step = max(1, int(carry_tau * FPS))
        kn = np.arange(0, n + step, step)
        kv = rng.normal(0, carry_sd, (len(kn), 2))
        ex, ey = np.interp(t, kn, kv[:, 0]), np.interp(t, kn, kv[:, 1])
        ball[at, 0] += carry_lead * ux[at] + ex[at]
        ball[at, 1] += carry_lead * uy[at] + ey[at]
    ball += rng.normal(0, bjit, ball.shape)
    # player track fragmentation: alternating on / off stretches per slot
    m = px.shape[1]
    drop = np.zeros(px.shape, dtype=bool)
    for j in range(m):
        k = int(rng.uniform(0, on_med * FPS))
        while k < n:
            off = max(1, int(_lognorm(rng, off_med, off_sig) * FPS))
            drop[k:k + off, j] = True
            k += off + max(FPS, int(_lognorm(rng, on_med, on_sig) * FPS))
    # occlusion: a player at the ball is lost with probability occ_p per contact
    if occ_p:
        ok = ~np.isnan(ball0[:, 0])
        d = np.full(px.shape, np.inf)
        d[ok] = np.hypot(px[ok] - ball0[ok, 0:1], py[ok] - ball0[ok, 1:2])
        near = d < occ_r
        for j in range(m):
            e2 = np.diff(np.concatenate(([0], near[:, j].astype(np.int8), [0])))
            for s in np.flatnonzero(e2 == 1):
                if rng.random() < occ_p:
                    L = max(1, int(_lognorm(rng, occ_med, 0.8) * FPS))
                    drop[s:s + L, j] = True
    px[drop] = np.nan
    py[drop] = np.nan
    return px, py, ball


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="folder holding g1/ g2/ g3/ with the sample files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-degrade", action="store_true")
    ap.add_argument("--real-only", action="store_true",
                    help="only (re)build the real copies from the <game>_clean.npz files in --out")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(SEED)
    if args.real_only:
        build_real(args.out)
        return
    games = [("g1", lambda: load_csv_game(os.path.join(args.src, "g1"), 1)),
             ("g2", lambda: load_csv_game(os.path.join(args.src, "g2"), 2)),
             ("g3", lambda: load_epts_game(os.path.join(args.src, "g3")))]
    for name, loader in games:
        px, py, ball, team, ev = loader()
        gt, blocks = gt_from_events(ev, len(ball))
        graded = np.isin(gt, ("H", "A", "D"))
        alive = np.isin(gt, ("H", "A"))
        changes = sum(1 for k in range(1, len(gt)) if gt[k] in "HA" and gt[k - 1] in "HA"
                      and gt[k] != gt[k - 1])
        print("{}: {} frames, {} graded, D {:.1f}%, ball missing {:.1f}% of alive / {:.1f}% of dead, "
              "{} H<->A changes".format(
                  name, len(gt), graded.sum(), 100 * (gt == "D").sum() / graded.sum(),
                  100 * (alive & np.isnan(ball[:, 0])).sum() / alive.sum(),
                  100 * ((gt == "D") & np.isnan(ball[:, 0])).sum() / max(1, (gt == "D").sum()),
                  changes))
        tags = simulate_tags(ev, gt, rng)
        np.savez_compressed(os.path.join(args.out, name + "_clean.npz"), px=px, py=py, ball=ball,
                            team=team, gt=gt, blocks=np.array(blocks), tags=tags)
        if not args.no_degrade:
            for vname, kw in VARIANTS.items():
                dpx, dpy, bd = degrade(px, py, ball, gt, rng, **kw)
                print("   {}: ball missing {:.1f}% of alive / {:.1f}% of dead / {:.1f}% of graded".format(
                    vname, 100 * (alive & np.isnan(bd[:, 0])).sum() / alive.sum(),
                    100 * ((gt == "D") & np.isnan(bd[:, 0])).sum() / max(1, (gt == "D").sum()),
                    100 * (graded & np.isnan(bd[:, 0])).sum() / graded.sum()))
                np.savez_compressed(os.path.join(args.out, "{}_{}.npz".format(name, vname)),
                                    px=dpx, py=dpy, ball=bd, team=team, gt=gt,
                                    blocks=np.array(blocks), tags=tags)

def build_real(folder):
    rng = np.random.default_rng(SEED + 16)
    for g in ("g1", "g2", "g3"):
        z = dict(np.load(os.path.join(folder, g + "_clean.npz"), allow_pickle=True))
        for vname, kw in REAL_VARIANTS.items():
            px, py, ball = degrade_real(z["px"], z["py"], z["ball"], z["gt"], rng, **kw)
            out = dict(z, px=px.astype(np.float32), py=py.astype(np.float32), ball=ball)
            np.savez_compressed(os.path.join(folder, "{}_{}.npz".format(g, vname)), **out)
            print("{} {}: ball tracked {:.1f}% of graded".format(
                g, vname, 100 * np.mean(~np.isnan(ball[np.isin(z["gt"], ("H", "A", "D")), 0]))))


if __name__ == "__main__":
    main()
