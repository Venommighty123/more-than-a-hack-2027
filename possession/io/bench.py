"""The test bench: Metrica's public sample games, shaped like BAR-ATH, with a possession truth.

From Ashwyn's ``test_scripts/bench/build_metrica_bench.py`` (logic unchanged, split into library
functions; the command line lives in ``scripts/bench_build.py``).

RULES.md forbids tuning on the provided match and on any human data about it, but allows data
about OTHER games in any amount. Metrica publishes three anonymised matches with tracking AND
event data (github.com/metrica-sports/sample-data). Each becomes::

    px, py        (frames, slots) player positions in metres (105 x 68), NaN when absent
    ball          (frames, 2) ball in metres, NaN when not tracked
    team          0 home / 1 away per slot
    blocks        [(period, first_frame, last_frame)] graded stretch of each half (0-based)
    gt            per-frame labels H / A / D / X (X = not graded)

GROUND TRUTH FROM EVENTS (an approximation of CHALLENGE.md's definition)
  alive  the team of the most recent on-the-ball event (PASS, CARRY, SET PIECE, SHOT, RECOVERY,
         BALL LOST). A contested / loose ball keeps the previous team. A change happens at the
         start of the new team's first event.
  dead   from what stopped play (ball out, a shot going out or in, a foul, an offside) to the
         restart (any SET PIECE, or a goal kick recorded as PASS / BALL LOST "GOAL KICK").
         Dead-time events in between (keeper's RECOVERY, CARD) are skipped. Before kick-off /
         after the end = X. (Ashwyn's Nov 7 fix; the first bench stopped D at the last event
         before a set piece and missed goal kicks.)
  The organisers' truth comes from corrected tracking, not events, so this is a proxy -- the same
  proxy for every version of a labeller, so it ranks versions fairly.

DEGRADED COPIES ("degraded", "hard", "tracked", "tracked_all") add live ball gaps, jitter, false ball blobs and a ball at the
restarter's feet late in dead stretches, calibrated so the starter baseline ranks its four
windows on the bench as it does on BAR-ATH. Only published baseline scores were used for that.
REAL COPIES ("real", "real_occ", Ashwyn Nov 7) use the noise MEASURED on BAR-ATH from tracking
only (backup/ashwyn_noise_profile.py): few long live ball gaps, fragmented player tracks, and the
ball at a player's feet drawn ~1 m off him (0.5 m ahead along his run). They are the copies the
v4 decisions were taken on. SIMULATED TAGS: every copy also stores Smart-Tagging-like set-piece
tags drawn from the timing measured on BAR-ATH (``simulate_tags``).
The bench reads ~0.07 higher than the board on spellF1: compare versions with each other on the
bench, never a bench number with a board number.

Bench files are written to outputs/bench/ (git-ignored). The sample data is public, so the bench
can also be built and scored in the cloud.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np

FPS = 25
PITCH = (105.0, 68.0)
OWNER_TYPES = {"PASS", "CARRY", "SET PIECE", "SHOT", "RECOVERY", "BALL LOST"}
SEED = 7
GAMES = ("g1", "g2", "g3")

# Degradation knobs. With them the starter baseline reproduces on the bench the pattern the
# organisers published for BAR-ATH (EVALUATION.md):
#   window      0         25        50        125          (macroF1 / spellF1)
#   BAR-ATH  .653/.049 .675/.200 .672/.243 .636/.224
#   degraded .649/.040 .684/.253 .685/.316 .635/.275       same order on both metrics
VARIANTS = {
    "degraded": dict(live=0.18, bnoise=0.3, pnoise=0.0, false_rate=10, false_len=0.6,
                     dead_present=0.3),
    "hard": dict(live=0.22, bnoise=0.5, pnoise=0.2, false_rate=6, false_len=1.0,
                 dead_present=0.3),
    # BAR-ATH keeps the ball tracked through much of a stoppage (carried or placed for the
    # restart). These copies make that the norm, so a dead-ball detector cannot lean on
    # "ball missing = dead".
    "tracked": dict(live=0.18, bnoise=0.3, pnoise=0.0, false_rate=10, false_len=0.6,
                    dead_present=0.7),
    "tracked_all": dict(live=0.18, bnoise=0.3, pnoise=0.0, false_rate=10, false_len=0.6,
                        dead_present=1.0),
}

# "real" copies (Ashwyn, Nov 7): noise measured on BAR-ATH by backup/ashwyn_noise_profile.py
#   real       reproduces the published baseline pattern and BAR-ATH's open-play distances
#   real_occ   + the player at the ball is lost 25% of contacts (harder stress copy)
REAL_VARIANTS = {
    "real": dict(live=0.18, carry_lead=0.5, carry_sd=1.0, carry_r=2.0),
    "real_occ": dict(live=0.18, carry_lead=0.5, carry_sd=0.9, carry_r=2.0, occ_p=0.25),
}
ALL_VARIANTS = ("clean",) + tuple(VARIANTS) + tuple(REAL_VARIANTS)

# Simulated Smart Tagging set-piece tags. The real feed is automatic and incomplete (BAR-ATH: 59
# set-piece tags for well over 100 restarts). Measured on BAR-ATH from automatic data only: a tag
# STARTS before the restart kick by a median 6.3 s (p10 1.5, p25 3.6, p75 8.1, p90 12.2). Each true
# restart is kept with TAG_KEEP and tagged that many seconds earlier; TAG_FALSE false tags per kept
# tag are added at random alive frames.
TAG_KEEP = 0.55
TAG_OFFSET_Q = ((0.0, 0.5), (0.10, 1.5), (0.25, 3.6), (0.50, 6.3), (0.75, 8.1), (0.90, 12.2),
                (1.0, 18.0))
TAG_FALSE = 0.05


def simulate_tags(gt, rng):
    """Set-piece tag frames for one bench game (see TAG_* above)."""
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


@dataclass
class BenchGame:
    name: str
    variant: str
    px: np.ndarray
    py: np.ndarray
    ball: np.ndarray
    team: np.ndarray
    gt: np.ndarray
    blocks: list
    fps: int = FPS
    pitch: tuple = PITCH
    tags: np.ndarray | None = None      # simulated set-piece tag frames (benches built after Nov 7)

    @property
    def n(self) -> int:
        return len(self.ball)

    @property
    def bx(self) -> np.ndarray:
        return self.ball[:, 0]

    @property
    def by(self) -> np.ndarray:
        return self.ball[:, 1]


# ------------------------------------------------------------------------------------------------
# Events -> ground truth
# ------------------------------------------------------------------------------------------------

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
        cur = -1
        marks = [(int(r.start), int(r.team)) for r in e.itertuples()
                 if r.type in OWNER_TYPES and r.team in (0, 1)]
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
                if q.type in ("RECOVERY", "CARD", "CHALLENGE", "BALL LOST") and \
                        s - q.start < 300 * FPS:
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


# ------------------------------------------------------------------------------------------------
# Loaders for the sample data
# ------------------------------------------------------------------------------------------------

def load_csv_game(folder, g):
    """Sample games 1 and 2 (CSV tracking + CSV events)."""
    import pandas as pd
    ev = pd.read_csv(os.path.join(folder, "Sample_Game_{}_RawEventsData.csv".format(g)))
    ev = pd.DataFrame({
        "team": ev["Team"].map({"Home": 0, "Away": 1}).fillna(-1).astype(int),
        "type": ev["Type"].astype(str).str.upper(), "subtype": ev["Subtype"].astype(str),
        "period": ev["Period"], "start": ev["Start Frame"] - 1, "end": ev["End Frame"] - 1})
    xs, ys, team = [], [], []
    frames = bx = by = None
    for side, code in (("Home", 0), ("Away", 1)):
        path = os.path.join(folder, "Sample_Game_{}_RawTrackingData_{}_Team.csv".format(g, side))
        t = pd.read_csv(path, skiprows=2)
        names = list(t.columns)
        frames = t["Frame"].values
        for p in [c for c in names if c.startswith("Player")]:
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


def _epts_layout(meta_path):
    """FIFA/EPTS metadata -> (home team id, {player: team id}, [(first, last, [player per column])])."""
    import xml.etree.ElementTree as ET
    root = ET.parse(meta_path).getroot()
    tag = lambda e: e.tag.rsplit("}", 1)[-1]                        # noqa: E731
    teams = [e.get("id") for e in root.iter() if tag(e) == "Team"]
    team_of = {e.get("id"): e.get("teamId") for e in root.iter() if tag(e) == "Player"}
    channel = {e.get("id"): (e.get("playerId"), e.get("channelId"))
               for e in root.iter() if tag(e) == "PlayerChannel"}
    specs = []
    for spec in (e for e in root.iter() if tag(e) == "DataFormatSpecification"):
        cols = [channel[r.get("playerChannelId")][0] for r in spec.iter()
                if tag(r) == "PlayerChannelRef" and channel[r.get("playerChannelId")][1] == "x"]
        specs.append((int(spec.get("startFrame")), int(spec.get("endFrame")), cols))
    return teams[0], team_of, specs


def _read_epts_tracking(meta_path, raw_path):
    """Stream an EPTS raw file (frame:x,y;x,y;...:bx,by) into per-player arrays, low memory.

    Column order changes at every DataFormatSpecification (substitutions), so each line is mapped
    through the specification covering its frame. Players are sorted by id, like kloppy's columns.
    """
    home_id, team_of, specs = _epts_layout(meta_path)
    pids = sorted({p for _, _, cols in specs for p in cols})
    col_of = {p: j for j, p in enumerate(pids)}
    n = max(last for _, last, _ in specs)
    px = np.full((n, len(pids)), np.nan, dtype=np.float32)
    py = np.full((n, len(pids)), np.nan, dtype=np.float32)
    ball = np.full((n, 2), np.nan)
    maps = [(a, b, np.array([col_of[p] for p in cols])) for a, b, cols in specs]
    m = 0
    with open(raw_path) as handle:
        for line in handle:
            head, _, rest = line.partition(":")
            if not head.isdigit():
                continue
            f = int(head)
            while m < len(maps) - 1 and f > maps[m][1]:
                m += 1
            a, b, cols = maps[m]
            if not a <= f <= b:
                continue
            players, _, bal = rest.strip().partition(":")
            vals = np.array(players.replace(";", ",").split(","), dtype=np.float32)
            if len(vals) != 2 * len(cols):
                continue
            px[f - 1, cols] = vals[0::2] * PITCH[0]
            py[f - 1, cols] = vals[1::2] * PITCH[1]
            bv = bal.split(",")
            if len(bv) >= 2 and bv[0][:1] not in ("N", "n"):
                ball[f - 1] = (float(bv[0]) * PITCH[0], float(bv[1]) * PITCH[1])
    team = np.array([0 if team_of.get(p) == home_id else 1 for p in pids], dtype=np.int8)
    return px, py, ball, team, home_id


def _read_kloppy_tracking(meta_path, raw_path):
    """Ashwyn's original path through kloppy (needs several GB of RAM for game 3)."""
    from kloppy import metrica
    ds = metrica.load_tracking_epts(meta_data=meta_path, raw_data=raw_path)
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
    for j, pid in enumerate(pids):                 # raw 0-1 Metrica values * pitch
        px[idx, j] = df[pid + "_x"].values * PITCH[0]
        py[idx, j] = df[pid + "_y"].values * PITCH[1]
    ball = np.full((n, 2), np.nan)
    ball[idx, 0] = df["ball_x"].values * PITCH[0]
    ball[idx, 1] = df["ball_y"].values * PITCH[1]
    return px, py, ball, np.array(team, dtype=np.int8), home_id


def load_epts_game(folder, use_kloppy=False):
    """Sample game 3 (FIFA/EPTS tracking, JSON events). Built-in streaming reader by default;
    ``use_kloppy=True`` takes Ashwyn's original kloppy path."""
    import pandas as pd
    meta = os.path.join(folder, "Sample_Game_3_metadata.xml")
    raw = os.path.join(folder, "Sample_Game_3_tracking.txt")
    reader = _read_kloppy_tracking if use_kloppy else _read_epts_tracking
    px, py, ball, team, home_id = reader(meta, raw)
    with open(os.path.join(folder, "Sample_Game_3_events.json")) as handle:
        events = json.load(handle)["data"]
    ev = pd.DataFrame({
        "team": [0 if (e.get("team") or {}).get("id") == home_id else
                 1 if e.get("team") else -1 for e in events],
        "type": [e["type"]["name"].upper() for e in events],
        "subtype": [str(e.get("subtypes")) for e in events],
        "period": [e["period"] for e in events],
        "start": [e["start"]["frame"] - 1 for e in events],
        "end": [(e["end"]["frame"] or e["start"]["frame"]) - 1 for e in events]})
    return px, py, ball, team, ev


# ------------------------------------------------------------------------------------------------
# Degradation
# ------------------------------------------------------------------------------------------------

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


# ------------------------------------------------------------------------------------------------
# Build and read the bench
# ------------------------------------------------------------------------------------------------

def _save(path, px, py, ball, team, gt, blocks, tags=None):
    extra = {} if tags is None else {"tags": np.asarray(tags, dtype=np.int64)}
    np.savez_compressed(path, px=px, py=py, ball=ball, team=team, gt=gt, blocks=np.array(blocks),
                        **extra)


def build_bench(src: str, out: str, with_degraded: bool = True, seed: int = SEED,
                verbose: bool = True, games=GAMES, use_kloppy: bool = False) -> list[dict]:
    """Write <game>_clean.npz (+ _degraded, _hard) for g1..g3 found under ``src``/g1..g3."""
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(seed)
    loaders = [("g1", lambda: load_csv_game(os.path.join(src, "g1"), 1)),
               ("g2", lambda: load_csv_game(os.path.join(src, "g2"), 2)),
               ("g3", lambda: load_epts_game(os.path.join(src, "g3"), use_kloppy))]
    loaders = [(name, fn) for name, fn in loaders if name in games]
    summary = []
    for name, loader in loaders:
        px, py, ball, team, ev = loader()
        gt, blocks = gt_from_events(ev, len(ball))
        graded = np.isin(gt, ("H", "A", "D"))
        alive = np.isin(gt, ("H", "A"))
        dead = gt == "D"
        changes = int(sum(1 for k in range(1, len(gt)) if gt[k] in "HA" and gt[k - 1] in "HA"
                          and gt[k] != gt[k - 1]))
        row = {"game": name, "frames": len(gt), "graded": int(graded.sum()),
               "dead_pct": round(100.0 * float(dead.sum()) / max(1, graded.sum()), 1),
               "ball_missing_alive_pct": round(100.0 * float((alive & np.isnan(ball[:, 0])).sum())
                                               / max(1, alive.sum()), 1),
               "changes": changes}
        tags = simulate_tags(gt, rng)
        _save(os.path.join(out, name + "_clean.npz"), px, py, ball, team, gt, blocks, tags)
        if with_degraded:
            for vname, kw in VARIANTS.items():
                dpx, dpy, bd = degrade(px, py, ball, gt, rng, **kw)
                row[vname + "_ball_missing_graded_pct"] = round(
                    100.0 * float((graded & np.isnan(bd[:, 0])).sum()) / max(1, graded.sum()), 1)
                _save(os.path.join(out, "{}_{}.npz".format(name, vname)), dpx, dpy, bd, team, gt,
                      blocks, tags)
        if verbose:
            print(row)
        summary.append(row)
    if with_degraded:
        build_real(out, seed, games=[g for g, _ in loaders], verbose=verbose)
    return summary


def build_real(folder: str, seed: int = SEED, games=GAMES, verbose: bool = True):
    """(Re)build the "real" copies from the <game>_clean.npz files (Ashwyn's --real-only)."""
    rng = np.random.default_rng(seed + 16)
    for g in games:
        z = dict(np.load(os.path.join(folder, g + "_clean.npz"), allow_pickle=True))
        for vname, kw in REAL_VARIANTS.items():
            px, py, ball = degrade_real(z["px"], z["py"], z["ball"], z["gt"], rng, **kw)
            out = dict(z, px=px.astype(np.float32), py=py.astype(np.float32), ball=ball)
            np.savez_compressed(os.path.join(folder, "{}_{}.npz".format(g, vname)), **out)
            if verbose:
                print("{} {}: ball tracked {:.1f}% of graded".format(
                    g, vname, 100 * np.mean(~np.isnan(ball[np.isin(z["gt"], ("H", "A", "D")), 0]))))


def load_bench_game(bench_dir: str, name: str, variant: str = "degraded") -> BenchGame:
    path = os.path.join(bench_dir, "{}_{}.npz".format(name, variant))
    z = np.load(path, allow_pickle=False)
    blocks = [tuple(int(v) for v in b) for b in z["blocks"]]
    tags = z["tags"] if "tags" in z.files else None
    return BenchGame(name, variant, z["px"], z["py"], z["ball"], z["team"], z["gt"], blocks,
                     tags=tags)


def iter_bench(bench_dir: str, variant: str = "degraded", games=GAMES):
    for name in games:
        yield load_bench_game(bench_dir, name, variant)
