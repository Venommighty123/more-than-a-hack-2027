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
  dead frames    from the end of the event that put the ball out (BALL OUT, a SHOT ending OUT/GOAL,
                 or the last event before a SET PIECE) to the start of the SET PIECE that restarts
                 play; from a FAULT RECEIVED to the free kick. Before kick-off / after the end = X.
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
        # dead stretches: up to every set piece
        prev_end = first
        for i, r in enumerate(e.itertuples()):
            s, t = int(r.start), int(max(r.end, r.start))
            if r.type == "SET PIECE" and s > first:
                # dead from the end of the last event that is not part of the restart
                k0 = prev_end
                for q in e.iloc[:i][::-1].itertuples():
                    if q.type in ("BALL OUT",) or (q.type == "SHOT" and any(
                            w in str(q.subtype).upper() for w in ("OUT", "GOAL"))):
                        k0 = int(max(q.end, q.start))
                        break
                    if q.type == "FAULT RECEIVED":
                        k0 = int(q.start)
                        break
                    if q.type in OWNER_TYPES or q.type == "CHALLENGE":
                        k0 = int(max(q.end, q.start))
                        break
                if s - k0 > 0 and s - k0 < 120 * FPS:
                    dead[k0 - first:s - first] = True
            if r.type not in ("CARD",):
                prev_end = max(prev_end, t)
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
}


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
            dd = np.hypot(px[k:t] - b[0], py[k:t] - b[1])
            j = np.where(np.isnan(dd), 1e9, dd).argmin(1)
            r = np.arange(k, t)
            ball[r, 0], ball[r, 1] = px[r, j], py[r, j]
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="folder holding g1/ g2/ g3/ with the sample files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-degrade", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(SEED)
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
        np.savez_compressed(os.path.join(args.out, name + "_clean.npz"), px=px, py=py, ball=ball,
                            team=team, gt=gt, blocks=np.array(blocks))
        if not args.no_degrade:
            for vname, kw in VARIANTS.items():
                dpx, dpy, bd = degrade(px, py, ball, gt, rng, **kw)
                print("   {}: ball missing {:.1f}% of alive / {:.1f}% of dead / {:.1f}% of graded".format(
                    vname, 100 * (alive & np.isnan(bd[:, 0])).sum() / alive.sum(),
                    100 * ((gt == "D") & np.isnan(bd[:, 0])).sum() / max(1, (gt == "D").sum()),
                    100 * (graded & np.isnan(bd[:, 0])).sum() / graded.sum()))
                np.savez_compressed(os.path.join(args.out, "{}_{}.npz".format(name, vname)),
                                    px=dpx, py=dpy, ball=bd, team=team, gt=gt,
                                    blocks=np.array(blocks))

if __name__ == "__main__":
    main()
