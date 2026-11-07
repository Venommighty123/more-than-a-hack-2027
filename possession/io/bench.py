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
  dead   from the end of the event that put the ball out (BALL OUT, a SHOT ending OUT/GOAL, or the
         last event before a SET PIECE) to the start of the SET PIECE that restarts play; from a
         FAULT RECEIVED to the free kick. Before kick-off / after the end = X.
  The organisers' truth comes from corrected tracking, not events, so this is a proxy -- the same
  proxy for every version of a labeller, so it ranks versions fairly.

DEGRADED COPIES ("degraded", "hard") add live ball gaps, jitter, false ball blobs and a ball at the
restarter's feet late in dead stretches, calibrated so the starter baseline ranks its four
windows on the bench as it does on BAR-ATH. Only published baseline scores were used for that.
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
}


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
        prev_end = first
        for i, r in enumerate(e.itertuples()):
            s, t = int(r.start), int(max(r.end, r.start))
            if r.type == "SET PIECE" and s > first:
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
                if 0 < s - k0 < 120 * FPS:
                    dead[k0 - first:s - first] = True
            if r.type not in ("CARD",):
                prev_end = max(prev_end, t)
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


# ------------------------------------------------------------------------------------------------
# Build and read the bench
# ------------------------------------------------------------------------------------------------

def _save(path, px, py, ball, team, gt, blocks):
    np.savez_compressed(path, px=px, py=py, ball=ball, team=team, gt=gt, blocks=np.array(blocks))


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
        _save(os.path.join(out, name + "_clean.npz"), px, py, ball, team, gt, blocks)
        if with_degraded:
            for vname, kw in VARIANTS.items():
                dpx, dpy, bd = degrade(px, py, ball, gt, rng, **kw)
                row[vname + "_ball_missing_graded_pct"] = round(
                    100.0 * float((graded & np.isnan(bd[:, 0])).sum()) / max(1, graded.sum()), 1)
                _save(os.path.join(out, "{}_{}.npz".format(name, vname)), dpx, dpy, bd, team, gt,
                      blocks)
        if verbose:
            print(row)
        summary.append(row)
    return summary


def load_bench_game(bench_dir: str, name: str, variant: str = "degraded") -> BenchGame:
    path = os.path.join(bench_dir, "{}_{}.npz".format(name, variant))
    z = np.load(path, allow_pickle=False)
    blocks = [tuple(int(v) for v in b) for b in z["blocks"]]
    return BenchGame(name, variant, z["px"], z["py"], z["ball"], z["team"], z["gt"], blocks)


def iter_bench(bench_dir: str, variant: str = "degraded", games=GAMES):
    for name in games:
        yield load_bench_game(bench_dir, name, variant)
