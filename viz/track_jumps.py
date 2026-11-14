"""Find sudden jumps in player tracks (our YOLO track ids; optionally the packed slots the labellers
read, or the ATD slots for diagnostics), list every one, and print the visualize_labeller command.

    python viz/track_jumps.py                                   # our YOLO tracks, both graded halves
    python viz/track_jumps.py --source slots                    # the cleaned slots v6 reads, vs YOLO
    python viz/track_jumps.py --source atd --range 73997 75081  # one window, ATD only
    python viz/track_jumps.py --top 30 --pad 50                 # 30 commands, +/- 2 s around each jump

WHAT IS A JUMP. The same rule as the Kaggle notebook's jumps_per_1000 (JUMP_SPEED, lag 5), so the
numbers are comparable: a track that moves faster than --speed (12 m/s) measured over --lag frames
(0.2 s). Per-frame jitter does not trigger it, an identity swap or a teleport does. Flagged steps of
one track that touch each other are merged into one event.
A second kind is checked too: a track that is unseen for 2..--max-gap frames and comes back
further away than it could have run (--reappear-speed, 8 m/s = the notebook's STITCH_MAX_SPEED).
For ATD this is the typical "slot reused for another player" case (46 slots for 22 players).

KIND (column `kind`)
  teleport   one track jumps on its own while continuously tracked
  swap       two tracks of the same team jump together and exchange places (pre of one ~ post of the other)
  reappear   the jump spans a gap: the track was lost, then came back somewhere else
  global     >= --global-min tracks jump within the same --lag frames: camera cut, homography or
             tracker reset, not a single player (look at the video for a cut)

CROSS-CHECK (columns `other_*`, `verdict`). Each jump is compared with the other source, nearest
person within --gate m (2 m, as head_to_head in the notebook):
  only_this_source   the other source has the person before, and it did NOT move with the jump -> error here
  both_jump          the other source moved the same way -> real fast move, or a shared camera / homography issue
  switched_person    after the jump this track sits on someone the other source tracks as a different id
  no_match_before / no_match_after   the other source has nobody within the gate -> can't tell
  other_missing      the other source was not loaded (--source atd or yolo only)

OUTPUT  outputs/track_audit/jumps_atd.csv, jumps_yolo.csv (one row per event), and on screen a
        summary per source plus commands such as
        python viz/visualize_labeller.py --compare --range 74010 74110 --ids --jumps outputs/track_audit/jumps_atd.csv

RULES. Same as visualize_labeller: these lists are for UNDERSTANDING tracking errors. Do not turn
BAR-ATH jumps into hand corrections or tuned constants (RULES.md rule 1). The CSVs are derived from
match data: keep them in outputs/ (git-ignored) and out of any cloud AI tool.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

FPS = 25
EVENT_COLS = ["source", "id", "team", "kind", "start", "end", "peak_speed", "jump_m", "gap_frames",
              "pre_x", "pre_y", "post_x", "post_y", "tracker_speed", "partner", "n_same_time",
              "other_id_before", "other_moved_m", "other_to_post_m", "other_id_after", "verdict"]


# -- 1. long format --------------------------------------------------------------------------------

def atd_long(px, py, ps, team, lo):
    """ATD matrices (n, slots) in metres -> rows (frame, id, team, x_m, y_m, tracker_speed).

    id = slot + 1, the number visualize_labeller --ids prints next to each ATD dot."""
    n, s = px.shape
    ok = np.isfinite(px) & np.isfinite(py)
    k, j = np.nonzero(ok)
    return pd.DataFrame({"frame": (lo + k).astype(np.int64), "id": (j + 1).astype(np.int64),
                         "team": np.asarray(team)[j].astype(np.int64), "x_m": px[k, j].astype(float),
                         "y_m": py[k, j].astype(float),
                         "tracker_speed": ps[k, j].astype(float) if ps is not None else np.nan})


def yolo_long(path, lo, hi):
    """Our tracks from the parquet -> the same columns (referees dropped, team H/A -> 0/1/-1)."""
    df = pd.read_parquet(path, filters=[("frame", ">=", lo), ("frame", "<=", hi)],
                         columns=["frame", "track_id", "role", "team", "x_m", "y_m"])
    df = df[(df.role != "referee") & np.isfinite(df.x_m) & np.isfinite(df.y_m)]
    return pd.DataFrame({"frame": df.frame.to_numpy(np.int64), "id": df.track_id.to_numpy(np.int64),
                         "team": df.team.map({"H": 0, "A": 1}).fillna(-1).to_numpy(np.int64),
                         "x_m": df.x_m.to_numpy(float), "y_m": df.y_m.to_numpy(float),
                         "tracker_speed": np.nan})


# -- 2. events -------------------------------------------------------------------------------------

def _merge(frames, gap):
    """Indices that start a new group in a sorted int array (a step > gap starts a new group)."""
    return np.r_[True, np.diff(frames) > gap]


def read_atd_slots(lo, hi):
    """The provided ATD player slots in metres (px, py, speed, team) - diagnostics only."""
    from games import GAMES, GAME, require
    from possession.io.teams import load_teams
    from viz.common import read_tracking
    meta_path, raw_path = require(["atd_meta", "atd_raw"])
    team = load_teams(meta_path).team
    players, _ = read_tracking(raw_path, lo, hi - lo + 1, len(team))
    L, W = GAMES[GAME]["pitch"]
    return players[:, :, 0] * L, players[:, :, 1] * W, players[:, :, 2], team


def find_jumps(T, source, speed=12.0, lag=5, max_gap=250, reappear_speed=8.0, min_jump=3.0, global_min=6,
               fps=FPS):
    """All jump events of one source. T: long rows (frame, id, team, x_m, y_m, tracker_speed)."""
    T = T.sort_values(["id", "frame"]).reset_index(drop=True)
    idv, fr = T.id.to_numpy(), T.frame.to_numpy()
    x, y, tm, ts = T.x_m.to_numpy(), T.y_m.to_numpy(), T.team.to_numpy(), T.tracker_speed.to_numpy()
    out = []

    # (a) continuous: row r vs row r-lag, same id, exactly lag frames apart.
    if len(T) > lag:
        same = (idv[lag:] == idv[:-lag]) & (fr[lag:] - fr[:-lag] == lag)
        sp = np.hypot(x[lag:] - x[:-lag], y[lag:] - y[:-lag]) * fps / lag
        hit = np.flatnonzero(same & (sp > speed)) + lag          # row index of the step's END
        if len(hit):
            new = np.r_[True, (idv[hit][1:] != idv[hit][:-1]) | (np.diff(fr[hit]) > lag)]
            for grp in np.split(hit, np.flatnonzero(new)[1:]):
                r_post, r_pre = grp[-1], grp[0] - lag
                pk = grp[np.argmax(sp[grp - lag])]
                out.append({"source": source, "id": int(idv[r_post]), "team": int(tm[r_post]),
                            "kind": "teleport", "start": int(fr[r_pre]), "end": int(fr[r_post]),
                            "peak_speed": float(sp[pk - lag]),
                            "jump_m": float(np.hypot(x[r_post] - x[r_pre], y[r_post] - y[r_pre])),
                            "gap_frames": 0, "pre_x": x[r_pre], "pre_y": y[r_pre],
                            "post_x": x[r_post], "post_y": y[r_post], "tracker_speed": ts[pk]})

    # (b) across a gap: consecutive rows of one id, 2..max_gap frames apart.
    if len(T) > 1:
        d_f = np.diff(fr)
        same = (idv[1:] == idv[:-1]) & (d_f > 1) & (d_f <= max_gap)
        dist = np.hypot(np.diff(x), np.diff(y))
        imp = dist / np.maximum(d_f, 1) * fps
        for r in np.flatnonzero(same & (imp > reappear_speed) & (dist > min_jump)):
            out.append({"source": source, "id": int(idv[r + 1]), "team": int(tm[r + 1]), "kind": "reappear",
                        "start": int(fr[r]), "end": int(fr[r + 1]), "peak_speed": float(imp[r]),
                        "jump_m": float(dist[r]), "gap_frames": int(d_f[r] - 1),
                        "pre_x": x[r], "pre_y": y[r], "post_x": x[r + 1], "post_y": y[r + 1],
                        "tracker_speed": ts[r + 1]})

    E = pd.DataFrame(out, columns=EVENT_COLS[:14])
    E["partner"] = -1
    E["n_same_time"] = 1
    return classify(E, lag, global_min).sort_values(["start", "id"]).reset_index(drop=True)


def classify(E, lag, global_min=6, swap_gate=2.0):
    """Mark global (many tracks at once) and swap (two tracks exchange places) events."""
    if E.empty:
        return E
    E = E.sort_values("end").reset_index(drop=True)
    ends = E.end.to_numpy()
    lo_i = np.searchsorted(ends, ends - lag, side="left")
    hi_i = np.searchsorted(ends, ends + lag, side="right")
    E["n_same_time"] = [E.id.iloc[a:b].nunique() for a, b in zip(lo_i, hi_i)]
    E.loc[E.n_same_time >= global_min, "kind"] = "global"
    for i in range(len(E)):
        if E.at[i, "kind"] == "global":
            continue
        for j in range(lo_i[i], hi_i[i]):
            if j == i or E.at[j, "id"] == E.at[i, "id"] or E.at[j, "team"] != E.at[i, "team"]:
                continue
            a_to_b = np.hypot(E.at[i, "post_x"] - E.at[j, "pre_x"], E.at[i, "post_y"] - E.at[j, "pre_y"])
            b_to_a = np.hypot(E.at[j, "post_x"] - E.at[i, "pre_x"], E.at[j, "post_y"] - E.at[i, "pre_y"])
            if a_to_b < swap_gate and b_to_a < swap_gate:
                E.at[i, "kind"], E.at[i, "partner"] = "swap", int(E.at[j, "id"])
                break
    return E


# -- 3. cross-check against the other source -------------------------------------------------------

def _nearest(by_frame, f, px, py, gate):
    """(id, x, y, dist) of the nearest row at frame f within gate, else None."""
    g = by_frame.get(f)
    if g is None or not len(g):
        return None
    d = np.hypot(g[:, 1] - px, g[:, 2] - py)
    k = int(np.argmin(d))
    return (int(g[k, 0]), g[k, 1], g[k, 2], d[k]) if d[k] <= gate else None


def cross_check(E, other, gate=2.0, speed=12.0, fps=FPS):
    """Fill other_* and verdict for every event of E using the other source's long rows."""
    E = E.copy()
    for c in ("other_id_before", "other_id_after"):
        E[c] = -1
    E["other_moved_m"] = np.nan
    E["other_to_post_m"] = np.nan
    if other is None:
        E["verdict"] = "other_missing"
        return E
    E["verdict"] = ""
    need = np.unique(np.r_[E.start.to_numpy(), E.end.to_numpy()])
    O = other[other.frame.isin(need)]
    by_frame = {int(f): g[["id", "x_m", "y_m"]].to_numpy(float) for f, g in O.groupby("frame")}
    for i, e in E.iterrows():
        before = _nearest(by_frame, int(e.start), e.pre_x, e.pre_y, gate)
        if before is None:
            E.at[i, "verdict"] = "no_match_before"
            continue
        oid = before[0]
        E.at[i, "other_id_before"] = oid
        g = by_frame.get(int(e.end))
        after_same = None
        if g is not None:
            m = g[:, 0] == oid
            if m.any():
                after_same = g[m][0]
        at_post = _nearest(by_frame, int(e.end), e.post_x, e.post_y, gate)
        if at_post is not None:
            E.at[i, "other_id_after"] = at_post[0]
        if after_same is not None:
            moved = float(np.hypot(after_same[1] - before[1], after_same[2] - before[2]))
            E.at[i, "other_moved_m"] = moved
            E.at[i, "other_to_post_m"] = float(np.hypot(after_same[1] - e.post_x, after_same[2] - e.post_y))
            if E.at[i, "other_to_post_m"] <= gate:
                E.at[i, "verdict"] = "both_jump"
                continue
        if at_post is None:
            E.at[i, "verdict"] = "no_match_after"
        elif at_post[0] != oid:
            E.at[i, "verdict"] = "switched_person"
        else:
            E.at[i, "verdict"] = "only_this_source"
        if E.at[i, "verdict"] != "switched_person" and after_same is not None:
            # The other source kept the person at a plausible speed: the jump is ours.
            if E.at[i, "other_moved_m"] <= speed * (e.end - e.start) / fps + gate:
                E.at[i, "verdict"] = "only_this_source"
    return E


# -- 4. report -------------------------------------------------------------------------------------

def per_1000(T, speed=12.0, lag=5, fps=FPS):
    """The notebook's jumps_per_1000: flagged lag-steps per 1,000 steps."""
    T = T.sort_values(["id", "frame"])
    idv, fr = T.id.to_numpy(), T.frame.to_numpy()
    x, y = T.x_m.to_numpy(), T.y_m.to_numpy()
    if len(T) <= lag:
        return float("nan")
    same = (idv[lag:] == idv[:-lag]) & (fr[lag:] - fr[:-lag] == lag)
    sp = np.hypot(x[lag:] - x[:-lag], y[lag:] - y[:-lag])[same] * fps / lag
    return 1000.0 * float(np.mean(sp > speed)) if len(sp) else float("nan")


def summary(name, T, E, speed, lag):
    print("\n== {} : {} tracks, {:,} rows, jumps_per_1000 = {:.2f}".format(
        name.upper(), T.id.nunique(), len(T), per_1000(T, speed, lag)))
    if E.empty:
        print("   no jumps")
        return
    print("   events by kind:     " + "  ".join("{} {}".format(k, v) for k, v in E.kind.value_counts().items()))
    print("   events by verdict:  " + "  ".join("{} {}".format(k, v) for k, v in E.verdict.value_counts().items()))
    top = E[E.kind != "global"].id.value_counts().head(8)
    print("   most jumpy ids:     " + "  ".join("{}x{}".format(i, c) for i, c in top.items()))


def commands(E, path, top, pad, n_frames):
    if E.empty:
        return
    pick = E[E.kind != "global"].sort_values("peak_speed", ascending=False).head(top)
    print("\n   watch the {} fastest non-global jumps (ids are the numbers --ids prints):".format(len(pick)))
    rel = os.path.relpath(path, ROOT)
    for e in pick.itertuples():
        a, b = max(0, e.start - pad), min(n_frames - 1, e.end + pad)
        print("   # id {} {} {:.0f} m/s {:.1f} m {} -> python viz/visualize_labeller.py --compare "
              "--range {} {} --ids --jumps {}".format(e.id, e.kind, e.peak_speed, e.jump_m, e.verdict, a, b, rel))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", choices=("yolo", "slots", "both", "atd"), default="yolo",
                    help="yolo = our raw tracks; slots = the packed, cleaned YOLO slots the labellers "
                         "read (possession.io.load); atd / both = also the provided ATD slots "
                         "(diagnostics only, the pipeline no longer uses them)")
    ap.add_argument("--range", type=int, nargs=2, metavar=("FIRST", "LAST"), help="video frames (default: graded halves)")
    ap.add_argument("--players-file", default=os.path.join(ROOT, "data", "kaggle_outputs", "final_tracks_full.parquet"))
    ap.add_argument("--speed", type=float, default=12.0, help="m/s over --lag frames that counts as a jump")
    ap.add_argument("--lag", type=int, default=5, help="frames (5 = 0.2 s, as the notebook)")
    ap.add_argument("--max-gap", type=int, default=250, help="longest unseen gap checked for reappear jumps")
    ap.add_argument("--reappear-speed", type=float, default=8.0, help="m/s implied across a gap")
    ap.add_argument("--global-min", type=int, default=6, help="tracks jumping together = a global event")
    ap.add_argument("--gate", type=float, default=2.0, help="metres to match a person across sources")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--pad", type=int, default=50, help="frames shown before / after a jump")
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "outputs", "track_audit"))
    args = ap.parse_args()

    from games import GAME, GAMES, half_video_blocks
    from possession.io import load
    n_frames = GAMES[GAME]["n_video_frames"]
    spans = [tuple(args.range)] if args.range else [(a, b) for _, a, b in half_video_blocks(GAME)]
    os.makedirs(args.out_dir, exist_ok=True)

    atd, yolo = [], []
    for lo, hi in spans:
        print("loading {}..{}".format(lo, hi), flush=True)
        if args.source in ("both", "atd"):
            atd.append(atd_long(*read_atd_slots(lo, hi), lo))
        if args.source == "slots":
            tr = load(lo, hi)
            atd.append(atd_long(tr.px, tr.py, tr.ps, tr.teams.team, tr.lo))
        if args.source in ("both", "yolo"):
            yolo.append(yolo_long(args.players_file, lo, hi))
    A = pd.concat(atd, ignore_index=True) if atd else None
    Y = pd.concat(yolo, ignore_index=True) if yolo else None

    kw = dict(speed=args.speed, lag=args.lag, max_gap=args.max_gap, reappear_speed=args.reappear_speed,
              global_min=args.global_min)
    first = "slots" if args.source == "slots" else "atd"
    for name, T, other in ((first, A, Y), ("yolo", Y, A)):
        if T is None:
            continue
        E = find_jumps(T, name, **kw)
        E = cross_check(E, other, args.gate, args.speed)[EVENT_COLS]
        path = os.path.join(args.out_dir, "jumps_{}.csv".format(name))
        E.to_csv(path, index=False, float_format="%.2f")
        summary(name, T, E, args.speed, args.lag)
        print("   -> {}".format(path))
        commands(E, path, args.top, args.pad, n_frames)


if __name__ == "__main__":
    main()
