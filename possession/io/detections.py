"""Our player tracks (Kaggle YOLO notebook) in the slot layout the labellers take.

The labellers read players as ``px, py`` of shape (n_frames, n_slots) in metres plus one team per
slot, like the ATD. Our tracks come as one row per frame per track instead
(``data/kaggle_outputs/final_tracks_full.parquet``, written by
kaggle/kaggle_player_tracking.ipynb):

    frame, half, track_id, role (player / goalkeeper / referee), team (H / A / ""),
    conf, x1, y1, x2, y2, x_m, y_m, team_p_home

``x_m, y_m`` come from the provided homography, so they are in the ATD's frame (0-1 x 105.3 x 68):
no conversion is needed (median distance to the ATD on matched players: 0.14 m).

Packing into slots
------------------
Referees are dropped (the ATD has none). Every track is cut into SEGMENTS wherever its team
changes, and each segment is given a slot of its team: the free slot released longest ago, and
never one released on the previous frame, so a slot never jumps from one person to another
between two consecutive frames (the stoppage detector reads speeds from these columns).

Team per row (``team_mode``)
----------------------------
* ``"track"``  the notebook's vote over the whole track. One wrong merge of two people of
               different teams (an ID switch) gives the whole track one team.
* ``"local"``  (default) the mean of ``team_p_home`` over +/- ``window`` frames inside the same
               track (team scores exist on every 3rd frame), falling back to the track vote where
               no score is near. Follows ID switches; costs the odd short flicker.

Clean-up (``clean=True``, default)
----------------------------------
The slots then go through possession/preprocess/fix_players.py with gap filling off: every slot is
cut at its jumps (> 12 m/s over 0.2 s) and a run that sits on another slot for most of its life
(one person tracked twice, the main cause of "more than 11 on a team") is blanked. Gaps are NOT
interpolated: a slot holds several tracks one after another, and filling between them would join
two different people.

    from possession.io.detections import load_detected, read_rows
    det = load_detected()                    # dict: px, py, team, stats   (the labeller input)
    rows = read_rows(None, 73997, 75081)     # long rows of a window, referees included (viz)

This is THE player source of the pipeline: possession.io.tracking.load() builds its Tracking from
it. The provided ATD player tracks are no longer used.
"""
from __future__ import annotations

import os

import numpy as np

from .teams import AWAY, HOME

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_FILE = os.path.join("data", "kaggle_outputs", "final_tracks_full.parquet")
CACHE_DIR = os.path.join(ROOT, "outputs", "cache")


def _read(path):
    try:
        import pandas as pd
        return pd.read_parquet(path, columns=["frame", "track_id", "role", "team", "x_m", "y_m",
                                               "team_p_home"])
    except ImportError as e:
        raise ImportError("Reading {} needs pandas + pyarrow: pip install pyarrow".format(path)) from e


def _local_team(df, window):
    """Mean P(home) over +/- window frames within each track (NaN-aware); NaN where none."""
    df = df.sort_values(["track_id", "frame"])
    p = df["team_p_home"].to_numpy(np.float64)
    ok = np.isfinite(p)
    tid = df["track_id"].to_numpy()
    fr = df["frame"].to_numpy()
    out = np.full(len(df), np.nan)
    starts = np.r_[0, np.flatnonzero(tid[1:] != tid[:-1]) + 1, len(df)]
    for a, b in zip(starts[:-1], starts[1:]):
        f = fr[a:b]
        cs = np.r_[0.0, np.cumsum(np.where(ok[a:b], p[a:b], 0.0))]
        cn = np.r_[0, np.cumsum(ok[a:b])]
        lo = np.searchsorted(f, f - window, side="left")
        hi = np.searchsorted(f, f + window, side="right")
        n = cn[hi] - cn[lo]
        out[a:b] = np.where(n > 0, (cs[hi] - cs[lo]) / np.maximum(n, 1), np.nan)
    return df, out


def build_slots(df, n_frames, team_mode="local", window=12):
    """(px, py, team, stats) from the long table. See the module docstring."""
    df = df[df["role"] != "referee"]
    vote = df["team"].map({"H": HOME, "A": AWAY}).astype("float64")
    df = df.assign(vote=vote.to_numpy())
    if team_mode == "local":
        df, p = _local_team(df, window)
        team = np.where(np.isfinite(p), np.where(p >= 0.5, HOME, AWAY), df["vote"].to_numpy())
    elif team_mode == "track":
        df = df.sort_values(["track_id", "frame"])
        team = df["vote"].to_numpy()
    else:
        raise ValueError("team_mode must be 'local' or 'track'")
    keep = np.isfinite(team)
    df = df[keep]
    team = team[keep].astype(np.int8)
    fr = df["frame"].to_numpy(np.int64)
    tid = df["track_id"].to_numpy()
    x = df["x_m"].to_numpy(np.float64)
    y = df["y_m"].to_numpy(np.float64)

    # segments: a new one at each track start or team change (rows are sorted by track, frame)
    new = np.r_[True, (tid[1:] != tid[:-1]) | (team[1:] != team[:-1])]
    seg = np.cumsum(new) - 1
    s_first = np.flatnonzero(new)
    s_last = np.r_[s_first[1:] - 1, len(fr) - 1]
    s_start, s_end, s_team = fr[s_first], fr[s_last], team[s_first]

    seg_slot = np.empty(len(s_first), np.int64)
    slot_team, slot_free_at = [], []          # free_at = last frame used + 1
    for s in np.argsort(s_start, kind="stable"):
        cands = [k for k in range(len(slot_team))
                 if slot_team[k] == s_team[s] and slot_free_at[k] < s_start[s]]
        if cands:
            k = min(cands, key=lambda c: slot_free_at[c])
        else:
            k = len(slot_team)
            slot_team.append(int(s_team[s])); slot_free_at.append(-1)
        seg_slot[s] = k
        slot_free_at[k] = s_end[s] + 1

    n_slots = len(slot_team)
    px = np.full((n_frames, n_slots), np.nan)
    py = np.full((n_frames, n_slots), np.nan)
    col = seg_slot[seg]
    px[fr, col] = x
    py[fr, col] = y
    team_arr = np.array(slot_team, np.int8)
    per_frame = {side: np.isfinite(px[:, team_arr == side]).sum(1) for side in (HOME, AWAY)}
    seen = (per_frame[HOME] + per_frame[AWAY]) > 0
    stats = {"team_mode": team_mode, "rows": int(len(fr)), "segments": int(len(s_first)),
             "slots": n_slots, "slots_home": int((team_arr == HOME).sum()),
             "slots_away": int((team_arr == AWAY).sum()),
             "overflow_pct": float(100 * np.mean(((per_frame[HOME] > 11) | (per_frame[AWAY] > 11))[seen])),
             "players_per_frame": float((per_frame[HOME] + per_frame[AWAY])[seen].mean())}
    return px, py, team_arr, stats


def overflow_pct(px, team):
    """% of frames with anyone tracked on which a team has more than 11 players."""
    ok = np.isfinite(px)
    h, a = ok[:, team == HOME].sum(1), ok[:, team == AWAY].sum(1)
    seen = (h + a) > 0
    return float(100 * np.mean(((h > 11) | (a > 11))[seen])) if seen.any() else float("nan")


def clean_slots(px, py, team):
    """fix_players with gap filling off (see the module docstring). Returns px, py, stats."""
    from possession.preprocess.fix_players import FixConfig, fix_players
    cfg = FixConfig(max_interp=0, max_bridge=0)
    px2, py2, _, rep = fix_players(px, py, team, yolo=None, cfg=cfg)
    return px2, py2, {"clean_jumps_cut": rep["jumps_cut"],
                      "clean_duplicate_runs": rep["duplicate_runs_removed"],
                      "clean_frames_blanked": rep["frames_blanked"],
                      "clean_jumps_per_1000_before": rep["jumps_per_1000_before"],
                      "clean_jumps_per_1000_after": rep["jumps_per_1000_after"],
                      "overflow_pct_after_clean": overflow_pct(px2, team)}


def load_detected(path=None, n_frames=199319, team_mode="local", window=12, clean=True, cache=True,
                  refresh=False):
    """Detected players as {'px', 'py', 'team', 'stats'}; cached in outputs/cache/ (``refresh``
    rebuilds and overwrites the cache)."""
    path = path or os.path.join(ROOT, DEFAULT_FILE)
    tag = "{}_{}_{}_{}{}".format(os.path.basename(path).rsplit(".", 1)[0], team_mode, window,
                                 int(os.path.getmtime(path)), "_clean" if clean else "")
    cpath = os.path.join(CACHE_DIR, "detected_{}.npz".format(tag))
    if cache and not refresh and os.path.exists(cpath):
        z = np.load(cpath, allow_pickle=False)
        stats = {k[6:]: z[k].item() for k in z.files if k.startswith("stats_")}
        return {"px": z["px"], "py": z["py"], "team": z["team"], "stats": stats, "cached": cpath}
    px, py, team, stats = build_slots(_read(path), n_frames, team_mode, window)
    stats["clean"] = bool(clean)
    if clean:
        px, py, cstats = clean_slots(px, py, team)
        stats.update(cstats)
    if cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.savez_compressed(cpath, px=px, py=py, team=team,
                            **{"stats_" + k: np.array(v) for k, v in stats.items()})
    return {"px": px, "py": py, "team": team, "stats": stats, "cached": None}


def read_rows(path=None, lo=0, hi=None, team_mode="local", window=12):
    """Long rows of frames lo..hi with the team rule of build_slots applied (column ``tm``:
    0 home, 1 away, NaN none), referees and goalkeepers included. For drawing boxes on the video."""
    import pandas as pd
    path = path or os.path.join(ROOT, DEFAULT_FILE)
    filters = [("frame", ">=", lo - window)] + ([("frame", "<=", hi + window)] if hi is not None else [])
    rows = pd.read_parquet(path, filters=filters,
                           columns=["frame", "half", "track_id", "role", "team", "conf", "x1", "y1",
                                    "x2", "y2", "x_m", "y_m", "team_p_home"])
    rows["vote"] = rows["team"].map({"H": HOME, "A": AWAY}).astype("float64")
    if team_mode == "local":
        rows, p = _local_team(rows, window)
        tm = np.where(np.isfinite(p), np.where(p >= 0.5, HOME, AWAY), rows["vote"].to_numpy())
    else:
        tm = rows["vote"].to_numpy()
    rows = rows.assign(tm=tm)
    rows.loc[rows.role == "referee", "tm"] = np.nan
    keep = (rows.frame >= lo) & ((rows.frame <= hi) if hi is not None else True)
    return rows[keep].sort_values(["frame", "track_id"]).reset_index(drop=True)
