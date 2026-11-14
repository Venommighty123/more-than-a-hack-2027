"""Tests for the YOLO player reader (possession/io/detections.py). Synthetic tracks only, no match data.

    pytest tests/test_yolo_io.py      or      python -m tests.test_yolo_io
"""
import os
import tempfile

import numpy as np
import pandas as pd

from possession.io import AWAY, HOME
from possession.io.detections import build_slots, clean_slots, overflow_pct, read_rows

N = 120


def _track(tid, team, frames, x0, y0=30.0, p_home=None, role="player", step=0.2):
    f = np.asarray(list(frames))
    p = (0.9 if team == "H" else 0.1) if p_home is None else p_home
    return pd.DataFrame({"frame": f, "half": 1, "track_id": tid, "role": role, "team": team,
                         "conf": 0.9, "x1": 0.0, "y1": 0.0, "x2": 10.0, "y2": 30.0,
                         "x_m": x0 + step * (f - f[0]), "y_m": np.full(len(f), y0),
                         "team_p_home": np.where(f % 3 == 0, p, np.nan)})


def _tracks():
    return pd.concat([
        _track(1, "H", range(0, 50), 10.0),             # slot A, frees at frame 50
        _track(2, "A", range(0, 100), 60.0),
        _track(3, "H", range(51, 100), 20.0),           # starts after a gap -> may reuse slot A
        _track(4, "H", range(50, 100), 30.0),           # starts right after 1 ends -> must NOT reuse it
        _track(9, "", range(0, 100), 50.0, role="referee"),
    ], ignore_index=True)


def test_shapes_teams_and_referees():
    px, py, team, st = build_slots(_tracks(), N, team_mode="track")
    assert px.shape == py.shape == (N, len(team))
    assert set(team.tolist()) <= {HOME, AWAY}
    assert st["slots_home"] == 2 and st["slots_away"] == 1          # referee dropped, slot reused
    assert np.isnan(px[100:]).all()                                  # nothing after the last frame
    assert int(np.isfinite(px[0]).sum()) == 2                         # tracks 1 and 2 at frame 0


def test_no_slot_changes_person_between_consecutive_frames():
    px, _, _, _ = build_slots(_tracks(), N, team_mode="track")
    step = np.abs(np.diff(px, axis=0))
    assert np.nanmax(step) < 1.0                                      # tracks move 0.2 m per frame


def test_local_team_follows_an_id_switch():
    flip = pd.concat([_track(5, "H", range(0, 40), 10.0, p_home=0.9),
                      _track(5, "H", range(40, 80), 10.0 + 0.2 * 40, p_home=0.1)], ignore_index=True)
    _, _, team_tr, _ = build_slots(flip, N, team_mode="track")
    _, _, team_lo, st = build_slots(flip, N, team_mode="local")
    assert team_tr.tolist() == [HOME]                                 # the vote gives one team
    assert sorted(team_lo.tolist()) == [HOME, AWAY]                   # local mode splits it
    assert st["segments"] == 2


def test_overflow_and_duplicate_clean_up():
    players = [_track(10 + i, "H", range(0, 60), 5.0 * i, y0=10.0 + i) for i in range(11)]
    dup = _track(30, "H", range(0, 60), 0.3, y0=10.2)                 # player 0 tracked twice
    px, py, team, _ = build_slots(pd.concat(players + [dup], ignore_index=True), 60, team_mode="track")
    assert overflow_pct(px, team) == 100.0                            # 12 home slots everywhere
    px2, _, st = clean_slots(px, py, team)
    assert st["clean_duplicate_runs"] >= 1
    assert overflow_pct(px2, team) == 0.0                             # the copy is gone: 11


def test_read_rows_window():
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        return                                                        # optional dependency
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "tracks.parquet")
        _tracks().to_parquet(path, index=False)
        rows = read_rows(path, 40, 60)
        assert rows.frame.between(40, 60).all()
        assert rows.loc[rows.role == "referee", "tm"].isna().all()
        assert set(rows.loc[rows.role == "player", "tm"].dropna().unique()) <= {HOME, AWAY}


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
