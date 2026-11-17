"""Conservative ball-gap filling on synthetic tracks (no match data)."""
import numpy as np

from possession.preprocess.ball_fill import (FILL_INTERP, FILL_POINT, FillConfig, chain_gap, fill_gaps, find_gaps,
                                             hide_and_refill)

FPS = 25


def _track(n=400, speed=8.0):
    """Ball rolling along x at `speed` m/s from (20, 30)."""
    t = np.arange(n) / FPS
    return 20.0 + speed * t, np.full(n, 30.0)


def test_find_gaps_inside_blocks_only():
    present = np.ones(100, bool); present[10:20] = False; present[50:60] = False; present[95:] = False
    assert find_gaps(present, [(1, 0, 40), (2, 45, 99)]) == [(1, 10, 20), (2, 50, 60), (2, 95, 100)]


def test_open_play_gap_filled_from_candidates_and_short_holes():
    x, y = _track()
    bx, by = x.copy(), y.copy(); bx[100:150] = np.nan; by[100:150] = np.nan
    cand = np.zeros(len(x), bool); cand[100:150] = True; cand[120:124] = False   # a 4-frame hole
    cx, cy = x + 0.3, y.copy()
    X, Y, code, rows = fill_gaps(bx, by, cx, cy, cand, [(1, 100, 150)], {}, {}, FillConfig())
    assert rows[0]["decision"] == "filled" and rows[0]["scope"] == "open_play"
    assert (code[100:150] > 0).all() and (code[120:124] == FILL_INTERP).all() and code[100] == FILL_POINT
    assert np.nanmax(np.abs(X[100:150] - x[100:150])) < 0.5


def test_outlier_dropped_and_far_chain_rejected():
    x, y = _track()
    bx, by = x.copy(), y.copy(); bx[100:150] = np.nan; by[100:150] = np.nan
    cand = np.zeros(len(x), bool); cand[100:150] = True
    cx, cy = x.copy(), y.copy()
    cx[130] += 30.0                                   # one wrong detection 30 m away
    ok, kept, _ = chain_gap(bx, by, cx, cy, cand, 100, 150, FillConfig())
    assert ok and 130 not in kept and len(kept) == 49
    cx2 = x + 25.0                                    # the whole run tracks something 25 m away
    ok2, _, reason = chain_gap(bx, by, cx2, cy, cand, 100, 150, FillConfig())
    assert not ok2 and "attached to an anchor" in reason


def test_no_anchor_no_fill():
    x, y = _track()
    bx, by = x.copy(), y.copy(); bx[:50] = np.nan
    cand = np.ones(len(x), bool)
    ok, _, reason = chain_gap(bx, by, x, y, cand, 0, 50, FillConfig())
    assert not ok and "anchor" in reason


def test_dead_stretch_needs_live_evidence_and_gesture_vetoes():
    x, y = _track()
    bx, by = x.copy(), y.copy()
    for a, b in ((50, 100), (150, 200), (250, 300)):
        bx[a:b] = np.nan; by[a:b] = np.nan
    cand = np.ones(len(x), bool)
    cx, cy = x.copy(), y.copy()
    cx[150:200] = bx[149]; cy[150:200] = by[149]        # gap 2: ball does not move (placed for a restart)
    bx[200] = bx[149]; by[200] = by[149]                # anchor after = same spot
    gaps = [(1, 50, 100), (1, 150, 200), (1, 250, 300)]
    X, Y, code, rows = fill_gaps(bx, by, cx, cy, cand, gaps, {0: [7], 1: [8], 2: [9]}, {2: True}, FillConfig())
    assert rows[0]["decision"] == "filled" and rows[0]["reason"] == "dead stretch with live evidence"
    assert rows[1]["decision"] == "skipped"
    assert rows[1]["decision"] == "skipped" and "without live evidence" in rows[1]["reason"]
    assert rows[2]["decision"] == "skipped" and rows[2]["reason"] == "referee gesture: dead"
    assert (code[150:200] == 0).all() and (code[250:300] == 0).all()


def test_hide_and_refill_measures_accuracy():
    x, y = _track(n=2000, speed=5.0)
    cx, cy = x + 0.4, y.copy()
    cand = np.ones(len(x), bool)
    rep = hide_and_refill(x, y, cx, cy, cand, np.ones(len(x), bool), [20, 50, 100], FillConfig(), n=50)
    assert rep["windows"] == 50 and rep["within_2m"] == 1.0 and rep["fill_rate"] == 1.0
    bad = cx + 20.0
    rep2 = hide_and_refill(x, y, bad, cy, cand, np.ones(len(x), bool), [20, 50, 100], FillConfig(), n=50)
    assert rep2["accepted_windows"] == 0 and rep2["fill_rate"] == 0.0


def test_floating_run_in_a_long_hole_is_not_trusted():
    x, y = _track(n=600)
    bx, by = x.copy(), y.copy(); bx[100:300] = np.nan; by[100:300] = np.nan
    cand = np.zeros(len(x), bool)
    cand[100:130] = True; cand[180:220] = True; cand[270:300] = True   # attached / floating / attached
    ok, kept, reason = chain_gap(bx, by, x, y, cand, 100, 300, FillConfig())
    assert ok and reason == "partial"
    assert set(range(100, 130)) <= set(kept) and set(range(270, 300)) <= set(kept)
    assert not set(range(180, 220)) & set(kept)
    X, Y, code, rows = fill_gaps(bx, by, x, y, cand, [(1, 100, 300)], {}, {}, FillConfig())
    assert rows[0]["decision"] == "partial" and rows[0]["filled_frames"] == 60
