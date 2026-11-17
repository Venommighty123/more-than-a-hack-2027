"""P2 broadcast windows on a synthetic sync file (no match data)."""
import numpy as np

from possession.referee.windows import (PairMap, WindowConfig, build_windows, describe_window, merge_ranges,
                                        summary)

FPS = 25


def _sync():
    # tactical: kickoff h1 at 100 s, h2 at 1000 s (clock 2700); broadcast: h1 at 40 s, h2 at 900 s
    return {"sources": {
        "tactical": {"fps": FPS, "n_frames": 50000, "segments": [
            {"half": 1, "t_first": 101.0, "t_last": 400.0, "offset": 100.0},
            {"half": 2, "t_first": 1001.0, "t_last": 1300.0, "offset": 1000.0 - 2700.0}]},
        "broadcast": {"fps": FPS, "n_frames": 40000, "segments": [
            {"half": 1, "t_first": 41.0, "t_last": 340.0, "offset": 40.0},
            {"half": 2, "t_first": 901.0, "t_last": 1200.0, "offset": 900.0 - 2700.0}]}},
        "pair": {"extra_sec": {"1": 0.2, "2": 0.0}, "atd_orientation": {"1": "identity", "2": "rot180"}}}


def test_mapping_through_the_match_clock():
    pm = PairMap(_sync())
    half, clock, bf = pm.map_frames([200 * FPS, 1100 * FPS])
    assert list(half) == [1, 2]
    assert abs(clock[0] - 99.8) < 1e-9 and abs(clock[1] - 2800.0) < 1e-9     # extra_sec removed in h1
    assert list(bf) == [int(round((99.8 + 40) * FPS)), int(round((2800 - 1800) * FPS))]


def test_windows_and_view_verdicts():
    cfg = WindowConfig(atd_step=5)
    rows = [{"n": 1, "d": 1, "kind": "stoppage", "direction": "H->D", "frame": 200 * FPS},
            {"n": 2, "d": 1, "kind": "restart", "direction": "D->A", "frame": 220 * FPS},
            {"n": 3, "d": 2, "kind": "stoppage", "direction": "A->D", "frame": 300 * FPS}]
    ws = build_windows(rows, cfg, FPS)
    assert [w["tactical_last"] - w["tactical_first"] + 1 for w in ws] == [201, 176, 201]
    pm = PairMap(_sync())
    labels = np.ones(40000, dtype=int)                       # all LIVE_WIDE ...
    _, _, bf3 = pm.map_frames([300 * FPS])
    labels[bf3[0] - 100: bf3[0] + 300] = 0                    # ... except around window 3: off the match
    rng = np.random.default_rng(0)
    players = {f: rng.uniform(0, 1, (12, 2)) for f in range(0, 50000, 1)}
    atd_t = players
    # broadcast ATD: window 1 matches the tactical moment, window 2 is a replay (other positions)
    atd_b = {}
    for w, same in ((ws[0], True), (ws[1], False)):
        tf = np.arange(w["tactical_first"], w["tactical_last"] + 1)
        _, _, b = pm.map_frames(tf)
        for t, f in zip(tf, b):
            atd_b[int(f)] = players[int(t)] + 0.001 if same else rng.uniform(0, 1, (12, 2))
    hv = np.ones(40000, dtype=bool)
    used = [describe_window(w, pm, labels, hv, atd_b, atd_t, cfg) for w in ws]
    assert ws[0]["view"] == "good" and ws[0]["atd_live_share"] == 1.0 and ws[0]["usable_share"] == 1.0
    assert ws[1]["view"] == "replay_suspect" and ws[1]["usable_frames"] < 10
    assert ws[2]["view"] == "none" and ws[2]["share_off_match"] == 1.0
    assert ws[0]["anchor_match_clock"] == "1:39.8"
    s = summary(ws, stretches_total=2)
    assert s["stretches_with_good_window"] == 1 and s["stretches_without_view"] == 1
    assert len(merge_ranges(np.concatenate(used))) == 1


def test_merge_ranges():
    assert merge_ranges([5, 1, 2, 3, 9, 10]) == [(1, 3), (5, 5), (9, 10)]
    assert merge_ranges([1, 3, 5], gap=2) == [(1, 5)]
    assert merge_ranges([]) == []
