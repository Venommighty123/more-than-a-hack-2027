"""Tests for possession/preprocess/ball_yolo.py on a synthetic camera and a synthetic lob.

    pytest tests/test_ball_yolo.py      or      python -m tests.test_ball_yolo
"""
import numpy as np

from possession.features.flight import FlightConfig
from possession.preprocess.ball_yolo import SRC_FLIGHT, clean, fix_ball, BallFixConfig

L, W, F, IMG = 105.3, 68.0, 1800.0, (1920, 1080)


def camera():
    C = np.array([52.6, -40.0, 30.0])
    fwd = np.array([52.6, 34.0, 0.0]) - C
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 0, 1.0]); right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.stack([right, down, fwd])
    K = np.array([[F, 0, IMG[0] / 2], [0, F, IMG[1] / 2], [0, 0, 1]])
    P = K @ np.column_stack([R, -R @ C])
    Hn = P[:, [0, 1, 3]] @ np.diag([L, W, 1.0])
    return P, Hn


def scene(n=400):
    """Ground roll, a lob from frame 100 (peak 7.34 m), ground roll again."""
    xyz = np.zeros((n, 3))
    t = np.arange(n) / 25.0
    xyz[:100, 0] = 20 + 2.0 * t[:100]
    xyz[:100, 1] = 30.0
    p0, v0 = np.array([xyz[99, 0], 30.0, 0.0]), np.array([15.0, 5.0, 12.0])
    T = 2 * v0[2] / 9.81
    k = 100
    while k < n and (k - 100) / 25.0 <= T:
        tt = (k - 100) / 25.0
        xyz[k] = p0 + v0 * tt + np.array([0, 0, -0.5 * 9.81 * tt * tt])
        k += 1
    land = xyz[k - 1].copy()
    for j in range(k, n):
        xyz[j] = land + np.array([3.0, 1.0, 0.0]) * (j - k + 1) / 25.0
    xyz[:, 2] = np.maximum(xyz[:, 2], 0.0)
    return xyz, (100, k - 1)


def observe(xyz, P, Hn):
    q = (P @ np.c_[xyz, np.ones(len(xyz))].T).T
    u, v = q[:, 0] / q[:, 2], q[:, 1] / q[:, 2]
    g = np.linalg.inv(Hn) @ np.stack([u, v, np.ones_like(u)])
    x, y = g[0] / g[2] * L, g[1] / g[2] * W
    n = len(u)
    return {"u": u, "v": v, "x": x, "y": y, "conf": np.full(n, 0.8), "jump": np.zeros(n, bool),
            "ran": np.ones(n, bool)}


def test_lob_gets_height_and_true_ground_position():
    P, Hn = camera()
    xyz, (s, e) = scene()
    raw = observe(xyz, P, Hn)
    n = len(xyz)
    Hs = np.repeat(Hn[None], n, axis=0)
    out = fix_ball(raw, Hs, np.ones(n, bool), FlightConfig(), verbose=False)
    air = [f for f in out["flights"] if f["is_airborne"]]
    assert len(air) >= 1
    assert abs(max(f["max_height_m"] for f in air) - 12.0 ** 2 / (2 * 9.81)) < 1.5
    mid = slice(s + 15, e - 15)
    naive_err = np.hypot(raw["x"][mid] - xyz[mid, 0], raw["y"][mid] - xyz[mid, 1])
    fixed_err = np.hypot(out["x"][mid] - xyz[mid, 0], out["y"][mid] - xyz[mid, 1])
    assert np.median(naive_err) > 5.0            # the naive projection lands far behind
    assert np.median(fixed_err) < 1.5            # the fit puts the ball back over its ground point
    assert (out["source"][mid] == SRC_FLIGHT).all()
    assert np.allclose(out["x"][:80], xyz[:80, 0], atol=0.05)    # the ground roll is untouched


def test_lob_with_pixel_noise():
    P, Hn = camera()
    xyz, (s, e) = scene()
    raw = observe(xyz, P, Hn)
    rng = np.random.default_rng(0)
    raw["u"] = raw["u"] + rng.normal(0, 1.5, len(xyz))
    raw["v"] = raw["v"] + rng.normal(0, 1.5, len(xyz))
    n = len(xyz)
    out = fix_ball(raw, np.repeat(Hn[None], n, axis=0), np.ones(n, bool), FlightConfig(), verbose=False)
    air = [f for f in out["flights"] if f["is_airborne"]]
    assert len(air) == 1 and abs(air[0]["max_height_m"] - 7.34) < 1.0
    err = np.hypot(out["x"][s:e + 1] - xyz[s:e + 1, 0], out["y"][s:e + 1] - xyz[s:e + 1, 1])
    assert np.median(err) < 1.0


def test_clean_drops_spikes_and_isolated_picks():
    P, Hn = camera()
    xyz, _ = scene(200)
    raw = observe(xyz, P, Hn)
    raw["u"][50] += 400.0                         # one-frame false blob
    raw["u"][150:200] = np.nan; raw["v"][150:200] = np.nan
    raw["u"][170] = 900.0; raw["v"][170] = 600.0  # isolated pick in a long gap
    Hs = np.repeat(Hn[None], 200, axis=0)
    uv, ground, observed, removed, bridged = clean(raw, Hs, BallFixConfig(), FlightConfig())
    assert removed[50] and removed[170]
    assert bridged[50] and np.isfinite(uv[50]).all()             # the hole is bridged back
    assert not observed[170] and np.isnan(uv[170]).all()


if __name__ == "__main__":
    test_lob_gets_height_and_true_ground_position()
    test_lob_with_pixel_noise()
    test_clean_drops_spikes_and_isolated_picks()
    print("ok")
