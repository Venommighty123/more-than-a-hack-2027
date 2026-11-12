"""prepare_ball / reconstruct on synthetic tracks (no match data). Run: pytest tests/ or
python -m tests.test_reconstruct"""
import numpy as np

from possession.preprocess.reconstruct import prepare_ball, reconstruct


def _line(n=100, speed=10.0, fps=25):
    t = np.arange(n) / fps
    return 10 + speed * t, np.full(n, 30.0)


def test_fills_plausible_gap_up_to_max_gap():
    x, y = _line()
    x[40:60] = y[40:60] = np.nan                       # 20 frames, ball at 10 m/s
    fx, fy, obs, rem, filled = prepare_ball(x, y, "hermite", 25)
    assert filled[40:60].all() and not filled[:40].any()
    assert np.nanmax(np.abs(fx[40:60] - (10 + 10 * np.arange(40, 60) / 25))) < 0.05


def test_leaves_long_gap():
    x, y = _line()
    x[30:60] = y[30:60] = np.nan                       # 30 frames > 25
    fx, _, _, _, filled = prepare_ball(x, y, "hermite", 25)
    assert not filled.any() and np.isnan(fx[30:60]).all()


def test_speed_gate_rejects_redetection():
    x, y = _line(speed=0.0)
    x[50:], y[50:] = x[50:] + 30.0, y[50:]             # reappears 30 m away
    x[45:50] = y[45:50] = np.nan                       # 5 frames: would be 150 m/s
    fx, _, _, _, filled = prepare_ball(x, y, "hermite", 25, max_speed_ms=35.0)
    assert not filled.any()
    fx2, _, _, _, filled2 = prepare_ball(x, y, "linear", 25, max_speed_ms=None)
    assert filled2[45:50].all()


def test_linear_6_equals_old_bridge():
    from possession.preprocess.ball import BallConfig, clean_ball
    rng = np.random.default_rng(0)
    x, y = _line(400)
    for s in rng.choice(380, 25, replace=False):
        x[s:s + rng.integers(1, 12)] = np.nan
    y[np.isnan(x)] = np.nan
    t = clean_ball(x, y, BallConfig(max_speed_ms=None, bridge_frames=6))
    fx, fy, *_ = prepare_ball(x, y, "linear", 6, max_speed_ms=None)
    assert np.array_equal(np.isnan(t.x), np.isnan(fx)) and np.allclose(t.x, fx, equal_nan=True)


def test_reconstruct_flags_only_filled():
    x, y = _line()
    x[10:13] = y[10:13] = np.nan
    _, _, filled = reconstruct(x, y, "linear")
    assert filled.sum() == 3


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
