"""Labeller v6 building blocks: smoothing, switch hysteresis, short-stoppage drop (no match data)."""
import numpy as np

from possession.labellers.v6 import DEFAULTS, drop_short_dead, smooth
from possession.model.resolve import ResolveConfig, resolve

H, A, D, NONE = 0, 1, 2, -1


def test_smooth_keeps_missing_frames_missing():
    a = np.array([[0.0], [1.0], [np.nan], [3.0], [4.0]])
    s = smooth(a, 1)
    assert np.isnan(s[2, 0])
    assert np.isclose(s[1, 0], 0.5) and np.isclose(s[3, 0], 3.5)   # NaN neighbour ignored
    assert np.array_equal(smooth(a, 0), a, equal_nan=True)


def test_hysteresis_blocks_weak_takeover():
    ctrl = np.array([H] * 10 + [A] * 8 + [H] * 10, dtype=np.int8)
    dead = np.zeros(len(ctrl), dtype=bool)
    cfg = ResolveConfig(confirm=6, hole=3)
    plain = resolve(ctrl, dead, 0, len(ctrl) - 1, cfg)
    assert (plain[10:18] == A).all()                       # v1 logic switches
    weak = np.where(ctrl == A, NONE, ctrl).astype(np.int8)  # A never clear by SW_MARGIN
    hyst = resolve(ctrl, dead, 0, len(ctrl) - 1, cfg, ctrl_sw=weak)
    assert (hyst == H).all()                               # v6: H keeps the ball
    hyst2 = resolve(ctrl, dead, 0, len(ctrl) - 1, cfg, ctrl_sw=ctrl)
    assert np.array_equal(hyst2, plain)                    # clear take-over still switches


def test_drop_short_dead():
    n = 200
    dead = np.zeros(n, dtype=bool)
    dead[50:70] = True        # 0.8 s at 25 fps -> dropped
    dead[100:180] = True      # 3.2 s -> kept
    b = np.full((n, 2), 50.0)
    p = dict(DEFAULTS)
    out = drop_short_dead(dead, b, p, 0, n - 1)
    assert not out[50:70].any() and out[100:180].all()
    b[55] = (-5.0, 30.0)      # ball off the pitch inside the short stretch
    keep = drop_short_dead(dead, b, dict(p, MIN_D_KEEP_OUT=True), 0, n - 1)
    assert keep[50:70].all()
    assert np.array_equal(drop_short_dead(dead, b, dict(p, MIN_D=0.0), 0, n - 1), dead)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
