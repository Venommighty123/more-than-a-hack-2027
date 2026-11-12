"""Labeller v5 building blocks: ZOI combination and the flight mask (no match data)."""
import numpy as np

from possession.detectors.control import AWAY, BOTH, HOME, NONE
from possession.labellers.v5 import combine, kicker_team, mask_flights


def test_zone_never_creates_control():
    d = np.array([NONE, NONE, NONE], dtype=np.int8)
    z = np.array([HOME, AWAY, BOTH], dtype=np.int8)
    for mode in ("off", "contest", "veto", "confirm"):
        assert (combine(d, z, mode) == NONE).all()


def test_veto_and_confirm():
    d = np.array([HOME, HOME, HOME, AWAY, AWAY], dtype=np.int8)
    z = np.array([HOME, NONE, BOTH, HOME, AWAY], dtype=np.int8)
    assert list(combine(d, z, "veto")) == [HOME, HOME, NONE, NONE, AWAY]
    assert list(combine(d, z, "contest")) == [HOME, HOME, NONE, AWAY, AWAY]
    assert list(combine(d, z, "confirm")) == [HOME, NONE, NONE, NONE, AWAY]
    assert list(combine(d, z, "off")) == list(d)


def test_flight_mask_opponent_only():
    ctrl = np.full(30, NONE, dtype=np.int8)
    ctrl[:5] = HOME                      # kicker
    ctrl[12:14] = AWAY                   # ball passes an opponent mid-flight
    ctrl[16:18] = HOME                   # team-mate near the ball
    ctrl[20:] = AWAY                     # interception at the flight's end
    flights = [(4, 20)]
    assert kicker_team(ctrl, 4, 10) == HOME
    out = mask_flights(ctrl, flights, "opponent")
    assert (out[12:14] == NONE).all() and (out[16:18] == HOME).all() and (out[20:] == AWAY).all()
    out = mask_flights(ctrl, flights, "all")
    assert (out[5:20] == NONE).all() and out[4] == HOME and out[20] == AWAY


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
