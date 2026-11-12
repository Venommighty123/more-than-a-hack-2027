"""Kinematic flight detection on synthetic tracks (no match data)."""
import numpy as np

from possession.features.flight import FlightConfig, flight_mask


def _pass(fps=25):
    """Ball at a player's feet 2 s, kicked at 20 m/s for 1 s, stops at a receiver, 2 s more."""
    x = np.r_[np.full(50, 20.0), 20.0 + 20.0 * np.arange(1, 26) / fps, np.full(50, 40.0)]
    return x, np.full(len(x), 34.0)


def test_one_pass_is_one_flight_starting_at_the_kick():
    x, y = _pass()
    flights, mask = flight_mask(x, y, FlightConfig())
    assert len(flights) == 1
    s, e = flights[0]
    assert 46 <= s <= 51 and 70 <= e <= 80
    assert mask[s:e + 1].all() and not mask[:s].any()


def test_redetection_jump_is_not_a_kick():
    y = np.full(150, 34.0)
    x = np.r_[np.full(75, 20.0), np.full(75, 60.0)]   # tracker jumps 40 m in one frame
    flights, _ = flight_mask(x, y, FlightConfig())
    assert flights == []


def test_reception_exit_with_players():
    x, y = _pass()
    x = x.copy()
    x[75:] = 40.0 + 6.0 * np.arange(len(x) - 75) / 25          # receiver dribbles on at 6 m/s
    px = np.c_[np.full(len(x), 20.0), x - 0.5]                  # kicker, receiver next to ball
    py = np.full_like(px, 34.0)
    px[:70, 1] = 60.0                                           # receiver away before the ball arrives
    flights, _ = flight_mask(x, y, FlightConfig(), px, py)
    assert len(flights) == 1 and flights[0][1] <= 80


def test_max_flight_duration():
    x = 10.0 + 13.0 * np.arange(300) / 25                         # 12 s at 13 m/s: not a flight
    flights, _ = flight_mask(x, np.full(300, 34.0), FlightConfig(max_flight_s=4.0))
    assert all((e - s + 1) / 25 <= 4.1 for s, e in flights)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
