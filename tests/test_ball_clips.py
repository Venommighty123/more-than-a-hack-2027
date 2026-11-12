"""viz/test_ball_clips.py: frame states and the clip plan, on a synthetic ball (no match data)."""
import numpy as np

from viz.test_ball_clips import FILLED, MISSING, REMOVED, TRACKED, frame_states, plan_clips


def _res(n=2000):
    x = np.full(n, 50.0)
    return dict(x=x, y=np.full(n, 30.0), observed=np.ones(n, bool),
                removed=np.zeros(n, bool), filled=np.zeros(n, bool))


def test_states_priority():
    r = _res(10)
    r["x"][2] = np.nan
    r["filled"][4] = True
    r["removed"][5] = True
    r["filled"][5] = True                      # removed and refilled: shown as removed
    st = frame_states(r)
    assert list(st[:6]) == [TRACKED, TRACKED, MISSING, TRACKED, FILLED, REMOVED]


def test_short_gap_one_clip_long_gap_two():
    r = _res()
    r["x"][500:540] = np.nan                   # 40 frames <= 2 x 75: one clip, gap +- 75
    r["x"][1000:1400] = np.nan                 # 400 frames: lost + found clips
    clips = plan_clips(r, [(1, 100, 1899)], pad=75, types=("not_tracked",))
    assert [c["part"] for c in clips] == ["whole", "lost", "found"]
    whole, lost, found = clips
    assert (whole["clip_lo"], whole["clip_hi"]) == (425, 614)
    assert (lost["clip_lo"], lost["clip_hi"]) == (925, 1075)
    assert (found["clip_lo"], found["clip_hi"]) == (1325, 1475)
    assert plan_clips(r, [(1, 100, 1899)], pad=75, min_gap=50, types=("not_tracked",))[0]["part"] == "lost"


def test_gap_at_half_start_is_censored_and_clamped():
    r = _res()
    r["x"][100:400] = np.nan
    clips = plan_clips(r, [(1, 100, 1899)], pad=75, types=("not_tracked",))
    assert [c["part"] for c in clips] == ["found"] and clips[0]["censored"] == "start"
    r = _res()
    r["x"][100:110] = np.nan
    c = plan_clips(r, [(1, 100, 1899)], pad=75, types=("not_tracked",))[0]
    assert c["clip_lo"] == 100                 # never before the graded half


def test_corrections_merge_when_windows_overlap():
    r = _res()
    r["removed"][600] = True
    r["filled"][650:653] = True                # 50 frames later: same clip
    r["removed"][1200] = True                  # far away: its own clip
    clips = plan_clips(r, [(1, 100, 1899)], pad=75, types=("corrected",))
    assert len(clips) == 2
    assert clips[0]["removed"] == 1 and clips[0]["filled"] == 3 and clips[0]["n_corrections"] == 2
    assert (clips[0]["clip_lo"], clips[0]["clip_hi"]) == (525, 727)


def test_outside_graded_halves_ignored():
    r = _res()
    r["x"][10:60] = np.nan
    r["removed"][30] = True
    assert plan_clips(r, [(1, 100, 1899)], pad=75) == []


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
