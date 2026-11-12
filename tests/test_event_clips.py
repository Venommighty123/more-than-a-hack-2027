"""viz/test_event_clips.py: folders, selection and clip names on synthetic events (no match data)."""
from viz.test_event_clips import clip_name, group_of, select


def ev(i, t="pass", sub=None, conf=0.5, team="H", **details):
    return dict(id="E{:05d}".format(i), type=t, subtype=sub, frame_start=1000 * i, frame_end=1000 * i + 50,
                team=team, confidence=conf, details=details)


def test_groups_split_restarts_and_fouls_only():
    assert group_of(ev(1)) == "pass"
    assert group_of(ev(1, "restart", "throw_in")).replace("\\", "/") == "restart/throw_in"
    assert group_of(ev(1, "restart", None)).replace("\\", "/") == "restart/unknown"
    assert group_of(ev(1, "foul_candidate", "penalty_candidate")).replace("\\", "/") == \
        "foul_candidate/penalty_candidate"


def test_spread_covers_first_and_last():
    evs = [ev(i) for i in range(1, 21)]
    got = select(evs, per_type=3, pick="spread")["pass"]
    assert [e["id"] for e in got] == ["E00001", "E00011", "E00020"]
    assert len(select(evs, per_type=0)["pass"]) == 20          # 0 = all
    assert len(select(evs[:2], per_type=8)["pass"]) == 2         # fewer events than asked


def test_confidence_picks_and_time_order():
    evs = [ev(1, conf=0.9), ev(2, conf=0.1), ev(3, conf=0.5), ev(4, conf=0.2)]
    low = select(evs, per_type=2, pick="low_conf")["pass"]
    assert [e["id"] for e in low] == ["E00002", "E00004"]        # lowest two, in time order
    high = select(evs, per_type=1, pick="high_conf")["pass"]
    assert high[0]["id"] == "E00001"
    r1 = select(evs, per_type=2, pick="random", seed=3)["pass"]
    assert r1 == select(evs, per_type=2, pick="random", seed=3)["pass"]


def test_clip_name():
    assert clip_name(ev(7, outcome="complete")) == "0007000_E00007_H_complete.mp4"
    assert clip_name(ev(7, team=None)) == "0007000_E00007_-.mp4"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
