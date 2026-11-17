"""Ball candidate selection on synthetic candidates (no match data)."""
import numpy as np

from possession.preprocess.ball_select import SelectConfig, expected_px, outside_m, static_mask, viterbi_run


def test_outside_m():
    d = outside_m(np.array([50.0, -3.0, 108.3, 50.0]), np.array([30.0, 30.0, 30.0, 72.0]), (105.3, 68.0))
    assert np.allclose(d, [0.0, 3.0, 3.0, 4.0])


def test_static_mask_flags_a_parked_object_not_a_moving_ball():
    cfg = SelectConfig(fps=25, static_s=4.0, static_frac=0.9)
    f = np.arange(200)
    u_ball, v_ball = 100 + 3.0 * f, 200 + 0.5 * f          # moving
    u_spare, v_spare = np.full(200, 900.0), np.full(200, 50.0)  # spare ball by the line
    frame = np.r_[f, f]
    m = static_mask(frame, np.r_[u_ball, u_spare], np.r_[v_ball, v_spare], cfg)
    assert not m[:200].any() and m[200 + 60:200 + 140].all()


def test_expected_px_scales_with_the_homography():
    Hn = np.diag([1053.0, 680.0, 1.0])                        # 10 px per metre
    assert abs(expected_px(Hn, 50.0, 30.0, (105.3, 68.0)) - 2.2) < 1e-6


def _cands(frames, true_xy, decoy_xy, true_conf=0.5, decoy_conf=0.9):
    out = {}
    for i, f in enumerate(frames):
        out[int(f)] = (np.array([true_xy[i][0], decoy_xy[i][0]]), np.array([true_xy[i][1], decoy_xy[i][1]]),
                       np.array([true_conf, decoy_conf]), np.array([True, True]))
    return out


def test_path_follows_the_continuous_ball_not_the_confident_decoy_with_atd_anchors():
    cfg = SelectConfig()
    frames = np.arange(100)
    true = [(30 + 0.3 * t, 34.0) for t in frames]
    decoy = [(80.0 + (5 if t % 2 else -5), 10.0) for t in frames]   # confident but jumping between two spots
    atd = {int(t): true[t] for t in frames if t < 10 or t > 90}      # ATD sees the ball at both ends only
    pick = viterbi_run(frames, _cands(frames, true, decoy), atd, cfg)
    assert sum(pick[int(t)] == 0 for t in frames) >= 95


def test_rejected_candidates_are_never_chosen_and_no_ball_is_allowed():
    cfg = SelectConfig()
    frames = np.arange(30)
    c = {int(t): (np.array([50.0]), np.array([30.0]), np.array([0.9]), np.array([t >= 15])) for t in frames}
    pick = viterbi_run(frames, c, {}, cfg)
    assert all(pick[int(t)] == -1 for t in range(15)) and all(pick[int(t)] == 0 for t in range(16, 30))


def test_impossible_jump_is_not_taken():
    cfg = SelectConfig()
    frames = np.arange(40)
    true = [(20 + 0.2 * t, 30.0) for t in frames]
    far = [(90.0, 60.0) for _ in frames]
    c = _cands(frames, true, far, true_conf=0.6, decoy_conf=0.62)
    pick = viterbi_run(frames, c, {0: true[0]}, cfg)
    chosen = [pick[int(t)] for t in frames]
    assert chosen.count(0) >= 38
