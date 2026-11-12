"""Noise profile of a tracking feed: label-free statistics of the ball and player tracking.

WHY. The v3 run showed the bench can reward fixes for noise BAR-ATH does not have (false balls: the
bench adds ~10 per minute, BAR-ATH has almost none). Before changing the labeller again we make the
bench's noise match the real feed. This script measures the noise the same way on BAR-ATH and on
every bench copy, so the two can be compared line by line.

RULES. Every number here comes from the tracking alone (positions, the team flag of each slot, and
the graded frame ranges). No ground truth, no labels, no video, no human data about the match is
read. Only aggregates are printed and saved, never frames or positions.

    python test_scripts/noise_profile.py                     # BAR-ATH, on your PC
    python test_scripts/noise_profile.py --bench bench/      # every bench copy (the GT is not read)

Output: a table on screen and evaluation/results/noise_profile_BAR-ATH.json (or
noise_profile_bench.json). Both are aggregates, safe to share.

WHAT IS MEASURED (over the graded frames from each kickoff)
  ball, availability   share of frames with a ball; gaps (number per 10 min, length percentiles,
                       and how the missing time splits by gap length)
  ball, false tracks   jumps > 2 m and > 5 m in one frame (per 10 min); short tracklets (<= 1.5 s,
                       per 10 min) - a false detection is a short track that jumps in or out
  ball, jitter         when the ball is nearly still (windowed speed < 1 m/s): the frame-to-frame
                       wobble around its 9-frame mean (median, p90, in m); the same on all
                       tracked frames ("wobble_all"), which also catches jitter on a moving ball
  ball, motion         windowed speed percentiles; share still (< 0.5 m/s); share off the pitch
  ball vs players      distance from the ball to the nearest player (percentiles); share of
                       tracked frames with someone within 1.5 m; share with nobody within 5 m;
                       control episodes per 10 min and switches of the controlling team per 10 min
                       (the v1 control rule, CTRL_R 1.5 / MARGIN 0.3)
  far-ball split        (added in the second version) the same distance stats split by situation:
                       ball still / slow / rolling / flying, players slow or moving, ball off the
                       pitch / near a line / interior, few players tracked. Shows WHERE the ball is
                       far from everyone, which decides whether it matters for possession
  offset check          (third version) on dribbling-pace balls in open play: nearest-player
                       distance with the ball track shifted in time against the players (a clock
                       offset shows as a minimum away from 0), and the ball's offset from that
                       player split into "ahead along his run" and "to the side"
  players              tracked players per team (median / p10); slot tracklets (number, median
                       length, share shorter than 60 s); position jitter; jumps > 2 m in a frame
                       per 10 min; mean player speed (median, m/s); share of frames slower than
                       0.7x and 0.5x the median (the v2 stoppage signal)
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import labeller_v1 as v1  # noqa: E402

FPS = v1.FPS
L, W = v1.PITCH


def _runs(mask):
    return v1._runs(mask)


def _pct(x, qs=(10, 50, 90)):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return [None] * len(qs)
    return [round(float(v), 3) for v in np.percentile(x, qs)]


def _wobble(x, y, still):
    """Distance of each position from its centred 9-frame mean, on 'still' frames."""
    k = np.ones(9) / 9
    out = []
    for s, e in zip(*_runs(still)):
        if e - s < 9:
            continue
        mx = np.convolve(x[s:e], k, mode="valid")
        my = np.convolve(y[s:e], k, mode="valid")
        out.append(np.hypot(x[s + 4:e - 4] - mx, y[s + 4:e - 4] - my))
    return np.concatenate(out) if out else np.array([])


def profile(px, py, ball, team, blocks):
    team = np.asarray(team)
    n = len(ball)
    graded = np.zeros(n, dtype=bool)
    for _, a, b in blocks:
        graded[int(a):int(b) + 1] = True
    mins10 = graded.sum() / FPS / 600.0
    r = {"graded_minutes": round(graded.sum() / FPS / 60.0, 1)}

    # ---- ball availability ---------------------------------------------------------------------
    raw = ball.astype(float)
    seen = ~np.isnan(raw[:, 0])
    r["ball_tracked_share"] = round(float(seen[graded].mean()), 3)
    miss = ~seen & graded
    gl = np.array([e - s for s, e in zip(*_runs(miss))]) / FPS
    r["ball_gaps_per10min"] = round(len(gl) / mins10, 1)
    r["ball_gap_len_s_p10_p50_p90"] = _pct(gl)
    r["ball_gap_len_s_p99_max"] = _pct(gl, (99, 100))
    bins = [(0, 1), (1, 5), (5, 15), (15, 60), (60, 1e9)]
    tot = max(gl.sum(), 1e-9)
    r["ball_missing_time_by_gap_len"] = {"{}-{}s".format(a, b if b < 1e9 else "inf"):
                                         round(float(gl[(gl >= a) & (gl < b)].sum() / tot), 3)
                                         for a, b in bins}

    # ---- false tracks ----------------------------------------------------------------------------
    both = seen[1:] & seen[:-1] & graded[1:]
    step = np.full(n, np.nan)
    step[1:][both] = np.hypot(*(raw[1:][both] - raw[:-1][both]).T)
    r["ball_jumps_gt2m_per10min"] = round(float((step > 2).sum() / mins10), 1)
    r["ball_jumps_gt5m_per10min"] = round(float((step > 5).sum() / mins10), 1)
    brk = ~seen | (step > 5)
    tl = []
    for s, e in zip(*_runs(seen & graded)):
        cuts = [s] + [k for k in range(s + 1, e) if brk[k]] + [e]
        tl += [b - a for a, b in zip(cuts[:-1], cuts[1:])]
    tl = np.array(tl) / FPS
    r["ball_tracklets_per10min"] = round(len(tl) / mins10, 1)
    r["ball_short_tracklets_le1.5s_per10min"] = round(float((tl <= 1.5).sum() / mins10), 1)
    r["ball_tracklet_len_s_p10_p50_p90"] = _pct(tl)

    # ---- ball jitter and motion ------------------------------------------------------------------
    v = v1.ball_speed(raw)
    still = seen & graded & (v < 1.0)
    r["ball_jitter_m_p50_p90"] = _pct(_wobble(raw[:, 0], raw[:, 1], still), (50, 90))
    r["ball_wobble_all_m_p50_p90"] = _pct(_wobble(raw[:, 0], raw[:, 1], seen & graded), (50, 90))
    vs = v[seen & graded]
    r["ball_speed_ms_p10_p50_p90_p99"] = _pct(vs, (10, 50, 90, 99))
    r["ball_still_share_lt0.5ms"] = round(float(np.mean(vs[np.isfinite(vs)] < 0.5)), 3)
    out = seen & ((raw[:, 0] < -0.5) | (raw[:, 0] > L + 0.5) | (raw[:, 1] < -0.5) | (raw[:, 1] > W + 0.5))
    r["ball_off_pitch_share_of_tracked"] = round(float(out[graded].sum() / max(1, (seen & graded).sum())), 3)

    # ---- ball vs players -------------------------------------------------------------------------
    ok = seen & graded
    d = np.hypot(px[ok] - raw[ok, 0:1], py[ok] - raw[ok, 1:2])
    d = np.where(np.isnan(d), np.inf, d)
    dn = d.min(1)
    r["nearest_player_m_p10_p50_p90"] = _pct(dn[np.isfinite(dn)])
    r["share_someone_within_1.5m"] = round(float(np.mean(dn < 1.5)), 3)
    r["share_nobody_within_5m"] = round(float(np.mean(dn > 5)), 3)
    b = v1.clean_ball(raw, v1.DEFAULTS)
    ctrl = v1.control(px, py, b, team, v1.DEFAULTS)
    ctrl[~graded] = -1
    r["controlled_share_of_tracked"] = round(float((ctrl >= 0).sum() / max(1, ok.sum())), 3)
    eps = []
    for t in (0, 1):
        for s, e in zip(*_runs(ctrl == t)):
            eps.append((s, t))
    eps.sort()
    r["control_episodes_per10min"] = round(len(eps) / mins10, 1)
    sw = sum(1 for a, b2 in zip(eps[:-1], eps[1:]) if a[1] != b2[1])
    r["control_team_switches_per10min"] = round(sw / mins10, 1)

    # ---- WHERE the ball is far from everyone (v2 of this script) ----------------------------------
    # Split the tracked frames by situation, using tracking only, and give for each: its share of
    # the tracked frames, the median distance to the nearest player, and the share with someone
    # within 1.5 m. This tells us whether the ball is far from players during play (a problem for
    # possession) or only while it flies, lies still, or sits off the pitch (harmless).
    sp_all = v1.player_speed(px, py)
    kk = int(3.0 * FPS)
    rel_all = np.convolve(sp_all, np.ones(kk) / kk, mode="same")
    rel_all = rel_all / max(float(np.median(rel_all[graded])), 1e-6)
    vv = v[ok]
    nl = (raw[ok, 0] < 3) | (raw[ok, 0] > L - 3) | (raw[ok, 1] < 3) | (raw[ok, 1] > W - 3)
    cnt = (~np.isnan(px[ok])).sum(1)
    groups = {
        "ball_still_lt0.5": vv < 0.5,
        "ball_slow_0.5-3": (vv >= 0.5) & (vv < 3),
        "ball_rolling_3-8": (vv >= 3) & (vv < 8),
        "ball_flying_gt8": vv >= 8,
        "players_slow_rel_lt0.7": rel_all[ok] < 0.7,
        "players_moving_rel_ge0.7": rel_all[ok] >= 0.7,
        "ball_off_pitch": out[ok],
        "ball_within_3m_of_line": nl & ~out[ok],
        "ball_interior": ~nl & ~out[ok],
        "le18_players_tracked": cnt <= 18,
        "moving_players_ball_lt8_interior": (rel_all[ok] >= 0.7) & (vv < 8) & ~nl & ~out[ok],
    }
    r["far_ball_split"] = {name: [round(float(g.mean()), 3),
                                  round(float(np.median(dn[g])), 2) if g.any() else None,
                                  round(float(np.mean(dn[g] < 1.5)), 3) if g.any() else None]
                           for name, g in groups.items()}

    # ---- is the ball offset from the player who has it? (third version) ---------------------------
    # On frames where the ball moves at dribbling pace (0.5-3 m/s) in open play, find how far the
    # ball is from its nearest player when the ball track is shifted in time against the players.
    # A minimum away from shift 0 means the ball and player clocks are out of sync. Then split the
    # offset to the nearest player into "ahead of him along his running direction" and "to the
    # side": a ball that leads is a dribble, a ball to the side or behind is a position error.
    sel = np.flatnonzero(ok)[(vv >= 0.5) & (vv < 3) & ~nl & ~out[ok] & (rel_all[ok] >= 0.7)]
    sel = sel[(sel > 60) & (sel < n - 60)][::3]
    lag = {}
    for k in (-50, -25, -15, -10, -5, -3, 0, 3, 5, 10, 15, 25, 50):
        bb = raw[sel + k]
        good = ~np.isnan(bb[:, 0])
        dd = np.hypot(px[sel[good]] - bb[good, 0:1], py[sel[good]] - bb[good, 1:2])
        dd = np.where(np.isnan(dd), np.inf, dd).min(1)
        lag[str(k)] = [round(float(np.median(dd)), 3), round(float(np.mean(dd < 1.5)), 3)]
    r["slowball_nearest_m_and_in1.5_by_ball_shift_frames"] = lag
    dd = np.hypot(px[sel] - raw[sel, 0:1], py[sel] - raw[sel, 1:2])
    dd = np.where(np.isnan(dd), np.inf, dd)
    j = dd.argmin(1)
    w = 12
    lo_, hi_ = sel - w, sel + w
    vx = (px[hi_, j] - px[lo_, j]) * FPS / (2 * w)
    vy = (py[hi_, j] - py[lo_, j]) * FPS / (2 * w)
    spd = np.hypot(vx, vy)
    m_ = np.isfinite(spd) & (spd > 1.0)
    ux, uy = vx[m_] / spd[m_], vy[m_] / spd[m_]
    ox = raw[sel[m_], 0] - px[sel[m_], j[m_]]
    oy = raw[sel[m_], 1] - py[sel[m_], j[m_]]
    along = ox * ux + oy * uy
    side = np.abs(-ox * uy + oy * ux)
    r["slowball_offset_along_run_m_p25_p50_p75"] = _pct(along, (25, 50, 75))
    r["slowball_offset_side_m_p25_p50_p75"] = _pct(side, (25, 50, 75))
    r["slowball_frames_used"] = int(m_.sum())

    # ---- players ---------------------------------------------------------------------------------
    on = ~np.isnan(px)
    for t, name in ((0, "home"), (1, "away")):
        c = on[graded][:, team == t].sum(1)
        r["players_tracked_{}_p10_p50".format(name)] = _pct(c, (10, 50))
    tls = []
    for j in range(px.shape[1]):
        tls += [e - s for s, e in zip(*_runs(on[:, j] & graded))]
    tls = np.array(tls) / FPS
    r["player_slots"] = int(px.shape[1])
    r["player_tracklets"] = int(len(tls))
    r["player_tracklet_len_s_p10_p50"] = _pct(tls, (10, 50))
    r["player_tracklets_lt60s_share"] = round(float(np.mean(tls < 60)), 3) if len(tls) else None
    pst = np.hypot(np.diff(px, axis=0), np.diff(py, axis=0))
    pst = pst[graded[1:]]
    r["player_jumps_gt2m_per10min"] = round(float(np.nansum(pst > 2) / mins10), 1)
    sp = v1.player_speed(px, py)
    sp1 = np.full(px.shape, np.nan, dtype=np.float32)
    w = 12
    sp1[w:-w] = np.hypot(px[2 * w:] - px[:-2 * w], py[2 * w:] - py[:-2 * w]) * FPS / (2 * w)
    wob = []
    for j in range(0, px.shape[1]):
        stl = on[:, j] & graded & (sp1[:, j] < 0.5)
        x = _wobble(px[:, j].astype(float), py[:, j].astype(float), stl)
        if len(x):
            wob.append(x[::5])
    r["player_jitter_m_p50_p90"] = _pct(np.concatenate(wob) if wob else [], (50, 90))
    k = int(3.0 * FPS)
    sm = np.convolve(sp, np.ones(k) / k, mode="same")
    med = float(np.median(sm[graded]))
    rel = sm[graded] / max(med, 1e-6)
    r["mean_player_speed_ms_median"] = round(med, 2)
    r["share_rel_lt0.7"] = round(float(np.mean(rel < 0.7)), 3)
    r["share_rel_lt0.5"] = round(float(np.mean(rel < 0.5)), 3)
    return r


def print_table(results):
    names = list(results)
    keys = [k for k in results[names[0]] if k not in ("far_ball_split",
            "slowball_nearest_m_and_in1.5_by_ball_shift_frames")]
    w = max(len(k) for k in keys)
    print("  ".join([" " * w] + ["{:>22}".format(n[:22]) for n in names]))
    for k in keys:
        cells = []
        for nm in names:
            val = results[nm].get(k)
            if isinstance(val, dict):
                val = "/".join("{:.2f}".format(x) for x in val.values())
            elif isinstance(val, list):
                val = "/".join("-" if x is None else "{:g}".format(x) for x in val)
            cells.append("{:>22}".format(str(val)))
        print("  ".join([k.ljust(w)] + cells))
    print("(ball_missing_time_by_gap_len is shown as 0-1 / 1-5 / 5-15 / 15-60 / 60+ s)")
    print()
    print("WHERE THE BALL IS FAR FROM PLAYERS: share of tracked frames / median nearest-player "
          "distance (m) / share with someone within 1.5 m")
    groups = list(results[names[0]]["far_ball_split"])
    w = max(len(g) for g in groups)
    print("  ".join([" " * w] + ["{:>22}".format(n[:22]) for n in names]))
    for g in groups:
        cells = ["{:>22}".format("/".join("-" if x is None else "{:g}".format(x)
                                          for x in results[nm]["far_ball_split"][g])) for nm in names]
        print("  ".join([g.ljust(w)] + cells))
    print()
    print("BALL SHIFTED IN TIME AGAINST THE PLAYERS (dribbling-pace ball, open play): shift in frames "
          "(+ = ball from later) -> median nearest-player distance (m) / share within 1.5 m")
    key = "slowball_nearest_m_and_in1.5_by_ball_shift_frames"
    for k in results[names[0]][key]:
        cells = ["{:>22}".format("/".join("-" if x is None else "{:g}".format(x)
                                          for x in results[nm][key][k])) for nm in names]
        print("  ".join(["shift {:>4}".format(k).ljust(w)] + cells))


def run_match():
    from games import GAME, GAMES, anchor_video_frame, half_video_blocks, require
    from pass_detection import load_tracking, team_of_each_track
    n_frames = GAMES[GAME]["n_video_frames"]
    meta_path, raw_path = require(["atd_meta", "atd_raw"])
    team = team_of_each_track(meta_path)
    print("loading tracking ...")
    px, py, ball = load_tracking(raw_path, len(team), 0, n_frames - 1)
    kickoff2 = anchor_video_frame(GAMES[GAME]["kickoff_2nd"])
    blocks = [(h, max(a, kickoff2) if h == 2 else a, b) for h, a, b in half_video_blocks(GAME)]
    return {GAME: profile(px, py, ball, np.asarray(team), blocks)}, GAME


def run_bench(folder):
    per = {}
    for path in sorted(glob.glob(os.path.join(folder, "g*_*.npz"))):
        g, copy = os.path.basename(path)[:-4].split("_", 1)
        z = np.load(path, allow_pickle=True)          # gt is in the file but NOT read
        per.setdefault(copy, []).append(profile(z["px"], z["py"], z["ball"], z["team"], z["blocks"]))
    out = {}
    for copy, rs in per.items():
        m = {}
        for k in rs[0]:
            vals = [r[k] for r in rs]
            if isinstance(vals[0], dict) and isinstance(next(iter(vals[0].values())), list):
                m[k] = {kk: [None if any(v[kk][i] is None for v in vals) else
                             round(float(np.mean([v[kk][i] for v in vals])), 3)
                             for i in range(len(vals[0][kk]))] for kk in vals[0]}
            elif isinstance(vals[0], dict):
                m[k] = {kk: round(float(np.mean([v[kk] for v in vals])), 3) for kk in vals[0]}
            elif isinstance(vals[0], list):
                m[k] = [None if any(v[i] is None for v in vals) else
                        round(float(np.mean([v[i] for v in vals])), 3) for i in range(len(vals[0]))]
            else:
                m[k] = round(float(np.mean(vals)), 3)
        out["bench_" + copy] = m
    order = ["clean", "degraded", "hard", "tracked", "tracked_all"]
    return {k: out[k] for k in ["bench_" + o for o in order if "bench_" + o in out]}, "bench"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bench", help="bench folder; without it, profile BAR-ATH")
    ap.add_argument("--out", help="JSON path (default evaluation/results/noise_profile_<name>.json)")
    args = ap.parse_args()
    t0 = time.time()
    results, name = run_bench(args.bench) if args.bench else run_match()
    print_table(results)
    out = args.out or os.path.join(ROOT, "evaluation", "results", "noise_profile_{}.json".format(name))
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as handle:
        json.dump(results, handle, indent=1)
    print("wrote {}  ({:.0f} s)".format(os.path.relpath(out, ROOT), time.time() - t0))


if __name__ == "__main__":
    main()
