"""Post-processing of the YOLO ball (kaggle/kaggle_ball_detection.ipynb): clean it and fix flights.

WHY. YOLO sees the ball where it is ON SCREEN. Through the homography (which maps the GROUND plane
to the image) an airborne ball therefore lands on the pitch where its line of sight hits the grass:
behind the true ball, too far from the camera, and moving too fast. The ATD fixed this for us (its
top view keeps the ball over its true ground point). For our own ball we do it here: every airborne
stretch is fitted with a 3D parabola seen through each frame's own camera, and its ground position
(X, Y) - the point under the ball - replaces the naive projection; the height Z is kept too.

    from possession.preprocess.ball_yolo import load_yolo_ball, fix_ball
    raw = load_yolo_ball("data/kaggle_outputs/ball_yolo_full.parquet", n_frames)
    out = fix_ball(raw, Hn, valid)              # dict: x, y, z, source, in_air, flights, report

STEPS (thresholds = FlightConfig / BallFixConfig defaults: untuned, never fitted on BAR-ATH)
  1. clean     drop picks the notebook flagged as jumps, NaN out spikes (jump away AND back faster
               than a kick), drop detections isolated between long gaps.
  2. bridge    linear fill of gaps <= `bridge` frames in PIXELS (the projection of a straight
               screen segment is a straight ground segment, so ground and pixels stay consistent).
  3. candidates  possession/features/flight.flight_mask on the naive ground track, lightly smoothed
               (moving median, 9 frames) because YOLO's pixel noise grows with depth (m/s, Savitzky-
               Golay derivative, kick walk-back, splits at re-detection jumps, max 4 s); pieces less
               than 3 frames apart are merged (noise splits one flight into several).
  4. fit       per candidate: camera P from each frame's homography (focal length estimated from
               the flight's own homographies: the camera zooms), the OBSERVED pixels, and two models
               - a ground roll (Z = 0) and a parabola (Z free) - with the ends softly pinned to the
               ground track. Airborne when the parabola peaks >= min_height_m, explains the pixels
               well (rms <= max_fit_rms_px) and improvement_ratio x better than the ground model,
               the ground model is clearly wrong (rms >= min_ground_rms_px), and P agrees with H on
               the ground (camera_gap_px <= max_camera_gap_px).
  5. replace   on airborne flights X(t), Y(t), Z(t) of the fit replace the naive positions on every
               frame of the flight (missing frames inside it included), grown by up to `extend`
               frames towards the parabola's own launch and landing where it still matches the pixels
               (the candidate detector starts / ends a few frames inside the real flight).

WHAT CHANGED FROM THE OLD SCRIPTS (flight_detector.py, desc.py, parabolic_solver.py)
  flight_detector.py  thresholds in pixels/frame (they change with zoom and depth)      -> m/s on the ground
                      speed = 1-frame difference of a smoothed track (noise = kicks)    -> SG derivative
                      frames across a missing stretch treated as consecutive            -> runs split at gaps / jumps
                      flight starts a few frames after the kick                         -> walked back to the kick
                      no maximum length, large debug print blocks, `start_frame`
                      printed after being set to None                                   -> 4 s cap, no prints
  desc.py             one hard-coded P matrix for the whole video (camera pans/zooms)   -> P per frame from the homography
                      airborne = projected speed > 130 km/h only (a hard driven pass
                      also exceeds it, a slow lob does not)                             -> ground vs parabola reprojection
                                                                                           test; 130 km/h kept as a flag
  parabolic_solver.py P from solvePnP on keypoints of a pitch model we no longer run    -> P from the provided homography
                      [X, Y, -Z] sign flip                                               -> Z up, no flip
                      one frame without calibration aborted the whole flight            -> frames without a valid H skipped
                      sequences hard-coded                                              -> found automatically
                      kept: the 6-parameter parabola, bounds, soft end anchors (now 0.5 px/m: the
                      candidate's ends are often already in the air, a strong pull biased the fit)

LIMITS. No drag or spin (long balls curve and slow down), principal point = image centre, a header or
a volley starts above the ground (the anchor is soft, so the fit can still lift Z0 up to 3 m). Height
is only as good as the camera model: check `camera_gap_px` and the clips before trusting it.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from possession.features.flight import (G, FlightConfig, estimate_focal, fill_short_gaps,
                                        finite_runs, flight_mask, ground_consistency_px,
                                        kinematics, projection_from_homography)
from possession.preprocess.ball import remove_spikes

SRC_NONE, SRC_OBSERVED, SRC_BRIDGED, SRC_FLIGHT, SRC_FALLBACK = 0, 1, 2, 3, 4
SOURCE_NAMES = {SRC_NONE: "none", SRC_OBSERVED: "yolo", SRC_BRIDGED: "bridged", SRC_FLIGHT: "flight_fit",
                SRC_FALLBACK: "provided"}


@dataclass
class BallFixConfig:
    min_conf: float = 0.25          # YOLO picks below this are ignored
    spike_step_m: float = 2.0       # m per frame, jump away AND back = spike (as reconstruct_ball.py)
    isolated_gap: int = 10          # a detection with gaps >= this on both sides is dropped ...
    isolated_len: int = 2           # ... when its run is this short or shorter
    bridge: int = 5                 # frames filled linearly in pixels
    fit_observed_only: bool = True  # fit on YOLO-seen frames only (bridged frames are not evidence)
    anchor_weight: float = 0.5      # px per metre pulling the fit's ends to the naive ground track.
                                    # Weak on purpose: YOLO pixels are direct observations, and a
                                    # candidate's first / last frame is often already in the air
                                    # (the ATD port needed 25 because its pixels were re-projected)
    detect_smooth: int = 9          # frames: moving median of the naive ground track used ONLY
                                    # to find candidates (YOLO pixel noise is magnified with depth)
    merge_gap: int = 3              # candidates closer than this are one flight (noise splits them)
    extend: int = 25                # frames the replacement may grow on each side, up to the
                                    # parabola's own launch / landing time ...
    extend_max_px: float = 12.0     # ... where it still explains the observed pixels this well
    refits: int = 3                 # grow-and-refit rounds
    max_fit_rms_px: float = 5.0     # airborne only if the parabola explains the pixels this well
                                    # (YOLO noise is ~1-3 px; a rolling ball that decelerates or is
                                    # touched can be "explained" by a high parabola, but badly)
    min_ground_rms_px: float = 4.0  # ... and the ground model is clearly wrong (a ratio alone passes
                                    # on tiny errors)

    def to_dict(self):
        return asdict(self)


# -- input ----------------------------------------------------------------------------------------

def load_yolo_ball(path, n_frames, frames_run=None, min_conf=0.25):
    """The notebook's ball_yolo_<mode>.parquet as per-frame arrays (pixels and naive ground metres)."""
    import pandas as pd
    df = pd.read_parquet(path)
    df = df[df.conf >= min_conf]
    out = {k: np.full(n_frames, np.nan) for k in ("u", "v", "x", "y", "conf")}
    f = df.frame.to_numpy(int)
    out["u"][f], out["v"][f] = df.u.to_numpy(float), df.v.to_numpy(float)
    out["x"][f], out["y"][f] = df.x_m.to_numpy(float), df.y_m.to_numpy(float)
    out["conf"][f] = df.conf.to_numpy(float)
    out["jump"] = np.zeros(n_frames, bool)
    if "jump" in df:
        out["jump"][f] = df.jump.to_numpy(bool)
    out["ran"] = np.load(frames_run).astype(bool) if frames_run is not None else ~np.isnan(out["u"])
    return out


def _pix_to_ground(Hn, u, v, pitch):
    """Pixels -> metres through each frame's homography (Hn maps normalised pitch -> pixels)."""
    n = len(u)
    out = np.full((n, 2), np.nan)
    ok = np.isfinite(u) & np.isfinite(v) & np.isfinite(Hn).all(axis=(1, 2))
    if ok.any():
        Hi = np.linalg.inv(Hn[ok])
        p = np.einsum("nij,nj->ni", Hi, np.stack([u[ok], v[ok], np.ones(ok.sum())], 1))
        out[ok, 0] = p[:, 0] / p[:, 2] * pitch[0]
        out[ok, 1] = p[:, 1] / p[:, 2] * pitch[1]
    return out


# -- 1-2. clean and bridge -------------------------------------------------------------------------

def clean(raw, Hn, cfg: BallFixConfig, fcfg: FlightConfig):
    """Returns uv (n, 2) pixels, ground (n, 2) metres, observed (n,), removed (n,), bridged (n,)."""
    u, v = raw["u"].copy(), raw["v"].copy()
    removed = raw["jump"] & np.isfinite(u)
    u[removed] = v[removed] = np.nan
    g = _pix_to_ground(Hn, u, v, fcfg.pitch)
    _, _, spikes = remove_spikes(g[:, 0], g[:, 1], cfg.spike_step_m)
    removed |= spikes
    u[spikes] = v[spikes] = np.nan
    ok = np.isfinite(u)
    runs = finite_runs(ok)
    for k, (a, b) in enumerate(runs):
        gap_before = a - runs[k - 1][1] if k else a
        gap_after = runs[k + 1][0] - b if k + 1 < len(runs) else len(u) - b
        if b - a <= cfg.isolated_len and gap_before >= cfg.isolated_gap and gap_after >= cfg.isolated_gap:
            u[a:b] = v[a:b] = np.nan
            removed[a:b] = True
    observed = np.isfinite(u)
    uv, bridged = fill_short_gaps(np.stack([u, v], 1), cfg.bridge)
    ground = _pix_to_ground(Hn, uv[:, 0], uv[:, 1], fcfg.pitch)
    return uv, ground, observed, removed, bridged


# -- 4. one flight ---------------------------------------------------------------------------------

def fit_parabola(t, uv, P, a0, a1, anchor_w, free_z):
    """Like features/flight.fit_trajectory, but a candidate may START MID-AIR (the detector often
    triggers late on a lob): Z0 up to 25 m and a descending Vz are allowed. Returns (theta, rms)."""
    from scipy.optimize import least_squares
    T = max(t[-1], 1e-6)
    v0 = (a1 - a0) / T
    if free_z:
        x0 = [a0[0], a0[1], 0.5, v0[0], v0[1], 0.5 * G * T]
        lo, hi = [-100, -100, 0.0, -60, -60, -30.0], [200, 200, 25.0, 60, 60, 40.0]
    else:
        x0, lo, hi = [a0[0], a0[1], v0[0], v0[1]], [-100, -100, -60, -60], [200, 200, 60, 60]
    x0 = np.clip(x0, np.array(lo) + 1e-6, np.array(hi) - 1e-6)

    def unpack(th):
        return th if free_z else np.array([th[0], th[1], 0.0, th[2], th[3], 0.0])

    def residuals(th):
        X0, Y0, Z0, Vx, Vy, Vz = unpack(th)
        X, Y = X0 + Vx * t, Y0 + Vy * t
        Z = np.maximum(0.0, Z0 + Vz * t - 0.5 * G * t * t) if free_z else np.zeros_like(t)
        q = np.einsum("nij,nj->ni", P, np.stack([X, Y, Z, np.ones_like(t)], axis=1))
        r = np.concatenate([q[:, 0] / q[:, 2] - uv[:, 0], q[:, 1] / q[:, 2] - uv[:, 1]])
        w = anchor_w
        return np.concatenate([r, [w * (X0 - a0[0]), w * (Y0 - a0[1]),
                                   w * (X0 + Vx * T - a1[0]), w * (Y0 + Vy * T - a1[1])]])

    res = least_squares(residuals, x0, bounds=(lo, hi), method="trf", loss="soft_l1", f_scale=3.0)
    if res.status <= 0:
        return None, np.nan
    r = res.fun[:-4]
    return unpack(res.x), float(np.sqrt(np.mean(r ** 2)))


def fit_flight(s, e, uv, ground, observed, Hn, valid, f_global, fcfg: FlightConfig, cfg: BallFixConfig):
    """Fit one candidate [s, e]. Returns (summary dict, theta or None)."""
    frames = np.arange(s, e + 1)
    use = valid[s:e + 1] & np.isfinite(uv[s:e + 1, 0])
    if cfg.fit_observed_only and (observed[s:e + 1] & use).sum() >= fcfg.min_obs:
        use &= observed[s:e + 1]
    seg = ground[s:e + 1]
    peak_kmh = float(np.nanmax(kinematics(np.nan_to_num(seg), fcfg)[0]) * 3.6) if len(seg) > 1 else 0.0
    summ = dict(start_frame=int(s), end_frame=int(e), n_frames=int(e - s + 1), n_fit=int(use.sum()),
                naive_peak_speed_kmh=peak_kmh, speed_flag=bool(peak_kmh > fcfg.ground_speed_limit_kmh),
                max_height_m=np.nan, peak_frame=int(s), rms_px_flight=np.nan, rms_px_ground=np.nan,
                focal_px=np.nan, camera_gap_px=np.nan, fit_ok=False, is_airborne=False,
                ground_shift_m=np.nan)
    if use.sum() < fcfg.min_obs or not np.isfinite(ground[s]).all() or not np.isfinite(ground[e]).all():
        return summ, None
    own = np.zeros(len(valid), bool)
    own[s:e + 1] = valid[s:e + 1]
    try:
        f, _ = estimate_focal(Hn, own, fcfg, max_samples=200)
    except ValueError:
        f = f_global
    sel = frames[use]
    P = np.stack([projection_from_homography(Hn[k], f, fcfg) for k in sel])
    gap = ground_consistency_px(P[0], Hn[sel[0]], fcfg)
    summ.update(focal_px=float(f), camera_gap_px=float(gap))
    t = (sel - s) / float(fcfg.fps)
    a0, a1 = ground[s], ground[e]
    th, rms_f = fit_parabola(t, uv[sel], P, a0, a1, cfg.anchor_weight, free_z=True)
    _, rms_g = fit_parabola(t, uv[sel], P, a0, a1, cfg.anchor_weight, free_z=False)
    if th is None or np.isnan(rms_g):
        return summ, None
    T = (e - s) / float(fcfg.fps)
    tt = np.linspace(0.0, T, 200)
    z = np.maximum(0.0, th[2] + th[5] * tt - 0.5 * G * tt * tt)
    j = int(np.argmax(z))
    h = float(z[j])
    air = bool(h >= fcfg.min_height_m and rms_g >= fcfg.improvement_ratio * max(rms_f, 1e-9)
               and gap <= fcfg.max_camera_gap_px and rms_f <= cfg.max_fit_rms_px
               and rms_g >= cfg.min_ground_rms_px)
    tf = (frames - s) / float(fcfg.fps)
    shift = np.hypot(th[0] + th[3] * tf - ground[s:e + 1, 0], th[1] + th[4] * tf - ground[s:e + 1, 1])
    summ.update(max_height_m=h, peak_frame=int(round(s + tt[j] * fcfg.fps)), rms_px_flight=rms_f,
                rms_px_ground=rms_g, fit_ok=True, is_airborne=air,
                ground_shift_m=float(np.nanmax(shift)) if np.isfinite(shift).any() else np.nan)
    return summ, (th, f)


def _moving_median(a, w):
    """Centred moving median along axis 0 over finite runs only (NaN stays NaN)."""
    out = a.copy()
    h = w // 2
    for s_, e_ in finite_runs(np.isfinite(a).all(axis=1)):
        seg = a[s_:e_]
        if len(seg) <= 2:
            continue
        pad = np.pad(seg, ((h, h), (0, 0)), mode="edge")
        win = np.lib.stride_tricks.sliding_window_view(pad, w, axis=0)      # (len, 2, w)
        out[s_:e_] = np.median(win, axis=2)
    return out


def find_candidates(ground, fcfg, cfg):
    """flight_mask on a lightly smoothed naive ground track, then merge pieces split by noise."""
    g = _moving_median(ground, cfg.detect_smooth) if cfg.detect_smooth > 1 else ground
    spans, _ = flight_mask(g[:, 0], g[:, 1], fcfg)
    merged = []
    for s_, e_ in spans:
        if merged and s_ - merged[-1][1] <= cfg.merge_gap:
            merged[-1][1] = max(merged[-1][1], e_)
        else:
            merged.append([s_, e_])
    return [tuple(m) for m in merged]


def replacement_span(s, e, th, f, uv, observed, Hn, valid, taken, fcfg, cfg):
    """[a, b] to overwrite: the candidate, grown up to cfg.extend frames on each side towards the
    parabola's own launch / landing time while the parabola still explains the observed pixels.
    (The candidate detector starts and ends a few frames inside the real flight.)"""
    X0, Y0, Z0, Vx, Vy, Vz = th
    disc = np.sqrt(max(Vz * Vz + 2 * G * Z0, 0.0))
    t_launch, t_land = (Vz - disc) / G, (Vz + disc) / G
    a_goal = max(s - cfg.extend, s + int(np.floor(t_launch * fcfg.fps)))
    b_goal = min(e + cfg.extend, s + int(np.ceil(t_land * fcfg.fps)))

    def fits(k):
        if k < 0 or k >= len(uv) or taken[k] and not (s <= k <= e) or not valid[k]:
            return False
        if not observed[k]:
            return True
        tt = (k - s) / float(fcfg.fps)
        Pk = projection_from_homography(Hn[k], f, fcfg)
        q = Pk @ np.array([X0 + Vx * tt, Y0 + Vy * tt, max(0.0, Z0 + Vz * tt - 0.5 * G * tt * tt), 1.0])
        return float(np.hypot(q[0] / q[2] - uv[k, 0], q[1] / q[2] - uv[k, 1])) <= cfg.extend_max_px

    a = s
    while a - 1 >= a_goal and fits(a - 1):
        a -= 1
    b = e
    while b + 1 <= b_goal and fits(b + 1):
        b += 1
    return a, b


# -- entry point ------------------------------------------------------------------------------------

def fix_ball(raw, Hn, valid, fcfg: FlightConfig | None = None, cfg: BallFixConfig | None = None,
             fallback=None, verbose=True, in_play=None):
    """raw from load_yolo_ball, Hn (n,3,3) normalised pitch -> pixels, valid (n,).

    fallback  optional (n, 2) metres (e.g. the fixed provided ball): used where YOLO has no ball
              after cleaning and fitting.
    in_play   optional (n,) bool: flights are fitted only inside it (the graded halves).
    Returns dict: x, y, z (metres; z = 0 on the ground, NaN where unknown), source (int8, see
    SOURCE_NAMES), in_air, observed, removed, bridged, flights (list of dicts), report.
    """
    fcfg = fcfg or FlightConfig()
    cfg = cfg or BallFixConfig()
    n = len(raw["u"])
    uv, ground, observed, removed, bridged = clean(raw, Hn, cfg, fcfg)
    try:
        f_global, spread = estimate_focal(Hn, valid, fcfg)
    except ValueError:
        f_global, spread = np.nan, np.nan
    spans = find_candidates(ground, fcfg, cfg)
    if in_play is not None:                      # graded halves only (halftime: camera off the pitch)
        spans = [(a, b) for a, b in spans if in_play[a] and in_play[b]]
    x, y = ground[:, 0].copy(), ground[:, 1].copy()
    z = np.where(np.isfinite(x), 0.0, np.nan)
    source = np.where(observed, SRC_OBSERVED, np.where(bridged, SRC_BRIDGED, SRC_NONE)).astype(np.int8)
    in_air = np.zeros(n, bool)
    flights = []
    fitted = []
    for s, e in spans:
        summ, fit = fit_flight(s, e, uv, ground, observed, Hn, valid, f_global, fcfg, cfg)
        flights.append(summ)
        if fit is not None and summ["is_airborne"]:
            fitted.append([s, e, fit[0], fit[1], summ])
    taken = np.zeros(n, bool)
    for s, e, th, f, summ in fitted:
        taken[s:e + 1] = True
    for item in fitted:
        s, e, th, f, summ = item
        # The detector often starts / ends inside the flight: grow to where the parabola still
        # explains the pixels, refit on the grown span once, grow again.
        a, b = replacement_span(s, e, th, f, uv, observed, Hn, valid, taken, fcfg, cfg)
        for _ in range(cfg.refits):
            if (a, b) == (s, e):
                break
            summ2, fit2 = fit_flight(a, b, uv, ground, observed, Hn, valid, f_global, fcfg, cfg)
            if fit2 is None or not summ2["is_airborne"] or \
                    summ2["rms_px_flight"] > 2 * summ["rms_px_flight"] + 1:
                break
            summ.update({k: v for k, v in summ2.items() if k not in ("start_frame", "end_frame")})
            s, e, th, f = a, b, fit2[0], fit2[1]
            a, b = replacement_span(s, e, th, f, uv, observed, Hn, valid, taken, fcfg, cfg)
        summ["replaced_from"], summ["replaced_to"] = int(a), int(b)
        tf = (np.arange(a, b + 1) - s) / float(fcfg.fps)
        x[a:b + 1] = th[0] + th[3] * tf
        y[a:b + 1] = th[1] + th[4] * tf
        z[a:b + 1] = np.maximum(0.0, th[2] + th[5] * tf - 0.5 * G * tf * tf)
        source[a:b + 1] = SRC_FLIGHT
        in_air[a:b + 1] = True
        taken[a:b + 1] = True
    if fallback is not None:
        fb = np.isnan(x) & np.isfinite(fallback[:, 0])
        x[fb], y[fb] = fallback[fb, 0], fallback[fb, 1]
        z[fb] = np.nan                       # height unknown on the provided ball
        source[fb] = SRC_FALLBACK
    air = [fl for fl in flights if fl["is_airborne"]]
    report = {"frames": int(n), "yolo_frames": int(np.isfinite(raw["u"]).sum()),
              "removed": int(removed.sum()), "bridged": int(bridged.sum()),
              "focal_px_global": float(f_global), "focal_relative_iqr": float(spread),
              "flight_candidates": len(flights), "airborne_flights": len(air),
              "airborne_frames": int(in_air.sum()),
              "median_height_m": float(np.median([fl["max_height_m"] for fl in air])) if air else float("nan"),
              "median_ground_shift_m": float(np.median([fl["ground_shift_m"] for fl in air])) if air else float("nan"),
              "frames_with_ball": int(np.isfinite(x).sum()),
              "by_source": {SOURCE_NAMES[k]: int((source == k).sum()) for k in SOURCE_NAMES},
              "config": cfg.to_dict(), "flight_config": asdict(fcfg)}
    if verbose:
        print("YOLO ball: {yolo_frames:,} frames; removed {removed:,}, bridged {bridged:,}".format(**report))
        print("flights: {} candidates, {} airborne ({} frames), median peak {:.1f} m, median ground "
              "correction {:.1f} m".format(len(flights), len(air), report["airborne_frames"],
                                           report["median_height_m"], report["median_ground_shift_m"]))
    return {"x": x, "y": y, "z": z, "source": source, "in_air": in_air, "observed": observed,
            "removed": removed, "bridged": bridged, "flights": flights, "report": report}
