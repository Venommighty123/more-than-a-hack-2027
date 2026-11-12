"""Ball-flight detection on the provided top-view (pitch-plane) tracking.

Port of the old side-view pipeline (flight_detector.py + desc.py + parabolic_solver.py) to the
More-than-a-Hack data. What changed and why:

  OLD                                        NEW
  ball in pixels (u, v), side camera         ball in normalised pitch coords (ATD) -> metres
  thresholds in pixels/frame                 thresholds in m/s (pixels/frame is not constant
                                             across the image, metres are)
  P (3x4) from PnP on pitch keypoints        P recovered from the provided 3x3 homography
  pixel -> ground via H to test Z=0 speed    kept as a *feature* (peak_ground_speed_kmh)
  parabola fit on pixel track                parabola fit on pixels obtained by projecting the
                                             ATD ball through the homography

Outputs per flight: start/end frame, time and ground position, max height (and the frame of the
peak), plus a per-frame `in_air` flag.

KEY ASSUMPTION (verify it: lofted passes should show a large `peak_ground_speed_kmh` and a clear `rms_px_ground` >> `rms_px_flight`): the ATD ball (x, y) is the ground-plane projection of
the ball's image position, i.e. an airborne ball appears displaced along the viewing ray. Then
H @ (x, y, 1) recovers the original ball pixel and a height can be fitted. If instead the ATD gives
the ball's true foot-point on the ground, height is unobservable from this data: every fit will
return ~0 m and `is_airborne` will be False everywhere. That outcome is itself informative.

Other approximations: principal point = image centre, square pixels, no lens distortion, one
global focal length (median over the match). Thresholds below are UNVALIDATED starting points:
calibrate them on your old labelled data (a different game, so allowed by RULES.md), never on the
public 5 minutes or on hand labels of this match.

FIXES (Nov 9, docs/changelog/flights.md). The first run on BAR-ATH gave 799 flights with peak
ground speeds up to 341 km/h (131 above 130 km/h) and flights up to 11 s long. Causes and fixes:
  1. it read the RAW ball: one-frame false blobs (1,071 on BAR-ATH) look like 50 m/s kicks.
     -> the source is now the prepared ball (spikes out, gaps <= 25 frames Hermite-filled):
        outputs/ball/ball_reconstructed.npz from scripts/reconstruct_ball.py, or the same
        computation in memory (preprocess/reconstruct.prepare_ball). Its own 12-frame linear
        bridge is no longer applied on top.
  2. speed was a one-frame difference of a 5-frame smoothed track (noise -> fake kicks / landings).
     -> velocity is the Savitzky-Golay DERIVATIVE over 7 frames.
  3. a flight started when the speed crossed 12 m/s, a few frames after the kick.
     -> the start is walked back to the contact frame (the speed minimum, <= back_max frames).
  4. nothing ended a flight that a player received without a clear deceleration.
     -> optional reception exit (players given): the ball within receive_r m of a player, slower
        than receive_speed m/s, after min_flight_frames; and a hard cap of max_flight_s.
  5. the height fit padded the last observation's camera/pixel, not the end frame's.
     -> the end frame's own homography and position are used.
  6. one focal length for the whole match, but the tactical camera zooms (relative IQR 0.18): P
     and H disagreed by ~100 px on the ground, so every height was meaningless.
     -> focal length estimated per flight from its own homographies; the P-vs-H gap is stored
        (camera_gap_px) and a flight is airborne only when it is below max_camera_gap_px.
  7. a jump faster than any kick between two consecutive frames (tracker switching detections)
     could start or end a flight. -> runs are split there (max_step_speed).
The kinematic part (``flight_mask``) needs no homography, so it also runs on the bench and is what
the labellers use (labeller v5: no team controls the ball inside a flight). The height test stays
a weak hint: the ATD ball looks like a ground foot-point, so most fits return ~0 m.

Needs numpy and scipy. Usage:
    python scripts/detect_flights.py   # -> outputs/flights/ball_flights.csv, in_air.npy, in_flight.npy
"""

import argparse
import os
import sys
from dataclasses import dataclass, asdict

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import savgol_filter

G = 9.81


@dataclass
class FlightConfig:
    fps: int = 25
    pitch: tuple = (105.3, 68.0)          # metres, as games.py
    image_size: tuple = (1920, 1080)
    # --- gap handling
    max_gap: int = 12                     # bridge used only on an UNPREPARED ball (prepared=False)
    # --- candidate detector (old flight_detector.py, converted to metres)
    smooth_window: int = 7                # Savitzky-Golay window (derivative = velocity)
    start_speed: float = 12.0             # m/s   flight starts above this
    stop_speed: float = 4.0               # m/s   flight ends below this
    impact_dv: float = 4.0                # m/s increase in one frame -> mid-air touch, split
    landing_dv: float = -3.0              # m/s decrease in one frame -> landing / trap
    turn_angle: float = 0.6               # rad direction change ...
    turn_min_speed: float = 6.0           # ... counted only above this speed
    min_flight_frames: int = 10
    back_max: int = 5                     # frames the start may be walked back to the kick
    max_flight_s: float = 4.0             # no flight lasts longer (the old run had 11 s ones)
    receive_r: float = 1.5                # m: reception exit, ball this close to a player ...
    receive_speed: float = 8.0            # ... and slower than this (m/s)
    max_step_speed: float = 40.0          # m/s between two frames: a re-detection, splits the run
    # --- height fit (old parabolic_solver.py)
    anchor_weight: float = 25.0           # px per metre pulling ends to the tracked position
    min_obs: int = 6
    min_height_m: float = 1.0             # airborne requires fitted peak >= this ...
    improvement_ratio: float = 1.5        # ... and ground-model RMS >= ratio * flight-model RMS
    ground_speed_limit_kmh: float = 130.0  # old desc.py rule, kept as a flag
    max_camera_gap_px: float = 10.0       # height trusted only if P(Z=0) matches H this well


@dataclass
class Flight:
    start_frame: int
    end_frame: int
    start_time_s: float
    end_time_s: float
    start_x_m: float
    start_y_m: float
    end_x_m: float
    end_y_m: float
    max_height_m: float
    peak_frame: int
    is_airborne: bool
    rms_px_flight: float
    rms_px_ground: float
    peak_ground_speed_kmh: float
    z0_speed_violation: bool
    n_observed: int
    fit_ok: bool
    focal_px: float = float("nan")        # focal length used for this flight (its own frames)
    camera_gap_px: float = float("nan")   # P at Z=0 vs the homography; large = height not trusted


# -- inputs -------------------------------------------------------------------------------------

def load_ball(raw_path, n_frames):
    """(n, 2) normalised ball (x, y) per VIDEO frame from the raw ATD, NaN where not tracked.

    Same parse as baseline.py: line = '<frame_id>:<players>:<ball>', frame_id counts from 1.
    """
    out = np.full((n_frames, 2), np.nan)
    with open(raw_path) as handle:
        for line in handle:
            parts = line.split(":")
            if len(parts) != 3:
                continue
            ball = parts[2].strip()
            if not ball or ball[0] == "N":
                continue
            values = ball.split(",")
            out[int(parts[0]) - 1] = (float(values[0]), float(values[1]))
    return out


# -- geometry -----------------------------------------------------------------------------------

def estimate_focal(Hn, valid, cfg, max_samples=2000):
    """Focal length (px) from the homographies, assuming centred principal point, square pixels.

    With H_m = K [r1 r2 t] and orthonormal r1, r2:  r1.r2 = 0  and |r1| = |r2|  each give f^2.
    Median over a sample of frames. Returns (f, relative_iqr); a large iqr means the camera zooms
    or the homographies are noisy, and a global f is then a poor model.
    """
    L, W = cfg.pitch
    cx, cy = cfg.image_size[0] / 2.0, cfg.image_size[1] / 2.0
    idx = np.flatnonzero(valid)
    if not len(idx):
        raise ValueError("no valid homography")
    idx = idx[np.linspace(0, len(idx) - 1, min(max_samples, len(idx))).astype(int)]
    D = np.diag([1.0 / L, 1.0 / W, 1.0])
    est = []
    for k in idx:
        p = Hn[k] @ D
        p[0] -= cx * p[2]
        p[1] -= cy * p[2]
        p1, p2 = p[:, 0], p[:, 1]
        den = p1[2] * p2[2]
        if abs(den) > 1e-12:
            f2 = -(p1[0] * p2[0] + p1[1] * p2[1]) / den
            if f2 > 0:
                est.append(np.sqrt(f2))
        den = p2[2] ** 2 - p1[2] ** 2
        if abs(den) > 1e-12:
            f2 = ((p1[0] ** 2 + p1[1] ** 2) - (p2[0] ** 2 + p2[1] ** 2)) / den
            if f2 > 0:
                est.append(np.sqrt(f2))
    if not est:
        raise ValueError("could not estimate a focal length from the homographies")
    est = np.array(est)
    med = float(np.median(est))
    return med, float(np.percentile(est, 75) - np.percentile(est, 25)) / med


def projection_from_homography(Hn, f, cfg):
    """3x4 camera matrix P with pixel ~ P @ [X, Y, Z, 1], X/Y in metres on the pitch, Z UP.

    Hn maps normalised pitch (0-1) to pixels. Note: your old P used [X, Y, -Z, 1] (Z down); here Z
    is up, so there is no sign flip when projecting.
    """
    L, W = cfg.pitch
    K = np.array([[f, 0, cfg.image_size[0] / 2.0], [0, f, cfg.image_size[1] / 2.0], [0, 0, 1.0]])
    M = np.linalg.inv(K) @ (Hn @ np.diag([1.0 / L, 1.0 / W, 1.0]))
    M = M * (2.0 / (np.linalg.norm(M[:, 0]) + np.linalg.norm(M[:, 1])))
    if M[2, 2] < 0:                       # pitch origin must be in front of the camera
        M = -M
    r1, r2, t = M[:, 0], M[:, 1], M[:, 2]
    U, _, Vt = np.linalg.svd(np.column_stack((r1, r2, np.cross(r1, r2))))
    R = U @ Vt
    if (-R.T @ t)[2] < 0:                 # camera must be above the pitch: make +Z point up
        R[:, 2] *= -1
    return K @ np.column_stack((R, t))


def ground_consistency_px(P, Hn, cfg):
    """Mean pixel gap between P at Z=0 and the provided homography over a pitch grid."""
    L, W = cfg.pitch
    gx, gy = np.meshgrid(np.linspace(0.1, 0.9, 5), np.linspace(0.1, 0.9, 5))
    xn, yn = gx.ravel(), gy.ravel()
    a = Hn @ np.stack([xn, yn, np.ones_like(xn)])
    b = P @ np.stack([xn * L, yn * W, np.zeros_like(xn), np.ones_like(xn)])
    return float(np.mean(np.hypot(a[0] / a[2] - b[0] / b[2], a[1] / a[2] - b[1] / b[2])))


def _pixels(Hn, xy_norm):
    """Pitch (0-1) -> pixels through one homography; xy_norm is (n, 2)."""
    q = Hn @ np.vstack([xy_norm[:, 0], xy_norm[:, 1], np.ones(len(xy_norm))])
    return np.stack([q[0] / q[2], q[1] / q[2]], axis=1)


# -- candidate detector (old flight_detector.py) ------------------------------------------------

def finite_runs(mask):
    """Half-open [start, stop) runs where mask is True."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def fill_short_gaps(xy, max_gap):
    """Linear interpolation of gaps <= max_gap frames. Returns (filled, interpolated_mask)."""
    filled = xy.copy()
    interp = np.zeros(len(xy), dtype=bool)
    idx = np.flatnonzero(~np.isnan(xy).any(axis=1))
    for g in np.flatnonzero(np.diff(idx) > 1):
        a, b = idx[g], idx[g + 1]
        if b - a - 1 <= max_gap:
            t = np.arange(a + 1, b)
            for k in range(2):
                filled[t, k] = np.interp(t, [a, b], [xy[a, k], xy[b, k]])
            interp[t] = True
    return filled, interp


def kinematics(xy, cfg):
    """Speed (m/s), heading, per-frame speed change and heading change from a Savitzky-Golay
    derivative of a gap-free run of metre positions."""
    n = len(xy)
    w = min(cfg.smooth_window, n if n % 2 else n - 1)
    if w >= 3:
        v = savgol_filter(xy, w, min(2, w - 1), deriv=1, axis=0) * cfg.fps
    else:
        v = np.diff(xy, axis=0, prepend=xy[:1]) * cfg.fps
    speed = np.hypot(v[:, 0], v[:, 1])
    ang = np.arctan2(v[:, 1], v[:, 0])
    dv = np.diff(speed, prepend=speed[:1])
    dang = np.abs(np.diff(ang, prepend=ang[:1]))
    dang = np.where(dang > np.pi, 2 * np.pi - dang, dang)
    return speed, dv, dang


def detect_candidates(xy, cfg, near=None):
    """Hysteresis state machine on one gap-free run of metre coordinates -> [(i0, i1)] inclusive.

    Exits: mid-air touch (split), landing deceleration, direction snap, low speed (old script),
    plus reception (``near`` = distance of the nearest player, optional) and max_flight_s.
    """
    n = len(xy)
    if n <= cfg.min_flight_frames:
        return []
    speed, dv, dang = kinematics(xy, cfg)
    longest = int(cfg.max_flight_s * cfg.fps)

    def kick(i):            # walk back from the threshold crossing to the contact frame
        j = i
        while j > 0 and i - j < cfg.back_max and speed[j - 1] < speed[j]:
            j -= 1
        return j

    out, in_flight, start = [], False, 0
    for i in range(n):
        if not in_flight:
            if speed[i] > cfg.start_speed:
                in_flight, start = True, kick(i)
                if out and start <= out[-1][1]:
                    start = out[-1][1] + 1
            continue
        if dv[i] > cfg.impact_dv:                                   # mid-air touch: split
            if i - start >= cfg.min_flight_frames:
                out.append((start, i - 1))
            start = i
            continue
        received = near is not None and i - start >= cfg.min_flight_frames and \
            near[i] < cfg.receive_r and speed[i] < cfg.receive_speed
        if dv[i] < cfg.landing_dv or received or i - start >= longest or \
                (dang[i] > cfg.turn_angle and speed[i] > cfg.turn_min_speed) or \
                speed[i] < cfg.stop_speed:                          # landing / reception / snap / slow
            if i - start >= cfg.min_flight_frames:
                out.append((start, i))
            in_flight = False
    if in_flight and n - 1 - start >= cfg.min_flight_frames:
        out.append((start, n - 1))
    return out


def _split_runs(ok, breaks):
    """finite_runs(ok), additionally split before every frame flagged in ``breaks``."""
    out = []
    for a, b in finite_runs(ok):
        cuts = [a] + [k for k in np.flatnonzero(breaks[a + 1:b]) + a + 1] + [b]
        out += [(c0, c1) for c0, c1 in zip(cuts[:-1], cuts[1:]) if c1 > c0]
    return out


def nearest_player_distance(px, py, x, y, chunk=20000):
    """(n,) distance from the ball to the nearest tracked player (inf when none / no ball)."""
    n = len(x)
    out = np.full(n, np.inf)
    for a in range(0, n, chunk):
        b = min(n, a + chunk)
        with np.errstate(invalid="ignore"):
            d = np.hypot(px[a:b] - x[a:b, None], py[a:b] - y[a:b, None])
        d = np.where(np.isnan(d), np.inf, d)
        out[a:b] = d.min(1) if d.shape[1] else np.inf
    return out


def flight_mask(x, y, cfg=None, px=None, py=None):
    """Kinematic flights on a PREPARED ball (metres, spikes out, short gaps filled).

    Returns (flights, in_flight): flights = [(start, end)] inclusive frames, in_flight (n,) bool.
    No homography needed: runs on BAR-ATH and on the bench alike. Pass px, py to enable the
    reception exit.
    """
    cfg = cfg or FlightConfig()
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    near = nearest_player_distance(px, py, x, y) if px is not None else None
    xy = np.stack([x, y], axis=1)
    ok = ~np.isnan(xy).any(axis=1)
    # a jump faster than any kick between two frames is the tracker switching detections: cut the
    # run there so it cannot start or end a flight
    with np.errstate(invalid="ignore"):
        step = np.r_[0.0, np.hypot(*np.diff(xy, axis=0).T) * cfg.fps]
    flights = []
    for a, b in _split_runs(ok, step > cfg.max_step_speed):
        for i0, i1 in detect_candidates(xy[a:b], cfg, None if near is None else near[a:b]):
            flights.append((a + i0, a + i1))
    in_flight = np.zeros(len(x), dtype=bool)
    for s, e in flights:
        in_flight[s:e + 1] = True
    return flights, in_flight


def detect_candidates_legacy(xy, cfg):
    """The Nov 7 candidate detector, unchanged (one-frame speed of a smoothed track, no kick
    walk-back, no reception exit). Kept because events/passes.py was built and checked on it;
    flight_mask uses detect_candidates. Pass cfg with smooth_window=5 for the original output.

    Hysteresis state machine on one gap-free run of metre coordinates -> [(i0, i1)] inclusive.

    Same four exits as the old script (mid-air touch, landing deceleration, direction snap, low
    speed), with kinematics from smoothed *positions* in m/s instead of raw pixels.
    """
    n = len(xy)
    if n <= cfg.min_flight_frames:
        return []
    s = savgol_filter(xy, cfg.smooth_window, 2, axis=0) if n >= cfg.smooth_window else xy
    d = np.diff(s, axis=0, prepend=s[:1])
    speed = np.hypot(d[:, 0], d[:, 1]) * cfg.fps
    ang = np.arctan2(d[:, 1], d[:, 0])
    dv = np.diff(speed, prepend=speed[:1])
    dang = np.abs(np.diff(ang, prepend=ang[:1]))
    dang = np.where(dang > np.pi, 2 * np.pi - dang, dang)

    out, in_flight, start = [], False, 0
    for i in range(n):
        if not in_flight:
            if speed[i] > cfg.start_speed:
                in_flight, start = True, i
            continue
        if dv[i] > cfg.impact_dv:                                   # mid-air touch: split
            if i - start >= cfg.min_flight_frames:
                out.append((start, i - 1))
            start = i
        elif dv[i] < cfg.landing_dv or \
                (dang[i] > cfg.turn_angle and speed[i] > cfg.turn_min_speed) or \
                speed[i] < cfg.stop_speed:                          # landing / snap / slow
            if i - start >= cfg.min_flight_frames:
                out.append((start, i))
            in_flight = False
    if in_flight and n - 1 - start >= cfg.min_flight_frames:
        out.append((start, n - 1))
    return out


# -- 3D parabola fit (old parabolic_solver.py, vectorised) --------------------------------------

def fit_trajectory(t, uv, P, a0, a1, cfg, free_z):
    """Fit X0,Y0,(Z0),Vx,Vy,(Vz); ends softly pinned to tracked ground positions a0, a1 (metres).

    Returns (theta_full[6], rms_px) or (None, nan).
    """
    T = max(t[-1], 1e-6)
    v0 = (a1 - a0) / T
    if free_z:
        x0 = [a0[0], a0[1], 0.3, v0[0], v0[1], 0.5 * G * T]
        lo = [-100, -100, 0.0, -60, -60, 0.0]
        hi = [200, 200, 3.0, 60, 60, 40.0]
    else:
        x0 = [a0[0], a0[1], v0[0], v0[1]]
        lo = [-100, -100, -60, -60]
        hi = [200, 200, 60, 60]
    x0 = np.clip(x0, np.array(lo) + 1e-6, np.array(hi) - 1e-6)

    def unpack(theta):
        if free_z:
            return theta
        return np.array([theta[0], theta[1], 0.0, theta[2], theta[3], 0.0])

    def residuals(theta):
        X0, Y0, Z0, Vx, Vy, Vz = unpack(theta)
        X, Y = X0 + Vx * t, Y0 + Vy * t
        Z = np.maximum(0.0, Z0 + Vz * t - 0.5 * G * t * t) if free_z else np.zeros_like(t)
        q = np.einsum("nij,nj->ni", P, np.stack([X, Y, Z, np.ones_like(t)], axis=1))
        r = np.concatenate([q[:, 0] / q[:, 2] - uv[:, 0], q[:, 1] / q[:, 2] - uv[:, 1]])
        w = cfg.anchor_weight
        anchors = [w * (X0 - a0[0]), w * (Y0 - a0[1]),
                   w * (X0 + Vx * T - a1[0]), w * (Y0 + Vy * T - a1[1])]
        return np.concatenate([r, anchors])

    res = least_squares(residuals, x0, bounds=(lo, hi), method="trf")
    if res.status <= 0:
        return None, np.nan
    r = res.fun[:-4]
    return unpack(res.x), float(np.sqrt(np.mean(r ** 2)))


def _characterise(s, e, filled_m, observed, Hn, valid, f, cfg):
    L, W = cfg.pitch
    scale = np.array([L, W])
    frames = np.arange(s, e + 1)
    ok = valid[s:e + 1]
    obs = observed[s:e + 1] & ok
    if obs.sum() < cfg.min_obs:
        obs = ok.copy()                     # too few real detections: also use interpolated frames
    base = dict(
        start_frame=s, end_frame=e, start_time_s=s / cfg.fps, end_time_s=e / cfg.fps,
        start_x_m=float(filled_m[s, 0]), start_y_m=float(filled_m[s, 1]),
        end_x_m=float(filled_m[e, 0]), end_y_m=float(filled_m[e, 1]),
        n_observed=int(obs.sum()))

    seg = filled_m[s:e + 1]
    peak_kmh = float(np.max(kinematics(seg, cfg)[0]) * 3.6) if len(seg) > 1 else 0.0
    base.update(peak_ground_speed_kmh=peak_kmh,
                z0_speed_violation=bool(peak_kmh > cfg.ground_speed_limit_kmh))

    failed = dict(base, max_height_m=float("nan"), peak_frame=s, is_airborne=False,
                  rms_px_flight=float("nan"), rms_px_ground=float("nan"), fit_ok=False)
    if obs.sum() < cfg.min_obs:
        return Flight(**failed)

    sel = frames[obs]
    own = np.zeros(len(valid), dtype=bool)
    own[s:e + 1] = valid[s:e + 1]
    try:
        f, _ = estimate_focal(Hn, own, cfg, max_samples=200)       # the camera zooms: per flight
    except ValueError:
        pass
    gap = ground_consistency_px(projection_from_homography(Hn[sel[0]], f, cfg), Hn[sel[0]], cfg)
    base.update(focal_px=float(f), camera_gap_px=float(gap))
    failed.update(focal_px=float(f), camera_gap_px=float(gap))
    t = (sel - s) / float(cfg.fps)
    P = np.stack([projection_from_homography(Hn[k_], f, cfg) for k_ in sel])
    uv = np.stack([_pixels(Hn[k_], (filled_m[k_] / scale)[None])[0] for k_ in sel])
    a0, a1 = filled_m[s], filled_m[e]
    # the ends are anchors, so make sure the first/last frames are in the fit's time span
    T = (e - s) / float(cfg.fps)
    t_full = np.append(t, T) if t[-1] < T else t
    if t_full is not t:                     # pad with the END frame's own camera and position
        P = np.concatenate([P, projection_from_homography(Hn[e], f, cfg)[None]])
        uv = np.concatenate([uv, _pixels(Hn[e], (filled_m[e] / scale)[None])])
        t = t_full

    th, rms_f = fit_trajectory(t, uv, P, a0, a1, cfg, free_z=True)
    _, rms_g = fit_trajectory(t, uv, P, a0, a1, cfg, free_z=False)
    if th is None or np.isnan(rms_g):
        return Flight(**failed)

    tt = np.linspace(0.0, T, 200)
    z = np.maximum(0.0, th[2] + th[5] * tt - 0.5 * G * tt * tt)
    j = int(np.argmax(z))
    h = float(z[j])
    airborne = bool(h >= cfg.min_height_m and rms_g >= cfg.improvement_ratio * max(rms_f, 1e-9)
                    and gap <= cfg.max_camera_gap_px)
    return Flight(**dict(base, max_height_m=h, peak_frame=int(round(s + tt[j] * cfg.fps)),
                         is_airborne=airborne, rms_px_flight=rms_f, rms_px_ground=rms_g,
                         fit_ok=True))


# -- entry point --------------------------------------------------------------------------------

def detect_flights(ball_norm, Hn, valid, cfg=None, f=None, verbose=True, prepared=True,
                   px=None, py=None, observed=None):
    """ball_norm (n,2) normalised ball, Hn (n,3,3) pitch->pixel, valid (n,) bool.

    prepared  True (default): ball_norm is the prepared ball (reconstruct.prepare_ball); False:
              a raw ball, bridged here over max_gap frames (the old behaviour, spikes kept)
    px, py    players in metres (optional): enables the reception exit
    observed  (n,) frames the tracker really saw (height fit prefers them); default = not NaN

    Returns (flights, in_air):
      flights : list[Flight]   every candidate, `is_airborne` says which survived the height test
      in_air  : (n,) bool      True on frames of airborne flights
    """
    cfg = cfg or FlightConfig()
    n = len(ball_norm)
    scale = np.array(cfg.pitch)
    ball_m = ball_norm * scale
    if prepared:
        filled = ball_m
    else:
        filled, _ = fill_short_gaps(ball_m, cfg.max_gap)
    if observed is None:
        observed = ~np.isnan(ball_m).any(axis=1)

    if f is None:
        f, spread = estimate_focal(Hn, valid, cfg)
        if verbose:
            print("focal length {:.0f} px (relative IQR {:.2f}{})".format(
                f, spread, "  <- large: camera zooms or H is noisy" if spread > 0.15 else ""))
    if verbose:
        k = int(np.flatnonzero(valid)[np.flatnonzero(valid).size // 2])
        print("P vs H on the ground plane, mean gap {:.2f} px (should be small)".format(
            ground_consistency_px(projection_from_homography(Hn[k], f, cfg), Hn[k], cfg)))

    spans, in_flight = flight_mask(filled[:, 0], filled[:, 1], cfg, px, py)
    flights = [_characterise(s, e, filled, observed, Hn, valid, f, cfg) for s, e in spans]

    in_air = np.zeros(n, dtype=bool)
    for fl in flights:
        if fl.is_airborne:
            in_air[fl.start_frame:fl.end_frame + 1] = True
    return flights, in_air, in_flight


def main(argv=None):
    import pandas as pd
    import possession.io  # noqa: F401  (repository root on sys.path)
    from possession.io import ROOT, load
    from possession.preprocess.reconstruct import DEFAULT_FILE, load_ball_file, prepare_ball
    from games import GAME, GAMES
    from starter.load_homography import load as load_h

    out_dir = os.path.join(ROOT, "outputs", "flights")
    parser = argparse.ArgumentParser(description="Ball-flight detection (outputs are derived match data)")
    parser.add_argument("--out", default=os.path.join(out_dir, "ball_flights.csv"))
    parser.add_argument("--frames-out", default=os.path.join(out_dir, "in_air.npy"))
    parser.add_argument("--flight-frames-out", default=os.path.join(out_dir, "in_flight.npy"))
    parser.add_argument("--ball-file", default=os.path.join(ROOT, DEFAULT_FILE),
                        help="prepared ball from scripts/reconstruct_ball.py (computed if missing)")
    parser.add_argument("--no-players", action="store_true", help="disable the reception exit")
    args = parser.parse_args(argv)
    for path in (args.out, args.frames_out, args.flight_frames_out):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    cfg = FlightConfig(fps=GAMES[GAME]["fps"], pitch=GAMES[GAME]["pitch"])
    tr = load(cache=True)
    if os.path.exists(args.ball_file):
        bf = load_ball_file(args.ball_file)
        x, y, observed = bf["x"], bf["y"], bf["observed"] & ~bf["removed"]
        print("ball source: {} ({} up to {} frames)".format(
            os.path.relpath(args.ball_file, ROOT), bf["method"], bf["max_gap"]))
    else:
        x, y, observed, removed, _ = prepare_ball(tr.bx, tr.by, fps=tr.fps)
        observed = observed & ~removed
        print("ball source: prepared in memory (run scripts/reconstruct_ball.py to store it)")
    scale = np.array(cfg.pitch)
    ball = np.stack([x, y], axis=1) / scale
    n = len(ball)
    Hn, _, valid = load_h()
    Hn, valid = Hn[tr.lo:tr.lo + n], valid[tr.lo:tr.lo + n]
    px, py = (None, None) if args.no_players else (tr.px, tr.py)
    flights, in_air, in_flight = detect_flights(ball, Hn, valid, cfg, px=px, py=py,
                                                observed=observed)

    table = pd.DataFrame([asdict(x) for x in flights])
    if len(table):
        table[["start_frame", "end_frame", "peak_frame"]] += tr.lo
    table.to_csv(args.out, index=False)
    np.save(args.frames_out, in_air)
    np.save(args.flight_frames_out, in_flight)
    print("{} flights ({} frames in flight), {} airborne ({} frames in air)".format(
        len(flights), int(in_flight.sum()),
        int(table["is_airborne"].sum()) if len(table) else 0, int(in_air.sum())))
    if len(table):
        dur = (table["end_frame"] - table["start_frame"] + 1) / cfg.fps
        print("duration s: median {:.2f}, max {:.2f};  peak ground speed km/h: median {:.0f}, "
              "max {:.0f}, above {:.0f}: {}".format(
                  dur.median(), dur.max(), table["peak_ground_speed_kmh"].median(),
                  table["peak_ground_speed_kmh"].max(), cfg.ground_speed_limit_kmh,
                  int(table["z0_speed_violation"].sum())))
        print("fit failures: {}".format(int((~table["fit_ok"]).sum())))
        print(table[table["is_airborne"]][["start_frame", "end_frame", "max_height_m",
                                           "peak_ground_speed_kmh"]].head(20).to_string())


if __name__ == "__main__":
    main()