"""Where the tactical camera points, as a stand-in for the ball (game plan idea 9, BAR-ATH only).

The camera operator follows the ball, and the homography exists for 100 % of tactical frames
(DATA.md). Mapping the image centre back to the pitch gives a rough ball location on every frame,
including the 41 % where the tracker loses the ball, and the camera usually stops panning at a
stoppage. It lags and leads the ball by a few metres: a weak signal, never a ball position.

Columns (metres, m/s): cam_x, cam_y, cam_speed (smoothed pan speed of the aim point),
cam_reproj (reprojection error, lower = more trustworthy). NaN where the homography is invalid.
Not available on the bench (no video there).
"""
from __future__ import annotations

import numpy as np

from games import GAME, GAMES
from starter.load_homography import load as load_homography

IMAGE = (1920, 1080)


def aim_points(matrices, valid, image=IMAGE):
    """(n,) normalised pitch coordinates of the image centre for each 3x3 pitch->pixel matrix."""
    n = len(matrices)
    x = np.full(n, np.nan)
    y = np.full(n, np.nan)
    ok = np.asarray(valid, dtype=bool) & np.isfinite(matrices.reshape(n, -1)).all(1)
    if ok.any():
        inv = np.linalg.inv(matrices[ok])
        u, v = image[0] / 2.0, image[1] / 2.0
        hx = inv[:, 0, 0] * u + inv[:, 0, 1] * v + inv[:, 0, 2]
        hy = inv[:, 1, 0] * u + inv[:, 1, 1] * v + inv[:, 1, 2]
        hw = inv[:, 2, 0] * u + inv[:, 2, 1] * v + inv[:, 2, 2]
        x[ok], y[ok] = hx / hw, hy / hw
    return x, y


def camera_features(lo=0, hi=None, game=GAME, data_dir=None, smooth_s=0.5) -> dict:
    """Columns for video frames lo..hi (inclusive) of the tactical recording."""
    matrices, reproj, valid = load_homography(game, data_dir)
    hi = len(matrices) - 1 if hi is None else hi
    sl = slice(lo, hi + 1)
    nx, ny = aim_points(matrices[sl], valid[sl])
    length, width = GAMES[game]["pitch"]
    fps = GAMES[game]["fps"]
    cx, cy = nx * length, ny * width
    step = np.hypot(np.diff(cx, prepend=np.nan), np.diff(cy, prepend=np.nan)) * fps
    w = max(1, int(round(smooth_s * fps)))
    speed = np.convolve(np.nan_to_num(step), np.ones(w) / w, mode="same")
    speed[np.isnan(cx)] = np.nan
    return {"cam_x": cx.astype(np.float32), "cam_y": cy.astype(np.float32),
            "cam_speed": speed.astype(np.float32),
            "cam_reproj": np.asarray(reproj[sl], dtype=np.float32)}
