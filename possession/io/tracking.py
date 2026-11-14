"""Players and ball as numpy arrays indexed by video frame: the one input of the pipeline.

    from possession.io import load
    tr = load()                        # whole match: YOLO players + the fixed ball
    tr = load(11185, 18684)            # one window, video frames inclusive

PLAYERS come from OUR tracking (kaggle/kaggle_player_tracking.ipynb, YOLO + tracker), packed into
slots by possession/io/detections.py and cleaned there (cut at jumps, duplicates removed). The
player tracks provided in the ATD are NOT used any more.

BALL is the fixed ball of scripts/reconstruct_ball.py (outputs/ball/ball_reconstructed.npz: ATD
ball with spikes removed and short gaps Hermite-filled). If that file is missing the raw ATD ball is
read instead (``load_ball``) and a note is printed; run scripts/reconstruct_ball.py once.

Everything is in metres (x along 105.3 m, y across 68 m, the ATD frame the homography maps from),
NaN where an object is not seen. ``ps`` / ``bs`` (tracker speeds) are NaN: neither source has them.

Raw ATD access (``iter_raw``, ``seek_to_frame_id``, ``load_ball``) stays for the ball only:
``frame_id`` counts from 1, video frames from 0, ``video_frame = frame_id - 1`` (DATA.md).

Caches (outputs/cache/) hold frame-level data derived from the match: outputs/ stays git-ignored
and is never uploaded (RULES.md).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from games import DEFAULT_SOURCE, GAME, GAMES, half_video_blocks, require, source_frames

from .teams import AWAY, HOME, TeamMap, load_teams

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE_DIR = os.path.join(ROOT, "outputs", "cache")
SEARCH_BLOCK = 1 << 20   # bisection stops when the byte window is this small, then reads lines
CHUNK = 4096             # lines parsed together in one numpy call


# ------------------------------------------------------------------------------------------------
# The container
# ------------------------------------------------------------------------------------------------

@dataclass
class Tracking:
    """Tracking for video frames ``lo .. lo + n - 1``. Row ``k`` is video frame ``lo + k``."""

    lo: int
    px: np.ndarray            # (n, slots) metres
    py: np.ndarray
    ps: np.ndarray            # (n, slots) tracker speed, m/s
    bx: np.ndarray            # (n,) metres
    by: np.ndarray
    bs: np.ndarray            # (n,) tracker ball speed, m/s (often NaN)
    teams: TeamMap
    fps: int = 25
    pitch: tuple = (105.3, 68.0)
    game: str = GAME
    source: str = DEFAULT_SOURCE
    stats: dict = field(default_factory=dict)

    # -- shape and indexing --------------------------------------------------------------------
    @property
    def n(self) -> int:
        return len(self.bx)

    @property
    def hi(self) -> int:
        """Last video frame covered (inclusive)."""
        return self.lo + self.n - 1

    @property
    def frames(self) -> np.ndarray:
        return np.arange(self.lo, self.lo + self.n)

    def row(self, video_frame: int) -> int:
        k = video_frame - self.lo
        if not 0 <= k < self.n:
            raise IndexError("video frame {} outside {}..{}".format(video_frame, self.lo, self.hi))
        return k

    def window(self, first: int, last: int) -> "Tracking":
        """A view on video frames first..last (inclusive). No copy."""
        a, b = self.row(first), self.row(last) + 1
        return Tracking(first, self.px[a:b], self.py[a:b], self.ps[a:b], self.bx[a:b],
                        self.by[a:b], self.bs[a:b], self.teams, self.fps, self.pitch, self.game,
                        self.source, dict(self.stats))

    # -- convenience ---------------------------------------------------------------------------
    @property
    def ball(self) -> np.ndarray:
        """(n, 2) ball position in metres."""
        return np.stack([self.bx, self.by], axis=1)

    @property
    def ball_present(self) -> np.ndarray:
        return ~(np.isnan(self.bx) | np.isnan(self.by))

    @property
    def player_present(self) -> np.ndarray:
        return ~(np.isnan(self.px) | np.isnan(self.py))

    def team_xy(self, side: int):
        """(px, py) restricted to one team's slots (HOME or AWAY)."""
        m = self.teams.mask(side)
        return self.px[:, m], self.py[:, m]

    def tracked_per_team(self) -> dict:
        ok = self.player_present
        return {side: ok[:, self.teams.mask(side)].sum(1) for side in (HOME, AWAY)}

    # -- persistence ---------------------------------------------------------------------------
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.savez_compressed(path, lo=self.lo, px=self.px, py=self.py, ps=self.ps, bx=self.bx,
                            by=self.by, bs=self.bs, fps=self.fps, pitch=np.array(self.pitch),
                            game=self.game, source=self.source)

    @classmethod
    def from_npz(cls, path: str, teams: TeamMap) -> "Tracking":
        z = np.load(path, allow_pickle=False)
        return cls(int(z["lo"]), z["px"], z["py"], z["ps"], z["bx"], z["by"], z["bs"], teams,
                   int(z["fps"]), tuple(float(v) for v in z["pitch"]), str(z["game"]),
                   str(z["source"]), {"cached": path})


# ------------------------------------------------------------------------------------------------
# Raw file access
# ------------------------------------------------------------------------------------------------

def _frame_id(line: bytes) -> int:
    return int(line[:line.index(b":")])


def seek_to_frame_id(handle, target_id: int) -> None:
    """Position ``handle`` (binary) at or just before the line of ``target_id``.

    The file is sorted by frame id, so a byte-offset bisection gets there without reading the
    200 MB from the top (same idea as visualize_tracking.py and event_detection.py).
    """
    handle.seek(0, os.SEEK_END)
    lo, hi = 0, handle.tell()
    while hi - lo > SEARCH_BLOCK:
        mid = (lo + hi) // 2
        handle.seek(mid)
        handle.readline()                     # finish the partial line
        line = handle.readline()
        if not line:
            hi = mid
            continue
        try:
            fid = _frame_id(line)
        except ValueError:
            hi = mid
            continue
        if fid < target_id:
            lo = mid
        else:
            hi = mid
    handle.seek(lo)
    if lo:
        handle.readline()


def iter_raw(raw_path: str, lo: int, hi: int):
    """Yield ``(video_frame, players_text, ball_text)`` for video frames lo..hi in file order.

    Malformed lines are skipped; the count of skipped lines is the generator's return value.
    """
    bad = 0
    with open(raw_path, "rb") as handle:
        if lo > 0:
            seek_to_frame_id(handle, lo + 1)
        for raw in handle:
            try:
                frame = _frame_id(raw) - 1
            except ValueError:
                bad += 1
                continue
            if frame < lo:
                continue
            if frame > hi:
                break
            parts = raw.decode("ascii", "replace").strip().split(":")
            if len(parts) != 3:
                bad += 1
                continue
            yield frame, parts[1], parts[2]
    return bad


# ------------------------------------------------------------------------------------------------
# Public loaders
# ------------------------------------------------------------------------------------------------

BALL_FILE = os.path.join("outputs", "ball", "ball_reconstructed.npz")


def _span(lo, hi, game, source):
    total = source_frames(source, game)
    lo = 0 if lo is None else max(0, int(lo))
    hi = total - 1 if hi is None else min(total - 1, int(hi))
    if hi < lo:
        raise ValueError("empty frame range {}..{}".format(lo, hi))
    return lo, hi


def _fixed_ball(total, game, source, data_dir, ball_file):
    """(bx, by, source text) for the whole recording, metres."""
    path = ball_file or os.path.join(ROOT, BALL_FILE)
    if os.path.isfile(path):
        from possession.preprocess.reconstruct import load_ball_file
        bf = load_ball_file(path)
        if bf["lo"] != 0 or len(bf["x"]) != total:
            raise ValueError("{} covers {}+{} frames, the recording has {}".format(
                path, bf["lo"], len(bf["x"]), total))
        return (np.asarray(bf["x"], np.float32), np.asarray(bf["y"], np.float32),
                "fixed ball {} ({} up to {} frames)".format(os.path.relpath(path, ROOT),
                                                           bf["method"], bf["max_gap"]))
    print("note: {} not found - using the raw ATD ball (run scripts/reconstruct_ball.py)".format(
        os.path.relpath(path, ROOT)))
    bx, by, _ = load_ball(game=game, source=source, data_dir=data_dir)
    return bx.astype(np.float32), by.astype(np.float32), "raw ATD ball"


def load(lo: int | None = None, hi: int | None = None, *, game: str = GAME,
         source: str = DEFAULT_SOURCE, data_dir: str | None = None, metres: bool = True,
         cache: bool = True, cache_dir: str | None = None, players_file: str | None = None,
         team_mode: str = "local", clean: bool = True, ball_file: str | None = None,
         refresh: bool = False) -> Tracking:
    """YOLO players + fixed ball for video frames lo..hi (inclusive; default: the whole recording).

    players_file  final_tracks_*.parquet (default data/kaggle_outputs/final_tracks_full.parquet)
    team_mode     "local" (team from a +/-0.5 s window, default) or "track" (whole-track vote)
    clean         cut jumps and drop duplicate runs (possession/io/detections.py)
    cache         keep the packed players in outputs/cache/ (``cache_dir`` is kept for old callers)
    refresh       pack the players again and overwrite that cache
    """
    from .detections import load_detected

    if source != "tactical":
        raise ValueError("our player tracking exists for the tactical recording only")
    meta = load_teams(game=game, data_dir=data_dir, source=source)     # club names only
    total = source_frames(source, game)
    lo, hi = _span(lo, hi, game, source)
    fps = GAMES[game]["fps"]
    pitch = tuple(GAMES[game]["pitch"])

    det = load_detected(players_file, n_frames=total, team_mode=team_mode, clean=clean, cache=cache,
                        refresh=refresh)
    bx, by, ball_src = _fixed_ball(total, game, source, data_dir, ball_file)
    team = np.asarray(det["team"], dtype=np.int8)
    teams = TeamMap(team=team, slot_ids=tuple("Y{}".format(i + 1) for i in range(len(team))),
                    team_ids=meta.team_ids, names=meta.names, channel_order_ok=None)
    sl = slice(lo, hi + 1)
    sx, sy = (1.0, 1.0) if metres else pitch
    px = (det["px"][sl] / sx).astype(np.float32)
    py = (det["py"][sl] / sy).astype(np.float32)
    nan_p = np.full(px.shape, np.nan, np.float32)
    stats = dict(det["stats"], metres=metres, players="YOLO " + os.path.relpath(
        players_file or os.path.join(ROOT, "data", "kaggle_outputs", "final_tracks_full.parquet"), ROOT),
        ball=ball_src, frames_with_ball=int(np.isfinite(bx[sl]).sum()))
    return Tracking(lo=lo, px=px, py=py, ps=nan_p, bx=bx[sl] / sx, by=by[sl] / sy,
                    bs=np.full(hi - lo + 1, np.nan, np.float32), teams=teams, fps=fps,
                    pitch=pitch, game=game, source=source, stats=stats)


def load_ball(lo: int | None = None, hi: int | None = None, *, game: str = GAME,
              source: str = DEFAULT_SOURCE, data_dir: str | None = None, metres: bool = True):
    """Ball only, (bx, by, bs) for video frames lo..hi. Several times faster than ``load``."""
    raw_path = require(["atd_raw"], game, data_dir, source)[0]
    lo, hi = _span(lo, hi, game, source)
    out = np.full((hi - lo + 1, 3), np.nan, dtype=np.float32)
    for frame, _, btxt in iter_raw(raw_path, lo, hi):
        vals = btxt.split(",")
        for i in range(min(3, len(vals))):
            try:
                out[frame - lo, i] = float(vals[i])
            except ValueError:
                pass
    sx, sy = GAMES[game]["pitch"] if metres else (1.0, 1.0)
    return out[:, 0] * sx, out[:, 1] * sy, out[:, 2]


def graded_blocks(game: str = GAME):
    """[(half, first_video_frame, last_video_frame)] of the graded halves (DATA.md)."""
    return half_video_blocks(game)


def graded_mask(n: int, lo: int = 0, game: str = GAME) -> np.ndarray:
    """Bool per row: True where row ``k`` (video frame ``lo + k``) is in a graded half."""
    mask = np.zeros(n, dtype=bool)
    for _, first, last in half_video_blocks(game):
        a, b = max(first - lo, 0), min(last - lo + 1, n)
        if b > a:
            mask[a:b] = True
    return mask
