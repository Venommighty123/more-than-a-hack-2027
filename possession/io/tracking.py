"""Read the ATD raw text (players and ball) into numpy arrays indexed by video frame.

One reader for the whole repository. It replaces ``load_tracking`` in event_detection.py,
possession_test_1.py and pass_detection.py, ``read_tracking`` in visualize_tracking.py,
``load_atd_arrays`` in Krish's ball_reconstruction.py and ``load_ball`` in ball_flight.py.

File format (DATA.md)::

    <frame_id>:<x,y,s;x,y,s;...one group per slot...>:<x,y,s>      NaN = not seen

* ``frame_id`` counts from 1, video frames from 0: ``video_frame = frame_id - 1``.
* x, y are normalised 0-1; they are scaled here to metres with ``games.GAMES[game]["pitch"]``
  (105.3 x 68.0 for BAR-ATH), because every threshold downstream is in metres.
* s is the speed in m/s that the tracker reports. The old loaders threw it away; it is kept here
  as ``ps`` / ``bs`` (it is often NaN for the ball).

Everything is NaN where an object is not seen, including frames the file does not contain.

    from possession.io import load
    tr = load(11185, 18684)            # one window, video frames inclusive
    tr = load(cache=True)              # whole match, cached in outputs/cache/ after the first run

The cache holds frame-level tracking derived from the match data: it lives in outputs/, which
must stay git-ignored and must never be uploaded (RULES.md).
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


def _groups_slow(text: str, n_groups: int, width: int = 3) -> np.ndarray:
    """Tolerant parse of 'x,y,s;x,y,s;...' for lines the fast path rejects."""
    out = np.full((n_groups, width), np.nan, dtype=np.float32)
    for g, group in enumerate(text.split(";")[:n_groups]):
        for i, v in enumerate(group.split(",")[:width]):
            try:
                out[g, i] = float(v)
            except ValueError:
                pass
    return out


def _parse_chunk(lines, n_slots):
    """Players (len, slots, 3) and ball (len, 3) for a list of (players_text, ball_text)."""
    m = len(lines)
    try:
        players = np.array(",".join(p.replace(";", ",") for p, _ in lines).split(","),
                           dtype=np.float32).reshape(m, n_slots, 3)
    except ValueError:
        players = np.stack([_groups_slow(p, n_slots) for p, _ in lines])
    try:
        ball = np.array(",".join(b for _, b in lines).split(","), dtype=np.float32).reshape(m, 3)
    except ValueError:
        ball = np.stack([_groups_slow(b, 1)[0] for _, b in lines])
    return players, ball


# ------------------------------------------------------------------------------------------------
# Public loaders
# ------------------------------------------------------------------------------------------------

def _span(lo, hi, game, source):
    total = source_frames(source, game)
    lo = 0 if lo is None else max(0, int(lo))
    hi = total - 1 if hi is None else min(total - 1, int(hi))
    if hi < lo:
        raise ValueError("empty frame range {}..{}".format(lo, hi))
    return lo, hi


def _cache_path(game, source, lo, hi, metres, cache_dir):
    unit = "m" if metres else "norm"
    return os.path.join(cache_dir or CACHE_DIR,
                        "atd_{}_{}_{}_{}_{}.npz".format(game, source, lo, hi, unit))


def load(lo: int | None = None, hi: int | None = None, *, game: str = GAME,
         source: str = DEFAULT_SOURCE, data_dir: str | None = None, metres: bool = True,
         cache: bool = False, cache_dir: str | None = None) -> Tracking:
    """Players and ball for video frames lo..hi (inclusive; default: the whole recording)."""
    raw_path, meta_path = require(["atd_raw", "atd_meta"], game, data_dir, source)
    teams = load_teams(meta_path)
    lo, hi = _span(lo, hi, game, source)
    fps = GAMES[game]["fps"]
    pitch = tuple(GAMES[game]["pitch"])

    cpath = _cache_path(game, source, lo, hi, metres, cache_dir)
    if cache and os.path.isfile(cpath) and os.path.getmtime(cpath) >= os.path.getmtime(raw_path):
        return Tracking.from_npz(cpath, teams)

    n, n_slots = hi - lo + 1, teams.n_slots
    players = np.full((n, n_slots, 3), np.nan, dtype=np.float32)
    ball = np.full((n, 3), np.nan, dtype=np.float32)
    rows, chunk = [], []
    gen = iter_raw(raw_path, lo, hi)
    bad = 0
    while True:
        try:
            frame, ptxt, btxt = next(gen)
        except StopIteration as stop:
            bad = stop.value or 0
            break
        rows.append(frame - lo)
        chunk.append((ptxt, btxt))
        if len(chunk) == CHUNK:
            p, b = _parse_chunk(chunk, n_slots)
            players[rows], ball[rows] = p, b
            rows, chunk = [], []
    if chunk:
        p, b = _parse_chunk(chunk, n_slots)
        players[rows], ball[rows] = p, b

    sx, sy = pitch if metres else (1.0, 1.0)
    tr = Tracking(lo=lo, px=players[:, :, 0] * sx, py=players[:, :, 1] * sy, ps=players[:, :, 2],
                  bx=ball[:, 0] * sx, by=ball[:, 1] * sy, bs=ball[:, 2], teams=teams, fps=fps,
                  pitch=pitch, game=game, source=source,
                  stats={"skipped_lines": bad, "metres": metres,
                         "frames_with_ball": int((~np.isnan(ball[:, 0])).sum())})
    if cache:
        tr.save(cpath)
    return tr


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
