import re
p = "possession/io/tracking.py"
s = open(p, encoding="utf-8").read()

# 1. docstring
a = s.index('"""Read the ATD raw text')
b = s.index('"""', a + 3) + 3
s = s[:a] + '''"""Players and ball as numpy arrays indexed by video frame: the one input of the pipeline.

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
"""''' + s[b:]

# 2. replace parse helpers + load()
a = s.index("def _groups_slow(")
b = s.index("def load_ball(")
new_mid = '''# ------------------------------------------------------------------------------------------------
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
         team_mode: str = "local", clean: bool = True, ball_file: str | None = None) -> Tracking:
    """YOLO players + fixed ball for video frames lo..hi (inclusive; default: the whole recording).

    players_file  final_tracks_*.parquet (default data/kaggle_outputs/final_tracks_full.parquet)
    team_mode     "local" (team from a +/-0.5 s window, default) or "track" (whole-track vote)
    clean         cut jumps and drop duplicate runs (possession/io/detections.py)
    cache         keep the packed players in outputs/cache/ (``cache_dir`` is kept for old callers)
    """
    from .detections import load_detected

    if source != "tactical":
        raise ValueError("our player tracking exists for the tactical recording only")
    meta = load_teams(game=game, data_dir=data_dir, source=source)     # club names only
    total = source_frames(source, game)
    lo, hi = _span(lo, hi, game, source)
    fps = GAMES[game]["fps"]
    pitch = tuple(GAMES[game]["pitch"])

    det = load_detected(players_file, n_frames=total, team_mode=team_mode, clean=clean, cache=cache)
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


'''
s = s[:a] + new_mid + s[b:]

# remove the now-duplicated old Public loaders header + _span/_cache_path if they precede (they were inside the replaced block)
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("ok", s.count("def _span("), s.count("def load("))
