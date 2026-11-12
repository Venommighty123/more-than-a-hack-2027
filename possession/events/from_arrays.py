"""Run the event detector on any game given as arrays (the bench), not only on BAR-ATH's ATD file.

    from possession.events.from_arrays import detect_events
    events = detect_events(g.px, g.py, g.ball, g.team, g.blocks, tag_frames=g.tags)

Same detector, same default thresholds as ``python scripts/detect_events.py --full`` (checked: on
BAR-ATH the array path and the file path give identical events). Differences on the bench:

  * kick-offs are taken as the first frame of each block (used for the attacking direction);
  * set-piece tags are frames only (the bench simulates when, not what or who), so restart
    type is "unknown" and the restart team never comes from Smart Tagging;
  * pitch-dependent constants stay those of games.py (105.3 x 68 m; Metrica is 105 x 68 m): a
    0.3 m difference on the far goal line, which only the out-of-play events would notice.

``cached_events`` stores the result per bench game / copy in outputs/bench_events/ (derived from
public bench data, still git-ignored like everything in outputs/).
"""
from __future__ import annotations

import json
import os

import numpy as np

from possession.events.detect import Context, SegmentAnalyser, renumber

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                         "outputs", "bench_events")
TAG_SPAN = 150          # frames: a simulated tag is treated as a 6 s set-piece window


def default_args(**overrides):
    """The detector's command-line defaults (scripts/detect_events.py), as a Namespace."""
    from scripts.detect_events import parse_args
    a = parse_args([])
    for k, v in overrides.items():
        setattr(a, k, v)
    return a


def detect_events(px, py, ball, team, blocks, tag_frames=None, names=("home", "away"), args=None):
    """Events (list of dicts, ids E00001...) for every block of one game given as arrays."""
    args = args or default_args()
    ball = np.asarray(ball, dtype=np.float64)
    tags = [{"code": "Set Pieces", "start_frame": int(f), "end_frame": int(f) + TAG_SPAN, "tags": {}}
            for f in (tag_frames if tag_frames is not None else [])]
    kickoff = {int(h): int(lo) for h, lo, _ in blocks}
    ctx = Context(None, np.asarray(team), list(names), tags, args,
                  arrays=(np.asarray(px, dtype=np.float64), np.asarray(py, dtype=np.float64),
                          ball[:, 0], ball[:, 1]), kickoff=kickoff)
    events = []
    for h, lo, hi in blocks:
        events.extend(SegmentAnalyser(ctx, int(h), int(lo), int(hi)).run())
    return renumber(events)


def cached_events(game, cache_dir=CACHE_DIR, refresh=False):
    """detect_events for a possession.io.bench.BenchGame, cached as <name>_<variant>.json."""
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, "{}_{}.json".format(game.name, game.variant))
    if os.path.isfile(path) and not refresh:
        with open(path) as handle:
            return json.load(handle)
    events = detect_events(game.px, game.py, game.ball, game.team, game.blocks, game.tags)
    with open(path, "w") as handle:
        json.dump(events, handle)
    return events
