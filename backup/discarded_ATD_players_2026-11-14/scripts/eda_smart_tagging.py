"""Smart Tagging at a glance: codes, team / half / type / tag-group counts, how set pieces line up
with the ball gaps, and the time-base check.

    python scripts/eda_smart_tagging.py

From Krish's smart_tagging.py, rebuilt on possession.io.tags (case-insensitive, teams resolved) and
possession.gaps. Prints counts only (no positions).
"""
from collections import Counter

import numpy as np

try:
    import _bootstrap  # noqa: F401  (python scripts/x.py)
except ImportError:
    from scripts import _bootstrap  # noqa: F401  (python -m scripts.x)

from possession.gaps.find import find_gaps
from possession.io import graded_blocks, load_ball, load_tags, load_teams
from possession.io.tags import overlapping, set_pieces
from starter.load_smart_tagging import check_time_base, load as load_raw

GROUPS = ("team", "half", "type", "side", "max players in the box", "direction of ball entry")


def table(title, counter):
    print("\n" + title)
    for k, v in sorted(counter.items(), key=lambda kv: (-kv[1], str(kv[0]))):
        print("  {:<34} {:>4}".format(str(k), v))


def main():
    teams = load_teams()
    tags = load_tags(teams)
    print("Smart Tagging instances: {}  (H = {}, A = {})".format(len(tags), *teams.names))
    table("Codes:", Counter(t.code.upper() for t in tags))
    table("Team (resolved):", Counter({0: "H", 1: "A", None: "N/A"}[t.team] for t in tags))
    table("Team N/A by code / type:", Counter("{} / {}".format(t.code, t.type) for t in tags
                                              if t.team is None))
    for g in GROUPS[1:]:
        table("Tag group '{}':".format(g), Counter(t.groups.get(g) for t in tags if g in t.groups))
    d = sorted(t.duration_s for t in tags)
    print("\nInterval length: median {:.1f} s, min {:.1f} s, max {:.1f} s".format(
        d[len(d) // 2], d[0], d[-1]))

    bx, _, _ = load_ball()
    gaps = find_gaps(np.isfinite(bx), graded_blocks(), min_len=25)   # ball missing >= 1 s
    pieces = set_pieces(tags)
    hit = [p for p in pieces if any(p.overlaps(g.start, g.end - 1) for g in gaps)]
    print("\nBall gaps >= 1 s in the graded halves: {}".format(len(gaps)))
    print("SET PIECES overlapping a ball gap: {} of {}".format(len(hit), len(pieces)))
    table("  ... by type:", Counter(p.type for p in hit))
    covered = sum(1 for g in gaps if overlapping(pieces, g.start, g.end - 1))
    print("Ball gaps overlapping a SET PIECES tag: {} of {}".format(covered, len(gaps)))

    print("\nTime base check (kick-offs): {}".format(check_time_base(load_raw())))


if __name__ == "__main__":
    main()
