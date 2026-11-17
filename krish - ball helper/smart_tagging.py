"""Smart Tagging EDA: analyse the automatic tactical annotation XML intervals.

This script loads the Smart Tagging XML data and produces:
  - code frequency distribution
  - Team/N/A counts per code
  - Half distribution
  - Type/Side/Max Players breakdowns
  - Gap analysis against ball-missing gaps from the ATD
  - Overlap of set-piece gaps with dead-ball states in ground truth

Intended to be run as:
    python eda/smart_tagging.py

Output goes to stdout. Reports are self-contained; no external files needed.
"""

import os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from games import GAME, DEFAULT_SOURCE, require  # noqa: E402
from starter.load_smart_tagging import load as load_tags, pattern_path  # noqa: E402
from starter.load_atd import check_alignment  # noqa: E402


def main():
    # --- Load Smart Tagging ---
    events = load_tags()
    print("Smart Tagging instances: %d" % len(events))

    # --- Code frequency ---
    counts = {}
    for e in events:
        c = (e["code"] or "").strip().upper()
        counts[c] = counts.get(c, 0) + 1
    print("\nCodes:")
    for code in sorted(counts, key=lambda c: -counts[c]):
        print("  {:<34} {:>4}".format(code, counts[code]))

    # --- Team tag values ---
    teams = sorted({e["tags"].get("Team") for e in events if e["tags"].get("Team") is not None})
    print("\nTeam tag values:")
    for t in teams:
        cnt = sum(1 for e in events if e["tags"].get("Team") == t)
        print("  {} : {}".format(t, cnt))
    na_cnt = sum(1 for e in events if e["tags"].get("Team") is None)
    print("  N/A: {}".format(na_cnt))

    # --- Half distribution ---
    half_counts = {}
    for e in events:
        h = e["tags"].get("Half") or "?"
        half_counts[h] = half_counts.get(h, 0) + 1
    print("\nHalf distribution:")
    for h in sorted(half_counts):
        print("  {:>8} {:>4}".format(h, half_counts[h]))

    # --- Type distribution ---
    type_counts = {}
    for e in events:
        t = (e["tags"].get("Type") or "").strip()
        if not t: continue
        type_counts[t] = type_counts.get(t, 0) + 1
    print("\nType distribution:")
    for t in sorted(type_counts, key=lambda x: -type_counts[x]):
        print("  {:<24} {:>4}".format(t, type_counts[t]))

    # --- Tags of interest ---
    tags_of_interest = ["Team", "Type", "Side", "Max Players in the box", "Direction of ball entry"]
    print("\nTags of interest:")
    for tag in tags_of_interest:
        vals = {}
        for e in events:
            v = e["tags"].get(tag)
            if v is not None:
                vals[v] = vals.get(v, 0) + 1
        print("  {}:".format(tag))
        for v in sorted(vals):
            print("    {:<24} {:>4}".format(v, vals[v]))

    # --- ATD ball-missing gaps overlap ---
    from eda.gaps.eda_ball_gaps import iter_ball_present, find_gaps
    raw = require(["atd_raw"], GAME, None, DEFAULT_SOURCE)[0]
    pres = list(iter_ball_present(raw))
    gaps = find_gaps(pres)
    print("\nBall-missing gaps:")
    print("  total gaps: %d" % len(gaps))
    # top 10 longest
    gaps_sorted = sorted(gaps, key=lambda g: -g[2])[:10]
    print("  top 10 longest (start_id, end_id, length):")
    for start, end, length in gaps_sorted:
        print("    {:>8} {:>8} {:>8}  ({:.1f}s)".format(start, end, length, length/25.0))

    # --- Overlap between SET PIECES and ball-missing gaps ---
    set_pieces = [e for e in events if (e["code"] or "").strip().upper() == "SET PIECES"]
    print("\nSET PIECES vs ball-missing gaps overlap:")
    overlap = 0
    for sp in set_pieces:
        ts = sp["start_frame"] + 1  # ATD frame_id 1-based
        te = sp["end_frame"] + 1
        for g_start, g_end, g_len in gaps:
            if not (te < g_start or ts > g_end):
                overlap += 1
                if overlap <= 5:
                    print("    SET PIECE %d..%d overlaps gap %d..%d (len %d)" % (
                        ts, te, g_start, g_end, g_len))
    print("  total SET PIECES gaps overlapping ball-missing gaps: %d" % overlap)

    # --- Check alignment ---
    check = check_alignment()
    print("\nAlignment check: %s" % check)


if __name__ == "__main__":
    main()