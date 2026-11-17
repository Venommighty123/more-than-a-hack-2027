import os, sys, glob
sys.path.insert(0, '.')
from eda.eda_ball_gaps import iter_ball_present, find_gaps
from games import GAME, DEFAULT_SOURCE, require, half_video_blocks
from starter.load_smart_tagging import load as load_tags

# Load Data
raw_path = require(['atd_raw'], GAME, None, DEFAULT_SOURCE)[0]
present = list(iter_ball_present(raw_path))
all_gaps = find_gaps(present)

tags = load_tags()
blocks = half_video_blocks(GAME) # [(half, first_frame, last_frame)]

# Map gap files
gap_files = glob.glob('gap_clips/gap_*.mp4')

classified = []

for f in sorted(gap_files):
    fname = os.path.basename(f)
    parts = fname.replace('gap_', '').replace('.mp4', '').split('_')
    v0 = int(parts[0])
    length = int(parts[1])

    start_id = v0 + 51
    end_id = start_id + length - 1

    # 1. Check Set Piece / Goal events overlapping or starting near end_id
    set_pieces = []
    for t in tags:
        ts = t['start_frame'] + 1
        te = t['end_frame'] + 1
        code = t['code'] or ''
        ttype = t['tags'].get('Type', '')
        team = t['tags'].get('Team', '')

        # Check if set piece overlaps [start_id - 50, end_id + 100]
        if 'set piece' in code.lower() or 'goal' in code.lower():
            if not (te < (start_id - 50) or ts > (end_id + 100)):
                set_pieces.append((code, ttype, team, ts, te))

    # 2. Check half boundaries (kickoff / halftime / end of game)
    is_graded = any(first <= end_id <= last for _, first, last in blocks)
    boundary_event = None
    if start_id < blocks[0][1]:
        boundary_event = "Pre-kickoff (1st Half Start)"
    elif blocks[0][2] < end_id < blocks[1][1]:
        boundary_event = "Halftime Break"
    elif end_id > blocks[1][2]:
        boundary_event = "Post-match (Full Time)"

    # 3. Check for Phase / Possession change right before vs right after gap
    tags_before = [t for t in tags if (t['start_frame'] + 1) <= start_id <= (t['end_frame'] + 1)]
    tags_after = [t for t in tags if (t['start_frame'] + 1) <= end_id + 25 <= (t['end_frame'] + 1)]

    teams_before = set(t['tags'].get('Team') for t in tags_before if t['tags'].get('Team') and t['tags'].get('Team') != 'N/A')
    teams_after = set(t['tags'].get('Team') for t in tags_after if t['tags'].get('Team') and t['tags'].get('Team') != 'N/A')
    possession_changed = len(teams_before) > 0 and len(teams_after) > 0 and teams_before != teams_after

    has_change = len(set_pieces) > 0 or (boundary_event is not None) or possession_changed

    reasons = []
    if set_pieces:
        sp_desc = [f"{sp[1] if sp[1] != 'N/A' else sp[0]} [{sp[2]}]" for sp in set_pieces]
        reasons.append("Set Piece: " + ", ".join(sp_desc))
    if boundary_event:
        reasons.append("Match Boundary: " + boundary_event)
    if possession_changed:
        reasons.append(f"Possession Handover ({'/'.join(teams_before)} -> {'/'.join(teams_after)})")

    classified.append({
        'filename': fname,
        'v0': v0,
        'length': length,
        'start_id': start_id,
        'end_id': end_id,
        'has_change': has_change,
        'reasons': reasons,
        'set_pieces': set_pieces
    })

with_change = [c for c in classified if c['has_change']]
without_change = [c for c in classified if not c['has_change']]

print(f"Total gap clips analyzed: {len(classified)}")
print(f"Gaps WITH set piece / restart / state change after gap: {len(with_change)}")
print(f"Gaps WITHOUT set piece / restart / state change after gap: {len(without_change)}")

print("\n=======================================================")
print("GAPS WITH SET PIECE OR STATE CHANGE AFTER GAP")
print("=======================================================")
for c in with_change:
    print(f"{c['filename']} (ATD {c['start_id']}..{c['end_id']}, length {c['length']}):")
    for r in c['reasons']:
        print(f"   -> {r}")

print("\n=======================================================")
print("GAP FILES WITHOUT SET PIECE OR STATE CHANGE (IN-PLAY OCCLUSIONS)")
print("=======================================================")
for c in without_change:
    print(f"{c['filename']}")
