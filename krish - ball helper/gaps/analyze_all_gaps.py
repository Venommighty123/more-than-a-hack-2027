import os, sys, glob
sys.path.insert(0, '.')
from eda.eda_ball_gaps import iter_ball_present, find_gaps
from games import GAME, DEFAULT_SOURCE, require
from starter.load_smart_tagging import load as load_tags

# Load Data
raw_path = require(['atd_raw'], GAME, None, DEFAULT_SOURCE)[0]
present = list(iter_ball_present(raw_path))
all_gaps = find_gaps(present)

tags = load_tags()
# Group tags by code
event_codes = {}
for t in tags:
    code = t['code']
    if code not in event_codes:
        event_codes[code] = []
    event_codes[code].append(t)

# Find gap clips
gap_files = glob.glob('gap_clips/gap_*.mp4')
results = []

for f in gap_files:
    fname = os.path.basename(f)
    parts = fname.replace('gap_', '').replace('.mp4', '').split('_')
    v0 = int(parts[0])
    length = int(parts[1])

    start_id = v0 + 51
    end_id = start_id + length - 1

    # Check for set pieces starting in [end_id, end_id + 75]
    restart_events = []
    for code, events in event_codes.items():
        if 'set piece' in code.lower() or 'goal' in code.lower():
            for e in events:
                # Check if event starts within [end_id, end_id + 75]
                # Allow a small buffer before (end_id - 5)
                if (end_id - 5) <= e['start_frame'] <= (end_id + 75):
                    restart_events.append({'code': code, 'type': e['tags'].get('Type', 'N/A'), 'start': e['start_frame']})

    has_restart = len(restart_events) > 0
    results.append({
        'filename': fname,
        'has_restart': has_restart,
        'events': restart_events,
        'end_id': end_id
    })

# Output
print('--- Gaps WITH restart / set-piece change ---')
with_restart = [r for r in results if r['has_restart']]
for r in with_restart:
    event_str = ', '.join([f"{e['code']} ({e['type']})" for e in r['events']])
    print(f'{r["filename"]}: {event_str}')

print('\n--- Gaps WITHOUT restart / set-piece change ---')
without_restart = [r for r in results if not r['has_restart']]
for r in without_restart:
    print(f'{r["filename"]}')
