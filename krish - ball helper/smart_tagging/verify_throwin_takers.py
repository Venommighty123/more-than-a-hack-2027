"""Independent re-derivation of the throw-in taker inference from raw ATD.

Reads the raw tracking file and the smart-tagging XML, recomputes the
centroid-in-own-half score for every throw-in, and compares against
eda/throw_in_takers.csv. No shared code with the original script.
"""
import csv
import xml.etree.ElementTree as ET

RAW = 'data/tactical/BAR-ATH_tactical_atd.txt'
TAG = 'data/tactical/BAR-ATH_tactical_smart_tagging.xml'
BAR_SLOTS = set(range(1, 25))
ATH_SLOTS = set(range(25, 47))
# own-half direction: positive score means centroid is in that team's own half
OWN_DIR = {('BAR', 1): -1.0, ('ATH', 1): +1.0,
           ('BAR', 2): +1.0, ('ATH', 2): -1.0}

# ---- stream the raw ATD into a dict: frame_id -> (players, ball) ----
idx = {}
with open(RAW) as f:
    for line in f:
        fid = int(line[:line.index(':')])
        parts = line.split(':')
        players = []
        for slot, group in enumerate(parts[1].split(';'), start=1):
            if group.startswith('N'):
                continue
            x, y = group.split(',')[:2]
            players.append((slot, float(x), float(y)))
        b = parts[2].strip()
        ball = None if b.startswith('N') else (float(b.split(',')[0]), float(b.split(',')[1]))
        idx[fid] = (players, ball)

def centroid(players, slots, half, team):
    """Non-GK centroid x; GK = player whose y is closest to a goal line (0 or 1)."""
    pts = [p for p in players if p[0] in slots]
    if len(pts) < 3:
        return None
    gk = max(pts, key=lambda p: min(abs(p[2] - 0.0), abs(p[2] - 1.0)))
    rest = [p for p in pts if p is not gk]
    if len(rest) < 2:
        return None
    cx = sum(p[1] for p in rest) / len(rest)
    return cx

# ---- throw-ins from the tagging XML ----
tree = ET.parse(TAG)
throws = []
for i in tree.getroot().find('ALL_INSTANCES').findall('instance'):
    lbls = {l.find('group').text: l.find('text').text for l in i.findall('label')}
    if i.find('code').text == 'SET PIECES' and lbls.get('Type') == 'Throw-in':
        throws.append((int(i.find('ID').text), float(i.find('start').text),
                       float(i.find('end').text), lbls.get('Half')))

# ---- compare ----
saved = {int(r['smart_id']): r for r in csv.DictReader(open('eda/throw_in_takers.csv'))}

print('%-5s %-8s %-9s %-6s %-6s  %8s %8s  %8s %8s  %s' % (
    'ID', 'Half', 'frame', 'BARcx', 'ATHcx', 'BARown', 'ATHown', 'BARsc', 'ATHsc', 'taker'))
mismatch = 0
for sid, start, end, half in throws:
    vid = int(round(start * 25))
    frame_id = vid + 1
    players, ball = idx.get(frame_id, ([], None))
    h = 1 if half == '1st Half' else 2
    bar_cx = centroid(players, BAR_SLOTS, h, 'BAR')
    ath_cx = centroid(players, ATH_SLOTS, h, 'ATH')
    bar_sc = OWN_DIR[('BAR', h)] * (bar_cx - 0.5) if bar_cx is not None else None
    ath_sc = OWN_DIR[('ATH', h)] * (ath_cx - 0.5) if ath_cx is not None else None
    if bar_sc is None or ath_sc is None:
        taker = 'MISSING'
    else:
        taker = 'BAR' if bar_sc > ath_sc else 'ATH'
    s = saved.get(sid)
    ok = s is not None and s['inferred_taker'] == taker
    if not ok:
        mismatch += 1
    print('%-5d %-8s %-9d %-6s %-6s  %8s %8s  %8s %8s  %s%s' % (
        sid, half, frame_id,
        ('%.3f' % bar_cx) if bar_cx is not None else '  n/a',
        ('%.3f' % ath_cx) if ath_cx is not None else '  n/a',
        ('%+.3f' % bar_sc) if bar_sc is not None else '    n/a',
        ('%+.3f' % ath_sc) if ath_sc is not None else '    n/a',
        ('%+.3f' % bar_sc) if bar_sc is not None else '     0',
        ('%+.3f' % ath_sc) if ath_sc is not None else '     0',
        taker, '' if ok else '   <-- MISMATCH vs saved=%s' % (s['inferred_taker'] if s else '?')))

print('\n%d/%d match saved CSV; %d mismatch' % (26 - mismatch, 26, mismatch))