# Project Overview & Dataset Architecture

This repository contains the codebase and EDA for the football possession tracking
competition ("More Than A Hack 2027"). This file mirrors the project memory so the key
facts are reachable from inside the `eda/` folder without consulting memory.

## Match & Data Specs
- **Match**: Barcelona vs Athletic Club (`BAR-ATH`).
- **FPS**: 25.0 Hz. Total video frames: 199,319 (~133 mins).
- **Pitch Dimensions**: 105.0m x 68.0m (center at 0,0).
- **Video frame to ATD frame**: `ATD frame_id = video_frame + 1` (1-based index).
- **Graded Blocks**: Graded play blocks defining valid possession periods in half 1 and
  half 2 (excluding pre-game, halftime, post-game).

Graded play (video frames), from `games.py` → `half_video_blocks`:
- **Half 1**: video frames **11185 → 90651** (kickoff at frame 11186 = 07:27)
- **Half 2**: video frames **110808 → 190197** (kickoff at frame 113741 = 75:49)

Anything before 11185 or between 90651 and 110808 is **not graded** (pre-game / halftime).

## Data Sources & Tools
- `games.py`: Game config, file paths resolver (`require`), half video frame boundaries.
- `starter/baseline.py`: Nearest-player-to-ball baseline model assigning possession
  (`BAR`, `ATH`, or `D`).
- `starter/load_atd.py`: Binary search & parsing for raw ATD tracking data (46 player
  slots + 1 ball slot).
- `starter/load_smart_tagging.py`: Event annotations (Set Pieces, Transitions, In-Box
  phases, Team tags).
- `eda/visualize_tracking.py`: Dual-panel visualizer (tactical video + pitch top-down)
  with lazy streaming via ffmpeg pipe.

## Smart Tagging schema (638 instances)
Codes: `SET PIECES` (59), `PLAYERS IN THE BOX` (97), `BALL IN FINAL THIRD` (93),
`BALL IN THE BOX` (72), `PROGRESSION` (66), `DEFENDING IN MIDDLE THIRD` (66),
`ATTACKING TRANSITION` (24), `DEFENSIVE TRANSITION` (24), `BUILD UP` (22),
`DEFENDING IN ATTACKING THIRD` (22), `DEFENDING IN DEFENSIVE THIRD` (30),
`CREATING CHANCES` (30), `LONG BALL` (29), `GOALS` (4).

Label groups: `Team` (Barcelona 314 · Atletic Club 288 · **N/A 36**), `Half`
(`1st Half` · `2nd Half` — strings), `Type` (Throw-in 26 · Corner Kick 12 · Goal Kick
10 · Free Kick 5 · Kick Off (After Goal) 4 · Kick Off (Start) 2), `Side`
(Central · Left · Right), `Direction of ball entry` (Horizontal · Vertical · Diagonal),
`Max Players in the box` (1…6, and `7+` — a **string**, not an integer).

The converted table is at `data/tactical/BAR-ATH_tactical_smart_tagging_csv.csv`
(columns: `ID, start, end, code, Team, Type, Max Players in the box, Side,
Direction of ball entry, Half`). `start`/`end` are **seconds from the first frame of the
tactical video** (0-based), so `round(seconds * 25)` is a video frame.

## ATD tracking (46 player slots + 1 ball slot)
- Slots 1–24 = **Barcelona** (home), slots 25–46 = **Atletic Club**
  (`baseline.py:91-100` reads this from the `<Player teamId="..."/>` entries).
- **No goalkeeper flag** — no `<Goalkeeper>`, no role/position field, no ShirtNumber.
  Slots are anonymous "Track 1"…"Track 46".
- Per-team pitch orientation is in the provider params:
  `attack_direction_first_half` = `left_to_right` for Barcelona, `right_to_left` for
  Atletic Club. Verified on the tracking: at 1st-half kickoff (ATD id 11187) BAR mean
  x = 0.38, ATH = 0.52; at 2nd-half kickoff (id 113742) they swap to 0.62 / 0.44.
- Raw line format: `<frame_id>:<player>;<player>;...:<ball>`, each group
  `x,y,speed` with `NaN,NaN,NaN` for a missing observation. The ball is the segment
  after the **last** colon.

## Standing constraint (session rule)
**Do not commit anything from this project** — no `git add`, no commits, no pushes, no
PRs. Read/analyze/write only. The repo's working tree is left untouched; the only
committed state is the snapshot from the start of the session (the `data/README.md`
deletion and untracked `eda/` are pre-existing, not ours).

## Related EDA files
- [`GAPS_AND_CLIPS.md`](GAPS_AND_CLIPS.md) — ball-tracking gap taxonomy and gap clips
- [`THROW_IN_SIDE.md`](THROW_IN_SIDE.md) — recovering the missing throw-in team tag
- [`thoughts.md`](thoughts.md) — running project log