p = "STRUCTURED_README.md"
s = open(p, encoding="utf-8").read()

def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (s.count(old), old[:80])
    s = s.replace(old, new)

def rep_block(start, end, new):
    """Replace from the line starting with `start` up to (not including) the line starting with `end`."""
    global s
    a = s.index(start)
    b = s.index(end, a)
    s = s[:a] + new + s[b:]

# 1. task paragraph
rep('''ranks on **spellF1**, which scores possessions, not frames: a hand-over written late, a possession
split in two or a missed stoppage all cost a lot. Our pipeline turns tracking data (players + ball,
ball missing ~41 % of graded frames) and Smart Tagging into those labels.''',
'''ranks on **spellF1**, which scores possessions, not frames: a hand-over written late, a possession
split in two or a missed stoppage all cost a lot. Our pipeline turns **our own player tracking**
(YOLO detection + tracking on the tactical video, `kaggle/`), the **fixed ball** (the provided ball,
spikes removed and short gaps filled; missing on ~41 % of graded frames) and Smart Tagging into
those labels. Since Nov 14 the provided ATD **player** tracks are not used by the pipeline.''')

# 2. quick start
rep_block("```bash\nuv sync", "```\n\nData: the match bundle", '''```bash
uv sync                                         # or: pip install -r requirements.txt (+ pyarrow)

# once per Kaggle run (kaggle/kaggle_player_tracking.ipynb -> data/kaggle_outputs/)
python scripts/reconstruct_ball.py              # fixed ball -> outputs/ball/ball_reconstructed.npz
python scripts/build_tracking.py                # YOLO players (packed + cleaned) + fixed ball -> outputs/cache/
python scripts/eval_tracking.py                 # label-free checks of our tracking -> outputs/track_eval/

# submissions
python scripts/run.py --labeller v6 --out outputs/submissions/v6.csv      # current candidate
python scripts/run.py --labeller v6 --out outputs/submissions/try.csv --compare outputs/submissions/v6.csv
python -m evaluation.run_evaluation --results outputs/submissions --out outputs/evaluation

# comparing labeller versions: ONLY on the bench (other games)
python scripts/bench_build.py                   # build the test bench (Metrica public games)
python scripts/bench_score.py --labeller v4 v6 --variant decision

# features, events, flights, passes (all on YOLO players + fixed ball)
python scripts/build_features.py
python scripts/detect_flights.py
python scripts/detect_events.py --full          # events -> outputs/events/events_full.json
python scripts/detect_passes.py

# looking at it (understand errors; never turn what you see into labels or settings)
python viz/visualize_labeller.py --range 73997 75081                     # labels on video, YOLO players
python viz/visualize_labeller.py --compare --range 73997 75081 --labels outputs/submissions/v6_atd.csv --labels2 outputs/submissions/v6.csv
python viz/track_jumps.py --source slots        # jumps left in the slots the labellers read
python viz/test_ball_clips.py --dry-run         # ball test clips
python viz/test_event_clips.py                  # event test clips
''')
rep('''Data: the match bundle goes in `data/tactical/` (see DATA.md); the bench sample games in
`data/external/metrica_sample/g1 g2 g3`. Both are git-ignored.''',
'''Data: the match bundle goes in `data/tactical/` (see DATA.md); our Kaggle outputs in
`data/kaggle_outputs/` (`final_tracks_full.parquet`, `raw_dets_full_*/`, `calib/`, `eval/`); the bench
sample games in `data/external/metrica_sample/g1 g2 g3`. All git-ignored.''')

# 3. diagram
rep('''data/ (match + bench)                     config/default.yaml (every threshold)
        │                                          │
        ▼                                          ▼
possession/io ──► possession/preprocess ──► possession/features ─────────────┐
 tracking, teams,   ball clean-up,            zoi, relational, time-to-ball, │
 tags, bench        players, reconstruct      pitch control, shape, camera   │''',
'''tactical video ──► kaggle/kaggle_player_tracking.ipynb (YOLO + tracker, T4 teams, GPU)
                          │  data/kaggle_outputs/final_tracks_full.parquet
                          ▼
possession/io/detections (slots, team per row, clean-up via preprocess/fix_players)
                          │        + fixed ball (outputs/ball, scripts/reconstruct_ball.py)
                          ▼
              possession/io/tracking.load()  ── THE input: Tracking(px, py, ball, teams)
                          │                         config/default.yaml (every threshold)
                          ▼                                        │
possession/io ──► possession/preprocess ──► possession/features ─────────────┐
 tags, teams,       ball clean-up,            zoi, relational, time-to-ball, │
 bench              players, reconstruct      pitch control, shape, camera   │''')
rep('''            possession/labellers  (baseline, v1, v2, v3, v3c)''',
    '''            possession/labellers  (baseline, v1-v7, v3c)''')

# 5. root files: add kaggle row
rep('''| `STRUCTURED_README.md` | ✅ | This file |''',
'''| `STRUCTURED_README.md` | ✅ | This file |
| `kaggle/kaggle_player_tracking.ipynb` | ✅ | Our player detection + tracking on Kaggle (T4 ×2): YOLOv8 player model, ByteTrack / pitch-metre tracker, team classifiers T1–T4 (T4 = home / away / official), officials / goalkeepers / duplicates fixed per track, label-free scorecard. Output → `data/kaggle_outputs/`. Plan and decisions: `docs/player_tracking_plan.md` |''')

# 6. organiser folders: starter / data
rep('''| `starter/` | Organisers' loaders (`load_atd`, `load_homography`, `load_smart_tagging`), the baseline, plotting helpers, notebooks. Our `possession/io` wraps the loaders |''',
'''| `starter/` | Organisers' loaders (`load_atd`, `load_homography`, `load_smart_tagging`), the baseline, plotting helpers, notebooks. We still use `load_homography` (and `baseline.team_of_each_track` in the ATD diagnostics); `load_atd` is no longer used for players |''')
rep('''| `data/` | Match bundle (`tactical/`, `broadcast/`) and `external/metrica_sample/` for the bench — git-ignored |''',
'''| `data/` | Match bundle (`tactical/`, `broadcast/`), our `kaggle_outputs/` and `external/metrica_sample/` for the bench — git-ignored |''')

# 8 intro
rep('''Every module is pure numpy on arrays in metres and runs on BAR-ATH and on the bench alike.''',
'''Every module is pure numpy on arrays in metres and runs on BAR-ATH and on the bench alike. On
BAR-ATH the arrays come from `possession.io.load()`: our YOLO players + the fixed ball.''')

# 8.1 io
rep('''| `tracking.py` | One reader for the ATD tracking file: players + ball in metres, tracker speeds, seek to any frame, cache in `outputs/cache/` | `load(lo, hi, cache=True) -> Tracking`, `load_ball()`, `graded_blocks()`, `graded_mask()` |
| `teams.py` | Team of each of the 46 track slots from the metadata. Unmapped slots are UNKNOWN (the old code made them AWAY) | `load_teams() -> TeamMap` (`.home`, `.away`, `.names`, `.side_of_name()`) |''',
'''| `tracking.py` | **The one input**: `load()` builds a `Tracking` from our YOLO players (`detections.py`) + the fixed ball (`outputs/ball/ball_reconstructed.npz`; raw ATD ball with a note if missing). Speeds `ps`/`bs` are NaN. The raw ATD reader stays for the ball only | `load(lo, hi, players_file=, team_mode="local", clean=True, ball_file=, refresh=False) -> Tracking`, `load_ball()` (raw ATD ball), `graded_blocks()`, `graded_mask()` |
| `detections.py` | Our YOLO tracks → labeller slots: referees dropped, team per row from a ±0.5 s window of the T4 score (`local`) or the track vote (`track`), each track cut where its team changes, slots reused least-recently and never on consecutive frames, then cleaned (jump cuts, duplicate runs; no gap filling). Cached in `outputs/cache/detected_*.npz` | `load_detected()`, `build_slots()`, `clean_slots()`, `overflow_pct()`, `read_rows(path, lo, hi)` (boxes incl. referees, for drawing) |
| `teams.py` | Club names and team ids from the metadata XML (used for our slots and for Smart Tagging). Its ATD slot → team map is no longer used by the pipeline | `load_teams() -> TeamMap` (`.home`, `.away`, `.names`, `.side_of_name()`) |''')

# 8.2 preprocess
rep('''| `reconstruct.py` | Fill ball positions inside a gap''',
'''| `fix_players.py` | Slot clean-up: cut slots at jumps (> 12 m/s over 0.2 s), blank duplicate runs (one person tracked twice); bridge / interpolate steps kept from the ATD era but off for YOLO slots | `fix_players(px, py, team, yolo=None, cfg=FixConfig(...))`, `FixConfig` |
| `reconstruct.py` | Fill ball positions inside a gap''')

# 8.9 events
rep('''| `passes.py` | ✅ | Pass detection on ball flights (Ashwyn) — published match stats, `--real` and `--sweep` removed; output identical otherwise. Defaults still to re-derive on other games |''',
'''| `passes.py` | ✅ | Pass detection on ball flights (Ashwyn) — published match stats, `--real` and `--sweep` removed. `main()` reads `possession.io.load()` (YOLO players + fixed ball). Defaults still to re-derive on other games |''')

# 9 scripts
rep('''| `run.py` | ✅ | Run a labeller on BAR-ATH → `outputs/submissions/<name>.csv`, validate, summarise, public-window sanity check. `--labeller`, `--out`, `--set KEY=VALUE`, `--ball-file` / `--no-ball-file`, `--no-cache`, `--bench-pitch` |''',
'''| `run.py` | ✅ | Run a labeller on BAR-ATH (YOLO players + fixed ball) → `outputs/submissions/<name>.csv`, validate, summarise, public-window sanity check. `--labeller`, `--out` (always set it: the default overwrites `<labeller>.csv`), `--set KEY=VALUE`, `--team-mode local/track`, `--raw-players`, `--players-file`, `--ball-file` / `--no-ball-file`, `--no-cache`, `--bench-pitch`, `--compare other.csv` |
| `build_tracking.py` | ✅ | Build the one input once (YOLO players packed + cleaned, fixed ball) → `outputs/cache/yolo_tracking.npz` + `_summary.json`; run after a new Kaggle run. `--rebuild`, `--team-mode`, `--raw-players` |
| `eval_tracking.py` | ✅ | Label-free checks of our tracking on the graded halves (people per frame, coverage 20–23, team overflow, jumps, fragmentation, ball, referees, visible goalkeepers) + the worst minutes → `outputs/track_eval/`. `--raw-players`, `--team-mode` |''')
rep('''| `audit_data.py` | ✅ | Data audit: ball-gap histogram, team mapping, homography, tagging coverage → `outputs/audit/` |
''', '')
rep('''| `detect_events.py` | ✅ | Events JSON + schema → `outputs/events/`; `--full`, `--start/--seconds`, `--validate`, `--schema-only` |''',
'''| `detect_events.py` | ✅ | Events JSON + schema on YOLO players + fixed ball → `outputs/events/` (merged with the old `fix_events.py`: ball-gap bridge off on the fixed ball, confidence on tracker-seen ball frames only). `--full`, `--start/--seconds`, `--validate`, `--schema-only`, `--compare-events old.json`, `--keep-max-gap` |''')
rep('''| `eda_smart_tagging.py` | ✅ | Smart Tagging counts, N/A teams, set pieces vs ball gaps, time-base check |
''', '')

# 10 viz
rep('''| `common.py` | Shared clip readers (tracking, homography, GT, tags), video reader, pitch panel, drawing helpers |
| `render_match.py` | Tactical video + top-down pitch + events, side by side (`--start 07:27 --duration 60`, `--event-id`, `--no-video`) |
| `render_possession.py` | The zone-of-influence possession test on video: zones, holder, contests, events, timeline |''',
'''| `common.py` | Shared clip readers: `read_window()` (YOLO players + fixed ball, the default), `read_tracking()` (raw ATD, diagnostics / raw ball only), homography, GT, tags; video reader, pitch panel, drawing helpers |
| `render_match.py` | Tactical video + top-down pitch + events, side by side, YOLO players (`--start 07:27 --duration 60`, `--event-id`, `--no-video`, `--new-ball`) |
| `visualize_labeller.py` | A labeller CSV on the video: label border and badge, YOLO players, top view, change banners, timeline; `_spells.csv` and a drawn-vs-CSV check. `--compare` (diagnostics) = five panels: original \\| video + ATD \\| video + YOLO / top ATD \\| top YOLO \\| info, with `--labels2`. From Ashwyn's branch, adapted |
| `track_jumps.py` | Every jump (teleport / swap / reappear) in our YOLO tracks (`--source yolo`, default), in the cleaned slots (`slots`) or the ATD (`atd`, diagnostics), cross-checked against the other source, with the `visualize_labeller` command to watch each → `outputs/track_audit/` |''')

# 11 experiments
rep('''| `possession_test_1.py` | The first zone-of-influence possession test (frozen; its logic is `detectors/control.py` method `zone`) |
''', '')
rep('''| `v5_ablation.py` | Every v5 change alone and together on bench v2 (table in `docs/changelog/labeller.md`) |
''', '')

# 12 tests
rep_block("## 12. `tests/`", "## 13. `docs/`", '''## 12. `tests/` — unit tests ✅

`test_yolo_io.py` (slot packing, team modes, clean-up, `read_rows`), `test_reconstruct.py`,
`test_flight.py`, `test_v5.py`, `test_v6.py`, `test_ball_clips.py`, `test_event_clips.py` — 29 tests.
Synthetic data only (no match data), so they run on a fresh clone with `pytest`, or one by one with
`python -m tests.test_yolo_io`. The empty placeholders were moved to `backup/` on Nov 14.

''')

# 13 docs: add plan doc
rep('''| `tech_doc.md` | ⬜ | The technical document required by SUBMISSION.md |''',
'''| `player_tracking_plan.md` | ✅ | Plan, decisions and status of our player tracking (Kaggle notebook, variants, metrics, the switch to YOLO players on Nov 14) |
| `tech_doc.md` | ⬜ | The technical document required by SUBMISSION.md |''')

# 14 backup
rep('''## 14. `backup/` — old code, reference only ⚠️
''', '''## 14. `backup/` — old code, reference only ⚠️

`backup/discarded_ATD_players_2026-11-14/` (same folder layout as the repo) holds what the switch to
YOLO players retired: `scripts/audit_data.py`, `eda_smart_tagging.py`, `fix_events.py` (merged into
`detect_events.py`), `experiments/possession_test_1.py`, `v5_ablation.py`, `v7_ablation.py`,
`viz/render_possession.py`, the empty tests and scratch helpers.
''')

# 15 outputs
rep('''| `cache/` | `io.tracking.load(cache=True)` — parsed tracking (41 MB) |''',
'''| `cache/` | `possession/io/detections.py` (packed YOLO players, `detected_*.npz`) and `scripts/build_tracking.py` (`yolo_tracking.npz`). Old `atd_*.npz` files are unused |
| `track_eval/` | `scripts/eval_tracking.py` |
| `track_audit/` | `viz/track_jumps.py` |
| `labeller_viz/` | `viz/visualize_labeller.py` |''')
rep('''Delete `outputs/cache/` after changing loader code; it refreshes on its own only when the raw file changes.''',
'''After changing loader code run `python scripts/build_tracking.py --rebuild` (the player cache
refreshes on its own only when the parquet file changes).''')

open(p, "w", encoding="utf-8", newline="\n").write(s)
print("README patched")
