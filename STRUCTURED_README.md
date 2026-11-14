# Structured README — Supa Strikers, More Than a Hack 2027

What every folder and file in this repository is for, how they fit together, and how to run them.
The organisers' own README.md, CHALLENGE.md, DATA.md, … describe the challenge; this file describes
**our** code.

**Status keys:** ✅ written and tested · ⬜ placeholder (empty, to be filled) · ⚠️ needs attention · 🔒 organiser file, do not edit

---

## 1. The task in one paragraph

For each of the 199,319 frames of the tactical video of FC Barcelona vs Athletic Club, write one
label: `H` (home has the ball), `A` (away has it), `D` (dead ball) or `X` (no answer). The board
ranks on **spellF1**, which scores possessions, not frames: a hand-over written late, a possession
split in two or a missed stoppage all cost a lot. Our pipeline turns **our own player tracking**
(YOLO detection + tracking on the tactical video, `kaggle/`), the **fixed ball** (the provided ball,
spikes removed and short gaps filled; missing on ~41 % of graded frames) and Smart Tagging into
those labels. Since Nov 14 the provided ATD **player** tracks are not used by the pipeline.

## 2. Quick start

Run everything from the repository root.

```bash
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
```

Data: the match bundle goes in `data/tactical/` (see DATA.md); our Kaggle outputs in
`data/kaggle_outputs/` (`final_tracks_full.parquet`, `raw_dets_full_*/`, `calib/`, `eval/`); the bench
sample games in `data/external/metrica_sample/g1 g2 g3`. All git-ignored.

## 3. How the pieces fit

```
tactical video ──► kaggle/kaggle_player_tracking.ipynb (YOLO + tracker, T4 teams, GPU)
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
 bench              players, reconstruct      pitch control, shape, camera   │
        │                  │                        │                        │
        │                  ▼                        ▼                        │
        │          possession/detectors      possession/gaps                 │
        │           control, dead_ball,       find, classify                 │
        │           gap_fill, tag_prior,                                     │
        │           throw_in_side                                            │
        │                  │                                                 │
        │                  ▼                                                 ▼
        │          possession/model ◄──────────────────────── features/build (table)
        │           resolve (state machine), hmm (Viterbi), classifier
        │                  │
        ▼                  ▼
            possession/labellers  (baseline, v1-v7, v3c)
                           │
            ┌──────────────┴──────────────┐
            ▼                             ▼
   scripts/run.py (BAR-ATH)       scripts/bench_score.py (bench)
   -> outputs/submissions/        -> the ONLY place versions are compared
```

## 4. Rules that shape the code (RULES.md)

1. **Never tune on BAR-ATH or on the public 5 minutes.** Tune on the bench (other games). The
   public window is a format / sanity check only.
2. **No hand labels**, not even "checking clips and correcting a setting".
3. **No match data or derived data in git.** Everything generated goes to `outputs/` (ignored).
4. **The submission must be reproducible** from this repository + the data bundle.

---

## 5. Root files

| File | Status | Purpose |
|---|---|---|
| `README.md`, `CHALLENGE.md`, `DATA.md`, `EVALUATION.md`, `FAQ.md`, `OVERVIEW.md`, `RULES.md`, `SUBMISSION.md` | 🔒 | The organisers' description of the challenge, data, metric, rules and submission |
| `CITATION.cff`, `LICENSE` | 🔒 | Citation and licence of the organisers' code |
| `games.py` | 🔒 | Per-game constants: file paths, frame counts, half boundaries, pitch size. Everything imports it |
| `pyproject.toml`, `uv.lock`, `.python-version` | | Project environment (Python ≥ 3.13). Build the bench in this environment: other numpy/pandas versions give different bench files |
| `requirements.txt`, `requirements-starter.txt` | 🔒 | Organisers' minimal requirements |
| `.gitignore` | | Ignores `data/`, `outputs/`, `*.npy`, `*.mp4`, … Keep `outputs/*` in it |
| `STRUCTURED_README.md` | ✅ | This file |
| `kaggle/kaggle_player_tracking.ipynb` | ✅ | Our player detection + tracking on Kaggle (T4 ×2): YOLOv8 player model, ByteTrack / pitch-metre tracker, team classifiers T1–T4 (T4 = home / away / official), officials / goalkeepers / duplicates fixed per track, label-free scorecard. Output → `data/kaggle_outputs/`. Plan and decisions: `docs/player_tracking_plan.md` |
| `ball_flights.csv`, `in_air.npy`, `viz_11185_events.csv`, `viz_11185_report.json` | ⚠️ | Old derived match data. Must never be pushed; move to `outputs/` or delete |

## 6. Organiser folders (🔒 do not edit)

| Path | Purpose |
|---|---|
| `evaluation/possession_metrics.py` | The official metric (spellF1, macroF1, validation). Our scripts call it |
| `evaluation/run_evaluation.py` | Scores every CSV in a folder. Run as `python -m evaluation.run_evaluation …` (its import breaks when run as a file) |
| `evaluation/validate_submission.py` | Checks a submission file's format |
| `evaluation/gt/` | Public ground truth: 5 minutes (frames 11185–18684), sanity check only |
| `evaluation/results/`, `evaluation/reports/`, `evaluation/scoreboard.csv` | Published baseline results; reports / scoreboard are regenerated by `run_evaluation` (use `--out outputs/evaluation` instead) |
| `starter/` | Organisers' loaders (`load_atd`, `load_homography`, `load_smart_tagging`), the baseline, plotting helpers, notebooks. We still use `load_homography` (and `baseline.team_of_each_track` in the ATD diagnostics); `load_atd` is no longer used for players |
| `data/` | Match bundle (`tactical/`, `broadcast/`), our `kaggle_outputs/` and `external/metrica_sample/` for the bench — git-ignored |

---

## 7. `config/`

| File | Status | Purpose |
|---|---|---|
| `default.yaml` | ✅ | Every threshold, grouped by module (`ball`, `players`, `zoi`, `control`). Untuned starting points; change only with a bench score and a changelog entry |
| `bench.yaml` | ✅ | Bench source / output folders, games, default variant and labellers, noise seed — defaults for `bench_build.py` / `bench_score.py` |

Read with `possession.utils.load_config()` → `section(cfg, "ball")` → `BallConfig.from_dict(...)`.

---

## 8. `possession/` — our package

Every module is pure numpy on arrays in metres and runs on BAR-ATH and on the bench alike. On
BAR-ATH the arrays come from `possession.io.load()`: our YOLO players + the fixed ball.

### 8.1 `possession/io/` — reading inputs ✅

| File | Purpose | Main API |
|---|---|---|
| `__init__.py` | Puts the repo root on `sys.path` (so `games`, `starter` import anywhere); re-exports the API | — |
| `tracking.py` | **The one input**: `load()` builds a `Tracking` from our YOLO players (`detections.py`) + the fixed ball (`outputs/ball/ball_reconstructed.npz`; raw ATD ball with a note if missing). Speeds `ps`/`bs` are NaN. The raw ATD reader stays for the ball only | `load(lo, hi, players_file=, team_mode="local", clean=True, ball_file=, refresh=False) -> Tracking`, `load_ball()` (raw ATD ball), `graded_blocks()`, `graded_mask()` |
| `detections.py` | Our YOLO tracks → labeller slots: referees dropped, team per row from a ±0.5 s window of the T4 score (`local`) or the track vote (`track`), each track cut where its team changes, slots reused least-recently and never on consecutive frames, then cleaned (jump cuts, duplicate runs; no gap filling). Cached in `outputs/cache/detected_*.npz` | `load_detected()`, `build_slots()`, `clean_slots()`, `overflow_pct()`, `read_rows(path, lo, hi)` (boxes incl. referees, for drawing) |
| `teams.py` | Club names and team ids from the metadata XML (used for our slots and for Smart Tagging). Its ATD slot → team map is no longer used by the pipeline | `load_teams() -> TeamMap` (`.home`, `.away`, `.names`, `.side_of_name()`) |
| `tags.py` | Smart Tagging as typed records: case-insensitive, team → HOME/AWAY/None, half → 1/2 | `load_tags()`, `select()`, `set_pieces()`, `active_at()`, `frame_mask()`, `team_votes()` |
| `bench.py` | The test bench (v2, Ashwyn Nov 7): Metrica's public sample games, ground truth from their events, simulated set-piece tags, copies `degraded`, `hard`, `tracked`, `tracked_all` and the realistic `real` / `real_occ` (noise measured on BAR-ATH) | `build_bench()`, `build_real()`, `load_bench_game()`, `iter_bench()`, `gt_from_events()`, `degrade()`, `degrade_real()`, `simulate_tags()` |
| `dfl.py` | Parked placeholder for the DFL Bundesliga data (the bench replaced it) | raises `NotImplementedError` |

### 8.2 `possession/preprocess/` — signal clean-up ✅

| File | Purpose | Main API |
|---|---|---|
| `ball.py` | Ball track: remove spikes and teleports, bridge gaps ≤ 6 frames, speed, touch signal, outside-pitch flag, distance to lines | `clean_ball(bx, by) -> BallTrack`, `BallConfig` |
| `players.py` | Player tracks: smoothing, velocities (jumps > 12 m/s = identity switch), mean speeds, centroids, attacking direction | `prepare_players() -> PlayerState`, `attack_signs()`, `to_attacking()` |
| `fix_players.py` | Slot clean-up: cut slots at jumps (> 12 m/s over 0.2 s), blank duplicate runs (one person tracked twice); bridge / interpolate steps kept from the ATD era but off for YOLO slots | `fix_players(px, py, team, yolo=None, cfg=FixConfig(...))`, `FixConfig` |
| `reconstruct.py` | Fill ball positions inside a gap (linear / Hermite / quadratic, from Krish), test them by hiding known stretches, and the **prepared ball** (spikes out, gaps <= 25 frames Hermite-filled unless the ends imply > 35 m/s) stored by `scripts/reconstruct_ball.py` | `reconstruct()`, `mask_and_score()`, `prepare_ball()`, `save_ball_file()`, `load_ball_file()` |

### 8.3 `possession/utils/` — shared helpers ✅

| File | Purpose |
|---|---|
| `frames.py` | Label codes (H=0, A=1, D=2, X=3, as the scorer), `runs`, `label_runs`, `to_codes` / `to_letters`, `clock`, `half_of` |
| `config.py` | `load_config()` / `section()`; uses PyYAML if installed, a built-in reader otherwise |

### 8.4 `possession/features/` — per-frame measurements

| File | Status | Purpose |
|---|---|---|
| `zoi.py` | ✅ | Zone of influence: an ellipse around each player stretched along their velocity; `zoi_q` (≤ 1 = ball inside) |
| `relational.py` | ✅ | Nearest / second-nearest per team, gap between teams, closing speed, pressure counts (3 / 5 / 10 m) |
| `time_to_ball.py` | ✅ | Time for each player to reach the ball (0.7 s reaction, 7 m/s), fastest per team |
| `pitch_control.py` | ✅ | P(home wins the ball) at the ball, and a whole-pitch grid for one frame |
| `shape.py` | ✅ | Team centroid, spread, length, width, defensive line, push-forward speed (goalkeeper excluded) |
| `camera.py` | ✅ | Where the tactical camera points (homography) and its pan speed — BAR-ATH only; weak position, possible stoppage cue |
| `flight.py` | ✅ | Ball flights: `flight_mask()` (kinematic, any data, used by v5) and `detect_flights()` (+ height fit, BAR-ATH). Fixed Nov 9 (`docs/changelog/flights.md`); height is mostly not observable |
| `build.py` | ✅ | Assembles all of the above into one table (62 columns, no labels); `BENCH_COLUMNS` = model inputs; `save_table` / `load_table` / `to_matrix` |

Column meanings: `docs/features.md`.

### 8.5 `possession/gaps/` — missing-ball stretches ✅

| File | Purpose |
|---|---|
| `find.py` | `find_gaps()` per half (censored gaps flagged), `bracket()` (where the ball vanished / reappeared), `length_histogram()` |
| `classify.py` | `classify_gaps()`: dead_kickoff / dead_tagged / dead_untagged / live, team before / after, flip, set-piece type and team, phase team; `class_table()` |

### 8.6 `possession/detectors/` — evidence per frame

| File | Status | Purpose |
|---|---|---|
| `control.py` | ✅ | Who controls the ball: `distance` rule (Ashwyn, default) or `zone` rule (zone of influence). Output HOME / AWAY / contest / nobody. Identical to the original code |
| `dead_ball.py` | ✅ | Dead-ball mask v1: out of pitch, gaps near a line or with slow players until the restart, joined stoppages; reason per frame |
| `stoppage.py` | ✅ | Dead-ball mask v2 (Ashwyn's v2.1 / v4): speeds relative to the match's own pace, rules A out / B gaps / C slow (off in v4) / D set-piece tags, restart on a strike |
| `gap_fill.py` | ✅ | Moves a hand-over into a live gap (end / fixed / evidence). Tested: did not help on the bench, default `end` |
| `tag_prior.py` | ✅ | Smart Tagging → per-frame home/away score; code directions checked against our control signal |
| `throw_in_side.py` | ✅ | Who takes each throw-in (tags say N/A): ball cue, then team-position cue |

### 8.7 `possession/model/` — from evidence to possessions

| File | Status | Purpose |
|---|---|---|
| `resolve.py` | ✅ | State machine: switch only after 6 frames of the other team's control, backdate the switch, merge short spells. Optional `ctrl_sw` = v6's switch hysteresis. Used by v1 / v2 / v4–v6 |
| `hmm.py` | ✅ | H/A/D hidden-state model decoded with Viterbi (switch and dead penalties). Used by v3 / v3c |
| `classifier.py` | ✅ | Per-frame P(H/A/D) from the feature table; numpy logistic regression (scikit-learn optional); trained on the bench only; `train_on_bench()` |

### 8.8 `possession/labellers/` — complete recipes ✅

All share `label(px, py, ball, team, blocks, params=None, *, fps, pitch)`. Pick by name with
`get_labeller(name)` or `--labeller name`.

| Name | File | What it is | Bench (degraded) spellF1 |
|---|---|---|---|
| `baseline0/25/50/125` | `baseline.py` | Nearest player + majority window (the organisers' floor) | 0.32 (window 50) |
| `v1` | `v1.py` | Ashwyn's labeller on the shared modules — **safe submission** | 0.5226 |
| `v2` | `v2.py` | v1 + gap fill (+ optional Smart Tagging); default identical to v1 | 0.5226 |
| `v3` | `v3.py` | v1 signals decoded by the HMM | 0.5056 |
| `v3c` | `v3.py` | HMM + bench-trained classifier — best on the OLD bench; needs `outputs/models/classifier.npz` | 0.5447 (held-out games) |
| `v4` | `v4.py` | Ashwyn's v4: v1 control at 2.0 m + `detectors/stoppage.py`, takes Smart Tagging set pieces. Identical to `backup/ashwyn_labeller_v4.py` | bench v2 R+O 0.4889 |
| `v5` | `v5.py` | v4 + prepared ball + flights (opponent masked) + zone-of-influence veto + CONFIRM 15. `--set CONFIRM=6` = best R+O | R+O 0.4450 (CONFIRM 6: 0.4909) |
| `v6` | `v6.py` | Ashwyn's v6 (his safe submission): v4 + position smoothing ±3 frames + 0.6 m margin to take the ball + stoppages < 2 s dropped. Identical to `backup/ashwyn_labeller_v6.py` (checked Nov 10). Not related to our `v5` | R+O 0.5101 |

Bench scores for `baseline`…`v3c` are on the first bench (`outputs/bench_pre_nov7/`); `v4`/`v5` on
bench v2 (spellF1, R+O = mean of `real` and `real_occ`). Full history and settings tried:
`docs/changelog/labeller.md`.

### 8.9 `possession/events/` — football events ✅

| File | Status | Purpose |
|---|---|---|
| `schema.py` | ✅ | JSON Schema of the 14 event types (safe to share, no data) |
| `detect.py` | ✅ | Pass, carry, hold, duel, tackle, interception, possession change, restart, out of play, shot, goal, foul candidate — output identical to the old `event_detection.py` |
| `passes.py` | ✅ | Pass detection on ball flights (Ashwyn) — published match stats, `--real` and `--sweep` removed. `main()` reads `possession.io.load()` (YOLO players + fixed ball). Defaults still to re-derive on other games |

---

## 9. `scripts/` — command lines

| File | Status | Purpose |
|---|---|---|
| `_bootstrap.py` | ✅ | Makes `python scripts/x.py` and `python -m scripts.x` both work |
| `run.py` | ✅ | Run a labeller on BAR-ATH (YOLO players + fixed ball) → `outputs/submissions/<name>.csv`, validate, summarise, public-window sanity check. `--labeller`, `--out` (always set it: the default overwrites `<labeller>.csv`), `--set KEY=VALUE`, `--team-mode local/track`, `--raw-players`, `--players-file`, `--ball-file` / `--no-ball-file`, `--no-cache`, `--bench-pitch`, `--compare other.csv` |
| `build_tracking.py` | ✅ | Build the one input once (YOLO players packed + cleaned, fixed ball) → `outputs/cache/yolo_tracking.npz` + `_summary.json`; run after a new Kaggle run. `--rebuild`, `--team-mode`, `--raw-players` |
| `eval_tracking.py` | ✅ | Label-free checks of our tracking on the graded halves (people per frame, coverage 20–23, team overflow, jumps, fragmentation, ball, referees, visible goalkeepers) + the worst minutes → `outputs/track_eval/`. `--raw-players`, `--team-mode` |
| `bench_build.py` | ✅ | Build the bench from `data/external/metrica_sample` → `outputs/bench/`. `--no-degrade`, `--games`, `--kloppy`, `--real-only` |
| `bench_score.py` | ✅ | Score labellers on the bench (simulated tags passed to labellers that take them). `--labeller v4 v5`, `--variant all / decision`, `--set KEY=VALUE`, `--no-tags`, `--quiet`; prints R+O |
| `build_features.py` | ✅ | Feature table for BAR-ATH or `--bench g1` → `outputs/features/` |
| `train_classifier.py` | ✅ | Train the classifier on the bench → `outputs/models/classifier.npz` |
| `classify_gaps.py` | ✅ | One row per ball gap with its class and teams → `outputs/gaps/gaps_classified.csv`; `--no-tags`, `--set` |
| `detect_events.py` | ✅ | Events JSON + schema on YOLO players + fixed ball → `outputs/events/` (merged with the old `fix_events.py`: ball-gap bridge off on the fixed ball, confidence on tracker-seen ball frames only). `--full`, `--start/--seconds`, `--validate`, `--schema-only`, `--compare-events old.json`, `--keep-max-gap` |
| `reconstruct_ball.py` | ✅ | Prepared ball (spikes out, short plausible gaps filled) → `outputs/ball/ball_reconstructed.npz`; read by `run.py` (v5) and `detect_flights.py`. `--max-gap`, `--method`, `--max-speed`, `--csv` |
| `detect_flights.py` | ✅ | Ball flights + in-air / in-flight flags → `outputs/flights/` (needs scipy); uses the prepared ball |
| `detect_passes.py` | ✅ | Passes JSON / CSV → `outputs/passes/` |
| `detect_features.py`, `evauate_dfl.py`, `make_dfl_testset.py` | ⚠️ | Deprecated shims that forward to `build_features.py` / `bench_score.py` / `bench_build.py` — delete |

## 10. `viz/` — videos and plots ✅

Needs opencv-python and ffmpeg (or imageio-ffmpeg). Videos go to `outputs/videos/`.

| File | Purpose |
|---|---|
| `common.py` | Shared clip readers: `read_window()` (YOLO players + fixed ball, the default), `read_tracking()` (raw ATD, diagnostics / raw ball only), homography, GT, tags; video reader, pitch panel, drawing helpers |
| `render_match.py` | Tactical video + top-down pitch + events, side by side, YOLO players (`--start 07:27 --duration 60`, `--event-id`, `--no-video`, `--new-ball`) |
| `visualize_labeller.py` | A labeller CSV on the video: label border and badge, YOLO players, top view, change banners, timeline; `_spells.csv` and a drawn-vs-CSV check. `--compare` (diagnostics) = five panels: original \| video + ATD \| video + YOLO / top ATD \| top YOLO \| info, with `--labels2`. From Ashwyn's branch, adapted |
| `track_jumps.py` | Every jump (teleport / swap / reappear) in our YOLO tracks (`--source yolo`, default), in the cleaned slots (`slots`) or the ATD (`atd`, diagnostics), cross-checked against the other source, with the `visualize_labeller` command to watch each → `outputs/track_audit/` |
| `render_gaps.py` | One clip per long ball gap (`--min-length`, `--max-clips`, `--dry-run`) |
| `flight_clips.py` | One clip per ball flight from `outputs/flights/ball_flights.csv` |
| `possession_definition.py` | What the public ground truth means: segment table, timeline PNG, optional clips → `outputs/possession_definition/` |
| `test_event_clips.py` | **Event test**: clips (event ±3 s) of the detected events, one folder per type (`restart/` and `foul_candidate/` split by subtype), via `render_match.py --event-id` → `outputs/test_events/` + `index.csv`, `summary.json`. Needs `scripts/detect_events.py --full` first. `--types`, `--per-type`, `--pick spread/first/low_conf/high_conf/random`, `--only`, `--focus-only`, `--dry-run` |
| `test_ball_clips.py` | **Ball-tracking test**: one clip (±75 frames) per stretch where the ball is still missing after cleaning (`not_tracked/`: lost / found / whole gap) and per correction the pipeline made (`corrected/`: removed detections, filled positions), with a per-frame state strip → `outputs/test_ball/` + `index.csv`, `summary.json`. `--pipeline v6/v4/prepared`, `--types`, `--only`, `--dry-run`, `--no-video` |

## 11. `experiments/` — frozen records ✅

| File | Purpose |
|---|---|
| `reconstruction_masking.py` | Krish's reconstruction comparison: Hermite best up to ~1 s (0.13 m vs 1.11 m linear), all methods drift by 5 s |

## 12. `tests/` — unit tests ✅

`test_yolo_io.py` (slot packing, team modes, clean-up, `read_rows`), `test_reconstruct.py`,
`test_flight.py`, `test_v5.py`, `test_v6.py`, `test_ball_clips.py`, `test_event_clips.py` — 29 tests.
Synthetic data only (no match data), so they run on a fresh clone with `pytest`, or one by one with
`python -m tests.test_yolo_io`. The empty placeholders were moved to `backup/` on Nov 14.

## 13. `docs/`

| File | Status | Purpose |
|---|---|---|
| `features.md` | ✅ | Dictionary of every feature column, with sanity checks |
| `changelog/labeller.md` | ✅ | Every labeller version and bench score (Ashwyn's v1 history + v2 / v3 / v3c) |
| `changelog/passes.md` | ⬜ | Pass-detection history (from `backup/ashwyn_pass_detection_CHANGELOG.md`, stats removed) |
| `changelog/flights.md` | ✅ | Flight-detection fixes (Nov 9) and their effect on BAR-ATH |
| `compliance.md` | ⬜ ⚠️ | Rule-compliance checklist — copy `backup/ashwyn_TESTING_ONLY_REMOVE_BEFORE_SUBMISSION.md` here and clear it before submitting |
| `research/ball_gaps.md`, `throw_in_side.md`, `ball_reconstruction.md` | ⬜ | Method notes from Krish's docs (no frame lists) |
| `log.md` | ⬜ | Daily "what changed + bench score" |
| `progress/ashwyn_progress_brief.md` | ✅ | Ashwyn's Nov 9 team brief (from his branch): rules, data findings, the three-truth bench, v1–v6 history, plan to Nov 17. Its file paths refer to his `test_scripts/` layout (here: `backup/ashwyn_*` and the package) |
| `player_tracking_plan.md` | ✅ | Plan, decisions and status of our player tracking (Kaggle notebook, variants, metrics, the switch to YOLO players on Nov 14) |
| `tech_doc.md` | ⬜ | The technical document required by SUBMISSION.md |
| `plans/Possession Challenge Game Plan.md` | | The game plan (to move to `docs/game_plan.md` with `git mv`) |

## 14. `backup/` — old code, reference only ⚠️

`backup/discarded_ATD_players_2026-11-14/` (same folder layout as the repo) holds what the switch to
YOLO players retired: `scripts/audit_data.py`, `eda_smart_tagging.py`, `fix_events.py` (merged into
`detect_events.py`), `experiments/possession_test_1.py`, `v5_ablation.py`, `v7_ablation.py`,
`viz/render_possession.py`, the empty tests and scratch helpers.

Flat copies of the code the package replaced: ours (`event_detection.py`, `ball_gap_analysis.py`,
`visualize_tracking.py`, …), Ashwyn's (`ashwyn_*`, updated Nov 9 from his branch: `labeller_v2/v3/v4`,
`run_labeller_v2/v3/v4`, `noise_profile`, bench builder / scorer; Nov 10: `labeller_v5/v6`,
`run_labeller_v6`, `build_trk_gt`, `fetch_dfl_bench`, `hutil`, changelog with v5 / v6) and Krish's (`krish_*`). Scripts there do not run
as they are. Use them to port the ⬜ modules, then remove the folder before the repository goes
public (`ashwyn_pass_detection.py` still holds published match statistics).

## 15. `outputs/` — everything generated (git-ignored)

| Folder | Written by |
|---|---|
| `cache/` | `possession/io/detections.py` (packed YOLO players, `detected_*.npz`) and `scripts/build_tracking.py` (`yolo_tracking.npz`). Old `atd_*.npz` files are unused |
| `track_eval/` | `scripts/eval_tracking.py` |
| `track_audit/` | `viz/track_jumps.py` |
| `labeller_viz/` | `viz/visualize_labeller.py` |
| `submissions/` | `scripts/run.py` |
| `evaluation/` | `python -m evaluation.run_evaluation --out outputs/evaluation` |
| `bench/` | `scripts/bench_build.py` (bench v2; the first bench is kept in `bench_pre_nov7/`) |
| `ball/` | `scripts/reconstruct_ball.py` |
| `flights/` | `scripts/detect_flights.py` |
| `features/` | `scripts/build_features.py`, `classifier.bench_tables()` |
| `models/` | `scripts/train_classifier.py` |

After changing loader code run `python scripts/build_tracking.py --rebuild` (the player cache
refreshes on its own only when the parquet file changes).

---

## 16. Before submitting

- [ ] `docs/compliance.md` cleared (published stats, spot-check tool, hand-verdict ideas removed)
- [ ] No data in git: `git ls-files | grep -E "npy|npz|csv|json|mp4"` shows only organiser files
- [ ] `backup/` and old root data files removed
- [ ] Bench rebuilt in the pinned environment; chosen labeller's score in the changelog
- [ ] `python scripts/run.py --labeller <chosen>` regenerates the submitted file exactly from a clean clone
- [ ] `docs/tech_doc.md` written
