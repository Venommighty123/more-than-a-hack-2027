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
split in two or a missed stoppage all cost a lot. Our pipeline turns tracking data (players + ball,
ball missing ~41 % of graded frames) and Smart Tagging into those labels.

## 2. Quick start

Run everything from the repository root.

```bash
uv sync                                         # or: pip install -r requirements.txt
python scripts/run.py                           # labeller v1 -> outputs/submissions/v1.csv (validated)
python -m evaluation.run_evaluation --results outputs/submissions --out outputs/evaluation

python scripts/bench_build.py                   # build the test bench (Metrica public games)
python scripts/bench_score.py --variant all     # compare labellers on the bench
python scripts/build_features.py                # per-frame feature table for BAR-ATH
python scripts/train_classifier.py              # train the classifier (bench only) for v3c
python scripts/run.py --labeller v3c            # submission from the classifier + HMM labeller
```

Data: the match bundle goes in `data/tactical/` (see DATA.md); the bench sample games in
`data/external/metrica_sample/g1 g2 g3`. Both are git-ignored.

## 3. How the pieces fit

```
data/ (match + bench)                     config/default.yaml (every threshold)
        │                                          │
        ▼                                          ▼
possession/io ──► possession/preprocess ──► possession/features ─────────────┐
 tracking, teams,   ball clean-up,            zoi, relational, time-to-ball, │
 tags, bench        players, reconstruct      pitch control, shape, camera   │
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
            possession/labellers  (baseline, v1, v2, v3, v3c)
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
| `ball_flights.csv`, `in_air.npy`, `viz_11185_events.csv`, `viz_11185_report.json` | ⚠️ | Old derived match data. Must never be pushed; move to `outputs/` or delete |

## 6. Organiser folders (🔒 do not edit)

| Path | Purpose |
|---|---|
| `evaluation/possession_metrics.py` | The official metric (spellF1, macroF1, validation). Our scripts call it |
| `evaluation/run_evaluation.py` | Scores every CSV in a folder. Run as `python -m evaluation.run_evaluation …` (its import breaks when run as a file) |
| `evaluation/validate_submission.py` | Checks a submission file's format |
| `evaluation/gt/` | Public ground truth: 5 minutes (frames 11185–18684), sanity check only |
| `evaluation/results/`, `evaluation/reports/`, `evaluation/scoreboard.csv` | Published baseline results; reports / scoreboard are regenerated by `run_evaluation` (use `--out outputs/evaluation` instead) |
| `starter/` | Organisers' loaders (`load_atd`, `load_homography`, `load_smart_tagging`), the baseline, plotting helpers, notebooks. Our `possession/io` wraps the loaders |
| `data/` | Match bundle (`tactical/`, `broadcast/`) and `external/metrica_sample/` for the bench — git-ignored |

---

## 7. `config/`

| File | Status | Purpose |
|---|---|---|
| `default.yaml` | ✅ | Every threshold, grouped by module (`ball`, `players`, `zoi`, `control`). Untuned starting points; change only with a bench score and a changelog entry |
| `bench.yaml` | ✅ | Bench source / output folders, games, default variant and labellers, noise seed — defaults for `bench_build.py` / `bench_score.py` |

Read with `possession.utils.load_config()` → `section(cfg, "ball")` → `BallConfig.from_dict(...)`.

---

## 8. `possession/` — our package

Every module is pure numpy on arrays in metres and runs on BAR-ATH and on the bench alike.

### 8.1 `possession/io/` — reading inputs ✅

| File | Purpose | Main API |
|---|---|---|
| `__init__.py` | Puts the repo root on `sys.path` (so `games`, `starter` import anywhere); re-exports the API | — |
| `tracking.py` | One reader for the ATD tracking file: players + ball in metres, tracker speeds, seek to any frame, cache in `outputs/cache/` | `load(lo, hi, cache=True) -> Tracking`, `load_ball()`, `graded_blocks()`, `graded_mask()` |
| `teams.py` | Team of each of the 46 track slots from the metadata. Unmapped slots are UNKNOWN (the old code made them AWAY) | `load_teams() -> TeamMap` (`.home`, `.away`, `.names`, `.side_of_name()`) |
| `tags.py` | Smart Tagging as typed records: case-insensitive, team → HOME/AWAY/None, half → 1/2 | `load_tags()`, `select()`, `set_pieces()`, `active_at()`, `frame_mask()`, `team_votes()` |
| `bench.py` | The test bench: Metrica's public sample games, ground truth from their events, degraded copies that behave like our data | `build_bench()`, `load_bench_game()`, `iter_bench()`, `gt_from_events()`, `degrade()` |
| `dfl.py` | Parked placeholder for the DFL Bundesliga data (the bench replaced it) | raises `NotImplementedError` |

### 8.2 `possession/preprocess/` — signal clean-up ✅

| File | Purpose | Main API |
|---|---|---|
| `ball.py` | Ball track: remove spikes and teleports, bridge gaps ≤ 6 frames, speed, touch signal, outside-pitch flag, distance to lines | `clean_ball(bx, by) -> BallTrack`, `BallConfig` |
| `players.py` | Player tracks: smoothing, velocities (jumps > 12 m/s = identity switch), mean speeds, centroids, attacking direction | `prepare_players() -> PlayerState`, `attack_signs()`, `to_attacking()` |
| `reconstruct.py` | Fill ball positions inside a gap (linear / Hermite / quadratic, from Krish) and test them by hiding known stretches | `reconstruct()`, `mask_and_score()` |

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
| `flight.py` | ✅ | Ball-flight / in-air detection (moved from `starter/ball_flight.py`); height is mostly not observable in this data |
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
| `dead_ball.py` | ✅ | Dead-ball mask: out of pitch, gaps near a line or with slow players until the restart, joined stoppages; reason per frame |
| `gap_fill.py` | ✅ | Moves a hand-over into a live gap (end / fixed / evidence). Tested: did not help on the bench, default `end` |
| `tag_prior.py` | ✅ | Smart Tagging → per-frame home/away score; code directions checked against our control signal |
| `throw_in_side.py` | ✅ | Who takes each throw-in (tags say N/A): ball cue, then team-position cue |

### 8.7 `possession/model/` — from evidence to possessions

| File | Status | Purpose |
|---|---|---|
| `resolve.py` | ✅ | State machine: switch only after 6 frames of the other team's control, backdate the switch, merge short spells. Used by v1 / v2 |
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
| `v3c` | `v3.py` | HMM + bench-trained classifier — **best on the bench**; needs `outputs/models/classifier.npz` | 0.5447 (held-out games) |

Full history and settings tried: `docs/changelog/labeller.md`.

### 8.9 `possession/events/` — football events ✅

| File | Status | Purpose |
|---|---|---|
| `schema.py` | ✅ | JSON Schema of the 14 event types (safe to share, no data) |
| `detect.py` | ✅ | Pass, carry, hold, duel, tackle, interception, possession change, restart, out of play, shot, goal, foul candidate — output identical to the old `event_detection.py` |
| `passes.py` | ✅ | Pass detection on ball flights (Ashwyn) — published match stats, `--real` and `--sweep` removed; output identical otherwise. Defaults still to re-derive on other games |

---

## 9. `scripts/` — command lines

| File | Status | Purpose |
|---|---|---|
| `_bootstrap.py` | ✅ | Makes `python scripts/x.py` and `python -m scripts.x` both work |
| `run.py` | ✅ | Run a labeller on BAR-ATH → `outputs/submissions/<name>.csv`, validate, summarise, public-window sanity check. `--labeller`, `--out`, `--no-cache`, `--bench-pitch` |
| `bench_build.py` | ✅ | Build the bench from `data/external/metrica_sample` → `outputs/bench/`. `--no-degrade`, `--games`, `--kloppy` |
| `bench_score.py` | ✅ | Score labellers on the bench. `--labeller v1 v3c`, `--variant all`, `--set KEY=VALUE` |
| `build_features.py` | ✅ | Feature table for BAR-ATH or `--bench g1` → `outputs/features/` |
| `train_classifier.py` | ✅ | Train the classifier on the bench → `outputs/models/classifier.npz` |
| `audit_data.py` | ✅ | Data audit: ball-gap histogram, team mapping, homography, tagging coverage → `outputs/audit/` |
| `classify_gaps.py` | ✅ | One row per ball gap with its class and teams → `outputs/gaps/gaps_classified.csv`; `--no-tags`, `--set` |
| `detect_events.py` | ✅ | Events JSON + schema → `outputs/events/`; `--full`, `--start/--seconds`, `--validate`, `--schema-only` |
| `detect_flights.py` | ✅ | Ball flights + in-air flag → `outputs/flights/` (needs scipy) |
| `detect_passes.py` | ✅ | Passes JSON / CSV → `outputs/passes/` |
| `eda_smart_tagging.py` | ✅ | Smart Tagging counts, N/A teams, set pieces vs ball gaps, time-base check |
| `detect_features.py`, `evauate_dfl.py`, `make_dfl_testset.py` | ⚠️ | Deprecated shims that forward to `build_features.py` / `bench_score.py` / `bench_build.py` — delete |

## 10. `viz/` — videos and plots ✅

Needs opencv-python and ffmpeg (or imageio-ffmpeg). Videos go to `outputs/videos/`.

| File | Purpose |
|---|---|
| `common.py` | Shared clip readers (tracking, homography, GT, tags), video reader, pitch panel, drawing helpers |
| `render_match.py` | Tactical video + top-down pitch + events, side by side (`--start 07:27 --duration 60`, `--event-id`, `--no-video`) |
| `render_possession.py` | The zone-of-influence possession test on video: zones, holder, contests, events, timeline |
| `render_gaps.py` | One clip per long ball gap (`--min-length`, `--max-clips`, `--dry-run`) |
| `flight_clips.py` | One clip per ball flight from `outputs/flights/ball_flights.csv` |
| `possession_definition.py` | What the public ground truth means: segment table, timeline PNG, optional clips → `outputs/possession_definition/` |

## 11. `experiments/` — frozen records ✅

| File | Purpose |
|---|---|
| `possession_test_1.py` | The first zone-of-influence possession test (frozen; its logic is `detectors/control.py` method `zone`) |
| `reconstruction_masking.py` | Krish's reconstruction comparison: Hermite best up to ~1 s (0.13 m vs 1.11 m linear), all methods drift by 5 s |

## 12. `tests/` — unit tests ⬜

`test_frames.py`, `test_teams.py`, `test_ball.py`, `test_control.py`, `test_dead_ball.py`,
`test_resolve.py`, `test_throw_in_side.py`. Synthetic data only (no match data), so they run on a
fresh clone with `pytest`.

## 13. `docs/`

| File | Status | Purpose |
|---|---|---|
| `features.md` | ✅ | Dictionary of every feature column, with sanity checks |
| `changelog/labeller.md` | ✅ | Every labeller version and bench score (Ashwyn's v1 history + v2 / v3 / v3c) |
| `changelog/passes.md` | ⬜ | Pass-detection history (from `backup/ashwyn_pass_detection_CHANGELOG.md`, stats removed) |
| `compliance.md` | ⬜ ⚠️ | Rule-compliance checklist — copy `backup/ashwyn_TESTING_ONLY_REMOVE_BEFORE_SUBMISSION.md` here and clear it before submitting |
| `research/ball_gaps.md`, `throw_in_side.md`, `ball_reconstruction.md` | ⬜ | Method notes from Krish's docs (no frame lists) |
| `log.md` | ⬜ | Daily "what changed + bench score" |
| `tech_doc.md` | ⬜ | The technical document required by SUBMISSION.md |
| `plans/Possession Challenge Game Plan.md` | | The game plan (to move to `docs/game_plan.md` with `git mv`) |

## 14. `backup/` — old code, reference only ⚠️

Flat copies of the code the package replaced: ours (`event_detection.py`, `ball_gap_analysis.py`,
`visualize_tracking.py`, …), Ashwyn's (`ashwyn_*`) and Krish's (`krish_*`). Scripts there do not run
as they are. Use them to port the ⬜ modules, then remove the folder before the repository goes
public (`ashwyn_pass_detection.py` still holds published match statistics).

## 15. `outputs/` — everything generated (git-ignored)

| Folder | Written by |
|---|---|
| `cache/` | `io.tracking.load(cache=True)` — parsed tracking (41 MB) |
| `submissions/` | `scripts/run.py` |
| `evaluation/` | `python -m evaluation.run_evaluation --out outputs/evaluation` |
| `bench/` | `scripts/bench_build.py` |
| `features/` | `scripts/build_features.py`, `classifier.bench_tables()` |
| `models/` | `scripts/train_classifier.py` |

Delete `outputs/cache/` after changing loader code; it refreshes on its own only when the raw file changes.

---

## 16. Before submitting

- [ ] `docs/compliance.md` cleared (published stats, spot-check tool, hand-verdict ideas removed)
- [ ] No data in git: `git ls-files | grep -E "npy|npz|csv|json|mp4"` shows only organiser files
- [ ] `backup/` and old root data files removed
- [ ] Bench rebuilt in the pinned environment; chosen labeller's score in the changelog
- [ ] `python scripts/run.py --labeller <chosen>` regenerates the submitted file exactly from a clean clone
- [ ] `docs/tech_doc.md` written
