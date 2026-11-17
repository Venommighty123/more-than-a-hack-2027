# Ball Reconstruction (Krish's approach): filling ATD ball gaps

This folder contains the code that fills missing ball positions in the tactical ATD **using the
ATD alone**: it interpolates between the ball positions on either side of a gap. It does not use
the YOLO ball (`fill_ball_gaps.py`, `ball_fill.py`, `fix_yolo_ball.py` and `select_ball_candidates.py`
are a separate approach and are not included).

Every file keeps its path from the repository. Copy the folders over a checkout of the repo to run them.

---

## 1. The idea in short

- The ball is missing on a large share of ATD frames (Krish's EDA found 37.5% of all frames, in
  196 gap runs; about 41% of the graded frames).
- **Short gaps** can be filled well from the positions and velocities at both ends of the gap.
- **Long gaps** cannot: every method drifts by several metres after a few seconds. They are left
  empty (NaN) for the dead-ball and gap rules to handle.
- Krish compared three methods by **masking**: hiding stretches where the ball is known, refilling
  them and measuring the error. **Velocity-aware cubic Hermite** was best.
- Filled frames are always flagged (`filled`). They are never silently mixed with observed frames.

### Masking results (Krish's original run, 100 samples per gap length)

| Gap | Linear RMSE | Hermite RMSE | Quadratic RMSE | Hermite within 5 m |
|---|---|---|---|---|
| 5 fr (0.2 s) | 0.069 m | **0.011 m** | 0.009 m | 100% |
| 15 fr (0.6 s) | 0.432 m | **0.018 m** | 0.082 m | 100% |
| 30 fr (1.2 s) | 1.307 m | **0.236 m** | 0.452 m | 99.6% |
| 60 fr (2.4 s) | 3.489 m | **1.641 m** | 2.042 m | 85.7% |
| 125 fr (5.0 s) | 7.381 m | **5.104 m** | 5.559 m | 53.2% |
| 250 fr (10 s) | 13.774 m | 12.506 m | 13.130 m | 31.7% |

The pipeline uses a **cap of 25 frames (1 s)**. That is more conservative than Krish's suggested
limit of 60 frames, because the real gaps turned out to be mostly tracker re-detections (see the
speed gate in section 3).

---

## 2. Run order

```bash
# 1. Build the reconstructed ball once (writes outputs/ball/ball_reconstructed.npz)
python scripts/reconstruct_ball.py
python scripts/reconstruct_ball.py --csv                        # also writes a frame,x,y,observed,filled CSV
python scripts/reconstruct_ball.py --max-gap 25 --method hermite --max-speed 35

# 2. Check the methods by masking known stretches
python experiments/reconstruction_masking.py                    # BAR-ATH graded halves
python experiments/reconstruction_masking.py --source bench --game g1 --variant clean

# 3. (optional) Find and classify the gaps that stay empty
python scripts/classify_gaps.py                                 # -> outputs/gaps/gaps_classified.csv

# 4. (optional) Render clips of long gaps to see what happens during them
python viz/render_gaps.py --min-length 125 --max-clips 10

# Tests
pytest tests/test_reconstruct.py
```

---

## 3. File by file

### Core pipeline

#### `possession/preprocess/reconstruct.py` (the heart of the approach)
Ported from Krish's `ball_reconstruction.py`. The methods are unchanged, but the code is rewritten
to work over gap runs and in metres.

| Function | What it does |
|---|---|
| `_linear(x, y, s, e)` | Straight line between the last point before the gap and the first point after it. |
| `_hermite(x, y, s, e, present)` | Cubic Hermite spline that matches **position and per-frame velocity** at both ends. A velocity comes from the neighbouring frame when it is tracked; otherwise the linear slope is used. |
| `_quadratic(x, y, s, e, present, context=4)` | Quadratic fit through up to 4 observed points on each side. Falls back to linear when fewer than 4 points exist or the fit fails. |
| `reconstruct(x, y, method, max_gap, mask, max_speed_ms, fps)` | Fills **interior** gaps only (both ends must be tracked). It skips a gap longer than `max_gap`, and a gap whose ends imply a speed above `max_speed_ms`. Returns `(x, y, filled)`. |
| `mask_and_score(x, y, gap_lengths, trials, methods, seed, within_m)` | The masking test. Picks random fully tracked stretches (with 4 tracked frames of context on each side), hides them, refills them with each method and returns rmse / mae / max error / % within 5 m. |
| `prepare_ball(bx, by, method="hermite", max_gap=25, max_step_m=2.0, fps=25, max_speed_ms=35.0)` | The **prepared ball** used by the pipeline: removes spikes, then fills gaps of up to 25 frames with Hermite. Returns `(x, y, observed, removed, filled)`. |
| `save_ball_file(...)` / `load_ball_file(path)` | Write and read `outputs/ball/ball_reconstructed.npz` (arrays plus the settings used). |

**Why the 35 m/s speed gate exists:** on BAR-ATH, 122 of the 129 gaps of 25 frames or fewer join
two points the ball could not have travelled between in the time (median implied speed 58 m/s, up to
650 m/s). The tracker jumped to another detection; it didn't lose the ball. Filling those gaps drew
fake 50 to 1000 m/s kicks (the cause of the 300+ km/h "flights"), so they stay NaN.

#### `scripts/reconstruct_ball.py`
The command-line entry point. It loads the BAR-ATH ATD, calls `prepare_ball`, saves the `.npz` and
prints the stats for the graded frames: % observed, spikes removed, frames and gaps reconstructed,
and the gaps left.

| Option | Default | Meaning |
|---|---|---|
| `--out` | `outputs/ball/ball_reconstructed.npz` | Output file |
| `--method` | `hermite` | `linear` / `hermite` / `quadratic` |
| `--max-gap` | `25` | Largest gap filled, in frames (25 = 1 s) |
| `--max-step` | `2.0` | Spike threshold, metres per frame |
| `--max-speed` | `35.0` | Speed gate in m/s (`0` = off) |
| `--csv` | off | Also write `frame,x_m,y_m,observed,filled` |
| `--no-cache` | off | Reload the ATD instead of using the cache |

Downstream readers: `scripts/run.py` (labeller v5) and `scripts/detect_flights.py`. Both compute the
same thing in memory if the file is missing.

#### `possession/preprocess/ball.py`
Ball clean-up and kinematics. `reconstruct.py` builds on it.

- `remove_spikes`: a point that jumps more than 2 m per frame away **and back** is a false blob.
- `remove_teleports`: a one-sided speed ceiling (40 m/s).
- `bridge_gaps`: the old linear bridge of up to 6 frames (`BallConfig.bridge_frames`).
- `runs`, `gap_table`: find runs of missing frames.
- `windowed_velocity`, `speed_of`, `heading_of`, `touch_signal`, `acceleration`: kinematics. Speed
  is measured over ±4 frames because 1-frame differences turn 0.3 m of jitter into about 5 m/s of
  fake speed.
- `outside_pitch`, `distance_to_lines`: pitch geometry (105.3 × 68 m).
- `clean_ball(...)` → `BallTrack`: the full clean-up in one call.

### Evaluation

#### `experiments/reconstruction_masking.py`
Runs `mask_and_score` and prints a table by gap length and method (frames, seconds, rmse, mae, max,
% within 5 m).
Options: `--source barath|bench`, `--game`, `--variant`, `--gaps 5 15 30 60 125 250`,
`--trials 200`, `--seed 0`. Use `--source bench` when the numbers are used to **choose** a method
or cap. Krish's long-gap fallback scoring on the public 5 minutes is deliberately left out
(RULES.md reserves those for sanity checks).

#### `tests/test_reconstruct.py`
| Test | Checks |
|---|---|
| `test_fills_plausible_gap_up_to_max_gap` | A gap on a ball moving at 10 m/s is filled, and the result is within 5 cm of the true line |
| `test_leaves_long_gap` | A gap longer than `max_gap` stays NaN |
| `test_speed_gate_rejects_redetection` | A gap whose ends imply an impossible speed is not filled, and is filled when the gate is off |
| `test_linear_6_equals_old_bridge` | Linear with `max_gap=6` matches the old `clean_ball` bridge exactly |
| `test_reconstruct_flags_only_filled` | Only the reconstructed frames are flagged |

#### `docs/research/ball_reconstruction.md`
This file.

### Gaps that stay empty (finding and classifying them)

#### `possession/gaps/find.py`
Merges Krish's `eda_ball_gaps.find_gaps` with the team's version.
- `Gap`: a dataclass with `half`, `start`, `end` (exclusive), `has_before`/`has_after` (False = the gap
  touches the edge of a block, so its true length is unknown), and `info`.
- `find_gaps(present, blocks, lo, min_len)`: gaps inside the graded halves only.
- `bracket(gaps, bx, by)`: the ball position either side, distance and implied speed.
- `gap_mask`, `length_histogram`: helpers.

#### `possession/gaps/classify.py`
Uses Krish's set-piece and goal overlap from `classify_all_135_gaps.py`, plus Ashwyn's rules.
Classes: `dead_kickoff`, `dead_tagged` (a Smart Tagging set piece overlaps the gap), `dead_untagged`
(the ball vanished near a line and the players were slow), and `live`. It also records
`team_before`/`team_after`, `flip`, the phase team and the median player speed.
Defaults (`GapClassConfig`): `min_gap_s=1.0`, `near_line_m=2.0`, `dead_speed_ms=1.8`,
`tag_slack_s=3.0`, `context_s=5.0`. These are untuned.

#### `scripts/classify_gaps.py`
Runner for the two modules above, writing `outputs/gaps/gaps_classified.csv` (one row per gap).
Options: `--no-tags` (as on the bench) and `--set key=value` for sensitivity checks.

#### `viz/render_gaps.py`
Renders one clip per long gap (`gap_<start>_<length>.mp4`) through `viz/render_match.py`, to see what
happens while the tracker is blind. From Krish's `render_gap_clips.py`. Options: `--min-length`
(default 60), `--max-clips`, `--pad`, `--extra-args`. Use the clips to understand the data only, not to
hand-label gaps (RULES.md).

### Krish's originals (`backup/`, kept for reference; these do not run as they are)

| File | Contents |
|---|---|
| `krish_ball_reconstruction.py` | The original suite: linear, PCHIP, velocity-aware Hermite and quadratic; masking on runs of 300+ frames; RMSE/MAE/max/reliability; and the long-gap fallbacks (last-touch carry-forward 41.9%, team-centroid 44.2% on the public window). |
| `krish_eda_ball_gaps.py` | Streams the raw ATD (the ball is the segment after the last `:`; missing = `NaN,NaN,NaN`), lists the longest gaps and runs a threshold sweep. |
| `krish_analyze_all_gaps.py` | Matches rendered gap clips to Smart Tagging events. |
| `krish_classify_all_135_gaps.py` | Classifies the 135 rendered clips: 93 restart or state change, 42 in-play occlusion. |
| `krish_render_gap_clips.py` | Original clip renderer (works in ATD `frame_id` = video frame + 1). |
| `krish_GAPS_AND_CLIPS.md` | Gap statistics, the clip naming scheme and the 93/42 breakdown. |
| `krish_thoughts.md` | Gap-filling strategy: Rule A (set piece → `D`), Rule B (tactical phase → flag), Rule C (no overlap → `D`), and ideas for the 42 in-play gaps. |
| `krish_README.md` | Full write-up: methods, masking method, result tables and fallbacks. |

---

## 4. Dependencies (not in this folder)
These scripts import other parts of the repo: `games.py`, `scripts/_bootstrap.py`,
`scripts/run.py` (used by `classify_gaps.py`), `possession/io/*` (`load`, `graded_mask`,
`graded_blocks`, `load_tags`, `bench`), `possession/utils/frames.py`, `possession/labellers/v1.py`,
`possession/detectors/control.py`, `viz/render_match.py` and `starter/*`. Python packages: numpy
(plus pandas for the backup scripts). Match data is not included; `outputs/` is git-ignored.

## 5. Known limits / next steps
- Gaps longer than 1 s are not filled. They need the gap rules or player-geometry priors.
- The 25-frame cap and the 35 m/s gate were picked on BAR-ATH. Re-check them on the bench.
- PCHIP from the original suite was not ported into `METHODS`.
