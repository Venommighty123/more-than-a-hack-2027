# How to run this repository — a guide for a first-time reader

This page takes you from a fresh clone to a validated submission, then shows the everyday tasks
(trying a labeller, comparing versions, watching clips, re-running the Kaggle notebooks). It assumes
you know Python but nothing about this repo.

- **What every file is for:** [`STRUCTURED_README.md`](STRUCTURED_README.md) (a reference, file by file).
- **What the challenge is:** the organisers' `README.md`, `CHALLENGE.md`, `DATA.md`, `EVALUATION.md`, `RULES.md`.
- **This page:** what to run, in which order, and what you should see.

---

## 1. The idea in two minutes

For each of the 199,319 frames of the **tactical video** of Barcelona vs Athletic Club we must write
one label: `H` (home has the ball), `A` (away has it), `D` (dead ball) or `X` (no answer). Only the two
halves are graded (frames 11,185–90,651 and 110,808–190,197). The board ranks on **spellF1**, which
scores whole possessions, so a late hand-over or a possession split in two costs a lot.

Our pipeline, from video to labels:

```
tactical video
   │  (1) Kaggle, GPU: kaggle/kaggle_player_tracking.ipynb   -> our player tracks (YOLO)
   │  (1b) Kaggle, GPU: kaggle/kaggle_ball_detection.ipynb   -> YOLO ball (being tested)
   ▼
data/kaggle_outputs/final_tracks_full.parquet          provided ball (ATD file, ball column only)
   │                                                          │  (2) scripts/reconstruct_ball.py
   │                                                          ▼
   │                                    outputs/ball/ball_reconstructed.npz  ("the fixed ball")
   ▼                                                          │
possession.io.load()  ◄───────────────────────────────────────┘   (3) scripts/build_tracking.py
   │  one Tracking object: players (slots, metres, team per slot) + ball (metres)
   ▼
possession/labellers (v1 … v7)   ── a labeller turns players + ball (+ Smart Tagging) into H/A/D
   │  (4) scripts/run.py
   ▼
outputs/submissions/<name>.csv   ── validated, summarised, ready to submit
```

Two facts that explain a lot of the code:

- **Players come from our own tracking**, not from the provided tracking file (the "ATD"). Since
  Nov 14 the ATD is used only for its **ball** column and for diagnostics.
- **Versions are compared on the bench, never on this match.** The bench is a set of other games
  with known possession (Metrica's public samples). Tuning on BAR-ATH or on its public 5 minutes is
  against the rules and also picks the wrong version (see §3).

---

## 2. Set up (once)

### 2.1 Python environment

Python **3.13** (see `.python-version`). Pick one:

```bash
# A. uv (recommended: same versions as everyone else, from uv.lock)
uv sync
# then prefix commands with "uv run", e.g.  uv run python scripts/run.py

# B. plain venv + pip
python -m venv .venv
.venv\Scripts\activate            # Windows PowerShell   (macOS / Linux: source .venv/bin/activate)
pip install numpy pandas scipy matplotlib opencv-python pyarrow fastparquet jsonschema kloppy imageio-ffmpeg pytest
```

You also need **ffmpeg** for videos (`imageio-ffmpeg` above provides one; a system ffmpeg is faster).
`pyarrow` is required: our player tracks are a Parquet file.

### 2.2 Data (nothing of it is in git)

| What | Where it goes | Where to get it |
|---|---|---|
| Match bundle: tactical video, ATD tracking (`*_atd.txt/.xml`), homography, Smart Tagging | `data/tactical/` (and `data/broadcast/`, unused) | The organisers' download link (DATA.md "Getting it"). Keep the file names |
| Our Kaggle outputs: `final_tracks_full.parquet`, `calib/`, `eval/`, `raw_dets_full_*/` | `data/kaggle_outputs/` | Ask Kushagra, or download the output of the last Kaggle run (§7) |
| Bench games (Metrica public samples g1, g2, g3) | `data/external/metrica_sample/` | `data/README.md` |

Check: `data/tactical/BAR-ATH_tactical.mp4` and `data/kaggle_outputs/final_tracks_full.parquet` exist.

### 2.3 Smoke test (no match data needed)

```bash
python -m pytest -q tests
```
Expected: `32 passed` in a few seconds. If this fails, fix the environment before anything else.

---

## 3. Rules that shape how you work (read once, RULES.md has the full text)

1. **No hand labels on this match, for any purpose.** Watching clips to understand an error is fine;
   turning what you saw into a label, a correction, a threshold or a choice of version is not.
2. **Never tune on BAR-ATH or its public 5 minutes.** `run.py` prints a "SANITY ONLY" score on the
   public window: it only proves the file is sane. Compare versions with `scripts/bench_score.py`.
3. **No match data in git.** Everything generated goes to `outputs/` (git-ignored). Never commit
   `.parquet`, `.npz`, `.npy`, `.mp4`, or CSVs made from the match, and never upload them to a
   cloud AI tool.

---

## 4. Your first run: from data to a submission (≈ 5 minutes)

Run every command **from the repository root**. On macOS / Linux use `python3` if `python` is not 3.13.

### Step 1 — fix the provided ball (once per data bundle)
```bash
python scripts/reconstruct_ball.py
```
Reads the ball column of the ATD file, removes spikes (jumps > 2 m per frame) and fills gaps of up
to 1 s with a Hermite curve when the two ends are physically reachable. Writes
`outputs/ball/ball_reconstructed.npz` — "the fixed ball" every later step uses.

### Step 2 — build the one input (once per Kaggle run)
```bash
python scripts/build_tracking.py
```
Packs our YOLO tracks into "slots" (columns of a players × frames table, the shape the labellers
expect), gives each a team, cuts tracks at impossible jumps and removes duplicates, and attaches the
fixed ball. You should see something like:
```
players : YOLO data/kaggle_outputs/final_tracks_full.parquet  (39 slots: H 23, A 16; team mode local; clean True)
          20.4 players/frame; a team over 11 on 20.6% of frames (before clean-up 21.1%)
ball    : fixed ball outputs/ball/ball_reconstructed.npz ...  (present on 59.3% of graded frames)
```
The first run takes ~40 s (it caches the slots in `outputs/cache/`); later runs take seconds.

### Step 3 — check the tracking (optional but recommended)
```bash
python scripts/eval_tracking.py
```
Prints label-free quality numbers (people per frame, how often a team has more than 11 players,
jumps, goalkeepers on the right side, ball coverage) and the worst minutes. Use it after every new
Kaggle run to see whether the tracking got better or worse.

### Step 4 — write a submission
```bash
python scripts/run.py --labeller v6 --out outputs/submissions/v6.csv
```
**Always pass `--out`.** Without it the file is `outputs/submissions/<labeller>.csv` and silently
overwrites the previous one. The printout tells you:

| Line | Meaning |
|---|---|
| `players: ...`, `ball: ...` | which inputs were used (always check these) |
| `validation: OK` | the file has the official format (199,319 rows, valid letters) |
| `graded frames: H ..% A ..% D ..%` | label shares on the graded halves |
| `possessions: N (mean .. s ...)` | how many possessions the labeller wrote and how long |
| `SANITY ONLY (public 5 min ...)` | a format check — **never** use it to choose anything |

### Step 5 — score a folder of submissions with the official scorer (optional)
```bash
python -m evaluation.run_evaluation --results outputs/submissions --out outputs/evaluation
```
Run it with `-m` (as a module); running the file directly breaks its imports.

---

## 5. Everyday tasks

### 5.1 Try another labeller or setting
```bash
python scripts/run.py --labeller v4 --out outputs/submissions/v4.csv
python scripts/run.py --labeller v6 --set SW_MARGIN=0.8 --out outputs/submissions/v6_sw08.csv
python scripts/run.py --labeller v6 --team-mode track --out outputs/submissions/v6_trackteam.csv
python scripts/run.py --labeller v6 --raw-players --out outputs/submissions/v6_raw.csv
```
Labellers: `baseline0/25/50/125`, `v1`–`v7`, `v3c` (needs a trained classifier, §5.6). What each
one does: `STRUCTURED_README.md` §8.8 and `docs/changelog/labeller.md`. `--set` only for settings
that were scored on the bench.

### 5.2 Compare two submissions frame by frame
```bash
python scripts/run.py --labeller v6 --out outputs/submissions/try.csv --compare outputs/submissions/v6.csv
```
Prints the agreement on graded frames and a confusion table (rows: this file, columns: the other).

### 5.3 Compare labeller versions properly — the bench
```bash
python scripts/bench_build.py                                   # once (several minutes): -> outputs/bench/
python scripts/bench_score.py --labeller v4 v6 v7 --variant decision
```
`--variant decision` scores the realistic copies (`real`, `real_occ`) and prints **R+O**, the
number we decide on. Other variants: `clean`, `degraded`, `hard`, `tracked`, `tracked_all`, `all`.
Note: the realistic copies imitate the ATD's errors; our YOLO tracks fail differently (short
tracks, team flips), so a YOLO-like bench copy is on the to-do list.

### 5.4 Watch what a labeller did on the video
```bash
# labels on the video with our players, a top view and a timeline (-> outputs/labeller_viz/)
python viz/visualize_labeller.py --labels outputs/submissions/v6.csv --range 73997 75081

# diagnostics: five panels, original | ATD players | our players / top views + info
python viz/visualize_labeller.py --compare --range 73997 75081 ^
    --labels outputs/submissions/v6_atd.csv --labels2 outputs/submissions/v6.csv

# list the possessions in a window, no video
python viz/visualize_labeller.py --range 73997 75081 --spells-only
```
`--range FIRST LAST` are video frames (= CSV rows). `--start 49:20 --duration 30` works too. `^` is
the PowerShell line break; on macOS / Linux use `\`, or write the command on one line.

Where to look: `outputs/track_eval/per_minute_*.csv` (from `eval_tracking.py`) lists the minutes
with the most problems; `python viz/track_jumps.py --source slots` lists jumps with the exact
`visualize_labeller` command for each.

### 5.5 Events, flights, passes
```bash
python scripts/detect_flights.py                 # ball flights -> outputs/flights/ (v5 / v7 use them)
python scripts/detect_events.py --full --validate  # passes, carries, duels, restarts... -> outputs/events/
python scripts/detect_passes.py                  # pass table -> outputs/passes/
python viz/test_event_clips.py                   # 8 clips per event type -> outputs/test_events/
```
Run `detect_flights.py` before labellers v5 / v7. `detect_events.py --start 11185 --seconds 120`
does a short window; `--compare-events old.json` lists what changed against an older run.

### 5.6 Use the YOLO ball (after a ball-notebook run)
```bash
python scripts/fix_yolo_ball.py          # data/kaggle_outputs/ball_yolo_full.parquet -> outputs/ball/ball_yolo_fixed.npz
python scripts/run.py --labeller v6 --ball-file outputs/ball/ball_yolo_fixed.npz --out outputs/submissions/v6_yoloball.csv --compare outputs/submissions/v6.csv
```
Why a post-processing step: YOLO sees the ball where it is **on screen**, so through the homography
an airborne ball lands on the grass behind its true position, moving too fast. `fix_yolo_ball.py`
finds each flight, fits a 3D parabola through every frame's camera and puts the ball back over its
true ground point (and records its height). Where YOLO has no ball the fixed provided ball is used
(`--fallback none` to switch that off). Check `outputs/ball/ball_yolo_fixed_flights.csv`: a real lob
has a clear height, `rms_px_flight` of a few pixels and a much larger `rms_px_ground`.

### 5.7 Features, the classifier labeller, ball gaps
```bash
python scripts/build_features.py                 # per-frame feature table -> outputs/features/
python scripts/train_classifier.py               # trains on the BENCH only -> outputs/models/classifier.npz
python scripts/run.py --labeller v3c --out outputs/submissions/v3c.csv
python scripts/classify_gaps.py                  # one row per missing-ball stretch -> outputs/gaps/
python viz/test_ball_clips.py --dry-run          # list ball-test clips (drop --dry-run to render)
python viz/render_gaps.py --dry-run              # clips of long ball gaps
```

---

## 6. When something changes, what to re-run

| You changed / received | Re-run |
|---|---|
| New Kaggle player run (new `final_tracks_full.parquet`) | `build_tracking.py --rebuild` → `eval_tracking.py` → `run.py` (new `--out` name) |
| Loader code in `possession/io/` | `build_tracking.py --rebuild` (the cache does not notice code changes) |
| A new data bundle / ball settings | `reconstruct_ball.py` → `build_tracking.py --rebuild` → `detect_flights.py` → `run.py` |
| New Kaggle ball run (`ball_yolo_full.parquet`) | `fix_yolo_ball.py` → `run.py --ball-file outputs/ball/ball_yolo_fixed.npz` |
| A labeller's code | `pytest` → `bench_score.py` → `run.py` |
| Anything in `possession/` | `python -m pytest -q tests` |

---

## 7. The Kaggle notebooks (GPU work)

Our own detection runs on Kaggle (free GPUs). Two notebooks live in `kaggle/`:

| Notebook | Produces | Time (T4 ×2) |
|---|---|---|
| `kaggle_player_tracking.ipynb` | players, goalkeepers, referees per frame with track IDs and teams → `final_tracks_full.parquet`, scorecard vs ATD, review clips | ~2.5 h full |
| `kaggle_ball_detection.ipynb` | YOLOv8 + SAHI ball where the provided ball is missing (+ a 1-in-10 sample to measure accuracy) → `ball_yolo_full.parquet` | depends on the probe estimate |

**Kaggle datasets** (private, attach with "Add Input"):
- `bar-ath-tactical`: the video, `BAR-ATH_tactical_atd.txt` + `.xml`, `BAR-ATH_tactical_homography.csv`
- `supa-strikers-models`: the player model `.pt` and the ball model `.pt` (name containing `ball`)
- optional `supa-strikers-outputs`: a previous run's `/kaggle/working`, **only** to resume an
  interrupted run (do not attach it for a fresh run: old calibration would be reused)

**Settings:** Accelerator *GPU T4 ×2*, Internet *on*.

**Procedure:** import the notebook → leave `RUN_MODE = "smoke"` → *Run All* (10–15 min) → check the
printed checks (alignment, crop grids, scorecard / probe) → set `RUN_MODE = "full"` →
*Save Version → Save & Run All* (keeps running if you close the browser) → when it finishes,
download the output into `data/kaggle_outputs/` (keep the previous folder under another name) →
§6 first row.

---

## 8. Where everything is written

All under `outputs/` (git-ignored):

| Folder | Written by |
|---|---|
| `ball/` | `reconstruct_ball.py` |
| `cache/` | `build_tracking.py` / `possession.io.load()` (packed players) |
| `track_eval/` | `eval_tracking.py` |
| `submissions/` | `run.py` |
| `evaluation/` | `python -m evaluation.run_evaluation` |
| `bench/`, `bench_results/` | `bench_build.py`, `bench_score.py` |
| `flights/`, `events/`, `passes/`, `features/`, `models/`, `gaps/` | the matching scripts in §5 |
| `labeller_viz/`, `track_audit/`, `videos/`, `test_ball/`, `test_events/` | the `viz/` tools |

---

## 9. Troubleshooting

| Symptom | Fix |
|---|---|
| `ImportError ... pyarrow` | `pip install pyarrow` (in the environment you run from) |
| `... not found under data/...` | a data file is missing or renamed; see §2.2 (keep the organisers' file names) |
| `note: outputs/ball/ball_reconstructed.npz not found - using the raw ATD ball` | run `scripts/reconstruct_ball.py` |
| Results did not change after editing loader code | `python scripts/build_tracking.py --rebuild` |
| A submission you compared against suddenly matches 100 % | it was overwritten by a run without `--out`; regenerate it |
| `SyntaxError ... null bytes` on a `.py` file | the file was saved as UTF-16 (PowerShell `>` does that). Save as UTF-8: `... \| Out-File -Encoding utf8 file.py` |
| `python -m evaluation.run_evaluation` import error | run it as a module from the repo root, exactly as in §4 step 5 |
| Video tools: `ffmpeg not found` | `pip install imageio-ffmpeg` or install ffmpeg; `--backend opencv` works without it (slower seeks) |
| `git` complains about `*.lock` files | another git process crashed; delete the `.lock` file inside `.git/` it names |

---

## 10. Glossary

| Term | Meaning |
|---|---|
| **frame** | a video frame of the tactical video, 0 … 199,318 at 25 fps; also the row of a submission |
| **graded halves** | 11,185–90,651 and 110,808–190,197; everything else is not scored |
| **ATD** | the provided Automatic Tracking Data (Metrica). We use only its ball column now |
| **fixed ball** | the ATD ball after `reconstruct_ball.py` (spikes out, short gaps filled) |
| **YOLO tracks** | our players from the Kaggle notebook (YOLOv8 detection + tracker + team classifier) |
| **slot** | one column of the players table the labellers read; holds one track at a time, several tracks over the match |
| **team mode** | how a YOLO row gets its team: `local` (classifier over ±0.5 s, default) or `track` (whole-track vote) |
| **T1–T4** | team classifiers in the notebook; T4 (home / away / official) is the one used |
| **SAHI** | slicing the frame into tiles so a small ball is still detected |
| **homography** | per-frame matrix mapping pitch coordinates to pixels; how video positions become metres |
| **bench / R+O** | other games with known possession; R+O = mean spellF1 on the realistic `real` and `real_occ` copies |
| **spellF1 / macroF1** | the board metric (possessions) / per-frame F1 (EVALUATION.md) |
| **H / A / D / X** | home / away possession, dead ball, no answer |

---

## 11. Read next

- `STRUCTURED_README.md` — every folder and file, with status.
- `docs/player_tracking_plan.md` — how our tracking works, what was tried, current status.
- `docs/changelog/labeller.md` — every labeller version and its bench score.
- `EVALUATION.md`, `RULES.md` — before changing how anything is chosen.
