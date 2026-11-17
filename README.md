# Possession labelling for BAR-ATH: More than a Hack 2027, Level 1

**Team:** SupaStrikerz#1605 · **Members:** Ashwyn Kumar, Kushagra Mathur, Krish Singh

One label per frame of the tactical video (`H` Barcelona in possession, `A` Athletic Club in possession,
`D` dead ball, `X` not graded), produced by a rule-based labeller (**v6**) that reads **player tracks we
built from the tactical video** and the **ball from the provided tracking (ATD)**.

| | |
|---|---|
| Submitted file | `<team_name>.csv`, 199,319 rows, passes `evaluation/validate_submission.py` |
| Expected md5 | **`587e9693e4ef247eeb4b47ecd80c4a9f`** (for the player-tracks file we used, see section 2.3) |
| Build time | step 1 (player tracking, GPU): several hours on Kaggle T4 x2 · step 2 (labeller, CPU): about 1 min |
| Hand labels | **none**, on any game. No human-produced data about this match. |

---

## Contents

1. [Quick start](#1-quick-start)
2. [Full reproduction, step by step](#2-full-reproduction-step-by-step)
3. [Repository layout](#3-repository-layout)
4. [How the labeller works](#4-how-the-labeller-works)
5. [Possession rules we implemented](#5-possession-rules-we-implemented)
6. [Decisions and why](#6-decisions-and-why)
7. [How we tuned without touching this match](#7-how-we-tuned-without-touching-this-match)
8. [What we tried and dropped](#8-what-we-tried-and-dropped)
9. [Compliance with RULES.md](#9-compliance-with-rulesmd)
10. [Limitations and next steps](#10-limitations-and-next-steps)

---

## 1. Quick start

```bash
# 0. install (Python 3.13)
uv sync                                   # exact versions from uv.lock (see 2.1)

# 1. put the match data in data/ (see data/README.md), then build the player tracks on Kaggle
#    with test_scripts/kaggle-player-tracking_new.ipynb and copy final_tracks_full_t5.parquet to
#    results_new/                                                       (section 2.3)

# 2. build and check the submission
python make_submission.py --name <team_name>
python evaluation/validate_submission.py submission/<team_name>.csv
```

`make_submission.py` prints `validation (official checks): OK`, the label shares and the md5.

---

## 2. Full reproduction, step by step

### 2.1 Software

| What | Version | Used for |
|---|---|---|
| Python | 3.13 (`.python-version`) | everything |
| numpy, pandas, scipy, opencv-python, matplotlib, imageio-ffmpeg, jsonschema | pinned in `uv.lock` | labeller, evaluation, visualisation |
| pyarrow | pinned in `uv.lock` | reading the player-tracks parquet |
| Kaggle notebook only: ultralytics, supervision 0.25.1, umap-learn, scikit-learn, torch, transformers | installed by the notebook's first cells | player detection, tracking, team model |

```bash
uv sync           # creates .venv with the exact versions in uv.lock
```

### 2.2 Match data (not in this repository)

The data is gated and must not be redistributed, so it is **not** committed (`.gitignore`). Unpack the
bundle into `data/` exactly as `data/README.md` shows, or point `MTAH_DATA_DIR` at it. The labeller reads:

| File | What we use from it |
|---|---|
| `data/tactical/BAR-ATH_tactical_atd.txt` | the **ball** position per frame (players are NOT used from it in the final run) |
| `data/tactical/BAR-ATH_tactical_atd.xml` | team names and the team of each ATD slot |
| `data/tactical/BAR-ATH_tactical_smart_tagging.xml` | start frames of the "Set Pieces" tags |
| `data/tactical/BAR-ATH_tactical.mp4`, `_homography.csv` | only in step 2.3 (the player tracking notebook) |

### 2.3 Step 1: player tracks from the tactical video (GPU, Kaggle)

Notebook: **`test_scripts/kaggle-player-tracking_new.ipynb`**.

| Setting | Value |
|---|---|
| Platform | Kaggle notebook, **Accelerator GPU T4 x2**, **Internet on** |
| Inputs (Add Input) | a dataset with `BAR-ATH_tactical.mp4`, `BAR-ATH_tactical_atd.txt`, `BAR-ATH_tactical_atd.xml`, `BAR-ATH_tactical_homography.csv` (keep it **private**: the data may not be redistributed) |
| Player detector | `model_file/model_file/yolo-football-player-detection.pt` (in this repository, 40.6 MB, md5 `61c83c9d261d8b07b6fa39e8070a7623`): an **Ultralytics YOLO11m** detector with 4 classes (ball, goalkeeper, player, referee), trained by the open-source project [martinjolif/football-game-tracking](https://github.com/martinjolif/football-game-tracking) (AGPL-3.0) on the public dataset [martinjolif/football-player-detection](https://huggingface.co/datasets/martinjolif/football-player-detection): 372 annotated images (298 train / 49 validation / 25 test) from [Roboflow Universe, "football-players-detection"](https://universe.roboflow.com/football-project-pifbc/football-players-detection-3zvbc-yyhdl), CC BY 4.0, labelled by its Roboflow authors ("Football project"), not by us and not from the competition data. The notebook's first code cell sets `MODEL_PATH = "/kaggle/input/datasets/kushubhai/yolo-teachers/models/yolo-football-player-detection.pt"` (our private Kaggle dataset holding this same file). To reproduce, upload `model_file/model_file/yolo-football-player-detection.pt` to Kaggle as a dataset, add it to the notebook as an input, and change `MODEL_PATH` to its path (`/kaggle/input/<your-dataset>/yolo-football-player-detection.pt`); check the md5 above. |
| Team model | `google/siglip2-so400m-patch14-384`, commit `e8e487298228002f3d8a82e0cd5c8ea9c567f57f` (fallback `google/siglip-base-patch16-224`, commit `7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed`), downloaded from Hugging Face by the notebook |
| Run | set `RUN_MODE = "smoke"`, Run All (about 15 min), check the printed checks; then `RUN_MODE = "full"`, *Save Version → Save & Run All* |
| Output | `/kaggle/working/final_tracks_full_t5.parquet` (download it to `results_new/final_tracks_full_t5.parquet`) |

What the notebook does (details in its markdown cells):

1. **Detection** of every person on both halves (frames 11,185 to 90,651 and 110,808 to 190,197) with the
   YOLO11m player model, on two GPUs, in resumable chunks.
2. **Feet to pitch metres** with the provided homography. Detections more than 3 m off the pitch (bench,
   ball boys, staff) are dropped.
3. **Tracking**: three trackers are compared (K1 ByteTrack, K2 tuned ByteTrack, K3 a tracker in pitch
   metres), with stitching (gaps up to 3 s, at most 8 m/s while unseen) and Savitzky-Golay smoothing.
   A label-free scorecard against the ATD picks the variant; the run picked **K3-T4+post**.
4. **Team and role (T5)**: every track is cut into 10 s segments; each segment is classified home /
   away / home GK / away GK / official by a logistic regression on SigLIP2 embeddings of whole-body
   crops. The training labels are **pseudo-labels from the provided ATD** (a detected person matched to an
   ATD player takes that slot's team), so no hand labels are involved. Held-out check (train on even
   2-minute blocks, test on odd ones): **97.1% accuracy**; team agreement with the ATD rises from 75.2%
   (colour models) to **97.0%**.

Result used for the submission: 3.77 M rows, 1,138 player/goalkeeper tracks; a team shows more than
11 players on 10.3% of graded frames (the provided ATD: 12.9%).

> **Reproducibility note.** Step 2 is fully deterministic: the same parquet always gives the same CSV
> (md5 above). Step 1 runs on GPUs in half precision, and GPU inference is not guaranteed to be
> bit-identical across hardware and driver versions, so a re-run can give slightly different tracks.
> The exact `final_tracks_full_t5.parquet` we used (md5 `351000f46bebf85221457121de52c6ea`) is
> derived from the match data, so it cannot be published here; **we can give it to the organisers on
> request** so step 2 can be checked byte for byte.

### 2.4 Step 2: the labeller (CPU, about one minute)

```bash
python make_submission.py --name <team_name>
```

It prints progress bars, then `validation (official checks): OK`, the label shares
(`H 33.2%  A 26.1%  D 40.7%  X 0.0%` of graded frames), the number of possessions (302) and stoppages (95),
and the md5 (`587e9693e4ef247eeb4b47ecd80c4a9f` with our parquet). Output: `submission/<team_name>.csv`.

### 2.5 Step 3: check before sending

```bash
python evaluation/validate_submission.py submission/<team_name>.csv
python -m evaluation.run_evaluation --results submission               # public 5 min: sanity check ONLY
```

---

## 3. Repository layout

Only what the pipeline and its documentation need is committed.

| Path | Role |
|---|---|
| `make_submission.py` | **entry point**: builds the submission CSV (section 2.4) |
| `test_scripts/kaggle-player-tracking_new.ipynb` | **step 1**: player detection, tracking and team model (section 2.3) |
| `model_file/model_file/yolo-football-player-detection.pt` | the player detector weights used in step 1 |
| `test_scripts/labeller_v6.py` | **the labeller** used for the submission (section 4) |
| `test_scripts/labeller_v1.py`, `labeller_v2.py`, `labeller_v4.py`, `labeller_v5.py` | building blocks `labeller_v6.py` imports (each version adds switches on top of the previous one; every switch and default is documented in its docstring) |
| `test_scripts/atd_io.py` | reading the ATD: team of each slot, team names, players and ball |
| `test_scripts/run_labeller_v6.py` | v6 on the ATD players (our earlier reference run) and the set-piece tag loader |
| `test_scripts/run_labeller_v6_newtracks.py` | readers for the video tracks (slot packing) and the run-to-run comparison |
| `test_scripts/run_labeller_v6_t5.py` | the final comparison run: same output as `make_submission.py`, plus a reproduction check and comparisons with the earlier runs |
| `test_scripts/bench/` | the test bench on **other** matches (section 7) |
| `test_scripts/noise_profile.py` | measures this match's tracking noise (counts only) so the bench copies can imitate it |
| `test_scripts/labeller_v1_CHANGELOG.md` | every labeller change with its bench result, including the rejected ones |
| `test_scripts/visualize_labeller.py`, `visualize_transitions.py`, `list_transitions.py`, `list_dead_transitions.py`, `recheck_transitions.py`, `gap_classification.py` | tools to look at a labeller CSV on the video (understanding only) |
| `visualize_tracking.py`, `ball_gap_analysis.py` | data exploration tools |
| `games.py`, `starter/`, `evaluation/`, `comp_docs/`, `data/README.md` | provided by the organisers (paths, loaders, official scorer and checks, competition documents) |

Not committed (`.gitignore`): `data/`, `results/`, `results_new/`, `bench_data/`, `reports/`,
`submission/`, every label CSV except the organisers' baselines, `*.parquet`, `*.npy`, `*.npz`, `*.mp4`, and every
`*.pt` except the player detector above.

---

## 4. How the labeller works

`labeller_v6.label(px, py, ball, team, blocks, params, tags)` turns positions in metres into one label per
frame. It is pure geometry plus a state machine; no learning, nothing fitted on this match.

**Inputs.** `px, py`: players (frames x slots); each video track is placed in a slot of its own team
(a slot holds one track at a time). `ball`: the ATD ball, NaN where it is missing (41% of graded frames).
`tags`: start frames of the Smart Tagging "Set Pieces". `blocks`: the two graded halves.

| Step | What happens | Settings |
|---|---|---|
| 1. Smooth | players and ball: centred moving average over ±3 frames | `SMOOTH_K=3` |
| 2. Clean the ball | a point that jumps more than 2 m in one frame **and back** is removed; holes of up to 6 frames are bridged linearly | `MAX_STEP=2.0`, `BRIDGE=6` |
| 3. Player speed | mean player speed over a 3 s window, divided by the match median (`rel`); this match's players are slow, so absolute thresholds fire everywhere | `SMOOTH_S=3.0` |
| 4. Control, per frame | a team **controls** the ball when its nearest player is within 2.0 m, the nearest opponent is at least 0.3 m further, and the ball is slower than 12 m/s; otherwise nobody does | `CTRL_R=2.0`, `MARGIN=0.3`, `CTRL_V=12` |
| 5. Dead ball (D) | see the table below | |
| 6. Possession | state machine per half: the team changes only when the other team **controls** the ball for 6 frames (holes of up to 3 frames allowed); to take the ball from the team in possession it must be **0.6 m** clearer (hysteresis); the change is dated back to the first frame of that control | `CONFIRM=6`, `HOLE=3`, `SW_MARGIN=0.6` |
| 7. After a stoppage | the first team to control the ball gets it | |
| 8. Clean-up | a spell shorter than 2 s between two spells of the same team is merged into them | `MIN_SPELL=2.0` |
| 9. Second half | graded frames before the second-half kickoff are `D` (FAQ.md) | |

**Dead ball (step 5).**

| Rule | The ball is dead when ... | until ... |
|---|---|---|
| A. Out of play | it is more than 0.5 m outside the pitch | it is struck (faster than 5 m/s) inside the pitch; players running after it do not end it |
| B. Ball missing | it is missing for at least 1.5 s **and** it vanished or reappeared within 2 m of a line, or the players were slow during the gap (median `rel` below 0.75) | the restart: ball struck faster than 5 m/s, or players back at open-play speed (`rel` above 1.0) |
| D. Set-piece tag | a "Set Pieces" tag starts: dead from the moment the players slowed down (up to 15 s before the tag) | the first strike within 8 s after the tag |
| Join | two dead stretches less than 1 s apart become one | |
| `MIN_D` | a dead stretch shorter than 2 s is treated as live (a real stoppage takes longer) | |

A gap shorter than 1.5 s, or a long gap in full flow away from the lines, is **live**: it keeps the last
team, and any change is decided when the ball is seen again. We never write `X` on a graded frame:
under `spellF1` an `X` is filled with the previous label anyway, and under `macroF1` it is a miss.

---

## 5. Possession rules we implemented

How each rule in `comp_docs/CHALLENGE.md` maps to the code:

| CHALLENGE.md | Our implementation |
|---|---|
| A change needs **controlled** possession by the other team | 6 frames (0.24 s) of control by the other team, with a 0.6 m clearer margin than the team in possession |
| Loose or contested ball: possession stays with the last team | no control (nobody close enough, both teams close, or the ball too fast) never changes the team |
| A simple touch, a deflection, a won header without control is **not** a change | a touch or a deflection gives at most a frame or two of control and a fast ball, so `CONFIRM` is not reached; a spell under 2 s between two spells of the same team is merged back |
| After a stoppage there is a clear possessor | the first team to control the ball after `D` |
| Dead ball is a stoppage, never a possession | rules A, B, D above; nothing is assigned to a team while dead |
| Three states only | `H`, `A`, `D` on every graded frame |

---

## 6. Decisions and why

| Decision | Why |
|---|---|
| Rule-based labeller, not a trained model | there is no legal labelled data for this match, and the rules are explicit; every rule can be explained and tested on other matches |
| Players from **our video tracks**, ball from the **ATD** | the video tracks with the T5 team model give cleaner teams (team over 11 players on 10.3% of frames vs 12.9%) and fewer very short possessions (23 vs 30); the video pipeline has no ball, and our YOLO ball was worse than the ATD ball (section 8) |
| Control radius 2.0 m | the ball sits about 1 m from the player on it (measured, counts only); a 1.5 m circle missed about 40% of on-the-ball frames on the bench |
| Hysteresis 0.6 m and smoothing ±3 frames | on the bench, 40% of the earlier version's possession changes were false (A-B-A blips and late switches); these two settings reduced them |
| No "ball present + players slow = dead" rule | it marked slow build-up as dead; dropping it improved all 6 realistic bench copies |
| Relative player speed, not m/s | this match's players are slow (median 1.9 m/s in play), so absolute thresholds fired everywhere |
| `MIN_D` 2 s | very short dead stretches were noise; improved all 12 bench cells |
| Changes dated back to the first frame of control | `spellF1` punishes lateness: 1 s late costs about a third of a possession's score |
| No `X` on graded frames | an `X` is never better than repeating the last label under `spellF1`, and it is a miss under `macroF1` |

---

## 7. How we tuned without touching this match

RULES.md does not allow tuning on this match, not even on the public 5 minutes. All settings were
chosen on a **bench of other matches**:

| Bench | Data | Truth |
|---|---|---|
| Metrica | the three public Metrica Sports sample games, with copies degraded to this match's measured noise (ball gaps, fragmented tracks, ~1 m ball offset; `test_scripts/noise_profile.py` measures counts only) | (a) from the Metrica events, (b) from the clean Metrica tracking |
| DFL / IDSSE | 7 Bundesliga matches of the DFL open data (`test_scripts/bench/fetch_dfl_bench.py`) | their own possession and ball-status channels |

v6 on the bench (spellF1): **0.510** (Metrica, event truth), **0.560** (Metrica, tracking truth),
**0.464** (DFL). The organisers' starter baseline on the full match is 0.2433. The degraded copies were
calibrated so that the starter baseline ranks the same way on them as on the organisers' board; only the
published baseline scores were used for that. A change was kept only if it did not lose on either truth.

The public 5 minutes were used only to check frames, format and meaning, as RULES.md allows.
On this match we only ever compared runs with each other ("how much and where do two files differ"),
never against a truth.

---

## 8. What we tried and dropped

| Idea | Result | Why dropped |
|---|---|---|
| Players from the provided ATD | valid, fully reproducible; 305 possessions | replaced by the video tracks (section 6); kept as `test_scripts/run_labeller_v6.py` |
| Video tracks with colour team models (before T5) | 395 possessions, team over 11 players on 52.6% of frames | team errors; fixed by T5 |
| YOLO ball (YOLOv8 + SAHI) to fill the ATD ball gaps | median 1.7 m from the ATD ball where both exist; filled gaps were 87% stoppages, so stoppages were split (95 to 112) | its error is close to the 2 m control radius, and it removed the main stoppage evidence |
| Referee pointing gestures (pose model on the broadcast) | referee visible on 28% of scanned frames; 10 gestures for 95 stoppages | too sparse to change any label |
| Ball gap interpolation (Hermite, gaps up to 1 s) | about 7 gaps filled; most short gaps are tracker jumps | v6 only treats gaps of 1.5 s or more as stoppages, so no effect |
| Hand-over detector (labeller v5: reception vote, density switch) | gains on one bench truth only | lost on the other truth |
| Longer `CONFIRM`, `MIN_SPELL` 1.5 s, reception vote, faster switching | -0.007 to -0.029 on the tracking truth | lost on the bench |
| Rule E (long slow stretch = dead) | -0.007 to -0.014 | split real possessions |
| Pass detection (kick and reception events from the ball track) | never reliable enough to feed the labeller | not used; its code is not in this repository |

---

## 9. Compliance with RULES.md

- **Algorithmic labels only.** Every label comes from `make_submission.py`; nothing is edited by hand.
- **No hand labels**, on this match or any other. The team model's training labels are pseudo-labels
  from the provided ATD (automatic data). The video was watched only to understand errors, never to make
  labels, rules, constants or a choice of version.
- **No human-produced data about this match** (no event data, no match statistics, no corrected tracking).
  Tuning used other matches only (section 7).
- **No data in the repository.** Match files, tracks, label files and video clips are git-ignored.
- **LLMs and pre-trained models.** We used Claude (Anthropic) as a coding assistant: it wrote scripts
  that we ran on our own machines; no video, tracking or label file was given to it, only printed run
  summaries (counts, percentages and frame numbers). Pre-trained models: the YOLO11m player detector (trained by a third party on
  372 public Roboflow images of other footage) and SigLIP2 (section 2.3). The video was processed in our own private Kaggle notebooks, not by any LLM.


---

## 10. Limitations and next steps

- **Ball.** Possession changes are only seen where the ATD ball is tracked; on 41% of graded frames it is
  missing, and long live gaps keep the last team until the ball reappears.
- **Restart timing.** When the tracker jumps between detections right after a gap, the jump can look like
  a kick and end a stoppage too early. Rejecting implausible jumps before the restart test is the first
  thing we would try, on the bench.
- **Who restarts.** After a stoppage the first team to control the ball gets it; a throw-in side model
  (team shape in its own half) looked promising but was not tested on the bench.
- **Video tracks.** v6 was tuned on ATD-like noise; the video tracks were not benched (there is no video
  for the bench matches). About 5% of frames swap team between the ATD-player run and the video-track run,
  and no label-free check can say which is right.
- **Reproducibility of step 1.** GPU inference may differ slightly between machines (section 2.3).
