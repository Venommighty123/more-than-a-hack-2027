# Player Detection & Tracking — Kaggle Build Plan

Nov 11, 2026 · Kushagra Mathur

## Goal and scope

One Kaggle notebook detects, tracks and team-labels every player, goalkeeper and referee in the tactical video, for the two halves only: **158,857 frames**. It produces several tracking variants and a scorecard comparing each one with the existing ATD.

| Segment | First frame | Last frame | Frames | Processed? |
| --- | --- | --- | --- | --- |
| Pre-match | 0 | 11,184 | 11,185 | No |
| 1st half | 11,185 | 90,651 | 79,467 | Yes, including dead-ball stretches |
| Halftime | 90,652 | 110,807 | 20,156 | No |
| 2nd half | 110,808 | 190,197 | 79,390 | Yes, including dead-ball stretches |
| Post-match | 190,198 | 199,318 | 9,121 | No |

Source: `DATA.md` / `games.half_video_blocks("BAR-ATH")`. Each half is processed with 25 extra warm-up frames before its first frame so the tracker has stable IDs at kickoff. Those warm-up rows are dropped from the outputs.

**Out of scope for now:** ball detection (the `BALL_MODEL` and SAHI code are removed) and the pitch-keypoint model. The provided homography CSV replaces the pitch model.

## Constraints

The hard rule: **no hand labels on this match, for any purpose** (`RULES.md`). Every team decision and correction has to come from code.

- **Models:** the original player detector was trained on other games, which is allowed. Record its training data in `docs/tech_doc.md`.
- **Team naming:** clusters are mapped to Home (Barcelona) or Away (Athletic) using the ATD team membership (`idLocalTeam` in `atd.xml`). Nobody picks by eye.
- **Goalkeepers:** assigned by side. At each half's kickoff, the team whose players sit on the left defends the left goal, so the left goalkeeper belongs to them. This is recomputed for the 2nd half because the teams swap ends.
- **Public ground truth** (frames 11,185–18,684) is for sanity checks only. We don't tune or choose variants on it.
- **Watching clips is allowed, but turning what we see into labels or fixes is not.** Visual review is a sanity check, never a data source.
- **Kaggle:** GPU T4 ×2, 12-hour session limit, about 20 GB of `/kaggle/working` output, 4 CPU cores, and 30 GPU-hours per week.

## Kaggle inputs

You upload three private Kaggle datasets. The notebook mounts them read-only under `/kaggle/input/`.

| Dataset (suggested name) | Files | Size | Used for |
| --- | --- | --- | --- |
| `bar-ath-tactical` | `BAR-ATH_tactical.mp4` | 5.2 GB | Video frames |
|  | `BAR-ATH_tactical_atd.txt`, `BAR-ATH_tactical_atd.xml` | 196 MB | ATD baseline, slot → team map, Home/Away mapping |
|  | `BAR-ATH_tactical_homography.csv` | 23 MB | Pixel → metre projection, pitch-plane tracking |
|  | `evaluation/gt/BAR-ATH_possession_gt_public.csv` | small | Sanity check only |
| `supa-strikers-models` | Original **player** detector `.pt` (not the combined `last.pt`) | ~50–150 MB | Detection |
| `supa-strikers-outputs` | Optional: a previous run's /kaggle/working, saved as a dataset, only if a session ends before detection finishes | grows | Resuming across sessions |

**Notebook settings:** Accelerator GPU T4 ×2, Internet on (to pip install `supervision`, `umap-learn`, `transformers` and pull SigLIP weights), Persistence: Files.

**Not needed:** the broadcast video, the smart-tagging XML, `BALL_MODEL`, the pitch model and the combined `last.pt`. The old `app/` modules are folded into the notebook, so you don't upload any code.

## What gets built

The GPU runs detection **once**. Everything after that works on cached detections on the CPU, so trying another tracker or team classifier costs minutes, not hours.

0. **Config:** paths, the half ranges, chunk size (5,000 frames), confidence thresholds and variant switches. Everything you'd want to change sits in this one cell.
1. **Setup:** pip installs, a GPU check (expects 2 × T4), and the player model's class names printed and checked.
2. **Load metadata:** parse the ATD into one row per frame per slot with team (H/A), load the homography, and build a helper that converts pixel coordinates to metres.
3. **Video sanity check:** confirm the frame count is 199,319, and test that seeking to a frame lands on that exact frame by comparing it with a sequential read at 5 spots. If seeking proves unreliable, fall back to PyAV.
4. **Calibration (GPU, ~10 min):** take ~150 frames spread over both halves and collect torso crops of the detected players. Fit the three team classifiers (T1–T3), then map each cluster to H/A by matching to ATD positions. Saves crop grids and cluster plots for review.
5. **Detection pass (GPU ×2, resumable):** split the 158,857 frames into ~32 chunks shared between two workers, one per GPU. For each frame, save every box, its confidence and class. Every 3rd frame, also save the team scores from T1–T3 for each box. Each chunk writes its own Parquet file, and chunks already done are skipped.
6. **Tracking (CPU):** run K1, K2 and K3 over the cached detections, one half at a time.
7. **Team and role (CPU):** take a majority vote of team and class over each track's life, apply the side rule for goalkeepers, and keep referees separate.
8. **Post-processing (CPU):** join broken tracks (same team, gap of 3 s or less, distance of 5 m or less), smooth paths (Savitzky–Golay), and project feet positions to metres.
9. **Evaluation:** build the scorecard comparing ATD and every variant, with plots (see How to compare).
10. **Clips (optional):** six 20-second side-by-side MP4s (ATD vs best variant) taken from the stretches where they disagree most.
11. **Export:** the best variant's tracks, the scorecard and the plots, written to `/kaggle/working`.

## Variants to compare

We compare 3 trackers × 3 team classifiers = **9 variants**, each with and without post-processing, against the ATD baseline. Detection is shared, so only the tracking and team stages differ.

| ID | Stage | Variant | What it tests |
| --- | --- | --- | --- |
| K1 | Tracker | `sv.ByteTrack()` defaults | Baseline, the same as the old script |
| K2 | Tracker | ByteTrack, tuned: `lost_track_buffer=100` (4 s), `frame_rate=25`, detection confidence 0.1, track activation 0.4, matching threshold 0.8 | Whether a longer memory and low-confidence re-association cut ID jumps |
| K3 | Tracker | Kalman + Hungarian matching in **pitch metres** via homography (gate of 2.5 m) | Removes camera pan and zoom from the motion model, the likely main cause of jumps |
| T1 | Team | Your `DeepTeamClassifier` features (HSV histogram + ResNet18), **fit once and frozen** | Your existing approach without the per-frame refit |
| T2 | Team | SigLIP embeddings + UMAP + KMeans (Roboflow sports style) | Stronger embeddings |
| T3 | Team | HSV histogram only + KMeans | Cheap baseline: is the ResNet adding anything? |
| ATD | Baseline | Metrica slots with fixed team per slot | The reference we have to beat |

If the tactical camera is mostly static, K2 and K3 will score close together, and K2 wins because it's simpler.

## How to compare

There's no ground-truth tracking for this match, and we can't make any. So the winner is picked on **label-free** metrics that ATD and our variants can all be scored on, computed in metres over the same 158,857 frames.

### A. Scorecard (one row per source: ATD, K1T1 … K3T3)

| Metric | How it's computed | Better | Catches |
| --- | --- | --- | --- |
| Coverage | % of frames with 20–23 people tracked (players + goalkeepers) | Higher | Missed detections |
| Team overflow | % of frames where one team has more than 11 (H or A) | Lower | Wrong team labels |
| Goalkeeper sanity | % of frames with exactly 1 goalkeeper per team inside its own half | Higher | Goalkeepers misclassified or put on the wrong team |
| Jump rate | Moves faster than 12 m/s measured over 0.2 s, per 1,000 steps (robust to per-frame jitter) | Lower | ID swaps, teleports |
| Fragmentation | Unique IDs per half, and median track length in seconds | Fewer, longer | Broken tracks |
| Team flip rate | % of track-frames whose per-frame team disagrees with the track's vote (ours only) | Lower | Unstable classifier |
| Off-pitch rate | % of positions more than 2 m outside 105.3 × 68 m | Lower | False positives, crowd, staff |

### B. ATD vs ours, head to head

Each frame, match ATD positions to ours (Hungarian matching, 2 m gate) and report: match rate, median distance, and **team agreement %**. Disagreement is aggregated per minute into a heat strip. The minutes where they disagree most become the review clips.

### C. Visual sanity check

Six 20-second side-by-side clips: ATD dots on the left, ours on the right, ID and team drawn on both. This is for spotting systematic failures only. Nothing seen there becomes a label or a fix.

### D. Downstream check (last, not for selection)

Run the existing possession pipeline once with the chosen source on the public window (frames 11,185–18,684) to confirm nothing broke. We don't pick a variant on this score.

**Decision rule:** a variant replaces ATD only if it beats ATD on team overflow, jump rate **and** coverage. Otherwise we keep ATD and use our tracks only to fill gaps where ATD is missing or jumps.

## Outputs and schema

All outputs are Parquet, one row per frame per object, written to `/kaggle/working/`.

| File | Columns | Approx. size |
| --- | --- | --- |
| `raw_dets/chunk_XXXX.parquet` | `frame, x1, y1, x2, y2, conf, cls, t1_p, t2_p, t3_p` (team scores only on every 3rd frame) | ~1.5 GB total |
| `calib/` | Fitted T1–T3 models, cluster → H/A map, crop grids, cluster plots | <100 MB |
| `tracks/{K}{T}.parquet` | `frame, track_id, role (player/gk/ref), team (H/A/None), conf, x1, y1, x2, y2, feet_px, feet_py, x_m, y_m` | ~150 MB each |
| `final_tracks.parquet` | The same schema, for the winning variant only | ~150 MB |
| `eval/scorecard.csv` + `eval/*.png` | Metrics in A and B, disagreement heat strip | small |
| `clips/*.mp4` | Side-by-side review clips | ~200 MB |

`x_m`/`y_m` use the same origin and orientation as the ATD multiplied by 105.3 × 68, so the possession code can read either source unchanged.

## Runtime plan on T4 ×2

The whole run should fit in **one 12-hour session**: estimated at about 3–5 hours, so there's roughly 2× headroom. These are estimates. Calibration measures real throughput on 500 frames and prints the projected total before the long pass starts.

| Stage | Where | Estimated time | Driver |
| --- | --- | --- | --- |
| Calibration | 1 GPU | ~10 min | ~150 frames, SigLIP + ResNet on ~3,000 crops |
| Detection pass | 2 GPUs in parallel | ~2–3.5 h | 158,857 frames at ~15–20 fps per GPU, FP16, batch 16; video decoding on the CPU is the likely bottleneck |
| Team scores | Same pass | +30–45 min | ~1.3 M crops (every 3rd frame) × 3 classifiers, batched |
| Tracking K1–K3 | CPU | ~20 min | Cached detections only |
| Team/role, post-processing, evaluation | CPU | ~20 min | 9 variants × 2 |
| Clips | CPU | ~10 min | 6 × 500 frames |

**Safety nets:** each finished chunk saves its own Parquet file, so a crash loses at most one chunk (about 5 minutes of work). If the session ends early, save the version, attach `/kaggle/working` as the `supa-strikers-outputs` dataset, and rerun. Finished chunks are skipped. Trackers run over each whole half from the cached detections, so chunk boundaries leave no seam.

## Fixes to the inherited code

| # | Problem in old code | File | Fix in notebook |
| --- | --- | --- | --- |
| 1 | KMeans refits as players lock in, so labels can drift; new IDs have no team for their first 5 samples | `team_classifier.py` | Fit once at calibration on both halves, freeze, then a per-track vote |
| 2 | Goalkeepers never get a team | `model_output.py` | Side rule at each kickoff |
| 3 | Default ByteTrack (short lost-track memory) and detection confidence 0.3, which skips ByteTrack's low-confidence second pass | `tracker.py` | K2 tuning: confidence 0.1, activation 0.4, buffer 100 |
| 4 | Player and goalkeeper class flicker frame to frame | `model_output.py` | Majority vote of class per track |
| 5 | ResNet runs on one crop at a time | `team_classifier.py` | Batched crops, FP16 |
| 6 | Fast-forwards by reading every frame; `cv2.imshow` crashes on Kaggle | `model_output.py`, `video_io.py` | Seek with a check, no GUI |
| 7 | Pickling the tracker for resume can fail and resets IDs | `model_output.py` | Chunk-level resume; tracking re-runs from cached detections |
| 8 | `if track_id` hides ID 0 in labels | `drawing.py` | `is not None` check |
| 9 | `app.settings` vs relative imports mismatch | all | One self-contained notebook |
| 10 | Ball / SAHI / pitch model code | all | Removed for now |

## Decisions (Nov 11)

- **Player model:** YOLOv8, pretrained. Class names are read from the model at runtime; ball and pitch classes are ignored. `IMGSZ` defaults to 1280.
- **tech_doc.md:** to be updated later.
- **Camera:** mostly fixed, with some pan and zoom, so K3 (tracking in pitch metres) stays in the comparison.
- **Referees:** detected and kept as their own role. They feed two extra signals: referees per frame, and ATD slots that sit on one of our referees with no player of ours nearby (ATD tracking a referee as a player).
- **Notebook:** `kaggle/kaggle_player_tracking.ipynb`. Run it in smoke mode first, then full.
