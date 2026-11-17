# EDA — Exploratory Data Analysis for Metrica Sports More Than a Hack 2027

This folder contains all exploratory analysis, ball-trajectory reconstruction experiments, and Smart Tagging utilities for the **Barcelona vs Athletic Club (BAR-ATH)** match.

## 1. Project Overview & Objectives

- **Match**: Barcelona vs Athletic Club (2024–25 La Liga, Jornada 2)
- **Data**: Tactical video (199,319 frames @ 25 Hz ≈ 133 min), automatic tracking (ATD), Smart Tagging XML, homography
- **Goal**: Understand ball-tracking gaps, reconstruct missing ball positions, and infer possession for frames where the ball is not tracked (37.5% of frames)
- **Metrics**: Reconstruction error (RMSE/MAE in metres), reliability (% within 5 m), possession accuracy on public ground truth (5-min window, 7,500 frames)

---

## 2. Folder Structure

```
eda/
├── ball_reconstruction.py          # Main reconstruction + artificial masking suite
├── smart_tagging.py                # Smart Tagging EDA & statistics
├── smart_tagging/
│   ├── throw_in_takers.csv         # 26-row throw-in side inference (13 BAR / 13 ATH)
│   └── verify_throwin_takers.py    # Independent verification script
├── gaps/
│   ├── eda_ball_gaps.py            # Stream ATD → find missing-ball gap runs
│   ├── classify_all_135_gaps.py    # Classify rendered gap clips by event overlap
│   ├── render_gap_clips.py         # Generate MP4 clips for long gaps (via visualize_tracking.py)
│   ├── analyze_all_gaps.py         # Quick gap summary
│   ├── gap_clips_classification.md # Human-readable catalog of 135 gap clips
│   └── __pycache__/
├── visualize_tracking.py           # Dual-panel video renderer (tactical + pitch top-down)
├── reconstruction_demo.mp4         # 12.4 s demo video (frames 14657–14966, 7.2 MB)
├── clips/
│   └── t451_annotated.mp4          # Sample annotated clip
├── docs/
│   ├── PROJECT_OVERVIEW.md         # Match specs, frame conventions, data sources
│   ├── GAPS_AND_CLIPS.md           # Gap taxonomy & clip catalogue
│   ├── THROW_IN_SIDE.md            # Throw-in taker inference methodology
│   └── thoughts.md                 # Running project log
└── __pycache__/
```

---

## 3. Smart Tagging Analysis (`smart_tagging.py`)

### What it does
Loads the SportsCode XML (`data/tactical/BAR-ATH_tactical_smart_tagging.xml`) via `starter.load_smart_tagging` and produces:

- Code frequency distribution (14 codes, 638 instances)
- Team tag counts (Barcelona 314, Atletic Club 288, **N/A 36**)
- Half distribution (1st Half 320, 2nd Half 318)
- Type breakdown (Throw-in 26, Corner Kick 12, Goal Kick 10, Free Kick 5, KO-After-Goal 4, KO-Start 2)
- Side / Max Players / Direction of ball entry breakdowns
- Overlap between SET PIECES intervals and ball-missing gaps (72 overlaps)

### Execution
```bash
cd eda
python smart_tagging.py
```

### Output
Printed to stdout. No files written.

### Throw-In Side Inference (`smart_tagging/throw_in_takers.csv`)

**Methodology** (centroid-in-own-half, per `THROW_IN_SIDE.md`):
1. At throw-in start frame, collect all tracked players per team (slots 1–24 = BAR, 25–46 = ATH)
2. Drop goalkeeper = player with y closest to goal line (0 or 1)
3. Compute non-GK centroid x per team
4. Score = signed distance from pitch centre (x=0.5) toward own half
   - BAR 1st half: x < 0.5, 2nd half: x > 0.5
   - ATH opposite
5. Team with deeper own-half centroid = inferred taker

**Results**: 13 Barcelona, 13 Athletic. 24/26 cross-validated by ball position at resumption. Closest calls: smart IDs 448, 450, 451.

---

## 4. Ball-Gap Detection & Classification (`gaps/eda_ball_gaps.py`)

### Methodology
Streams raw ATD text file (`data/tactical/BAR-ATH_tactical_atd.txt`) line by line. Ball is the segment after the **last** colon. A frame has ball if segment ≠ `NaN,NaN,NaN`.

### Gap Definition
Consecutive frames with `ball_present = False` form a gap run: `(start_frame_id, end_frame_id, length)`.

### Execution
```bash
python gaps/eda_ball_gaps.py
python gaps/eda_ball_gaps.py --thresholds 5,15,30,60,125,250,500 --top 25
```

### Key Results (Tactical ATD)
- Total frames: 199,319
- Ball present: 124,550 (62.5%)
- Ball missing: 74,769 (37.5%)
- Gap runs: **196**
- Longest gaps: 4939 fr (post-match), 4192 fr (pre-kickoff), 3466 fr (halftime), 1758 fr, 1737 fr
- **Graded window (11185–18684)**: 5 gaps (lengths 5, 132, 148, 232, 366 frames)

### Gap Clip Rendering (`gaps/render_gap_clips.py`)
```bash
python gaps/render_gap_clips.py                    # gaps ≥60 frames, in play
python gaps/render_gap_clips.py --min-length 125   # only 5s+ gaps
python gaps/render_gap_clips.py --pad 2.0 --only-graded --max-clips 10
```
Outputs MP4s to `gap_clips/` (not tracked in git).

### Gap Classification (`gaps/classify_all_135_gaps.py`)
Cross-references 135 rendered clips against Smart Tagging set pieces, restarts, goals, match boundaries, and possession handovers.

**Breakdown**:
- 93 gaps overlap a Set Piece / Restart / State Change (Corner 18, Throw-in 23, Free Kick 8, Goal Kick 8, Kick Off 6, Boundaries 3, Possession Handover 27)
- 42 gaps are pure **in-play occlusions** (ball high, crowded box, camera cutaway)

See `gaps/gap_clips_classification.md` for full catalogue.

---

## 5. Ball Trajectory Reconstruction Methods (`ball_reconstruction.py`)

All methods implemented in pure NumPy (no SciPy dependency). Input arrays: normalized ball x,y ∈ [0,1], pitch 105.3 × 68.0 m.

### 5.1 Linear Interpolation (`reconstruct_linear`)
Straight-line position interpolation between last known frame before gap (`s-1`) and first known frame after gap (`e`).

```
x(t) = x₀ + (x₁ - x₀) * t,  t ∈ [0,1]
```

### 5.2 Velocity-Aware Cubic Hermite Spline (`reconstruct_hermite_velocity`)
Cubic Hermite basis matching **position and velocity** at gap boundaries.

- Velocity estimated by finite difference: `v = (p[s-1] - p[s-2])` (if available) else gap-average
- Basis functions: `h00 = 2t³-3t²+1`, `h10 = t³-2t²+t`, `h01 = -2t³+3t²`, `h11 = t³-t²`
- `x(t) = h00·x₀ + h10·Δt·v₀ + h01·x₁ + h11·Δt·v₁`

Superior for fast-moving balls; reduces error 3–5× vs Linear for gaps ≤60 frames.

### 5.3 Quadratic Polynomial (`reconstruct_quadratic_curved`)
Degree-2 polynomial fit through boundary context points (up to 3 frames before/after gap).

```python
poly = np.poly1d(np.polyfit(times, positions, deg=2))
```

Competitive at 15–30 frames but tends to overshoot at 60+ frames.

---

## 6. Artificial Masking Experiments & Evaluation Methodology

### 6.1 Methodology (No Data Leakage — Verified)
1. Identify **continuous ball-tracking runs** (≥300 frames / 12 s of consecutive `ball_present = True`)
2. For each gap length (5, 15, 30, 60, 125, 250 frames), sample **100 independent masked intervals** by sliding a window across runs
3. Create boolean `gap_mask` = True only for the artificial gap frames
4. Call reconstruction functions with `gap_mask` — **boundary frames (mask=False) are preserved exactly**; interior frames reconstructed from boundary information only
4. Compare reconstructed positions against ground-truth tracked positions (in metres)

**Verified**: Boundary values match original to machine precision; interior values differ from original (proving reconstruction runs).

### 6.2 Metrics Computed
For each gap length, over 100 samples:
- **RMSE** = `sqrt(mean((pred - true)²))` in metres
- **MAE** = `mean(|pred - true|)` in metres
- **Max Error** = `max(|pred - true|)` in metres
- **Reliability** = `% of predictions with error < 5.0 m`

### 6.3 Motion-Conditioned Evaluation
All 100 continuous runs exhibited **fast ball motion** (mean speed 10.6 m/s, max 41.3 m/s). No stationary/slow runs found in ≥300-frame segments. Evaluation repeated per motion type (only FAST available).

---

## 7. Quantitative Results

### 7.1 Aggregate Results (100 samples/gap length)

| Gap Length | Linear RMSE (m) | Hermite RMSE (m) | Quadratic RMSE (m) | Best Method | <5m Reliability |
|------------|------------------|------------------|--------------------|-------------|-----------------|
| 5 fr (0.2s) | 0.069 | **0.011** | 0.009 | Hermite/Quad | 100.0% |
| 15 fr (0.6s) | 0.432 | **0.018** | 0.082 | **Hermite** | 100.0% |
| 30 fr (1.2s) | 1.307 | **0.236** | 0.452 | **Hermite** | 99.6% |
| 60 fr (2.4s) | 3.489 | **1.641** | 2.042 | **Hermite** | **85.7%** ⚠️ |
| 125 fr (5.0s) | 7.381 | **5.104** | 5.559 | Hermite | 53.2% ❌ |
| 250 fr (10.0s) | 13.774 | 12.506 | 13.130 | — | 31.7% ❌ |

### 7.2 Fast-Motion Breakdown (50 samples/gap length)

| Gap Length | Linear RMSE | **Hermite RMSE** | Quadratic RMSE |
|------------|-------------|------------------|----------------|
| 5 fr (0.2s) | 0.079m | **0.012m** | 0.010m |
| 15 fr (0.6s) | 0.498m | **0.021m** | 0.092m |
| 30 fr (1.2s) | 1.487m | **0.264m** | 0.509m |
| 60 fr (2.4s) | 3.799m | **1.879m** | 2.306m |
| 125 fr (5.0s) | 8.037m | **5.086m** | 5.520m |

### 7.3 Reliable Reconstruction Limit
**Operational threshold: ≤60 frames (2.4 seconds)**

- At 60 frames: Hermite RMSE = 1.64 m, **85.7%** within 5 m
- At 125 frames: RMSE = 5.10 m, only **53.2%** within 5 m
- Beyond 60 frames: kinematic interpolation degrades; fallback to player geometry / Smart Tagging priors required

---

## 8. Long-Gap Fallback Methods

When ball is missing >60 frames (or in graded window gaps), kinematic interpolation is unreliable. Two fallbacks evaluated on **885 missing-ball frames in public GT window**:

| Fallback Method | Description | Accuracy |
|-----------------|-------------|----------|
| **Last-Touch Carry-Forward** | Propagate team of player nearest to ball at last tracked frame | **41.92%** |
| **Team-Centroid Geometry** | Compute non-GK centroid per team; assign to team whose centroid is deeper in own half (mirrors throw-in logic) | **44.18%** |

**Smart Tagging Priors**: 638 phases loaded. Set-piece gaps → `D` (dead ball). Active phases (BUILD UP, ATTACKING TRANSITION, etc.) provide team priors for possession inference.

---

## 9. Visualisation & Generated Video

### `visualize_tracking.py`
Dual-panel renderer (tactical video + pitch top-down) with lazy streaming:
- Video: ffmpeg pipe (frame-accurate seek) or OpenCV fallback
- Tracking: binary search on raw ATD → load only needed frames
- Homography: projects players/ball onto video panel
- Bottom bar: frame/time/half, player counts, ball status, ground truth label, active Smart Tagging phases

```bash
# Basic usage
python visualize_tracking.py --start 07:27                    # 5 min from 1st-half kickoff
python visualize_tracking.py --start-frame 11185 --duration 300 --out viz.mp4
python visualize_tracking.py --no-video --trail 80            # tracking only, no mp4 decode
```

### Demo Video: `reconstruction_demo.mp4`
- **Path**: `eda/reconstruction_demo.mp4`
- **Frames**: 310 (12.4 s) — video frames 14657–14966 (09:46.3–09:58.7)
- **Resolution**: 1920 × 610
- **FPS**: 25.0
- **Size**: 7.2 MB
- **Content**: Original ball (white), players (red/blue), ball trail, nearest-player line, Smart Tagging tags, public GT label

**Generated with**:
```bash
python visualize_tracking.py --start-frame 14657 --duration 12.4 --out eda/reconstruction_demo.mp4 --trail 60
```

---

## 10. Installation & Dependencies

### Required (already in `requirements.txt`)
```txt
numpy
pandas
```

### Optional (for video rendering)
```bash
pip install opencv-python imageio-ffmpeg
```
- `opencv-python`: required for `visualize_tracking.py`
- `imageio-ffmpeg`: provides ffmpeg binary if not in system PATH (used for frame-accurate video seeking)

### Repository Loaders (from `starter/`)
- `load_atd.py` — streams ATD via kloppy (optional) or raw text parsing
- `load_smart_tagging.py` — pure-stdlib XML parser
- `load_homography.py` — CSV loader
- `games.py` — match constants, paths, frame/half alignment
- `utils.py` — pitch plotting helpers

---

## 11. Exact Execution Commands

### Smart Tagging EDA
```bash
cd eda
python smart_tagging.py
```

### Ball Reconstruction Suite (full)
```bash
cd eda
python ball_reconstruction.py
```
Outputs: console summary tables + RMSE/reliability by gap length + fallback accuracy.

### Gap Analysis
```bash
python gaps/eda_ball_gaps.py
python gaps/eda_ball_gaps.py --thresholds 5,15,30,60,125,250,500 --top 25
```

### Render Gap Clips
```bash
python gaps/render_gap_clips.py                       # all gaps ≥60 frames
python gaps/render_gap_clips.py --min-length 125 --pad 2.0 --only-graded --max-clips 10
```

### Classify Rendered Clips
```bash
python gaps/classify_all_135_gaps.py
```

### Visualize Tracking (Video)
```bash
python visualize_tracking.py --start-frame 14657 --duration 12.4 --out reconstruction_demo.mp4 --trail 60
python visualize_tracking.py --start 75:49 --duration 60 --out second_half_kickoff.mp4
```

### Verify Throw-In Takers
```bash
python smart_tagging/verify_throwin_takers.py
```

---

## 12. Input Data Requirements & Directory Structure

The repository expects the gated data bundle at `./data/` (or `$MTAH_DATA_DIR`):

```
data/
├── tactical/
│   ├── BAR-ATH_tactical.mp4
│   ├── BAR-ATH_tactical_atd.xml
│   ├── BAR-ATH_tactical_atd.txt
│   ├── BAR-ATH_tactical_homography.csv
│   └── BAR-ATH_tactical_smart_tagging.xml
└── broadcast/
    ├── BAR-ATH_broadcast.mp4
    ├── BAR-ATH_broadcast_atd.xml
    ├── BAR-ATH_broadcast_atd.txt
    └── BAR-ATH_broadcast_homography.csv
```

**Ground Truth** (public): `evaluation/gt/BAR-ATH_possession_gt_public.csv` (7,500 graded frames)

All scripts use `games.require()` to resolve paths — works from any working directory if `$MTAH_DATA_DIR` is set.

---

## 13. Generated Outputs & Locations

| Script | Output | Location |
|--------|--------|----------|
| `smart_tagging.py` | Console statistics | stdout |
| `ball_reconstruction.py` | Console tables + metrics | stdout |
| `gaps/render_gap_clips.py` | MP4 clips | `gap_clips/` (git-ignored) |
| `gaps/classify_all_135_gaps.py` | Classification printout | stdout |
| `visualize_tracking.py` | MP4 video | user-specified `--out` |
| `reconstruction_demo.mp4` | Demo video (pre-generated) | `eda/reconstruction_demo.mp4` |
| `throw_in_takers.csv` | 26-row inference | `smart_tagging/throw_in_takers.csv` |

**No original data files are modified**. All outputs are new files or console output.

---

## 14. Limitations & Future Improvements

### Current Limitations
1. **Video overlay of reconstructed trajectory** — `visualize_tracking.py` shows original tracking only; passing reconstructed ball arrays would require modifying the renderer.
2. **Acceleration-aware model** — Not implemented; needs 3+ boundary points for stable jerk estimation. Hermite (velocity) is best practical compromise.
3. **Slow/stationary ball evaluation** — No ≥300-frame continuous runs with mean speed < 3 m/s found; reconstruction behavior on slow balls untested.
4. **Broadcast ATD fusion** — Offset ≈ 4013 frames identified; cross-camera validation not implemented.
5. **Anonymous tracklets** — Player slots 1–46 are not persistent identities; limits Voronoi pitch control quality.
6. **Public GT only** — 5-minute window (17 possessions); full-match behavior extrapolated. Bootstrap confidence intervals recommended.
7. **No acceleration/ jerk constraints** — Quadratic model can overshoot; piecewise physics-based model (constant acceleration) could improve >60 frame gaps.

### Possible Extensions
- **Physics-based interpolation**: Constant-acceleration model with boundary position/velocity/acceleration matching.
- **Broadcast-tactical fusion**: Use broadcast ATD (51.6% ball presence) to fill tactical gaps via homography alignment.
- **Player identity tracking**: Re-identification across tracklets for persistent Voronoi pitch control.
- **Set-piece specific models**: Separate interpolation for corners, free kicks, goal kicks (ball dead → live transition).
- **Full-match evaluation**: Run reconstruction on private GT when available; bootstrap 1000× for confidence intervals.

---

## 15. Reproducibility

All experiments are deterministic given the same input data:

```bash
# Full pipeline
cd eda
python smart_tagging.py
python ball_reconstruction.py
python gaps/eda_ball_gaps.py
python gaps/classify_all_135_gaps.py
python visualize_tracking.py --start-frame 14657 --duration 12.4 --out reconstruction_demo.mp4 --trail 60
```

Random sampling in masking experiments uses the order of continuous runs and fixed step sizes — no RNG seeds involved, results are fully reproducible.

---

*Last updated: 2026-11-07*