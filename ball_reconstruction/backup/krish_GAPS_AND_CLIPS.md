# Tracking Gaps & Rendered Gap Clips

In the tactical ATD tracking data for BAR-ATH (199,319 frames total), the ball is missing
in ~74.7k frames (37.5%) across 196 missing-ball gap runs.

## Key Gap Analysis Scripts & Outputs
- `eda/eda_ball_gaps.py`: Stream-scans raw ATD file for runs where the ball is
  `NaN,NaN,NaN`, counts thresholds, and matches against graded half blocks.
- `eda/render_gap_clips.py`: Renders `.mp4` video clips in `gap_clips/` for gaps
  exceeding specified thresholds (e.g. >=60 frames / 2.4s) with padded context
  before/after.
- `eda/classify_all_135_gaps.py`: Script that cross-references all 135 rendered gap clips
  against Smart Tagging set pieces, restarts, goals, match boundaries, and possession
  handovers.
- `eda/gap_clips_classification.md`: Full catalog mapping each gap clip to its restart
  event or classifying it as an in-play occlusion.
- `gap_clips/`: Directory containing 135 rendered MP4 clips of long ball-tracking gaps
  (named e.g. `gap_0058031_03466.mp4`).

## Gap Classification Breakdown (135 Gap Clips)
1. **93 Gaps HAD a Set Piece, Restart, or State Change**:
   - **Corner Kicks**: 18 gaps
   - **Throw-ins**: 23 gaps
   - **Free Kicks**: 8 gaps
   - **Goal Kicks**: 8 gaps
   - **Kick Offs (Start / After Goal)**: 6 gaps
   - **Match / Half Boundaries**: 3 gaps (Pre-kickoff, Halftime, Full Time)
   - **Possession Handovers / Phase Changes**: 27 gaps
2. **42 Gaps DID NOT HAVE a Set Piece or State Change**:
   - Pure **in-play occlusions** (ball high in air, crowded box, temporary broadcast/camera
     occlusion while play continued continuously).

## Gap clip naming
Clip names encode a padded start and length, e.g. `gap_0012781_00148.mp4` =
video frame 12781, 148 frames. The padding means the clip start does **not** equal the
real ATD gap start; match each clip to its actual gap run by scanning `find_gaps()` for
the run nearest `start + pad_frames`.

## The 42 "C — No overlap" gaps (in-play occlusions)
These overlap no Smart Tagging interval — the cleanest case for a `D` (dead-ball) fill,
since the ground truth also labels those frames `D`. They are spread across the whole
match, from 8:33 (1st half) to ~121:00 (2nd half), and include the two longest in-play
occlusions: ~53s at 46:47 and ~69s at 57:44. Full list with real ATD frame ranges and
match clocks is in `gap_clips_classification.md` (section 2).

## Why
Avoid re-scanning ATD lines and re-calculating gap clip parameters every time gap analysis
or baseline strategy is updated.

## How to apply
Refer to this context when analyzing possession handovers, handling missing ball tracking,
or auditing restart transitions. Link with [[PROJECT_OVERVIEW]] and
[[THROW_IN_SIDE]].