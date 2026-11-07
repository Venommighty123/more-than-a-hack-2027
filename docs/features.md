# Feature dictionary

One row per frame, written by `possession/features/build.py` (`python scripts/build_features.py`).
No labels. Metres, seconds, m/s. NaN = not computable (ball unseen, team unseen). The table is
derived from match data: it lives in `outputs/features/`, never in git.

**Bench columns** (`BENCH_COLUMNS`) exist on the bench and on BAR-ATH; only these are model inputs.
BAR-ATH-only columns: `tag_score`, `cam_*`. Not model inputs: `ball_x`, `ball_y`, `graded`, `n_tracked_*`.

| column | source | meaning |
|---|---|---|
| `ball_observed` | preprocess/ball | tracker saw the ball (before cleaning) |
| `ball_present` | preprocess/ball | usable position (kept, or bridged gap <= 6 frames) |
| `ball_x`, `ball_y` | preprocess/ball | cleaned position |
| `ball_speed` | preprocess/ball | displacement over +/- 4 frames |
| `ball_touch` | preprocess/ball | change of velocity vector across +/- 3 frames |
| `ball_outside` | preprocess/ball | beyond a line by > 0.5 m |
| `ball_line_m` | preprocess/ball | distance to the nearest line (negative = outside) |
| `ctrl_dist` | detectors/control | control, distance rule (0 home, 1 away, -1 nobody) |
| `ctrl_zone` | detectors/control | control, zone-of-influence rule (2 = contest) |
| `dead_rule` | detectors/dead_ball | labeller v1's dead-ball mask |
| `in_gap`, `gap_len_s`, `gap_pos`, `gap_class` | gaps/classify | inside a ball gap >= 1 s; its length (model input capped at 30 s); position 0-1 through it; class 1 kick-off, 2 tagged set piece, 3 untagged dead, 4 live |
| `mean_speed`, `speed_home`, `speed_away` | preprocess/players | mean player speed, smoothed 1 s |
| `d1_*`, `d2_*` | features/relational | nearest / second-nearest player of the team to the ball |
| `closing_*` | features/relational | nearest player's speed towards the ball (+ = closing in) |
| `relspeed_*` | features/relational | ball speed relative to that player |
| `n3_*`, `n5_*`, `n10_*` | features/relational | players of the team within 3 / 5 / 10 m of the ball |
| `gap_d1` | features/relational | `d1_away - d1_home` (+ = home closer) |
| `ttb_home`, `ttb_away`, `ttb_diff` | features/time_to_ball | fastest arrival time per team (0.7 s reaction, 7 m/s); diff + = home first |
| `pc_home` | features/pitch_control | P(home wins a ball at the ball's position), sigma 0.45 s |
| `cx_att_*`, `cy_*` | features/shape | outfield centroid along the attacking direction / across |
| `spread_*`, `length_*`, `width_*` | features/shape | RMS spread, 10-90 % extents |
| `def_line_*` | features/shape | deepest outfield player along the attacking direction |
| `advance_*`, `advance_diff` | features/shape | mean velocity towards the opponent goal (+ = pushing) |
| `spread_diff`, `cx_att_diff` | features/shape | home - away |
| `n_tracked_*` | features/shape | players tracked (always 11 on the bench: not a model input) |
| `tag_score` | detectors/tag_prior | Smart Tagging evidence, + home / - away (BAR-ATH only) |
| `cam_x`, `cam_y`, `cam_speed`, `cam_reproj` | features/camera | image-centre aim point, its pan speed, reprojection error (BAR-ATH only) |
| `graded` | — | row inside a graded block |

Sanity checks on BAR-ATH (no labels, against our own control signal): when home controls the ball
`pc_home` averages 0.78 (away: 0.22), `ttb_diff` +0.70 s (away: -0.79), `advance_diff` +0.77 m/s
(away: -0.59). The camera aim point is a median 23 m from the ball (useless as a position) but pans
at 0.1 m/s during dead-ball stretches vs 0.5 m/s in play (a possible stoppage cue).
