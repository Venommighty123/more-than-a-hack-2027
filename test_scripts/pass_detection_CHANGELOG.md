# pass_detection.py: changelog

Real stats used as the target: home 568 passes (87.5% accurate), away 293 (79.2%).

| # | Date (UTC) | What was wrong | What changed | Result on the match |
|---|---|---|---|---|
| 1 | Nov 5 | (first version) | Kicks from `ball_flight.detect_candidates`; passer = nearest player within 2 m; receiver = first player within 1.5 m of a slow or touched ball; completion = receiver's team | H 362 (82%), A 252 (59%) |
| 2 | Nov 6 05:00 | Pylance could not resolve `ball_flight` / `load_homography` | Imports changed to `from starter.ball_flight ...` | warnings gone, behaviour unchanged |
| 3 | Nov 6 05:07 | One-touch passes were never detected: `ball_flight` only starts a flight when the ball speeds up from slow, and in a one-touch pass it stays fast | A reception followed within 8 frames by a fast ball starts a new kick by the receiver | (run 2 below) |
| 4 | Nov 6 05:09 | 53 "passes" came from tracking during halftime / before kickoff | Frames outside the two halves are blanked after loading | H 426 (69%), A 326 (53%): interceptions doubled |
| 5 | Nov 6 05:14 | Balls rolling past a defender counted as received: the touch test used the ball's direction (pure noise on a slow ball) and any ball under 8 m/s near a player counted as controlled | Touch = change of the velocity VECTOR over ±3 frames ≥ 3 m/s; "controlled" speed lowered 8 → 4 m/s | H 514 (62%), A 368 (47%) |
| 6 | Nov 6 05:19 | Still too many interceptions; each one also created a fake pass for the other team | (a) An opponent's brush/deflection where the ball flies on is followed to where it ends up (`deflected_by_slot`). (b) Touch threshold auto-set from the match's own tracking noise | H 347 (82%), A 233 (57%): noise is huge (p90 4.7 m/s), auto threshold 6.4 killed one-touch passes |
| 7 | Nov 6 05:23 | Tuning one setting per run was too slow | `--sweep`: 96 settings in one run, ranked against the real stats | best: kick 5, radius 1.5, touch 3.0: H 499 (73.5%), A 319 (65.3%) |
| 8 | Nov 6 05:30 | Accuracy ~14 points low on every setting: the "first player the ball reaches" is often the marker next to the receiver | Receiving team = team nearest the ball most often over the 8 frames after reception (`--confirm`); best sweep settings made the defaults | H 459 (78.8%), A 311 (65.4%) |
| 9 | Nov 6 05:40 | Tested the same vote for the passer (tackler nearest at the kick) | Added `--confirm-passer`; it made every setting worse, so it is OFF by default | no change to defaults |
| 10 | Nov 6 05:47 | Blind tuning plateaued | `pass_spot_check.py`: random sample of passes with tactical-video seek times and a verdict column | (needs Ashwyn's verdicts) |
| 11 | Nov 6 13:16 | Spot check: 5 of 20 "interceptions" (and 1 "complete") moved the ball < 0.3 m: the kick's own velocity change was read as a defender's touch | A "reception" less than 3 m from the kick point is a `duel` (opponent) or `touch` (same team), not a pass (`MIN_PASS_M`) | H 423 (82.3%), A 268 (65.0%). Home incompletes now match real (68 vs 71); the gap is MISSED completed passes (316 vs 497 home, 158 vs 232 away) |
| 12 | Nov 6 13:30 | Diagnosis: incompletes are about right for home, so the remaining error is passes we never count. Candidates: the 260 home / 140 away rows classed as "touch" | `pass_spot_check.py --touches`: samples rows the script did NOT count as passes (touch / duel), to check on video whether they were real passes | no change to the detector |
