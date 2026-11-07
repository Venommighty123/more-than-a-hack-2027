# Possession labeller: changelog

Every change says what was wrong, what changed, and what it did on the bench.

**The bench** (`test_scripts/bench/`) is three Metrica public sample matches (other games, so tuning on them is allowed by RULES.md). Ground truth for those games is derived from their event data. Each game comes in three copies:

- **clean**: the data as published.
- **degraded**: noise added until the starter baseline behaves on it the way the organisers say it behaves on BAR-ATH (same ranking of the four smoothing windows on both metrics).
- **hard**: noisier than that.

Scores are the mean over the three games, computed with the official `evaluation/possession_metrics.py`. The bench reads about 0.07 higher than the board on spellF1, so compare versions with each other on the bench and never compare a bench number with a board number.

Nothing in this file was tuned on BAR-ATH, on the public 5 minutes, or on any published statistic of that match.

## Reference: the starter baseline on the bench

| labeller | clean spellF1 / macroF1 | degraded | hard |
|---|---|---|---|
| baseline_smooth50 (the published floor) | 0.409 / 0.776 | 0.323 / 0.687 | 0.295 / 0.662 |
| baseline_smooth125 | 0.295 / 0.706 | 0.267 / 0.632 | 0.251 / 0.616 |

## v1.0: first version (2026-11-06)

The logic is described in the docstring of `labeller_v1.py`:

- control rule;
- confirmed changes, backdated to the first frame of control;
- dead-ball gaps;
- restart detection;
- live gaps hold the team;
- short possessions merged.

Result: clean 0.622 / 0.836. Degraded 0.304 / 0.741, which is worse than the baseline's 0.323.

## v1.1: speeds measured over a window

**What was wrong.** Ball speed was taken from one frame to the next. With 0.3 m of detection jitter, a ball that is standing still reads as moving at about 5 m/s. That fired the restart test at once, so stoppages broke into hundreds of slivers: 541 D stretches against 73 true ones in game 1, and twice as many possessions as the truth.

**What changed.**

- Ball speed is now the displacement over ±4 frames.
- Alive slivers shorter than 1 s inside a stoppage are joined into it (`D_JOIN`).
- `RESTART_V` was raised from 3 to 5 m/s.

**Effect.** Degraded went from 0.304 / 0.741 to 0.524 / 0.799. Clean is unchanged.

## v1.2: player speed measured over a window

**What was wrong.** The "players are slow, so the ball is dead" test used frame-to-frame player speed, which position jitter inflates. On the hard copy, dead-ball recall was 0.46.

**What changed.** Mean player speed is now taken from the displacement over ±12 frames, ignoring jumps faster than 12 m/s (these are identity switches).

**Effect.** Hard went from 0.426 / 0.678 to 0.458 / 0.742. D recall on the hard copy of game 1 went from 0.46 to 0.82.

## Sweeps after v1.2 (no change kept)

- `DEAD_SPEED` 1.5 to 2.2 and `DEAD_MIN` 1 to 3 s: flat within 0.01.
- `CTRL_R` 1.5 to 2.5, `MARGIN` 0 to 0.5, `CONFIRM` 3 to 6, `MIN_SPELL` 1 to 2: the defaults are within 0.005 of the best.
- `MIN_SPELL` 1 gives too many possessions (×1.1 to 1.3).

**Current v1 on the bench:**

| copy | spellF1 | macroF1 | vs baseline_smooth50 (spellF1) |
|---|---|---|---|
| clean | 0.617 | 0.839 | +0.21 |
| degraded | 0.523 | 0.798 | +0.20 |
| hard | 0.458 | 0.742 | +0.16 |

**Known weaknesses, to fix next:**

- About 40% of true hand-overs are missed (174 found against 304 true in game 1). These are mostly quick changes where the new team never holds the ball near one player.
- About 40% of predicted stoppages are not real.

---

# v2 (2026-11-06): new dead-ball detector (`labeller_v2.py`, run with `run_labeller_v2.py`)

## Why

On BAR-ATH, v1 labelled D almost exactly where the ball was missing (D 40.2% of graded frames, ball missing 41.1%). Two reasons:

1. **The speed threshold doesn't transfer.** v1's "players slow, so the ball is dead" test used an absolute 1.8 m/s. On BAR-ATH the mean player speed in play has a median of only 1.9 m/s, so the test fired in nearly every gap.
2. **Stoppages with the ball still tracked.** In the public window the ball is tracked on 66% of truly dead frames, and v1 could only find stoppages where the ball went missing.

## Changes to the bench (`bench/build_metrica_bench.py`, `bench/score_bench.py`)

1. **The bench's own truth missed stoppages.**
   - **What was wrong:** goal kicks are often recorded as PASS or BALL LOST with subtype GOAL KICK, not as SET PIECE. Stoppages longer than 120 s (injuries) were dropped.
   - **Effect:** many real stoppages were labelled as possession, so a correct D prediction was counted as wrong. On the clean copy, "false" stoppages fell from 1,700 s to 280 s once fixed.
   - **What changed:**
     - A restart is now any SET PIECE or any GOAL KICK subtype.
     - What stops play: ball out, a shot going out or in, a foul, or an offside.
     - Dead-time events in between (the keeper's RECOVERY, a CARD) are skipped.
     - The cap is now 300 s.
   - **Result:** D is now 30–43% of graded time, which fits real football (the ball is typically in play 55–65% of the time).
2. **Two new copies, `tracked` and `tracked_all`.** The ball stays visible through 70% or all of each stoppage, at the restarter's feet, carried by that one player. Before this, the simulated ball jumped between players, which gave it fake speed.
3. **Simulated Smart Tagging tags.** Each true restart is kept with probability 0.55, its time is jittered by 1 s, and 5% false tags are added.
4. **Calibration still holds.** The starter baseline on the `hard` copy is now almost exactly its published BAR-ATH pattern (macroF1 / spellF1):

   | | w0 | w25 | w50 | w125 |
   |---|---|---|---|---|
   | hard copy | .634 / .023 | .672 / .224 | .674 / .272 | .625 / .242 |
   | BAR-ATH (published) | .653 / .049 | .675 / .200 | .672 / .243 | .636 / .224 |

   On the `tracked` copies the baseline's macroF1 falls to 0.45–0.53, far below BAR-ATH's 0.67. So over the whole match BAR-ATH's dead ball is mostly *missing*. Stoppages with a tracked ball exist, as the public window shows, but they aren't the majority. The `degraded` and `hard` copies stay the main reference.

## Changes to the labeller

1. **Relative speed.** `rel` is the mean player speed, smoothed over 3 s, divided by the match's own median over graded frames. It needs no labels, so it adapts to any match.
   - On the bench, dead frames sit at about 0.6× and live frames at about 1.2×.
   - On its own it separates dead from live with AUC 0.84–0.89.
2. **Ball-gap stoppage** (gap of at least 1.5 s): dead if near a line, or if the median `rel` < 0.75. This replaces v1's 1.8 m/s.
3. **New: stoppage with the ball tracked.** Dead when:
   - the players are collectively slow (`rel` < 0.7) for at least 4 s, and
   - the ball is quiet (missing, slower than 2 m/s, or off the pitch) on at least half of those frames.

   The start is walked back while `rel` < 0.95 (up to 10 s), because players decelerate after the whistle: on the bench, `rel` falls from 1.0 to 0.6 through a stoppage.
4. **Restart (the end of D).** Either the ball is struck (> 5 m/s), or the players move as in open play again (`rel` > 1.0). Without the second test, a ball that reappears at a dribbler's feet kept D running until the next pass.
5. **Set-piece tags:** implemented, but **off by default**. On the bench they gave no gain (0.489 with, 0.494 without). `run_labeller_v2.py --tags` switches them on.

## Bench results (spellF1, mean of 3 games, new bench truth)

| copy | baseline_smooth50 | v1 | v2 |
|---|---|---|---|
| clean | 0.415 | 0.637 | 0.608 |
| degraded | 0.291 | 0.490 | 0.482 |
| hard (closest to BAR-ATH) | 0.272 | 0.425 | **0.439** |
| tracked | 0.277 | 0.453 | **0.486** |
| tracked_all | 0.281 | 0.392 | **0.454** |
| mean | | 0.479 | **0.494** |

- **Stoppages on the hardest copy:** v2's recall is 0.75, against v1's 0.24. Precision is about 0.88 on every copy.
- **Where v2 loses:** about 0.03 on the clean copy and 0.01 on `degraded`. These are copies where the ball almost always vanishes when dead, so "missing means dead" is nearly free there.

## Sweeps after v2 (no change kept)

- `MARGIN` 0 / 0.3, `CONFIRM` 3 / 6, `MIN_SPELL` 1 / 1.5 / 2: the defaults are best.
- `STOP_MIN` 4 / 8, `EXT_REL` 0.85–1.05, `LIVE_REL` 0.9–1.3: within 0.005.

## Biggest remaining weakness: possession changes

Only about half of the true hand-overs are found within 2 s, and the predicted possessions number 0.8× the true ones. Lowering the confirmation thresholds finds more changes but adds more false ones, so the score drops. The next version needs a smarter change detector (ball trajectory: a pass ending at an opponent, interceptions), not looser thresholds.

---

# v2.1 (2026-11-06): dead-ball fixes from the BAR-ATH v2 run

`run_labeller_v2.py` now writes `evaluation/results/labeller_v2_1.csv`, so the v2 file is kept. Each fix has a `FIX_*` switch in `labeller_v2.py`, so the bench can measure it on its own. A fix was kept only if it raised the bench mean.

| change | what was wrong (seen in the v2 run) | bench mean spellF1 (all 5 copies) | kept? |
|---|---|---|---|
| Bench tags re-simulated | Real tags START a median 6.3 s before the restart kick (measured on BAR-ATH from tracking only). The bench put them at the kick ±1 s. | (bench change) | yes |
| `FIX_CARRY`: a ball under 4 m/s, or within 3 m of a line, counts as quiet | A ball carried to a throw-in moves at ~3 m/s and failed the < 2 m/s test | +0.003 alone | yes |
| `FIX_OUT`: after the ball leaves the pitch, D lasts until a strike inside the pitch (max 60 s; if the ball is never seen again, until the players speed up after 8 s) | Players kept running after the ball went out, which ended D too early | +0.004 alone; −0.008 when removed | yes |
| `FIX_DROP`: a sudden collective slowdown in a ball gap counts as dead | A short foul stoppage mid-pitch was missed | +0.001 alone, 0.000 combined | no (no measurable gain) |
| `FIX_LINE`: "near a line" counts only with slowing players | A 9 s false stoppage in live play | −0.006 alone | no (it loses more real stoppages than it removes false ones) |
| Tags rebuilt (`TAG_STRIKE`): D from where the players slowed (up to 15 s before the tag) to the first strike within 8 s after it. Now ON by default (`--no-tags` turns them off). | Tags gave no gain with the wrong simulation | +0.003 (0.498 to 0.501) | yes |

**Bench spellF1 after v2.1:**

| copy | v1 | v2 | v2.1 |
|---|---|---|---|
| clean | 0.637 | 0.608 | 0.608 |
| degraded | 0.490 | 0.482 | 0.488 |
| hard | 0.425 | 0.439 | 0.466 |
| tracked | 0.453 | 0.486 | 0.499 |
| tracked_all | 0.392 | 0.454 | 0.446 |
| mean | 0.479 | 0.494 | **0.501** |

**Honest reading:**

- **Bench gain from v2:** +0.007. The fixes are real but small. The bench's noise band is about ±0.01 per copy.
- **The two bench copies closest to BAR-ATH** (`hard` and `degraded`) gained +0.027 and +0.006.
- **Biggest lever still left:** possession changes. Spell counts are still only 0.78–0.85× the truth on the noisy copies.

---

# v3 (2026-11-06): possession changes (`labeller_v3.py`, run with `run_labeller_v3.py`, writes `labeller_v3.csv`)

v2.1 was frozen first as `evaluation/results/SAFE_SUBMISSION_labeller_v2_1.csv`.

## Diagnosis

On the clean bench, v2.1 finds only 50% of the true hand-overs within 3 s.

- **The evidence is there:** for 82% of true changes, the new team controls the ball for at least 6 frames near the change.
- **The logic throws it away:** most misses are short real possessions that `MIN_SPELL` merges away, or changes that never reach `CONFIRM`.
- **Loosening those rules backfires:** it finds them, but it also makes false changes on noisy data.
- **Timing is good:** of the changes found, 64% are within 0.5 s.

## Tried: optimal segmentation (`RESOLVE="viterbi"`). Rejected.

Instead of switching greedily, this picks the best H/A sequence over each live stretch, with a fixed cost per change. Three forms of evidence were tried:

- **Per-frame control** (best: cost 16, mean 0.472).
  - **Fixed along the way:** a tie-break bug placed changes up to 2 s early. Across a ball flight every switch point scores the same, so ties now go to the latest frame.
  - **The "nearest player within 3 m" soft evidence** pulled changes about 1 s early, so it was removed.
- **One impulse per touch** (best: cost 1.5, minimum touch 4 frames, mean 0.489).
- **Result:** both stay below v2.1's greedy switch (0.501) on every noisy copy. False control from noise costs the global method more than the greedy one, which has short-spell merging. The code stays in `labeller_v3.py` as an option and is off by default.

## Kept: false-ball filter (`drop_blobs`)

- **What it does:** a ball track shorter than 1.5 s that is entered and left by a jump of more than 5 m in one frame (or by a gap) is a false detection somewhere else on the pitch, not the ball. A real ball cannot teleport, so this needs no labels. These tracks are removed before the control rule.
- **Bench mean:** 0.501 → **0.511**.

  | copy | v2.1 | v3 |
  |---|---|---|
  | degraded | 0.488 | 0.501 |
  | hard | 0.466 | 0.481 |
  | tracked | 0.499 | 0.515 |
  | tracked_all | 0.446 | 0.450 |
  | clean | 0.608 | 0.608 (unchanged) |

- **Threshold choice:** jump 3 / 5 / 8 m and length 1 / 1.5 / 2.5 s all land within 0.005, so the choice is not fragile.
- **Caveat:** the bench's false balls are simulated. `run_labeller_v3.py` prints how much of the real ball track the filter removes on BAR-ATH, so we can see whether our data has such blobs. A large share would mean it is removing real ball, and we would turn it off.
- **Re-swept after the filter:** `CONFIRM` 4 / 6 / 9 and `MIN_SPELL` 1 / 1.5 / 2 are all within 0.006, and the defaults stay.

## Where this leaves possession changes

Thresholds and segmentation on per-player control have plateaued at about 0.51 on the bench. The next real step needs a different source of evidence: the ball's path. A pass that ends with an opponent, or a ball that changes direction at an opponent's feet, marks an interception even when nobody controls the ball for long.

---

# v4 (Nov 7, session 2): rule C off, control radius 2.0 m

Files: `test_scripts/labeller_v4.py` (v2.1 + three switches), `test_scripts/run_labeller_v4.py`.
Scored on the rebuilt bench (it reproduces session 1 exactly: baseline50 `real` 0.287 / 0.657,
v1 0.516 / 0.473, v2.1 0.483 / 0.438). Ranked by the mean of `real` and `real_occ` (R+O); the old
copies are a regression check only. Mean spellF1 over Metrica games 1-3.

**What was wrong.** On the realistic copies, rule C (ball present + players slow = dead) marks slow
live build-up as dead and costs about 0.02. It was built for stoppages with a tracked ball, which
the full match mostly does not have (ball tracked only 31% of the 10 s before set-piece tags).
And the real ball sits ~1 m off the player on it (+0.46 m ahead), so the 1.5 m circle misses
about 40% of on-ball frames.

## Kept: `RULE_STOP_MODE="off"` (rule C off)

| setting | real | real_occ | hard | degraded | R+O |
|---|---|---|---|---|---|
| v2.1, rule C full | 0.483 | 0.438 | 0.466 | 0.488 | 0.461 |
| rule C weak (tag / out / line evidence) | 0.488 | 0.443 | 0.462 | 0.488 | 0.466 |
| rule C weak, line test stricter (0.6 / 0.9) | 0.490-0.492 | 0.447-0.448 | 0.458-0.460 | | 0.469 |
| rule C weak, tag / out only | 0.492 | 0.449 | 0.459 | | 0.471 |
| rule C weak, out only | 0.500 | 0.459 | 0.449 | | 0.480 |
| rule C full, `STOP_REL` 0.55 | 0.497 | 0.453 | 0.467 | | 0.475 |
| rule C full, `STOP_MIN` 8 s | 0.487 | 0.445 | 0.467 | | 0.466 |
| **rule C off** | **0.504** | **0.461** | 0.450 | **0.499** | **0.483** |

- Off beats full on all 6 realistic game copies (real +0.038/+0.022/+0.039 with R 2.0; real_occ
  +0.033/+0.008/+0.029), so the +0.02 is consistent, not noise.
- "Weak" does not rescue rule C. Each piece of evidence also occurs in live play: build-up runs
  along the touchline, and tags are already handled by rule D, so rule C around a tag only drags
  D backwards into live play. Every weaker form lands between full and off.
- It loses on `hard` (-0.016), the old copy with the ball tracked through most dead time. That
  copy does not look like BAR-ATH, so it does not decide.

## Kept: `CTRL_R` 1.5 -> 2.0 (circle)

| rule C off + | real | real_occ | hard | degraded | R+O | spells (real, gt ~380) |
|---|---|---|---|---|---|---|
| circle 1.5 | 0.504 | 0.461 | 0.450 | 0.499 | 0.483 | 256 |
| circle 1.75 | 0.512 | 0.458 | 0.448 | 0.492 | 0.485 | 265 |
| **circle 2.0** | **0.516** | **0.462** | 0.443 | 0.480 | **0.489** | 270 |
| circle 2.25 | 0.508 | 0.461 | 0.438 | 0.476 | 0.485 | 279 |
| circle 2.0, `MARGIN` 0.15 / 0.5 | 0.522 / 0.521 | 0.463 / 0.460 | 0.441 | | 0.493 / 0.490 | |

- +0.006 R+O: inside the noise band. Kept because it matches the measured ~1 m offset and does
  not lose on either realistic copy. `MARGIN` 0.15-0.5 is flat, so it stays 0.3.

## Rejected: directional control (`CTRL_MODE="shift"` / `"ellipse"`)

| rule C off + | real | real_occ | hard | R+O |
|---|---|---|---|---|
| shift 0.5 m, R 1.5 / 2.0 | 0.492 / 0.489 | 0.450 / 0.461 | 0.414 | 0.471 / 0.475 |
| shift 0.8 m, R 1.5 / 2.0 | 0.468 / 0.475 | 0.439 / 0.441 | 0.395 | 0.453 / 0.458 |
| ellipse fwd 2.5, back 1.5, side 1.5 | 0.503 | 0.461 | 0.436 | 0.482 |
| ellipse 2.0 / 1.2 / 1.5 | 0.504 | 0.448 | 0.443 | 0.476 |
| ellipse 3.0 / 1.5 / 1.5 | 0.488 | 0.444 | 0.429 | 0.466 |

- Worse than the plain 2.0 m circle everywhere, even though the bench does put the ball 0.5 m
  ahead of the carrier. Why: the defender chasing the carrier runs the same way, so his zone is
  also pushed toward the ball. The margin test then fails more often or flips, and possessions
  fragment (spells 266-292 vs 270). The real limit is the ~1 m random error, not the lead.

## Rejected: adaptive radius (`CTRL_R_AUTO` = quantile of open-play nearest distance)

| q | radius g1/g2/g3 (real) | real | real_occ | R+O |
|---|---|---|---|---|
| 0.6 | 1.33 / 1.32 / 1.63 | 0.497 | 0.459 | 0.478 |
| 0.7 | 1.60 / 1.60 / 2.05 | 0.507 | 0.454 | 0.480 |
| 0.8 | 2.00 / 2.02 / 2.51 | 0.517 | 0.424 | 0.471 |

- Label-free, but fragile: when the player on the ball is lost (real_occ) the nearest distance
  grows, the quantile inflates the radius, and real_occ falls to 0.42. A fixed 2.0 m is better.

## v1 compared

v1 (its own simple dead-ball logic) scores R+O 0.495, the same as v4 (0.489) within noise;
per game it is mixed. v1 with R 2.0: real 0.528, real_occ 0.472. So once rule C is off, v2.1's
remaining dead-ball rules add nothing measurable on the realistic copies, and nothing loses either.
All settings sit at R+O 0.47-0.49: threshold tuning of the control rule has plateaued. The next
gain has to come from new evidence about hand-overs (session1.md section 7.3).

**v4 defaults:** `RULE_STOP_MODE="off"`, `CTRL_MODE="circle"`, `CTRL_R=2.0`, `CTRL_R_AUTO=None`;
everything else = v2.1. `--stop-mode full --ctrl-r 1.5` reproduces v2.1 exactly (checked on the bench).

## Checked: v4's long possessions (> 40 s) on BAR-ATH, 6 -> 13. Not a problem. Rule E rejected.

- **Question:** on BAR-ATH, v4 doubled possessions over 40 s (6 -> 13) against v2.1. Are these
  stoppages that rule C used to catch?
- **Bench answer:** the Metrica ground truth itself has 11-14 possessions over 40 s per match, and
  v4 produces 12-19 on the realistic copies. Only about 8% of the time inside them is truly dead
  (real: 58 / 755 s, 69 / 1017 s, 44 / 719 s). 13 on BAR-ATH is therefore normal for a full match.
- The few long possessions that do hide a stoppage have one signature: players slow (rel < 0.7)
  for 20-40 s in a row. Live long possessions never have a slow run above 7 s.
- **Tried rule E (`LONG_SLOW`):** dead when rel < 0.7 for at least N s.

  | v4 + | real | real_occ | hard | degraded | R+O |
  |---|---|---|---|---|---|
  | nothing (v4) | **0.516** | **0.462** | 0.443 | 0.480 | **0.489** |
  | E 10 s | 0.501 | 0.449 | 0.459 | 0.472 | 0.475 |
  | E 15 s | 0.505 | 0.454 | 0.455 | 0.474 | 0.480 |
  | E 20 s | 0.509 | 0.455 | 0.454 | 0.477 | 0.482 |
  | E 15 s, extended to the restart | 0.505 | 0.453 | 0.455 | 0.474 | 0.479 |
  | E 15 s, rel < 0.6 | 0.512 | 0.458 | 0.451 | 0.479 | 0.485 |

- **Rejected:** it loses on both realistic copies at every setting. Slow 15-20 s runs also happen
  in live play (keeper on the ball, slow build-up), and a D in the middle of a real possession
  splits it into two, which costs spellF1 more than fixing the few hidden stoppages gains.
  `LONG_SLOW=None` stays the default; with it, v4's output is unchanged.
