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
