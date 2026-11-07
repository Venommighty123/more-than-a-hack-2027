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

## Restructure check (2026-11-07)

`possession/labellers/v1.py` (rebuilt on the shared modules) gives output identical to the original
`labeller_v1.label` on BAR-ATH (0 frames differ at 105 x 68) and on the bench. Bench rebuilt with
`scripts/bench_build.py` in the project environment:

| labeller | clean | degraded | hard |
|---|---|---|---|
| v1 | 0.6174 / 0.8391 | 0.5226 / 0.7977 | 0.4577 / 0.7422 |

Note: building the bench with other numpy / pandas versions changes the degraded copies (the
noise draws differ) and the scores with them (a Linux build gave v1 degraded 0.492). Build it once,
in the pinned environment, and share the files.

## v2.0: hand-overs moved into ball gaps (2026-11-07) -- not kept

`detectors/gap_fill.py` moves a switch that the resolver writes at the end of a live gap to a frame
inside it: `fixed` (a fraction of the gap) or `evidence` (team-shape signal: which team pushes
towards the opponent goal).

| GAP_SPLIT | degraded spellF1 / macroF1 |
|---|---|
| end (= v1) | 0.5226 / 0.7977 |
| fixed 0.25 | 0.5186 / 0.7943 |
| fixed 0.5 | 0.5198 / 0.7958 |
| fixed 0.75 | 0.5206 / 0.7964 |
| evidence | 0.5208 / 0.7973 |
| evidence, min_evidence 0 | 0.5210 / 0.7973 |

Hard: end 0.4577, evidence 0.4520, fixed 0.75 0.4513. Clean: end 0.6174, evidence 0.6167.
Only 54 switches move over the three games, and for half of them the true hand-over is after the
gap. Default stays `end`, so v2 == v1. The Smart Tagging prior and throw-in sides (USE_TAGS) cannot
be scored on the bench and stay off.

## v3: HMM decoding, rules and bench-trained classifier (2026-11-07)

`model/hmm.py` (H/A/D Viterbi with switch / dead penalties) replaces `model/resolve.py` in
`labellers/v3.py`.

**Rules only** (`v3`, emissions from v1's control + dead rule), degraded: best 0.5056 / 0.7973
(SWITCH 12, TO/FROM_DEAD 10) vs v1 0.5226 -- tried SWITCH 6 / 8 / 12 / 14 / 20, dead penalties
3 / 6 / 10 / 12. Not better than the state machine.

**Rules + classifier** (`v3c`): multinomial logistic regression (`model/classifier.py`, numpy) on
the bench columns of `features/build.py` plus 1 s / 3 s moving averages, trained on bench degraded
games. Leave-one-game-out (each game scored by a model that never saw it), degraded:

| setting | spellF1 / macroF1 | per game spellF1 |
|---|---|---|
| v1 (reference) | 0.5226 / 0.7977 | |
| classifier only (RULE_WEIGHT 0) | 0.4940 / 0.8063 | first feature set |
| rules + classifier, PROBA_WEIGHT 1.0 | 0.5374 / 0.8074 | 0.520 0.563 0.528 |
| rules + classifier, PROBA_WEIGHT 0.5 (**default**) | **0.5447 / 0.8068** | 0.526 0.573 0.536 |
| rules + classifier, PROBA_WEIGHT 0.25 | 0.5450 / 0.8063 | 0.526 0.586 0.522 |

Fix found on the way: `n_tracked_*` is always 11 on the bench, so after standardisation it
exploded on BAR-ATH (|z| > 200) and pushed v3c to 11 % D. Dropped from the inputs, inputs clipped
to |z| <= 5, `gap_len_s` capped at 30 s. After the fix v3c on BAR-ATH gives 44 % D, 346 possessions
(v1: 41 %, 321). Domain shift bench -> BAR-ATH remains the main risk of v3c.
