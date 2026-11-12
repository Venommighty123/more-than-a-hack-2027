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

## Bench v2 (Ashwyn, Nov 7; ported Nov 9)

`possession/io/bench.py` now builds Ashwyn's updated bench (`backup/ashwyn_build_metrica_bench.py`):

- **ground truth**: D runs from what stopped play (ball out, shot out / goal, foul, offside) to the
  restart (any SET PIECE or a goal kick recorded as PASS / BALL LOST); dead-time events (keeper's
  RECOVERY, CARD) are skipped. The first bench missed goal kicks and stopped D at the wrong event.
- **simulated set-piece tags** in every copy, timed as measured on BAR-ATH (55 % of restarts tagged,
  a median 6.3 s before the kick, 5 % false tags). `bench_score.py` passes them to labellers that
  take `tags`.
- new copies: `tracked`, `tracked_all` (ball visible through stoppages) and the **realistic**
  `real` / `real_occ`, whose noise was measured on BAR-ATH from tracking only
  (`backup/ashwyn_noise_profile.py`): few long live gaps, fragmented player tracks, the ball drawn
  ~1 m off the player on it. **R+O** = mean of real and real_occ: the number decisions are taken on.
- Checked: g1 / g2 files identical to Ashwyn's builder; g3 within 1e-5 m (repo EPTS reader).
- The old bench is kept in `outputs/bench_pre_nov7/`; numbers above this section are on it and are
  not comparable with the ones below. v1 on the new bench: real 0.516, real_occ 0.473, hard 0.428,
  degraded 0.502 (R+O 0.494).

## v4: Ashwyn's v2.1 dead-ball rules, rule C off, control radius 2.0 m (Nov 7-8; ported Nov 9)

Ashwyn's history v2 / v2.1 / v3 / v4 is in `backup/ashwyn_labeller_v1_CHANGELOG.md`. Ported to
`labellers/v4.py` on the shared modules, with his dead-ball rules in `detectors/stoppage.py`.
Checked identical to the original: 0 differing frames on every bench copy (also with
`RULE_STOP_MODE` full / weak and `LONG_SLOW` 15), and the BAR-ATH file from
`scripts/run.py --labeller v4 --bench-pitch` equals `run_labeller_v4.py`'s on all 199,319 rows.

| bench v2 | real | real_occ | hard | degraded | R+O |
|---|---|---|---|---|---|
| v4 | 0.5160 | 0.4617 | 0.4430 | 0.4796 | **0.4889** |

## v5: reconstructed ball, flights, zone of influence, 15-frame confirm (Nov 9)

Requested changes, each behind a switch (`labellers/v5.py`; with all switches off v5 == v4, 0
differing frames). Reproduce: `python experiments/v5_ablation.py`. Bench v2, spellF1, mean of
games 1-3, simulated tags on; "spells" = predicted possessions on the real copies (truth: 1,140).

| configuration | real | real_occ | hard | degraded | R+O | spells |
|---|---|---|---|---|---|---|
| v4 | 0.5160 | 0.4617 | 0.4430 | 0.4796 | 0.4889 | 810 |
| + ball: Hermite fill <= 25 frames, 35 m/s gate | 0.5180 | 0.4630 | 0.4419 | 0.4774 | **0.4905** | 808 |
| + ball: same, no gate | 0.5172 | 0.4630 | 0.4418 | 0.4797 | 0.4901 | 808 |
| + flights, mask ALL control in flight | 0.4828 | 0.4437 | 0.4455 | 0.4776 | 0.4633 | 762 |
| + flights, mask the OPPONENT only | 0.5145 | 0.4651 | 0.4419 | **0.4936** | 0.4898 | 801 |
| + ZOI veto, zone radius 2.0 m (earlier run) | 0.5164 | 0.4448 | 0.4202 | 0.4681 | 0.4806 | 786 |
| + ZOI contest-only veto, 2.0 m (earlier run) | 0.5159 | 0.4448 | 0.4206 | 0.4681 | 0.4803 | 787 |
| + ZOI veto, 1.5 m | 0.5258 | 0.4546 | 0.4370 | 0.4746 | **0.4902** | 802 |
| + ZOI contest-only veto, 1.5 m | 0.5232 | 0.4543 | 0.4370 | 0.4748 | 0.4888 | 800 |
| + ZOI veto, 1.2 m / 1.0 m | 0.5155 / 0.5149 | 0.4625 / 0.4644 | 0.4458 / 0.4456 | 0.4781 / 0.4774 | 0.4890 / 0.4897 | 803 / 814 |
| + ZOI confirm (both rules must hold), 1.5 m | 0.4336 | 0.3608 | 0.3902 | 0.4298 | 0.3972 | 625 |
| + CONFIRM 8 | 0.5175 | 0.4483 | 0.4447 | 0.4862 | 0.4829 | 782 |
| + CONFIRM 10 | 0.5129 | 0.4409 | 0.4418 | 0.4844 | 0.4769 | 769 |
| + CONFIRM 15 | 0.4861 | 0.4145 | 0.4242 | 0.4732 | 0.4503 | 706 |
| + CONFIRM 15, HOLE 6 | 0.4880 | 0.4139 | 0.4273 | 0.4776 | 0.4510 | 706 |
| **v5 (defaults: all four, CONFIRM 15)** | 0.4824 | 0.4075 | 0.4199 | 0.4619 | 0.4450 | 684 |
| v5 with CONFIRM 6 | **0.5253** | 0.4564 | 0.4368 | 0.4827 | **0.4909** | 795 |
| v5 with CONFIRM 8 | 0.5139 | 0.4431 | 0.4413 | 0.4868 | 0.4785 | 752 |
| v5 without flights | 0.4843 | 0.4072 | 0.4199 | 0.4620 | 0.4457 | 691 |
| v5 without ZOI | 0.4847 | 0.4162 | 0.4249 | 0.4756 | 0.4505 | 698 |
| v5 with ZOI confirm | 0.3893 | 0.3375 | 0.3709 | 0.4033 | 0.3634 | 549 |

What it says:

- **Ball fill** (+0.002 R+O): small, because on BAR-ATH and the realistic copies almost every gap is
  long. On BAR-ATH 122 of the 129 gaps of <= 25 frames are re-detections (the ball reappears where
  it could not have travelled: median 58 m/s, up to 650 m/s); the 35 m/s gate leaves them NaN, so
  only 7 gaps (88 frames) are filled. Ungated filling drew fake 50-1000 m/s kicks.
- **Flights**: removing ALL control inside a flight loses real evidence (on clean data the distance
  rule inside flights agrees with the truth 97 % of the time: the kicker and his receiver).
  Removing only the non-kicking team's control is neutral on R+O and +0.014 on degraded: kept.
- **ZOI**: with a 2.0 m zone, two players' zones overlap and every close marking becomes a "contest";
  at 1.5 m (inside the 2.0 m distance circle) the veto helps real (+0.010) and costs real_occ /
  hard (-0.007): net +0.001. "confirm" (both rules must hold) halves the control frames: rejected.
- **CONFIRM 15**: monotonic loss 6 > 8 > 10 > 15. The labeller already predicts too few
  possessions (810 vs 1,140 true); a longer confirmation hides more short real possessions
  (684), and the backdating cannot recover a change that is never confirmed.
- v5 as requested (CONFIRM 15) scores 0.445 R+O, 0.044 below v4. With CONFIRM 6 the other three
  changes together reach 0.4909 (+0.002 over v4, within noise; real +0.009, real_occ / hard -0.005).

BAR-ATH (`scripts/run.py`): v5 275 possessions (median 8.7 s), 39.9 % D; v5 CONFIRM 6: 306
possessions; v4: 316. The zone veto removes 2,160 of 47,276 distance-control frames, the flight mask
394. The public-5-minutes sanity score is not used to choose (RULES.md).

## Ashwyn's v5 and v6 (his branch, Nov 8-10; brought in Nov 10)

Naming: Ashwyn's "v5" (a hand-over detector, not adopted) is **not** our `v5` above. His v6 builds
on v4 and is registered here as `v6`. Raw files: `backup/ashwyn_labeller_v5.py`,
`backup/ashwyn_labeller_v6.py`, `backup/ashwyn_run_labeller_v6.py`, bench helpers
`backup/ashwyn_build_trk_gt.py` (tracking-based truth), `backup/ashwyn_fetch_dfl_bench.py` (7 DFL
matches), `backup/ashwyn_hutil.py`. His full write-up: `docs/progress/ashwyn_progress_brief.md` and
the v5 / v6 sections at the end of `backup/ashwyn_labeller_v1_CHANGELOG.md`.

- **His v5** (reception vote, flight, gap vote, density switching): up to +0.017 on the event truth
  but -0.010 on his new tracking-based truth. Not adopted, not ported.
- **v6** = v4 + `SMOOTH_K` 3 (positions averaged over +/- 3 frames) + `SW_MARGIN` 0.6 (0.6 m clearer
  needed to *take* the ball) + `MIN_D` 2 s (shorter stoppages become live play). Rejected switches
  (weak-blip merge, velocity following, gap split, gap-start delay) are not ported.

Port (`possession/labellers/v6.py`; hysteresis added to `model/resolve.py` as the optional
`ctrl_sw`, default None so v1-v5 are unchanged). **Checked Nov 10: identical to Ashwyn's original on
every frame of g1-g3 x real / real_occ / hard / degraded**, and `params=v6.AS_V4` is identical to
our v4. Tests: `tests/test_v6.py`.

Bench v2, event truth, spellF1 (our run of both, Nov 10):

| | g1 real | g2 real | g3 real | real | g1 occ | g2 occ | g3 occ | real_occ | R+O |
|---|---|---|---|---|---|---|---|---|---|
| v4 | 0.5453 | 0.5149 | 0.4879 | 0.516 | 0.4885 | 0.4241 | 0.4724 | 0.462 | 0.489 |
| **v6** | 0.5532 | 0.5412 | 0.5311 | **0.542** | 0.4837 | 0.4420 | 0.5093 | **0.478** | **0.510** |

Ashwyn's other truths: tracking truth 0.596 / 0.525 (v4 0.563 / 0.518); DFL 7 matches 0.464 vs v4
0.468 (tie). The +0.02 is inside the 0.04 noise band; v6 loses only on g1 real_occ (-0.005).
On BAR-ATH (his run): 305 possessions, 95 stoppages, 39.9 % D, 99.0 % same labels as v4.
**Reproduced on BAR-ATH, Nov 10:** `python scripts/run.py --labeller v6 --bench-pitch` writes a file
with md5 `6d084c06`, the same as Ashwyn's `SAFE_SUBMISSION_labeller_v6.csv` (305 possessions, 95
stoppages, 39.9 % D). `--bench-pitch` (105 x 68 m) is required: Ashwyn's loader scales to 105 x 68,
the package default is games.py's 105.3 x 68, which gives a slightly different file.
