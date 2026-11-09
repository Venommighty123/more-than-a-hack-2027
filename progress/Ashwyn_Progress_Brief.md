# Ashwyn's progress brief: Level 1 possession labeller (Nov 1 to Nov 9)

Written 2026-11-09 for the team meeting. It covers **everything** from session 1 (Nov 1-7) and
session 2 (Nov 7-9). The Nov 7 executive summary was never presented, so this brief stands alone:
you do not need to read anything else first.

Status in one paragraph: we have a **validated safe submission** (`labeller_v6`) that scores
roughly twice the published baseline on our test benches. It is ready to upload to the Google
Form. Our own team target for that upload is **Nov 10** (the organisers' only deadline is Nov 17). We reached it in six versions, each one tested
only on other matches, as the rules require. Tuning of the current approach is close to its
ceiling; one last experiment is paused (section 10), and after it the plan is to move to the
freeze work: rule cleanup, a one-command reproducible pipeline, and the technical document.

---

## Contents

1. The goal and how we are scored
2. The rules we must not break
3. The data, and what makes this task hard
4. How we measure progress without cheating: the test benches
5. Code map: every file and what it is for
6. Core logic of the labeller, step by step
7. Version history v1 to v6, with numbers
8. Challenges we hit and how we solved them
9. Where we stand against the goal
10. What Ashwyn is doing right now
11. Plan of action and strategy to Nov 17
12. Key points to discuss in the meeting
13. Glossary
14. Appendix: commands and file hashes

---

## 1. The goal and how we are scored

**The goal is to win Level 1**, not to beat the baseline. The ranking is decided only by the
score. There is no jury and no presentation mark.

- **Match:** FC Barcelona vs Athletic Club (`BAR-ATH`), one full match.
- **Output:** a CSV `frame,label` with exactly **199,319 rows**, one per frame of the tactical
  video. Labels:

  | Label | Meaning |
  |---|---|
  | `H` | home team (Barcelona) has the ball |
  | `A` | away team (Athletic) has the ball |
  | `D` | dead ball: play stopped (out of play, foul, offside, goal, waiting for a restart) |
  | `X` | not graded / abstain |

- **Graded frames:** half 1 = frames 11185 to 90651, half 2 = 110808 to 190197. Frames 110808 to
  113741 (before the second-half kickoff) are all `D` (FAQ.md).
- **Ranking metric: `spellF1`.** A "spell" is one unbroken run of the same label, that is, one
  possession or one stoppage. Our spells are matched one to one against the true spells by
  overlap in time. This punishes both mistakes we can make:
  - **too many changes** (one true possession cut into pieces: only one piece can match, the
    rest count as false spells);
  - **too few changes** (two true possessions merged into one: one of them goes unmatched).
  So spellF1 rewards getting the **number and timing of possession changes** right, not just
  the share of frames that are right.
- **Tie-break: `macroF1`.** The mean frame-by-frame F1 of H, A and D. An `X` on a graded frame
  counts as wrong, so **we never write `X` on a graded frame.**
- **The published floor** is `baseline_smooth50` at **spellF1 0.2433** (nearest player to the
  ball, majority vote over a 50-frame window).
- **Noise:** EVALUATION.md says differences below about **0.04 spellF1** are noise on a full
  match. So a small gain on our bench is only believed if it is consistent across every copy
  and every truth we test on.
- **Deliverables (Google Form, deadline Nov 17):** the CSV; a **public repository that
  regenerates exactly that CSV** from the data bundle with one command; a short technical
  document.
- **Organisers' deadline:** only **Nov 17** (OVERVIEW.md, SUBMISSION.md). One form response per
  team, editable until the deadline; the last response is the one scored. **There is no live
  leaderboard and no feedback before the deadline.**
- **Name the files after the team:** `<team_name>.csv` and `<team_name>.pdf` (SUBMISSION.md: the
  board names each row after the file).
- **Our own team milestones** (from our Nov 3 game plan, not the organisers): safe submission
  uploaded **by Nov 10**, code freeze **Nov 15**, submit **Nov 16** (one day of buffer).

---

## 2. The rules we must not break

Breaking any of these gets us disqualified, or makes our numbers untrustworthy.

1. **No hand labels on BAR-ATH, in any amount, for any purpose** (RULES.md rule 1). That includes
   validation and tuning. Watching the video to *understand* an error is fine; turning what you
   saw into a label, a correction, a constant or a rule is not.
2. **No human-produced data about BAR-ATH** (RULES.md rule 2): no Opta, StatsBomb, Wyscout or
   FotMob stats, events or possession numbers. Early pass-detection tests compared against
   published pass stats. Those items are listed in `TESTING_ONLY_REMOVE_BEFORE_SUBMISSION.md`
   (repo root) and **must be removed before we submit**. Nothing in the labeller depends on them.
3. **The public 5-minute ground truth is a format check only.** EVALUATION.md shows that tuning
   on it picks the wrong option. Our run scripts print a `SANITY` line from it, and **we never
   choose a version because of that number.**
4. **Tune only on other matches.** We use Metrica's 3 public sample games and 7 public DFL
   Bundesliga matches (section 4).
5. **Match data never goes to a cloud AI tool.** Claude writes the scripts; they run on
   Ashwyn's PC; only summary numbers come back. The two videos (tactical ~4 GB, broadcast
   ~13 GB) stay on the PC.
6. Smart Tagging and LLMs for writing code are explicitly allowed (FAQ.md).

**Ask for the team:** if your branch uses anything from rules 1-3 (hand-checked frames, match
stats, a setting picked by the public 5 minutes), flag it in the meeting so we can remove it
before it reaches the submitted pipeline.

---

## 3. The data, and what makes this task hard

What the bundle gives us:

- **ATD tracking** at 25 fps: ball x/y plus players in **anonymous slots** with a team flag. Not
  named players, and 46 slots for 22 players (tracks break and restart in new slots).
- **Smart Tagging:** 15 phase codes plus a few set-piece tags. **There is no event data** (no
  passes, no shots, no fouls). Everything has to be derived from the tracking.
- Homography and the broadcast video (not used so far).

**What we measured (label-free statistics, from `noise_profile.py` and earlier audits)**:

| Finding | Number | Why it matters |
|---|---|---|
| Ball missing | **41% of graded frames**, mostly in long gaps (67% of missing time in gaps > 20 s), mostly near the lines | Most gaps are dead balls, so a gap is strong D evidence, but not always |
| The ball track itself | Very clean: 0 jumps > 5 m, 0.7 false tracks per 10 min | We don't need a ball-noise filter (v3 taught us this) |
| **Ball-to-player distance** | Median **1.24 m** from the nearest player in open play; someone within 1.5 m only 58% of the time | **The key difficulty.** "Who is nearest" is ambiguous, so hand-overs are seen late or not at all |
| Players | 1,144 track pieces, median 41 s each; 9 of 11 players per team visible at p10 | The player on the ball is sometimes missing |
| Ball during stoppages | Mostly **not** tracked (only 31% of the 10 s before a set-piece tag) | Rules built for a tracked ball at a stoppage misfire |
| Smart Tagging set pieces | Only 59 tags for well over 100 restarts; a tag starts a median **6.3 s before** the kick | Useful for D timing, not enough on its own |
| Smart Tagging phase "team" field | Near coin flip | Dropped |
| Team shape / movement | Lags turnovers (spellF1 0.65 even when perfectly 1 s late) | Not usable as the main labeller |

---

## 4. How we measure progress without cheating: the test benches

We cannot score ourselves on BAR-ATH (rule 1, rule 3), and the leaderboard only shows our final
score. So we built a **bench**: other matches, made to look like BAR-ATH, scored with the
**official** metric code (`evaluation/possession_metrics.py`, so bench numbers are computed the
same way as the board's).

### 4.1 Matches

| Source | Matches | Licence / status | Truth comes from |
|---|---|---|---|
| Metrica public sample games 1-3 | 3 | public | their event data (truth 1) and their clean tracking (truth 2) |
| DFL / IDSSE open data (Bassek et al., Sci Data 2025) | 7 Bundesliga matches | CC-BY 4.0, other matches so allowed | the DFL's own per-frame `BallPossession` / `BallStatus` channels (truth 3) |

### 4.2 Three truths, because none of them is the organisers' truth

The organisers built their truth from corrected tracking. We can't see it, so we test against
three independent stand-ins and require a change to hold on all of them.

| Truth | How it's built | How well it mimics the board (baseline window shape*) | Role |
|---|---|---|---|
| **EV**, Metrica event truth | owner = team of the last pass / carry / set piece / shot / recovery; D from a stop to the restart | w25/w50 0.87, w125/w50 0.82 | regression check |
| **TRK**, Metrica tracking truth (`build_trk_gt.py`) | team of the player the ball is on in the CLEAN tracking: ball within 0.5 m, other team 0.5 m further away, for 3+ frames; flights and contested balls keep the last team; D copied from EV | **0.81 / 0.93** | **primary judge** |
| **DFL** truth | DFL's provider channels | ~1.0 / 0.76 in all 7 matches | independent check; counts ~406 changes per match (probably every touch), many more than the organisers seem to |

\* "Baseline window shape": we score the published baseline at smoothing windows 25, 50 and 125
on each truth and compare the ratios with the board's published ratios (**0.82 / 0.92**). The
truth whose ratios match best is the one that most likely counts possessions the way the
organisers do. TRK matches almost exactly; that's why it is the primary judge.

### 4.3 Degraded copies: making clean data look like BAR-ATH

Clean tracking is far easier than BAR-ATH, so each match is degraded with noise **measured** on
BAR-ATH (label-free):

| Copy | What it adds | Check |
|---|---|---|
| `real` | ball gaps (live 18%, dead 30% tracked), ball ~1 m off the carrier (+0.5 m ahead, sd 1.0), light jitter, player tracks broken into pieces (median 40 s), Smart Tagging tags simulated with the measured 6.3 s lead | the baseline scores on it reproduce the board's macroF1 within ~0.01 and its spellF1 ranking; bench reads ~0.04 easier |
| `real_occ` | as `real`, plus the player on the ball lost in 25% of contacts | harder stress copy (baseline below board) |
| (old) `clean`, `degraded`, `hard` | session-1 copies | regression only; they had the wrong noise (section 8) |

DFL copies use the same recipe **without** the carry offset, because the DFL ball is already
~1.37 m from the player, like BAR-ATH (1.24 m).

### 4.4 How a change is accepted

1. Score it on every match, on `real` and `real_occ`, against every truth.
2. **Rank on the mean of `real` and `real_occ`** ("R+O"); TRK first.
3. **A change must not lose on any truth or any copy.** A gain on one truth that loses on
   another is a sign we're fitting that truth, not the game.
4. Compare versions with each other on the bench. **Never compare a bench number with a board
   number** (the bench reads easier).
5. Log the decision, with numbers, in `labeller_v1_CHANGELOG.md`.

---

## 5. Code map: every file and what it is for

All of the possession work is in `test_scripts/`. Every labeller has the same interface:

```python
label(px, py, ball, team, blocks, params=None, tags=None) -> array of "H"/"A"/"D"/"X", one per frame
```

- `px, py`: (frames x slots) player positions in metres on a 105 x 68 pitch, NaN when missing
- `ball`: (frames x 2) ball position, NaN when missing
- `team`: 0 (home) / 1 (away) per slot
- `blocks`: `[(half, first_frame, last_frame)]`, the graded halves
- `params`: overrides for any constant or switch (every rule has one, so it can be measured alone)
- `tags`: Smart Tagging set-piece start frames

### 5.1 Labellers (each builds on the previous one)

| File | What it is | Status |
|---|---|---|
| `labeller_v1.py` | Base pipeline: ball cleaning, control rule, confirmed switches, minimum spell, first dead-ball rules | superseded |
| `labeller_v2.py` (now v2.1) | v1 + a new dead-ball detector based on **relative** player speed; rules A-D | superseded (was the safe file until Nov 7) |
| `labeller_v3.py` | v2.1 + false-ball filter + Viterbi segmentation (rejected) | output identical to v2.1 on BAR-ATH |
| `labeller_v4.py` | v2.1 + rule C off + control radius 2.0 m; directional/adaptive radius and rule E available as switches (rejected) | backup safe file |
| `labeller_v5.py` | experimental hand-over detector (reception vote, flight, gap vote, density switching) | **not adopted**; v6 imports pieces of it |
| `labeller_v6.py` | v4 + position smoothing, switch margin, minimum stoppage length (+ more switches, off) | **current safe submission** |

### 5.2 Run scripts (run on Ashwyn's PC against BAR-ATH)

| File | What it does |
|---|---|
| `run_labeller_v1.py` ... `run_labeller_v4.py`, `run_labeller_v6.py` | load BAR-ATH, run that labeller, write `evaluation/results/labeller_vN.csv`, validate it with the official checks, print label shares, possession count, stoppage count, % same as the safe file, and the `SANITY` line. `run_labeller_v6.py --as-v4` turns every v6 switch off and reproduces v4 byte for byte |

### 5.3 Bench and diagnostics

| File | What it does |
|---|---|
| `bench/build_metrica_bench.py` | builds the Metrica bench: event truth + the noisy copies (`--real-only` builds `real`/`real_occ`) |
| `bench/build_trk_gt.py` | builds the tracking-based truth (TRK) from the clean copy |
| `bench/fetch_dfl_bench.py` | downloads the 7 DFL matches via `kloppy` and saves them in bench format (`bench_data/dfl/<match>_clean.npz`, gitignored). Runs on Ashwyn's PC because the cloud can't reach huggingface. Has a progress bar and `--skip-existing` |
| `bench/score_bench.py` | scores any labeller on any bench copy with the official `evaluate()` |
| `bench/hutil.py` | fast (vectorised) possession-change finder used by the diagnostics |
| `noise_profile.py` | label-free noise statistics of a feed, measured the same way on BAR-ATH and on every bench copy (ball gaps, false tracks, ball-to-player distance, control rate, fragmentation, relative speed) |
| `gap_classification.py` | Nov 6 audit of ball gaps (dead tagged / untagged / kickoff / live) |
| `labeller_v1_CHANGELOG.md` | the full history: what was wrong, what changed, the bench effect, for every version. **Every labeller change is logged here** |

### 5.4 Parked or early work

| File | Status |
|---|---|
| `pass_detection.py` (+ changelog), `pass_spot_check.py` | pass detector, parked: passes don't decide possession labels. Contains rule-2 items to remove |
| `possession_test_1.py`, `visualize_events.py` | first zone-of-influence test (Nov 4) and the video/pitch visualiser |

### 5.5 Results (`evaluation/results/`, on Ashwyn's PC)

| File | md5 (first 8) | Status |
|---|---|---|
| `SAFE_SUBMISSION_labeller_v6.csv` | `6d084c06` | **current safe file, validated, to upload by Nov 10** |
| `SAFE_SUBMISSION_labeller_v4.csv` | `68952c15` | backup |
| `SAFE_SUBMISSION_labeller_v2_1.csv` | `ed254a5a` | older backup |

### 5.6 Handoff docs

`sessions/session1.md` (full session-1 handoff) and this brief.

---

## 6. Core logic of the labeller, step by step

This is the v6 pipeline as it runs today, in order. Constants are the v6 defaults.

### 6.1 Overview

```
tracking ──► 1 smooth positions ──► 2 clean ball ──► 3 speeds (ball, players, relative)
                                                           │
             ┌─────────────────────────────────────────────┤
             ▼                                             ▼
   4 control test per frame                        6 dead-ball mask (rules A, B, D)
   (who has the ball now?)                         7 drop D stretches < 2 s
             │                                             │
             └──────────────► 5 resolve: confirmed switches, ◄┘
                                backdating, hysteresis
                                         │
                                8 merge spells < 2 s
                                         │
                                  H / A / D per frame
```

### 6.2 Step by step

**1. Smooth positions (v6, `SMOOTH_K=3`).** Ball and player positions are averaged over ±3
frames (0.12 s each side) before anything else. Missing frames stay missing. *Why:* the ~1 m
ball offset wobbles frame to frame; smoothing stops the "nearest player" from flickering.

**2. Clean the ball (v1, `clean_ball`).** Drop jumps over `MAX_STEP` 2 m per frame; bridge gaps up
to `BRIDGE` 6 frames.

**3. Speeds, always over windows.**
- Ball speed over ±4 frames; player speed over ±12 frames.
- **Relative speed `rel`** (v2): the mean player speed, smoothed over 3 s, divided by **this
  match's own median**. So "players slowed down" means slower than usual *for this match*,
  which carries over between matches (BAR-ATH players are slow on average: median 1.9 m/s).
- *Why windows:* frame-to-frame jitter broke every stoppage in v1.0. This was the first big lesson.

**4. Control test per frame (v1, radius from v4, margin from v6).** For each frame, a team
**controls** the ball when:
- one of its players is within `CTRL_R` **2.0 m** of the ball (v1 had 1.5 m), and
- that player is at least `MARGIN` **0.3 m** nearer than the nearest opponent, and
- the ball is slower than `CTRL_V` 12 m/s (a ball in flight is nobody's).

Otherwise nobody controls it on that frame (a loose or contested ball).

**5. Resolve into possessions (`resolve_hyst`).** Frame-level control is noisy, so a change of
possession needs confirmation:
- **Confirmation:** the other team must control the ball for `CONFIRM` **6 frames** (0.24 s), with
  holes up to `HOLE` 3 frames allowed, before we switch.
- **Backdating:** when a switch is confirmed, it is placed at the **first** frame of the new
  team's control, not the 6th.
- **Hysteresis (v6, `SW_MARGIN=0.6`):** to *take* the ball from the team that has it, the
  challenger must be **0.6 m** clearer than the nearest opponent, not just 0.3 m. Keeping the ball
  needs less evidence than taking it. *Why:* 21% of our false changes were A-B-A blips where the
  "new" team barely touched the ball.
- **Nobody in control** (flight, loose ball, ball missing in live play): the last team keeps it.
  CHALLENGE.md says a contested ball stays with the team that had it.
- After a stoppage, the first team to control the ball gets the whole stretch back to the restart.

**6. Dead-ball mask (v2.1 rules, rule C off since v4).** A frame is D when one of these fires:

| Rule | Fires when | Notes |
|---|---|---|
| **A** ball out (`FIX_OUT`) | the ball goes past a line by more than 0.5 m | D until the next strike (> 5 m/s) inside the pitch |
| **B-line** | a ball gap of at least 1.5 s starts within 2 m of a line | most stoppages on BAR-ATH look like this |
| **B-slow** | a ball gap of at least 1.5 s with median `rel` < 0.75 | players slowed = whistle |
| ~~**C** stop with ball~~ | ~~ball present but quiet and `rel` < 0.7 for ≥ 4 s~~ | **OFF since v4**: it marked slow live build-up as dead (−0.02) |
| **D** set-piece tag | a Smart Tagging set-piece tag | D from where players slowed down to the first strike after the tag |

A stoppage ends at the **restart**: a strike over 5 m/s inside the pitch, or players back to normal
speed (`rel` > 1.0). D stretches less than 1 s apart are joined.

**7. Drop short stoppages (v6, `MIN_D=2.0`).** Any D stretch shorter than **2 s** is turned back
into live play. *Why:* a real stoppage lasts longer than 2 s, and a 0.8 s false D in the middle of
a possession splits it in two, which costs two spells in spellF1.

**8. Merge short spells (v1, `MIN_SPELL=2.0`).** A possession shorter than 2 s is merged into its
neighbours.

Output: H / A / D for every graded frame (never X).

### 6.3 Important cases and how they're handled

| Case | What the labeller does |
|---|---|
| Ball carried by a dribbler, ~1 m off | 2.0 m radius + smoothing still sees control |
| Defender chasing the carrier | 0.6 m switch margin stops him "taking" it on proximity alone |
| Pass between teammates | ball flies (> 12 m/s = nobody), receiver is a teammate: no change |
| Interception | the opponent controls for 6+ frames: change, backdated to his first touch |
| Contested ball / 50-50 | nobody controls clearly: last team keeps it |
| Ball vanishes in live play, no line, players at speed | last team keeps it (no D) |
| Ball vanishes near a line, or players slow | D (rule B), until the restart |
| Ball goes out of play | D (rule A) until the next strike inside the pitch |
| Throw-in, corner, goal kick with a tag | D from the slow-down to the strike (rule D) |
| Kickoff after half time | frames 110808-113741 are D (FAQ) |
| Very short "stoppage" (< 2 s) | treated as live (MIN_D) |
| Very short "possession" (< 2 s) | merged (MIN_SPELL) |

---

## 7. Version history v1 to v6, with numbers

### 7.1 Timeline

| Date | Stage | Outcome |
|---|---|---|
| Nov 1-4 | Data audit, movement idea, game plan, data explainer | no event data; ball missing 41%; shape lags turnovers |
| Nov 5-6 | Pass detection | plateaued; parked (passes don't decide possession; rule-2 contamination) |
| Nov 6 | Gap audit; bench v1; **labeller v1** | ~31% of graded time clearly dead; v1 runs, but D ≈ "ball missing" |
| Nov 6 | **v2 / v2.1**: relative-speed dead-ball detector | frozen as the first safe file |
| Nov 6-7 | **v3**: Viterbi (rejected), false-ball filter (bench artefact) | exposed that the bench had the wrong noise |
| Nov 7 | Noise profile of BAR-ATH; new `real` / `real_occ` copies | ball ~1 m off the carrier found |
| Nov 7 | **v4**: rule C off, radius 2.0 m | new safe file |
| Nov 8 | **v5**: hand-over detector; new tracking truth | v5 gains didn't hold; **not adopted** |
| Nov 8 | **v6**: false-change cuts + dead-ball timing | **current safe file** |
| Nov 8-9 | DFL third truth, 7 matches | v4/v6 confirmed; conflict on "more vs fewer changes" |
| Nov 9 | 3-truth grid (reception vote in v6) | **paused** for this brief |

### 7.2 What each version changed and why

**v1 (Nov 6).** First complete pipeline (sections 6.2 steps 2-5, 8). On BAR-ATH: 321
possessions, D 40.2%. *Problem:* it used an **absolute** player-speed threshold for dead balls
(1.8 m/s). BAR-ATH players are slow on average, so it fired everywhere and D was basically
"ball missing".

**v2 / v2.1 (Nov 6).** Replaced the dead-ball logic with **relative** speed (`rel`) and rules
A-D (including rule C, the "stop with the ball tracked" rule), carried-ball handling at
throw-ins, and set-piece tags. On BAR-ATH: 289 possessions, D 43.5%, 96 stoppages. *Significance:*
the first dead-ball detector that adapts to the match. Frozen as the first safe file.

**v3 (Nov 6-7).** Tried Viterbi segmentation (global best path instead of greedy switching):
**rejected**, it placed changes about 1 s early (0.489 vs 0.501). Added a false-ball filter: +0.010
on the old bench, **zero effect on BAR-ATH**. *Significance:* this is how we found out the old
bench had the wrong kind of noise (section 8, challenge 2).

**v4 (Nov 7).** After measuring BAR-ATH's real noise and rebuilding the bench:
- **Rule C off:** +0.02, and better on **all 6** realistic game copies. Rule C was built for
  stoppages with a tracked ball, which BAR-ATH mostly doesn't have; on BAR-ATH it was marking slow
  live build-up as dead.
- **Control radius 1.5 → 2.0 m:** +0.006, small but it matches the measured ~1 m offset and loses
  nowhere.
- Rejected, with numbers: weakened rule C (every weaker form lands between full and off),
  directional/ellipse radius (the chasing defender gets the same stretch), adaptive radius (inflates
  under occlusion, real_occ falls to 0.42), rule E "long slow run = dead" (also happens in live play).
- Checked: v4 has 13 possessions over 40 s on BAR-ATH (v2.1 had 6). Normal: the bench truth itself
  has 11-14 per match.
- On BAR-ATH: 316 possessions, D 39.9%, 98 stoppages, 91.5% same labels as v2.1.

**v5 (Nov 8): hand-over detector. Not adopted.**
- *Why we tried it:* oracles showed the headroom is in the team label, not the dead ball. With a
  perfect team label on live frames the bench would score ~0.86-0.89; with perfect D only ~0.54-0.60.
  v4 found only 40% of true hand-overs within 2 s.
- *What it does:* `HO_RECV` (when a ball flight over 7 m/s ends, vote on which team receives it),
  `HO_FLIGHT` (no control while the ball flies), `HO_GAP` (vote when the ball reappears),
  `HO_KEEP` / `HO_PROTECT`, and "density" switching (switch on 6 of the last 25 frames instead of a
  run of 6).
- *Result:* up to +0.017 on the event truth, but **−0.010 on the new tracking truth**. The gains
  came from chasing the event truth's extra hand-overs (an event is logged when the tracking often
  shows nothing yet). Not adopted.
- *Significance:* it led to the tracking truth (section 4.2), which is now our primary judge.

**v6 (Nov 8): cutting false changes and false stoppages. Current safe file.**
- *Diagnosis (tracking truth):* v4 predicts 640 changes where there are 585, and **40% are false**:
  56% right team but more than 2 s off (mostly late), 21% A-B-A blips, 12% wrong team before a
  stoppage, 8% wrong team after a stoppage. And of 322 predicted stoppages, 55 were false, median
  0.8 s long.
- *Kept:* `SMOOTH_K` 3 + `SW_MARGIN` 0.6 (+0.011) and `MIN_D` 2 s (+0.013, better in all 12
  game x copy x truth cells, flat from 1.5 to 4 s so not fragile).
- *Rejected:* weak-blip merging and longer MIN_SPELL (help the tracking truth, hurt the event
  truth: they remove short real possessions too); velocity-following control (adds nothing);
  gap split, shorter walk-back, delayed gap start (all lose).
- On BAR-ATH: **305 possessions, 95 stoppages, D 39.9%, 99.0% same labels as v4**. MIN_D touched
  only 3 stoppages (the bench's short false D's come partly from its own ball offset pushing the
  ball over the touchline, which BAR-ATH doesn't have). Expected real gain: small, perhaps
  +0.005-0.01 (a guess, not a measurement).
- Ashwyn chose v6 as the safe file on Nov 8 after a pros-and-cons review: bench gain up on all four
  copy x truth means, worst single cell −0.005, and 99% identical to v4 on BAR-ATH, so the downside
  is tiny.

### 7.3 Scores side by side (spellF1, mean of 3 Metrica games)

Bench copies, event truth (EV) and tracking truth (TRK). Higher is better.

| Version | EV real | EV real_occ | TRK real | TRK real_occ | BOTH (mean of EV and TRK R+O) |
|---|---|---|---|---|---|
| baseline_smooth50 | 0.287 | 0.202 | 0.282 | | |
| v1 | 0.516 | 0.473 | | | |
| v2.1 | 0.483 | 0.438 | 0.544 | 0.512 | ~0.49 |
| v4 | 0.516 | 0.462 | 0.563 | 0.518 | 0.515 |
| v5 best (recv + density) | 0.528 | 0.484 | 0.561 | 0.506 | ~0.52 |
| **v6** | **0.542** | **0.478** | **0.596** | **0.525** | **0.535** |

DFL truth (7 matches, mean of real and real_occ):

| Config | real | real_occ | mean | macroF1 | changes predicted / true |
|---|---|---|---|---|---|
| v2.1 | 0.466 | 0.436 | 0.451 | 0.830 | 190 / 406 |
| v4 | 0.483 | 0.452 | **0.468** | 0.831 | 205 / 406 |
| **v6** | 0.478 | 0.451 | **0.464** | 0.830 | 200 / 406 |
| v6 smoothing + margin only | 0.478 | 0.452 | 0.465 | 0.830 | 200 / 406 |
| v6 MIN_D only | 0.483 | 0.452 | 0.468 | 0.831 | 205 / 406 |
| v5 reception vote | 0.490 | 0.455 | 0.473 | 0.832 | 211 / 406 |
| v4, CONFIRM 3 + MIN_SPELL 1 (fast switching) | 0.524 | 0.481 | 0.503 | 0.833 | 282 / 406 |
| v4, MIN_SPELL 3 (slow switching) | 0.450 | 0.431 | 0.440 | 0.828 | 181 / 406 |

What the DFL results say:
- **v4 > v2.1 holds on a third, independent truth** (+0.017). The v4 decision is confirmed.
- **v6 ≈ v4 on DFL** (−0.004, a tie). v6's smoothing and margin cut changes, and DFL rewards more
  changes.
- **Fast switching wins on DFL (+0.035) but loses about −0.08 on the Metrica tracking truth.** The
  truths disagree on whether more or fewer changes is better. Because DFL's baseline window shape
  is the least board-like (it counts ~406 changes per match, probably every touch), we trust TRK
  here and did **not** adopt fast switching.

### 7.4 BAR-ATH plausibility checks (label-free, no score)

| Version | Possessions | Stoppages | D share | Same labels as previous safe |
|---|---|---|---|---|
| v1 | 321 | | 40.2% | |
| v2.1 | 289 | 96 | 43.5% | |
| v4 | 316 | 98 | 39.9% | 91.5% vs v2.1 |
| **v6** | **305** | **95** | **39.9%** | **99.0% vs v4** |

The `SANITY` line (public 5 min) moved 0.48 → 0.375 → 0.388 over these versions. It is ~17
possessions, so **it is noise and we ignore it on purpose** (rule 3).

---

## 8. Challenges we hit and how we solved them

| # | Challenge | What we did | Why it matters |
|---|---|---|---|
| 1 | **No way to score on the real match** (no hand labels, public 5 min misleads) | Built benches on other matches, scored with the official metric code, and calibrated them so the published baseline behaves on them as it does on the board | Every decision we've made is legal and measured |
| 2 | **The first bench had the wrong noise.** It reproduced the board's baseline scores, but with ~10 false balls per minute (BAR-ATH has almost none) and a ball snapped 8 cm from the player (BAR-ATH: ~1 m). v3's false-ball filter "won" on it and changed nothing on BAR-ATH | Wrote `noise_profile.py`, measured BAR-ATH and the bench the same way, rebuilt the copies (`real`, `real_occ`) to match the measured noise, then re-tested old decisions | Showed that matching scores is not enough; the noise itself must match. Led straight to rule C off and the 2 m radius |
| 3 | **Ball ~1 m off the player.** The 1.5 m control rule fired on only 40% of tracked frames | 2.0 m radius, position smoothing, a stricter margin to take the ball | Possessions now start closer to the true touch |
| 4 | **Which truth to trust?** v5 won on the event truth and lost on the tracking truth | Built a tracking-based truth, picked the judge by which truth reproduces the board's baseline shape, then added a third (DFL) | Stops us fitting the quirks of one truth |
| 5 | **Rule C misfired** in slow live build-up | Measured it on 6 copies; switched it off | +0.02, the single biggest bench gain of session 2 |
| 6 | **Absolute speed thresholds** fired everywhere (BAR-ATH players are slow) | Relative speed (`rel`) against the match's own median | Dead-ball logic that adapts to the match without labels |
| 7 | **False changes** (40% of v4's) | Diagnosed them by type, then targeted each type with a switch | v6 |
| 8 | **Short false stoppages** split possessions | `MIN_D` 2 s | +0.013, all 12 cells up |
| 9 | **DFL data blocked in the cloud** (huggingface unreachable; adding it to the environment was declined) | `fetch_dfl_bench.py` runs in Ashwyn's own Python and writes the bench files locally | 7 more matches, no data rules touched (DFL is CC-BY, other matches) |
| 10 | **Slow scripts looked hung** (275 s per DFL match with no output) | Rule: every script expected to run over 1 minute prints a live progress bar | Ashwyn can tell a slow run from a hung one |
| 11 | **Stale file on the PC** (Nov 6: Ashwyn ran old code twice) | Every file copied to the PC is checked by md5 on the PC before we say it's there | No more wasted runs |
| 12 | **A slow diagnostic hung** (O(n²) Python loop) | Rewrote it vectorised (`hutil.py`) | Diagnostics run in seconds |

---

## 9. Where we stand against the goal

- **Safe file:** v6, validated, ~**2x the baseline** on every bench copy and truth **(e.g. TRK real 0.596 vs baseline 0.282; EV real 0.542 vs 0.287)**
- **Rough board estimate:** scaling the bench by baseline-on-bench vs baseline-on-board
  (0.243 / ~0.285) gives **roughly 0.45-0.51 spellF1** for v6. This is a guess with a wide error,
  **not a measurement**. There is no feedback before the deadline, so we will only know when the
  results are out.
- **Where the headroom is:** the team label on live frames. With a perfect team label the bench
  would score ~0.86; with perfect D only ~0.60. So the next real gain needs **new evidence about
  who has the ball**, not more threshold tuning.
- **The control-rule approach has plateaued.** Every variant since v4 lands within 0.51-0.53 on
  the combined score; gains are now inside the 0.04 noise band. This is a real finding, not a
  failure: it tells us where *not* to spend the last week.
- **Biggest unresolved question:** the truths disagree on how many changes the organisers count.
  TRK (best calibrated) says "fewer, cleaner changes"; DFL says "more". v6 sits on the TRK side.

---

## 10. What Ashwyn is doing right now

**Adding the reception vote as a switch in v6, then running a grid on all three truths.**

- **What:** v5's best single piece, the **reception vote** (`HO_RECV`: when a ball flight over
  7 m/s ends, the team that receives it gets control evidence for the next 6 frames), is now a
  switch inside `labeller_v6.py` (default **off**, so v6's output does not change). The switch
  exists in the cloud test copy; it goes to the PC only if it is adopted.
- **Why:** it was the only v5 piece that gained on DFL (+0.009 on v4) and on the event truth
  (+0.010), but it lost on the tracking truth with v4 (−0.010). v6's stricter switch margin might
  remove the false receptions that made it lose. If it now holds on all three truths, it's a free
  gain; if not, we stop tuning.
- **The grid:** 9 configurations x 3 truths (EV, TRK, DFL) x 2 copies, on all 10 matches:

  | # | Config | What it tests |
  |---|---|---|
  | 1 | v6 | reference |
  | 2 | v6, CONFIRM 4 | slightly faster switching |
  | 3 | v6, CONFIRM 5 | |
  | 4 | v6, MIN_SPELL 1.5 | allow shorter possessions |
  | 5 | CONFIRM 4 + MIN_SPELL 1.5 | both |
  | 6 | CONFIRM 5 + MIN_SPELL 1.5 | |
  | 7 | v6 + reception vote | the new switch |
  | 8 | reception vote + CONFIRM 5 | |
  | 9 | reception vote + MIN_SPELL 1.5 | |

  Configs 2-6 probe the "more vs fewer changes" conflict carefully: small steps toward faster
  switching, to find out whether there is a middle point that loses on no truth.
- **Decision rule:** adopt a setting **only if it doesn't lose on any truth**. Otherwise v6 stays
  and tuning stops.
- **Status:** the grid is about 45% done and **paused** for this brief. It resumes when Ashwyn
  says so (~30-40 min of compute left).

---

## 11. Plan of action and strategy to Nov 17

### 11.1 Timeline

| When | What | Owner (proposed) |
|---|---|---|
| **Nov 10** (team target) | **Upload `SAFE_SUBMISSION_labeller_v6.csv` to the Google Form**, renamed `<team_name>.csv` (responses are editable, so this only protects us) | Ashwyn |
| Nov 9-10 | Finish the 3-truth grid; adopt a setting only if it loses nowhere; log the result | Ashwyn |
| Nov 10-12 | **Merge the team's branches**: review each teammate's work against the rules and the bench (section 12) | whole team |
| Nov 10-13 | Any last idea that adds **new evidence** (11.3), tested on the bench with the same acceptance rule | whoever owns it |
| Nov 12-14 | **Rule cleanup**: work through `TESTING_ONLY_REMOVE_BEFORE_SUBMISSION.md`, delete the rule-2 items, delete the file | one owner |
| Nov 12-14 | **Reproducible pipeline**: one entry point (e.g. `python make_submission.py`) from the data bundle to the exact CSV; the final labeller moved out of `test_scripts/` into a clean package; pinned dependencies (`pyproject.toml` / `uv.lock`, add anything imported); fixed seeds | one owner |
| Nov 13-15 | **Technical document**: approach, assets used and dropped, training data (only Metrica + DFL public matches, nothing from BAR-ATH), the possession rules implemented, limitations, how to reproduce | one owner (this brief is most of the raw material) |
| **Nov 15** | **Code freeze.** Re-run the pipeline from a clean checkout; the md5 must equal the submitted CSV | Ashwyn |
| Nov 16 | Make the repo public (no `data/`, no videos), validate with `evaluation/validate_submission.py`, submit the form | team |
| Nov 17 | Deadline (buffer day) | |

### 11.2 Strategy

1. **Protect what we have first.** Upload v6 by Nov 10. A missed deadline or a broken repo is a
   total loss; a slightly better labeller is a small gain.
2. **Stop tuning thresholds.** Gains inside the 0.04 noise band can't be trusted to carry over, and
   the control rule has plateaued. Only new *kinds* of evidence are worth the remaining time.
3. **Keep the acceptance rule strict:** no loss on any truth or copy. It is our only protection
   against overfitting a bench.
4. **Reproducibility is a hard requirement** (RULES.md rule 3). The freeze work is not optional
   polish; without it the file is not allowed.
5. **Integrate, don't fork.** Three people working three days without talking is the biggest risk
   right now. Everything that goes into the final pipeline must go through the same bench and the
   same changelog.

### 11.3 Ideas that could still add new evidence (only if time allows)

| Idea | Why it might help | Risk |
|---|---|---|
| Camera aim via homography as a ball proxy during gaps | 41% of frames have no ball; the camera follows the ball | needs work to test on the bench (no camera data in Metrica/DFL) |
| Check the ATD XML for ball-state or owner channels | could be free, strong evidence | if it's human-corrected, rule 2 may forbid it: check first |
| Ensemble / vote of v1 and v6 dead-ball logic | v1 ties v4 on the bench with different errors | small expected gain |
| Teammates' approaches (pass/flight detection, zone-of-influence, movement) | may contain new evidence | must pass the same bench and rules |

---

## 12. Key points to discuss in the meeting

1. **Safe upload by Nov 10 (our own target).** Confirm who uploads v6, who has the form link (it
   comes by email to registered participants), the exact team name for the file names, and that
   whoever uploads has a Google account.
2. **What has each of you built in the last 3 days?** For each branch: what it does, whether it
   was tested on other matches only, and its score on our bench (we can score any `label()`
   function on all three truths; send it over).
3. **Rule check of every branch.** Any hand labels, video verdicts, match stats or settings picked
   by the public 5 minutes on BAR-ATH? Anything found goes on the cleanup list before Nov 15.
4. **One integration path.** Agree on one `main`-line labeller (v6 unless a branch beats it on the
   bench by the acceptance rule) and one place to log decisions (`labeller_v1_CHANGELOG.md`).
5. **Owners for the freeze work:** rule cleanup, the reproducible entry point and pinned deps, the
   technical document, the final clean-checkout check.
6. **The "more vs fewer changes" question.** Our best-calibrated truth says fewer, DFL says more.
   v6 bets on fewer. Does anyone's work or reading of CHALLENGE.md suggest otherwise?
7. **Should we spend time on one more evidence source** (camera aim, XML channels) or go straight
   to freeze work? Proposed: freeze work first, new evidence only with spare hands.
8. **Repository hygiene:** `.gitignore` covers `data/`, `bench_data/` and generated reports; no
   match data in any branch before the repo goes public.

---

## 13. Glossary

| Term | Meaning |
|---|---|
| spell | an unbroken run of the same label (one possession or one stoppage) |
| spellF1 | the ranking metric: F1 of spells matched one to one by overlap |
| macroF1 | tie-break: mean per-frame F1 of H, A, D |
| bench | our test set: other matches, degraded to look like BAR-ATH |
| `real` / `real_occ` | the two realistic bench copies (`real_occ` also hides the player on the ball 25% of the time) |
| R+O | mean of `real` and `real_occ` |
| EV / TRK / DFL | the three truths: Metrica events, Metrica clean tracking, DFL provider channels |
| BOTH | mean of the EV and TRK R+O scores |
| control | a team has a player within 2.0 m of the ball, 0.3 m clearer than any opponent, ball < 12 m/s |
| `rel` | relative player speed: mean speed smoothed over 3 s / the match's median |
| CONFIRM | frames of control the other team needs before a switch (6) |
| SW_MARGIN | the stricter margin needed to take the ball (0.6 m) |
| MIN_SPELL / MIN_D | shortest possession / shortest stoppage we allow (2 s each) |
| oracle | a test where one part is made perfect from the truth, to see how much headroom that part has |
| SANITY line | score on the public 5 minutes; printed for format checking only, never used for decisions |

---

## 14. Appendix: commands and file hashes

Run on BAR-ATH (Ashwyn's PC, from the repo root):

```
python test_scripts/run_labeller_v6.py            # v6, writes evaluation/results/labeller_v6.csv
python test_scripts/run_labeller_v6.py --as-v4    # every v6 switch off = v4, byte for byte
python evaluation/validate_submission.py evaluation/results/SAFE_SUBMISSION_labeller_v6.csv
python test_scripts/bench/fetch_dfl_bench.py --skip-existing   # DFL bench (needs internet + kloppy)
```

File hashes on the PC (md5, first 8 characters):

| File | md5 |
|---|---|
| `test_scripts/labeller_v6.py` | `20d55518` |
| `test_scripts/run_labeller_v6.py` | `c0a7dd72` |
| `test_scripts/labeller_v4.py` | `f05f06f6` |
| `test_scripts/run_labeller_v4.py` | `70e506e2` |
| `test_scripts/labeller_v5.py` | `e7064fee` |
| `test_scripts/bench/build_trk_gt.py` | `6623d73e` |
| `test_scripts/bench/hutil.py` | `76175a82` |
| `test_scripts/bench/fetch_dfl_bench.py` | `3572d1b6` |
| `evaluation/results/SAFE_SUBMISSION_labeller_v6.csv` | `6d084c06` |
| `evaluation/results/SAFE_SUBMISSION_labeller_v4.csv` | `68952c15` |
| `evaluation/results/SAFE_SUBMISSION_labeller_v2_1.csv` | `ed254a5a` |

Earlier documents (optional background): game plan (Nov 3)
https://claude.ai/artifact/6pRyWAgBpJV8BnENkYGd6F, data explainer (Nov 4)
https://claude.ai/artifact/H9TF3Mt4kYHZUHXAM2AKAp, full session-1 handoff `sessions/session1.md`.
