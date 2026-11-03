# Possession Challenge Game Plan

Nov 3, 2026 · @Ashwyn

## Summary

We should build a three-part system: a **ball-control detector** that decides *when* possession changes, a **dead-ball detector** that splits possessions at stoppages, and a **state model** that turns both into clean, realistic possessions. Team shape and movement fills in *who* has the ball during the 41% of graded frames where the tracking loses it.

Two new ideas matter most. First, we have no way to measure progress today, because the 5 public minutes are not allowed for tuning. Open tracking data from other matches with real possession and ball-status labels (the DFL Bundesliga open dataset) would give us an honest test set. Second, writing `X` is never better than writing our previous label ourselves, so we should never abstain on graded frames.

The target is a safe submission that clearly beats 0.2433 by Nov 10, and the best version locked by Nov 15.

## What the score rewards

spellF1 scores *possessions*, not frames. Each predicted run of `H` or `A` must overlap one true possession of the same team by a temporal IoU of 0.3 to 0.7 (averaged over five thresholds), matched one to one. Five facts follow from this and should drive every design choice:

1. **Timing beats coverage.** A perfect labelling that is 1 s late scores 0.65; 3 frames late scores 0.98. Short possessions are the ones that die from lateness, so any smoothing must be centred or lag-corrected.
2. **Fragmenting is fatal.** Splitting one true possession into two halves usually fails both at IoU 0.5 and adds a false positive. This is why the raw baseline scores 0.05.
3. **Dead balls split possessions.** `H H D D H H` is two possessions. Missing a stoppage merges two true possessions into one long prediction that matches neither. Good `D` detection is a spellF1 lever, not just a macroF1 one.
4. **`X` is dominated on graded frames.** The scorer forward-fills `X` with our last label, so `X` equals writing that label ourselves on spellF1, but costs a miss on macroF1 (the tie-break). We should write the fill ourselves, or our best guess.
5. **The public window cannot pick between versions.** It holds 17 possessions; one miss moves spellF1 about 6 points, and tuning on it picked the wrong baseline. It is also forbidden as a tuning set. It is only for checking alignment and file format.

Scale check: the public window averages 14.2 s per possession, so the full match likely has a few hundred possessions. Our predicted count is a sanity check, not a tuning target.

## What we already know

| Finding | Source | What it means for us |
| --- | --- | --- |
| Ball missing on 40.7% of graded frames, in 150 gaps | Data audit (Nov 1) | Ball-only methods leave almost half the match blind |
| No gaps between 1 s and 5 s; 67% of missing time is in gaps over 20 s | Ball gap analysis | Short gaps are trivial to bridge; long gaps are the real problem |
| Long gaps cluster near the goal lines | Ball gap analysis | Many long gaps are probably dead balls (goal kicks, corners) |
| Team in possession flips across about 40% of long gaps | Ball gap analysis | We can't just copy the label from before the gap |
| Team shape lags a turnover by about 1 s or more | Viability report (Nov 2) | Shape can say *who*, but must not decide *when* |
| Ground truth comes mechanically from corrected tracking: `D` where it marks the ball dead, otherwise the team it names as owner | CHALLENGE.md | We are predicting what a tracking system would mark, not what a human would judge |
| Smart Tagging: 638 automatic phases with team tags; throw-ins have no team | DATA.md | A free prior on team and on restarts, with known holes |
| Baseline (nearest player, ±2 s vote) scores 0.2433 | EVALUATION.md | The floor; it has no dead-ball model and no possession structure |

Still open: the movement probe (possession\_movement\_probe.py) has not been run yet. Its results decide how much weight shape gets in the gap filler. The teammates' local work is undocumented, so some of the ideas below may already be started.

## The ideas, in priority order

P0 ideas are foundations we need before anything else counts. P1 ideas are where the score comes from. P2 ideas are stretch bets for after a safe submission exists. Effort is in person-days.

### P0. Foundations

**1. An honest test set from other matches**

- **What:** Use the [DFL / IDSSE open dataset](https://floodlight.readthedocs.io/en/latest/modules/io/datasets.html): seven full Bundesliga matches of tracking and event data, which include per-match possession and ball-status codes. Those codes are the same kind of labels our ground truth is built from. We then *degrade* the clean tracking to look like ours: delete the ball in gaps that copy our gap-length mix, drop and fragment player tracks, and remove player identities. Our scorer then gives a real spellF1 on seven matches.
- **Why it helps:** It is the only legal way to tune thresholds and compare versions. Without it every choice is a guess, and the public window already proved it picks the wrong answer.
- **Effort:** 2 to 3 days. **Risks:** Bundesliga labels may define possession slightly differently from Metrica's; check the licence and cite it in the tech doc. A bonus: this data is not competition data, so it can be analysed in the cloud.

**2. Check the tracking files for hidden ball fields (1 hour)**

- **What:** Read the ATD metadata XML and list every data channel it declares for the ball. If the automatic tracking carries a ball-state or owning-team channel, it is a direct noisy copy of what the ground truth is built from.
- **Why:** Possibly the single strongest feature, found in an hour. **Risk:** It may simply not be there.

**3. A pipeline skeleton that always produces a valid file**

- **What:** One command: load data, run modules, write `frame,label` CSV, run `validate_submission.py`, print possession count and mean length. Fix the broken `run_evaluation.py` import first.
- **Why:** Everyone plugs into the same pipeline from day one, and reproducibility is a submission requirement. **Effort:** 1 day.

### P1. The core system

**4. Ball-control detector (decides WHEN)**

- **What:** On frames with the ball, find the nearest player of each team and the ball speed. A player *controls* the ball when it stays within about 1.5 m of them, moving with them, for at least a few frames. A change of possession fires at the first frame of the opponent's controlled touch, not when the ball leaves the old team. Loose balls, deflections and lost aerial duels keep the last team, as CHALLENGE.md says.
- **Why it helps:** Fixes timing, which is worth more than anything else on spellF1, and stops the baseline's flicker at the source rather than by blurring.
- **Effort:** 3 days. **Risks:** Tracker identity switches near the ball; the "how many frames is control" threshold must be tuned on the test set from idea 1.

**5. Dead-ball detector**

- **What:** Combine cheap signals: ball outside the pitch lines, a long ball gap starting near a line, a sudden drop in average player speed, and set-piece shapes (players crowded in the box for a corner, a wall for a free kick, both teams in their own halves for a kickoff). Smart Tagging `SET PIECES` intervals mark restarts. The end of a dead ball is the first ball movement after the restart, and the restarting team is the one whose player touches it first.
- **Why it helps:** Each stoppage we find splits a merged prediction into two matchable possessions. The baseline has no model of this at all. It also decides the second half's first 2,934 frames, which are all `D`.
- **Effort:** 3 days. **Risks:** Quick restarts (throw-ins taken fast) look like live play; the ground truth may mark dead slightly before or after what we see.

**6. Gap filler from team shape and movement (decides WHO in gaps)**

- **What:** For ball gaps under about 20 s that are not dead balls, decide the team from shape: which team is pushing forward, whose defensive line drops, which team's players are more spread out (attackers spread, defenders compact), and where pressure gathers. If both sides of the gap agree and shape agrees, keep the label. If they disagree, place the change where shape changes fastest.
- **Why it helps:** Covers the 41% blind spot, and directly uses our original brainstorm idea where it is strongest.
- **Effort:** 3 days, after the probe results. **Risks:** Shape lags by about 1 s, so it should only pick the team and roughly place the change, never fine-tune timing.

**7. State model to join it all (clean possessions)**

- **What:** A hidden-state model over three states (H, A, D) with a minimum-duration rule. Each frame's evidence (ball control, dead-ball score, shape, tags) votes for a state, and a Viterbi pass picks the single most likely sequence, charging a penalty for every switch. This replaces majority-vote smoothing.
- **Why it helps:** Removes fragments without delaying real changes, which smoothing cannot do. One penalty knob controls the possession count.
- **Effort:** 2 days. **Risk:** Over-penalising merges short real possessions; tune on idea 1.

**8. Smart Tagging as a prior**

- **What:** Turn tags into evidence: a `BUILD UP` or `PROGRESSION` tagged Barcelona favours `H` across its interval; `DEFENDING IN ...` tagged Athletic also favours `H`; `ATTACKING TRANSITION` marks a recent change; set pieces mark `D` then a restart.
- **Why it helps:** Free team information inside ball gaps. **Effort:** 1 day. **Risk:** Phases are not possessions; use as a soft vote only, never copy intervals.

### P2. Stretch bets (only after a safe submission)

**9. Camera aim as a ball proxy**

- **What:** The tactical camera operator follows the ball. The homography tells us which pitch point sits at the image centre on every frame (100% valid), giving a rough ball location even when the ball is lost.
- **Why:** A cheap, always-available signal for gaps and for spotting stoppages (the camera often stops panning). **Effort:** 1 day. **Risk:** It lags and leads the ball by a few metres.

**10. Our own ball detector on the video**

- **What:** Run a pretrained ball detector on the tactical video, only on the \~65,000 graded frames where the ball is missing, and map hits to the pitch with the homography. Must run on Ashwyn's machine.
- **Why:** Shrinks the blind spot directly. **Effort:** 4 to 5 days plus compute. **Risks:** Highest effort; false detections; needs a GPU to be practical.

**11. Learned model on the test set**

- **What:** Train a small classifier (for example gradient boosting) on features from ideas 4 to 9, using the degraded DFL matches as labelled data, and use its outputs as the state model's evidence.
- **Why:** Lets data set the weights instead of us. **Effort:** 2 days once idea 1 exists. **Risk:** Overfitting to Bundesliga habits.

**Dropped:** the broadcast video (not synced, a third of frames unusable; the viability report covers why).

## How to split the work across three people

Each person owns one part of the pipeline with a fixed interface, so work runs in parallel and merges cleanly. Every module outputs a per-frame score array of the same length (199,319), and the state model reads them all.

| Role | Owns | Ideas | Interface they deliver |
| --- | --- | --- | --- |
| Integrator and evaluation (suggest Ashwyn) | Pipeline, test set from other matches, state model, final file, tech doc | 1, 2, 3, 7, 11 | `run.py` that writes and validates the submission; a scoring report on the DFL matches |
| Ball and timing | When possession changes | 4, 9 | Per-frame controlling team and confidence; list of candidate change frames |
| Dead ball and gaps | Stoppages and who has the ball in gaps | 5, 6, 8 | Per-frame dead-ball probability; per-frame team evidence in gaps |

Three team habits keep this safe:

- One shared config file for every threshold, so tuning never means editing code.
- Every change is judged on the DFL test set, never on the public window. The public window is only checked for alignment.
- A short daily note in the repo (`docs/log.md`) of what changed and its score, which later becomes the tech doc's "what we tried and dropped" section.

## Timeline to Nov 17

&#91;embedded content: timeline · Nov 3 to Nov 17\]

The safe submission on Nov 10 is our insurance: it only needs ideas 3 to 5 working and should beat 0.2433 clearly. After that, each improvement goes in only if it wins on the DFL test set. Nothing new after the Nov 15 freeze; the last two days are for reproducing the file from a clean clone, the tech doc, and submitting a day early.

## Risks, rules and next steps

| Risk | Fallback |
| --- | --- |
| DFL labels define possession differently from ours | Use it to compare versions, not for absolute scores; check alignment on the public window only |
| We run out of time on the core system | The Nov 10 safe submission is already in the form; the form takes edits until the deadline |
| File not reproducible from the repo | Pin dependencies, fix random seeds, and regenerate from a clean clone on Nov 15 |
| Match data leaks into the public repo | Add `data/` and `evaluation/results/` to `.gitignore` now (it is missing in our fork) |

Rules to keep in mind:

- No hand labels or hand corrections on this match, and no thresholds tied to specific frames of it.
- The public 5 minutes are for checking alignment and format only.
- Raw match data and videos never go to a cloud model; scripts run on Ashwyn's machine and only summaries come back.

Next steps this week:

- [ ] Ashwyn: run possession\_movement\_probe.py and share `reports/movement_probe/probe.md`
- [ ] Ashwyn: share the list of channel names in the ATD metadata XML (idea 2; no data values needed)
- [ ] Team: agree on owners for the three roles
- [ ] Download the DFL open dataset and confirm its licence
- [ ] Fix the `run_evaluation.py` import and add the `.gitignore`
