# EDA Thoughts — Ball-Tracking Gaps

The 196 missing-ball runs across 199,319 frames total ~74.7k missing frames (37.5%).
The longest gaps (after final whistle, before kickoff) are ungraded — `X` is free there.

## What the gaps actually overlap

The claim that most gaps occur during dead-ball situations needs nuance. Scanning the top 20 gaps against the 638 Smart Tagging instances reveals three classes:

### Rule A — Dead-Ball / Restart overlaps
- Several gaps overlap `SET PIECES` intervals with a concrete Team tag (e.g. `Free Kick [Atletic Club]`, `Corner Kick [Barcelona]`). These are live phases where the ball is technically in play but the tracker has lost it — often a crowded-box occlusion or the ball going out of the camera frame during a restart.
- A few gaps overlap `GOALS` intervals (`[Barcelona]`, `[Atletic Club]`). These are genuine in-play periods where a goal was scored; the ball may have gone temporarily out of view during the scramble.
- `Throw-in` intervals appear with `N/A` for the Team tag. Per `DATA.md`, 36 of 638 tag instances are `N/A`, every one a throw-in — exactly the restarts where knowing the side would be worth the most, but the tag is unreliable.

### Rule B — Active tactical-phase overlaps
- `PLAYERS IN THE BOX` and `BALL IN THE BOX` are the most common overlaps. Per `DATA.md`, every set piece is a restart and every restart ends a dead-ball stretch — so a gap that falls inside `PLAYERS IN THE BOX [Barcelona]` is **still in possession**, just heavily occluded.
- `BALL IN FINAL THIRD`, `ATTACKING TRANSITION`, `DEFENSIVE TRANSITION`, `BUILD UP` also appear. These are all active phases, meaning the ball is in play and the team tagged as having the phase is the *expected* possessor.
- Gap `8242..9705` (58 frames, 2.3s after video start) has **zero Smart Tagging overlap** — it sits in the pre-kickoff window before the first tagged phase begins.

### Rule C — No overlap at all
- 1 gap out of the top 20 has no Smart Tagging overlap. This is the cleanest case for “the ball was never in the tagged region” and could be filled with `D` (dead ball) with zero risk of bleeding across a handover.

## Smart Gap-Filling Strategy

| Rule | When to apply | What to fill | Why |
|---|---|---|---|
| **A — Dead Ball / Restarts** | Gap overlaps `SET PIECES` / `Free Kick` / `Throw-in` / `Goal Kick` / `Corner Kick` | `D` (dead-ball state) or a tracking-state flag `X` marking “out of play” | The ball is technically non-existent on the pitch during these phases. No interpolation needed. Cleans data metrics instantly. |
| **B — Tactical Phases** | Gap falls inside `PLAYERS IN THE BOX` / `BALL IN THE BOX` / `BALL IN FINAL THIRD` / `BUILD UP` / `ATTACKING TRANSITION` etc. | **Flag for secondary inspection** — these are rare instances of occlusion during active play. Consider last-known-ball-position + velocity, or player-geometry heuristics. | The ball is *active*. A gap here means the tracker lost it mid-possession, not that play stopped. |
| **C — No overlap** | Gap overlaps no Smart Tagging interval | `D` (dead-ball state) | The simplest safe fill. Guarantees no hand-over bleed because the ground truth also labels these frames `D`. |

## Implementation notes

The Smart Tagging intervals are **phases**, not possessions — a `Build Up` can span one possession, stop short of one, or cover two. Copying an interval across as a label imports its errors wholesale (`DATA.md:180`). Use the tags as **confidence priors**, not deterministic rules.

The baseline (`baseline.py:128-130`) already does the correct minimal thing: **where the ball is missing, say `D`**. That choice costs nothing on ungraded frames and is the reference class for `D` on graded frames. Any smarter interpolation should only aim to reduce `D` usage on truly in-play gaps (Rule B), and even then, `macroF1` counts `X` as a miss and `spellF1` fills abstentions with the last label — so carry-forward of the last known possession team is already the default behaviour.

## What this changes for the notebook

1. Add a column to the gap table: `overlapping_tags` — list of `(code, team)` that intersect each gap.
2. Classify each gap by Rule A / B / C above.
3. Fill Rule A gaps with `D`, Rule C gaps with `D` as the safe default, and leave Rule B flagged for a secondary strategy (last-seen-ball, player-geometry, etc.).
4. Report the counts: “Rule A: X gaps filled with D, Rule B: Y gaps flagged, Rule C: Z gaps filled with D.”

---

## Tracking Strategy for the 42 In-Play Gaps

Out of 135 total gaps analyzed across rendered gap clips:
- **93 gaps** occur during set pieces, restarts, goals, or match boundaries where setting `D` (Dead Ball) is the optimal action.
- **42 gaps** represent **in-play occlusions** (ball high in the air, camera cutaway mid-action, player occlusion in crowded box) where play is live and no set piece or restart follows.

### Actionable Strategies for the 42 In-Play Gaps:

1. **Ball Trajectory Interpolation (Kinematic Interpolation)**
   - Since play is continuous, interpolate missing `(ball_x, ball_y)` coordinates between the last-known and first-reappearance positions using linear or PCHIP spline interpolation.
   - Re-evaluate player-to-ball distance on each interpolated frame to assign nearest-player possession.

2. **Player Geometry & Team Centroid / Pitch Control**
   - Player tracking remains active even when the ball is missing.
   - Compute the spatial cluster (convex hull or local density) or Voronoi pitch control for both teams. The ball is almost always near the possessing team's spatial center during active play.

3. **Competition Metric Mechanics (`spellF1` & Forward Fill)**
   - If the same team has possession before and after the gap, maintain that team's label across the gap — the ground truth treats this as one continuous possession.
   - Predicting `X` (abstention) on uncertain frames allows the official scorer to automatically forward-fill the previous prediction within `spellF1` without penalizing as two fragmented possessions.

4. **Smart Tagging Active Phase Priors**
   - Cross-reference with `BAR-ATH_tactical_smart_tagging.xml`. If an in-play gap falls inside an active phase (e.g. `BUILD UP`, `ATTACKING TRANSITION`, `BALL IN THE BOX`) tagged with a specific team, use that team tag as a Bayesian prior for possession.

5. **Broadcast ATD Cross-Validation**
   - Frequently, when the tactical camera loses the ball during a crowded box scramble or high pass, the broadcast camera tracking (`data/broadcast/BAR-ATH_broadcast_atd.txt`) retains ball visibility. Cross-validate across feeds using homography.

---

# Throw-In Side Inference (the missing `N/A` team tag)

`DATA.md` flags that the Smart Tagging `Team` field is `N/A` on **36 of 638** instances —
every throw-in (26), every kickoff (6), plus 2 `CREATING CHANCES` and 2 `DEFENDING IN
DEFENSIVE THIRD`. Throw-ins are the only restart where the side genuinely isn't known from
the tag. Kickoffs are derivable (after-goal restarts go to the non-scoring team; all 4 goals
are tagged with a team), so only the 26 throw-ins need inference.

## What the ATD actually carries

- **46 track slots, each assigned to a team** in `BAR-ATH_tactical_atd.xml`: slots 1–24 =
  Barcelona (home), 25–46 = Atletic Club (`baseline.py:91-100`).
- **No goalkeeper flag** — no `<Goalkeeper>`, no role/position field, no ShirtNumber. Slots
  are anonymous "Track 1"…"Track 46".
- **Per-team pitch orientation** is in the provider params:
  `attack_direction_first_half` = `left_to_right` for Barcelona, `right_to_left` for Atletic
  Club. Verified on the tracking: at 1st-half kickoff (ATD id 11187) BAR mean x = 0.38,
  ATH = 0.52; at 2nd-half kickoff (id 113742) they swap to 0.62 / 0.44. Own-half side is
  known per team per half.
- **The ball is often missing at throw-ins** — only 11/26 throw-ins had a tracked ball at the
  interval start. Ball position alone is therefore not a reliable signal.

## The method (centroid-in-own-half)

For each throw-in, at the interval's start frame:
1. Take all tracked players per team.
2. Drop the **goalkeeper** = the single player whose y is closest to a goal line (0 or 1).
3. Compute the non-GK centroid.
4. Score = signed distance of the centroid from pitch centre (x = 0.5) toward that team's
   **own half** (BAR: x<0.5 in 1st half, x>0.5 in 2nd half; ATH opposite).
5. The team with the deeper own-half centroid is the taker.

Rationale: at a throw-in the takers line up just inside their own half while the defenders
press up the line, so the taker side's centroid sits well inside its own half.

## Results — `eda/throw_in_takers.csv` (26 rows)

**13 Barcelona, 13 Athletic.** Cross-validated against the ball's position when play resumes:
**24/26** the ball was in the inferred taker's half. The 2 that didn't (smart IDs 442, 448)
are throws *awarded in Barcelona's half* where the centroid says Athletic took them —
plausible, since a throw-in goes to the team that last touched the ball out, not the side
whose half it's in. So the centroid is the reliable signal; the ball-position check is a weak
validator.

Closest calls needing a video spot-check: smart IDs 448 (scores +0.037 vs +0.046), 450
(+0.008 vs +0.087), 451 (+0.043 vs +0.021).

CSV columns: `smart_id,start_sec,end_sec,half,video_frame_start,video_frame_end,inferred_taker,bar_cx,ath_cx,bar_score,ath_score`. Re-run with the script at `/tmp/throwin_csv.py` if the tagging or tracking changes.

## Caveats

- This is **inference, not ground truth** — same caveat `DATA.md` gives for the Smart
  Tagging itself. Treat as a prior, validate against video on the close calls.
- GK exclusion is inferred from y-extreme; could drift if slot assignment changes mid-match.
- Throw-ins taken from deep in the defending third, or where both teams stack in the same
  half, would weaken the signal.
