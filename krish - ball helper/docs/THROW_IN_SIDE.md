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

CSV columns: `smart_id,start_sec,end_sec,half,video_frame_start,video_frame_end,inferred_taker,bar_cx,ath_cx,bar_score,ath_score`. Re-run with the script at
`/tmp/throwin_csv.py` if the tagging or tracking changes.

## Caveats
- This is **inference, not ground truth** — same caveat `DATA.md` gives for the Smart
  Tagging itself. Treat as a prior, validate against video on the close calls.
- GK exclusion is inferred from y-extreme; could drift if slot assignment changes mid-match.
- Throw-ins taken from deep in the defending third, or where both teams stack in the same
  half, would weaken the signal.

## Why
Closes the one restart where possession side is genuinely unknown from the provided tags,
using only data that is actually in the ATD.

## How to apply
When a possession model needs the throw-in taker, read `eda/throw_in_takers.csv`.
Link with [[GAPS_AND_CLIPS]] and [[PROJECT_OVERVIEW]].