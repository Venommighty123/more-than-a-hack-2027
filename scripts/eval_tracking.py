"""Label-free checks of our player tracking: the numbers that replaced the ATD comparison.

    python scripts/eval_tracking.py                          # cleaned slots (what the labellers read)
    python scripts/eval_tracking.py --raw-players            # the same before the clean-up
    python scripts/eval_tracking.py --team-mode track        # team from the whole-track vote

There is no ground-truth tracking for BAR-ATH and RULES.md forbids making any, so every check is a
property true football must have. All on the graded halves, positions in metres:

  people_per_frame    players + goalkeepers tracked per frame (22 when everyone is in view)
  coverage_20_23      share of frames with 20-23 people
  team_overflow       share of frames where one team has more than 11  -> wrong team / duplicates
  jumps_per_1000      steps faster than 12 m/s over 0.2 s per 1,000 slot steps  -> ID switches
  segments_per_half   continuous pieces of slot tracks (a gap > 1 s splits)  -> fragmentation
  median_segment_s    their median length
  ball_present        share of frames with a ball position (fixed ball)
  referees_per_frame  referee rows in the YOLO tracks (ref + assistants ~ 1-3)
  gk_visible_share    share of frames with at least one goalkeeper row
  gk_sane             on those frames: every goalkeeper on its own side, at most one per team

OUTPUT outputs/track_eval/ (git-ignored): summary_<tag>.json and per_minute_<tag>.csv (overflow,
low coverage, ball missing and jumps per video minute: where to look with viz/visualize_labeller.py).
Never tune anything on these numbers for this match beyond fixing evident errors (RULES.md rule 1).
"""
import argparse
import json
import os

try:
    import _bootstrap  # noqa: F401  (python scripts/x.py)
except ImportError:
    from scripts import _bootstrap  # noqa: F401  (python -m scripts.x)

import numpy as np
import pandas as pd

from possession.io import AWAY, HOME, graded_blocks, graded_mask, load
from possession.io.detections import DEFAULT_FILE as PLAYERS_FILE

ROOT = _bootstrap.ROOT
JUMP_SPEED, LAG, SPLIT_GAP = 12.0, 5, 25


def slot_metrics(tr, graded):
    ok = np.isfinite(tr.px) & graded[:, None]
    team = tr.teams.team
    h, a = ok[:, team == HOME].sum(1)[graded], ok[:, team == AWAY].sum(1)[graded]
    cnt = h + a
    px, py = tr.px.astype(np.float64), tr.py.astype(np.float64)
    d = np.hypot(px[LAG:] - px[:-LAG], py[LAG:] - py[:-LAG]) * tr.fps / LAG
    both = ok[LAG:] & ok[:-LAG]
    jumps = (d > JUMP_SPEED) & both
    seg_len = []
    for s in range(px.shape[1]):
        f = np.flatnonzero(ok[:, s])
        if len(f):
            cut = np.r_[0, np.flatnonzero(np.diff(f) > SPLIT_GAP) + 1, len(f)]
            seg_len += list(np.diff(cut))
    seg_len = np.array(seg_len, float) / tr.fps
    return {"people_per_frame": float(cnt.mean()), "home_per_frame": float(h.mean()),
            "away_per_frame": float(a.mean()),
            "coverage_20_23": float(np.mean((cnt >= 20) & (cnt <= 23))),
            "team_overflow": float(np.mean((h > 11) | (a > 11))),
            "jumps_per_1000": float(1000 * jumps.sum() / max(both.sum(), 1)),
            "segments_per_half": float(len(seg_len) / len(graded_blocks())),
            "median_segment_s": float(np.median(seg_len)) if len(seg_len) else float("nan"),
            "ball_present": float(np.isfinite(tr.bx[graded]).mean())}, (h, a, jumps)


def left_team(tr, first):
    """Team whose players stand on the left (small x) in the first 10 s of a half."""
    sl = slice(first - tr.lo, first - tr.lo + 10 * tr.fps)
    team = tr.teams.team
    mx = [np.nanmean(tr.px[sl][:, team == t]) for t in (HOME, AWAY)]
    return HOME if mx[0] < mx[1] else AWAY


def row_metrics(tr, path):
    """Referees and goalkeepers from the raw YOLO rows (they are not in the labeller slots)."""
    rows = pd.read_parquet(path, columns=["frame", "role", "team", "x_m"])
    rows = rows[graded_mask(tr.n, tr.lo)[rows.frame.to_numpy() - tr.lo]]
    graded_frames = np.flatnonzero(graded_mask(tr.n, tr.lo)) + tr.lo
    n = len(graded_frames)
    out = {"referees_per_frame": float((rows.role == "referee").sum() / n)}
    g = rows[rows.role == "goalkeeper"].copy()
    if not len(g):
        out.update(gk_visible_share=0.0, gk_sane=float("nan"))
        return out
    g["tm"] = g.team.map({"H": HOME, "A": AWAY})
    g["ok_side"] = False
    L = tr.pitch[0]
    for half, first, last in graded_blocks():
        lt = left_team(tr, first)
        m = (g.frame >= first) & (g.frame <= last)
        own_left = g.tm == lt
        g.loc[m, "ok_side"] = np.where(own_left[m], g.x_m[m] < L / 2, g.x_m[m] >= L / 2)
    per = g.groupby("frame").agg(n_bad=("ok_side", lambda s: int((~s).sum())),
                                 n_h=("tm", lambda s: int((s == HOME).sum())),
                                 n_a=("tm", lambda s: int((s == AWAY).sum())))
    sane = (per.n_bad == 0) & (per.n_h <= 1) & (per.n_a <= 1)
    out.update(gk_visible_share=float(len(per) / n), gk_sane=float(sane.mean()))
    return out


def per_minute(tr, graded, h, a, jumps):
    frames = np.flatnonzero(graded) + tr.lo
    m = pd.DataFrame({"frame": frames, "overflow": (h > 11) | (a > 11), "low_coverage": (h + a) < 20,
                      "ball_missing": ~np.isfinite(tr.bx[graded])})
    j = np.zeros(tr.n, int)
    j[LAG:] = jumps.sum(1)
    m["jumps"] = j[graded]
    m["minute"] = m.frame // (60 * tr.fps)
    return m.groupby("minute").agg(first_frame=("frame", "min"), overflow=("overflow", "mean"),
                                   low_coverage=("low_coverage", "mean"),
                                   ball_missing=("ball_missing", "mean"), jumps=("jumps", "sum"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--players-file", default=os.path.join(ROOT, PLAYERS_FILE))
    ap.add_argument("--team-mode", choices=("local", "track"), default="local")
    ap.add_argument("--raw-players", action="store_true", help="evaluate before the clean-up")
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "outputs", "track_eval"))
    args = ap.parse_args()

    tr = load(players_file=args.players_file, team_mode=args.team_mode, clean=not args.raw_players)
    graded = graded_mask(tr.n, tr.lo)
    res, (h, a, jumps) = slot_metrics(tr, graded)
    res.update(row_metrics(tr, args.players_file))
    tag = "{}_{}".format(args.team_mode, "raw" if args.raw_players else "clean")
    res.update(source=tr.stats["players"], ball=tr.stats["ball"], team_mode=args.team_mode,
               clean=not args.raw_players, graded_frames=int(graded.sum()))

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "summary_{}.json".format(tag)), "w") as handle:
        json.dump(res, handle, indent=2)
    pm = per_minute(tr, graded, h, a, jumps)
    pm.round(4).to_csv(os.path.join(args.out_dir, "per_minute_{}.csv".format(tag)))

    print("players: {} | team mode {} | {}".format(res["source"], args.team_mode,
                                                  "raw (no clean-up)" if args.raw_players else "cleaned"))
    fmt = {"people_per_frame": "{:.2f}", "home_per_frame": "{:.2f}", "away_per_frame": "{:.2f}",
           "coverage_20_23": "{:.1%}", "team_overflow": "{:.1%}", "jumps_per_1000": "{:.2f}",
           "segments_per_half": "{:,.0f}", "median_segment_s": "{:.1f} s", "ball_present": "{:.1%}",
           "referees_per_frame": "{:.2f}", "gk_visible_share": "{:.1%}", "gk_sane": "{:.1%}"}
    for k, f in fmt.items():
        v = res.get(k)
        print("  {:<20} {}".format(k, "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f.format(v)))
    worst = pm.sort_values("overflow", ascending=False).head(5)
    print("minutes with the most team overflow:")
    print(worst[["first_frame", "overflow", "low_coverage", "ball_missing", "jumps"]].round(2).to_string())
    print("-> {}".format(os.path.relpath(args.out_dir, ROOT)))


if __name__ == "__main__":
    main()
