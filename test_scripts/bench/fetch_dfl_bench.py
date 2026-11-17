"""Download the 7 public DFL / IDSSE Bundesliga matches and save them in our bench format.

    python test_scripts/bench/fetch_dfl_bench.py                  # all 7 matches
    python test_scripts/bench/fetch_dfl_bench.py --matches J03WPY # just one (quick test)
    python test_scripts/bench/fetch_dfl_bench.py --skip-existing  # only the ones not saved yet

WHY. Our bench truths so far both come from Metrica's 3 sample games. This adds 7 independent
full matches whose per-frame truth is the DFL's own BallPossession (1 home / 2 away) and BallStatus
(1 alive / 0 dead) channels from their position data: a tracking-based truth from a real provider,
the closest thing we have to how the organisers build theirs.

DATA AND RULES. Bassek, Rein, Weber et al., "An integrated dataset of spatiotemporal and event
data in elite soccer", Sci Data 12, 195 (2025), CC-BY 4.0, data owner DFL. These are other matches,
so RULES.md allows using them in any amount. Nothing from BAR-ATH is read. kloppy downloads the
files from huggingface.co/datasets/pysport/idsse-data (needs internet on this PC).

OUTPUT  bench_data/dfl/<match>_clean.npz with the same keys as the Metrica bench:
    px, py   (frames x slots) players in metres on a 105 x 68 pitch, NaN when not on the pitch
    ball     (frames x 2) ball in metres, NaN when not tracked
    team     0 home / 1 away for every slot
    blocks   [(period, first_frame, last_frame)] 0-based rows
    gt       per frame "H" / "A" / "D" from the DFL channels (BallStatus 0 -> D)
    tags     rows of set-piece events from the DFL event data (empty if the events fail to load)
It prints one summary line per match (sizes, shares, possessions). Nothing else is written.

Needs: pip install kloppy numpy
"""

import argparse
import os
import time

import numpy as np

MATCHES = ["J03WMX", "J03WN1", "J03WPY", "J03WOH", "J03WQQ", "J03WOY", "J03WR9"]
L, W = 105.0, 68.0
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "bench_data", "dfl")


def progress(label, i, n, t0, width=30):
    """One-line progress bar, updated in place (Ashwyn's rule: anything over 1 minute shows one)."""
    f = (i + 1) / max(1, n)
    el = time.time() - t0
    eta = el / f - el if f > 0 else 0
    bar = "#" * int(width * f) + "-" * (width - int(width * f))
    print("\r  {} [{}] {:5.1f}%  {}/{}  {:4.0f}s elapsed  ETA {:4.0f}s".format(
        label, bar, 100 * f, i + 1, n, el, eta), end="", flush=True)
    if i + 1 >= n:
        print(flush=True)


def convert(match_id):
    from kloppy import sportec
    from kloppy.domain import BallState

    t0 = time.time()
    print("  {}: downloading + parsing tracking (no progress available from kloppy, ~1-3 min) ...".format(
        match_id), flush=True)
    ds = sportec.load_open_tracking_data(match_id=match_id, coordinates="kloppy", only_alive=False)
    print("  {}: loaded in {:.0f} s".format(match_id, time.time() - t0), flush=True)
    home, away = ds.metadata.teams[0], ds.metadata.teams[1]
    players = list(home.players) + list(away.players)
    slot = {p: i for i, p in enumerate(players)}
    team = np.array([0] * len(home.players) + [1] * len(away.players), dtype=np.int8)
    n, m = len(ds.frames), len(players)
    px = np.full((n, m), np.nan, dtype=np.float32)
    py = np.full((n, m), np.nan, dtype=np.float32)
    ball = np.full((n, 2), np.nan, dtype=np.float32)
    gt = np.full(n, "X", dtype="<U1")
    period = np.zeros(n, dtype=np.int8)
    tsec = np.zeros(n, dtype=np.float64)
    t1 = time.time()
    step = max(1, n // 200)
    for i, fr in enumerate(ds.frames):
        if i % step == 0 or i == n - 1:
            progress("converting frames", i, n, t1)
        period[i] = fr.period.id
        tsec[i] = fr.timestamp.total_seconds()
        if fr.ball_coordinates is not None and fr.ball_coordinates.x is not None:
            ball[i] = (fr.ball_coordinates.x * L, fr.ball_coordinates.y * W)
        for p, pd in fr.players_data.items():
            if pd.coordinates is not None and p in slot:
                px[i, slot[p]] = pd.coordinates.x * L
                py[i, slot[p]] = pd.coordinates.y * W
        if fr.ball_state == BallState.DEAD:
            gt[i] = "D"
        elif fr.ball_owning_team is not None:
            gt[i] = "H" if fr.ball_owning_team.team_id == home.team_id else "A"
    # a frame with the ball alive but no owner keeps the last owner
    last = "D"
    for i in range(n):
        if gt[i] == "X":
            gt[i] = last if last in "HA" else "D"
        last = gt[i]
    blocks = []
    for pid in np.unique(period):
        rows = np.flatnonzero(period == pid)
        blocks.append((int(pid), int(rows[0]), int(rows[-1])))

    tags = []
    try:
        from kloppy.domain import SetPieceQualifier
        ev = sportec.load_open_event_data(match_id=match_id, coordinates="kloppy")
        for e in ev.events:
            if e.get_qualifier_value(SetPieceQualifier) is None:
                continue
            rows = np.flatnonzero(period == e.period.id)
            if not len(rows):
                continue
            k = rows[np.clip(np.searchsorted(tsec[rows], e.timestamp.total_seconds()), 0, len(rows) - 1)]
            tags.append(int(k))
    except Exception as error:  # events are optional; the truth comes from the position data
        print("  note: no set-piece tags for {} ({})".format(match_id, str(error).splitlines()[0][:120]))

    return dict(px=px, py=py, ball=ball, team=team, gt=gt, blocks=np.array(blocks),
                tags=np.array(sorted(set(tags)), dtype=int)), ds.metadata.frame_rate


def summary(match_id, d, fps, secs):
    gt = d["gt"]
    ch = np.flatnonzero(gt[1:] != gt[:-1]) + 1
    starts = np.r_[0, ch]
    poss = sum(gt[s] in "HA" for s in starts)
    direct = sum(1 for s in ch if gt[s] in "HA" and gt[s - 1] in "HA")
    alive = gt != "D"
    print("{}: {} frames @ {} fps, {} slots, blocks {} | D {:.1f}% | possessions {} | direct H<->A "
          "{} | ball missing {:.1f}% of alive / {:.1f}% of dead | tags {} | {:.0f} s".format(
              match_id, len(gt), fps, d["px"].shape[1], d["blocks"].tolist(), 100 * np.mean(~alive),
              poss, direct, 100 * np.mean(np.isnan(d["ball"][alive, 0])),
              100 * np.mean(np.isnan(d["ball"][~alive, 0])) if (~alive).any() else 0.0,
              len(d["tags"]), secs))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--matches", nargs="*", default=MATCHES)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--skip-existing", action="store_true", help="skip matches already saved")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for j, mid in enumerate(args.matches):
        print("match {}/{}: {}".format(j + 1, len(args.matches), mid), flush=True)
        if args.skip_existing and os.path.exists(os.path.join(args.out, "{}_clean.npz".format(mid))):
            print("  already saved, skipped")
            continue
        t0 = time.time()
        try:
            d, fps = convert(mid)
        except Exception as error:
            print("{}: FAILED - {}".format(mid, str(error).splitlines()[0][:200]))
            continue
        path = os.path.join(args.out, "{}_clean.npz".format(mid))
        np.savez_compressed(path, **d)
        summary(mid, d, fps, time.time() - t0)
    print("saved to {}".format(os.path.relpath(args.out, ROOT)))


if __name__ == "__main__":
    main()
