"""Whole-match picture of the ball data, raw vs prepared, in one PNG (no video, seconds to run).

    python viz/ball_overview.py                       # -> outputs/ball/ball_overview.png
    python viz/ball_overview.py --bin 30              # 30 s bins instead of 60 s

Rows, per minute of video over the whole match:
  1. what the ball is on each frame: tracked / reconstructed / raw detection removed as a spike /
     missing (stacked %, graded halves shaded)
  2. frames in flight (outputs/flights/in_flight.npy) and the number of flights starting
  3. the longest missing-ball stretch in the bin (s)
plus the distribution of frame-to-frame ball speed, raw vs prepared (re-detections show up as an
impossible tail above ~35 m/s), and the gap-length histogram before / after.
Use it to find the minutes worth a closer look, then render them with
    python viz/render_match.py --new-ball --no-events --start-frame <frame> --duration 60
The image is derived match data: it stays in outputs/ (git-ignored).
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

import possession.io  # noqa: E402,F401
from possession.io import graded_blocks, load  # noqa: E402
from possession.preprocess.reconstruct import DEFAULT_FILE, load_ball_file  # noqa: E402
from possession.utils.frames import runs  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ball-file", default=os.path.join(ROOT, DEFAULT_FILE))
    ap.add_argument("--flights", default=os.path.join(ROOT, "outputs", "flights", "in_flight.npy"))
    ap.add_argument("--bin", type=float, default=60.0, help="seconds per bar")
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "ball", "ball_overview.png"))
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not os.path.isfile(args.ball_file):
        sys.exit("no {} - run scripts/reconstruct_ball.py first".format(args.ball_file))
    tr = load(cache=True)
    bf = load_ball_file(args.ball_file)
    n, fps = tr.n, tr.fps
    raw_x, raw_y = tr.bx.astype(float), tr.by.astype(float)
    x, y = bf["x"], bf["y"]
    filled, removed = bf["filled"], bf["removed"]
    tracked = ~np.isnan(x) & ~filled
    missing = np.isnan(x)
    spike = removed & missing                       # removed and not refilled
    missing_other = missing & ~spike
    flight = np.load(args.flights) if os.path.isfile(args.flights) else np.zeros(n, dtype=bool)
    starts = np.zeros(n, dtype=bool)
    starts[runs(flight)[0]] = True

    k = int(args.bin * fps)
    edges = np.arange(0, n + k, k)
    t_min = (edges[:-1] + tr.lo) / fps / 60.0

    def per_bin(mask):
        return np.array([mask[a:a + k].mean() * 100 if a < n else 0 for a in edges[:-1]])

    def longest_gap(a):
        seg = missing[a:a + k]
        s, e = runs(seg)
        return (e - s).max() / fps if len(s) else 0.0

    fig = plt.figure(figsize=(18, 13))
    gs = fig.add_gridspec(4, 2, height_ratios=[2.2, 1.2, 1.2, 1.8], hspace=0.45, wspace=0.18)
    ax = fig.add_subplot(gs[0, :])
    w = args.bin / 60.0
    cols = [("tracked", tracked, "#4c78a8"), ("reconstructed", filled, "#54a24b"),
            ("spike removed", spike, "#e45756"), ("missing", missing_other, "#c9c9c9")]
    bottom = np.zeros(len(t_min))
    for name, m, c in cols:
        v = per_bin(m)
        ax.bar(t_min, v, width=w, bottom=bottom, align="edge", color=c, label=name, linewidth=0)
        bottom += v
    for half, a, b in graded_blocks():
        ax.axvspan(a / fps / 60, b / fps / 60, color="gold", alpha=0.08, zorder=0)
    ax.set_ylabel("% of frames")
    ax.set_xlim(t_min[0], t_min[-1] + w)
    ax.set_title("Ball per {:.0f} s of video: tracked {:.1f}%, reconstructed {} frames, spikes "
                 "removed {}, missing {:.1f}% (graded halves shaded)".format(
                     args.bin, 100 * tracked.mean(), int(filled.sum()), int(removed.sum()),
                     100 * missing.mean()))
    ax.legend(loc="lower right", ncol=4, fontsize=9)

    ax2 = fig.add_subplot(gs[1, :], sharex=ax)
    ax2.bar(t_min, per_bin(flight), width=w, align="edge", color="#f58518", linewidth=0)
    ax2b = ax2.twinx()
    ax2b.plot(t_min + w / 2, [starts[a:a + k].sum() for a in edges[:-1]], color="k", lw=1)
    ax2.set_ylabel("% in flight")
    ax2b.set_ylabel("flights started")
    ax2.set_title("Flights (outputs/flights/in_flight.npy): {} flights, {:.1f}% of frames".format(
        int(starts.sum()), 100 * flight.mean()))

    ax3 = fig.add_subplot(gs[2, :], sharex=ax)
    ax3.bar(t_min, [longest_gap(a) for a in edges[:-1]], width=w, align="edge", color="#9d755d",
            linewidth=0)
    ax3.set_ylabel("s")
    ax3.set_xlabel("video time (min)")
    ax3.set_title("Longest missing-ball stretch per bin")

    def steps(xx, yy):
        st = np.hypot(np.diff(xx), np.diff(yy)) * fps
        return st[np.isfinite(st)]

    ax4 = fig.add_subplot(gs[3, 0])
    bins = np.linspace(0, 120, 121)
    for name, (xx, yy), c in (("raw", (raw_x, raw_y), "#e45756"), ("prepared", (x, y), "#4c78a8")):
        st = steps(xx, yy)
        ax4.hist(np.clip(st, 0, 120), bins=bins, histtype="step", color=c, lw=1.5, log=True,
                 label="{} ({} steps > 35 m/s)".format(name, int((st > 35).sum())))
    ax4.axvline(35, color="k", ls=":", lw=1)
    ax4.set_xlabel("frame-to-frame ball speed (m/s, clipped at 120)")
    ax4.set_ylabel("frames (log)")
    ax4.set_title("Ball speed between consecutive frames")
    ax4.legend(fontsize=9)

    ax5 = fig.add_subplot(gs[3, 1])
    gb = np.logspace(0, np.log10(max(10, n)), 40)
    for name, m, c in (("raw", np.isnan(raw_x), "#e45756"), ("prepared", missing, "#4c78a8")):
        s, e = runs(m)
        ax5.hist((e - s), bins=gb, histtype="step", color=c, lw=1.5,
                 label="{}: {} gaps".format(name, len(s)))
    ax5.axvline(int(bf["max_gap"]), color="k", ls=":", lw=1)
    ax5.set_xscale("log")
    ax5.set_xlabel("gap length (frames, log); dotted = fill limit")
    ax5.set_ylabel("gaps")
    ax5.set_title("Missing-ball gaps before / after preparation")
    ax5.legend(fontsize=9)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=110, bbox_inches="tight")
    print("wrote {}".format(os.path.relpath(args.out, ROOT)))


if __name__ == "__main__":
    main()
