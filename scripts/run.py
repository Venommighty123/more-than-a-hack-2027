"""Run a labeller on BAR-ATH and write a full-match submission CSV, then validate and summarise it.

    python scripts/run.py                                  # labeller v1 -> outputs/submissions/v1.csv
    python scripts/run.py --labeller baseline50
    python scripts/run.py --out outputs/submissions/try.csv --no-cache
    python scripts/run.py --labeller v5                    # uses outputs/ball/ball_reconstructed.npz
    python scripts/run.py --labeller v5 --set CONFIRM=6 --out outputs/submissions/v5_confirm6.csv

Labellers that accept ``ball_source`` (v5) get the prepared ball written by
scripts/reconstruct_ball.py when the file exists (--no-ball-file to ignore it); they compute the
same thing in memory otherwise. ``--set KEY=VALUE`` overrides a labeller parameter (as in
scripts/bench_score.py) -- only for settings already scored on the bench.

Replaces Ashwyn's test_scripts/run_labeller_v1.py. What it prints is safe to share (no raw data):

  1. the official validation of the file (evaluation/possession_metrics.validate_submission)
  2. what the file says: label shares, number and length of possessions, stoppages
  3. a SANITY score on the public 5 minutes. RULE: format / sanity check only -- it must never be
     used to choose between versions or settings (EVALUATION.md: tuning on it picks the wrong
     option). Versions are compared on the bench (scripts/bench_score.py).

Frames outside the graded halves are X. The labeller sees play from each kick-off: half 2 is graded
from 110808 but kicks off at 113742, and FAQ.md says those 2,934 frames are all D, so they are
written as D directly.

The submission is derived from the match data: outputs/ is git-ignored, keep it that way.
"""
import argparse
import inspect
import os
import time

try:
    import _bootstrap  # noqa: F401  (python scripts/x.py)
except ImportError:
    from scripts import _bootstrap  # noqa: F401  (python -m scripts.x)

import numpy as np

import possession_metrics as pm
from games import GAME, GAMES, anchor_video_frame
from possession.io import graded_blocks, load, load_tags
from possession.labellers import get_labeller
from possession.preprocess.reconstruct import DEFAULT_FILE, load_ball_file

ROOT = _bootstrap.ROOT
N_FRAMES = GAMES[GAME]["n_video_frames"]
FPS = GAMES[GAME]["fps"]
PUBLIC_GT = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
OUT_DIR = os.path.join(ROOT, "outputs", "submissions")


def play_blocks(game=GAME):
    """Graded halves, with half 2 starting at its kick-off."""
    kickoff2 = anchor_video_frame(GAMES[game]["kickoff_2nd"])
    return [(h, max(a, kickoff2) if h == 2 else a, b) for h, a, b in graded_blocks(game)], kickoff2


def describe(labels, blocks):
    graded = np.zeros(len(labels), dtype=bool)
    for _, a, b in blocks:
        graded[a:b + 1] = True
    g = labels[graded]
    shares = {k: 100.0 * np.mean(g == k) for k in "HADX"}
    spells, stops = [], 0
    for _, a, b in blocks:
        seg = labels[a:b + 1]
        edges = np.r_[0, np.flatnonzero(seg[1:] != seg[:-1]) + 1, len(seg)]
        for s, e in zip(edges[:-1], edges[1:]):
            if seg[s] in "HA":
                spells.append(e - s)
            elif seg[s] == "D":
                stops += 1
    spells = np.array(spells, dtype=float) / FPS
    print("graded frames: H {H:.1f}%  A {A:.1f}%  D {D:.1f}%  X {X:.1f}%".format(**shares))
    if len(spells):
        print("possessions: {}  (mean {:.1f} s, median {:.1f} s, shorter than 2 s: {})".format(
            len(spells), spells.mean(), np.median(spells), int((spells < 2).sum())))
    print("stoppages (D stretches): {}".format(stops))


def _value(text):
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return {"True": True, "False": False, "None": None}.get(text, text)


def tag_input(fn, tr):
    """Smart Tagging in the form the labeller expects: Tag records (v2) or set-piece start frames
    (v4 / v5 / v6, Ashwyn's run_labeller_v4.load_set_piece_frames)."""
    tags = load_tags(tr.teams)
    module = getattr(fn, "__module__", "")
    if module.endswith((".v4", ".v5", ".v6")):
        frames = sorted(t.start_frame for t in tags
                        if (getattr(t, "code", "") or "").strip().lower() == "set pieces")
        print("set-piece tags: {}".format(len(frames)))
        return np.array(frames, dtype=int) - tr.lo
    return tags


def write_csv(path, labels):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, v) for i, v in enumerate(labels))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--labeller", default="v1", help="registered name or package.module")
    ap.add_argument("--out", default=None, help="default outputs/submissions/<labeller>.csv")
    ap.add_argument("--no-cache", action="store_true", help="re-read the raw tracking file")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a labeller parameter, e.g. --set CONFIRM=6")
    ap.add_argument("--ball-file", default=os.path.join(ROOT, DEFAULT_FILE),
                    help="prepared ball for labellers that take ball_source (v5)")
    ap.add_argument("--no-ball-file", action="store_true", help="always prepare the ball in memory")
    ap.add_argument("--bench-pitch", action="store_true",
                    help="use 105 x 68 like the bench (reproduces the original v1 file exactly)")
    args = ap.parse_args()
    out = args.out or os.path.join(OUT_DIR, "{}.csv".format(args.labeller.replace(".", "_")))

    t0 = time.time()
    tr = load(cache=not args.no_cache)
    print("H = {}, A = {}, {} tracking slots ({} unknown)".format(
        tr.teams.names[0], tr.teams.names[1], tr.teams.n_slots, int(tr.teams.unknown.sum())))
    print("tracking: {} frames, ball on {:.1f}%, loaded in {:.1f} s".format(
        tr.n, 100 * tr.ball_present.mean(), time.time() - t0))

    blocks, kickoff2 = play_blocks()
    pitch = (105.0, 68.0) if args.bench_pitch else tr.pitch
    fn = get_labeller(args.labeller)
    sig = inspect.signature(fn).parameters
    params = dict((k, _value(v)) for k, v in (x.split("=", 1) for x in args.set)) or None
    if params:
        print("parameter overrides: {}".format(params))
    extra = {}
    if "tags" in sig:                                   # labellers that can use Smart Tagging
        extra["tags"] = tag_input(fn, tr)
    if "ball_source" in sig and not args.no_ball_file:
        if os.path.exists(args.ball_file):
            extra["ball_source"] = load_ball_file(args.ball_file)
            bs = extra["ball_source"]
            print("ball source file: {} ({} up to {} frames, gate {} m/s)".format(
                os.path.relpath(args.ball_file, ROOT), bs["method"], bs["max_gap"],
                bs["max_speed_ms"]))
        else:
            print("no {}: the ball is prepared in memory (scripts/reconstruct_ball.py writes it)"
                  .format(os.path.relpath(args.ball_file, ROOT)))
    labels = np.asarray(fn(tr.px, tr.py, tr.ball, tr.teams.team, blocks, params, fps=tr.fps,
                           pitch=pitch, **extra))
    for half, a, _ in graded_blocks():
        if half == 2:
            labels[a:kickoff2] = "D"

    write_csv(out, labels)
    print("wrote {} ({} rows)".format(os.path.relpath(out, ROOT), len(labels)))
    frame = pm.read_label_csv(out)
    problem = pm.validate_submission(frame, N_FRAMES)
    print("validation: {}".format("OK" if problem is None else "FAILED - " + str(problem)))
    describe(labels, graded_blocks())

    if os.path.exists(PUBLIC_GT):
        gt = pm.label_codes(pm.read_label_csv(PUBLIC_GT))
        r = pm.evaluate(gt, pm.label_codes(frame))
        print("SANITY ONLY (public 5 min, never tune on it): spellF1 {:.4f}  macroF1 {:.4f}  "
              "possessions {}/{}".format(r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"],
                                         r["n_gt_spells"]))
    print("done in {:.0f} s".format(time.time() - t0))


if __name__ == "__main__":
    main()
