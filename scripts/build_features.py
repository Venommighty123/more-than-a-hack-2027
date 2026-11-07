"""Write the per-frame feature table (no labels) for BAR-ATH or a bench game.

    python scripts/build_features.py                       # BAR-ATH -> outputs/features/BAR-ATH.npz
    python scripts/build_features.py --bench g1 --variant degraded
    python scripts/build_features.py --no-camera --no-tags  # BAR-ATH with bench columns only

Column meanings: docs/features.md. The table is derived from match data: outputs/ stays git-ignored.
"""
import argparse
import os
import time

try:
    import _bootstrap  # noqa: F401  (python scripts/x.py)
except ImportError:
    from scripts import _bootstrap  # noqa: F401  (python -m scripts.x)

import numpy as np

from possession.features.build import build_features, save_table

ROOT = _bootstrap.ROOT
OUT = os.path.join(ROOT, "outputs", "features")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bench", default=None, help="bench game (g1/g2/g3) instead of BAR-ATH")
    ap.add_argument("--variant", default="degraded")
    ap.add_argument("--bench-dir", default=os.path.join(ROOT, "outputs", "bench"))
    ap.add_argument("--no-tags", action="store_true")
    ap.add_argument("--no-camera", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    t0 = time.time()
    if args.bench:
        from possession.io.bench import load_bench_game
        g = load_bench_game(args.bench_dir, args.bench, args.variant)
        table = build_features(g.px, g.py, g.ball, g.team, g.blocks, fps=g.fps, pitch=g.pitch)
        out = args.out or os.path.join(OUT, "bench_{}_{}.npz".format(args.bench, args.variant))
    else:
        from games import GAMES, GAME, anchor_video_frame
        from possession.io import load, load_tags
        try:
            from scripts.run import play_blocks
        except ImportError:
            from run import play_blocks
        tr = load(cache=True)
        blocks, kickoff2 = play_blocks()
        tags = None if args.no_tags else load_tags(tr.teams)
        camera = None
        if not args.no_camera:
            from possession.features.camera import camera_features
            camera = camera_features()
        table = build_features(tr.px, tr.py, tr.ball, tr.teams.team, blocks, fps=tr.fps,
                               pitch=tr.pitch, tags=tags, camera=camera,
                               kickoffs={1: blocks[0][1], 2: kickoff2})
        out = args.out or os.path.join(OUT, "{}.npz".format(GAME))
    save_table(table, out)
    cols = table["columns"]
    print("{} columns x {} rows -> {} ({:.0f} s)".format(len(cols), table["meta"]["n"],
                                                         os.path.relpath(out, ROOT), time.time() - t0))
    nan = {k: float(np.isnan(v).mean()) for k, v in cols.items()}
    print("most-missing columns:", sorted(nan.items(), key=lambda kv: -kv[1])[:5])


if __name__ == "__main__":
    main()
