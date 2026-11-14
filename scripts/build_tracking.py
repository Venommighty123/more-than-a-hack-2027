"""Build the pipeline's one input once: our YOLO players (packed + cleaned) and the fixed ball.

    python scripts/build_tracking.py                         # -> outputs/cache/yolo_tracking.npz
    python scripts/build_tracking.py --rebuild               # ignore the cached slots, pack again
    python scripts/build_tracking.py --team-mode track --raw-players --out outputs/cache/yolo_raw.npz

Every other script calls possession.io.load(), which builds the same thing on demand and caches the
packed players in outputs/cache/detected_*.npz. Run this script after a new Kaggle run (new
final_tracks_*.parquet) or a new fixed ball, to rebuild in one place and to see what changed:

  players  data/kaggle_outputs/final_tracks_full.parquet (kaggle/kaggle_player_tracking.ipynb),
           packed into slots and cleaned by possession/io/detections.py
  ball     outputs/ball/ball_reconstructed.npz (scripts/reconstruct_ball.py)

OUTPUT (git-ignored, derived match data: never commit or upload it, RULES.md)
  <out>.npz           px, py (frames, slots) metres, bx, by (frames,), team (slots,), names, stats
  <out>_summary.json  what was built: slots, players per frame, team overflow, clean-up counts, ball
"""
import argparse
import json
import os
import time

try:
    import _bootstrap  # noqa: F401  (python scripts/x.py)
except ImportError:
    from scripts import _bootstrap  # noqa: F401  (python -m scripts.x)

import numpy as np

from possession.io import graded_mask, load
from possession.io.detections import DEFAULT_FILE as PLAYERS_FILE
from possession.preprocess.reconstruct import DEFAULT_FILE as BALL_FILE

ROOT = _bootstrap.ROOT


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--players-file", default=os.path.join(ROOT, PLAYERS_FILE))
    ap.add_argument("--ball-file", default=os.path.join(ROOT, BALL_FILE))
    ap.add_argument("--team-mode", choices=("local", "track"), default="local")
    ap.add_argument("--raw-players", action="store_true", help="skip the clean-up (jump cuts, duplicates)")
    ap.add_argument("--rebuild", action="store_true", help="pack the players again even if cached")
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "cache", "yolo_tracking.npz"))
    args = ap.parse_args()

    t0 = time.time()
    tr = load(players_file=args.players_file, team_mode=args.team_mode, clean=not args.raw_players,
              ball_file=args.ball_file, refresh=args.rebuild)
    graded = graded_mask(tr.n, tr.lo)
    st = dict(tr.stats)
    st.update(frames=int(tr.n), graded_frames=int(graded.sum()),
              ball_on_graded_pct=round(100 * float(np.isfinite(tr.bx[graded]).mean()), 2),
              names=list(tr.teams.names), built_in_s=round(time.time() - t0, 1))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, px=tr.px, py=tr.py, bx=tr.bx, by=tr.by, team=tr.teams.team,
                        names=np.array(tr.teams.names), lo=np.array(tr.lo), fps=np.array(tr.fps),
                        pitch=np.array(tr.pitch), stats=np.array(json.dumps(st, default=str)))
    summary = os.path.splitext(args.out)[0] + "_summary.json"
    with open(summary, "w") as handle:
        json.dump(st, handle, indent=2, default=str)

    print("players : {}  ({} slots: H {}, A {}; team mode {}; clean {})".format(
        st["players"], st["slots"], st["slots_home"], st["slots_away"], st["team_mode"], st.get("clean")))
    print("          {:.1f} players/frame; a team over 11 on {:.1f}% of frames{}".format(
        st["players_per_frame"], st.get("overflow_pct_after_clean", st["overflow_pct"]),
        "" if not st.get("clean") else " (before clean-up {:.1f}%)".format(st["overflow_pct"])))
    if st.get("clean"):
        print("          clean-up: {} jumps cut, {} duplicate runs removed, {} frames blanked".format(
            st["clean_jumps_cut"], st["clean_duplicate_runs"], st["clean_frames_blanked"]))
    print("ball    : {}  (present on {:.1f}% of graded frames)".format(st["ball"], st["ball_on_graded_pct"]))
    print("wrote {}\n      {}  ({:.0f} s)".format(os.path.relpath(args.out, ROOT),
                                               os.path.relpath(summary, ROOT), time.time() - t0))


if __name__ == "__main__":
    main()
