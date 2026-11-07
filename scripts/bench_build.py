"""Build the Metrica-sample bench (see possession/io/bench.py for what it contains and why).

    python scripts/bench_build.py                                   # data/external/metrica_sample -> outputs/bench
    python scripts/bench_build.py --src <folder with g1/ g2/ g3/> --out outputs/bench --no-degrade

The source folder holds Metrica's public sample games (github.com/metrica-sports/sample-data):
  g1/Sample_Game_1_RawEventsData.csv, ..._RawTrackingData_Home_Team.csv, ..._Away_Team.csv
  g2/ the same for game 2
  g3/Sample_Game_3_metadata.xml, Sample_Game_3_tracking.txt, Sample_Game_3_events.json (needs kloppy)
Other games, public data: tuning on them is allowed (RULES.md).
"""
import argparse
import os

try:
    import _bootstrap  # noqa: F401  (python scripts/x.py)
except ImportError:
    from scripts import _bootstrap  # noqa: F401  (python -m scripts.x)

from possession.io.bench import GAMES, SEED, build_bench
from possession.utils.config import load_config, section

ROOT = _bootstrap.ROOT
CFG = section(load_config(os.path.join(ROOT, "config", "bench.yaml")), "bench")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", default=os.path.join(ROOT, CFG.get("src", "data/external/metrica_sample")))
    ap.add_argument("--out", default=os.path.join(ROOT, CFG.get("dir", "outputs/bench")))
    ap.add_argument("--no-degrade", action="store_true", help="write only the clean copies")
    ap.add_argument("--seed", type=int, default=int(CFG.get("seed", SEED)))
    ap.add_argument("--games", nargs="+", default=list(CFG.get("games", GAMES)),
                    help="subset, e.g. --games g3")
    ap.add_argument("--kloppy", action="store_true", help="read game 3 with kloppy (needs RAM)")
    args = ap.parse_args()
    build_bench(args.src, args.out, with_degraded=not args.no_degrade, seed=args.seed,
                games=args.games, use_kloppy=args.kloppy)
    print("bench written to {}".format(os.path.relpath(args.out, ROOT)))


if __name__ == "__main__":
    main()
