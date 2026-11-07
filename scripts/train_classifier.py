"""Train the per-frame classifier on the bench games and save it for labeller v3c.

    python scripts/train_classifier.py                       # degraded copies -> outputs/models/classifier.npz
    python scripts/train_classifier.py --variants clean degraded hard

Bench data only (other games, allowed by RULES.md). Never train or tune on BAR-ATH.
"""
import argparse
import os

try:
    import _bootstrap  # noqa: F401  (python scripts/x.py)
except ImportError:
    from scripts import _bootstrap  # noqa: F401  (python -m scripts.x)

from possession.model.classifier import ClassifierConfig, train_on_bench

ROOT = _bootstrap.ROOT


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bench", default=os.path.join(ROOT, "outputs", "bench"))
    ap.add_argument("--variants", nargs="+", default=["degraded"])
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "models", "classifier.npz"))
    ap.add_argument("--epochs", type=int, default=ClassifierConfig.epochs)
    args = ap.parse_args()
    clf = train_on_bench(args.bench, tuple(args.variants), cfg=ClassifierConfig(epochs=args.epochs))
    clf.save(args.out)
    print("saved {} ({} features, class prior H/A/D {})".format(
        os.path.relpath(args.out, ROOT), clf.n_features, [round(float(p), 3) for p in clf.prior]))


if __name__ == "__main__":
    main()
