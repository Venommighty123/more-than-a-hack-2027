"""DEPRECATED -- misspelt, and the DFL test set was replaced by the Metrica bench. Kept only so old commands keep working; delete this file.

    python scripts/bench_score.py
"""
import sys

try:
    import _bootstrap  # noqa: F401  (python scripts/x.py)
except ImportError:
    from scripts import _bootstrap  # noqa: F401  (python -m scripts.x)

if __name__ == "__main__":
    print("note: scripts/evauate_dfl.py is deprecated, running scripts/bench_score.py")
    try:
        from scripts.bench_score import main
    except ImportError:
        from bench_score import main
    main()
