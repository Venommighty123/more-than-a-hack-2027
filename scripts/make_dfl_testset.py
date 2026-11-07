"""DEPRECATED -- the DFL test set was replaced by the Metrica bench. Kept only so old commands keep working; delete this file.

    python scripts/bench_build.py
"""
import sys

try:
    import _bootstrap  # noqa: F401  (python scripts/x.py)
except ImportError:
    from scripts import _bootstrap  # noqa: F401  (python -m scripts.x)

if __name__ == "__main__":
    print("note: the DFL test set is parked (possession/io/dfl.py); running scripts/bench_build.py")
    try:
        from scripts.bench_build import main
    except ImportError:
        from bench_build import main
    main()
