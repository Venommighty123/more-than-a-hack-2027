"""DEPRECATED -- use scripts/build_features.py. Kept only so old commands keep working; delete this file.

    python scripts/build_features.py
"""
import sys

try:
    import _bootstrap  # noqa: F401  (python scripts/x.py)
except ImportError:
    from scripts import _bootstrap  # noqa: F401  (python -m scripts.x)

if __name__ == "__main__":
    print("note: scripts/detect_features.py is deprecated, running scripts/build_features.py")
    try:
        from scripts.build_features import main
    except ImportError:
        from build_features import main
    main()
