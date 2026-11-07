"""Detect every pass (passer, receiver, outcome) on top of the ball flights.

    python scripts/detect_passes.py                                   # whole match -> outputs/passes/
    python scripts/detect_passes.py --start 11185 --seconds 120       # a window
    python scripts/detect_passes.py --height                          # add the ball-height fit (slower)

Logic and every option: possession/events/passes.py. Thresholds still need re-deriving on other
games (docs/compliance.md). Output is derived from match data: outputs/ stays git-ignored.
"""
try:
    import _bootstrap  # noqa: F401  (python scripts/x.py)
except ImportError:
    from scripts import _bootstrap  # noqa: F401  (python -m scripts.x)

from possession.events.passes import main

if __name__ == "__main__":
    main()
