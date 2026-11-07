"""Detect ball flights (start / end, ground speed, fitted height) and a per-frame in-air flag.

    python scripts/detect_flights.py                    # -> outputs/flights/ball_flights.csv, in_air.npy
    python scripts/detect_flights.py --out my.csv --frames-out my.npy

Logic: possession/features/flight.py (needs scipy). Ball height is mostly NOT observable in this
data (13 of 799 flights airborne): treat in_air as a weak hint. Outputs are derived match data.
"""
try:
    import _bootstrap  # noqa: F401  (python scripts/x.py)
except ImportError:
    from scripts import _bootstrap  # noqa: F401  (python -m scripts.x)

from possession.features.flight import main

if __name__ == "__main__":
    main()
