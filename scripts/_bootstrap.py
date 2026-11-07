"""Put the repository root on sys.path so ``python scripts/<name>.py`` works like ``python -m``."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "evaluation")):
    if p not in sys.path:
        sys.path.insert(0, p)
