"""Readers for everything the pipeline takes in: tracking, slot teams, Smart Tagging, bench games.

Importing this package puts the repository root on ``sys.path`` so that the organisers' modules
(``games``, ``starter.*``) import the same way from a script, a notebook or ``python -m``.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from .teams import HOME, AWAY, UNKNOWN, LETTER, TeamMap, load_teams, team_of_each_track  # noqa: E402,F401
from .tracking import Tracking, load, load_ball, graded_blocks, graded_mask  # noqa: E402,F401
from .tags import Tag, load_tags  # noqa: E402,F401
