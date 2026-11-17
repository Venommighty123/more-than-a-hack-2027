"""ATD loading helpers used by the labeller pipeline.

Copied VERBATIM from the old test_scripts/pass_detection.py (removed for the submission because it
held published match statistics, RULES.md rule 2). Only these functions and constants are kept; the
code is unchanged, so every run reads the ATD exactly as before.

    team_of_each_track(meta_path)  0 home / 1 away per ATD slot (first Team in the XML is home)
    team_names(meta_path)          [home name, away name]
    load_tracking(raw_path, n_slots, lo, hi)  players (frames x slots) and ball in metres
    clean_ball(ball)               drop single-frame teleports (> MAX_BALL_SPEED m/s)
    PITCH_L, PITCH_W, FPS, HOME, AWAY, MAX_BALL_SPEED
"""

import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAME, GAMES  # noqa: E402

FPS = GAMES[GAME]["fps"]
PITCH_L, PITCH_W = GAMES[GAME]["pitch"]
HOME, AWAY = 0, 1
MAX_BALL_SPEED = 40.0    # m/s   faster frame-to-frame jumps are tracking errors


def team_of_each_track(meta_path):
    """0 for home slots, 1 for away, in Player order (first Team in the XML is home)."""
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    if len(team_ids) != 2:
        raise ValueError("expected 2 teams in {}, found {}".format(meta_path, len(team_ids)))
    slots = [player.get("teamId") for player in root.iter("Player")]
    return np.array([HOME if t == team_ids[0] else AWAY for t in slots], dtype=np.int8)


def team_names(meta_path):
    root = ET.parse(meta_path).getroot()
    return [team.findtext("Name") for team in root.iter("Team")]


def load_tracking(raw_path, n_slots, lo, hi):
    """Player (frames x slots) and ball positions in metres, video frames lo..hi inclusive.

    '<frame>:<x,y,s;x,y,s;...>:<ball x,y,...>', ATD frames count from 1, video frames from 0.
    NaN where an object is not seen.
    """
    n = hi - lo + 1
    px = np.full((n, n_slots), np.nan, dtype=np.float32)
    py = np.full((n, n_slots), np.nan, dtype=np.float32)
    ball = np.full((n, 2), np.nan)
    per_slot = None
    with open(raw_path) as handle:
        for line in handle:
            head = line[:line.index(":")] if ":" in line else ""
            if not head.isdigit():
                continue
            frame = int(head) - 1
            if frame < lo:
                continue
            if frame > hi:
                break
            parts = line.strip().split(":")
            if len(parts) != 3:
                continue
            values = np.array(parts[1].replace(";", ",").split(","), dtype=np.float32)
            if per_slot is None:
                per_slot = len(values) // n_slots
                if per_slot < 2 or per_slot * n_slots != len(values):
                    raise ValueError("cannot split {} values into {} slots".format(
                        len(values), n_slots))
            values = values.reshape(n_slots, per_slot)
            k = frame - lo
            px[k] = values[:, 0] * PITCH_L
            py[k] = values[:, 1] * PITCH_W
            b = parts[2].split(",")
            if len(b) >= 2 and b[0][:1] != "N":
                ball[k] = (float(b[0]) * PITCH_L, float(b[1]) * PITCH_W)
    return px, py, ball


def clean_ball(ball):
    """Drop single-frame teleports (implausible speed) from the ball track."""
    step = np.hypot(*np.diff(ball, axis=0, prepend=np.nan).T) * FPS
    out = ball.copy()
    out[step > MAX_BALL_SPEED] = np.nan
    return out
