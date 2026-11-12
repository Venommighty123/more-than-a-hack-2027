"""Signal clean-up between the raw readers (possession.io) and the features.

Everything here is pure numpy on arrays in metres, so the same code runs on BAR-ATH and on the
bench games. Frame rate and pitch size are arguments (bench games are 105 x 68, BAR-ATH 105.3 x 68).
"""
import possession.io  # noqa: F401  (puts the repository root on sys.path)

from .ball import BallConfig, BallTrack, clean_ball  # noqa: E402,F401
from .players import PlayerConfig, PlayerState, prepare_players  # noqa: E402,F401
from .reconstruct import load_ball_file, prepare_ball, reconstruct  # noqa: E402,F401
