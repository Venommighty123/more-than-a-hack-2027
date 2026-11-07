"""DFL / IDSSE open Bundesliga matches -- not used for now.

The game plan proposed these as the honest test set. Ashwyn's bench (``possession/io/bench.py``)
does the same job with Metrica's public sample games, which are already in the right format, so
this loader is parked. If we add it later, it must return the same shape as
``bench.BenchGame`` (px, py, ball in metres, team 0/1 per slot, gt H/A/D/X, graded blocks) so
every labeller and ``scripts/bench_score.py`` work on it unchanged.
"""


def load_dfl_game(*_args, **_kwargs):
    raise NotImplementedError("DFL loader parked; use possession.io.bench (Metrica sample games)")
