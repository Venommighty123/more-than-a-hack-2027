"""A per-frame classifier P(H / A / D | features), trained on the bench games only (game plan
idea 11). Its probabilities become HMM emissions (model/hmm.py), so the classifier says what each
frame looks like and the HMM decides where possessions start and end.

Dependency-light on purpose (the repository avoids sklearn): a multinomial logistic regression in
numpy -- standardised inputs, L2 penalty, mini-batch Adam, fixed seed, so training is reproducible.
If scikit-learn is installed, ``kind="hgb"`` uses HistGradientBoostingClassifier instead.

Features come from features/build.py (BENCH_COLUMNS only: they exist on the bench and on BAR-ATH)
plus centred moving averages over ``context_s`` windows, because one frame alone cannot tell a pass
in flight from a loose ball. RULES.md: train and tune on other games only, never on BAR-ATH or the
public 5 minutes; BAR-ATH is only ever passed to ``predict_proba``.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass

import numpy as np

from possession.features.build import BENCH_COLUMNS, to_matrix


@dataclass
class ClassifierConfig:
    kind: str = "logreg"           # "logreg" (numpy) | "hgb" (scikit-learn, if installed)
    context_s: tuple = (1.0, 3.0)  # centred moving-average windows added as extra features
    l2: float = 1e-3
    lr: float = 0.05
    epochs: int = 8
    batch: int = 4096
    max_rows: int = 300000         # training frames sampled across games
    z_clip: float = 5.0            # standardised inputs clipped to +/- this (robust to shift)
    seed: int = 0


def add_context(X, fps=25, windows_s=(1.0, 3.0)):
    """X plus a centred moving average of every column for each window."""
    out = [X]
    for w_s in windows_s:
        w = max(1, int(round(w_s * fps)))
        c = np.cumsum(np.pad(X, ((w // 2 + 1, w - w // 2 - 1), (0, 0)), mode="edge"), axis=0)
        out.append(((c[w:] - c[:-w]) / w).astype(np.float32))
    return np.concatenate(out, axis=1)


def design(table, cfg: ClassifierConfig, fps=25, names=BENCH_COLUMNS):
    return add_context(to_matrix(table, names), fps, cfg.context_s)


class Classifier:
    def __init__(self, cfg: ClassifierConfig | None = None):
        self.cfg = cfg or ClassifierConfig()
        self.mean = self.std = self.W = self.b = None
        self.model = None
        self.prior = None
        self.n_features = None

    # -- training ------------------------------------------------------------------------------
    def fit(self, X, y):
        cfg = self.cfg
        rng = np.random.default_rng(cfg.seed)
        y = np.asarray(y, dtype=np.int64)
        keep = (y >= 0) & (y <= 2)
        X, y = X[keep], y[keep]
        if len(y) > cfg.max_rows:
            idx = rng.choice(len(y), cfg.max_rows, replace=False)
            X, y = X[idx], y[idx]
        self.n_features = X.shape[1]
        self.prior = np.bincount(y, minlength=3) / float(len(y))
        if cfg.kind == "hgb":
            from sklearn.ensemble import HistGradientBoostingClassifier
            self.model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1,
                                                        random_state=cfg.seed).fit(X, y)
            return self
        self.mean = X.mean(0)
        self.std = X.std(0) + 1e-6
        Z = np.clip((X - self.mean) / self.std, -cfg.z_clip, cfg.z_clip)
        k = Z.shape[1]
        W, b = np.zeros((k, 3)), np.zeros(3)
        mW, vW, mb, vb = np.zeros_like(W), np.zeros_like(W), np.zeros(3), np.zeros(3)
        Y = np.eye(3)[y]
        step = 0
        for _ in range(cfg.epochs):
            order = rng.permutation(len(y))
            for s in range(0, len(y), cfg.batch):
                j = order[s:s + cfg.batch]
                p = self._softmax(Z[j] @ W + b)
                g = p - Y[j]
                gW = Z[j].T @ g / len(j) + cfg.l2 * W
                gb = g.mean(0)
                step += 1
                for prm, grd, m, v in ((W, gW, mW, vW), (b, gb, mb, vb)):
                    m *= 0.9
                    m += 0.1 * grd
                    v *= 0.999
                    v += 0.001 * grd * grd
                    prm -= cfg.lr * (m / (1 - 0.9 ** step)) / (np.sqrt(v / (1 - 0.999 ** step)) + 1e-8)
        self.W, self.b = W, b
        return self

    @staticmethod
    def _softmax(z):
        z = z - z.max(1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(1, keepdims=True)

    # -- inference -----------------------------------------------------------------------------
    def predict_proba(self, X) -> np.ndarray:
        if self.model is not None:
            return self.model.predict_proba(X)
        z = np.clip((X - self.mean) / self.std, -self.cfg.z_clip, self.cfg.z_clip)
        return self._softmax(z @ self.W + self.b)

    # -- persistence (logreg only) ----------------------------------------------------------------
    def save(self, path):
        if self.model is not None:
            raise ValueError("save() supports kind='logreg' only")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.savez_compressed(path, mean=self.mean, std=self.std, W=self.W, b=self.b,
                            prior=self.prior, cfg=json.dumps(asdict(self.cfg)))

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        d = json.loads(str(z["cfg"]))
        d["context_s"] = tuple(d["context_s"])
        c = cls(ClassifierConfig(**d))
        c.mean, c.std, c.W, c.b, c.prior = z["mean"], z["std"], z["W"], z["b"], z["prior"]
        c.n_features = len(c.mean)
        return c


# ------------------------------------------------------------------------------------------------
# Training on the bench
# ------------------------------------------------------------------------------------------------

def bench_tables(bench_dir, variant="degraded", games=("g1", "g2", "g3"), cache_dir=None):
    """[(name, feature table, gt codes)] for bench games; tables cached under outputs/features."""
    from possession.features.build import build_features, load_table, save_table
    from possession.io.bench import load_bench_game
    from possession.utils.frames import to_codes
    cache_dir = cache_dir or os.path.join(os.path.dirname(bench_dir.rstrip("/\\")), "features")
    out = []
    for name in games:
        g = load_bench_game(bench_dir, name, variant)
        path = os.path.join(cache_dir, "bench_{}_{}.npz".format(name, variant))
        if os.path.isfile(path) and os.path.getmtime(path) >= os.path.getmtime(
                os.path.join(bench_dir, "{}_{}.npz".format(name, variant))):
            table = load_table(path)
        else:
            table = build_features(g.px, g.py, g.ball, g.team, g.blocks, fps=g.fps, pitch=g.pitch)
            save_table(table, path)
        out.append((name, table, to_codes(g.gt)))
    return out


def train_on_bench(bench_dir, variants=("degraded",), games=("g1", "g2", "g3"),
                   cfg: ClassifierConfig | None = None, exclude=None):
    """Fit on the graded frames of the bench games (minus ``exclude``, for leave-one-game-out)."""
    cfg = cfg or ClassifierConfig()
    Xs, ys = [], []
    for variant in variants:
        for name, table, gt in bench_tables(bench_dir, variant, games):
            if name == exclude:
                continue
            X = design(table, cfg)
            graded = gt <= 2
            Xs.append(X[graded])
            ys.append(gt[graded])
    return Classifier(cfg).fit(np.concatenate(Xs), np.concatenate(ys))
