"""P2: broadcast windows around labeller v6's stoppages and restarts.

Input is the v6 transitions table (outputs/dead/dead_transitions_*.csv): each ``stoppage`` row is
the first dead frame of a stretch, each ``restart`` row the first frame after it, in TACTICAL video
frames (its ``clock`` column is tactical video time, frame / 25, not the match clock).

For every transition a window is opened where a referee signal is expected:

    stoppage   [frame - stop_pre_s,    frame + stop_post_s]     whistle, then the arm signal
    restart    [frame - restart_pre_s, frame + restart_post_s]  the signal is often repeated before the restart

Each window is mapped to the broadcast through the sync notebook's output (outputs/sync/**/sync_full.json):

    tactical frame -> (half, match clock) -> broadcast frame,
    match_clock = t_T - tactical_offset(half) - extra(half),  t_B = match_clock + broadcast_offset(segment)

and described frame by frame: the sync label (LIVE_WIDE / LIVE_CLOSE / OFF_MATCH / UNKNOWN), the
broadcast homography ``valid`` flag, and an ATD check every ``atd_step`` frames: the median
distance from each broadcast-tracked player to the nearest tactical-tracked player at the mapped
frame. Aligned live frames sit near 0.3 m; replays (an earlier moment) are far off. The sync
labelled no frame REPLAY, which suggests this broadcast keeps the scoreboard on during replays,
so this check is the only replay detector for wide shots. Close-ups have no ATD, and stay unverified.

``usable`` frames (what P3 scans) are mapped, labelled live and not ATD-flagged as a replay. Each
window gets a ``view`` verdict:

    good            usable share >= good_share
    partial         some usable frames (>= min_usable_s)
    replay_suspect  most ATD-checked frames disagree with the tactical moment
    none            nothing usable: broadcast cut away, off the match or unmapped

Pure functions on dicts / numpy; scripts/build_broadcast_windows.py does the I/O.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

LABELS = {0: "OFF_MATCH", 1: "LIVE_WIDE", 2: "LIVE_CLOSE", 3: "REPLAY", 4: "UNKNOWN"}
LIVE_CODES = (1, 2)
VIEWS = ("good", "partial", "replay_suspect", "none")

COLUMNS = (
    "window_id", "d", "n", "kind", "direction", "half",
    "anchor_tactical", "tactical_first", "tactical_last", "anchor_video_time", "anchor_match_clock",
    "broadcast_anchor", "broadcast_first", "broadcast_last", "frames",
    "share_live_wide", "share_live_close", "share_off_match", "share_unknown", "share_unmapped",
    "share_hom_valid", "atd_checked", "atd_live_share", "atd_replay_share",
    "usable_frames", "usable_share", "view",
)


@dataclass
class WindowConfig:
    stop_pre_s: float = 2.0
    stop_post_s: float = 6.0
    restart_pre_s: float = 6.0
    restart_post_s: float = 1.0
    atd_step: int = 5              # ATD check every N tactical frames
    atd_min_players: int = 6
    atd_live_m: float = 1.0        # aligned live frames: ~0.3 m (sync refinement)
    atd_replay_m: float = 2.5      # a 2 s shift already gives ~1.7-2.0 m
    min_usable_s: float = 1.0
    good_share: float = 0.6
    replay_share: float = 0.5

    @classmethod
    def from_dict(cls, d: dict | None) -> "WindowConfig":
        d = d or {}
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict:
        return asdict(self)


def mmss(seconds) -> str:
    if seconds is None or not np.isfinite(seconds):
        return ""
    sign = "-" if seconds < 0 else ""
    m, s = divmod(abs(float(seconds)), 60.0)
    return "{}{:d}:{:04.1f}".format(sign, int(m), s)


class PairMap:
    """Tactical frame -> (half, match clock, broadcast frame) from the sync JSON (segments only)."""

    def __init__(self, sync: dict):
        src = sync["sources"]
        self.fps_t = float(src["tactical"]["fps"])
        self.fps_b = float(src["broadcast"]["fps"])
        self.n_b = int(src["broadcast"]["n_frames"])
        self.t_segs = src["tactical"]["segments"]
        self.b_segs = src["broadcast"]["segments"]
        self.extra = {int(h): float(v) for h, v in sync["pair"].get("extra_sec", {}).items() if v is not None}
        self.orientation = {int(h): v for h, v in sync["pair"].get("atd_orientation", {}).items()}

    @staticmethod
    def _bounds(segs):
        return np.array([(a["t_last"] + b["t_first"]) / 2.0 for a, b in zip(segs, segs[1:])])

    def tactical(self, frames):
        """(half, match_clock_sec) for tactical frames (arrays)."""
        t = np.asarray(frames, dtype=np.float64) / self.fps_t
        k = np.searchsorted(self._bounds(self.t_segs), t, side="right")
        half = np.array([self.t_segs[i]["half"] for i in k], dtype=int)
        off = np.array([self.t_segs[i]["offset"] for i in k])
        extra = np.array([self.extra.get(int(h), 0.0) for h in half])
        return half, t - off - extra

    def broadcast(self, half, clock):
        """Broadcast frame (float, NaN if no broadcast segment of that half covers it)."""
        half = np.atleast_1d(half); clock = np.atleast_1d(np.asarray(clock, dtype=np.float64))
        bounds = self._bounds(self.b_segs)
        out = np.full(clock.shape, np.nan)
        for k, seg in enumerate(self.b_segs):
            sel = half == seg["half"]
            if not sel.any():
                continue
            tb = clock[sel] + seg["offset"]
            ok = np.searchsorted(bounds, tb, side="right") == k
            tmp = out[sel]; tmp[ok] = tb[ok] * self.fps_b; out[sel] = tmp
        return out

    def map_frames(self, frames):
        """(half, clock, broadcast_frame int, -1 where unmapped / outside the video)."""
        half, clock = self.tactical(frames)
        bf = self.broadcast(half, clock)
        bi = np.where(np.isfinite(bf), np.round(np.nan_to_num(bf, nan=-1)), -1).astype(np.int64)
        bi[(bi < 0) | (bi >= self.n_b)] = -1
        return half, clock, bi


def build_windows(transitions, cfg: WindowConfig, fps: float = 25.0):
    """One window per stoppage / restart row of the v6 transitions table."""
    out = []
    for r in transitions:
        kind = r["kind"]
        f = int(r["frame"])
        if kind == "stoppage":
            a, b = f - int(round(cfg.stop_pre_s * fps)), f + int(round(cfg.stop_post_s * fps))
        elif kind == "restart":
            a, b = f - int(round(cfg.restart_pre_s * fps)), f + int(round(cfg.restart_post_s * fps))
        else:
            continue
        out.append({"d": int(r["d"]), "n": int(r["n"]), "kind": kind, "direction": r.get("direction", ""),
                    "anchor_tactical": f, "tactical_first": max(0, a), "tactical_last": b})
    for i, w in enumerate(out, start=1):
        w["window_id"] = i
    return out


def chamfer_m(pb, pt, orientation="identity", pitch=(105.3, 68.0), min_players=6):
    """Median distance (m) from each broadcast-tracked player to the nearest tactical-tracked one."""
    if pb is None or pt is None or len(pb) < min_players or len(pt) < min_players:
        return np.nan
    a = np.asarray(pb, dtype=np.float64).copy()
    if orientation in ("flip_x", "rot180"):
        a[:, 0] = 1.0 - a[:, 0]
    if orientation in ("flip_y", "rot180"):
        a[:, 1] = 1.0 - a[:, 1]
    a *= pitch
    b = np.asarray(pt, dtype=np.float64) * pitch
    return float(np.median(np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)).min(1)))


def describe_window(w, pm: PairMap, labels, hom_valid, atd_b, atd_t, cfg: WindowConfig, pitch=(105.3, 68.0)):
    """Fill window ``w`` (in place) with its broadcast mapping, shares and view verdict.

    labels     int codes per broadcast frame (sync JSON broadcast_frames.label), index = frame
    hom_valid  bool per broadcast frame or None
    atd_b/t    {frame: (k, 2) normalised players} or None
    Returns the usable broadcast frames (int array) for the P3 scan list.
    """
    tf = np.arange(w["tactical_first"], w["tactical_last"] + 1)
    half, clock, bf = pm.map_frames(tf)
    ok = bf >= 0
    n = len(tf)
    lab = np.full(n, -1)
    lab[ok] = labels[bf[ok]]
    hv = np.zeros(n, dtype=bool)
    if hom_valid is not None:
        hv[ok] = hom_valid[bf[ok]]
    # ATD check every atd_step frames
    verdict = np.zeros(n, dtype=np.int8)        # 0 not checked, 1 live, 2 replay, 3 in between
    if atd_b is not None and atd_t is not None:
        for i in range(0, n, max(1, cfg.atd_step)):
            if not ok[i]:
                continue
            d = chamfer_m(atd_b.get(int(bf[i])), atd_t.get(int(tf[i])), pm.orientation.get(int(half[i]), "identity"),
                          pitch, cfg.atd_min_players)
            if np.isfinite(d):
                verdict[i] = 1 if d < cfg.atd_live_m else (2 if d > cfg.atd_replay_m else 3)
        # spread each check to its neighbours (the step) so shares are per frame
        idx = np.flatnonzero(verdict)
        if len(idx):
            j = np.clip(np.searchsorted(idx, np.arange(n)), 0, len(idx) - 1)
            jl = np.clip(j - 1, 0, len(idx) - 1)
            near = np.where(np.abs(idx[jl] - np.arange(n)) <= np.abs(idx[j] - np.arange(n)), idx[jl], idx[j])
            spread = np.where(np.abs(near - np.arange(n)) <= cfg.atd_step // 2, verdict[near], 0)
            verdict = spread.astype(np.int8)
    checked = verdict > 0
    usable = ok & np.isin(lab, LIVE_CODES) & (verdict != 2)
    ai = len(tf) // 2 if w["anchor_tactical"] < tf[0] else int(w["anchor_tactical"] - tf[0])
    ai = min(max(ai, 0), n - 1)
    w.update({
        "half": int(half[ai]),
        "anchor_video_time": mmss(w["anchor_tactical"] / pm.fps_t),
        "anchor_match_clock": mmss(clock[ai]),
        "broadcast_anchor": int(bf[ai]),
        "broadcast_first": int(bf[ok].min()) if ok.any() else -1,
        "broadcast_last": int(bf[ok].max()) if ok.any() else -1,
        "frames": n,
        "share_live_wide": round(float((lab == 1).mean()), 3),
        "share_live_close": round(float((lab == 2).mean()), 3),
        "share_off_match": round(float((lab == 0).mean()), 3),
        "share_unknown": round(float(np.isin(lab, (3, 4)).mean()), 3),
        "share_unmapped": round(float((~ok).mean()), 3),
        "share_hom_valid": round(float(hv.mean()), 3),
        "atd_checked": int(checked.sum()),
        "atd_live_share": round(float((verdict[checked] == 1).mean()), 3) if checked.any() else "",
        "atd_replay_share": round(float((verdict[checked] == 2).mean()), 3) if checked.any() else "",
        "usable_frames": int(usable.sum()),
        "usable_share": round(float(usable.mean()), 3),
    })
    rep = (verdict[checked] == 2).mean() if checked.sum() >= 3 * cfg.atd_step else 0.0
    if rep >= cfg.replay_share:
        view = "replay_suspect"
    elif usable.mean() >= cfg.good_share:
        view = "good"
    elif usable.sum() >= cfg.min_usable_s * pm.fps_t:
        view = "partial"
    else:
        view = "none"
    w["view"] = view
    return np.unique(bf[usable])


def merge_ranges(frames, gap: int = 1):
    """Sorted int frames -> [(first, last)] runs, joining runs at most ``gap`` apart."""
    f = np.unique(np.asarray(frames, dtype=np.int64))
    if not len(f):
        return []
    cut = np.flatnonzero(np.diff(f) > gap) + 1
    return [(int(p[0]), int(p[-1])) for p in np.split(f, cut)]


def summary(windows, stretches_total: int) -> dict:
    """Aggregates only: windows by kind x view, stretches with a usable view (the M1 go/no-go)."""
    out = {"windows": len(windows), "by_kind_view": {}}
    for w in windows:
        k = out["by_kind_view"].setdefault(w["kind"], {v: 0 for v in VIEWS})
        k[w["view"]] += 1
    per_d = {}
    for w in windows:
        per_d.setdefault(w["d"], set()).add(w["view"])
    good = sum(1 for v in per_d.values() if "good" in v)
    any_use = sum(1 for v in per_d.values() if v & {"good", "partial"})
    out["stretches"] = stretches_total
    out["stretches_with_good_window"] = good
    out["stretches_with_usable_window"] = any_use
    out["stretches_without_view"] = stretches_total - any_use
    out["usable_frames"] = int(sum(w["usable_frames"] for w in windows))
    return out
