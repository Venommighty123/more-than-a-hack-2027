"""Assemble the per-frame feature table: one row per frame, one named column per feature.

The same function runs on BAR-ATH (with Smart Tagging and camera columns) and on a bench game
(without them), so a model trained on the bench sees the same columns on BAR-ATH. The table holds
NO labels. docs/features.md is the dictionary of every column.

    from possession.features.build import build_features, save_table, BENCH_COLUMNS
    table = build_features(tr.px, tr.py, tr.ball, tr.teams.team, blocks, pitch=tr.pitch,
                           tags=tags, camera=camera_features())
    save_table(table, "outputs/features/BAR-ATH.npz")

Derived from match data: outputs/ is git-ignored, never commit or upload the table (RULES.md).
"""
from __future__ import annotations

import json
import os

import numpy as np

from possession.detectors.control import ControlConfig, detect_control
from possession.detectors.tag_prior import build_prior
from possession.detectors.throw_in_side import fill_tag_teams, infer_throw_ins
from possession.features.pitch_control import pitch_control_at_ball
from possession.features.relational import relational
from possession.features.shape import team_shape
from possession.features.time_to_ball import time_to_ball
from possession.gaps.classify import CLASSES, classify_gaps
from possession.gaps.find import find_gaps
from possession.labellers import v1
from possession.preprocess.ball import distance_to_lines
from possession.preprocess.players import attack_signs, prepare_players

GAP_CLASS = {c: i + 1 for i, c in enumerate(CLASSES)}   # 0 = not in a classified gap

# Columns that exist on the bench too (no Smart Tagging, no camera): what a model may be trained on.
BENCH_COLUMNS = (
    "ball_observed", "ball_present", "ball_speed", "ball_touch", "ball_outside", "ball_line_m",
    "ctrl_dist", "ctrl_zone", "dead_rule", "in_gap", "gap_len_s", "gap_pos", "gap_class",
    "mean_speed", "speed_home", "speed_away",
    "d1_home", "d2_home", "closing_home", "relspeed_home", "n3_home", "n5_home", "n10_home",
    "d1_away", "d2_away", "closing_away", "relspeed_away", "n3_away", "n5_away", "n10_away",
    "gap_d1", "ttb_home", "ttb_away", "ttb_diff", "pc_home",
    "cx_att_home", "spread_home", "length_home", "width_home", "def_line_home", "advance_home",
    "cx_att_away", "spread_away", "length_away", "width_away", "def_line_away", "advance_away",
    "advance_diff", "spread_diff", "cx_att_diff",
)
# n_tracked_* are NOT model inputs: on the bench every team is always 11 tracked players, so the
# column is constant there and explodes after standardisation on BAR-ATH (~10.2 tracked).
CAPS = {"gap_len_s": 30.0}   # bench gaps (synthetic) top out near 30 s; BAR-ATH stoppages are longer


def _gap_columns(present, blocks, track, ctrl, mean_speed, tags, kickoffs, fps, pitch):
    n = len(present)
    in_gap = np.zeros(n, dtype=np.float32)
    gap_len = np.zeros(n, dtype=np.float32)
    gap_pos = np.full(n, np.nan, dtype=np.float32)
    gap_cls = np.zeros(n, dtype=np.float32)
    gaps = classify_gaps(find_gaps(present, blocks), track.x, track.y, ctrl, mean_speed,
                         tags=tags, kickoffs=kickoffs, fps=fps, pitch=pitch)
    for g in gaps:
        sl = slice(g.start, g.end)
        in_gap[sl] = 1.0
        gap_len[sl] = g.length / float(fps)
        gap_pos[sl] = (np.arange(g.length) + 0.5) / g.length
        gap_cls[sl] = GAP_CLASS[g.info["cls"]]
    return {"in_gap": in_gap, "gap_len_s": gap_len, "gap_pos": gap_pos, "gap_class": gap_cls}, gaps


def build_features(px, py, ball, team, blocks, *, fps=25, pitch=(105.0, 68.0), tags=None,
                   camera=None, kickoffs=None, params=None) -> dict:
    """{"columns": {name: (n,) float32}, "meta": {...}} for rows 0..n-1."""
    team = np.asarray(team)
    base = v1.run(px, py, ball, team, blocks, params, fps=fps, pitch=pitch)
    track, ctrl = base["ball"], base["control"]
    ps = prepare_players(px, py, team, blocks, 0, fps=fps)
    signs = ps.signs or attack_signs(px, team, blocks)
    cols = {
        "ball_observed": track.observed, "ball_present": track.present,
        "ball_x": track.x, "ball_y": track.y, "ball_speed": track.speed,
        "ball_touch": track.touch, "ball_outside": track.outside,
        "ball_line_m": distance_to_lines(track.x, track.y, pitch),
        "ctrl_dist": ctrl.ctrl, "dead_rule": base["dead"],
        "mean_speed": base["mean_speed"], "speed_home": ps.team_speed[0],
        "speed_away": ps.team_speed[1],
    }
    zone = detect_control(ps.x, ps.y, track.x, track.y, team, ControlConfig(method="zone"),
                          ball_v=(track.vx, track.vy), player_v=(ps.vx, ps.vy), fps=fps)
    cols["ctrl_zone"] = zone.ctrl
    gap_cols, gaps = _gap_columns(track.present, blocks, track, ctrl.ctrl, base["mean_speed"],
                                  tags, kickoffs, fps, pitch)
    cols.update(gap_cols)
    cols.update(relational(px, py, track.x, track.y, team, pvx=ps.vx, pvy=ps.vy,
                           bvx=track.vx, bvy=track.vy))
    ttb = time_to_ball(ps.x, ps.y, ps.vx, ps.vy, track.x, track.y, team)
    cols.update({k: v for k, v in ttb.items() if not k.startswith("ttb_slot")})
    cols["pc_home"] = pitch_control_at_ball(ps.x, ps.y, ps.vx, ps.vy, track.x, track.y, team)
    cols.update(team_shape(ps.x, ps.y, ps.vx, team, signs, blocks, pitch=pitch))
    if tags:
        filled = fill_tag_teams(tags, infer_throw_ins(tags, px, py, track.x, track.y, team, signs,
                                                      dead=base["dead"], pitch=pitch))
        cols["tag_score"] = build_prior(filled, len(ball)).score
    if camera:
        cols.update(camera)
    graded = np.zeros(len(ball), dtype=np.float32)
    for _, a, b in blocks:
        graded[int(a):int(b) + 1] = 1.0
    cols["graded"] = graded
    cols = {k: np.asarray(v, dtype=np.float32) for k, v in cols.items()}
    meta = {"n": len(ball), "fps": fps, "pitch": list(pitch), "blocks": [list(map(int, b)) for b in blocks],
            "signs": {"{}_{}".format(h, s): v for (h, s), v in signs.items()},
            "n_gaps": len(gaps), "columns": sorted(cols)}
    return {"columns": cols, "meta": meta}


def save_table(table, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(path, __meta__=json.dumps(table["meta"]), **table["columns"])


def load_table(path):
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["__meta__"]))
    return {"columns": {k: z[k] for k in z.files if k != "__meta__"}, "meta": meta}


def to_matrix(table, names=BENCH_COLUMNS, clip=50.0):
    """(n, len(names)) float32 for a model: inf -> clip, NaN -> 0, plus a NaN-indicator per column
    that has any NaN (so the model can tell 'missing' from 'zero')."""
    cols, extra = [], []
    for k in names:
        v = np.asarray(table["columns"][k], dtype=np.float32)
        nan = np.isnan(v)
        v = np.clip(np.where(nan, 0.0, v), -clip, CAPS.get(k, clip))
        cols.append(v)
        if nan.any():
            extra.append(nan.astype(np.float32))
    return np.stack(cols + extra, axis=1)
