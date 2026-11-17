"""P2: map labeller v6's stoppages and restarts to broadcast windows and say what the broadcast shows.

    python scripts/build_broadcast_windows.py                    # -> outputs/referee/
    python scripts/build_broadcast_windows.py --set stop_post_s=8 --set atd_live_m=1.2
    python scripts/build_broadcast_windows.py --no-atd           # skip the replay check (faster)

Inputs (all already in the repo):
    outputs/dead/dead_transitions_labeller_v6_newtracks.csv   v6 stoppage / restart frames (tactical)
    outputs/dead/dead_stretches_labeller_v6_newtracks.csv     v6 dead stretches (for the counts)
    outputs/sync/**/sync_full.json                            the sync notebook's full run
    data/broadcast/BAR-ATH_broadcast_homography.csv           `valid` per broadcast frame
    data/{broadcast,tactical}/BAR-ATH_*_atd.txt               replay check (players only)

Outputs (outputs/ is git-ignored; this is derived match data):
    outputs/referee/broadcast_windows.csv          one row per stoppage / restart window (see possession/referee/windows.py)
    outputs/referee/broadcast_scan_ranges.csv      merged usable broadcast frame ranges = what P3 decodes
    outputs/referee/broadcast_windows_summary.json aggregates, config, mapping check, M1 go/no-go numbers

Only aggregates are printed.
"""
import argparse
import csv
import glob
import json
import os
import time

try:
    import _bootstrap  # noqa: F401  (python scripts/x.py)
except ImportError:
    from scripts import _bootstrap  # noqa: F401  (python -m scripts.x)

import numpy as np
import pandas as pd

from games import GAMES, GAME, asset_paths
from possession.referee.windows import (COLUMNS, VIEWS, PairMap, WindowConfig, build_windows, describe_window,
                                        merge_ranges, summary)

ROOT = _bootstrap.ROOT
OUT_DIR = os.path.join(ROOT, "outputs", "referee")
TRANSITIONS = os.path.join(ROOT, "outputs", "dead", "dead_transitions_labeller_v6_newtracks.csv")
STRETCHES = os.path.join(ROOT, "outputs", "dead", "dead_stretches_labeller_v6_newtracks.csv")
SYNC_GLOB = os.path.join(ROOT, "outputs", "sync", "**", "sync_full.json")


def _value(text):
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text


def _read_csv(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def load_players(raw_path, wanted):
    """{video_frame: (k, 2) normalised player xy} for the wanted frames, straight off the raw ATD."""
    want = set(int(f) for f in wanted)
    out = {}
    with open(raw_path, "rb") as handle:
        for line in handle:
            i = line.find(b":")
            if i <= 0:
                continue
            try:
                f = int(line[:i]) - 1
            except ValueError:
                continue
            if f not in want:
                continue
            parts = line.decode("ascii", "replace").strip().split(":")
            if len(parts) != 3:
                continue
            pts = []
            for g in parts[1].split(";"):
                if not g or g[0] == "N":
                    continue
                v = g.split(",")
                try:
                    x, y = float(v[0]), float(v[1])
                except (ValueError, IndexError):
                    continue
                if np.isfinite(x) and np.isfinite(y):
                    pts.append((x, y))
            out[f] = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--transitions", default=TRANSITIONS)
    ap.add_argument("--stretches", default=STRETCHES)
    ap.add_argument("--sync", default=None, help="sync_full.json (default: search outputs/sync)")
    ap.add_argument("--no-atd", action="store_true", help="skip the ATD replay check")
    ap.add_argument("--scan-gap", type=int, default=12, help="join usable broadcast runs this close (frames)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a WindowConfig field")
    args = ap.parse_args()
    cfg = WindowConfig.from_dict({k: _value(v) for k, v in (s.split("=", 1) for s in args.set)})
    t0 = time.time()

    sync_path = args.sync or next(iter(sorted(glob.glob(SYNC_GLOB, recursive=True))), None)
    for p in (args.transitions, args.stretches, sync_path):
        if not p or not os.path.exists(p):
            raise SystemExit("missing input: {}".format(p or "sync_full.json under outputs/sync"))
    with open(sync_path) as handle:
        sync = json.load(handle)
    pm = PairMap(sync)

    bfr = sync["broadcast_frames"]
    labels = np.full(pm.n_b, 4, dtype=np.int64)                      # UNKNOWN outside the table
    lab = np.asarray(bfr["columns"]["label"], dtype=np.int64)
    labels[bfr["start"]:bfr["start"] + len(lab)] = lab

    hom_path = asset_paths(GAME, source="broadcast")["homography"]
    hom_valid = None
    if os.path.exists(hom_path):
        h = pd.read_csv(hom_path, usecols=["frame", "valid"])
        hom_valid = np.zeros(pm.n_b, dtype=bool)
        fr = h["frame"].to_numpy(); ok = (fr >= 0) & (fr < pm.n_b)
        hom_valid[fr[ok]] = h["valid"].to_numpy()[ok].astype(bool)

    transitions = _read_csv(args.transitions)
    stretches = _read_csv(args.stretches)
    windows = build_windows(transitions, cfg, pm.fps_t)

    atd_b = atd_t = None
    if not args.no_atd:
        want_t, want_b = [], []
        for w in windows:
            tf = np.arange(w["tactical_first"], w["tactical_last"] + 1, max(1, cfg.atd_step))
            _, _, bf = pm.map_frames(tf)
            want_t.append(tf); want_b.append(bf[bf >= 0])
        atd_t = load_players(asset_paths(GAME, source="tactical")["atd_raw"], np.concatenate(want_t))
        atd_b = load_players(asset_paths(GAME, source="broadcast")["atd_raw"], np.concatenate(want_b))

    pitch = tuple(GAMES[GAME]["pitch"])
    usable = [describe_window(w, pm, labels, hom_valid, atd_b, atd_t, cfg, pitch) for w in windows]

    # mapping check against the sync notebook's own per-frame tactical -> broadcast index
    tfr = sync.get("tactical_frames")
    diffs = []
    if tfr:
        idx = np.asarray(tfr["columns"]["broadcast_frame"], dtype=np.int64)
        for w in windows:
            k = w["anchor_tactical"] - tfr["start"]
            if 0 <= k < len(idx) and idx[k] >= 0 and w["broadcast_anchor"] >= 0:
                diffs.append(abs(int(idx[k]) - w["broadcast_anchor"]))
    mapping_check = {"anchors_compared": len(diffs),
                     "max_abs_diff_frames": int(max(diffs)) if diffs else None,
                     "share_within_1_frame": round(float(np.mean(np.array(diffs) <= 1)), 3) if diffs else None}

    # scan list for P3
    ranges = merge_ranges(np.concatenate(usable) if usable else [], gap=args.scan_gap)
    scan_rows = []
    for i, (a, b) in enumerate(ranges, start=1):
        ids = [w["window_id"] for w in windows if w["broadcast_first"] <= b and w["broadcast_last"] >= a
               and w["view"] in ("good", "partial")]
        scan_rows.append({"range_id": i, "broadcast_first": a, "broadcast_last": b, "frames": b - a + 1,
                          "window_ids": "|".join(map(str, ids))})

    os.makedirs(args.out_dir, exist_ok=True)
    wpath = os.path.join(args.out_dir, "broadcast_windows.csv")
    with open(wpath, "w", newline="") as handle:
        wr = csv.DictWriter(handle, fieldnames=COLUMNS, extrasaction="ignore")
        wr.writeheader(); wr.writerows(windows)
    spath = os.path.join(args.out_dir, "broadcast_scan_ranges.csv")
    with open(spath, "w", newline="") as handle:
        wr = csv.DictWriter(handle, fieldnames=["range_id", "broadcast_first", "broadcast_last", "frames", "window_ids"])
        wr.writeheader(); wr.writerows(scan_rows)

    summ = summary(windows, stretches_total=len(stretches))
    checked = [w for w in windows if w["atd_checked"]]
    summ.update({
        "config": cfg.to_dict(), "scan_gap": args.scan_gap, "atd_check": not args.no_atd,
        "inputs": {"transitions": os.path.relpath(args.transitions, ROOT), "stretches": os.path.relpath(args.stretches, ROOT),
                   "sync": os.path.relpath(sync_path, ROOT), "sync_ocr_min_score": sync.get("ocr", {}).get("min_score")},
        "mapping_check": mapping_check,
        "windows_with_atd_check": len(checked),
        "scan_ranges": len(scan_rows), "scan_frames": int(sum(r["frames"] for r in scan_rows)),
        "runtime_sec": round(time.time() - t0, 1),
    })
    with open(os.path.join(args.out_dir, "broadcast_windows_summary.json"), "w") as handle:
        json.dump(summ, handle, indent=1)

    print("{} windows ({} stoppage, {} restart) -> {}".format(
        len(windows), sum(w["kind"] == "stoppage" for w in windows), sum(w["kind"] == "restart" for w in windows),
        os.path.relpath(wpath, ROOT)))
    print("\n{:<10}".format("kind") + "".join("{:>16}".format(v) for v in VIEWS))
    for k, row in summ["by_kind_view"].items():
        print("{:<10}".format(k) + "".join("{:>16}".format(row[v]) for v in VIEWS))
    print("\nM1 go/no-go: of {} v6 stretches, {} have a good window, {} a usable one, {} no view".format(
        summ["stretches"], summ["stretches_with_good_window"], summ["stretches_with_usable_window"],
        summ["stretches_without_view"]))
    print("mapping vs sync index: {}".format(mapping_check))
    print("ATD-checked windows: {} | P3 scan: {} ranges, {} broadcast frames ({:.1f} min) -> {}".format(
        len(checked), summ["scan_ranges"], summ["scan_frames"], summ["scan_frames"] / pm.fps_b / 60.0,
        os.path.relpath(spath, ROOT)))
    print("done in {:.1f}s".format(summ["runtime_sec"]))


if __name__ == "__main__":
    main()
