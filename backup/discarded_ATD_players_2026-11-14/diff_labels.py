"""Print where two submission CSVs disagree, longest stretch first.

    python diff_labels.py                                   # v6_det_local vs v6_atd
    python diff_labels.py outputs/submissions/a.csv outputs/submissions/b.csv 30
"""
import sys

import pandas as pd

a_path = sys.argv[1] if len(sys.argv) > 1 else "outputs/submissions/v6_det_local.csv"
b_path = sys.argv[2] if len(sys.argv) > 2 else "outputs/submissions/v6.csv"
top = int(sys.argv[3]) if len(sys.argv) > 3 else 20

a = pd.read_csv(a_path)["label"].astype(str).str.strip()
b = pd.read_csv(b_path)["label"].astype(str).str.strip()
diff = a != b

# a new stretch starts wherever the (a, b) pair changes
pair = a + b
new = (pair != pair.shift()) | (diff != diff.shift())
df = pd.DataFrame({"frame": range(len(a)), "a": a, "b": b, "diff": diff, "run": new.cumsum()})
runs = (df[df["diff"]].groupby("run")
        .agg(start=("frame", "first"), end=("frame", "last"), a=("a", "first"), b=("b", "first")))
runs["frames"] = runs.end - runs.start + 1
runs["seconds"] = (runs.frames / 25).round(1)
runs["video_time"] = runs.start.map(lambda f: "{:02d}:{:02d}".format(f // 25 // 60, f // 25 % 60))
runs = runs.sort_values("frames", ascending=False).reset_index(drop=True)

print("A = {}\nB = {}".format(a_path, b_path))
print("differ on {} frames ({:.1f}% of all rows), {} stretches".format(
    int(diff.sum()), 100 * diff.mean(), len(runs)))
print(runs.groupby(["a", "b"]).frames.agg(["count", "sum"]).rename(
    columns={"count": "stretches", "sum": "frames"}).to_string())
print("\nlongest {}:".format(top))
print(runs.head(top)[["start", "end", "frames", "seconds", "video_time", "a", "b"]].to_string())
