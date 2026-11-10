"""Per run of metrics/out/runs.csv (no breast MRI cohort): GPU model and median seconds per training step (between
consecutive logged iterations, validation excluded) from the run logs. python parse_logs.py <runs.csv> <steps.csv>"""
import os
import re, sys, glob
from datetime import datetime
from multiprocessing import Pool
import numpy as np, pandas as pd
RES = (os.environ.get("BFE_RESULTS") or "/projects/cvpr2026_challenge_fomo4ct/beyond_frozen_encoders") + "/"  # structured results folder (tools/organize_results.py); BFE_RESULTS overrides
TS = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3})\]")
IT = re.compile(r"LoggingHook.*- iteration: (\d+),")
def parse(path):
    out = dict(path=path, gpu=None, n_logs=0, s_per_step=np.nan, n_intervals=0, val_s=np.nan)
    rates, vals = [], []
    for f in sorted(glob.glob(RES + path + "/logs/log_*.txt")):
        out["n_logs"] += 1
        prev = None; vstart = None
        with open(f, errors="replace") as fh:
            for line in fh:
                if out["gpu"] is None and "Device 0:" in line:
                    out["gpu"] = line.split("Device 0:")[1].split("|")[0].strip()
                m = TS.match(line)
                if not m: continue
                t = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S,%f")
                if "Starting validation" in line:
                    vstart = t; prev = None; continue
                mi = IT.search(line)
                if mi:
                    it = int(mi.group(1))
                    if vstart is not None:
                        vals.append((t - vstart).total_seconds()); vstart = None
                    elif prev is not None and it > prev[0]:
                        rates.append((t - prev[1]).total_seconds() / (it - prev[0]))
                    prev = (it, t)
    if rates:
        out["s_per_step"] = float(np.median(rates)); out["n_intervals"] = len(rates)
    if vals:
        out["val_s"] = float(np.median(vals))
    return out
if __name__ == "__main__":
    runs = pd.read_csv(sys.argv[1])
    with Pool(8) as p:
        rows = p.map(parse, runs.path.tolist(), chunksize=8)
    d = runs.merge(pd.DataFrame(rows), on="path")
    d.to_csv(sys.argv[2], index=False)
    print(d.gpu.value_counts(dropna=False)); print(d.s_per_step.describe())
