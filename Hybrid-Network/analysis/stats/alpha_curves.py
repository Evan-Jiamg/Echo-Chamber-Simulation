"""Alpha curves pooled across networks and topics, at the current n."""
import collections
import csv
import glob
import json
import os

import numpy as np
import scipy.stats

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
import hcog_paths

# The extract carries everything these tables need. They read convergence.json,
# metrics.csv and poa_components.csv only.
G = hcog_paths.grid_root()


def ci95(a):
    a = np.asarray(a, float)
    if len(a) < 2:
        return 0.0
    return float(scipy.stats.t.ppf(0.975, len(a) - 1) * a.std(ddof=1) / np.sqrt(len(a)))


def load_csv(p):
    cols = collections.defaultdict(list)
    with open(p) as f:
        for row in csv.DictReader(f):
            for k, v in row.items():
                try:
                    cols[k].append(float(v))
                except (TypeError, ValueError):
                    cols[k].append(np.nan)
    return cols


rows = collections.defaultdict(list)
rows_t = collections.defaultdict(list)
rows_n = collections.defaultdict(list)

for f in glob.glob(G + "/*/*/alpha_*/seed_*/convergence.json"):
    d = json.load(open(f))
    p = f.split("/phi4/")[1].split("/")
    topic, net, a = p[0], p[1], float(p[2].split("_")[1])
    c = load_csv(f.replace("convergence.json", "metrics.csv"))
    if not c.get("step"):
        continue
    steps = np.array(c["step"], int)
    tc = d.get("t_conv")
    # A run that never converged has no t_conv + post_window; its last step is
    # the honest readout, and it is flagged separately rather than dropped.
    i = len(steps) - 1
    if tc is not None:
        hit = np.where(steps == tc + 10)[0]
        if len(hit):
            i = int(hit[0])
    rec = {m: c[m][i] for m in ("polarization", "modularity", "Q_norm", "poa", "C_out")
           if m in c}
    rows[a].append(rec)
    rows_t[(a, topic)].append(rec)
    rows_n[(a, net)].append(rec)

M = ("poa", "polarization", "Q_norm", "C_out")
print("=" * 94)
print("pooled over 3 networks x 2 topics, at t_conv + 10")
print("=" * 94)
print("  %-7s %-4s %-21s %-21s %-21s %s" % ("alpha", "n", "PoA", "Pz", "Q_norm", "C_out"))
for a in sorted(rows):
    rs = rows[a]
    line = "  %-7g %-4d " % (a, len(rs))
    for m in M:
        v = [r[m] for r in rs if m in r and np.isfinite(r[m])]
        line += ("%.3f+-%.3f" % (np.mean(v), ci95(v))).ljust(21) if v else "-".ljust(21)
    print(line)

print()
print("=" * 94)
print("PoA by topic")
print("=" * 94)
print("  %-7s %-26s %-26s" % ("alpha", "gun_control", "abortion"))
for a in sorted(rows):
    line = "  %-7g " % a
    for t in ("gun_control", "abortion"):
        v = [r["poa"] for r in rows_t.get((a, t), []) if np.isfinite(r.get("poa", np.nan))]
        line += ("%.3f+-%.3f (n=%d)" % (np.mean(v), ci95(v), len(v))).ljust(26) if v else "-".ljust(26)
    print(line)

print()
print("=" * 94)
print("PoA by network   <- Q1 with LLM agents present")
print("=" * 94)
print("  %-7s %-23s %-23s %-23s" % ("alpha", "scale_free", "random", "small_world"))
for a in sorted(rows):
    line = "  %-7g " % a
    for n in ("scale_free", "random", "small_world"):
        v = [r["poa"] for r in rows_n.get((a, n), []) if np.isfinite(r.get("poa", np.nan))]
        line += ("%.3f+-%.3f" % (np.mean(v), ci95(v))).ljust(23) if v else "-".ljust(23)
    print(line)

print()
print("=" * 94)
print("Pz by network")
print("=" * 94)
print("  %-7s %-23s %-23s %-23s" % ("alpha", "scale_free", "random", "small_world"))
for a in sorted(rows):
    line = "  %-7g " % a
    for n in ("scale_free", "random", "small_world"):
        v = [r["polarization"] for r in rows_n.get((a, n), [])
             if np.isfinite(r.get("polarization", np.nan))]
        line += ("%.3f+-%.3f" % (np.mean(v), ci95(v))).ljust(23) if v else "-".ljust(23)
    print(line)
