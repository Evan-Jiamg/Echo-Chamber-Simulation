"""
Is the PoA rise from alpha=0 to alpha=0.125 real?

Pooling across networks and topics hides that the two topics behave differently,
and the two topics are not independent (same agent personas and stubbornness,
stance correlation +0.881), so they are tested separately rather than pooled or
treated as 2x the sample.

Welch's t-test throughout: variance is visibly unequal across alpha.
"""
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


def load(p):
    cols = collections.defaultdict(list)
    with open(p) as f:
        for row in csv.DictReader(f):
            for k, v in row.items():
                try:
                    cols[k].append(float(v))
                except (TypeError, ValueError):
                    cols[k].append(np.nan)
    return cols


vals = collections.defaultdict(list)          # (alpha, topic) -> [poa]
pooled = collections.defaultdict(list)        # alpha -> [poa]
for f in glob.glob(G + "/*/*/alpha_*/seed_*/convergence.json"):
    d = json.load(open(f))
    p = f.split("/phi4/")[1].split("/")
    topic, a = p[0], float(p[2].split("_")[1])
    c = load(f.replace("convergence.json", "metrics.csv"))
    steps = np.array(c["step"], int)
    tc = d.get("t_conv")
    i = len(steps) - 1
    if tc is not None:
        h = np.where(steps == tc + 10)[0]
        if len(h):
            i = int(h[0])
    vals[(a, topic)].append(c["poa"][i])
    pooled[a].append(c["poa"][i])

ALPHAS = sorted(pooled)


def welch(a, b):
    t, p = scipy.stats.ttest_ind(a, b, equal_var=False)
    na, nb = len(a), len(b)
    sd = np.sqrt(((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2))
    return t, p, (np.mean(a) - np.mean(b)) / sd


print("=" * 78)
print("Is PoA at alpha=0.125 higher than at alpha=0?   Welch's t-test")
print("=" * 78)
tests = []
for topic in ("gun_control", "abortion", "POOLED"):
    a0 = pooled[0.0] if topic == "POOLED" else vals[(0.0, topic)]
    a1 = pooled[0.125] if topic == "POOLED" else vals[(0.125, topic)]
    t, p, dz = welch(a1, a0)
    tests.append((topic, p))
    print("  %-12s  alpha=0: %.3f (n=%d)   alpha=0.125: %.3f (n=%d)   "
          "t=%+.2f  p=%.4f  d=%.2f"
          % (topic, np.mean(a0), len(a0), np.mean(a1), len(a1), t, p, dz))

print()
print("=" * 78)
print("Peak location: is the maximum significantly above alpha=0, per topic?")
print("=" * 78)
for topic in ("gun_control", "abortion"):
    series = [(a, np.mean(vals[(a, topic)])) for a in ALPHAS]
    peak_a = max(series[:5], key=lambda x: x[1])[0]   # search alpha <= 0.5
    t, p, dz = welch(vals[(peak_a, topic)], vals[(0.0, topic)])
    print("  %-12s peak at alpha=%g (%.3f) vs alpha=0 (%.3f):  p=%.4f  d=%.2f  -> %s"
          % (topic, peak_a, np.mean(vals[(peak_a, topic)]),
             np.mean(vals[(0.0, topic)]), p, dz,
             "SIGNIFICANT" if p < 0.05 else "not significant"))

print()
print("=" * 78)
print("Adjacent-alpha comparisons, pooled, with BH-FDR")
print("=" * 78)
ps, labels = [], []
for i in range(len(ALPHAS) - 1):
    a, b = ALPHAS[i], ALPHAS[i + 1]
    t, p, dz = welch(pooled[b], pooled[a])
    ps.append(p)
    labels.append((a, b, np.mean(pooled[a]), np.mean(pooled[b]), t, dz))
order = np.argsort(ps)
m = len(ps)
adj = np.empty(m)
prev = 1.0
for rank, idx in enumerate(order[::-1]):
    k = m - rank
    prev = min(prev, ps[idx] * m / k)
    adj[idx] = prev
for (a, b, ma, mb, t, dz), p, q in zip(labels, ps, adj):
    flag = "*" if q < 0.05 else " "
    print("  %-6g -> %-6g  %6.3f -> %6.3f  t=%+6.2f  p=%.4g  q=%.4g %s"
          % (a, b, ma, mb, t, p, q, flag))

print()
print("=" * 78)
print("Q1: are the three networks equivalent?  TOST, delta = 0.2 "
      "(see Statistics in the root README)")
print("=" * 78)
netvals = collections.defaultdict(list)
for f in glob.glob(G + "/*/*/alpha_*/seed_*/convergence.json"):
    d = json.load(open(f))
    p = f.split("/phi4/")[1].split("/")
    net, a = p[1], float(p[2].split("_")[1])
    c = load(f.replace("convergence.json", "metrics.csv"))
    steps = np.array(c["step"], int)
    tc = d.get("t_conv")
    i = len(steps) - 1
    if tc is not None:
        h = np.where(steps == tc + 10)[0]
        if len(h):
            i = int(h[0])
    netvals[(a, net)].append(c["poa"][i])

DELTA = 0.2
print("  %-8s %-26s %-10s %s" % ("alpha", "pair", "diff", "TOST p (equivalence)"))
for a in ALPHAS:
    nets = ("scale_free", "random", "small_world")
    worst_p, worst_pair, worst_d = 0.0, None, None
    for i in range(3):
        for j in range(i + 1, 3):
            x, y = netvals[(a, nets[i])], netvals[(a, nets[j])]
            diff = np.mean(x) - np.mean(y)
            se = np.sqrt(np.var(x, ddof=1) / len(x) + np.var(y, ddof=1) / len(y))
            df = len(x) + len(y) - 2
            p1 = scipy.stats.t.sf((diff + DELTA) / se, df)
            p2 = scipy.stats.t.cdf((diff - DELTA) / se, df)
            p = max(p1, p2)
            if p > worst_p:
                worst_p, worst_pair, worst_d = p, (nets[i], nets[j]), diff
    verdict = "EQUIVALENT" if worst_p < 0.05 else "not shown"
    print("  %-8g %-26s %+.3f      %.4f  %s"
          % (a, "%s/%s" % worst_pair, worst_d, worst_p, verdict))
