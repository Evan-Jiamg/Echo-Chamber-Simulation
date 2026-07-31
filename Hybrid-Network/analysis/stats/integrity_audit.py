"""Integrity audit of what M-1 has actually persisted."""
import collections
import csv
import glob
import json
import os

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
import hcog_paths

# The extract carries everything these tables need. audit.py checks agents_data.json and
# edges_per_step.json, which the extract omits, so it wants the raw grid.
G = hcog_paths.raw_grid_root()
EXPECTED_COLS = ["step", "polarization", "modularity", "poa", "dS", "C_out", "dL",
                 "deltacon", "n_comm", "max_comm_share", "ari_camp", "in_gini",
                 "parse_fail", "Q_rand_mean", "Q_rand_sd", "Q_norm", "z_Q", "swap_fail"]
EXPECTED_FILES = ["metrics.csv", "convergence.json", "agents_data.json",
                  "edges_per_step.json", "model_overview.json", "agent_assignment.json"]

runs = sorted(glob.glob(G + "/*/*/alpha_*/seed_*/convergence.json"))
print("completed runs: %d" % len(runs))

missing = collections.defaultdict(list)
badcols = []
lenmismatch = []
empty = []
sizes = collections.Counter()

for conv in runs:
    d = os.path.dirname(conv)
    rel = d.split("/phi4/")[1]
    for fn in EXPECTED_FILES:
        p = os.path.join(d, fn)
        if not os.path.exists(p):
            missing[fn].append(rel)
        elif os.path.getsize(p) == 0:
            empty.append(rel + "/" + fn)
        else:
            sizes[fn] += os.path.getsize(p)

    p = os.path.join(d, "metrics.csv")
    if os.path.exists(p) and os.path.getsize(p) > 0:
        with open(p) as f:
            rd = csv.reader(f)
            hdr = next(rd, [])
            n = sum(1 for _ in rd)
        if hdr != EXPECTED_COLS:
            badcols.append((rel, len(hdr)))
        cj = json.load(open(conv))
        if cj.get("steps_run") is not None and n != cj["steps_run"]:
            lenmismatch.append((rel, n, cj["steps_run"]))

print()
print("=" * 78)
print("FILE PRESENCE")
print("=" * 78)
for fn in EXPECTED_FILES:
    m = missing.get(fn, [])
    tag = "OK" if not m else ("MISSING in %d runs" % len(m))
    print("  %-28s %-24s total %8.1f MB"
          % (fn, tag, sizes[fn] / 1e6))
    for r in m[:3]:
        print("      - " + r)

print()
print("=" * 78)
print("METRICS.CSV INTEGRITY")
print("=" * 78)
print("  expected 18 columns: %s" % ("OK, all runs match"
                                     if not badcols else "%d runs differ" % len(badcols)))
for r, n in badcols[:5]:
    print("      - %s has %d cols" % (r, n))
print("  row count == steps_run: %s" % ("OK, all runs match"
                                        if not lenmismatch else "%d mismatch" % len(lenmismatch)))
for r, a, b in lenmismatch[:5]:
    print("      - %s rows=%d steps_run=%d" % (r, a, b))
print("  zero-byte files: %s" % ("none" if not empty else empty[:5]))

print()
print("=" * 78)
print("WHAT IS INSIDE agents_data.json / agents_interaction_data.json")
print("=" * 78)
d0 = os.path.dirname(runs[0])
ad = json.load(open(os.path.join(d0, "agents_data.json")))
k0 = list(ad)[0]
print("  agents_data.json   : %d agents, fields per agent = %s"
      % (len(ad), list(ad[k0].keys())))
for f, v in ad[k0].items():
    print("      %-14s len=%-5s sample=%s"
          % (f, len(v) if isinstance(v, list) else "-", str(v[:1])[:70]))

ip = os.path.join(d0, "agents_interaction_data.json")
if os.path.exists(ip):
    it = json.load(open(ip))
    print("  agents_interaction_data.json : type=%s len=%d" % (type(it).__name__, len(it)))
    if isinstance(it, dict):
        kk = list(it)[0]
        inner = it[kk]
        print("      key=%s -> %s" % (kk, type(inner).__name__))
        if isinstance(inner, dict):
            print("      fields: %s" % list(inner.keys()))
            for f, v in list(inner.items())[:6]:
                print("        %-16s %s" % (f, str(v)[:80]))
        elif isinstance(inner, list) and inner:
            print("      first elem: %s" % str(inner[0])[:200])
else:
    print("  agents_interaction_data.json : MISSING")

print()
print("=" * 78)
print("REASONING TRACES  (needed for A-2)")
print("=" * 78)
found = False
for fn in ("agents_data.json", "agents_interaction_data.json", "model_overview.json"):
    p = os.path.join(d0, fn)
    if not os.path.exists(p):
        continue
    raw = open(p, encoding="utf-8", errors="replace").read()
    hit = ("reasoning" in raw.lower())
    print("  %-32s contains 'reasoning': %s" % (fn, hit))
    found = found or hit
if not found:
    print("  -> CoT reasoning is generated every step but NOT persisted anywhere.")

print()
print("=" * 78)
print("DISK")
print("=" * 78)
tot = sum(sizes.values())
print("  recorded so far : %.2f GB over %d runs (%.1f MB/run)"
      % (tot / 1e9, len(runs), tot / 1e6 / max(len(runs), 1)))
print("  projected @540  : %.2f GB" % (tot / 1e9 / max(len(runs), 1) * 540))
