"""Steps and t_conv per alpha -- what dynamic stopping actually costs."""
import collections
import glob
import json
import os

import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
import hcog_paths

# The extract carries everything these tables need. They read convergence.json,
# metrics.csv and poa_components.csv only.
G = hcog_paths.grid_root()

by_alpha = collections.defaultdict(list)
by_alpha_topic = collections.defaultdict(list)
by_alpha_net = collections.defaultdict(list)
attr = collections.defaultdict(collections.Counter)

for f in glob.glob(G + "/*/*/alpha_*/seed_*/convergence.json"):
    d = json.load(open(f))
    p = f.split("/phi4/")[1].split("/")
    topic, net, a = p[0], p[1], float(p[2].split("_")[1])
    rec = (d["t_conv"], d["steps_run"], d["hit_T_max"], d["C_plateau"])
    by_alpha[a].append(rec)
    by_alpha_topic[(a, topic)].append(rec)
    by_alpha_net[(a, net)].append(rec)
    attr[a][d["attractor"]] += 1

print("=" * 88)
print("T per alpha  (T = steps_run = t_conv + post_window(10))")
print("=" * 88)
print("  %-7s %-4s | %-28s | %-28s | %s"
      % ("alpha", "n", "t_conv  mean [min-max] p90", "steps   mean [min-max] p90", "hit_T_max"))
tot_steps = 0
for a in sorted(by_alpha):
    rs = by_alpha[a]
    tc = np.array([r[0] for r in rs if r[0] is not None])
    st = np.array([r[1] for r in rs])
    hm = sum(1 for r in rs if r[2])
    tot_steps += st.sum()
    print("  %-7g %-4d | %5.1f  [%3d-%3d]  %5.1f     | %5.1f  [%3d-%3d]  %5.1f     | %d/%d"
          % (a, len(rs), tc.mean(), tc.min(), tc.max(), np.percentile(tc, 90),
             st.mean(), st.min(), st.max(), np.percentile(st, 90), hm, len(rs)))

print("")
print("  overall mean steps = %.1f   (T_max = 120)" % (tot_steps / sum(len(v) for v in by_alpha.values())))
print("  planning assumption was ~40 steps; the fixed-T=35 setting in the old")
print("  paper would have truncated most of these before convergence.")

print("")
print("=" * 88)
print("steps by topic")
print("=" * 88)
print("  %-7s %-22s %-22s" % ("alpha", "gun_control", "abortion"))
for a in sorted(by_alpha):
    row = "  %-7g " % a
    for t in ("gun_control", "abortion"):
        rs = by_alpha_topic.get((a, t), [])
        if rs:
            st = np.array([r[1] for r in rs])
            row += "%5.1f (n=%2d, %3d-%3d)   " % (st.mean(), len(rs), st.min(), st.max())
        else:
            row += "%-22s" % "-"
    print(row)

print("")
print("=" * 88)
print("steps by network")
print("=" * 88)
print("  %-7s %-20s %-20s %-20s" % ("alpha", "scale_free", "random", "small_world"))
for a in sorted(by_alpha):
    row = "  %-7g " % a
    for n in ("scale_free", "random", "small_world"):
        rs = by_alpha_net.get((a, n), [])
        if rs:
            st = np.array([r[1] for r in rs])
            row += "%5.1f (n=%2d)        " % (st.mean(), len(rs))
        else:
            row += "%-20s" % "-"
    print(row)

print("")
print("=" * 88)
print("attractor mix per alpha")
print("=" * 88)
for a in sorted(attr):
    items = ", ".join("%s=%d" % (k, v) for k, v in sorted(attr[a].items()))
    print("  %-7g %s" % (a, items))

print("")
print("=" * 88)
print("C_plateau per alpha  (neighbour retention per step)")
print("=" * 88)
for a in sorted(by_alpha):
    cp = np.array([r[3] for r in by_alpha[a] if r[3] is not None])
    if len(cp):
        print("  %-7g n=%-3d mean=%.4f  sd=%.4f  [%.4f - %.4f]"
              % (a, len(cp), cp.mean(), cp.std(ddof=1) if len(cp) > 1 else 0,
                 cp.min(), cp.max()))
