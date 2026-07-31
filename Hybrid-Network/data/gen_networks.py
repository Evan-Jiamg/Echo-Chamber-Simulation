#!/usr/bin/env python3
"""
gen_networks.py — generate the network assets for the H-COG sweep.

Produces BA / ER / WS graphs for seeds 1..30 at N=50, with the mean degree of
ER and WS aligned to BA. Degree alignment is a precondition for the Q1
comparison: without it, any difference between topologies is confounded with a
difference in how many neighbours agents start with.

Alignment procedure
  BA   barabasi_albert_graph(N, m=2) — the reference; mean degree d̄ is measured
       over the 30 seeds.
  ER   gnm_random_graph(N, M) with M = round(d̄·N/2) — matches the edge count
       exactly, so the mean degree matches by construction.
  WS   watts_strogatz_graph(N, k, p=0.1) with k the even integer nearest d̄.
       k must be even, so a small residual mismatch is unavoidable; it is
       reported and must stay under 5%.

All graphs are required to be connected. Disconnected draws are resampled with
a perturbed seed, so no agent starts with an empty neighbourhood (which would
silently freeze that agent's Friedkin–Johnsen update).

Output matches the existing schema:
  {nodes, edges, clustering_coefficient, average_path_length, density, diameter}
"""

import argparse
import json
from pathlib import Path

import networkx as nx
import numpy as np

N_DEFAULT = 50
SEEDS_DEFAULT = range(1, 31)
BA_M = 2
WS_P = 0.1
MAX_RESAMPLE = 200


def _connected(G):
    return G.number_of_nodes() > 0 and nx.is_connected(G)


def make_ba(n, seed):
    return nx.barabasi_albert_graph(n, BA_M, seed=seed)


def make_er(n, seed, m_edges):
    for attempt in range(MAX_RESAMPLE):
        G = nx.gnm_random_graph(n, m_edges, seed=seed * 1000 + attempt)
        if _connected(G):
            return G, attempt
    raise RuntimeError(f"ER seed {seed}: no connected draw in {MAX_RESAMPLE} attempts")


def make_ws(n, seed, k):
    for attempt in range(MAX_RESAMPLE):
        G = nx.watts_strogatz_graph(n, k, WS_P, seed=seed * 1000 + attempt)
        if _connected(G):
            return G, attempt
    raise RuntimeError(f"WS seed {seed}: no connected draw in {MAX_RESAMPLE} attempts")


def describe(G):
    return {
        "nodes": sorted(G.nodes()),
        "edges": [list(e) for e in sorted(G.edges())],
        "clustering_coefficient": float(nx.average_clustering(G)),
        "average_path_length": float(nx.average_shortest_path_length(G)),
        "density": float(nx.density(G)),
        "diameter": int(nx.diameter(G)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True,
                    help="output directory (Hybrid-Network/data/networks)")
    ap.add_argument("--num_agents", type=int, default=N_DEFAULT)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS_DEFAULT))
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    n, seeds = args.num_agents, args.seeds
    out = Path(args.out)

    # 1) BA is the reference; measure its mean degree across the seed set.
    ba = {s: make_ba(n, s) for s in seeds}
    ba_deg = np.array([2 * G.number_of_edges() / n for G in ba.values()])
    d_bar = float(ba_deg.mean())

    m_edges = int(round(d_bar * n / 2))
    k_ws = max(2, int(round(d_bar / 2)) * 2)

    print(f"N={n}  seeds={len(seeds)}")
    print(f"BA(m={BA_M})  mean degree d̄ = {d_bar:.4f}  (edges {ba_deg.mean()*n/2:.1f})")
    print(f"ER  gnm with M = {m_edges}  -> degree {2*m_edges/n:.4f}")
    print(f"WS  k = {k_ws} (nearest even), p = {WS_P}  -> degree {float(k_ws):.4f}")

    er, ws, resamples = {}, {}, {"ER": 0, "WS": 0}
    for s in seeds:
        er[s], a = make_er(n, s, m_edges); resamples["ER"] += a
        ws[s], a = make_ws(n, s, k_ws); resamples["WS"] += a

    stats = {}
    for name, gs in [("scale_free", ba), ("random", er), ("small_world", ws)]:
        degs = np.array([2 * G.number_of_edges() / n for G in gs.values()])
        stats[name] = degs
        dev = 100 * abs(degs.mean() - d_bar) / d_bar
        flag = "OK" if dev < 5 else "FAIL"
        print(f"  {name:12s} mean_deg={degs.mean():.4f} ({degs.min():.2f}-{degs.max():.2f})"
              f"  deviation vs BA {dev:.2f}%  [{flag}]"
              f"  clustering={np.mean([nx.average_clustering(G) for G in gs.values()]):.3f}")
    print(f"  resamples for connectivity: ER={resamples['ER']}  WS={resamples['WS']}")

    worst = max(100 * abs(stats[k].mean() - d_bar) / d_bar for k in stats)
    if worst >= 5:
        raise SystemExit(f"FAILED: largest mean-degree deviation {worst:.2f}% >= 5%")
    print(f"OK: largest mean-degree deviation {worst:.2f}% < 5%")

    if args.dry_run:
        print("dry run - nothing written")
        return

    out.mkdir(parents=True, exist_ok=True)
    written = 0
    for name, gs in [("scale_free", ba), ("random", er), ("small_world", ws)]:
        for s, G in gs.items():
            path = out / f"{name}_network_num_agents_{n}_seed_{s}.json"
            path.write_text(json.dumps(describe(G)), encoding="utf-8")
            written += 1
    print(f"wrote {written} files to {out}")


if __name__ == "__main__":
    main()
