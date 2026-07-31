#!/usr/bin/env python3
"""
decompose_poa.py — split the social cost behind PoA into its two terms.

    C(z) = sum_i sum_{j in N(i)} (z_i - z_j)^2      "conflict": disagreeing with
                                                     the neighbours you kept
         + sum_i rho_i * K * (z_i - s_i)^2          "conformity": having moved
                                                     off your own intrinsic view

PoA is the ratio of that total to the central planner's minimum, so a single PoA
number cannot say whether a population is inefficient because it argues or
because it caves. The split says which.

Everything is recomputed from artefacts each run already writes
(agents_data.json, edges_per_step.json) so metrics.csv keeps its schema and runs
finished before this existed are covered too — nothing is re-simulated.

The recomputed total is checked against the stored `poa` column; a run whose
check fails is reported, not silently included.

Usage:
  python analysis/decompose_poa.py                      # whole M-1 grid
  python analysis/decompose_poa.py --limit 6            # spot-check
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJ_ROOT = Path(__file__).resolve().parent.parent
K = 5


def load_stub_and_s(topic, n=50):
    p = (PROJ_ROOT / "data" / "agents" /
         f"numeric_sim_opnions_and_stubbornness_num_agents_{n}_{topic}.json")
    raw = json.load(open(p))
    s = np.array([float(raw["opinions"][str(i)]) for i in range(n)])
    rho = np.array([float(raw["stubbornness"][str(i)]) for i in range(n)])
    return s, rho


def costs(z, s, rho, A):
    """Conflict and conformity terms of C(z) on directed adjacency A."""
    dz = z[:, None] - z[None, :]
    conflict = float(np.sum(A * dz ** 2))
    conformity = float(np.dot(rho * K, (z - s) ** 2))
    return conflict, conformity


def optimal(s, rho, A):
    """Central planner's minimiser and its two terms — mirrors model.py."""
    n = len(s)
    D_out = np.diag(A.sum(axis=1))
    D_in = np.diag(A.sum(axis=0))
    L_sym = D_out + D_in - A - A.T
    W = K * np.diag(rho)
    M = L_sym + W
    if np.linalg.matrix_rank(M) < n:
        M = M + 1e-8 * np.eye(n)
    z_opt = np.linalg.solve(M, W @ s)
    return z_opt, costs(z_opt, s, rho, A)


def decompose_run(run_dir, tol=0.02):
    agents = json.load(open(run_dir / "agents_data.json"))
    edges_per_step = json.load(open(run_dir / "edges_per_step.json"))
    topic = run_dir.parent.parent.parent.name
    s, rho = load_stub_and_s(topic)
    n = len(s)

    beliefs = np.array([agents[str(i)]["beliefs"] for i in range(n)])  # (n, T+1)

    stored = {}
    with open(run_dir / "metrics.csv") as f:
        for row in csv.DictReader(f):
            stored[int(float(row["step"]))] = float(row["poa"])

    # beliefs[:, 0] is the initial state; step t's graph is edges_per_step[t-1].
    # Confirm that alignment against the stored PoA rather than assuming it.
    def series(offset):
        out = []
        for t_idx, edges in enumerate(edges_per_step):
            step = t_idx + 1
            bi = t_idx + offset
            if bi >= beliefs.shape[1] or step not in stored:
                continue
            A = np.zeros((n, n))
            for a, b in edges:
                A[int(a), int(b)] = 1.0
            z = beliefs[:, bi]
            cf, cm = costs(z, s, rho, A)
            _, (ocf, ocm) = optimal(s, rho, A)
            oc = ocf + ocm
            out.append({
                "step": step,
                "conflict": cf, "conformity": cm, "total": cf + cm,
                "opt_conflict": ocf, "opt_conformity": ocm, "opt_total": oc,
                "poa": (cf + cm) / oc if oc > 0 else 1.0,
                "poa_stored": stored[step],
            })
        return out

    best, best_err = None, np.inf
    for offset in (1, 0):
        rows = series(offset)
        if not rows:
            continue
        err = float(np.median([abs(r["poa"] - r["poa_stored"]) /
                               max(r["poa_stored"], 1e-9) for r in rows]))
        if err < best_err:
            best, best_err = rows, err

    if best is None:
        return None, "no comparable steps"
    if best_err > tol:
        return None, f"PoA mismatch (median rel. err {best_err:.3f})"

    for r in best:
        tot = r["conflict"] + r["conformity"]
        r["conflict_share"] = r["conflict"] / tot if tot else 0.0
        r["conformity_share"] = r["conformity"] / tot if tot else 0.0
        # Excess over the planner's own split: where the inefficiency sits.
        r["poa_conflict"] = (r["conflict"] / r["opt_total"]) if r["opt_total"] else 0.0
        r["poa_conformity"] = (r["conformity"] / r["opt_total"]) if r["opt_total"] else 0.0
    return best, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="M-1_main-grid")
    ap.add_argument("--model", default="phi4")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = PROJ_ROOT / "experiments" / args.run_id / args.model
    runs = sorted(root.glob("*/*/alpha_*/seed_*/convergence.json"))
    if args.limit:
        runs = runs[:args.limit]
    print(f"{len(runs)} completed runs under {root}")

    per_cond = defaultdict(list)
    failures = []
    for i, conv in enumerate(runs, 1):
        d = conv.parent
        rows, err = decompose_run(d)
        if err:
            failures.append((str(d).split(args.model + "/")[1], err))
            continue
        (d / "poa_components.csv").write_text(
            ",".join(rows[0].keys()) + "\n"
            + "\n".join(",".join(f"{v:.6g}" for v in r.values()) for r in rows) + "\n")

        cfg = json.load(open(conv))
        t_conv = cfg.get("t_conv")
        target = (t_conv + 10) if t_conv else rows[-1]["step"]
        pick = min(rows, key=lambda r: abs(r["step"] - target))
        per_cond[(d.parent.parent.parent.name, d.parent.parent.name,
                  float(d.parent.name.split("_")[1]))].append(pick)
        if i % 25 == 0:
            print(f"  {i}/{len(runs)}")

    if failures:
        print(f"\n{len(failures)} runs failed the PoA cross-check:")
        for name, why in failures[:10]:
            print(f"  {name}: {why}")

    print("\nPoA decomposition at t_conv + 10")
    print(f"  {'topic':<12}{'network':<13}{'alpha':<7}{'n':<4}"
          f"{'PoA':<9}{'conflict':<10}{'conformity':<12}{'conflict%'}")
    out_rows = []
    for key in sorted(per_cond):
        topic, net, alpha = key
        rs = per_cond[key]
        poa = np.mean([r["poa"] for r in rs])
        pc = np.mean([r["poa_conflict"] for r in rs])
        pf = np.mean([r["poa_conformity"] for r in rs])
        sh = np.mean([r["conflict_share"] for r in rs])
        print(f"  {topic:<12}{net:<13}{alpha:<7g}{len(rs):<4}"
              f"{poa:<9.3f}{pc:<10.3f}{pf:<12.3f}{sh:.1%}")
        out_rows.append(dict(topic=topic, network=net, alpha=alpha, n=len(rs),
                             poa=poa, poa_conflict=pc, poa_conformity=pf,
                             conflict_share=sh))

    out = Path(args.out) if args.out else (PROJ_ROOT / "analysis" /
                                           f"poa_decomposition_{args.run_id}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(out_rows, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
