#!/usr/bin/env python3
"""
build_summary.py — aggregate M-1 runs into per-condition summaries.

Replaces the version that assumed a single topology and equal-length runs.
Two things changed under it:

  Layout    {run_id}/{model}/{topic}/{network}/alpha_{a:.3f}/seed_{s:02d}/
            The old code globbed experiments/{topic}/alpha_{a}/agents_50_.../
            and had no network level at all.

  Length    Dynamic stopping ends each run at t_conv + post_window, so runs
            differ in length. `np.stack` on those raises ValueError. Worse, the
            old code took mat[:, -1] as "final step", which under dynamic
            stopping compares a run that stopped at 51 against one that stopped
            at 102 — different points in the dynamics, silently.

Comparison points follow the root README (Convergence): the primary readout is
t_conv + post_window (every run has converged there, by construction), with
fixed t=20 and t=35 reported alongside as a check that the choice of readout
is not doing the work.

Usage:
  python plots/build_summary.py
  python plots/build_summary.py --run-id M-1_main-grid --model phi4
  python plots/build_summary.py --out-prefix partial   # snapshot mid-sweep
"""

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.stats

PROJ_ROOT = Path(__file__).resolve().parent.parent
EXP_ROOT = PROJ_ROOT / "experiments"
ANALYSIS_DIR = PROJ_ROOT / "analysis"
# Generated tables go in their own directory; analysis/ holds the scripts.
SUMMARY_DIR  = ANALYSIS_DIR / "summaries"

# Reported per condition. The first three are the paper's headline metrics and
# are never dropped; Q_norm supersedes bare modularity for any claim (§3.4).
METRICS = [
    "polarization", "modularity", "Q_norm", "poa",
    "C_out", "dS", "dL", "deltacon",
    "n_comm", "max_comm_share", "ari_camp", "in_gini", "z_Q",
]

POST_WINDOW = 10
FIXED_POINTS = [20, 35]


def ci95(a):
    """95% CI half-width. Returns 0 for n<2 — with n=10 seeds this is thin, and
    the CI reflects topology/assignment variance only (§3.6)."""
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) < 2:
        return 0.0
    se = a.std(ddof=1) / np.sqrt(len(a))
    return float(scipy.stats.t.ppf(0.975, df=len(a) - 1) * se)


def load_csv(path):
    cols = defaultdict(list)
    with open(path) as f:
        for row in csv.DictReader(f):
            for k, v in row.items():
                try:
                    cols[k].append(float(v))
                except (TypeError, ValueError):
                    cols[k].append(np.nan)
    return {k: np.array(v) for k, v in cols.items()}


def read_run(run_dir):
    """Return one run's readouts, or None if it has not finished.

    A run without convergence.json was interrupted; W-6 re-runs it from
    scratch, so including its partial trajectory here would mix a truncated
    run into the summary.
    """
    conv_path = run_dir / "convergence.json"
    csv_path = run_dir / "metrics.csv"
    if not conv_path.exists() or not csv_path.exists():
        return None

    conv = json.load(open(conv_path))
    cols = load_csv(csv_path)
    if "step" not in cols or len(cols["step"]) == 0:
        return None

    steps = cols["step"].astype(int)
    last = len(steps) - 1

    def at(step):
        """Index of `step`, or None if the run stopped before reaching it."""
        hits = np.where(steps == step)[0]
        return int(hits[0]) if len(hits) else None

    t_conv = conv.get("t_conv")
    idx_conv = at(t_conv + POST_WINDOW) if t_conv is not None else None
    if idx_conv is None:
        # Converged runs stop at t_conv + post_window, so the readout is the
        # last row; fall back to it rather than dropping the run.
        idx_conv = last

    out = {
        "t_conv": t_conv,
        "steps_run": conv.get("steps_run"),
        "attractor": conv.get("attractor"),
        "period": conv.get("period"),
        "hit_T_max": conv.get("hit_T_max"),
        "parse_failures": conv.get("parse_failures"),
        "llm_calls": conv.get("llm_calls"),
        "n_llm": conv.get("n_llm"),
        "at_conv": {},
        "fixed": {t: {} for t in FIXED_POINTS},
        "series": {},
    }
    for m in METRICS:
        if m not in cols:
            continue
        out["at_conv"][m] = float(cols[m][idx_conv])
        out["series"][m] = cols[m].tolist()
        for t in FIXED_POINTS:
            i = at(t)
            out["fixed"][t][m] = float(cols[m][i]) if i is not None else None
    return out


def discover(exp_dir):
    """Walk {topic}/{network}/alpha_*/seed_*/ and group runs by condition."""
    conditions = defaultdict(dict)
    for conv in sorted(exp_dir.glob("*/*/alpha_*/seed_*/convergence.json")):
        run_dir = conv.parent
        seed = int(run_dir.name.split("_")[1])
        alpha = float(run_dir.parent.name.split("_")[1])
        network = run_dir.parent.parent.name
        topic = run_dir.parent.parent.parent.name
        r = read_run(run_dir)
        if r is not None:
            conditions[(topic, network, alpha)][seed] = r
    return conditions


def _timing(values):
    """Summarise t_conv, ignoring runs that never converged."""
    got = [v for v in values if v is not None]
    n_missing = len(values) - len(got)
    if not got:
        return {"values": values, "mean": None, "ci95": None,
                "min": None, "max": None, "n_not_converged": n_missing}
    return {
        "values": values,
        "mean": float(np.mean(got)),
        "ci95": ci95(got),
        "min": int(np.min(got)),
        "max": int(np.max(got)),
        "n_not_converged": n_missing,
    }


def summarise(conditions):
    summary, timeseries = {}, {}

    for (topic, network, alpha), runs in sorted(conditions.items()):
        key = f"{topic}|{network}|{alpha:.3f}"
        seeds = sorted(runs)
        rs = [runs[s] for s in seeds]

        entry = {
            "topic": topic, "network": network, "alpha": alpha,
            "n_seeds": len(seeds), "seeds": seeds,
            "convergence": {
                # A run with attractor "none" has t_conv = null. Averaging
                # over it fails outright, and coercing it to 0 would understate
                # the cost of stopping, so it is dropped here and counted below.
                "t_conv": _timing([r["t_conv"] for r in rs]),
                "attractor_mix": dict(
                    zip(*np.unique([r["attractor"] for r in rs], return_counts=True))
                ),
                "hit_T_max": int(sum(bool(r["hit_T_max"]) for r in rs)),
                "parse_failures": int(sum(r["parse_failures"] or 0 for r in rs)),
                "llm_calls": int(sum(r["llm_calls"] or 0 for r in rs)),
            },
            "at_t_conv": {},
            "at_fixed": {str(t): {} for t in FIXED_POINTS},
        }
        # numpy ints are not JSON-serialisable
        entry["convergence"]["attractor_mix"] = {
            str(k): int(v) for k, v in entry["convergence"]["attractor_mix"].items()
        }

        for m in METRICS:
            vals = [r["at_conv"][m] for r in rs if m in r["at_conv"]]
            if vals:
                entry["at_t_conv"][m] = {
                    "mean": float(np.mean(vals)), "ci95": ci95(vals),
                    "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                    "n": len(vals),
                }
            for t in FIXED_POINTS:
                fv = [r["fixed"][t][m] for r in rs
                      if m in r["fixed"][t] and r["fixed"][t][m] is not None]
                if fv:
                    entry["at_fixed"][str(t)][m] = {
                        "mean": float(np.mean(fv)), "ci95": ci95(fv), "n": len(fv),
                    }
        summary[key] = entry

        # Ragged series: mean over whatever runs are still alive at step t, with
        # the count so a thinning tail is visible rather than silent.
        ts = {}
        for m in METRICS:
            series = [r["series"][m] for r in rs if m in r["series"]]
            if not series:
                continue
            L = max(len(s) for s in series)
            mean, ci, n = [], [], []
            for t in range(L):
                v = [s[t] for s in series if t < len(s) and np.isfinite(s[t])]
                mean.append(float(np.mean(v)) if v else None)
                ci.append(ci95(v) if len(v) > 1 else 0.0)
                n.append(len(v))
            ts[m] = {"mean": mean, "ci95": ci, "n_runs": n}
        timeseries[key] = ts

    return summary, timeseries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="M-1_main-grid")
    ap.add_argument("--model", default="phi4")
    ap.add_argument("--out-prefix", default=None,
                    help="prefix for output files; defaults to the run id")
    args = ap.parse_args()

    exp_dir = EXP_ROOT / args.run_id / args.model
    if not exp_dir.is_dir():
        raise SystemExit(f"not found: {exp_dir}")

    print(f"scanning {exp_dir}")
    conditions = discover(exp_dir)
    n_runs = sum(len(v) for v in conditions.values())
    print(f"  {n_runs} completed runs in {len(conditions)} conditions")
    if not n_runs:
        raise SystemExit("no completed runs yet")

    summary, timeseries = summarise(conditions)

    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    prefix = args.out_prefix or args.run_id
    sp = SUMMARY_DIR / f"summary_{prefix}.json"
    tp = SUMMARY_DIR / f"timeseries_{prefix}.json"
    sp.write_text(json.dumps(summary, indent=2))
    tp.write_text(json.dumps(timeseries, indent=2))
    print(f"  wrote {sp.name}, {tp.name}")

    # Coverage, so a partial sweep never reads as a complete one.
    by_topic = defaultdict(lambda: defaultdict(dict))
    for (topic, network, alpha), runs in conditions.items():
        by_topic[topic][network][alpha] = len(runs)
    print("\ncoverage (seeds per cell):")
    for topic in sorted(by_topic):
        for network in sorted(by_topic[topic]):
            cells = by_topic[topic][network]
            done = sum(cells.values())
            print(f"  {topic:<12} {network:<13} {done:>3} runs over "
                  f"{len(cells)} alphas: "
                  + " ".join(f"{a:g}:{n}" for a, n in sorted(cells.items())))

    print("\nheadline metrics at t_conv + 10:")
    hdr = f"  {'topic':<12}{'network':<13}{'alpha':<7}{'n':<4}"
    print(hdr + f"{'Pz':<18}{'Q_norm':<18}{'PoA':<18}")
    for key in sorted(summary):
        e = summary[key]
        row = f"  {e['topic']:<12}{e['network']:<13}{e['alpha']:<7g}{e['n_seeds']:<4}"
        for m in ("polarization", "Q_norm", "poa"):
            d = e["at_t_conv"].get(m)
            row += f"{d['mean']:.3f}±{d['ci95']:.3f}".ljust(18) if d else "-".ljust(18)
        print(row)


if __name__ == "__main__":
    main()
