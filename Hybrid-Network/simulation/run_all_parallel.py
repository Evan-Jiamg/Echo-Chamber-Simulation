#!/usr/bin/env python3
"""
run_all_parallel.py — Re-run all experiments in parallel using vLLM backend.

Replaces existing data in experiments/ for all topic × alpha × seed combinations.

Prerequisites:
  1. Start vLLM server first:
       bash scripts/start_vllm.sh &
  2. Set the backend URL:
       export VLLM_BASE_URL=http://localhost:8000/v1

Usage:
  python simulation/run_all_parallel.py
  python simulation/run_all_parallel.py --workers 4 --topics gun_control
  python simulation/run_all_parallel.py --dry-run     # print jobs only
"""

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────
PROJ_ROOT  = Path(__file__).resolve().parent.parent
SIM_DIR    = PROJ_ROOT / "simulation"
PYTHON     = os.environ.get("PYTHON", sys.executable)
RUNNER     = str(SIM_DIR / "run_hybrid.py")

TOPICS   = ["gun_control", "abortion"]
ALPHAS   = ["0.0", "0.125", "0.25", "0.375", "0.5", "0.625", "0.75", "0.875", "1.0"]
NETWORKS = ["scale_free", "random", "small_world"]   # BA / ER / WS, degree-aligned (W-2)
SEEDS    = list(range(1, 11))                        # halved to fund the
#                                                    three-network x three-model
#                                                    matrix; supersedes §2's 30

# Records live under Hybrid-Network/experiments as requested; the bytes sit on
# /mnt/NewSSD via a symlink because / is at 87% and M-1 plus the M-3 models
# would add ~30 GB (G-7).
RUN_ID   = "M-1_main-grid"
MODEL    = "phi4"
EXP_ROOT = str(PROJ_ROOT / "experiments")
EXP_DIR  = os.path.join(EXP_ROOT, RUN_ID, MODEL)

# Parallelism: LLM-heavy jobs (alpha < 1) share GPU; alpha=1 is pure numeric.
# Default workers chosen so total concurrent vLLM requests stay manageable.
DEFAULT_WORKERS = 1   # safe for 70% VRAM with Phi-4
FAST_WORKERS    = 4  # alpha=1.0 has no LLM calls — run more in parallel


def build_cmd(topic: str, alpha: str, seed: int, network: str,
              exp_dir: str) -> list[str]:
    # alpha=1.0 has no Type-L agents, so the scorer is never called on real
    # text. Keeping it off the GPU leaves vLLM its 0.85 allocation intact and
    # lets these jobs run several at a time.
    device = "cpu" if alpha == "1.0" else "cuda"
    return [
        PYTHON, RUNNER,
        "--topic",        topic,
        "--alpha",        alpha,
        "--seed",         str(seed),
        "--network_type", network,
        "--exp_dir",      exp_dir,
        "--gpt_model",    "phi4",
        "--temp",         "0.0",
        "--scorer",       "roberta",
        "--scorer_device", device,
        "--field_order",  "cot_first",
        "--num_agents",   "50",
        "--K",            "5",
    ]


def is_complete(topic, alpha, seed, network, exp_dir) -> bool:
    """Same rule W-6 applies, evaluated before paying to start the run.

    run_hybrid.py imports torch and puts the scorer on the GPU before it reaches
    its own resume check, so letting it decide costs ~4s per finished run --
    22 minutes to walk 335 of them after a restart.
    """
    return os.path.exists(os.path.join(
        exp_dir, topic, network, f"alpha_{float(alpha):.3f}",
        f"seed_{int(seed):02d}", "convergence.json"))


def run_one(job: tuple) -> dict:
    topic, alpha, seed, network, exp_dir = job
    if is_complete(topic, alpha, seed, network, exp_dir):
        return {"job": job, "ok": True, "elapsed": 0.0, "skipped": True}
    cmd = build_cmd(topic, alpha, seed, network, exp_dir)
    label = f"{topic}  {network}  alpha={alpha}  seed={seed}"
    t0 = time.time()

    env = os.environ.copy()
    env.setdefault("VLLM_BASE_URL", "http://localhost:8000/v1")

    result = subprocess.run(
        cmd,
        cwd=str(SIM_DIR),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    elapsed = time.time() - t0
    ok = result.returncode == 0
    status = "OK " if ok else "ERR"
    print(f"  [{status}] {label}  ({elapsed:.0f}s)", flush=True)
    if not ok:
        print(f"       STDERR: {result.stderr[-400:]}", flush=True)
    return {"job": job, "ok": ok, "elapsed": elapsed}


def _cleanup_and_rebuild():
    """Rebuild every derived artefact from fresh run data.

    analysis/figures/ is NOT deleted first. It used to hold throwaway per-run
    charts and was cleared wholesale; it now holds the paper figures, and the
    generators overwrite what they produce anyway. Per-run charts go beside
    their own metrics.csv, so nothing accumulates here.

    Stale summaries are removed by name from analysis/summaries/, where
    build_summary.py writes them.
    """
    analysis_dir = PROJ_ROOT / "analysis"
    summaries    = analysis_dir / "summaries"

    print("\n── Cleaning stale summaries ──")
    summaries.mkdir(parents=True, exist_ok=True)
    for pattern in ("summary_*.json", "timeseries_*.json"):
        for f in summaries.glob(pattern):
            f.unlink()
            print(f"  deleted {f.name}")

    print("\n── Rebuilding summaries ──")
    _run_step([PYTHON, str(analysis_dir / "build_summary.py")])

    print("\n── Refreshing the committed extract ──")
    _run_step([PYTHON, str(analysis_dir / "build_results_bundle.py"),
               "--grid", str(PROJ_ROOT / "experiments" / "M-1_main-grid" / "phi4"),
               "--out",  str(PROJ_ROOT / "results" / "M-1_main-grid" / "phi4")])

    print("\n── Rebuilding figures ──")
    _run_step([PYTHON, str(analysis_dir / "make_official_figs.py")])
    _run_step([PYTHON, str(analysis_dir / "make_convergence_figs.py")])

    print("\n── Statistics ──")
    for s in ("stopping_cost.py", "alpha_curves.py", "peak_test.py"):
        _run_step([PYTHON, str(analysis_dir / "stats" / s)])

    print("\nAll done — fresh data, figures and tables ready.")


def _run_step(cmd: list[str]):
    label = " ".join(cmd[-1:])
    print(f"  running {label} ...", flush=True)
    r = subprocess.run(cmd, cwd=str(PROJ_ROOT), capture_output=False, text=True)
    if r.returncode != 0:
        print(f"  [WARN] {label} exited with code {r.returncode}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topics",   nargs="+", default=TOPICS)
    parser.add_argument("--alphas",   nargs="+", default=ALPHAS)
    parser.add_argument("--seeds",    nargs="+", type=int, default=SEEDS)
    parser.add_argument("--networks", nargs="+", default=NETWORKS)
    parser.add_argument("--exp_dir",  default=EXP_DIR)
    parser.add_argument("--no-rebuild", action="store_true",
                        help="skip build_summary/plots at the end; they still "
                             "assume one topology and equal-length runs")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help="Parallel workers for LLM-heavy jobs (alpha<1)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print jobs without running them")
    args = parser.parse_args()

    # Separate pure-numeric (fast) from LLM-heavy jobs
    # Seed-major: finishing seed k means a complete grid at n=k, so an
    # interrupted sweep loses precision rather than whole conditions.
    grid = [(t, a, s, n)
            for s in args.seeds
            for t in args.topics
            for a in args.alphas
            for n in args.networks]
    llm_jobs  = [(t, a, s, n, args.exp_dir) for t, a, s, n in grid if a != "1.0"]
    fast_jobs = [(t, a, s, n, args.exp_dir) for t, a, s, n in grid if a == "1.0"]
    all_jobs  = llm_jobs + fast_jobs

    already = sum(1 for t, a, sd, n in grid
                  if is_complete(t, a, sd, n, args.exp_dir))
    print(f"Output dir : {args.exp_dir}")
    print(f"Already done: {already} (skipped instantly)")
    print(f"Networks   : {args.networks}")
    print(f"Seeds      : {min(args.seeds)}-{max(args.seeds)} ({len(args.seeds)})")
    print(f"Total jobs : {len(all_jobs)}")
    print(f"  LLM jobs : {len(llm_jobs)}  (workers={args.workers})")
    print(f"  Fast jobs: {len(fast_jobs)}  (workers={FAST_WORKERS})")
    print(f"VLLM_BASE_URL: {os.environ.get('VLLM_BASE_URL', 'http://localhost:8000/v1')}")

    if args.dry_run:
        for job in all_jobs:
            print(" ", " ".join(build_cmd(*job)))
        return

    os.makedirs(args.exp_dir, exist_ok=True)

    # The manifest carries what the directory tree deliberately does not:
    # the parameters held fixed across every cell of this run.
    manifest = {
        "run_id": RUN_ID,
        "model": MODEL,
        "num_agents": 50,
        "K": 5,
        "temperature": 0.0,
        "scorer": "roberta (fine-tuned stance regressor, BWS-calibrated)",
        "field_order": "cot_first",
        "guided_decoding": False,
        "stopping": "dynamic; C_out plateau, W=10, eps_C=0.01, "
                    "patience=5, post_window=10, T_max=120",
        "topics": args.topics,
        "networks": args.networks,
        "alphas": args.alphas,
        "seeds": args.seeds,
        "total_runs": len(all_jobs),
        "layout": "{topic}/{network}/alpha_{a:.3f}/seed_{s:02d}/",
    }
    try:
        manifest["commit"] = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(PROJ_ROOT),
            text=True).strip()
    except Exception:
        manifest["commit"] = None
    with open(os.path.join(EXP_ROOT, RUN_ID, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    # Verify vLLM is reachable before starting
    import urllib.request
    url = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1") + "/models"
    try:
        urllib.request.urlopen(url, timeout=5)
        print("\nvLLM server: reachable ✓")
    except Exception as e:
        print(f"\n[ERROR] Cannot reach vLLM at {url}")
        print(f"        Start it first: bash scripts/start_vllm.sh")
        sys.exit(1)

    results = []
    t_start = time.time()

    # Run LLM-heavy jobs with limited parallelism
    ABORT_AFTER = 3   # consecutive failures that mean "endpoint down"
    aborted = False

    if llm_jobs:
        print(f"\n── LLM jobs ({len(llm_jobs)}) ──")
        consecutive = 0
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(run_one, j): j for j in llm_jobs}
            for f in as_completed(futures):
                r = f.result()
                results.append(r)
                consecutive = 0 if r["ok"] else consecutive + 1
                if consecutive >= ABORT_AFTER:
                    print(f"\n[ABORT] {consecutive} consecutive failures — the LLM\n        endpoint is down, not these parameter combinations. Stopping so the\n        supervisor can restart vLLM; completed runs are kept and resumed.",
                          flush=True)
                    for pending in futures:
                        pending.cancel()
                    aborted = True
                    break

    # Run fast (alpha=1) jobs with higher parallelism
    if fast_jobs and not aborted:
        print(f"\n── Fast jobs ({len(fast_jobs)}) ──")
        with ProcessPoolExecutor(max_workers=FAST_WORKERS) as pool:
            futures = {pool.submit(run_one, j): j for j in fast_jobs}
            for f in as_completed(futures):
                results.append(f.result())

    # Summary
    elapsed = time.time() - t_start
    if aborted:
        print(f"\nAborted after {elapsed/60:.1f} min — endpoint down.")
        sys.exit(2)
    ok_count  = sum(1 for r in results if r["ok"])
    err_count = len(results) - ok_count
    print(f"\n{'='*55}")
    print(f"Done in {elapsed/60:.1f} min  |  OK={ok_count}  ERR={err_count}")
    if err_count:
        print("Failed jobs:")
        for r in results:
            if not r["ok"]:
                t, a, s, n, _ = r["job"]
                print(f"  {t}  {n}  alpha={a}  seed={s}")
        print("="*55)
        print("[SKIP] Errors detected — old data/figures kept intact.")
        sys.exit(1)

    print("="*55)
    if args.no_rebuild:
        print("[SKIP] --no-rebuild: run plots/build_summary.py when the "
              "sweep finishes (it handles the network dimension and "
              "variable-length runs; the plot scripts do not yet).")
        return
    _cleanup_and_rebuild()


if __name__ == "__main__":
    main()
