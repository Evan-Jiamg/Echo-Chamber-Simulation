import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

K_LIST = [3, 5, 10, 20, 49]
NETWORK_TYPES = ["scale_free", "small_world", "random"]
NETWORK_LABELS = {"scale_free": "Scale-Free", "small_world": "Small-World", "random": "Random"}
METHODS = ["degroot", "fj", "bcm"]
METHOD_LABELS = {"degroot": "DeGroot", "fj": "FJ", "bcm": "BCM"}
METHOD_COLORS = {"degroot": "tab:blue", "fj": "tab:orange", "bcm": "tab:green"}


def load_metrics(exp_base, network_type, K, num_agents=50, sigma=0.5, seed=50,
                 update_method='degroot', epsilon=0.3):
    if update_method == 'bcm':
        exp_name = f"agents_{num_agents}_{update_method}_epsilon_{epsilon}_K_{K}_sigma_{sigma}_seed_{seed}"
    else:
        exp_name = f"agents_{num_agents}_{update_method}_K_{K}_sigma_{sigma}_seed_{seed}"
    path = os.path.join(exp_base, network_type, 'euthanasia', exp_name, 'model_overview.json')
    if not os.path.exists(path):
        return None
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def plot_metric_across_K(exp_base, network_type, metric, ax, K_list, num_agents, sigma, seed,
                         update_method='degroot', epsilon=0.3):
    cmap = cm.get_cmap('tab10')
    for idx, K in enumerate(K_list):
        records = load_metrics(exp_base, network_type, K, num_agents, sigma, seed,
                               update_method, epsilon)
        if records is None:
            continue
        steps = [r['step'] for r in records]
        values = [r.get(metric, float('nan')) for r in records]
        ax.plot(steps, values, label=f"K={K}", color=cmap(idx), linewidth=1.8)


def main(args):
    exp_base = os.path.join(os.path.dirname(__file__), '..', 'experiments_dynamic')
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'Opinion Dynamic')
    os.makedirs(out_dir, exist_ok=True)

    metrics_cfg = [
        ("poa",          "Price of Anarchy (PoA)",  "PoA"),
        ("polarization", "Polarization $P_z$",      "$P_z$"),
        ("modularity",   "Modularity $Q$",           "$Q$"),
    ]

    method = args.update_method

    # ── Figure 1: one row per network, one column per metric ────────────────
    fig, axes = plt.subplots(
        nrows=len(NETWORK_TYPES), ncols=len(metrics_cfg),
        figsize=(15, 10), sharey='col'
    )
    fig.suptitle(f"Opinion Dynamics [{METHOD_LABELS[method]}] — PoA, Polarization $P_z$, Modularity $Q$",
                 fontsize=15, y=1.01)

    for row, network_type in enumerate(NETWORK_TYPES):
        for col, (metric, title, ylabel) in enumerate(metrics_cfg):
            ax = axes[row][col]
            plot_metric_across_K(exp_base, network_type, metric, ax,
                                 args.K_list, args.num_agents, args.sigma, args.seed,
                                 method, args.epsilon)
            if row == 0:
                ax.set_title(title, fontsize=11)
            if col == 0:
                ax.set_ylabel(NETWORK_LABELS[network_type], fontsize=11)
            ax.set_xlabel("Step")
            ax.legend(fontsize=7, ncol=2)
            ax.grid(True, alpha=0.3)
            if metric == 'poa':
                ax.axhline(1.0, color='red', linestyle='--', linewidth=0.8, alpha=0.6)

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"metrics_overview_{method}.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")

    # ── Figure 2: PoA final value bar chart (K on x-axis, grouped by network) ──
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    x = np.arange(len(args.K_list))
    width = 0.25
    cmap2 = cm.get_cmap('Set2')

    for i, network_type in enumerate(NETWORK_TYPES):
        final_poas = []
        for K in args.K_list:
            records = load_metrics(exp_base, network_type, K, args.num_agents, args.sigma, args.seed,
                                   method, args.epsilon)
            if records:
                final_poas.append(records[-1]['poa'])
            else:
                final_poas.append(float('nan'))
        ax2.bar(x + i * width, final_poas, width, label=NETWORK_LABELS[network_type], color=cmap2(i))

    ax2.axhline(1.0, color='red', linestyle='--', linewidth=1, alpha=0.7, label='PoA = 1 (optimal)')
    ax2.set_xticks(x + width)
    ax2.set_xticklabels([f"K={K}" for K in args.K_list])
    ax2.set_ylabel("Final PoA (step 30)")
    ax2.set_title(f"Final PoA by K and Network Type [{METHOD_LABELS[method]}]")
    ax2.legend()
    ax2.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    out_path2 = os.path.join(out_dir, f"final_poa_bar_{method}.png")
    plt.savefig(out_path2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path2}")

    # ── Figure 3: per-network line plot (K vs step) for each metric ─────────
    for network_type in NETWORK_TYPES:
        fig3, axes3 = plt.subplots(1, 3, figsize=(14, 4))
        fig3.suptitle(
            f"{NETWORK_LABELS[network_type]} [{METHOD_LABELS[method]}] — PoA, $P_z$, Modularity $Q$ by K",
            fontsize=13)
        for col, (metric, title, ylabel) in enumerate(metrics_cfg):
            ax = axes3[col]
            plot_metric_across_K(exp_base, network_type, metric, ax,
                                 args.K_list, args.num_agents, args.sigma, args.seed,
                                 method, args.epsilon)
            ax.set_title(title, fontsize=11)
            ax.set_xlabel("Step")
            ax.set_ylabel(ylabel)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            if metric == 'poa':
                ax.axhline(1.0, color='red', linestyle='--', linewidth=0.8, alpha=0.6)
        plt.tight_layout()
        out_path3 = os.path.join(out_dir, f"metrics_{network_type}_{method}.png")
        plt.savefig(out_path3, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {out_path3}")

    # ── Figure 4: Method comparison at fixed K (DeGroot vs FJ vs BCM) ───────
    for network_type in NETWORK_TYPES:
        fig4, axes4 = plt.subplots(1, 3, figsize=(14, 4))
        fig4.suptitle(
            f"{NETWORK_LABELS[network_type]} — Method Comparison (K={args.compare_K})",
            fontsize=13)
        for col, (metric, title, ylabel) in enumerate(metrics_cfg):
            ax = axes4[col]
            for m in METHODS:
                records = load_metrics(exp_base, network_type, args.compare_K,
                                       args.num_agents, args.sigma, args.seed,
                                       m, args.epsilon)
                if records is None:
                    continue
                steps = [r['step'] for r in records]
                values = [r.get(metric, float('nan')) for r in records]
                ax.plot(steps, values, label=METHOD_LABELS[m],
                        color=METHOD_COLORS[m], linewidth=2)
            ax.set_title(title, fontsize=11)
            ax.set_xlabel("Step")
            ax.set_ylabel(ylabel)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            if metric == 'poa':
                ax.axhline(1.0, color='red', linestyle='--', linewidth=0.8, alpha=0.6)
        plt.tight_layout()
        out_path4 = os.path.join(out_dir, f"method_comparison_{network_type}_K{args.compare_K}.png")
        plt.savefig(out_path4, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {out_path4}")

    # ── Figure 5: Final PoA bar chart grouped by method, across K ────────────
    for network_type in NETWORK_TYPES:
        fig5, ax5 = plt.subplots(figsize=(10, 5))
        x = np.arange(len(args.K_list))
        width = 0.25
        for mi, m in enumerate(METHODS):
            final_poas = []
            for K in args.K_list:
                records = load_metrics(exp_base, network_type, K,
                                       args.num_agents, args.sigma, args.seed,
                                       m, args.epsilon)
                final_poas.append(records[-1]['poa'] if records else float('nan'))
            ax5.bar(x + mi * width, final_poas, width,
                    label=METHOD_LABELS[m], color=METHOD_COLORS[m])
        ax5.axhline(1.0, color='red', linestyle='--', linewidth=1, alpha=0.7, label='PoA=1')
        ax5.set_xticks(x + width)
        ax5.set_xticklabels([f"K={K}" for K in args.K_list])
        ax5.set_ylabel("Final PoA (step 30)")
        ax5.set_title(f"{NETWORK_LABELS[network_type]} — Final PoA by K and Method")
        ax5.legend()
        ax5.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        out_path5 = os.path.join(out_dir, f"final_poa_methods_{network_type}.png")
        plt.savefig(out_path5, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {out_path5}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_agents", default=50, type=int)
    parser.add_argument("--sigma", default=0.5, type=float)
    parser.add_argument("--seed", default=50, type=int)
    parser.add_argument("--K_list", nargs='+', type=int, default=[3, 5, 10, 20, 49])
    parser.add_argument("--update_method", default="degroot",
                        choices=["degroot", "fj", "bcm"],
                        help="Method for Figures 1-3")
    parser.add_argument("--epsilon", default=0.3, type=float)
    parser.add_argument("--compare_K", default=5, type=int,
                        help="Fixed K used in the method comparison figure (Fig 4)")
    args = parser.parse_args()
    main(args)
