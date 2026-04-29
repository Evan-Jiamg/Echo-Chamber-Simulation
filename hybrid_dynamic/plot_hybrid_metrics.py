import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

ALPHA_LIST  = [0.0, 0.25, 0.5, 0.75, 1.0]
ALPHA_COLORS = {
    0.0:  "tab:red",
    0.25: "tab:orange",
    0.5:  "tab:green",
    0.75: "tab:blue",
    1.0:  "tab:gray",
}
ALPHA_LABELS = {a: f"α={a} ({'pure LLM' if a==0.0 else 'pure Numeric' if a==1.0 else f'LLM={round((1-a)*50)}'})" for a in ALPHA_LIST}

BASE     = os.path.join(os.path.dirname(__file__), '..')
HYB_DIR  = os.path.join(BASE, 'experiments_hybrid')
NUM_DIR  = os.path.join(BASE, 'experiments_dynamic')
OUT_DIR  = os.path.join(BASE, 'Opinion Dynamic', 'Hybrid')
os.makedirs(OUT_DIR, exist_ok=True)


def load_hybrid(alpha, network_type='scale_free', K=5, sigma=0.5, seed=50, num_agents=50):
    exp = f"agents_{num_agents}_K_{K}_alpha_{alpha}_sigma_{sigma}_seed_{seed}"
    path = os.path.join(HYB_DIR, network_type, 'euthanasia', exp, 'model_overview.json')
    return _parse(path)


def load_numeric_baseline(network_type='scale_free', K=5, sigma=0.5, seed=50, num_agents=50,
                           update_method='degroot'):
    exp = f"agents_{num_agents}_{update_method}_K_{K}_sigma_{sigma}_seed_{seed}"
    path = os.path.join(NUM_DIR, network_type, 'euthanasia', exp, 'model_overview.json')
    return _parse(path)


def _parse(path):
    if not os.path.exists(path):
        return None
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records if records else None


def plot_alpha_comparison(network_type='scale_free', K=5):
    metrics = [
        ('poa',          'Price of Anarchy (PoA)',   'PoA'),
        ('polarization', 'Polarization $P_z$',       '$P_z$'),
        ('modularity',   'Modularity $Q$',            '$Q$'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Hybrid Dynamic — Scale-Free, K={K}\n(alpha: 0=pure LLM → 1=pure Numeric)',
                 fontsize=13, y=1.02)

    for col, (metric, title, ylabel) in enumerate(metrics):
        ax = axes[col]
        for alpha in ALPHA_LIST:
            if alpha == 1.0:
                records = load_numeric_baseline(network_type, K)
            else:
                records = load_hybrid(alpha, network_type, K)
            if records is None:
                print(f"  [missing] alpha={alpha}")
                continue
            steps  = [r['step']            for r in records]
            values = [r.get(metric, float('nan')) for r in records]
            ax.plot(steps, values,
                    label=ALPHA_LABELS[alpha],
                    color=ALPHA_COLORS[alpha],
                    linewidth=2.2,
                    linestyle='--' if alpha == 1.0 else '-')

        ax.set_title(title, fontsize=11)
        ax.set_xlabel('Step')
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        if metric == 'poa':
            ax.axhline(1.0, color='black', linestyle=':', linewidth=1, alpha=0.7, label='PoA=1')

    plt.tight_layout()
    out = os.path.join(OUT_DIR, f'hybrid_alpha_comparison_{network_type}_K{K}.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')


def plot_final_bar(network_type='scale_free', K=5):
    metrics = [
        ('poa',          'Final PoA',          'PoA'),
        ('polarization', 'Final Polarization', '$P_z$'),
        ('modularity',   'Final Modularity',   '$Q$'),
    ]

    x = np.arange(len(ALPHA_LIST))
    width = 0.55
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Hybrid Dynamic — Final Values at Step 30\nScale-Free, K={K}', fontsize=13, y=1.02)

    for col, (metric, title, ylabel) in enumerate(metrics):
        ax = axes[col]
        vals, colors = [], []
        for alpha in ALPHA_LIST:
            if alpha == 1.0:
                records = load_numeric_baseline(network_type, K)
            else:
                records = load_hybrid(alpha, network_type, K)
            if records:
                vals.append(records[-1].get(metric, float('nan')))
            else:
                vals.append(float('nan'))
            colors.append(ALPHA_COLORS[alpha])

        bars = ax.bar(x, vals, width, color=colors, edgecolor='white', linewidth=0.5)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels([f'α={a}' for a in ALPHA_LIST], fontsize=9)
        ax.grid(True, axis='y', alpha=0.3)
        if metric == 'poa':
            ax.axhline(1.0, color='black', linestyle='--', linewidth=1, alpha=0.7)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.01,
                        f'{v:.3f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, f'hybrid_final_bar_{network_type}_K{K}.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')


def plot_poa_heatmap_placeholder(network_type='scale_free', K=5):
    """PoA over steps × alpha — shows convergence speed"""
    n_alpha = len(ALPHA_LIST)
    steps_data = []
    max_steps = 0

    for alpha in ALPHA_LIST:
        if alpha == 1.0:
            records = load_numeric_baseline(network_type, K)
        else:
            records = load_hybrid(alpha, network_type, K)
        if records:
            poas = [r.get('poa', float('nan')) for r in records]
            steps_data.append(poas)
            max_steps = max(max_steps, len(poas))
        else:
            steps_data.append([float('nan')])

    for i, row in enumerate(steps_data):
        while len(row) < max_steps:
            row.append(float('nan'))

    matrix = np.array(steps_data)

    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(matrix, aspect='auto', cmap='RdYlGn_r', vmin=0, vmax=2)
    ax.set_yticks(range(n_alpha))
    ax.set_yticklabels([f'α={a}' for a in ALPHA_LIST])
    ax.set_xlabel('Step')
    ax.set_title(f'PoA Heatmap — Hybrid Dynamic, Scale-Free, K={K}\n(Green=low PoA=better)', fontsize=12)
    plt.colorbar(im, ax=ax, label='PoA')
    plt.tight_layout()
    out = os.path.join(OUT_DIR, f'hybrid_poa_heatmap_{network_type}_K{K}.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')


if __name__ == '__main__':
    print('Generating hybrid comparison charts...')
    plot_alpha_comparison()
    plot_final_bar()
    plot_poa_heatmap_placeholder()
    print('Done.')
