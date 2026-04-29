import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import json
import argparse
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.cm as cm
import matplotlib.colors as mcolors


def load_static_network(network_file):
    with open(network_file, 'r') as f:
        data = json.load(f)
    G = nx.Graph()
    G.add_nodes_from(data["nodes"])
    G.add_edges_from(data["edges"])
    return G


def load_beliefs(agents_interaction_file):
    with open(agents_interaction_file, 'r') as f:
        agent_data = json.load(f)
    time_steps = len(next(iter(agent_data.values()))["beliefs"])
    # index 0 = initial, skip; index 1..end = frame 0..end-1
    beliefs_at_steps = {}
    for step in range(1, time_steps):
        beliefs_at_steps[step - 1] = {
            agent_id: info["beliefs"][step]
            for agent_id, info in agent_data.items()
        }
    return beliefs_at_steps


def load_edges(edges_file):
    with open(edges_file, 'r') as f:
        return json.load(f)


def update_frame(frame, pos, num_nodes, ax, beliefs_at_steps, edges_per_step, norm, cmap):
    ax.clear()

    beliefs = beliefs_at_steps[frame]
    edges = edges_per_step[frame]

    G = nx.DiGraph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(edges)

    in_deg = dict(G.in_degree())
    node_sizes = [max(60, in_deg.get(n, 0) * 70) for n in range(num_nodes)]
    colors = [cmap(norm(beliefs[str(n)])) for n in range(num_nodes)]

    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=node_sizes, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color='black', arrows=True,
                           arrowsize=8, alpha=0.35, ax=ax,
                           connectionstyle='arc3,rad=0.1')
    ax.set_title(f"Step {frame + 1}", fontsize=14, pad=8)
    ax.axis('off')


def generate_gif(run_dir, network_file, network_type, fps=2):
    interaction_file = os.path.join(run_dir, 'agents_interaction_data.json')
    edges_file = os.path.join(run_dir, 'edges_per_step.json')
    output_file = os.path.join(run_dir, 'opinion_dynamics.gif')

    if not os.path.exists(interaction_file):
        print(f"Skip (no agents_interaction_data.json): {run_dir}")
        return
    if not os.path.exists(edges_file):
        print(f"Skip (no edges_per_step.json): {run_dir}")
        return

    beliefs_at_steps = load_beliefs(interaction_file)
    edges_per_step = load_edges(edges_file)
    total_steps = min(len(beliefs_at_steps), len(edges_per_step))
    num_nodes = len(beliefs_at_steps[0])

    G_init = load_static_network(network_file)
    if network_type == 'small_world':
        pos = nx.shell_layout(G_init)
    else:
        pos = nx.kamada_kawai_layout(G_init)

    cmap = cm.coolwarm
    norm = mcolors.Normalize(vmin=-1.0, vmax=1.0)

    fig, ax = plt.subplots(figsize=(8, 8))

    ani = FuncAnimation(
        fig, update_frame, frames=total_steps,
        fargs=(pos, num_nodes, ax, beliefs_at_steps, edges_per_step, norm, cmap),
        interval=1000 / fps
    )
    ani.save(output_file, writer='pillow', fps=fps)
    plt.close()
    print(f"GIF saved: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_all", action="store_true",
                        help="為所有 network_type × K 組合生成 GIF")
    parser.add_argument("--network_type", default="scale_free",
                        choices=["small_world", "scale_free", "random"])
    parser.add_argument("--K", default=5, type=int)
    parser.add_argument("--sigma", default=0.5, type=float)
    parser.add_argument("--num_agents", default=50, type=int)
    parser.add_argument("--seed", default=50, type=int)
    parser.add_argument("--fps", default=2, type=int)
    args = parser.parse_args()

    exp_base = os.path.join(os.path.dirname(__file__), '..', 'experiments_dynamic')
    data_base = os.path.join(os.path.dirname(__file__), '..', 'data')

    def run_single(network_type, K):
        exp_name = f"agents_{args.num_agents}_K_{K}_sigma_{args.sigma}_seed_{args.seed}"
        run_dir = os.path.join(exp_base, network_type, 'euthanasia', exp_name)
        network_file = os.path.join(
            data_base, f'{network_type}_network_num_agents_{args.num_agents}_seed_{args.seed}.json'
        )
        if not os.path.exists(network_file):
            print(f"Skip (no network file): {network_file}")
            return
        print(f"Generating GIF: {network_type} | K={K}")
        generate_gif(run_dir, network_file, network_type, fps=args.fps)

    if args.run_all:
        for network_type in ["scale_free", "small_world", "random"]:
            for K in [3, 5, 10, 20, 49]:
                run_single(network_type, K)
    else:
        run_single(args.network_type, args.K)
