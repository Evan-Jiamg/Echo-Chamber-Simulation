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


def load_network(network_file):
    with open(network_file, 'r') as f:
        data = json.load(f)
    G = nx.Graph()
    G.add_nodes_from(data["nodes"])
    G.add_edges_from(data["edges"])
    return G


def extract_beliefs(agents_interaction_file):
    with open(agents_interaction_file, 'r') as f:
        agent_data = json.load(f)
    time_steps = len(next(iter(agent_data.values()))["beliefs"])
    beliefs_at_steps = {}
    for step in range(time_steps):
        beliefs_at_steps[step] = {
            agent_id: info["beliefs"][step]
            for agent_id, info in agent_data.items()
        }
    return beliefs_at_steps


def update_frame(frame, G, pos, node_size, ax, beliefs_at_steps, time_step_text, norm, cmap):
    ax.clear()
    beliefs = beliefs_at_steps[frame]
    colors = [cmap(norm(beliefs[str(node)])) for node in G.nodes()]
    nx.draw(G, pos, node_color=colors, node_size=node_size,
            edge_color='gray', with_labels=False, ax=ax)
    time_step_text = ax.text(0.05, 0.95, f"Step {frame + 1}",
                             transform=ax.transAxes, fontsize=14,
                             verticalalignment='top')


def generate_gif(network_file, agents_interaction_file, output_file, network_type, fps=1):
    G = load_network(network_file)
    beliefs_at_steps = extract_beliefs(agents_interaction_file)
    total_steps = len(beliefs_at_steps)

    if network_type == 'small_world':
        pos = nx.shell_layout(G)
    else:
        pos = nx.kamada_kawai_layout(G)

    node_size = [G.degree[n] * 80 for n in G.nodes()]

    # 連續 colormap，適用浮點 belief（-1 ~ 1）
    cmap = cm.coolwarm
    norm = mcolors.Normalize(vmin=-1.0, vmax=1.0)

    fig, ax = plt.subplots(figsize=(8, 8))
    time_step_text = ax.text(0.05, 0.95, '', transform=ax.transAxes,
                             fontsize=14, verticalalignment='top')

    ani = FuncAnimation(
        fig, update_frame, frames=total_steps,
        fargs=(G, pos, node_size, ax, beliefs_at_steps, time_step_text, norm, cmap),
        interval=1000 / fps
    )

    ani.save(output_file, writer='pillow', fps=fps)
    plt.close()
    print(f"GIF saved: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_all", action="store_true", help="為所有實驗生成 GIF")
    parser.add_argument("--network_type", default="scale_free",
                        choices=["small_world", "scale_free", "random"])
    parser.add_argument("--update_method", default="degroot",
                        choices=["degroot", "fj", "bcm"])
    parser.add_argument("--num_agents", default=50, type=int)
    parser.add_argument("--seed", default=50, type=int)
    parser.add_argument("--epsilon", default=0.5, type=float)
    parser.add_argument("--fps", default=2, type=int)
    args = parser.parse_args()

    exp_base = os.path.join(os.path.dirname(__file__), '..', 'experiments_numeric')
    data_base = os.path.join(os.path.dirname(__file__), '..', 'data')

    def run_single(network_type, update_method):
        exp_name = f"agents_{args.num_agents}_{update_method}_seed_{args.seed}"
        if update_method == 'bcm':
            exp_name += f"_epsilon_{args.epsilon}"

        run_dir = os.path.join(exp_base, network_type, 'euthanasia', exp_name)
        network_file = os.path.join(data_base, f'{network_type}_network_num_agents_{args.num_agents}_seed_{args.seed}.json')
        interaction_file = os.path.join(run_dir, 'agents_interaction_data.json')
        output_file = os.path.join(run_dir, 'opinion_dynamics.gif')

        if not os.path.exists(interaction_file):
            print(f"Skip (no data): {run_dir}")
            return

        print(f"Generating GIF: {network_type} | {update_method}")
        generate_gif(network_file, interaction_file, output_file, network_type, fps=args.fps)

    if args.run_all:
        for network_type in ["scale_free", "small_world", "random"]:
            for update_method in ["degroot", "fj", "bcm"]:
                run_single(network_type, update_method)
    else:
        run_single(args.network_type, args.update_method)
