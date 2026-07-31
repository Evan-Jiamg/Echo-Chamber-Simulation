"""
Recompute PoA for existing experiment results using the corrected formula:
  PoA(step t) = C(z_t, G_t) / C_opt(G_t)
where both numerator and denominator use the post-KNN-rewiring graph G_t.

Overwrites model_overview.json and metrics.csv in-place.
"""
import os
import json
import csv
import numpy as np
import networkx as nx
import community as community_louvain
from scipy.linalg import solve

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
EXP_DIR  = os.path.join(os.path.dirname(__file__), '..', 'experiments', 'scale_free')
K = 5


def load_stubbornness(topic, num_agents=50):
    topic_file = os.path.join(DATA_DIR, f'numeric_sim_opnions_and_stubbornness_num_agents_{num_agents}_{topic}.json')
    default_file = os.path.join(DATA_DIR, f'numeric_sim_opnions_and_stubbornness_num_agents_{num_agents}.json')
    path = topic_file if os.path.exists(topic_file) else default_file
    with open(path) as f:
        raw = json.load(f)
    opinions     = {int(k): float(v) for k, v in raw['opinions'].items()}
    stubbornness = {int(k): float(v) for k, v in raw['stubbornness'].items()}
    # Both LLM and Numeric agents use continuous Reddit stance as s_i baseline.
    # LLM agents discretize beliefs for prompt output, but s_i should reflect
    # the true intrinsic opinion from the data, not the rounded representation.
    init_s = opinions   # same for both agent types
    return stubbornness, init_s, init_s


def compute_optimal_cost(edges, n, stubbornness, init_s):
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edges)
    A     = nx.to_numpy_array(G, nodelist=list(range(n)))
    D_out = np.diag(A.sum(axis=1))
    D_in  = np.diag(A.sum(axis=0))
    L_sym = D_out + D_in - A - A.T
    rho   = np.array([stubbornness[i] for i in range(n)])
    s     = np.array([init_s[i] for i in range(n)])
    W     = K * np.diag(rho)
    M     = L_sym + W
    if np.linalg.matrix_rank(M) < n:
        M += 1e-8 * np.eye(n)
    z_opt = solve(M, W @ s)
    dz    = z_opt[:, None] - z_opt[None, :]
    return float(np.sum(A * dz**2) + float(np.dot(rho * K, (z_opt - s)**2)))


def compute_social_cost(edges, n, z, stubbornness, init_s):
    neighbors = {i: [] for i in range(n)}
    for u, v in edges:
        neighbors[u].append(v)
    total = 0.0
    for i in range(n):
        zi = z[i]
        si = init_s[i]
        nbr_cost = sum((zi - z[j])**2 for j in neighbors[i])
        stub_cost = stubbornness[i] * K * (zi - si)**2
        total += nbr_cost + stub_cost
    return float(total)


def recompute_exp(exp_dir, topic, num_agents=50):
    stub, init_llm, init_numeric = load_stubbornness(topic, num_agents)

    with open(os.path.join(exp_dir, 'agent_assignment.json')) as f:
        aa = json.load(f)
    llm_ids     = set(aa['llm_agents'])
    numeric_ids = set(aa['numeric_agents'])

    # Initial canonical beliefs per agent
    init_s = {}
    for i in range(num_agents):
        init_s[i] = init_llm[i] if i in llm_ids else init_numeric[i]

    with open(os.path.join(exp_dir, 'agents_data.json')) as f:
        agents_data = json.load(f)

    with open(os.path.join(exp_dir, 'edges_per_step.json')) as f:
        edges_per_step = json.load(f)

    # Read existing model_overview to keep other metrics intact
    overview_path = os.path.join(exp_dir, 'model_overview.json')
    existing = []
    with open(overview_path) as f:
        for line in f:
            line = line.strip()
            if line:
                existing.append(json.loads(line))

    n_steps = len(edges_per_step)
    new_records = []

    for t in range(n_steps):
        step_num = t + 1  # 1-indexed, matching model_overview
        edges = [tuple(e) for e in edges_per_step[t]]

        # Canonical beliefs at step t+1 (after step t's update)
        z = {}
        for agent_id_str, data in agents_data.items():
            i = int(agent_id_str)
            raw_belief = data['beliefs'][step_num]  # beliefs[0]=initial, beliefs[1]=step1 ...
            z[i] = float(raw_belief) / 2.0 if i in llm_ids else float(raw_belief)

        c_opt = compute_optimal_cost(edges, num_agents, stub, init_s)
        c_act = compute_social_cost(edges, num_agents, z, stub, init_s)
        poa   = c_act / c_opt if c_opt > 0 else 1.0

        # Recompute modularity from graph
        G_t = nx.Graph()
        G_t.add_nodes_from(range(num_agents))
        G_t.add_edges_from(edges)
        if G_t.number_of_edges() > 0:
            partition = community_louvain.best_partition(G_t)
            modularity = float(community_louvain.modularity(partition, G_t))
        else:
            modularity = 0.0

        old = existing[t] if t < len(existing) else {}
        rec = {
            "step":         step_num,
            "polarization": old.get("polarization", float('nan')),
            "modularity":   modularity,
            "poa":          poa,
        }
        new_records.append(rec)

    # Overwrite model_overview.json
    with open(overview_path, 'w') as f:
        for rec in new_records:
            json.dump(rec, f)
            f.write('\n')

    # Overwrite metrics.csv
    fields = ["step", "polarization", "modularity", "poa"]
    csv_path = os.path.join(exp_dir, 'metrics.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for rec in new_records:
            writer.writerow({k: rec.get(k, '') for k in fields})

    final_poa = new_records[-1]['poa']
    return final_poa


def run_all():
    topics = ['abortion', 'gun_control']
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    seeds  = list(range(1, 6))

    for topic in topics:
        print(f'\n=== {topic.upper()} ===')
        print(f'{"Alpha":<8} {"Seed":<6} {"Final PoA":>10}')
        for alpha in alphas:
            poas = []
            for seed in seeds:
                exp = f'agents_50_K_5_alpha_{alpha}_sigma_0.5_seed_{seed}'
                exp_dir = os.path.join(EXP_DIR, topic, exp)
                if not os.path.exists(exp_dir):
                    continue
                try:
                    poa = recompute_exp(exp_dir, topic)
                    poas.append(poa)
                    print(f'  α={alpha:<5} seed={seed}  PoA={poa:.4f}')
                except Exception as e:
                    print(f'  α={alpha:<5} seed={seed}  ERROR: {e}')
            if poas:
                print(f'  α={alpha:<5} → mean={np.mean(poas):.4f}  std={np.std(poas):.4f}')


if __name__ == '__main__':
    run_all()
