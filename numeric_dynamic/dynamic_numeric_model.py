import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import mesa
import json
import numpy as np
from tqdm import tqdm
import networkx as nx
import community as community_louvain

from dynamic_numeric_agent import DynamicAgent


def _load_network(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    G = nx.Graph()
    G.add_nodes_from(data["nodes"])
    G.add_edges_from(data["edges"])
    return G


class DynamicWorld(mesa.Model):
    def __init__(self, network_type, K, sigma=0.5, num_agents=50, seed=50,
                 exp_dir='../experiments_dynamic',
                 update_method='degroot', epsilon=0.3):
        super().__init__()

        self.network_type = network_type
        self.K = K
        self.sigma = sigma
        self.num_agents = num_agents
        self.update_method = update_method
        self.epsilon = epsilon
        self.current_step = 0

        if update_method == 'bcm':
            exp_name = f"agents_{num_agents}_{update_method}_epsilon_{epsilon}_K_{K}_sigma_{sigma}_seed_{seed}"
        else:
            exp_name = f"agents_{num_agents}_{update_method}_K_{K}_sigma_{sigma}_seed_{seed}"
        self.run_dir = os.path.join(exp_dir, network_type, 'euthanasia', exp_name)
        os.makedirs(self.run_dir, exist_ok=True)

        # 載入初始 belief 與 stubbornness
        belief_file = os.path.join(
            os.path.dirname(__file__), '..',
            'data', f'numeric_sim_opnions_and_stubbornness_num_agents_{num_agents}.json'
        )
        with open(belief_file, 'r') as f:
            data = json.load(f)

        self.agents_list = []
        for i in range(num_agents):
            agent = DynamicAgent(
                unique_id=i,
                model=self,
                initial_belief=float(data['opinions'][str(i)]),
                stubbornness=float(data['stubbornness'][str(i)])
            )
            self.agents_list.append(agent)

        # 載入靜態初始網路（提供初始意見多樣性）
        network_file = os.path.join(
            os.path.dirname(__file__), '..',
            'data', f'{network_type}_network_num_agents_{num_agents}_seed_{seed}.json'
        )
        self.G_static = _load_network(network_file)

        # 初始鄰居 = 靜態網路鄰居（Step 0 使用，確保初始社會成本 > 0）
        self.neighbors = {
            i: list(self.G_static.neighbors(i))
            for i in range(num_agents)
        }

        # DiGraph：隨 K-NN 每步更新
        self.G = nx.DiGraph()
        self.G.add_nodes_from(range(num_agents))
        for i, nbrs in self.neighbors.items():
            for j in nbrs:
                self.G.add_edge(i, j)

        # 每步 K-NN 邊紀錄（frame 0 = step 1 的邊）
        self.edges_log = []

        # Optimal cost：在靜態初始網路上解 Nash Equilibrium 線性系統，固定為 PoA 分母
        # 系統：(L_static + K·diag(ρ)) z* = K·diag(ρ)·s
        self.optimal_cost = self._compute_optimal_cost()

    # ── Gaussian 相似度函數 g(O_i, O_j) ∈ [0, 1] ─────────────────────────
    def gaussian_g(self, O_i, O_j):
        return float(np.exp(-(O_i - O_j) ** 2 / (2 * self.sigma ** 2)))

    # ── Optimal Cost：靜態網路 NE 解，作為 PoA 固定基準 ──────────────────
    def _compute_optimal_cost(self):
        n = self.num_agents
        nodes = list(range(n))

        L = nx.laplacian_matrix(self.G_static, nodelist=nodes).toarray().astype(float)
        rho = np.array([self.agents_list[i].stubbornness for i in nodes])
        s = np.array([self.agents_list[i].initial_belief for i in nodes])

        W = self.K * np.diag(rho)
        z_star = np.linalg.solve(L + W, W @ s)

        total = 0.0
        for i in nodes:
            for j in self.neighbors[i]:              # 靜態網路鄰居
                total += (z_star[i] - z_star[j]) ** 2
            total += rho[i] * self.K * (z_star[i] - s[i]) ** 2
        return float(total)

    # ── K-NN Rewiring：每步依當前 belief 重選 K 個最相似的 out-neighbors ──
    def _update_knn(self):
        for i in range(self.num_agents):
            O_i = self.agents_list[i].belief
            scores = sorted(
                [(j, self.gaussian_g(O_i, self.agents_list[j].belief))
                 for j in range(self.num_agents) if j != i],
                key=lambda x: x[1], reverse=True
            )
            self.neighbors[i] = [j for j, _ in scores[:self.K]]

        self.G.remove_edges_from(list(self.G.edges()))
        for i, nbrs in self.neighbors.items():
            for j in nbrs:
                self.G.add_edge(i, j)

    # ── 同步更新（snapshot 確保所有人同步讀 t 時刻的意見） ──────────────────
    def _update_opinions(self):
        snapshot = {i: self.agents_list[i].belief for i in range(self.num_agents)}
        for i in range(self.num_agents):
            neighbor_beliefs = [snapshot[j] for j in self.neighbors[i]]
            agent = self.agents_list[i]
            if self.update_method == 'fj':
                agent.step_fj(neighbor_beliefs)
            elif self.update_method == 'bcm':
                agent.step_bcm(neighbor_beliefs, self.epsilon)
            else:
                agent.step_degroot(neighbor_beliefs)

    # ── 指標計算 ──────────────────────────────────────────────────────────
    def compute_total_social_cost(self):
        total = 0.0
        for i in range(self.num_agents):
            nbr_beliefs = [self.agents_list[j].belief for j in self.neighbors[i]]
            total += self.agents_list[i].compute_social_cost(nbr_beliefs, self.K)
        return float(total)

    def compute_polarization(self):
        z = np.array([a.belief for a in self.agents_list], dtype=float)
        return float(np.mean((z - z.mean()) ** 2))

    def compute_modularity(self):
        G_u = self.G.to_undirected()
        if G_u.number_of_edges() == 0:
            return 0.0
        partition = community_louvain.best_partition(G_u)
        return float(community_louvain.modularity(partition, G_u))

    def compute_echo_chamber_effect(self):
        total_g = sum(
            self.gaussian_g(self.agents_list[i].belief, self.agents_list[j].belief)
            for i in range(self.num_agents)
            for j in self.neighbors[i]
        )
        count = sum(len(nbrs) for nbrs in self.neighbors.values())
        return float(total_g / count) if count > 0 else 0.0

    # ── 主迴圈 ────────────────────────────────────────────────────────────
    def step(self):
        self._update_opinions()   # 1. 同步 DeGroot 更新
        self._update_knn()        # 2. K-NN 重連
        self.edges_log.append([list(e) for e in self.G.edges()])
        for agent in self.agents_list:
            agent.record()
        self.current_step += 1
        self.save_model_data()

    def save_model_data(self):
        cost = self.compute_total_social_cost()
        poa = cost / self.optimal_cost if self.optimal_cost > 0 else 1.0
        data = {
            "step": self.current_step,
            "polarization": self.compute_polarization(),
            "modularity": self.compute_modularity(),
            "echo_chamber_effect": self.compute_echo_chamber_effect(),
            "social_cost": cost,
            "poa": poa,
        }
        with open(os.path.join(self.run_dir, "model_overview.json"), 'a') as f:
            json.dump(data, f)
            f.write('\n')

    def save_agents_data(self):
        data = {
            a.unique_id: {"beliefs": a.beliefs, "opinions": a.opinions}
            for a in self.agents_list
        }
        with open(os.path.join(self.run_dir, "agents_data.json"), 'w') as f:
            json.dump(data, f, indent=4)

    def save_interaction_data(self):
        data = {
            str(a.unique_id): {"beliefs": a.beliefs, "opinions": a.opinions}
            for a in self.agents_list
        }
        with open(os.path.join(self.run_dir, "agents_interaction_data.json"), 'w') as f:
            json.dump(data, f, indent=4)

    def save_edges_data(self):
        with open(os.path.join(self.run_dir, "edges_per_step.json"), 'w') as f:
            json.dump(self.edges_log, f)

    def run_model(self, step_count):
        open(os.path.join(self.run_dir, "model_overview.json"), 'w').close()
        for _ in tqdm(range(step_count), desc=f"Dynamic {self.network_type} K={self.K}"):
            self.step()
        self.save_agents_data()
        self.save_interaction_data()
        self.save_edges_data()
        final_cost = self.compute_total_social_cost()
        print(f"\nOptimal cost (NE on static): {self.optimal_cost:.4f}")
        print(f"Final social cost (K-NN):    {final_cost:.4f}")
        print(f"Final PoA:                   {final_cost / self.optimal_cost:.4f}")
        print(f"Saved to: {self.run_dir}")
