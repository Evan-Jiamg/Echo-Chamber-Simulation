import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import json
import random
import numpy as np
from tqdm import tqdm
import networkx as nx
import community as community_louvain

from agent import SocialAgent   # 原始 agent，完全不修改
from utils import update_day    # 原始 GPT opinion 更新函式


def _load_network(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    G = nx.Graph()
    G.add_nodes_from(data["nodes"])
    G.add_edges_from(data["edges"])
    return G


class DynamicLLMWorld:
    """
    Dynamic K-NN LLM 社會網路模型。

    與 numeric_sim/DynamicWorld 設計相同，差異：
    - Agent 使用 SocialAgent（GPT 驅動），belief 為整數 {-2,-1,0,1,2}
    - sigma 預設 1.0（belief 尺度為 numeric 的 2 倍）
    - 不使用 mesa.space.NetworkGrid；鄰居由 self.neighbors 手動管理
    - 輸出格式與 numeric 完全相同，可直接用 plot_metrics.py 分析
    """

    def __init__(self, network_type, K, sigma=1.0, num_agents=50, seed=50,
                 topic="euthanasia", gpt_model="gpt-4o-mini-2024-07-18",
                 belief_keywords_file=None, exp_dir=None,
                 leaders=None, temp=0.5, with_long_memory=True):

        if leaders is None:
            leaders = [10, 30]
        if belief_keywords_file is None:
            belief_keywords_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'belief_keywords.json')
        if exp_dir is None:
            exp_dir = os.path.join(os.path.dirname(__file__), '..', 'experiments_dynamic_llm')

        self.network_type = network_type
        self.K = K
        self.sigma = sigma
        self.num_agents = num_agents
        self.leaders = leaders
        self.gpt_model = gpt_model
        self.temp = temp
        self.with_long_memory = with_long_memory
        self.current_step = 0

        exp_name = f"agents_{num_agents}_K_{K}_sigma_{sigma}_seed_{seed}"
        self.run_dir = os.path.join(exp_dir, network_type, topic, exp_name)
        os.makedirs(self.run_dir, exist_ok=True)

        # Topic 描述字串
        opinions_file = os.path.join(os.path.dirname(__file__), '..', 'opinions.json')
        with open(opinions_file, 'r') as f:
            self.topic_str = json.load(f)[topic]

        # Belief keywords
        with open(belief_keywords_file, 'r') as f:
            self.belief_keywords = json.load(f)

        # 初始 belief（LLM 用 int×2）與 stubbornness
        data_file = os.path.join(
            os.path.dirname(__file__), '..',
            'data', f'numeric_sim_opnions_and_stubbornness_num_agents_{num_agents}.json'
        )
        with open(data_file, 'r') as f:
            raw = json.load(f)
        self.initial_beliefs = {int(k): int(float(v) * 2) for k, v in raw['opinions'].items()}
        self.stubbornness = {int(k): float(v) for k, v in raw['stubbornness'].items()}

        # 靜態初始網路（計算 optimal cost 與初始鄰居用）
        network_file = os.path.join(
            os.path.dirname(__file__), '..',
            'data', f'{network_type}_network_num_agents_{num_agents}_seed_{seed}.json'
        )
        self.G_static = _load_network(network_file)

        # Agent backgrounds
        gpt_label = "gpt-4o-mini" if gpt_model == "gpt-4o-mini-2024-07-18" else gpt_model
        backgrounds_file = os.path.join(
            os.path.dirname(__file__), '..',
            'data', f'agents_backgrounds_num_agents_{num_agents}_{topic}_{gpt_label}.json'
        )
        if not os.path.exists(backgrounds_file):
            raise FileNotFoundError(
                f"Agent backgrounds not found: {backgrounds_file}\n"
                "Please run the original main.py once to generate agent backgrounds."
            )
        with open(backgrounds_file, 'r') as f:
            self.backgrounds = json.load(f)["backgrounds"]

        # 建立 SocialAgent 列表
        self.agents_list = []
        for i in tqdm(range(num_agents), desc="Creating agents"):
            agent = self._create_agent(i)
            self.agents_list.append(agent)

        # 初始鄰居 = 靜態網路鄰居
        self.neighbors = {
            i: list(self.G_static.neighbors(i))
            for i in range(num_agents)
        }

        # DiGraph：每步 K-NN 更新
        self.G = nx.DiGraph()
        self.G.add_nodes_from(range(num_agents))
        for i, nbrs in self.neighbors.items():
            for j in nbrs:
                self.G.add_edge(i, j)

        # Optimal cost（靜態 NE，固定基準）
        self.optimal_cost = self._compute_optimal_cost()

        self.edges_log = []

    # ── Agent 建立 ─────────────────────────────────────────────────────────
    def _create_agent(self, i):
        bg = self.backgrounds[str(i)]
        system_prompt = bg.get("system_prompt", "You are a helpful assistant")

        # Leaders 系統提示強化（同原始 main.py 邏輯）
        initial_opinion = bg.get("initial_opinion")
        if i in self.leaders:
            system_prompt += (
                f" You are an information distributor, you must firmly hold to your own opinion"
                f" {initial_opinion} and refrain from adopting the views of others."
            )

        agent = SocialAgent(
            model=self,
            unique_id=i,
            name=bg["name"],
            gender=bg["gender"],
            age=bg["age"],
            traits=bg["traits"],
            qualification=bg["education level"],
            initial_belief=self.initial_beliefs[i],
            topic=self.topic_str,
            belief_keywords=self.belief_keywords,
            gpt_model=self.gpt_model,
            temp=self.temp,
            initial_opinion=initial_opinion,
            initial_reasoning=bg.get("initial_reasoning"),
            system_prompt=system_prompt,
            with_long_memory=self.with_long_memory,
        )
        return agent

    # ── Gaussian 相似度 ─────────────────────────────────────────────────────
    def gaussian_g(self, O_i, O_j):
        return float(np.exp(-(O_i - O_j) ** 2 / (2 * self.sigma ** 2)))

    # ── Optimal Cost（靜態 NE，一次計算固定） ───────────────────────────────
    def _compute_optimal_cost(self):
        n = self.num_agents
        nodes = list(range(n))
        L = nx.laplacian_matrix(self.G_static, nodelist=nodes).toarray().astype(float)
        rho = np.array([self.stubbornness[i] for i in nodes])
        s = np.array([self.initial_beliefs[i] for i in nodes], dtype=float)
        W = self.K * np.diag(rho)
        z_star = np.linalg.solve(L + W, W @ s)
        total = 0.0
        for i in nodes:
            for j in self.G_static.neighbors(i):
                total += (z_star[i] - z_star[j]) ** 2
            total += rho[i] * self.K * (z_star[i] - s[i]) ** 2
        return float(total)

    # ── K-NN 重連 ────────────────────────────────────────────────────────────
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

    # ── 設定本輪互動對象（K-NN 鄰居） ────────────────────────────────────────
    def _set_agent_interactions(self):
        for i in range(self.num_agents):
            self.agents_list[i].agent_interaction = [
                self.agents_list[j] for j in self.neighbors[i]
            ]

    # ── 指標計算 ─────────────────────────────────────────────────────────────
    def compute_total_social_cost(self):
        total = 0.0
        for i in range(self.num_agents):
            zi = float(self.agents_list[i].belief)
            si = float(self.initial_beliefs[i])
            rho_i = self.stubbornness[i]
            nbr_beliefs = [float(self.agents_list[j].belief) for j in self.neighbors[i]]
            total += sum((zi - b) ** 2 for b in nbr_beliefs)
            total += rho_i * self.K * (zi - si) ** 2
        return float(total)

    def compute_polarization(self):
        z = np.array([a.belief / 2.0 for a in self.agents_list], dtype=float)
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

    # ── 主迴圈 ────────────────────────────────────────────────────────────────
    def step(self):
        # 1. 設定本輪 K-NN 互動對象
        self._set_agent_interactions()

        # 2. 每個 agent 互動：讀取鄰居意見，更新短/長期記憶（隨機順序）
        shuffled = list(self.agents_list)
        random.shuffle(shuffled)
        for agent in shuffled:
            agent.step()   # → agent.interact()

        # 3. GPT 更新：根據記憶體輸出新 opinion & belief
        for agent in self.agents_list:
            update_day(agent)

        # 4. K-NN 重連（基於新 belief）
        self._update_knn()
        self.edges_log.append([list(e) for e in self.G.edges()])

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
            str(a.unique_id): {
                "beliefs": a.beliefs,
                "opinions": a.opinions,
                "reasonings": a.reasonings,
                "short_memory": a.short_memory_full,
                "long_memory": a.long_memory_full,
            }
            for a in self.agents_list
        }
        with open(os.path.join(self.run_dir, "agents_data.json"), 'w') as f:
            json.dump(data, f, indent=4)

    def save_interaction_data(self):
        data = {
            str(a.unique_id): {
                "beliefs": a.beliefs,
                "opinions": a.opinions,
            }
            for a in self.agents_list
        }
        with open(os.path.join(self.run_dir, "agents_interaction_data.json"), 'w') as f:
            json.dump(data, f, indent=4)

    def save_edges_data(self):
        with open(os.path.join(self.run_dir, "edges_per_step.json"), 'w') as f:
            json.dump(self.edges_log, f)

    def run_model(self, step_count):
        open(os.path.join(self.run_dir, "model_overview.json"), 'w').close()
        for _ in tqdm(range(step_count), desc=f"Dynamic LLM {self.network_type} K={self.K}"):
            self.step()
        self.save_agents_data()
        self.save_interaction_data()
        self.save_edges_data()
        final_cost = self.compute_total_social_cost()
        print(f"\nOptimal cost (NE on static): {self.optimal_cost:.4f}")
        print(f"Final social cost (K-NN):    {final_cost:.4f}")
        print(f"Final PoA:                   {final_cost / self.optimal_cost:.4f}")
        print(f"Saved to: {self.run_dir}")
