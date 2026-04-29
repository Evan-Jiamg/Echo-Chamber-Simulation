import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'numeric_dynamic'))

import mesa
import json
import random
import numpy as np
from tqdm import tqdm
import networkx as nx
import community as community_louvain

from dynamic_numeric_agent import DynamicAgent   # numeric agent, unchanged
from agent import SocialAgent                    # LLM agent, unchanged
from utils import update_day                     # GPT opinion update, unchanged


def _load_network(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    G = nx.Graph()
    G.add_nodes_from(data["nodes"])
    G.add_edges_from(data["edges"])
    return G


class HybridDynamicWorld(mesa.Model):
    """
    Hybrid Dynamic Network:
      alpha=0 -> pure LLM  (all SocialAgent, GPT-driven)
      alpha=1 -> pure Numeric (all DynamicAgent, DeGroot)
      0 < alpha < 1 -> mixed

    Alpha > 0 requires LLM API key in utils.py and agent backgrounds file.
    Alpha = 1 runs without any API requirement.

    Belief scales:
      Numeric  : float [-1, 1]  (canonical as-is)
      LLM      : int {-2,-1,0,1,2} / 2 = canonical [-1,-0.5,0,0.5,1]
    All metrics use canonical beliefs in [-1, 1].
    """

    def __init__(self, network_type, K, alpha=1.0, sigma=0.5, num_agents=50, seed=50,
                 topic="euthanasia", gpt_model="gpt-4o-mini-2024-07-18",
                 belief_keywords_file=None, exp_dir=None,
                 leaders=None, temp=0.5, with_long_memory=True):
        super().__init__()

        if leaders is None:
            leaders = [10, 30]
        if belief_keywords_file is None:
            belief_keywords_file = os.path.join(
                os.path.dirname(__file__), '..', 'data', 'belief_keywords.json')
        if exp_dir is None:
            exp_dir = os.path.join(
                os.path.dirname(__file__), '..', 'experiments_hybrid')

        self.network_type = network_type
        self.K = K
        self.alpha = alpha
        self.sigma = sigma
        self.num_agents = num_agents
        self.leaders = leaders
        self.gpt_model = gpt_model
        self.temp = temp
        self.with_long_memory = with_long_memory
        self.current_step = 0

        exp_name = f"agents_{num_agents}_K_{K}_alpha_{alpha}_sigma_{sigma}_seed_{seed}"
        self.run_dir = os.path.join(exp_dir, network_type, topic, exp_name)
        os.makedirs(self.run_dir, exist_ok=True)

        # Load beliefs and stubbornness
        data_file = os.path.join(
            os.path.dirname(__file__), '..',
            'data', f'numeric_sim_opnions_and_stubbornness_num_agents_{num_agents}.json'
        )
        with open(data_file, 'r') as f:
            raw = json.load(f)
        self._init_beliefs_numeric = {int(k): float(v) for k, v in raw['opinions'].items()}
        self._init_beliefs_llm = {int(k): int(float(v) * 2) for k, v in raw['opinions'].items()}
        self.stubbornness = {int(k): float(v) for k, v in raw['stubbornness'].items()}

        # Determine agent type assignment
        n_llm = round((1.0 - alpha) * num_agents)
        n_numeric = num_agents - n_llm

        if n_llm == 0:
            self.llm_agent_ids = set()
            self.numeric_agent_ids = set(range(num_agents))
        elif n_numeric == 0:
            self.llm_agent_ids = set(range(num_agents))
            self.numeric_agent_ids = set()
        else:
            # Leaders first in LLM pool, then fill randomly
            self.llm_agent_ids = set(leaders[:min(len(leaders), n_llm)])
            remaining = [i for i in range(num_agents) if i not in self.llm_agent_ids]
            random.shuffle(remaining)
            extra = n_llm - len(self.llm_agent_ids)
            if extra > 0:
                self.llm_agent_ids.update(remaining[:extra])
            self.numeric_agent_ids = set(range(num_agents)) - self.llm_agent_ids

        # Load initial static network
        network_file = os.path.join(
            os.path.dirname(__file__), '..',
            'data', f'{network_type}_network_num_agents_{num_agents}_seed_{seed}.json'
        )
        self.G_static = _load_network(network_file)

        # Load LLM backgrounds only if needed
        self.backgrounds = {}
        self.topic_str = ""
        self.belief_keywords = {}
        if self.llm_agent_ids:
            gpt_label = "gpt-4o-mini" if gpt_model == "gpt-4o-mini-2024-07-18" else gpt_model
            backgrounds_file = os.path.join(
                os.path.dirname(__file__), '..',
                'data', f'agents_backgrounds_num_agents_{num_agents}_{topic}_{gpt_label}.json'
            )
            if not os.path.exists(backgrounds_file):
                raise FileNotFoundError(
                    f"Backgrounds not found: {backgrounds_file}\n"
                    "Run main.py once to generate backgrounds, or use alpha=1.0 (pure numeric)."
                )
            with open(backgrounds_file, 'r') as f:
                self.backgrounds = json.load(f)["backgrounds"]
            opinions_file = os.path.join(os.path.dirname(__file__), '..', 'opinions.json')
            with open(opinions_file, 'r') as f:
                self.topic_str = json.load(f)[topic]
            with open(belief_keywords_file, 'r') as f:
                self.belief_keywords = json.load(f)

        # Create agents
        self.agents_list = []
        for i in tqdm(range(num_agents), desc=f"Creating agents (LLM={n_llm} Num={n_numeric})"):
            agent = self._create_llm_agent(i) if i in self.llm_agent_ids \
                else self._create_numeric_agent(i)
            self.agents_list.append(agent)

        # Initial neighbors from static network
        self.neighbors = {
            i: list(self.G_static.neighbors(i))
            for i in range(num_agents)
        }

        # DiGraph: updated by K-NN each step
        self.G = nx.DiGraph()
        self.G.add_nodes_from(range(num_agents))
        for i, nbrs in self.neighbors.items():
            for j in nbrs:
                self.G.add_edge(i, j)

        self.optimal_cost = self._compute_optimal_cost()
        self.edges_log = []

        # Log agent type assignment for reference
        assignment = {
            "llm_agents": sorted(self.llm_agent_ids),
            "numeric_agents": sorted(self.numeric_agent_ids),
            "alpha": alpha,
            "n_llm": n_llm,
            "n_numeric": n_numeric,
        }
        with open(os.path.join(self.run_dir, "agent_assignment.json"), 'w') as f:
            json.dump(assignment, f, indent=4)

    # ── Agent factories ────────────────────────────────────────────────────
    def _create_numeric_agent(self, i):
        return DynamicAgent(
            unique_id=i,
            model=self,
            initial_belief=self._init_beliefs_numeric[i],
            stubbornness=self.stubbornness[i]
        )

    def _create_llm_agent(self, i):
        bg = self.backgrounds[str(i)]
        system_prompt = bg.get("system_prompt", "You are a helpful assistant")
        initial_opinion = bg.get("initial_opinion")
        if i in self.leaders:
            system_prompt += (
                f" You are an information distributor, you must firmly hold to your own"
                f" opinion {initial_opinion} and refrain from adopting the views of others."
            )
        return SocialAgent(
            model=self,
            unique_id=i,
            name=bg["name"],
            gender=bg["gender"],
            age=bg["age"],
            traits=bg["traits"],
            qualification=bg["education level"],
            initial_belief=self._init_beliefs_llm[i],
            topic=self.topic_str,
            belief_keywords=self.belief_keywords,
            gpt_model=self.gpt_model,
            temp=self.temp,
            initial_opinion=initial_opinion,
            initial_reasoning=bg.get("initial_reasoning"),
            system_prompt=system_prompt,
            with_long_memory=self.with_long_memory,
        )

    # ── Canonical belief: all agents unified to [-1, 1] ───────────────────
    def _canonical(self, i):
        a = self.agents_list[i]
        return float(a.belief) / 2.0 if i in self.llm_agent_ids else float(a.belief)

    def _initial_canonical(self, i):
        if i in self.llm_agent_ids:
            return self._init_beliefs_llm[i] / 2.0
        return self._init_beliefs_numeric[i]

    # ── Gaussian similarity (on canonical scale) ───────────────────────────
    def gaussian_g(self, ci, cj):
        return float(np.exp(-(ci - cj) ** 2 / (2 * self.sigma ** 2)))

    # ── Optimal cost (static NE, computed once) ────────────────────────────
    def _compute_optimal_cost(self):
        n = self.num_agents
        nodes = list(range(n))
        L = nx.laplacian_matrix(self.G_static, nodelist=nodes).toarray().astype(float)
        rho = np.array([self.stubbornness[i] for i in nodes])
        s = np.array([self._initial_canonical(i) for i in nodes])
        W = self.K * np.diag(rho)
        z_star = np.linalg.solve(L + W, W @ s)
        total = 0.0
        for i in nodes:
            for j in self.G_static.neighbors(i):
                total += (z_star[i] - z_star[j]) ** 2
            total += rho[i] * self.K * (z_star[i] - s[i]) ** 2
        return float(total)

    # ── K-NN rewiring ──────────────────────────────────────────────────────
    def _update_knn(self):
        canonical = [self._canonical(i) for i in range(self.num_agents)]
        for i in range(self.num_agents):
            scores = sorted(
                [(j, self.gaussian_g(canonical[i], canonical[j]))
                 for j in range(self.num_agents) if j != i],
                key=lambda x: x[1], reverse=True
            )
            self.neighbors[i] = [j for j, _ in scores[:self.K]]
        self.G.remove_edges_from(list(self.G.edges()))
        for i, nbrs in self.neighbors.items():
            for j in nbrs:
                self.G.add_edge(i, j)

    # ── Main step ──────────────────────────────────────────────────────────
    def step(self):
        canonical_snapshot = [self._canonical(i) for i in range(self.num_agents)]

        # LLM agents: memory interaction + GPT belief update
        if self.llm_agent_ids:
            for i in self.llm_agent_ids:
                self.agents_list[i].agent_interaction = [
                    self.agents_list[j] for j in self.neighbors[i]
                ]
            shuffled_llm = [self.agents_list[i] for i in self.llm_agent_ids]
            random.shuffle(shuffled_llm)
            for agent in shuffled_llm:
                agent.step()        # → agent.interact()
            for agent in shuffled_llm:
                update_day(agent)   # GPT → new belief/opinion

        # Numeric agents: synchronous DeGroot on canonical snapshot
        for i in self.numeric_agent_ids:
            nbr_beliefs = [canonical_snapshot[j] for j in self.neighbors[i]]
            if nbr_beliefs:
                self.agents_list[i].belief = float(np.mean(nbr_beliefs))
            self.agents_list[i].record()

        # K-NN rewiring based on updated beliefs
        self._update_knn()
        self.edges_log.append([list(e) for e in self.G.edges()])
        self.current_step += 1
        self.save_model_data()

    # ── Metrics ───────────────────────────────────────────────────────────
    def compute_total_social_cost(self):
        total = 0.0
        for i in range(self.num_agents):
            zi = self._canonical(i)
            si = self._initial_canonical(i)
            nbr = [self._canonical(j) for j in self.neighbors[i]]
            total += sum((zi - b) ** 2 for b in nbr)
            total += self.stubbornness[i] * self.K * (zi - si) ** 2
        return float(total)

    def compute_polarization(self):
        z = np.array([self._canonical(i) for i in range(self.num_agents)])
        return float(np.mean((z - z.mean()) ** 2))

    def compute_modularity(self):
        G_u = self.G.to_undirected()
        if G_u.number_of_edges() == 0:
            return 0.0
        partition = community_louvain.best_partition(G_u)
        return float(community_louvain.modularity(partition, G_u))

    def compute_echo_chamber_effect(self):
        canonical = [self._canonical(i) for i in range(self.num_agents)]
        total_g = sum(
            self.gaussian_g(canonical[i], canonical[j])
            for i in range(self.num_agents)
            for j in self.neighbors[i]
        )
        count = sum(len(v) for v in self.neighbors.values())
        return float(total_g / count) if count > 0 else 0.0

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
        data = {}
        for a in self.agents_list:
            entry = {"beliefs": a.beliefs, "opinions": a.opinions}
            if a.unique_id in self.llm_agent_ids:
                entry["reasonings"] = a.reasonings
                entry["short_memory"] = a.short_memory_full
                entry["long_memory"] = a.long_memory_full
            data[str(a.unique_id)] = entry
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
        n_llm = len(self.llm_agent_ids)
        n_num = len(self.numeric_agent_ids)
        for _ in tqdm(range(step_count),
                      desc=f"Hybrid {self.network_type} K={self.K} alpha={self.alpha}"):
            self.step()
        self.save_agents_data()
        self.save_interaction_data()
        self.save_edges_data()
        final_cost = self.compute_total_social_cost()
        print(f"\nalpha={self.alpha}  LLM={n_llm}  Numeric={n_num}")
        print(f"Optimal cost : {self.optimal_cost:.4f}")
        print(f"Final PoA    : {final_cost / self.optimal_cost:.4f}")
        print(f"Saved to     : {self.run_dir}")
