import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import mesa
import json
import numpy as np
import random
from tqdm import tqdm
from scipy.stats import pearsonr
import networkx as nx

from numeric_agent import NumericAgent


def load_network_structure(file_path):
    with open(file_path, 'r') as f:
        network_data = json.load(f)
    G = nx.Graph()
    G.add_nodes_from(network_data["nodes"])
    G.add_edges_from(network_data["edges"])
    return G


class NumericWorld(mesa.Model):
    def __init__(self, network_type, update_method, num_agents=50, seed=50,
                 epsilon=0.5, exp_dir='../experiments_numeric'):
        super().__init__()

        self.network_type = network_type
        self.update_method = update_method
        self.num_agents = num_agents
        self.epsilon = epsilon
        self.current_step = 0

        # 建立實驗輸出路徑
        exp_name = f"agents_{num_agents}_{update_method}_seed_{seed}"
        if update_method == 'bcm':
            exp_name += f"_epsilon_{epsilon}"
        self.run_dir = os.path.join(exp_dir, network_type, 'euthanasia', exp_name)
        os.makedirs(self.run_dir, exist_ok=True)

        # 載入網路結構
        network_file = os.path.join(
            os.path.dirname(__file__), '..',
            'data', f'{network_type}_network_num_agents_{num_agents}_seed_{seed}.json'
        )
        self.G = load_network_structure(network_file)
        self.grid = mesa.space.NetworkGrid(self.G)
        self.agents_list = []

        # 載入初始 belief 與 stubbornness
        belief_file = os.path.join(
            os.path.dirname(__file__), '..',
            'data', f'numeric_sim_opnions_and_stubbornness_num_agents_{num_agents}.json'
        )
        with open(belief_file, 'r') as f:
            data = json.load(f)

        opinions = data['opinions']
        stubbornness = data['stubbornness']

        # 建立 Agent
        for node in tqdm(self.G.nodes(), desc="Creating agents"):
            initial_belief = float(opinions[str(node)])
            s = float(stubbornness[str(node)])
            agent = NumericAgent(
                unique_id=node,
                model=self,
                initial_belief=initial_belief,
                stubbornness=s,
                update_method=update_method,
                epsilon=epsilon
            )
            self.agents_list.append(agent)
            self.grid.place_agent(agent, node)

    def compute_polarization(self):
        beliefs = [agent.belief for agent in self.agents_list]
        return float(np.mean(beliefs))

    def compute_nci(self):
        beliefs = {agent.unique_id: agent.belief for agent in self.agents_list}
        nodes = list(self.G.nodes())
        node_beliefs = np.array([beliefs[n] for n in nodes])
        neighbor_avgs = []
        for node in nodes:
            neighbors = list(self.G.neighbors(node))
            if neighbors:
                avg = np.mean([beliefs[n] for n in neighbors])
            else:
                avg = beliefs[node]
            neighbor_avgs.append(avg)
        neighbor_avgs = np.array(neighbor_avgs)
        if np.std(node_beliefs) == 0 or np.std(neighbor_avgs) == 0:
            return 0.0
        nci, _ = pearsonr(node_beliefs, neighbor_avgs)
        return float(nci)

    def compute_echo_chamber_effect(self):
        total_similarity = 0
        total_connections = 0
        for agent in self.agents_list:
            neighbors = self.grid.get_neighbors(agent.pos, include_center=False)
            for neighbor in neighbors:
                similarity = 1 - abs(agent.belief - neighbor.belief)
                total_similarity += similarity
                total_connections += 1
        if total_connections == 0:
            return 0.0
        return float(total_similarity / total_connections)

    def step(self):
        agents_shuffled = self.agents_list[:]
        random.shuffle(agents_shuffled)
        for agent in agents_shuffled:
            agent.step()
        self.current_step += 1
        self.save_model_data()

    def save_model_data(self):
        model_data = {
            "step": self.current_step,
            "polarization": self.compute_polarization(),
            "neighbor_correlation_index": self.compute_nci(),
            "echo_chamber_effect": self.compute_echo_chamber_effect(),
        }
        file_path = os.path.join(self.run_dir, "model_overview.json")
        with open(file_path, 'a') as f:
            json.dump(model_data, f)
            f.write('\n')

    def save_agents_data(self):
        agents_data = {}
        for agent in self.agents_list:
            agents_data[agent.unique_id] = {
                "beliefs": agent.beliefs,
                "opinions": agent.opinions,
            }
        file_path = os.path.join(self.run_dir, "agents_data.json")
        with open(file_path, 'w') as f:
            json.dump(agents_data, f, indent=4)

    def save_interaction_data(self):
        # 相容 utils.generate_belief_animation() 的格式
        agents_data = {}
        for agent in self.agents_list:
            agents_data[str(agent.unique_id)] = {
                "beliefs": agent.beliefs,
                "opinions": agent.opinions,
            }
        file_path = os.path.join(self.run_dir, "agents_interaction_data.json")
        with open(file_path, 'w') as f:
            json.dump(agents_data, f, indent=4)

    def run_model(self, step_count):
        for _ in tqdm(range(step_count), desc=f"Running {self.update_method} on {self.network_type}"):
            self.step()
            print(f"Step {self.current_step} | Polarization: {self.compute_polarization():.4f} | NCI: {self.compute_nci():.4f}")

        self.save_agents_data()
        self.save_interaction_data()
        print(f"\nResults saved to: {self.run_dir}")
