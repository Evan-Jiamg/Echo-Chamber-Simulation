import mesa
import numpy as np


class DynamicAgent(mesa.Agent):
    def __init__(self, unique_id, model, initial_belief, stubbornness):
        super().__init__(unique_id, model)
        self.unique_id = unique_id
        self.initial_belief = initial_belief  # intrinsic opinion s_i, fixed
        self.belief = initial_belief          # expressed opinion z_i, updated each step
        self.stubbornness = stubbornness      # rho_i, the conformity term of the social cost

        self.beliefs = [self.belief]
        self.opinions = [str(round(self.belief, 4))]

    def step_degroot(self, neighbor_beliefs):
        if not neighbor_beliefs:
            return
        self.belief = float(np.mean(neighbor_beliefs))

    def step_fj(self, neighbor_beliefs):
        if not neighbor_beliefs:
            return
        social = float(np.mean(neighbor_beliefs))
        # Best response of C_i = Σ(z_i-z_j)² + ρ_i·K·(z_i-s_i)²
        # FOC → z_i = mean(z_j)/(1+ρ_i) + ρ_i·s_i/(1+ρ_i)
        w = self.stubbornness / (1.0 + self.stubbornness)
        self.belief = (1.0 - w) * social + w * self.initial_belief

    def step_bcm(self, neighbor_beliefs, epsilon):
        in_bound = [b for b in neighbor_beliefs if abs(self.belief - b) <= epsilon]
        if in_bound:
            self.belief = float(np.mean(in_bound))

    def compute_social_cost(self, neighbor_beliefs, K):
        # C_i = Σ_{j∈N_i}(z_i - z_j)² + ρ_i · K · (z_i - s_i)²
        neighbor_cost = sum((self.belief - b) ** 2 for b in neighbor_beliefs)
        intrinsic_cost = self.stubbornness * K * (self.belief - self.initial_belief) ** 2
        return float(neighbor_cost + intrinsic_cost)

    def record(self):
        self.beliefs.append(self.belief)
        self.opinions.append(str(round(self.belief, 4)))
