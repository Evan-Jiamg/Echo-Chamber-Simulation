import mesa
import numpy as np


class NumericAgent(mesa.Agent):
    def __init__(self, unique_id, model, initial_belief, stubbornness=0.5, update_method='degroot', epsilon=0.5):
        super().__init__(model)
        self.unique_id = unique_id

        self.initial_belief = initial_belief
        self.belief = initial_belief
        self.stubbornness = stubbornness      # 用於 FJ
        self.update_method = update_method
        self.epsilon = epsilon                # 用於 BCM

        self.beliefs = [self.belief]          # 記錄每步的 belief（供儲存結果）
        self.opinions = [str(round(self.belief, 4))]  # 供 GIF 相容格式

    def get_neighbor_beliefs(self):
        neighbors = self.model.grid.get_neighbors(self.pos, include_center=False)
        return [n.belief for n in neighbors]

    def step_degroot(self):
        neighbor_beliefs = self.get_neighbor_beliefs()
        if not neighbor_beliefs:
            return
        all_beliefs = neighbor_beliefs + [self.belief]
        self.belief = float(np.mean(all_beliefs))

    def step_fj(self):
        neighbor_beliefs = self.get_neighbor_beliefs()
        if not neighbor_beliefs:
            return
        neighbor_avg = float(np.mean(neighbor_beliefs))
        self.belief = (1 - self.stubbornness) * neighbor_avg + self.stubbornness * self.initial_belief

    def step_bcm(self):
        neighbor_beliefs = self.get_neighbor_beliefs()
        eligible = [b for b in neighbor_beliefs if abs(self.belief - b) <= self.epsilon]
        if not eligible:
            return
        self.belief = float(np.mean(eligible))

    def step(self):
        if self.update_method == 'degroot':
            self.step_degroot()
        elif self.update_method == 'fj':
            self.step_fj()
        elif self.update_method == 'bcm':
            self.step_bcm()

        self.beliefs.append(self.belief)
        self.opinions.append(str(round(self.belief, 4)))
