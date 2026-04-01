import mesa
import numpy as np


class NumericAgent(mesa.Agent):
    def __init__(self, unique_id, model, initial_belief, stubbornness=0.5, update_method='degroot', epsilon=0.5):
        super().__init__(model)
        self.unique_id = unique_id
        
        # model.py:83 將 Opinion Value * 2，Range = [-2.0, 2.0]，再做為 LLM 的 Belief Value，是為了讓 GPT 能明確理解 Value 強度
        # 因此，換為以 Numeric 計算 Belief Value 時，僅需要以 Opinioin Value 即可，Range = [-1.0, 1.0]

        self.initial_belief = initial_belief   # 初始 Belief 數值為 Discrete，從 JSON 載入，只有 5 個固定值：-1.0, -0.5, 0.0, 0.5, 1.0 
        self.belief = initial_belief
        self.stubbornness = stubbornness       # 將原先用於 Scale-Free Network，做為調整 Agents 的 Stubbornness 數值，用於 FJ
        self.update_method = update_method
        self.epsilon = epsilon                 # 用於 BCM，自行預設為 0.5
        
        self.beliefs = [self.belief]                  # 記錄每次 Step 的 Belief Value，用於 Experimental Analysis
        self.opinions = [str(round(self.belief, 4))]  # 讓 Numeric Opinion，相容於 GIF 格式

    def get_neighbor_beliefs(self):           # 取得 Neighbor 的 Belief 數值
        neighbors = self.model.grid.get_neighbors(self.pos, include_center=False)
        return [n.belief for n in neighbors]

    def step_degroot(self):                    # 將當前 Agent 自身 + 所有 Neighbor 的 Belief 數值，相加再取平均，得到新的 Belief Value
        neighbor_beliefs = self.get_neighbor_beliefs()
        if not neighbor_beliefs:
            return
        all_beliefs = neighbor_beliefs + [self.belief]
        self.belief = float(np.mean(all_beliefs))

    def step_fj(self):                          # 將 Agent 的「初始 Belief 數值」 和「當前 Neighbor Belief 數值加總取平均」，透過 Stubbornness 來調整權重，得到新的 Belief Value，越高越固執，Agent 越難被鄰居說服 
        neighbor_beliefs = self.get_neighbor_beliefs()
        if not neighbor_beliefs:
            return
        neighbor_avg = float(np.mean(neighbor_beliefs))
        self.belief = (1 - self.stubbornness) * neighbor_avg + self.stubbornness * self.initial_belief

    def step_bcm(self):                         # 將當前 Agent 自身 Belief 數值 + Neighbor 的 Belief 數值（只選與自身差距 ≤ Epsilon 的 Neighbor），相加再取平均更新 Belief Value
        neighbor_beliefs = self.get_neighbor_beliefs()
        eligible = [b for b in neighbor_beliefs if abs(self.belief - b) <= self.epsilon]
        if not eligible:
            return
        self.belief = float(np.mean(eligible))

    def step(self):                             # 根據 update_method 執行對應 Numeric Update Method，並記錄 Belief 數值和 Opinion
        if self.update_method == 'degroot':
            self.step_degroot()
        elif self.update_method == 'fj':
            self.step_fj()
        elif self.update_method == 'bcm':
            self.step_bcm()

        self.beliefs.append(self.belief)
        self.opinions.append(str(round(self.belief, 4)))
