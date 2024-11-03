import numpy as np
import torch
from agent.base import AgentBase
from sbx import PPO
import os
from collections import Counter

class AgentPPO(AgentBase):
    def __init__(
        self,
        model_path: str,
    ):
        seeds_dir = os.listdir(model_path)
        print(f'Loading {len(seeds_dir)} seeds, {seeds_dir}')
        self.agents = [PPO.load(os.path.join(model_path, seed_dir)) for seed_dir in seeds_dir]
        self.device = torch.device("cpu")

    def action(
        self,
        state: np.ndarray,
    ):  
        tensor_state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        actions = [agent.predict(tensor_state, deterministic=True)[0][0] for agent in self.agents]
        return Counter(actions).most_common(1)[0][0]
