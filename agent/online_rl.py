import os
import numpy as np
import torch

from agent.base import AgentBase
from collections import Counter
from stable_baselines3.common.base_class import BaseAlgorithm

class AgentOnlineRl(AgentBase):
    def __init__(
        self,
        agent_class: BaseAlgorithm,
        model_path: str,
        deterministic: bool = True,
    ):
        seeds_dir = os.listdir(model_path)
        print(f'Loading {len(seeds_dir)} seeds, {seeds_dir}')
        self.agents = [agent_class.load(os.path.join(model_path, seed_dir)) for seed_dir in seeds_dir]
        self.device = torch.device("cpu")
        self.deterministic = deterministic

    def action(
        self,
        state: np.ndarray,
    ):  
        tensor_state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        actions = [agent.predict(tensor_state, deterministic=self.deterministic)[0][0] for agent in self.agents]        
        return Counter(actions).most_common(1)[0][0]
