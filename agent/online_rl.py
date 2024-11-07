import os
from pathlib import Path
import typing
import numpy as np
import torch

from agent.base import AgentBase
from collections import Counter

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_util import make_vec_env
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt

from erl_config import build_env
from trade_simulator import TradeSimulator

class AgentOnlineRl(AgentBase):
    def __init__(
        self,
        agent_class: typing.Union[PPO, DQN],
        model_path: str,
        deterministic: bool = True,
    ):
        assert [agent_class == x for x in [PPO, DQN]], 'Only stable_baseline3 PPO and DQN are supported (prefer not to use sbx for now)'
        
        
        self.device = torch.device("cpu")
        self.gpu_id = -1
        self.deterministic = deterministic
        self.model_path = model_path
        self.agent_class = agent_class
        self.agents = []

    def action(
        self,
        state: np.ndarray,
    ):  
        tensor_state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        actions = [agent.predict(tensor_state, deterministic=self.deterministic)[0][0] for agent in self.agents]        
        return Counter(actions).most_common(1)[0][0]
    
    
    def load(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f'Model path {self.model_path} does not exist')
            
        seeds_dir = os.listdir(self.model_path)
        print(f'Loading {len(seeds_dir)} seeds, {seeds_dir}')
        self.agents = [self.agent_class.load(os.path.join(self.model_path, seed_dir)) for seed_dir in seeds_dir]
        
        
    def train(self, days, seed=None, save_dir: str = ".", 
              max_step:int=480, n_envs:int=1, n_episodes:int=300,
              progress_bar:bool=True, save_tb_plots:bool=True,
              model_params:dict={}, learn_params:dict={}):
        assert len(days) == 2 and days[0] < days[1] and days[0] >= 7 and days[1] <= 14, 'Error on days'
        if seed is None:
            np.random.seed()
            seed = np.random.randint(2**32 - 1, dtype="int64")
        
        env_args = {
            "env_name": "TradeSimulator-v0",
            "num_envs": 1,
            "num_sims": 1,
            "state_dim": 10,
            "action_dim": 3,
            "if_discrete": True,
            "max_position": 1,
            "slippage": 7e-7,
            "step_gap": 2,
            "eval_sequential": False,
            "env_class": TradeSimulator,
            "max_step": max_step,
            "days": days,
            "seed": seed,
        }            

        if save_tb_plots:
            tb_log_path = os.path.join(save_dir, f'seed_{seed}', "tb")       
            os.makedirs(tb_log_path, exist_ok=True)
        else:
            tb_log_path = None
        
        env = make_vec_env(
            lambda: build_env(TradeSimulator, env_args, gpu_id=self.gpu_id),
            n_envs=n_envs,
            seed=seed
        )
        print(f'Training with seed: {seed} on days: [{self.start_day}, {self.end_day}]')
        agent = self.agent_class("MlpPolicy", env, verbose=0, 
                                 seed=seed, 
                                 tensorboard_log=tb_log_path, 
                                 **model_params)
            
        agent.learn(total_timesteps=max_step * n_episodes, 
                    progress_bar=progress_bar, 
                    **learn_params)
        
        # Plot tb plots
        if tb_log_path is not None:
            tb_dirs = AgentOnlineRl._find_all_directories(tb_log_path)
            for tb_dir in tb_dirs:
                tb_curr_plot_dir = os.path.join(tb_dirs, tb_dir.split('/')[-1])
                os.makedirs(tb_curr_plot_dir, exist_ok=True)
                AgentOnlineRl._save_tensorboard_plots(tb_dir, tb_curr_plot_dir)
        
        return agent
    

    @staticmethod
    def save(agent: typing.Union[PPO, DQN], save_dir: str = "."):
        os.makedirs(save_dir, exist_ok=True)
        agent.save(save_dir)
        
    
    @staticmethod
    def _find_all_directories(path: str | Path) -> list[str]:
        if isinstance(path, str):
            path = Path(path)
        directories = [str(p) for p in path.rglob('*') if p.is_dir()]
        return directories
    
    
    @staticmethod
    def _save_tensorboard_plots(log_dir: str, output_dir: str):
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Load the tensorboard logs
        event_acc = EventAccumulator(log_dir)
        event_acc.Reload()
        
        # Get all the scalar tags (metrics names)
        tags = event_acc.Tags()['scalars']
        
        for tag in tags:
            # Retrieve the scalar data for each tag
            events = event_acc.Scalars(tag)
            steps = [e.step for e in events]
            values = [e.value for e in events]
            
            # Plot the data
            plt.figure(figsize=(10, 6))
            plt.plot(steps, values, label=tag)
            plt.title(f'Training Plot for {tag}')
            plt.xlabel('Steps')
            plt.ylabel(tag)
            plt.grid(True)
            plt.legend()
            
            # Save the figure
            output_path = os.path.join(output_dir, f'{tag.replace("/", "_")}.png')
            plt.savefig(output_path)
            plt.close()
