import os
from pathlib import Path
import typing
import numpy as np
from sb3_contrib import RecurrentPPO
from scipy import stats
import torch
from agent.base import AgentBase
from collections import Counter
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_util import make_vec_env
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
from erl_config import build_env
from trade_simulator import TradeSimulator


ONLINE_RL_NAME_TO_CLASS_DICT = {
    "ppo": PPO,
    "dqn": DQN,
}


class AgentOnlineRl(AgentBase):
    def __init__(
        self,
        agent_class: str,
        deterministic: bool = True,
        device: str = "cpu",
        gpu_id: int = -1,
    ):
        assert agent_class in ['ppo', 'dqn'], 'Only stable_baseline3 PPO and DQN are supported (prefer not to use sbx for now)'
    
        self.device = torch.device(device)
        self.gpu_id = gpu_id
        self.deterministic = deterministic
        self.agent_class = ONLINE_RL_NAME_TO_CLASS_DICT[agent_class]
        self.agent = None

    def action(
        self,
        state: np.ndarray,
    ):  
        tensor_state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        
        results = np.empty((len(self.agents), state.shape[0]))
        for i, agent in enumerate(self.agents):
            results[i, :] = agent.predict(tensor_state, deterministic=self.deterministic)[0]
        
        return stats.mode(results, axis=0, keepdims=False)[0]    
    
    def load(self, model_path: str):
        self.agents = [self.agent_class.load(os.path.join(model_path))]
        
        
    def train(self,
              env_args: dict,
              model_args: dict,
              learn_args: dict = {},
              n_episodes: int = 200):
        
        days = env_args.get("days", None)
        assert days is not None and days[0] <= days[1] and days[0] >= 7 and days[1] <= 16, 'Correct days must be provided'            

        n_envs = env_args.get("num_envs", 1)
        # Setting num_envs externally in stable baseline, in the wrapped env setting num_envs to 1
        env_args["num_envs"] = 1
        env_args['num_sims'] = 1  # Only num_sims=1 is supported

        seed = env_args.get("seed", None)
        if seed is None:
            np.random.seed()
            seed = int(np.random.randint(2**32 - 1, dtype="int64"))
            print(f'Seed not provided, using random seed {seed}')
            env_args["seed"] = seed
            
        env = make_vec_env(
            lambda: build_env(TradeSimulator, env_args, gpu_id=self.gpu_id),
            n_envs=n_envs,
            seed=seed
        )
        progress_bar = model_args.get('progress_bar', False)
        model_args.pop('progress_bar', None)
        
        print(f'Training with seed: {seed} on days: [{env_args["days"][0]}, {env_args["days"][0]}]')
        agent = self.agent_class("MlpPolicy", env, verbose=0, 
                                 seed=seed,
                                 **model_args)
        

        agent.learn(total_timesteps=env_args['max_step'] * n_episodes, progress_bar=progress_bar, **learn_args)
        
        # Plot tb plots
        tb_log_path = model_args.get('tensorboard_log', None)
        if tb_log_path is not None:
            tb_dirs = AgentOnlineRl._find_all_directories(tb_log_path)
            for tb_dir in tb_dirs:
                tb_curr_plot_dir = os.path.join(tb_dirs, tb_dir.split('/')[-1])
                os.makedirs(tb_curr_plot_dir, exist_ok=True)
                AgentOnlineRl._save_tensorboard_plots(tb_dir, tb_curr_plot_dir)
        
        return agent
    
    def save(self, save_path: str):
        self.agents[0].save(save_path)
            
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
