import os
from pathlib import Path
import numpy as np
import torch
from agent.base import AgentBase
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_util import make_vec_env
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
from erl_config import build_env
from trade_simulator import TradeSimulator
from torch import nn as nn

def get_factors(number: int) -> list:
    factors = []
    for i in range(1, int(number ** 0.5) + 1):
        if number % i == 0:
            factors.append(i)
            if i != number // i:
                factors.append(number // i)
    return sorted(factors)

def find_closest_factor(number, y):
    factors = get_factors(y)
    return min(factors, key=lambda x: abs(x - number))

def preprocess_ppo_params(params: dict, n_envs: int):
    new_params = params.copy()
    n_steps = params["n_steps"]
    batch_size = params["batch_size"]

    if (n_steps * n_envs) % batch_size != 0:        
        batch_size = find_closest_factor(batch_size, n_steps * n_envs)

    net_arch_type = params["net_arch"]
    del new_params["net_arch"]
    
    net_arch = {
        "tiny": dict(pi=[64], vf=[64]),
        "small": dict(pi=[64, 64], vf=[64, 64]),
        "medium": dict(pi=[256, 256], vf=[256, 256]),
    }[net_arch_type]
    
    activation_fn_type = params["activation_fn"]
    del new_params["activation_fn"]
    
    activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU, "leaky_relu": nn.LeakyReLU}[activation_fn_type]

    new_params.update({
        "policy_kwargs": dict(
            net_arch=net_arch,
            activation_fn=activation_fn,
            ortho_init=False,
        ),
    })
    return new_params

def preprocess_dqn_params(params: dict, n_envs:int):
    new_params = params.copy()
    net_arch_type = params["net_arch"]
    del new_params["net_arch"]
    
    net_arch = {
        "tiny": [64],
        "small": [64, 64],
        "medium": [256, 256],
    }[net_arch_type]
        
    new_params.update({
        "policy_kwargs": dict(
            net_arch=net_arch,
        ),
    })
    return new_params

PREPROCESS_ONLINE_RL_PARAMS = {
    'ppo': preprocess_ppo_params,
    'dqn': preprocess_dqn_params
}

ONLINE_RL_TYPES = {
    "ppo": PPO,
    "dqn": DQN,
}

ONLINE_RL_EPISODES = 50


class AgentOnlineRl(AgentBase):
    def __init__(
        self,
        agent_type: str,
        device: str = "cpu",
        gpu_id: int = -1,
    ):   
        self.device = torch.device(device)
        self.gpu_id = gpu_id
        self.agent_type_name = agent_type
        self.agent_type = ONLINE_RL_TYPES[agent_type]
        self.agent = None

    def action(
        self,
        state: np.ndarray,
    ):  
        tensor_state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        return self.agent.predict(tensor_state, deterministic=True)[0]
    
    def load(
        self, 
        model_path: str,
    ):
        self.agent = self.agent_type.load(model_path)
    
    def save(
        self, 
        model_path: str,
    ):
        self.agent.save(model_path)
    
    def train(
        self,
        env_args: dict,
        model_args: dict,
        learn_args: dict = {},
        n_episodes: int = ONLINE_RL_EPISODES
    ):
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
        progress_bar = model_args.get('progress_bar', True)
        model_args.pop('progress_bar', None)
        
        print(f'Training with seed: {seed} on days: [{env_args["days"][0]}, {env_args["days"][0]}]')
        
        model_args = PREPROCESS_ONLINE_RL_PARAMS[self.agent_type_name](model_args, n_envs=n_envs)
        self.agent = self.agent_type(
            "MlpPolicy", 
            env, 
            verbose=0,
            seed=seed,
            **model_args,
        )
        
        self.agent.learn(
            total_timesteps=env_args['max_step'] * n_episodes, 
            progress_bar=progress_bar, 
            **learn_args,
        )
        
        # Plot tb plots
        tb_log_path = model_args.get('tensorboard_log', None)
        if tb_log_path is not None:
            tb_dirs = AgentOnlineRl._find_all_directories(tb_log_path)
            for tb_dir in tb_dirs:
                tb_curr_plot_dir = os.path.join(tb_dirs, tb_dir.split('/')[-1])
                os.makedirs(tb_curr_plot_dir, exist_ok=True)
                AgentOnlineRl._save_tensorboard_plots(tb_dir, tb_curr_plot_dir)
            
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