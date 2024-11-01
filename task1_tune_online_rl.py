from typing import Any
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
import torch
import tqdm
from erl_config import build_env
from trade_simulator import EvalTradeSimulator, TradeSimulator
from trlib.algorithms.reinforcement.fqi import FQI
from trlib.policies.qfunction import ZeroQ
from trlib.policies.valuebased import EpsilonGreedy
from joblib import Parallel, delayed
import optuna
from ast import Dict, literal_eval
import argparse
import os
from generate_experience_fqi import generate_experience
import flax.linen as nn # for JAX
import optuna
from stable_baselines3.common.base_class import BaseAlgorithm
from datetime import datetime

from sbx import PPO, DQN

def get_cli_args():
    """Create CLI parser and return parsed arguments"""
    parser = argparse.ArgumentParser()
    # Example-specific args.
    parser.add_argument(
        '--start_day_train',
        type=int,
        default=7,
        help="starting day to train (included) "
    )

    parser.add_argument(
        '--end_day_train',
        type=int,
        default=15,
        help="ending day to train (included) "
    )
    
    parser.add_argument(
        '--n_trials',
        type=int,
        default=50,
        help="number of iterations of optuna"
    )
    parser.add_argument(
        '--n_seeds',
        type=int,
        default=5
    )
    parser.add_argument(
        '--n_windows',
        type=int,
        default=8
    )
    parser.add_argument(
        '--out_dir',
        type=str,
        default="."
    )

    return parser.parse_args()

def interquartile_mean(data: np.ndarray, q_min: int = 25, q_max: int = 75) -> float:
    assert data.ndim == 1, "Input data must be 1D"
    sorted_data = np.sort(data)
    
    q_min = np.percentile(sorted_data, q_min)
    q_max = np.percentile(sorted_data, q_max)
    filtered_data = sorted_data[(sorted_data >= q_min) & (sorted_data <= q_max)]    
    iqm = np.mean(filtered_data)
    return iqm

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

def sample_ppo_params(trial: optuna.Trial, n_actions: int, n_envs: int, additional_args: dict):
    n_steps_range = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    batch_size_range = [8, 16, 32, 64, 128, 256, 512]
        
    n_steps = trial.suggest_categorical("n_steps", n_steps_range)        
    batch_size = trial.suggest_categorical("batch_size", batch_size_range)
    gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1, log=True)
    ent_coef = trial.suggest_float("ent_coef", 0.00000001, 0.1, log=True)
    clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3, 0.4])
    n_epochs = trial.suggest_categorical("n_epochs", [1, 5, 10, 20])
    gae_lambda = trial.suggest_categorical("gae_lambda", [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0])
    max_grad_norm = trial.suggest_categorical("max_grad_norm", [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 5])
    vf_coef = trial.suggest_float("vf_coef", 0, 1)
    net_arch_type = trial.suggest_categorical("net_arch", ["tiny", "small", "medium"])

    ortho_init = False
    activation_fn_name = trial.suggest_categorical("activation_fn", ["tanh", "relu"])

            
    if (n_steps * n_envs) % batch_size != 0:        
        batch_size = find_closest_factor(batch_size, n_steps * n_envs)

    net_arch = {
        "tiny": dict(pi=[64], vf=[64]),
        "small": dict(pi=[64, 64], vf=[64, 64]),
        "medium": dict(pi=[256, 256], vf=[256, 256]),
    }[net_arch_type]

    # activation_fn_name = 'relu'
    activation_fn = {"tanh": nn.tanh, "relu": nn.relu, "elu": nn.elu, "leaky_relu": nn.leaky_relu}[activation_fn_name]

    return {
        "n_steps": n_steps,
        "batch_size": batch_size,
        "gamma": gamma,
        "learning_rate": learning_rate,
        "ent_coef": ent_coef,
        "clip_range": clip_range,
        "n_epochs": n_epochs,
        "gae_lambda": gae_lambda,
        "max_grad_norm": max_grad_norm,
        "vf_coef": vf_coef,
        "policy_kwargs": dict(
            net_arch=net_arch,
            activation_fn=activation_fn,
            ortho_init=ortho_init,
        ),
    }
    
def sample_dqn_params(trial: optuna.Trial, n_actions: int, n_envs: int, additional_args: dict):
    """
    Sampler for DQN hyperparams.

    :param trial:
    :return:
    """
    gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 100, 128, 256, 512])
    buffer_size = trial.suggest_categorical("buffer_size", [int(1e4), int(5e4), int(1e5), int(1e6)])
    exploration_final_eps = trial.suggest_float("exploration_final_eps", 0, 0.2)
    exploration_fraction = trial.suggest_float("exploration_fraction", 0, 0.5)
    target_update_interval = trial.suggest_categorical("target_update_interval", [1, 1000, 5000, 10000, 15000, 20000])
    learning_starts = trial.suggest_categorical("learning_starts", [0, 1000, 5000, 10000, 20000])

    train_freq = trial.suggest_categorical("train_freq", [1, 4, 8, 16, 128, 256, 1000])
    subsample_steps = trial.suggest_categorical("subsample_steps", [1, 2, 4, 8])
    gradient_steps = max(train_freq // subsample_steps, 1)

    net_arch_type = trial.suggest_categorical("net_arch", ["tiny", "small", "medium"])

    net_arch = {"tiny": [64], "small": [64, 64], "medium": [256, 256]}[net_arch_type]

    hyperparams = {
        "gamma": gamma,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "buffer_size": buffer_size,
        "train_freq": train_freq,
        "gradient_steps": gradient_steps,
        "exploration_fraction": exploration_fraction,
        "exploration_final_eps": exploration_final_eps,
        "target_update_interval": target_update_interval,
        "learning_starts": learning_starts,
        "policy_kwargs": dict(net_arch=net_arch),
    }

    return hyperparams

SAMPLER = {
    PPO.__name__: sample_ppo_params,
    DQN.__name__: sample_dqn_params
}
    

class TradeSimulatorOptimizer:
    def __init__(self, agent_class: BaseAlgorithm, device, out_dir, plot_dir, start_day_train, end_day_train, max_steps, n_episodes = 1000, storage = None, n_seeds=5, n_trials=100, n_days_val = 1, gpu_id = -1):
        self.agent_class = agent_class
        self.device = device
        self.gpu_id = gpu_id
        self.out_dir = out_dir
        self.plot_dir = plot_dir
        self.start_day_train = start_day_train
        self.end_day_train = end_day_train
        self.n_days_val = n_days_val
        self.max_steps = max_steps
        self.n_episodes = n_episodes
        self.n_seeds = n_seeds
        self.n_trials = n_trials
        self.storage = storage
        self.n_envs = 1
        self.n_actions = 3
        self.env_class = TradeSimulator
        self.eval_env_class = EvalTradeSimulator
        
        self.env_args = self._initialize_env_args()
        self._setup_directories()
        
        np.random.seed()
        self.seeds = [np.random.randint(2**32 - 1, dtype="int64").item() for i in range(self.n_seeds)]
        
        self.study = None
    
    def create_study(self):
        self.study = optuna.create_study(study_name=f'{self.agent_class.__name__}_tuning_{self.n_seeds}seeds_{self.start_day_train}_{self.end_day_train}',
                                         direction="maximize", storage=self.storage)
    def _setup_directories(self):
        os.makedirs(self.out_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)
        
    def _initialize_env_args(self):
        return {
            "env_name": "TradeSimulator-v0",
            "num_envs": self.n_envs,
            "max_step": self.max_steps,
            "state_dim": 10,
            "action_dim": 3,
            "if_discrete": True,
            "max_position": 1,
            "slippage": 7e-7,
            "num_sims": 1,
            "step_gap": 2,
            "env_class": self.env_class,
            "days": [self.start_day_train, self.end_day_train]
        }
        
    def train_agent(self, model_params, learn_params = {}, seed=123):
        env_args = self.env_args.copy()
        env_args["seed"] = seed
        env_args["max_step"] = 480
        env = build_env(self.env_class, env_args, self.gpu_id)
        agent = self.agent_class("MlpPolicy", env, verbose=0, seed=seed, **model_params)
        agent.learn(total_timesteps=self.max_steps * self.n_episodes, progress_bar=True, **learn_params)
        return agent
    
    def evaluate_agent(self, agent, trade_env, seed=123):        
        state, _ = trade_env.reset(seed=seed)
        reward = 0
        for _ in range(trade_env.max_step):
            tensor_state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
            action, _ = agent.predict(tensor_state, deterministic=True)
            state, r, terminal, truncated, _ = trade_env.step(action=action)
            reward += r
            
            if terminal or truncated:
                break
        return reward
                    
    def optimize_hyperparameters(self):
        def objective(trial):
            model_params = SAMPLER[self.agent_class.__name__](trial, n_actions=self.n_actions, n_envs=self.n_envs, additional_args={})
            
            rewards = []
            rewards_train = []
            for seed in self.seeds:
                agent = self.train_agent(model_params, {}, seed=seed)
                
                eval_env_args = self.env_args.copy()
                first_day_val = eval_env_args["days"][-1] + 1
                eval_env_args.update({
                    "eval": True,
                    "days": [first_day_val, first_day_val + self.n_days_val - 1],
                    "num_envs": 1,
                    "num_sims": 1,
                    "env_class": self.eval_env_class,
                    "seed": seed
                })
                eval_env = build_env(self.eval_env_class, eval_env_args, self.gpu_id)
                reward = self.evaluate_agent(agent, eval_env, seed=seed).item()
                
                train_env_args = self.env_args.copy()
                train_env_args.update({
                    "eval": True,
                    "num_envs": 1,
                    "num_sims": 1,
                    "env_class": self.eval_env_class,
                    "seed": seed
                })            
                train_env = build_env(self.env_class, train_env_args, self.gpu_id)
                train_reward = self.evaluate_agent(agent, train_env, seed=seed).item()
                
                
                
                print(f"seed: {seed}, reward: {reward}, train_reward: {train_reward}")
                rewards.append(reward)
                rewards_train.append(train_reward)
            
            if trial.number > 1:
                self._plot_results(trial)
            
            trial.set_user_attr("rewards", rewards)
            trial.set_user_attr("rewards_train", rewards_train)
            
            mean, median, iqm = np.mean(rewards), np.median(rewards), interquartile_mean(rewards)
            print(rewards)
            print(f"Mean: {mean}, Median: {median}, IQM: {iqm}")
            
            return iqm
        
        print(f"Optimizing {self.agent_class.__name__} hyperparameters")
        print(f'Using seeds: {self.seeds}')
        self.study.optimize(objective, n_trials=self.n_trials)
    
    def _plot_results(self, trial):
        plots = [
            ("ParamsOptHistory.png", optuna.visualization.plot_optimization_history(self.study)),
            ("ParamsImportance.png", optuna.visualization.plot_param_importances(self.study)),
            ("ParamsContour.png", optuna.visualization.plot_contour(self.study)),
            ("ParamsSlice.png", optuna.visualization.plot_slice(self.study))
        ]
        for filename, fig in plots:
            fig.write_image(f"{self.plot_dir}/{filename}")
    
    def run(self):
        self.optimize_hyperparameters()

if __name__ == "__main__":
    args = get_cli_args()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = f'{args.out_dir}_{timestamp}'
    plot_dir = f'{out_dir}/plots'
    storage = f'sqlite:///{out_dir}/optuna_study.db'

    num_ignore_step = 60
    step_gap = 2
    slippage = 7e-7
    max_steps = (4800 - num_ignore_step) // step_gap
    # max_steps = 480
    
    optimizer = TradeSimulatorOptimizer(
        agent_class=DQN, # PPO, DQN
        device=device,
        gpu_id=-1,
        out_dir=out_dir,
        plot_dir=plot_dir,
        start_day_train=args.start_day_train,
        end_day_train=args.end_day_train,
        max_steps=max_steps,
        n_seeds=args.n_seeds,
        n_trials=args.n_trials,
        storage=storage,
        n_episodes=50
    )
    optimizer.create_study()
    optimizer.run()