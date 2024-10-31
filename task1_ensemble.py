import argparse
import os
import shutil
import time
# from stable_baselines3 import PPO
from sbx import PPO, DQN, DDPG

import torch
import numpy as np
from erl_config import Config, build_env
from erl_replay_buffer import ReplayBuffer
from erl_evaluator import Evaluator
from trade_simulator import TradeSimulator, EvalTradeSimulator
from erl_agent import AgentD3QN, AgentDoubleDQN, AgentTwinD3QN
from collections import Counter
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from metrics import *

def save_tensorboard_plots(log_dir: str, output_dir: str):
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


class Ensemble:
    def __init__(self, log_rules, save_path, starting_cash, agent_classes, args: Config):

        self.log_rules = log_rules

        # ensemble configs
        self.save_path = save_path
        self.starting_cash = starting_cash
        self.current_btc = 0
        self.position = [0]
        self.btc_assets = [0]
        self.net_assets = [starting_cash]
        self.cash = [starting_cash]
        self.agent_classes = agent_classes

        self.from_env_step_is = None

        # args
        self.args = args
        self.agents: list[BaseAlgorithm] = []
        self.thresh = 0.001
        self.num_envs = 1
        self.state_dim = 8 + 2
        # gpu_id = 0
        self.device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
        eval_env_class = args.eval_env_class
        eval_env_class.num_envs = 1

        eval_env_args = args.eval_env_args
        eval_env_args["num_envs"] = 1
        eval_env_args["num_sims"] = 1
        eval_env_args['eval'] = True

        self.trade_env = build_env(eval_env_class, eval_env_args, gpu_id=args.gpu_id)

        self.actions = []

        self.firstbpi = True

    def save_ensemble(self):
        """Saves the ensemble of agents to a directory."""
        for idx, agent in enumerate(self.agents):            
            agent_name = self.agent_classes[idx].__name__
            agent_dir = os.path.join(self.save_path, agent_name, f'{agent_name}.zip')
            agent.save(agent_dir)
        print(f"Ensemble models saved in directory: {self.save_path}")

    def ensemble_train(self):
        args = self.args

        for agent_class in self.agent_classes:
            args.agent_class = agent_class
            agent = self.train_agent(args=args)
            self.agents.append(agent)

        self.save_ensemble()

    def _majority_vote(self, actions):
        """handles tie breaks by returning first element of the most common ones"""
        count = Counter(actions)
        majority_action, _ = count.most_common(1)[0]
        return majority_action
    
    def train_agent(self, args: Config):
        agent_name = args.agent_class.__name__
        agent_dir = os.path.join(self.save_path, agent_name)
        plot_dir = os.path.join(agent_dir, "plots")
        train_logs_dir = os.path.join(agent_dir, "logs")
        tb_dir = os.path.join(train_logs_dir, "tb")
        monitor_dir = os.path.join(train_logs_dir, "monitor")
        os.makedirs(monitor_dir, exist_ok=True)
        
        """init environment"""
        env = build_env(args.env_class, args.env_args, args.gpu_id)  
        env = Monitor(env, monitor_dir, info_keywords=("asset_v", 'mid','new_cash', 'old_cash', "action_exec", "position"))
 
        agent = args.agent_class("MlpPolicy", env, verbose=0, tensorboard_log=tb_dir,
                                 gamma=args.gamma)

        
        total_timesteps = args.max_step * 1000
        n_evals = 10
        eval_freq = int(total_timesteps / n_evals)
        eval_callback = EvalCallback(self.trade_env,
                             log_path=train_logs_dir, eval_freq=eval_freq , n_eval_episodes=20,
                             deterministic=True, render=False)

        agent.learn(total_timesteps=total_timesteps, callback=[eval_callback], progress_bar=True)
        
        tb_dirs_name = [name for name in os.listdir(tb_dir) if os.path.isdir(os.path.join(tb_dir, name))]
        for name in tb_dirs_name:   
            save_tensorboard_plots(os.path.join(tb_dir, name), os.path.join(plot_dir, "tb", name))
        
        return agent


def run(save_path, agent_list, days, log_rules=False):
    gpu_id = -1

    num_sims = 1
    num_ignore_step = 60
    max_position = 1
    step_gap = 2
    slippage = 7e-7

    # max_step = (4800 - num_ignore_step) // step_gap
    max_step = 480

    env_args = {
        "env_name": "TradeSimulator-v0",
        "num_envs": num_sims,
        "max_step": max_step,
        "state_dim": 8 + 2,  # factor_dim + (position, holding)
        "action_dim": 3,  # long, 0, short
        "if_discrete": True,
        "max_position": max_position,
        "slippage": slippage,
        "num_sims": num_sims,
        "step_gap": step_gap,
        "eval": False,
        "days": days,
    }
    args = Config(agent_class=None, env_class=TradeSimulator, env_args=env_args)
    args.gpu_id = gpu_id
    args.random_seed = gpu_id


    args.eval_env_class = EvalTradeSimulator
    args.eval_env_args = env_args.copy()
    args.eval_env_args["eval"] = True
    eval_day = env_args['days'][-1] + 1
    args.eval_env_args['days'] = [eval_day, eval_day]

    ensemble_env = Ensemble(
        log_rules,
        save_path,
        1e6,
        agent_list,
        args,
    )
    ensemble_env.ensemble_train()

def get_cli_args():
    """Create CLI parser and return parsed arguments"""
    parser = argparse.ArgumentParser()
    # Example-specific args.
    parser.add_argument(
        '--start_day',
        type=int,
        default=7,
        help="starting day (included) "
    )

    parser.add_argument(
        '--end_day',
        type=int,
        default=15,
        help="ending day (included) "
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_cli_args()
    start_day, end_day = args.start_day, args.end_day
    print(start_day, end_day)
    agent_list = [PPO, DQN]

    agent_names = sorted([x.__name__ for x in agent_list])
    save_path = f'experiments/ensemble_polimi/train/{start_day}_{end_day}'
    
    
    run(
        save_path,
        agent_list,
        [start_day, end_day],
    )
