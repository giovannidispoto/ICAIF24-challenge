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

def can_buy(action, mid_price, cash, current_btc):
    if action == 1 and cash > mid_price:  # can buy
        last_cash = cash
        new_cash = last_cash - mid_price
        current_btc += 1
    elif action == -1 and current_btc > 0:  # can sell
        last_cash = cash
        new_cash = last_cash + mid_price
        current_btc -= 1
    else:
        new_cash = cash

    return new_cash, current_btc


def winloss(action, last_price, mid_price):
    if action > 0:
        if last_price < mid_price:
            correct_pred = 1
        elif last_price > mid_price:
            correct_pred = -1
        else:
            correct_pred = 0
    elif action < 0:
        if last_price < mid_price:
            correct_pred = -1
        elif last_price > mid_price:
            correct_pred = 1
        else:
            correct_pred = 0
    else:
        correct_pred = 0
    return correct_pred


class Ensemble:
    def __init__(self, log_rules, save_path, starting_cash, agent_classes, args: Config):

        self.log_rules = log_rules

        # ensemble configs
        self.save_path = save_path
        self.ensemble_path = os.path.join(save_path, "ensemble_models")
        shutil.rmtree(self.ensemble_path)
        os.makedirs(self.ensemble_path, exist_ok=True)
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
            agent_dir = os.path.join(self.ensemble_path, agent_name, f'{agent_name}.zip')
            agent.save(agent_dir)
        print(f"Ensemble models saved in directory: {self.ensemble_path}")

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
        agent_dir = os.path.join(self.ensemble_path, agent_name)
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

        
        total_timesteps = args.max_step * 500
        n_evals = 5
        eval_freq = int(total_timesteps / n_evals)
        eval_callback = EvalCallback(self.trade_env,
                             log_path=train_logs_dir, eval_freq=eval_freq , n_eval_episodes=100,
                             deterministic=True, render=False)

        agent.learn(total_timesteps=total_timesteps, callback=[eval_callback], progress_bar=True)
        
        tb_dirs_name = [name for name in os.listdir(tb_dir) if os.path.isdir(os.path.join(tb_dir, name))]
        for name in tb_dirs_name:   
            save_tensorboard_plots(os.path.join(tb_dir, name), os.path.join(plot_dir, "tb", name))
        
        return agent


def run(save_path, agent_list, log_rules=False):
    import sys
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else -1  # 从命令行参数里获得GPU_ID

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
        "days": [7],
    }
    args = Config(agent_class=None, env_class=TradeSimulator, env_args=env_args)
    args.gpu_id = gpu_id
    args.random_seed = gpu_id
    args.net_dims = (128, 128, 128)

    args.gamma = 0.995
    args.explore_rate = 0.005
    args.state_value_tau = 0.01
    args.soft_update_tau = 2e-6
    args.learning_rate = 2e-6
    args.batch_size = 512
    args.break_step = int(32)  # TODO reset to 32e4
    args.buffer_size = int(max_step * 32)
    args.repeat_times = 2
    args.horizon_len = int(max_step * 4)
    args.eval_per_step = int(max_step)
    args.num_workers = 1
    args.save_gap = 8

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


if __name__ == "__main__":

    run(
        "experiments/ensemble_polimi",
        [PPO]
    )
