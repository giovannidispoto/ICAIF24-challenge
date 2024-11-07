import os
import time
import torch
import numpy as np

from agent.factory import AgentsFactory
from erl_config import Config, build_env
from erl_replay_buffer import ReplayBuffer
from erl_evaluator import Evaluator
from trade_simulator import TradeSimulator, EvalTradeSimulator
from erl_agent import AgentD3QN, AgentDoubleDQN, AgentTwinD3QN
from collections import Counter
import pickle

from metrics import *

PROJECT_FOLDER = "./"

AGENTS_FOLDER = os.path.join(PROJECT_FOLDER, "agents")
os.makedirs(AGENTS_FOLDER, exist_ok=True)

class Ensemble:
    def __init__(self, starting_cash, hyperparameters, env_args):


        # ensemble configs
        self.starting_cash = starting_cash
        self.current_btc = 0
        self.position = [0]
        self.btc_assets = [0]
        self.net_assets = [starting_cash]
        self.cash = [starting_cash]
        self.hyperparameters = hyperparameters
        self.from_env_step_is = None
        # args
        self.agents = []
        self.thresh = 0.001
        self.num_envs = 1
        self.state_dim = 8 + 2
        self.env_args = env_args
        # gpu_id = 0
        self.device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
        self.trade_env  = build_env(TradeSimulator, env_args, -1)


    def save_ensemble(self):
        """Saves the ensemble of agents to a directory."""
        ensemble_dir = os.path.join(self.save_path, "ensemble_models")
        os.makedirs(ensemble_dir, exist_ok=True)
        for idx, agent in enumerate(self.agents):
            agent_name = self.agent_classes[idx].__name__
            agent_dir = os.path.join(ensemble_dir, agent_name)
            os.makedirs(agent_dir, exist_ok=True)
            agent.save_or_load_agent(agent_dir, if_save=True)
        print(f"Ensemble models saved in directory: {ensemble_dir}")

    def ensemble_train(self):
        for agent_type in self.hyperparameters.keys():
            for agent_name in self.hyperparameters[agent_type].keys():
                env_args = self.env_args
                env_args["days"] = self.hyperparameters[agent_type][agent_name]["days"]
                if agent_type == "ppo":
                    env_args['n_envs'] = 4
                AgentsFactory.train({"type": agent_type, "file":  self.hyperparameters[agent_type][agent_name]['file'], "model_args": self.hyperparameters[agent_type][agent_name]["model_args"]}, env_args=env_args)




def run(hyperparameters, log_rules=False):
    import sys

    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else -1  # 从命令行参数里获得GPU_ID

    from erl_agent import AgentD3QN

    num_sims = 1
    num_ignore_step = 60
    max_position = 1
    step_gap = 2
    slippage = 7e-7

    max_step = (4800 - num_ignore_step) // step_gap

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
    }

    ensemble_env = Ensemble(
        1e6,
        hyperparameters,
        env_args
    )
    ensemble_env.ensemble_train()


if __name__ == "__main__":

    hyperparameters = {
        "ppo": {
            "w1": {
                "file":"./agents/ppo_w1.zip",
                "days" : [9, 9],
                "model_args": { },
            }

        },
        "fqi" : {
            "w1": {
                "file" : "./agents/fqi_w1.pkl",
                "days" : [9, 9],
                "model_args": {
                    "n_estimators": 100,
                    "max_depth": 20,
                    "iterations": 3,
                    "min_samples_split": 10000
                },

            },
            "w2": {
                "file": "./agents/fqi_w2.pkl",
                "days": [10, 10],
                "model_args": {
                    "n_estimators": 100,
                    "max_depth": 20,
                    "iterations": 3,
                    "min_samples_split": 10000
                },
            },
            "w3": {
                "file": "./agents/fqi_w3.pkl",
                "days": [11, 11],
                "model_args": {
                    "n_estimators": 100,
                    "max_depth": 20,
                    "iterations": 3,
                    "min_samples_split": 10000
                },
            },
            "w4": {
                "file": "./agents/fqi_w4.pkl",
                "days": [12, 12],
                "model_args": {
                    "n_estimators": 100,
                    "max_depth": 20,
                    "iterations": 3,
                    "min_samples_split": 10000
                },
            },
            "w5": {
                "file": "./agents/fqi_w5.pkl",
                "days": [13, 13],
                "model_args": {
                    "n_estimators": 100,
                    "max_depth": 20,
                    "iterations": 3,
                    "min_samples_split": 10000
                },
            },
            "w6": {
                "file": "./agents/fqi_w6.pkl",
                "days": [14, 14],
                "model_args": {
                    "n_estimators": 100,
                    "max_depth": 20,
                    "iterations": 3,
                    "min_samples_split": 10000
                },
            },
            "w7": {
                "file" : "./agents/fqi_w7.pkl",
                "days": [15, 15],
                "model_args": {
                    "n_estimators": 100,
                    "max_depth": 20,
                    "iterations": 3,
                    "min_samples_split": 10000
                },
            },
            "w8": {
                "file": "./agents/fqi_w8.pkl",
                "days": [16, 16],
                "model_args": {
                    "n_estimators": 100,
                    "max_depth": 20,
                    "iterations": 3,
                    "min_samples_split": 10000
                },
            },
            "w9": {
                "file": "./agents/fqi_w9.pkl",
                "days": [17, 17],
                "model_args": {
                    "n_estimators": 100,
                    "max_depth": 20,
                    "iterations": 3,
                    "min_samples_split": 10000
                },
            },
        }
    }

    run(
        hyperparameters,
    )
