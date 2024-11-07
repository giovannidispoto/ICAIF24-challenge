import json
import os
import time
from typing import Optional
import torch as th
import numpy as np

from agent.base import AgentBase
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
        self.device = th.device(f"cuda" if th.cuda.is_available() else "cpu")

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
    
    def evaluate_agent(self, agent: AgentBase, eval_env, eval_sequential: bool = False, verbose: int = 0):
        num_eval_sims = eval_env.num_sims

        state, _ = eval_env.reset(seed=eval_env.seed, eval_sequential=eval_sequential)
        
        total_reward = th.zeros(num_eval_sims, dtype=th.float32, device=self.device)
        rewards = th.empty((0, num_eval_sims), dtype=th.float32, device=self.device)
        
            
        for i in range(eval_env.max_step):
            
            action = agent.action(state)
            action = th.from_numpy(action).to(self.device)            
            state, reward, terminated, truncated, _ = eval_env.step(action=action)
            
            rewards = th.cat((rewards, reward.unsqueeze(0)), dim=0)
                
            total_reward += reward

            if terminated.any() or truncated:
                break
        
        
        mean_total_reward = total_reward.mean().item()
        std_total_reward = total_reward.std().item() if num_eval_sims > 1 else 0.
        mean_std_steps = rewards.std(dim=0).mean().item()
        
        if verbose:
            print(f'Sims mean: {mean_total_reward} Sims std: {std_total_reward}, Mean std steps: {mean_std_steps}')
        
        
        return mean_total_reward, std_total_reward, mean_std_steps


    def model_selection(self, agent_path: str, num_sims: int = 10, eval_sequential: bool = False, save_path: Optional[str] = None):
        eval_env_args = self.env_args.copy()
        eval_env_args["num_envs"] = 1
        eval_env_args["num_sims"] = num_sims
        eval_env_args["eval_sequential"] = eval_sequential
        eval_env_args["eval"] = True
        eval_env_args["env_class"] = EvalTradeSimulator
        
        agent_file_names = [x for x in os.listdir(agent_path) if x.split('_')[0] in ['ppo', 'fqi', 'dqn']]
        
        print(f'All found agents: {agent_file_names}')
        results = {}
        for w in range(1, 8):
            curr_agents = [a for a in agent_file_names if f'_w{w-1}.' in a] # Get agents trained of the previous day
            
            curr_eval_env_args = eval_env_args.copy()
            curr_eval_env_args["days"] = [w + 7, w + 7]
            eval_env = build_env(curr_eval_env_args["env_class"], curr_eval_env_args, gpu_id=-1)
            
            results[w] = {
                "agents": [],
                "mean_total_rewards": [],
                "std_simulations": []
            }
            for agent_file in curr_agents:
                agent_type = agent_file.split('_')[0]
                agent = AgentsFactory.load_agent({"type": agent_type, "file": os.path.join(agent_path, agent_file)})
                print(f"Evaluating {agent_file.split('.')[0]} on window {w}")
                mean_total_reward, std_simulations, mean_std_steps = self.evaluate_agent(agent, eval_env, eval_sequential, verbose=1)
                results[w]["agents"].append(agent_file)
                results[w]["mean_total_rewards"].append(mean_total_reward)
                results[w]["std_simulations"].append(std_simulations)
                results[w]["mean_std_steps"] = mean_std_steps
                # print(f'Agent: {agent_file} Mean Total Reward: {mean_total_reward} Std Simulations: {std_simulations} Mean std steps: {mean_std_steps}')
            if len(results[w]["agents"]) > 0:
                best_idx = np.argmax(results[w]["mean_total_rewards"])
                results[w]["best_agent"] = results[w]["agents"][best_idx]
                results[w]["best_mean_total_reward"] = results[w]["mean_total_rewards"][best_idx]
        
        if save_path is not None:
            with open(save_path, "w") as file:
                json.dump(results, file, indent=4)
            
        return results


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
    
    # model_selection_results = ensemble_env.model_selection(AGENTS_FOLDER, num_sims=10, eval_sequential=False, save_path=f"{AGENTS_FOLDER}/model_selection_results.json")


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