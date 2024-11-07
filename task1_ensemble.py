import json
import os
import numpy as np
import torch as th

from agent.base import AgentBase
from agent.factory import AgentsFactory
from erl_config import build_env
from trade_simulator import EvalTradeSimulator
from metrics import *

PROJECT_FOLDER = "./"
AGENTS_FOLDER = os.path.join(PROJECT_FOLDER, "agents")
os.makedirs(AGENTS_FOLDER, exist_ok=True)
AGENTS_TYPES = ['ppo', 'fqi', 'dqn']

class Ensemble:
    def __init__(self, env_args: dict):
        self.env_args = env_args
        self.device = th.device(f"cuda" if th.cuda.is_available() else "cpu")

    def agents_training(self):
        # Loading agents hyperparameters
        with open(os.path.join(AGENTS_FOLDER, 'agents_args.json'), 'r') as file:
            agents_args = json.load(file)
        # Training agents
        for agent_type in agents_args.keys():
            for agent_name in agents_args[agent_type].keys():
                env_args = self.env_args
                env_args["days"] = agents_args[agent_type][agent_name]["days"]
                if agent_type == "ppo":
                    env_args['n_envs'] = 4
                AgentsFactory.train(
                    {
                        "type": agent_type, 
                        "file": agents_args[agent_type][agent_name]['file'], 
                        "model_args": agents_args[agent_type][agent_name]["model_args"]
                    },
                    env_args=env_args
                )

    def agents_selection(self, num_sims: int = 10):
        # Generating evaluation env
        eval_env_args = self.env_args.copy()
        eval_env_args["num_envs"] = 1
        eval_env_args["num_sims"] = num_sims
        eval_env_args["env_class"] = EvalTradeSimulator
        
        agents_file_names = [x for x in os.listdir(AGENTS_FOLDER) if x.split('_')[0] in AGENTS_TYPES]
        
        results = {
            'best_agents':[]
        }
        for w in range(2, 9):
            curr_agents = [a for a in agents_file_names if f'_w{w}.' in a] # Get agents trained of the previous day
            eval_env_args["days"] = [8 + w , 8 + w]
            eval_env = build_env(eval_env_args["env_class"], eval_env_args, gpu_id=-1)
            
            results[w] = {
                "agents": [],
                "returns_mean": [],
                "returns_std": []
            }
            for agent_file in curr_agents:
                agent_type = agent_file.split('_')[0]
                agent = AgentsFactory.load_agent({"type": agent_type, "file": agent_file})
                returns_mean, returns_std = self.agent_evaluation(agent, eval_env)
                results[w]["agents"].append(agent_file)
                results[w]["returns_mean"].append(returns_mean)
                results[w]["returns_std"].append(returns_std)
            
            if len(results[w]["agents"]) > 0:
                best_idx = np.argmax(results[w]["returns_mean"])
                results['best_agents'].append(curr_agents[best_idx])
        
        with open(os.path.join(AGENTS_FOLDER, 'agents_best.json'), "w") as file:
            json.dump(results, file, indent=4)
    
   
    def agent_evaluation(self, agent: AgentBase, eval_env, seed=0):

        state, _ = eval_env.reset(seed=seed, _if_sequential=False)
        returns = th.zeros(eval_env.num_sims, dtype=th.float32, device=self.device)
            
        for _ in range(eval_env.max_step):
            action = agent.action(state)        
            state, reward, terminated, truncated, _ = eval_env.step(action=action)            
            returns += reward
            if terminated.any() or truncated:
                break
        
        returns_mean = returns.mean().item()
        returns_std = returns.std().item() if eval_env.num_sims > 1 else 0.       
        return returns_mean, returns_std


def run():
    num_sims = 1
    num_ignore_step = 60
    max_position = 1
    step_gap = 2
    slippage = 7e-7

    max_step = 480 #(4800 - num_ignore_step) // step_gap

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

    ensemble_method = Ensemble(env_args)
    ensemble_method.agents_training()
    ensemble_method.agents_selection(num_sims=200)


if __name__ == "__main__":
    run()