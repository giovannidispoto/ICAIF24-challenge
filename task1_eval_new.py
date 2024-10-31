import os
# from stable_baselines3 import PPO
from sbx import PPO

import torch
import numpy as np
from erl_config import Config, build_env
from trade_simulator import EvalTradeSimulator
from erl_agent import AgentD3QN, AgentDoubleDQN, AgentTwinD3QN
from collections import Counter
from metrics import sharpe_ratio, max_drawdown, return_over_max_drawdown
from stable_baselines3.common.base_class import BaseAlgorithm


def to_python_number(x):
    if isinstance(x, torch.Tensor):
        return x.cpu().item()
    else:
        return x


class EnsembleEvaluator:
    def __init__(self, save_path, agent_classes, args: Config):
        self.save_path = save_path
        self.ensemble_path = os.path.join(save_path, "ensemble_models")
        self.agent_classes = agent_classes

        # args
        self.args = args
        self.agents = []
        self.thresh = 0.001
        self.num_envs = 1
        self.state_dim = 8 + 2
        self.device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

        self.trade_env = build_env(args.env_class, args.env_args, gpu_id=args.gpu_id)

        self.current_btc = 0
        self.cash = [args.starting_cash]
        self.btc_assets = [0]
        # self.net_assets = [torch.tensor(args.starting_cash, device=self.device)]
        self.net_assets = [args.starting_cash]
        self.starting_cash = args.starting_cash
        self.deterministic = args.deterministic
        
    def load_agents(self):
        for agent_class in self.agent_classes:
            agent_name = agent_class.__name__
            agent_dir = os.path.join(self.ensemble_path, agent_name, f"{agent_name}.zip")
            agent = agent_class.load(agent_dir)
            print(agent == None)
            self.agents.append(agent)

    def multi_trade(self):
        """Evaluation loop using ensemble of agents"""

        agents: list[BaseAlgorithm] = self.agents
        trade_env = self.trade_env
        state, _ = trade_env.reset()

        last_state = state
        last_price = 0

        positions = []
        action_ints = []
        correct_pred = []
        current_btcs = [self.current_btc]

        for _ in range(trade_env.max_step):
            actions = []
            intermediate_state = last_state

            
            for agent in agents:
                tensor_state = torch.as_tensor(intermediate_state, dtype=torch.float32, device=agent.device)
                action, _ = agent.predict(tensor_state, deterministic=self.deterministic)
                actions.append(action)

            action = self._ensemble_action(actions=actions)
            action_int = action.item() - 1

            state, reward, terminal, truncated, _ = trade_env.step(action=action)

            action_ints.append(action_int)
            positions.append(trade_env.position)

            # Manually compute cumulative returns
            mid_price = trade_env.price_ary[trade_env.step_i, 2].to(self.device)

            new_cash = self.cash[-1]

            if action_int > 0 and self.cash[-1] > mid_price:  # Buy
                last_cash = self.cash[-1]
                new_cash = last_cash - mid_price
                self.current_btc += 1
            elif action_int < 0 and self.current_btc > 0:  # Sell
                last_cash = self.cash[-1]
                new_cash = last_cash + mid_price
                self.current_btc -= 1

            self.cash.append(new_cash)
            self.btc_assets.append((self.current_btc * mid_price).item())
            self.net_assets.append((to_python_number(self.btc_assets[-1]) + to_python_number(new_cash)))

            last_state = state

            # Log win rate
            if action_int == 1:
                correct_pred.append(1 if last_price < mid_price else -1 if last_price > mid_price else 0)
            elif action_int == -1:
                correct_pred.append(-1 if last_price < mid_price else 1 if last_price > mid_price else 0)
            else:
                correct_pred.append(0)

            last_price = mid_price
            current_btcs.append(self.current_btc)

        # Save results
        np.save(f"{self.save_path}/positions.npy", positions)
        np.save(f"{self.save_path}/net_assets.npy", np.array(self.net_assets))
        np.save(f"{self.save_path}/btc_positions.npy", np.array(self.btc_assets))
        np.save(f"{self.save_path}/correct_predictions.npy", np.array(correct_pred))

        # Compute metrics
        returns = np.diff(self.net_assets) / self.net_assets[:-1]
        final_sharpe_ratio = sharpe_ratio(returns)
        final_max_drawdown = max_drawdown(returns)
        final_roma = return_over_max_drawdown(returns)

        print(f'P&L: {returns.sum() * 100}%')
        print(f'Net gain: {self.net_assets[-1] - self.starting_cash}, Starting cash: {self.starting_cash}, Ending cash: {self.net_assets[-1]}')
        print(f"Sharpe Ratio: {final_sharpe_ratio}")
        print(f"Max Drawdown: {final_max_drawdown}")
        print(f"Return over Max Drawdown: {final_roma}")

    def _ensemble_action(self, actions):
        """Returns the majority action among agents. Our code uses majority voting, you may change this to increase performance."""
        count = Counter([a.item() for a in actions])
        majority_action, _ = count.most_common(1)[0]
        return torch.tensor([majority_action], dtype=torch.int32)


def run_evaluation(save_path, agent_list):
    import sys

    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else -1  # Get GPU_ID from command line arguments

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
        "state_dim": 8 + 2,
        "action_dim": 3,
        "if_discrete": True,
        "max_position": max_position,
        "slippage": slippage,
        "num_sims": num_sims,
        "step_gap": step_gap,
        "days": [9, 9], #[ 7  8  9 10 11 12 13 14 15 16 17]
        "deterministic": True,
        "eval": True,
    }
    args = Config(agent_class=None, env_class=EvalTradeSimulator, env_args=env_args)
    args.gpu_id = gpu_id
    args.random_seed = gpu_id
    args.net_dims = (128, 128, 128)
    args.starting_cash = 1e6

    ensemble_evaluator = EnsembleEvaluator(
        save_path,
        agent_list,
        args,
    )
    ensemble_evaluator.load_agents()
    ensemble_evaluator.multi_trade()


if __name__ == "__main__":
    save_path = "experiments/ensemble_polimi/"    
    agent_list = [PPO]
    run_evaluation(save_path, agent_list)


