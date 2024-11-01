import os
import copy
import torch
import numpy as np
import sys

from tqdm import tqdm

from agent.base import AgentBase
from agent.factory import AgentsFactory
from erl_config import Config, build_env
from trade_simulator import EvalTradeSimulator
from metrics import sharpe_ratio, max_drawdown, return_over_max_drawdown
from oamp.oamp import OAMP
from oamp.oamp_config import ConfigOAMP

PROJECT_FOLDER = "."

AGENTS_FOLDER = os.path.join(PROJECT_FOLDER, "agents")
os.makedirs(AGENTS_FOLDER, exist_ok=True)
RUNS_FOLDER = os.path.join(PROJECT_FOLDER, "runs")
os.makedirs(RUNS_FOLDER, exist_ok=True)


def to_python_number(x):
    if isinstance(x, torch.Tensor):
        return x.cpu().item()
    else:
        return x


def trade(action, price, cur_cash, cur_btc):
    if action == 1:
        new_cash = cur_cash - price
        new_btc = cur_btc + 1
    elif action == -1:
        new_cash = cur_cash + price
        new_btc = cur_btc - 1
    else:
        new_cash = cur_cash
        new_btc = cur_btc
    return new_cash, new_btc


def winloss(action, last_price, price):
    if action > 0:
        if last_price < price:
            correct_pred = 1
        elif last_price > price:
            correct_pred = -1
        else:
            correct_pred = 0
    elif action < 0:
        if last_price < price:
            correct_pred = -1
        elif last_price > price:
            correct_pred = 1
        else:
            correct_pred = 0
    else:
        correct_pred = 0
    return correct_pred


class EnsembleEvaluator:
    def __init__(
        self,
        run_name,
        agents_info,
        oamp_args,
        args: Config,
    ):
        self.save_path = os.path.join(RUNS_FOLDER, run_name)
        os.makedirs(self.save_path, exist_ok=True)
        # Initializing trading env
        self.args = args
        self.device = torch.device("cpu") #torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.trade_env = build_env(args.env_class, args.env_args, gpu_id=args.gpu_id)
        # Initializing trading agents
        self.agents_info = agents_info
        self.agents_names = []
        self.agents: list[AgentBase] = []
        self.ensemble: OAMP = OAMP(len(agents_info), oamp_args)
        # Initializing trading portfolio
        self.current_btc = 0
        self.cash = [args.starting_cash]
        self.btc_assets = [0]
        self.net_assets = [args.starting_cash]
        self.starting_cash = args.starting_cash

    def load_agents(self):
        # Loading trading agents
        for agent_name, agent_info in self.agents_info.items():
            self.agents_names.append(agent_name)
            self.agents.append(AgentsFactory.load_agent(agent_info))

    def multi_trade(self):
        # Initializing trading history
        positions = []
        action_ints = []
        correct_pred = []
        current_btcs = [self.current_btc]
        # Initializing last state and price
        last_state, _ = self.trade_env.reset(eval=True)
        last_price = 0
        # Initializing trading agents rewards
        agents_rewards_old = [0] * len(self.agents)
        # Trading
        return_ = 0
        for _ in tqdm(range(self.trade_env.max_step)):
            agents_actions = []
            agents_rewards = []
            # Collecting actions from each agent
            for agent in self.agents:
                # Computing agent curr action
                agent_action = agent.action(last_state)
                agents_actions.append(agent_action)
                # Computing agent last reward
                agent_env = copy.deepcopy(self.trade_env)
                _, agent_reward, _, _, _ = agent_env.step(agent_action)
                agents_rewards.append(agent_reward.item())
            # Computing ensemble action
            action = self._ensemble_action(agents_actions, agents_rewards_old)
            agents_rewards_old = agents_rewards
            action_int = action - 1
            state, reward, _, _, _ = self.trade_env.step(action=action)
            # Upadting trading portfolio
            price = self.trade_env.price_ary[self.trade_env.step_i, 2].to(self.device)
            new_cash, self.current_btc = trade(
                action_int, price, self.cash[-1], self.current_btc
            )
            self.cash.append(new_cash)
            self.btc_assets.append((self.current_btc * price).item())
            self.net_assets.append(
                (to_python_number(self.btc_assets[-1]) + to_python_number(new_cash))
            )
            # Upadting trading history
            positions.append(self.trade_env.position)
            action_ints.append(action_int)
            current_btcs.append(self.current_btc)
            correct_pred.append(winloss(action_int, last_price, price))
            # Updating last state and price
            last_state = state
            last_price = price
            return_ += reward
        # Saving trading history
        np.save(
            os.path.join(self.save_path, "positions.npy"),
            positions,
        )
        np.save(
            os.path.join(self.save_path, "net_assets.npy"),
            np.array(self.net_assets),
        )
        np.save(
            os.path.join(self.save_path, "btc_positions.npy"),
            np.array(self.btc_assets),
        )
        np.save(
            os.path.join(self.save_path, "correct_predictions.npy"),
            np.array(correct_pred),
        )
        # Computing trading metrics
        returns = np.diff(self.net_assets) / self.net_assets[:-1]
        final_sharpe_ratio = sharpe_ratio(returns)
        final_max_drawdown = max_drawdown(returns)
        final_roma = return_over_max_drawdown(returns)
        print(f"Sharpe Ratio: {final_sharpe_ratio}")
        print(f"Max Drawdown: {final_max_drawdown}")
        print(f"Return over Max Drawdown: {final_roma}")
        self.ensemble.plot_stats(self.save_path)
        return final_sharpe_ratio, return_

    def _ensemble_action(self, actions, rewards):
        return self.ensemble.step(np.array(rewards), np.array(actions))


def run_evaluation(
    run_name: str,
    agents_info: dict,
    oamp_args: dict=None,
    env_args=None
):

    gpu_id =-1  # Get GPU_ID from command line arguments
    if env_args is None:
        num_sims = 1
        num_ignore_step = 60
        step_gap = 2
        max_step = (4800 - num_ignore_step) // step_gap
        max_position = 1
        slippage = 7e-7
        # max_ste not used but set to full len
        env_args = {
            "env_name": "TradeSimulator-v0",
            "num_envs": num_sims,
            "num_sims": num_sims,
            "max_step": max_step,
            "step_gap": step_gap,
            "state_dim": 8 + 2,
            "action_dim": 3,
            "if_discrete": True,
            "max_position": max_position,
            "slippage": slippage,
            "dataset_path": "data\BTC_1sec_predict.npy",  # Replace with your evaluation dataset path
            "days": [15, 16],
            "eval_sequential": True
        }
    args = Config(agent_class=None, env_class=EvalTradeSimulator, env_args=env_args)
    args.gpu_id = gpu_id
    args.random_seed = gpu_id
    args.starting_cash = 1e6

    oamp_args = ConfigOAMP(oamp_args)

    ensemble_evaluator = EnsembleEvaluator(
        run_name,
        agents_info,
        oamp_args,
        args,
    )
    ensemble_evaluator.load_agents()
    return ensemble_evaluator.multi_trade()


if __name__ == "__main__":
    RUN_NAME = "oamp"
    AGENTS_INFO = {
        "agent_0": {
            'type': 'fqi',
            'file': 'agent_0.pkl',
        },
        "agent_1": {
            'type': 'fqi',
            'file': 'agent_1.pkl',
        },
        "agent_2": {
            'type': 'fqi',
            'file': 'agent_2.pkl',
        },
        "agent_3": {
            'type': 'fqi',
            'file': 'agent_3.pkl',
        },
    }
    OAMP_ARGS = {}
    run_evaluation(RUN_NAME, AGENTS_INFO, OAMP_ARGS)
