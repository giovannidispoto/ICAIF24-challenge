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
EXPERIMENTS_FOLDER = os.path.join(PROJECT_FOLDER, "experiments")
os.makedirs(EXPERIMENTS_FOLDER, exist_ok=True)


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
        self.save_path = os.path.join(EXPERIMENTS_FOLDER, run_name)
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
        print(f"loaded {self.agents_names}")

    def multi_trade(self):
        # Initializing trading history
        positions = []
        action_ints = []
        correct_pred = []
        current_btcs = [self.current_btc]
        # Initializing last state and price
        # last_state, _ = self.trade_env.reset(eval_sequential=True)
        last_price = 0
        # Initializing trading agents rewards
        agents_rewards_old = [0] * len(self.agents)
        # Trading
        envs = []
        last_states = []
        for i in range(len(self.agents)):
            env = build_env(self.args.env_class, self.args.env_args, gpu_id=self.args.gpu_id)
            envs.append(env)
            last_state, _ = envs[i].reset(eval_sequential=True)
            last_states.append(last_state)

        return_ = np.zeros(len(self.agents))
        for step_ in tqdm(range(self.trade_env.max_step)):
            agents_actions = []
            agents_rewards = []
            # Collecting actions from each agent
            states = []
            for i, agent in enumerate(self.agents):
                # Computing agent curr action
                agent_action = agent.action(last_states[i])
                # agent_action = np.random.choice(3)
                agents_actions.append(agent_action)
                # Computing agent last reward
                # agent_env = copy.deepcopy(self.trade_env)
                state, agent_reward, _, _, _ = envs[i].step(agent_action)
                # state , rewards, terminals, truncates, info_dict = self.trade_env.step(np.array(agents_actions))
                states.append(state)
                agents_rewards.append(agent_reward)
            # print(agents_rewards)
            self.ensemble.stats['rewards'].append(agents_rewards)


            last_states = states
            # last_price = price
            # return_ += rewards.numpy()
            if step_ % 100 == 0:
                # np.save(
                #     os.path.join(self.save_path, "positions.npy"),
                #     positions,
                # )
                # np.save(
                #     os.path.join(self.save_path, "net_assets.npy"),
                #     np.array(self.net_assets),
                # )
                # np.save(
                #     os.path.join(self.save_path, "btc_positions.npy"),
                #     np.array(self.btc_assets),
                # )
                # np.save(
                #     os.path.join(self.save_path, "correct_predictions.npy"),
                #     np.array(correct_pred),
                # )
                self.ensemble.plot_stats(self.save_path, independent=True, agent_names=self.agents_names)

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
        self.ensemble.plot_stats(self.save_path, independent=True, agent_names=self.agents_names)
        return final_sharpe_ratio, return_

    def _ensemble_action(self, actions, rewards):
        return self.ensemble.step(np.array(rewards), np.array(actions))


def run_evaluation(
    run_name: str,
    agents_info: dict,
    oamp_args: dict=None,
    env_args=None,
    days=None
):

    gpu_id =-1  # Get GPU_ID from command line arguments
    if env_args is None:
        num_sims = 1 #len(agents_info.keys())
        num_ignore_step = 60
        step_gap = 2
        max_step = (4800 - num_ignore_step) // step_gap
        max_position = 1
        slippage = 7e-7
        if days is None:
            days = [16, 16]
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
            "days": days,
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
    agent_dir = "results_agents/results_agents/completed/results/saved_agents/"
    AGENTS_INFO = {}
    for i in range(5):
        AGENTS_INFO[f"dqn_{i}"] = {"type": "dqn", "file": agent_dir + f"DQN_window_{i}"}
        AGENTS_INFO[f"ppo_{i}"] = {"type": "ppo", "file": agent_dir + f"PPO_window_{i}"}
    agent_dir = "ppos_new/"
    AGENTS_INFO = {}
    for i in range(2):
        AGENTS_INFO[f"ppo_{i}"] = {"type": "ppo", "file": agent_dir + f"{1}"}
    # for i in range(3):
    #     AGENTS_INFO[f"fqi_{i}"] = {"type": "fqi", "file": agent_dir + f"fqi/{i+1}.pkl"}

    agent_dir = "ppos/"
    AGENTS_INFO = {}
    # for i in range(5):
    #     AGENTS_INFO[f"ppo_{i}"] = {"type": "ppo", "file": agent_dir + f"{i+1}"}
    AGENTS_INFO[f"lo"] = {"type": "lo", "file": ""}
    AGENTS_INFO[f"sho"] = {"type": "sho", "file": ""}
    AGENTS_INFO[f"random"] = {"type": "random", "file": ""}

    OAMP_ARGS = {}
    RUN_NAME = "oamp_7_baselines"
    days = [7, 7]
    run_evaluation(RUN_NAME, AGENTS_INFO, OAMP_ARGS, days=days)
