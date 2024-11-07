import os
import torch
import numpy as np

from tqdm import tqdm

from agent.base import AgentBase
from agent.factory import AgentsFactory
from erl_config import Config, build_env
from trade_simulator import EvalTradeSimulator
from metrics import sharpe_ratio, max_drawdown, return_over_max_drawdown
from oamp.oamp import OAMP
from oamp.oamp_config import ConfigOAMP

PROJECT_FOLDER = "/home/trading/antonio/ICAIF24-challenge"

AGENTS_FOLDER = os.path.join(PROJECT_FOLDER, "agents")
os.makedirs(AGENTS_FOLDER, exist_ok=True)
EXPERIMENTS_FOLDER = os.path.join(PROJECT_FOLDER, "experiments")
os.makedirs(EXPERIMENTS_FOLDER, exist_ok=True)


def to_python_number(x):
    if isinstance(x, torch.Tensor):
        return x.cpu().item()
    else:
        return x


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
        self.agents: list[AgentBase] = []
        self.agents_info = agents_info
        self.agents_envs = []
        self.agents_names = []
        self.ensemble: OAMP = OAMP(args.env_args['max_step'], len(agents_info), oamp_args)
        # Initializing trading portfolio
        self.current_btc = 0
        self.cash = [args.starting_cash]
        self.btc_assets = [0]
        self.net_assets = [args.starting_cash]
        self.starting_cash = args.starting_cash

    def load_agents(self):
        # Loading trading agents
        for agent_name, agent_info in self.agents_info.items():
            self.agents.append(AgentsFactory.load_agent(agent_info))
            self.agents_envs.append(
                build_env(
                    self.args.env_class, 
                    self.args.env_args, 
                    gpu_id=self.args.gpu_id
                )
            )
            self.agents_names.append(agent_name)

    def multi_trade(self, evaluation_steps_count):
        # Initializing trading history
        positions = []
        action_ints = []
        correct_pred = []
        current_btcs = [self.current_btc]
        # Initializing last state and price
        last_price = 0.0
        last_state = self.trade_env.reset()[0]
        agents_last_state = [agent_env.reset()[0] for agent_env in self.agents_envs]
        # Initializing trading agents rewards
        reward = 0.0
        agents_rewards_old = [0.0] * len(self.agents)
        # Trading
        for step in tqdm(range(evaluation_steps_count)):
            if step > 0 and step % self.args.env_args['max_step'] == 0:
                self.ensemble.update_agents_weights()
            agents_actions = []
            agents_rewards = []
            # Collecting actions from each agent
            for ai, (agent, agent_env) in enumerate(zip(self.agents, self.agents_envs)):
                # Computing agent current action and reward
                agent_action = agent.action(agents_last_state[ai])
                agent_state, agent_reward, _, _, info, = agent_env.step(agent_action)
                agents_actions.append(info['action'])
                agents_rewards.append(agent_reward.item())
                # Updating agent last state
                agents_last_state[ai] = agent_state
            # Computing ensemble action
            action = self._ensemble_action(agents_actions, agents_rewards_old, reward)
            agents_rewards_old = agents_rewards
            action_int = action - 1
            last_state, reward, _, _, info = self.trade_env.step(action=action)
            # Upadting trading portfolio
            new_cash = info['new_cash']
            price = info['price']
            self.cash.append(self.starting_cash + new_cash)
            self.btc_assets.append(self.current_btc * price)
            self.net_assets.append(
                (to_python_number(self.btc_assets[-1]) + to_python_number(self.cash[-1]))
            )
            # Upadting trading history
            positions.append(self.trade_env.position)
            action_ints.append(action_int)
            current_btcs.append(self.current_btc)
            correct_pred.append(winloss(action_int, last_price, price))
            # Updating last price
            last_price = price
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
        self.ensemble.plot_stats(self.save_path, self.agents_names)

    def _ensemble_action(self, actions, rewards, reward):
        if not isinstance(reward, float):
            reward = reward.item()
        return self.ensemble.step(np.array(rewards), np.array(actions), reward)


def run_evaluation(
    run_name: str,
    agents_info: dict,
    oamp_args: dict=None,
):
    evaluation_steps_count = 5000

    gpu_id = -1

    num_sims = 1
    num_ignore_step = 60
    step_gap = 2
    max_step = 50
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
        "dataset_path": "data/BTC_1sec_predict.npy",  # Replace with your evaluation dataset path
        "days": [7, 8],
    }

    args = Config(agent_class=None, env_class=EvalTradeSimulator, env_args=env_args)
    args.gpu_id = gpu_id
    args.random_seed = gpu_id
    args.starting_cash = 1e6

    oamp_args = ConfigOAMP(oamp_args)

    ensemble_evaluator = EnsembleEvaluator(
        'test',
        agents_info,
        oamp_args,
        args,
    )
    ensemble_evaluator.load_agents()
    ensemble_evaluator.multi_trade(evaluation_steps_count)


if __name__ == "__main__":
    RUN_NAME = "test"
    AGENTS_INFO = {
        "agent_0": {
            'type': 'fqi',
            'file': "fqi_w2.pkl",
        },
        "agent_1": {
            'type': 'fqi',
            'file': "fqi_w3.pkl",
        },
        "agent_2": {
            'type': 'fqi',
            'file': "fqi_w4.pkl",
        },
        "agent_3": {
            'type': 'fqi',
            'file': "fqi_w5.pkl",
        },
    }
    OAMP_ARGS = {}
    run_evaluation(RUN_NAME, AGENTS_INFO, OAMP_ARGS)
