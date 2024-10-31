import os
import torch
import numpy as np
from erl_config import Config, build_env
from oamp.oamp import OAMP
from trade_simulator import TradeSimulator, EvalTradeSimulator
from tqdm import tqdm
from agent.base import AgentBase
from agent.factory import AgentsFactory
# from metrics import sharpe_ratio, max_drawdown, return_over_max_drawdown


def trade(action, mid_price, cur_cash, cur_btc):
    if action == 1:
        new_cash = cur_cash - mid_price
        new_btc = cur_btc + 1
    elif action == -1:
        new_cash = cur_cash + mid_price
        new_btc = cur_btc - 1
    else:
        new_cash = cur_cash
        new_btc = cur_btc
    return new_cash, new_btc


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
    def __init__(
        self,
        starting_cash: float,
        agents_info: dict,
        save_path: str,
        args: Config,
    ):
        # Initializing portfolio
        self.starting_cash = starting_cash
        self.current_btc = 0
        self.position = [0]
        self.btc_assets = [0]
        self.net_assets = [starting_cash]
        self.cash = [starting_cash]

        # Setting args env
        self.args = args
        self.num_envs = 1
        self.state_dim = 8 + 2

        # Setting device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Setting save path
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)

        # Loading agents
        self.agents: list[AgentBase] = []
        self.agents_names: list[str] = []
        for agent_name, agent_info in agents_info.items():
            self.agents.append(AgentsFactory.load_agent(agent_info))
            self.agents_names.append(agent_name)

    def ensemble_trade(self):
        # Building trading env
        args = self.args
        eval_env_class = args.eval_env_class
        eval_env_class.num_envs = 1
        eval_env_args = args.eval_env_args
        eval_env_args["num_envs"] = 1
        eval_env_args["num_sims"] = 1
        eval_env = build_env(eval_env_class, eval_env_args, gpu_id=args.gpu_id)
        eval_env.step_is[0] = 1000
        # Initializing states, actions, rewards, dones
        states = torch.zeros(
            (eval_env.max_step, self.num_envs, self.state_dim),
            dtype=torch.float32,
        ).to(self.device)
        actions = torch.zeros(
            (eval_env.max_step, self.num_envs, 1),
            dtype=torch.int32,
        ).to(self.device)
        rewards = torch.zeros(
            (eval_env.max_step, self.num_envs),
            dtype=torch.float32,
        ).to(self.device)
        dones = torch.zeros(
            (eval_env.max_step, self.num_envs),
            dtype=torch.bool,
        ).to(self.device)
        # Initializing trading history
        trade_ary = []
        position_ary = []
        correct_pred = []
        current_btcs = [self.current_btc]
        # Initializing last state and price
        last_state, _ = eval_env.reset()
        last_price = 0
        # Initializing OAMP
        oamp = OAMP(len(self.agents))
        # Trading
        for i in tqdm(range(eval_env.max_step)):
            agents_rewards = []
            agents_actions = []
            for ai, agent in enumerate(self.agents):
                # Computing agent last reward
                agent_reward = ai
                agents_rewards.append(agent_reward)
                # Computing agent curr action
                agent_action = agent.action(last_state)
                agents_actions.append(agent_action)
            # Computing ensemble action
            action = oamp.step(agents_rewards, agents_actions)
            action_int = action - 1
            state, reward, done, _, _ = eval_env.step(action=action)
            # Saving states, actions, rewards, dones
            actions[i] = action
            states[i] = state
            rewards[i] = reward
            dones[i] = done
            # Saving trading history
            trade_ary.append(eval_env.action_int.data.cpu().numpy())
            position_ary.append(eval_env.position.data.cpu().numpy())
            price = eval_env.price_ary[eval_env.step_i, 2].to(self.device)
            new_cash, self.current_btc = trade(
                action_int, price, self.cash[-1], self.current_btc
            )
            self.cash.append(new_cash)
            self.btc_assets.append((self.current_btc * price).item())
            self.net_assets.append((self.btc_assets[-1] + new_cash))
            current_btcs.append(self.current_btc)
            correct_pred.append(winloss(action_int, last_price, price))
            # Updating last state and price
            last_state = state
            last_price = price
        # Saving
        np.save(f"{self.save_path}_position_ary.npy", position_ary)
        np.save(f"{self.save_path}_net_assets.npy", np.array(self.net_assets))
        np.save(f"{self.save_path}_btc_pos.npy", np.array(self.btc_assets))
        np.save(f"{self.save_path}_correct_preds.npy", np.array(correct_pred))
        # Computing returns
        returns = []
        for t in range(len(self.net_assets) - 1):
            r_t = self.net_assets[t]
            r_t_plus_1 = self.net_assets[t + 1]
            return_t = (r_t_plus_1 - r_t) / r_t
            returns.append(return_t)
        returns = np.array(returns)
        oamp.plot_stats(self.save_path)
        # Computing metrics
        # final_sharpe_ratio = sharpe_ratio(returns)
        # final_max_drawdown = max_drawdown(returns)
        # final_roma = return_over_max_drawdown(returns)


def run(
    save_path: str,
    agents_info: list[str],
):
    starting_cash = 1e6

    import sys

    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else -1

    from erl_agent import AgentD3QN

    num_sims = 2**12
    num_ignore_step = 60
    step_gap = 2
    max_step = 300  # (4800 - num_ignore_step) // step_gap
    max_position = 1
    slippage = 7e-7

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
    }
    args = Config(agent_class=AgentD3QN, env_class=TradeSimulator, env_args=env_args)

    # Args for reward shaping
    args.gamma = 0.995
    args.reward_scale = 2**0

    # Args for training
    args.random_seed = gpu_id
    args.net_dims = (64, 64)
    args.learning_rate = 2e-6
    args.state_value_tau = 0.01
    args.soft_update_tau = 2e-6
    args.batch_size = 512
    args.buffer_size = int(max_step * 8)
    args.horizon_len = int(max_step * 1)
    args.repeat_times = 2
    args.explore_rate = 0.005
    args.break_step = int(32e2)

    # Args for device
    args.gpu_id = gpu_id
    args.num_workers = 1

    # Args for testing
    args.save_gap = 8
    args.eval_per_step = int(max_step)
    args.eval_env_args = env_args.copy()
    args.eval_env_class = EvalTradeSimulator

    ensemble_env = Ensemble(
        starting_cash,
        agents_info,
        save_path,
        args,
    )
    ensemble_env.ensemble_trade()


if __name__ == "__main__":
    SAVE_PATH = "C:\\Users\\anton\\Desktop\\ICAIF24\\ICAIF24-challenge\\oamp"
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

    run(
        SAVE_PATH,
        AGENTS_INFO,
    )
