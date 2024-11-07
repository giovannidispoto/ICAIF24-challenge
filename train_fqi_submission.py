import numpy as np
from joblib import Parallel, delayed
import argparse
from erl_config import build_env
from trade_simulator import TradeSimulator
from agent.fqi import AgentFQI


def get_cli_args():
    """Create CLI parser and return parsed arguments"""
    parser = argparse.ArgumentParser()
    # Example-specific args.
    parser.add_argument(
        '--start_day_train',
        type=int,
        default=7,
        help="starting day to train (included) "
    )
    parser.add_argument(
        '--n_train_days',
        type=int,
        default=1,
        help="number of training days per window"
    )
    parser.add_argument(
        '--max_steps',
        type=int,
        default=480
    )
    parser.add_argument(
        '--n_windows',
        type=int,
        default=8
    )
    parser.add_argument(
        '--train_episodes',
        type=int,
        default=1000
    )
    parser.add_argument(
        '--out_dir',
        type=str,
        default="."
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default="./data_final/"
    )

    return parser.parse_args()

args = get_cli_args()
n_windows = args.n_windows
start_day = args.start_day_train
n_train_days = args.n_train_days
data_dir = args.data_dir
base_out_dir = args.out_dir


def train_window(window):
    print(f"training window {window}")
    start_day_train = start_day + window
    end_day_train = start_day_train + n_train_days - 1
    sample_days_train = [start_day_train, end_day_train]
    np.random.seed()
    max_steps = args.max_steps
    env_args = {
        "env_name": "TradeSimulator-v0",
        "num_envs": 1,
        "max_step": max_steps,
        "state_dim": 8 + 2,  # factor_dim + (position, holding)
        "action_dim": 3,  # long, 0, short
        "if_discrete": True,
        "max_position": 1,
        "slippage": 7e-7,
        "num_sims": 1,
        "step_gap": 1,
        "env_class": TradeSimulator,
        "eval": True,
        "days": [end_day_train + 1, end_day_train + 1]
    }

    eval_env = build_env(TradeSimulator, env_args, -1)
    agent = AgentFQI()
    state_actions, rewards, next_states, absorbing, policies_unread = agent.read_dataset(sample_days_train,
                                                                                         data_dir=data_dir)
    if len(policies_unread) > 0:
        for policy in policies_unread:
            sa, r, ns, a = agent.generate_experience(days_to_sample=sample_days_train,
                                             max_steps=args.max_steps, episodes=args.train_episodes, policy=policy,
                                             data_dir=data_dir)
            if len(state_actions) > 0:
                state_actions = np.concatenate([state_actions, sa], axis=0)
                rewards = np.concatenate([rewards, r], axis=0)
                next_states = np.concatenate([next_states, ns], axis=0)
                absorbing = np.concatenate([absorbing, a], axis=0)
            else:
                state_actions = sa
                rewards = r
                next_states = ns
                absorbing = a
    agent.train(state_actions=state_actions, rewards=rewards, next_states=next_states, absorbing=absorbing,
                env=eval_env, args={})

    out_dir = base_out_dir + f"/submission_/{window}_window/"
    agent.save(out_dir)


Parallel(n_jobs=n_windows)(delayed(train_window)(window) for window in range(n_windows))


