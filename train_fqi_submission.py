import pandas as pd
import numpy as np
import pickle
from train_fqi import read_dataset, generate_dataset, get_cli_args, prepare_dataset
from erl_config import build_env
from trade_simulator import TradeSimulator
from agent.fqi import AgentFQI

args = get_cli_args()
n_windows = args.n_windows
start_day = args.start_day_train
n_train_days = 1
data_dir = "./data_final"
base_out_dir = '.'

for window in range(n_windows):
    start_day_train = start_day + window
    end_day_train = start_day_train + n_train_days - 1
    sample_days_train = [start_day_train, end_day_train]
    policies = ['random_policy', 'long_only_policy', 'short_only_policy', 'flat_only_policy']
    dfs, dfs_unread = read_dataset(sample_days_train, policies=policies, data_dir=data_dir)
    if len(dfs_unread) > 0:
        dfs_train = generate_dataset(days_to_sample=sample_days_train,
                                     max_steps=args.max_steps, episodes=args.train_episodes, policies=dfs_unread,
                                     data_dir=data_dir)
        dfs += dfs_train

    if len(dfs) > 0:
        dfs = pd.concat(dfs)
    else:
        raise ValueError("No dataset!!")

    state_actions, rewards, next_states, absorbing = prepare_dataset(dfs)
    actions_values = [0, 1, 2]
    np.random.seed()
    seed = np.random.randint(100000)
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
    agent.train(state_actions=state_actions, rewards=rewards, next_states=next_states, absorbing=absorbing,
                env=eval_env, args={})

    out_dir = base_out_dir + f"/submission/trial_{window}_window/"
    agent.save(out_dir)
    # model_name = out_dir + f'Policy_iter{iteration}.pkl'
    # with open(model_name, 'wb+') as f:
    #     pickle.dump(algorithm._policy, f)

