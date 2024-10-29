import pandas as pd
import numpy as np
import pickle
import optuna
from train_fqi import read_dataset, generate_dataset, get_cli_args, prepare_dataset
from erl_config import build_env
from trade_simulator import TradeSimulator
from trlib.algorithms.reinforcement.fqi import FQI
from trlib.policies.qfunction import ZeroQ
from trlib.policies.valuebased import EpsilonGreedy
from sklearn.ensemble import ExtraTreesRegressor


def load_policy(model_path):
    policy = pickle.load(open(model_path, "rb"))
    return policy

args = get_cli_args()
n_windows = 8
start_day = 7
n_train_days = 1
n_validation_days = 1
base_dir = "sqlite:///."
base_out_dir = '.'
for window in range(n_windows):
    study_path = base_dir + f"/trial_{window}_window/optuna_study.db"
    out_dir = base_out_dir + "/trial_{window}_window/"
    loaded_study = optuna.load_study(study_name=None, storage=study_path)
    start_day_train = start_day + window
    end_day_train = start_day_train + n_train_days - 1
    sample_days_train = [start_day, start_day + n_train_days - 1]
    policies = ['random_policy', 'long_only_policy', 'short_only_policy', 'flat_only_policy']
    dfs, dfs_unread = read_dataset(sample_days_train, policies=policies)
    if len(dfs_unread) > 0:
        dfs_train = generate_dataset(days_to_sample=sample_days_train,
                                     max_steps=args.max_steps, episodes=args.train_episodes, policies=dfs_unread)
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
        "days": [end_day_train + 1, end_day_train + n_validation_days]
    }
    eval_env = build_env(TradeSimulator, env_args, -1)
    pi = EpsilonGreedy(actions_values, ZeroQ(), epsilon=0)
    max_iterations = loaded_study.best_params['iterations']
    n_estimators = loaded_study.best_params['n_estimators']
    max_depth = loaded_study.best_params['max_depth']
    min_split = loaded_study.best_params['min_split']
    algorithm = FQI(mdp=eval_env, policy=pi, actions=actions_values, batch_size=5, max_iterations=max_iterations,
                    regressor_type=ExtraTreesRegressor, random_state=seed, n_estimators=n_estimators, n_jobs=-1,
                    max_depth=max_depth, min_samples_split=min_split)

    for i in range(max_iterations):
        iteration = i + 1
        algorithm._iter(
            state_actions.to_numpy(dtype=np.float32),
            rewards.to_numpy(dtype=np.float32),
            next_states.to_numpy(dtype=np.float32),
            absorbing,
        )
        model_name = out_dir + f'Policy_iter{iteration}.pkl'
        with open(model_name, 'wb+') as f:
            pickle.dump(algorithm._policy, f)

