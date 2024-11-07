import os

import pandas as pd
import numpy as np
import pickle
import optuna
from joblib import Parallel, delayed
from matplotlib import pyplot as plt

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

def evaluation(policy, eval_env):
    for i in range(3):
        policy.Q._regressors[i].n_jobs = 1
    reward = 0
    rewards = []
    actions = []
    s, _ = eval_env.reset(eval_sequential=True)
    print(f"step_is: {eval_env.step_is}")
    for st in range(eval_env.max_step):
        a = policy.sample_action(s)
        actions.append(a)
        sp, r, done, truncated, _ = eval_env.step(a)
        reward = reward + r
        rewards.append(r.item())
        s = sp
        if done or truncated:
            break

    return rewards, actions

args = get_cli_args()
n_windows = 2
start_day = 9
n_train_days = 1
n_validation_days = 1
performances = {}
for window in range(n_windows):
    performances[window] = {}
    out_dir = "./agents"
    start_day_train = start_day + window
    end_day_train = start_day_train + n_train_days - 1
    sample_days_train = [start_day_train, start_day_train + n_train_days - 1]
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
    policy = pickle.load(open(os.path.join(out_dir, f"fqi_w{window + 1}.pkl"), "rb"))

    for day_to_test in [7, 8]:
        print(f"Test {day_to_test}")
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
            "step_gap": 2,
            "env_class": TradeSimulator,
            "eval": True,
            "eval_sequential": True,
            "days": [day_to_test, day_to_test]
        }
        eval_env = build_env(TradeSimulator, env_args, -1)
        rewards_obtained = Parallel(n_jobs=1)(
            delayed(evaluation)(policy, eval_env) for i in range(1))
        rewards_obtained_t = []
        actions_t = []
        for e in range(1):
            rewards_obtained_t.append(rewards_obtained[e][0])
            actions_t.append(rewards_obtained[e][1])

        rewards_obtained = np.asarray(rewards_obtained_t)
        _, c = np.unique(np.asarray(actions_t).flatten(), return_counts=True)
        print(c)
        # plt.figure()
        # plt.title(f"FQI Return on day {day_to_test} | Iteration {iteration}")
        # mean = np.asarray(rewards_obtained_t[0]).cumsum()
        # print(mean[-1])
        # std = np.std(np.asarray(rewards_obtained_t).cumsum(), axis = 0)
        # plt.plot(mean)
        # plt.fill_between(mean - std, mean+std,  alpha=0.3,)
        # plt.savefig(f"{day_to_test}_return_iteration_{i}")
        performances[window][day_to_test] = np.asarray(rewards_obtained_t[0]).cumsum()
        # np.mean(np.asarray(rewards_obtained).cumsum(axis=1), axis=0)[-1]
        # np.std(np.asarray(rewards_obtained).cumsum(axis=1), axis=0)[-1]
        print(
            f"Reward obtained on next validation [{day_to_test}]: {np.mean(np.asarray(rewards_obtained).cumsum(axis=1), axis=0)[-1]} +/- {np.std(np.asarray(rewards_obtained).cumsum(axis=1), axis=0)[-1]}")
        print(f"SR = {np.mean(np.asarray(rewards_obtained)) / np.std(np.asarray(rewards_obtained))}")
        # with open('saved_dictionary_mean_step_gap_2_train_validation_on_first.pkl', 'wb') as f:
        #    pickle.dump(test_mean_rewards_per_windows_test, f)
        # with open('saved_dictionary_std_step_gap_2_train_validation_on_first.pkl', 'wb') as f:
        #    pickle.dump(test_std_rewards_per_windows_test, f)
        # with open('saved_dictionary_sr_step_gap_2_train_validation_on_first.pkl', 'wb') as f:
        #    pickle.dump(test_sr, f)
plt.figure()
for day in range(7, 9):
    win_p = []
    plt.figure()
    for w in performances.keys():
        plt.plot(performances[w][day], label=f'{w}')

    plt.title(f"Performance of FQI on day {day}")
    plt.legend()
    plt.savefig(f"FQI_{day}")



