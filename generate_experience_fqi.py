import random
from abc import ABC

import pandas as pd

from erl_config import build_env
from trade_simulator import TradeSimulator, EvalTradeSimulator
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.results_plotter import load_results, ts2xy
import numpy as np
import os
import matplotlib.pyplot as plt

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback



max_steps = 360

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
        'days': [7, 13]
    }

env = build_env(TradeSimulator, env_args, -1)
#env = Monitor(env, log_dir, info_keywords=("asset_v", 'mid','new_cash', 'old_cash', "action_exec", "position"))
#eval_env = Monitor(eval_env, log_dir_eval, info_keywords=("asset_v", 'mid','new_cash', 'old_cash', "action_exec", "position"))


EPISODES = 1000

#rendom policy execution

episode_rewards = []
states = []
actions = []
rewards = []
absorbing_state = []
next_states = []

for episode in range(EPISODES):
    print("Episode: " + str(episode))
    s, _ = env.reset()
    for step in range(max_steps):
            states.append(s[0].numpy())
            a = random.randint(0, 2)
            s, r, done, truncated, info = env.step(a)
            actions.append(a)
            rewards.append(r.numpy()[0])
            next_states.append(s[0].numpy())
            if done:
                absorbing_state.append(True)
                break
            else:
                absorbing_state.append(False)


df = pd.DataFrame({'state': states, 'action': actions, 'reward': rewards, "next_state": next_states, 'absorbing_state':absorbing_state})
df.to_json('./data/random_policy.json')

#long only

states = []
actions = []
rewards = []
next_states = []
absorbing_state = []

for episode in range(EPISODES):
    print("Episode: " + str(episode))
    s, _ = env.reset()
    for step in range(max_steps):
            states.append(s[0].numpy())
            a = 2
            s, r, done, truncated, info = env.step(a)
            actions.append(a)
            rewards.append(r.numpy()[0])
            next_states.append(s[0].numpy())
            if done:
                absorbing_state.append(True)
                break
            else:
                absorbing_state.append(False)

df = pd.DataFrame({'state': states, 'action': actions, 'reward': rewards, "next_state": next_states, 'absorbing_state':absorbing_state})
df.to_json('./data/long_only_policy.json')

#short only

states = []
actions = []
rewards = []
next_states = []
absorbing_state = []

for episode in range(EPISODES):
    print("Episode: " + str(episode))
    s, _ = env.reset()
    for step in range(max_steps):
            states.append(s[0].numpy())
            a = 0
            s, r, done, truncated, info = env.step(a)
            actions.append(a)
            rewards.append(r.numpy()[0])
            next_states.append(s[0].numpy())
            if done:
                absorbing_state.append(True)
                break
            else:
                absorbing_state.append(False)

df = pd.DataFrame({'state': states, 'action': actions, 'reward': rewards, "next_state": next_states, 'absorbing_state':absorbing_state})
df.to_json('./data/short_only_policy.json')

#flat only

states = []
actions = []
rewards = []
next_states = []
absorbing_state = []

for episode in range(EPISODES):
    print("Episode: " + str(episode))
    s, _ = env.reset()
    for step in range(max_steps):
            states.append(s[0].numpy())
            a = 1
            s, r, done, truncated, info = env.step(a)
            actions.append(a)
            rewards.append(r.numpy()[0])
            next_states.append(s[0].numpy())
            if done:
                absorbing_state.append(True)
                break
            else:
                absorbing_state.append(False)

df = pd.DataFrame({'state': states, 'action': actions, 'reward': rewards, "next_state": next_states, 'absorbing_state':absorbing_state})
df.to_json('./data/flat_only_policy.json')

