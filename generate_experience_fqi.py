import random
import os
import pandas as pd
from erl_config import build_env
from trade_simulator import TradeSimulator

def random_policy(state):
    return random.randint(0, 2)

def long_only_policy(state):
    return 2

def short_only_policy(state):
    return 0

def flat_only_policy(state):
    return 1

policies = {
    'random_policy': random_policy,
    'long_only_policy':long_only_policy,
    'short_only_policy':short_only_policy,
    'flat_only_policy':flat_only_policy
}

def generate_experience(days_to_sample, policy, max_steps=360, episodes=1000, save=False, testing=False,
                        data_dir='./data/'):
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
            'days': days_to_sample
        }
    pi = policies[policy]
    env = build_env(TradeSimulator, env_args, -1)
    episode_rewards = []
    states = []
    actions = []
    rewards = []
    absorbing_state = []
    next_states = []

    for episode in range(episodes):
        # print("Episode: " + str(episode) + "; policy:" + policy)
        s, _ = env.reset()
        for step in range(max_steps):
                states.append(s[0].numpy())
                a = pi(s)
                s, r, done, truncated, info = env.step(a)
                actions.append(a)
                rewards.append(r.numpy()[0])
                next_states.append(s[0].numpy())
                if done:
                    absorbing_state.append(True)
                    break
                else:
                    absorbing_state.append(False)

    df = pd.DataFrame({'state': states, 'action': actions, 'reward': rewards, "next_state": next_states,
                       'absorbing_state': absorbing_state})

    print(f"Generated dataset days: {days_to_sample} ; policy:{policy}")

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if save:
        if testing is True:
            df.to_json(f'{data_dir}{policy}_testing_{days_to_sample}.json')
        else:
            df.to_json(f'{data_dir}{policy}_{days_to_sample}.json')
    return df