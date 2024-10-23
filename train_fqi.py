import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

import tqdm
from erl_config import build_env
from trade_simulator import TradeSimulator
from trlib.algorithms.reinforcement.fqi import FQI
from trlib.policies.qfunction import ZeroQ
from trlib.policies.valuebased import EpsilonGreedy
from ast import literal_eval

dfs = None

for p in ['random_policy', 'long_only_policy', 'short_only_policy','flat_only_policy']:
    df = pd.read_json(f"./data/{p}.json", )
    if dfs is None:
        dfs = df
    else:
        dfs = pd.concat([dfs, df])

max_steps = 3600

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
        "env_class": TradeSimulator
    }

dfs = dfs.sample(frac = 1)
dfs['state'] = dfs['state']
dfs['next_state'] = dfs['next_state']

state = pd.DataFrame(dfs['state'].to_list())
state_actions = pd.concat([state, dfs['action'].reset_index(drop=True)], axis = 1)
episodes = 5
rewards = dfs['reward']
next_states = pd.DataFrame(dfs['next_state'].to_list())
absorbing = dfs['absorbing_state']
actions_values = [0 ,1 ,2]
env_args["eval"] = True
eval_env = build_env(TradeSimulator, env_args, -1)
max_iterations = 5
seed = 12345

pi = EpsilonGreedy(actions_values, ZeroQ(), epsilon=0)
algorithm = FQI(mdp=eval_env, policy=pi, actions=actions_values, batch_size=5, max_iterations=max_iterations,
                regressor_type=ExtraTreesRegressor, random_state=seed, n_jobs=-1)

rewards_iteration = dict()
for i in range(max_iterations):
    rewards_iteration[i] = list()

    print(f"Start {i + 1} training")
    iteration = i + 1

    algorithm._iter(
        state_actions.to_numpy(dtype=np.float32),
        rewards.to_numpy(dtype=np.float32),
        next_states.to_numpy(dtype=np.float32),
        absorbing,
    )
    print(f"Iteration {i + 1} trained")
    print("Testing")
    continue
    rewards_obtained = list()
    for e in tqdm.tqdm(range(episodes)):
        reward = 0
        s, _ = eval_env.reset()
        for st in range(max_steps):
            a = algorithm._policy.sample_action(s)
            sp, r, done, truncated,  _ = eval_env.step(a)
            reward = reward + r
            s = sp
            if done or truncated:
                break
        rewards_obtained.append(reward)
    print(f"Reward: {np.mean(rewards_obtained)} +/- {np.std(rewards_obtained)}")
    rewards_iteration[i] = rewards_obtained



