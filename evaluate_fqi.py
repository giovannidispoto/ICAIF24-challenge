import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
import tqdm
from erl_config import build_env
from trade_simulator import TradeSimulator
from trlib.algorithms.reinforcement.fqi import FQI
from trlib.policies.qfunction import ZeroQ
from trlib.policies.valuebased import EpsilonGreedy
from joblib import Parallel, delayed
import optuna
from ast import literal_eval

dfs = None


def evaluation(algorithm, eval_env):
    reward = 0
    s, _ = eval_env.reset()
    for st in range(max_steps):
        a = algorithm._policy.sample_action(s)
        sp, r, done, truncated, _ = eval_env.step(a)
        reward = reward + r
        s = sp
        if done or truncated:
            break

    return reward



for p in ['random_policy', 'long_only_policy', 'short_only_policy',
          'flat_only_policy']:  # aggiungere anche politiche addestrate con PPO (anche senza tuning)
    df = pd.read_json(f"./data/{p}.json", )
    if dfs is None:
        dfs = df
    else:
        dfs = pd.concat([dfs, df])

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
    "env_class": TradeSimulator
}

dfs = dfs.sample(frac=1)
dfs['state'] = dfs['state']
dfs['next_state'] = dfs['next_state']

state = pd.DataFrame(dfs['state'].to_list())
state_actions = pd.concat([state, dfs['action'].reset_index(drop=True)], axis=1)
episodes = 50
rewards = dfs['reward']
next_states = pd.DataFrame(dfs['next_state'].to_list())
absorbing = dfs['absorbing_state']
actions_values = [0, 1, 2]
np.random.seed()
seeds = []
for _ in range(4):
    seeds.append(np.random.randint(100000))



max_iterations = 2 #
max_depth = 10 #
min_split = 660 #
rewards_seed_iterations = dict()

for seed in seeds:
    rewards_seed_iterations[seed] = dict()
    env_args["eval"] = True
    env_args["seed"] = seed
    env_args["days"] = [14, 15]
    eval_env = build_env(TradeSimulator, env_args, -1)
    pi = EpsilonGreedy(actions_values, ZeroQ(), epsilon=0)
    algorithm = FQI(mdp=eval_env, policy=pi, actions=actions_values, batch_size=5, max_iterations=max_iterations,
                    regressor_type=ExtraTreesRegressor, random_state=seed, n_jobs=-1, max_depth=max_depth, min_samples_split = min_split)

    for i in range(max_iterations):
        rewards_seed_iterations[seed][i] = list()

        iteration = i + 1

        algorithm._iter(
            state_actions.to_numpy(dtype=np.float32),
            rewards.to_numpy(dtype=np.float32),
            next_states.to_numpy(dtype=np.float32),
            absorbing,
        )
        #print(f"Iteration {i + 1} trained")
        #print("Testing")
        rewards_obtained = np.asarray(Parallel(n_jobs=10)(delayed(evaluation)(algorithm, eval_env) for i in range(episodes)))
        #print(f"Reward: {np.mean(rewards_obtained)} +/- {np.std(rewards_obtained)}")
        rewards_seed_iterations[seed][i] = np.mean(rewards_obtained)


print(pd.DataFrame.from_dict(rewards_seed_iterations, orient='index').mean().iloc[-1])