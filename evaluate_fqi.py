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
from scipy import stats
import matplotlib.pyplot as plt
import os
import pickle
import optuna
from ast import literal_eval

dfs = None


def evaluation(algorithm, eval_env):
    reward = 0
    rewards = []
    s, _ = eval_env.reset()
    for st in range(max_steps):
        a = algorithm._policy.sample_action(s)
        sp, r, done, truncated, _ = eval_env.step(a)
        reward = reward + r
        rewards.append(r.item())
        s = sp
        if done or truncated:
            break

    return rewards



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
    "env_class": TradeSimulator,
    "days": [7, 13]
}

dfs = dfs.sample(frac=1)
dfs['state'] = dfs['state']
dfs['next_state'] = dfs['next_state']
env = build_env(TradeSimulator, env_args, -1)
state = pd.DataFrame(dfs['state'].to_list())
state_actions = pd.concat([state, dfs['action'].reset_index(drop=True)], axis=1)
episodes = 50
train_agents = True
rewards = dfs['reward']
next_states = pd.DataFrame(dfs['next_state'].to_list())
absorbing = dfs['absorbing_state']
actions_values = [0, 1, 2]
np.random.seed()
seeds = []

if train_agents is True:
    for _ in range(4):
      seeds.append(np.random.randint(100000))
else:
    seeds = [int(s.split("d")[1]) for s in os.listdir("checkpoints")] #load saved seeds

year_set = {"train": [7, 13], "val": [14, 15]}


max_iterations = 2 # 2 was the iteration number selected
max_depth = 10 #
min_split = 660 #
rewards_seed_iterations = dict()

for s in year_set.keys():
    rewards_df_overall = [None] * max_iterations
    for seed in seeds:
        env_args["eval"] = True
        env_args["seed"] = seed
        env_args["days"] = year_set[s]
        eval_env = build_env(TradeSimulator, env_args, -1)
        print(year_set[s])
        pi = EpsilonGreedy(actions_values, ZeroQ(), epsilon=0)
        algorithm = FQI(mdp=env, policy=pi, actions=actions_values, batch_size=5, max_iterations=max_iterations,
                        regressor_type=ExtraTreesRegressor, random_state=seed, n_jobs=-1, max_depth=max_depth,
                        min_samples_split=min_split)

        for i in range(max_iterations):
            iteration = i + 1
            if train_agents is True:
                algorithm._iter(
                    state_actions.to_numpy(dtype=np.float32),
                    rewards.to_numpy(dtype=np.float32),
                    next_states.to_numpy(dtype=np.float32),
                    absorbing,
                )
                model_name = f'Policy_iter{iteration}.pkl'
                os.makedirs(f"./checkpoints/seed{seed}", exist_ok=True)
                with open(os.path.join(f"./checkpoints/seed{seed}", model_name), 'wb+') as f:
                    pickle.dump(algorithm._policy, f)
            else:
                with open(os.path.join("./checkpoints", f'Policy_iter{iteration}.pkl'), 'rb') as load_file:
                    algorithm._policy = pickle.load(load_file)

            #print(f"Iteration {i + 1} trained")
            #print("Testing")
            rewards_res = Parallel(n_jobs=10)(delayed(evaluation)(algorithm, eval_env) for i in range(episodes))
            rewards_df = pd.DataFrame(rewards_res)
            if rewards_df_overall[i] is None:
                rewards_df_overall[i] = rewards_df
            else:
                rewards_df_overall[i] = pd.concat([rewards_df_overall, rewards_df], ignore_index=True)

        for i in range(max_iterations):
            rewards_df_overall[i] = rewards_df_overall[i].cumsum(axis=1) #calculate the cumulative sum of the rewards
            mean_rewards = np.mean(rewards_df_overall[i], axis=0)
            sem_rewards = stats.sem(rewards_df_overall[i], axis=0)

            # Compute 95% confidence interval (CI)
            ci = 1.96 * sem_rewards
            plt.figure(figsize=(10, 6))
            steps = np.arange(len(mean_rewards))

            plt.figure()
            plt.plot(steps, mean_rewards, label='Mean reward', color='b')
            plt.fill_between(steps, mean_rewards - ci, mean_rewards + ci, color='b', alpha=0.2, label='95% CI')

            plt.title(f'Phase = {s} | Mean Rewards with 95% Confidence Interval: Iteration {i+1}')
            plt.xlabel('Steps')
            plt.ylabel('Reward cumsum')
            plt.legend()
            plt.grid(True)
            plt.show()
            plt.savefig(f"plot/return_{s}_phase_{i+1}_iteration.png")
        #print(f"Reward: {np.mean(rewards_obtained)} +/- {np.std(rewards_obtained)}")
        #rewards_seed_iterations[seed][i] = np.mean(rewards_obtained)


#for i in range(max_iterations):
#    print(f"Iteration {i}, Reward:", pd.DataFrame.from_dict(rewards_seed_iterations, orient='index').mean().iloc[i], " +/- ", pd.DataFrame.from_dict(rewards_seed_iterations, orient='index').std().iloc[i])