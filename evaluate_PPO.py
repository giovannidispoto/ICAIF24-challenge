import random

import pandas as pd
import numpy as np
import shutil
from sklearn.ensemble import ExtraTreesRegressor
import tqdm
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback

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
from stable_baselines3 import PPO


class SaveLastModelsCallback(BaseCallback):
    def __init__(self, save_path, save_freq, n_last=5, verbose=0):
        super(SaveLastModelsCallback, self).__init__(verbose)
        self.save_path = save_path
        self.save_freq = save_freq
        self.n_last = n_last

        # Ensure the save directory exists
        os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.save_freq == 0:
            # Shift all model files up by 1 position (model_0 becomes model_1, etc.)
            for i in range(self.n_last - 1):
                old_model_path = os.path.join(self.save_path, f"model_{i + 1}.zip")
                new_model_path = os.path.join(self.save_path, f"model_{i}.zip")
                if os.path.exists(old_model_path):
                    os.rename(old_model_path, new_model_path)

            # Save the latest model as model_4.zip (or model_{n_last-1}.zip)
            latest_model_path = os.path.join(self.save_path, f"model_{self.n_last - 1}.zip")
            self.model.save(latest_model_path)

        return True



def evaluation(algorithm, eval_env):
    reward = 0
    rewards = []
    s, _ = eval_env.reset()
    for st in range(max_steps):
        if algorithm is None:
            a = random.randint(0, 2)
        else:
            a = algorithm.predict(s, deterministic=True)
        sp, r, done, truncated, _ = eval_env.step(a[0][0])
        reward = reward + r
        rewards.append(r.item())
        s = sp
        if done or truncated:
            break

    return rewards


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

env = build_env(TradeSimulator, env_args, -1)
eval_episodes = 50
keep_last_models = 5
train_episodes = 1000
save_checkpoint_every = 5
train_agents = True
actions_values = [0, 1, 2]
np.random.seed()
seeds = []

if train_agents is True:
    if "checkpoints_PPO" in os.listdir("."):
        shutil.rmtree("./checkpoints_PPO")
    os.makedirs("./checkpoints_PPO")
    for _ in range(4):
      seeds.append(np.random.randint(100000))
else:
    seeds = [int(s.split("d")[1]) for s in os.listdir("./checkpoints_PPO")] #load saved seeds

year_set = {"val": [14, 15]}

rewards_seed_iterations = dict()

for s in year_set.keys():
    rewards_df_overall = [None]*keep_last_models
    for seed in seeds:
        env_args["eval"] = True
        env_args["seed"] = seed
        env_args["days"] = year_set[s]
        eval_env = build_env(TradeSimulator, env_args, -1)

        model = PPO("MlpPolicy", env, verbose=0, tensorboard_log="./ppo_tensorboard/", seed=seed, n_steps = max_steps )
        save_path = os.path.join("PPO_Saved", f"seed{seed}")
        if f"seed{seed}" in os.listdir("PPO_Saved"):
            shutil.rmtree(save_path)
        os.makedirs(save_path, exist_ok=False)
        save_callback = SaveLastModelsCallback(save_path=save_path, save_freq=max_steps*save_checkpoint_every, n_last=keep_last_models)
        model.learn(total_timesteps=max_steps * train_episodes, callback=[save_callback], progress_bar=True)

        #print(f"Iteration {i + 1} trained")
        #print("Testing")
        for i in range(keep_last_models):
            rewards_res = Parallel(n_jobs=10, prefer="threads")(delayed(evaluation)(model, eval_env) for i in range(eval_episodes))
            rewards_df = pd.DataFrame(rewards_res)
            if rewards_df_overall[i] is None:
                rewards_df_overall[i] = rewards_df
            else:
                rewards_df_overall[i] = pd.concat([rewards_df_overall[i], rewards_df], ignore_index=True)

    for i in range(keep_last_models):
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

        plt.title(f'Phase = {s} | Mean Rewards with 95% Confidence Interval: PPO')
        plt.xlabel('Steps')
        plt.ylabel('Reward cumsum')
        plt.legend()
        plt.grid(True)
        #plt.show()
        plt.savefig(f"plot/return_{s}_phase_PPO_{i}.png")
        #print(f"Reward: {np.mean(rewards_obtained)} +/- {np.std(rewards_obtained)}")
        #rewards_seed_iterations[seed][i] = np.mean(rewards_obtained)


#for i in range(max_iterations):
#    print(f"Iteration {i}, Reward:", pd.DataFrame.from_dict(rewards_seed_iterations, orient='index').mean().iloc[i], " +/- ", pd.DataFrame.from_dict(rewards_seed_iterations, orient='index').std().iloc[i])