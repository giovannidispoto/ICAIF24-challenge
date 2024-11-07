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
import argparse
import os
from generate_experience_fqi import generate_experience
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
        '--end_day_train',
        type=int,
        default=15,
        help="ending day to train (included) "
    )

    parser.add_argument(
        '--num_days_validation',
        type=int,
        default=2,
        help="number of days to use for validation after training"
    )
    parser.add_argument(
        '--n_trials',
        type=int,
        default=50,
        help="number of iterations of optuna"
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=50
    )
    parser.add_argument(
        '--n_seeds',
        type=int,
        default=4
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

    return parser.parse_args()


def read_dataset(sample_days, policies=None, data_dir='./data/'):
    dfs = []
    dfs_unread = []
    if policies is None:
        policies = ['random_policy', 'long_only_policy', 'short_only_policy', 'flat_only_policy']
    for p in policies:  # aggiungere anche politiche addestrate con PPO (anche senza tuning)
        try:
            df = pd.read_json(f"{data_dir}{p}_{sample_days}.json", )
            dfs.append(df)
        except:
            dfs_unread.append(p)
    return dfs, dfs_unread

def generate_dataset(days_to_sample, max_steps=360, episodes=1000, policies=None, data_dir='./data/'):
    dfs = []
    if policies is None:
        policies = ['random_policy', 'long_only_policy', 'short_only_policy', 'flat_only_policy']
    for policy in policies:  # aggiungere anche politiche addestrate con PPO (anche senza tuning)
        df = generate_experience(days_to_sample, policy, max_steps=max_steps, episodes=episodes, save=True,
                                 testing=False, data_dir=data_dir)
        dfs.append(df)
    return dfs

def prepare_dataset(dfs, sample_frac=1.):
    dfs = dfs.sample(frac=sample_frac)
    dfs['state'] = dfs['state']
    dfs['next_state'] = dfs['next_state']
    state = pd.DataFrame(dfs['state'].to_list())
    state_actions = pd.concat([state, dfs['action'].reset_index(drop=True)], axis=1)
    rewards = dfs['reward']
    next_states = pd.DataFrame(dfs['next_state'].to_list())
    absorbing = dfs['absorbing_state']
    return state_actions, rewards, next_states, absorbing

def tune():
    args = get_cli_args()
    out_dir = args.out_dir
    plot_dir = out_dir + "/plots"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    sample_days_train = [args.start_day_train, args.end_day_train]
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
    max_steps = args.max_steps
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
        "env_class": TradeSimulator
    }
    state_actions, rewards, next_states, absorbing = prepare_dataset(dfs)
    actions_values = [0, 1, 2]
    np.random.seed()
    seeds = []
    for _ in range(args.n_seeds):
        seeds.append(np.random.randint(100000))
    study = optuna.create_study(direction="maximize", storage= f'sqlite:///{out_dir}/optuna_study.db')

    def objective(trial):
        max_iterations = trial.suggest_int("iterations", low=1, high=10, step=1)
        max_depth = trial.suggest_int("max_depth", low=1, high=30, step=5)
        n_estimators = trial.suggest_int("n_estimators", low=50, high=150, step=10)
        min_split = trial.suggest_int("min_samples_split", low=10, high=1000, step=50)
        rewards_seed_iterations = dict()
        for seed in seeds:
            rewards_seed_iterations[seed] = dict()
            env_args["eval"] = True
            env_args["seed"] = seed
            env_args["days"] = [args.end_day_train + 1, args.end_day_train + args.num_days_validation]
            eval_env = build_env(TradeSimulator, env_args, -1)
            pi = EpsilonGreedy(actions_values, ZeroQ(), epsilon=0)
            algorithm = FQI(mdp=eval_env, policy=pi, actions=actions_values, batch_size=5, max_iterations=max_iterations,
                            regressor_type=ExtraTreesRegressor, random_state=seed, n_estimators=n_estimators, n_jobs=-1,
                            max_depth=max_depth, min_samples_split=min_split)

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
                rewards_obtained = np.asarray(Parallel(n_jobs=10)(delayed(evaluation)(algorithm, eval_env) for i in range(args.episodes)))
                #print(f"Reward: {np.mean(rewards_obtained)} +/- {np.std(rewards_obtained)}")
                rewards_seed_iterations[seed][i] = np.mean(rewards_obtained)

        if trial.number > 1:
            fig = optuna.visualization.plot_optimization_history(study)
            fig.write_image(plot_dir + '/ParamsOptHistory.png')
            fig = optuna.visualization.plot_param_importances(study)
            fig.write_image(plot_dir + '/ParamsImportance.png')
            fig = optuna.visualization.plot_contour(study)
            fig.write_image(plot_dir + '/ParamsContour.png', width=3000, height=1750)
            fig = optuna.visualization.plot_slice(study)
            fig.write_image(plot_dir + '/ParamsSlice.png')
        return pd.DataFrame.from_dict(rewards_seed_iterations, orient='index').mean().iloc[-1]

    study.optimize(objective, n_trials=args.n_trials)

if __name__ == "__main__":
    tune()