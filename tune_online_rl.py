import numpy as np
from stable_baselines3 import A2C
import torch
from erl_config import build_env
from metrics import sharpe_ratio
from sample_online_rl import SAMPLER
from task1_eval import to_python_number, trade, winloss
from trade_simulator import EvalTradeSimulator, TradeSimulator
from stable_baselines3.common.vec_env import DummyVecEnv
import optuna
import argparse
import os
import optuna
from stable_baselines3.common.base_class import BaseAlgorithm
from datetime import datetime
from sbx import PPO, DQN, SAC
from stable_baselines3.common.env_util import make_vec_env


def get_cli_args():
    """Create CLI parser and return parsed arguments"""
    parser = argparse.ArgumentParser()
    # Example-specific args.
    parser.add_argument(
        '--agent',
        type=str,
        default="PPO",
        help="Agent class name"
    )
    
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
        '--start_day_val',
        type=int,
        default=7,
        help="starting day to train (included) "
    )

    parser.add_argument(
        '--end_day_val',
        type=int,
        default=15,
        help="ending day to train (included) "
    )
    
    parser.add_argument(
        '--n_trials',
        type=int,
        default=50,
        help="number of iterations of optuna"
    )
    parser.add_argument(
        '--n_seeds',
        type=int,
        default=5
    )
    parser.add_argument(
        '--n_windows',
        type=int,
        default=8
    )
    parser.add_argument(
        '--out_dir',
        type=str,
        default="."
    )
    parser.add_argument(
        '--progress',
        action='store_true',
        help='Enable progress output',
        default=False
    )

    return parser.parse_args()

def interquartile_mean(data: np.ndarray, q_min: int = 25, q_max: int = 75) -> float:
    assert data.ndim == 1, "Input data must be 1D"
    sorted_data = np.sort(data)
    
    q_min = np.percentile(sorted_data, q_min)
    q_max = np.percentile(sorted_data, q_max)
    filtered_data = sorted_data[(sorted_data >= q_min) & (sorted_data <= q_max)]    
    iqm = np.mean(filtered_data)
    return iqm


class TradeSimulatorOptimizer:
    def __init__(self, agent_class: BaseAlgorithm, device, out_dir, plot_dir,
                 num_eval_sims, n_envs,
                 start_day_train, end_day_train, 
                 start_day_val, end_day_val,
                 max_step, eval_max_step = None, deterministic_eval=True,
                 show_progress=False,
                 n_episodes = 1000, storage = None, n_seeds=5, n_trials=100, n_days_val = 1, gpu_id = -1):
        self.agent_class = agent_class
        self.device = device
        self.gpu_id = gpu_id
        self.out_dir = out_dir
        self.plot_dir = plot_dir
        self.start_day_train = start_day_train
        self.end_day_train = end_day_train
        self.start_day_val = start_day_val
        self.end_day_val = end_day_val
        assert self.start_day_train <= self.end_day_train, "start_day_train must be less than end_day_train"
        assert self.start_day_val <= self.end_day_val, "start_day_val must be less than end_day_val"
        assert self.end_day_train < self.start_day_val or self.end_day_val < self.start_day_train, "No overlap between train and val days"

        self.n_days_val = n_days_val
        self.max_step = max_step
        self.eval_max_step = eval_max_step if eval_max_step is not None else max_step
        self.n_episodes = n_episodes
        self.n_envs=n_envs
        self.n_seeds = n_seeds
        self.n_trials = n_trials
        self.storage = storage
        self.n_actions = 3
        self.num_eval_sims=num_eval_sims
        self.deterministic_eval = deterministic_eval
        self.show_progress = show_progress
        
        self.env_args = self._initialize_env_args()
        self._setup_directories()
        
        np.random.seed()
        self.seeds = [np.random.randint(2**32 - 1, dtype="int64").item() for i in range(self.n_seeds)]
        
        self.study = None
    
    def create_study(self):
        self.study = optuna.create_study(study_name=f'{self.agent_class.__name__}_tuning_{self.n_seeds}seeds_{self.start_day_train}_{self.end_day_train}',
                                         direction="maximize", storage=self.storage)
    def _setup_directories(self):
        os.makedirs(self.out_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)
        
    def _initialize_env_args(self):
        return {
            "env_name": "TradeSimulator-v0",
            "num_envs": 1,
            "num_sims": 1,
            "max_step": self.max_step,
            "state_dim": 10,
            "action_dim": 3,
            "if_discrete": True,
            "max_position": 1,
            "slippage": 7e-7,
            "step_gap": 2,
            "env_class": TradeSimulator,
            "days": [self.start_day_train, self.end_day_train],
            "eval_sequential": False,
        }
        
    def train_agent(self, model_params, learn_params = {}, seed=123):        
        env = make_vec_env(
            lambda: build_env(TradeSimulator, {**self.env_args, "seed": seed}, gpu_id=self.gpu_id),
            n_envs=self.n_envs,
            seed=seed
        ) 
        
        agent = self.agent_class("MlpPolicy", env, verbose=0, device="cpu", seed=seed, **model_params)
        agent.learn(total_timesteps=self.max_step * self.n_episodes, progress_bar=self.show_progress, **learn_params)
        return agent
    
    def evaluate_agent(self, agent, days, seed=None):
        eval_env_args = self.env_args.copy()
        eval_env_args.update({
            "eval": True,
            "days": days,
            "num_sims": self.num_eval_sims,
            "num_envs": 1,
            "max_step": self.eval_max_step,
            "env_class": EvalTradeSimulator,
            "seed": seed
        })
        eval_env = build_env(EvalTradeSimulator, eval_env_args, gpu_id=self.gpu_id)
        
        # last_price = 0        
        # current_btc = 0
        # starting_cash = 1e6

        # cash = [starting_cash]
        # btc_assets = [0]
        # net_assets = [starting_cash]

        # positions = []
        # action_ints = []
        # correct_pred = []
        # current_btcs = [current_btc]
        
        state, _ = eval_env.reset(seed=seed)
                
        total_reward = torch.zeros(self.num_eval_sims, dtype=torch.float32, device=self.device)
        for _ in range(eval_env.max_step):
            tensor_state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
            action, _ = agent.predict(tensor_state, deterministic=self.deterministic_eval)
            # action_int = action.item() - 1                        
            state, reward, terminated, truncated, _ = eval_env.step(action=action)
            total_reward += reward
            # reward_int = reward.item()
            
            # price = eval_env.price_ary[eval_env.step_i, 2].to(self.device)
            # new_cash, current_btc = trade(
            #     action_int, price, cash[-1], current_btc
            # )
            
            # cash.append(new_cash)
            # btc_assets.append((current_btc * price).item())
            # net_assets.append(
            #     (to_python_number(btc_assets[-1]) + to_python_number(new_cash))
            # )
            # # Upadting trading history
            # positions.append(eval_env.position)
            # action_ints.append(action_int)
            # current_btcs.append(current_btc)
            # correct_pred.append(winloss(action_int, last_price, price))
            # # Updating last state and price
            # last_price = price
            # total_reward += reward_int
            if terminated.any() or truncated:
                break
                
            # returns = np.diff(net_assets) / net_assets[:-1]
            # final_sharpe_ratio = sharpe_ratio(returns)
            
        mean_total_reward = total_reward.mean()
        return to_python_number(mean_total_reward), 0
                    
    def optimize_hyperparameters(self):
        def objective(trial):
            model_params = SAMPLER[self.agent_class.__name__](trial, n_actions=self.n_actions, n_envs=self.n_envs, additional_args={})
            
            rewards = []
            sharpe_ratios = []
            rewards_train = []
            sharpe_ratios_train = []
            for seed in self.seeds:
                agent = self.train_agent(model_params, {}, seed=seed)
                
                val_days = [self.start_day_val, self.end_day_val]
                reward, sharpe_ratio = self.evaluate_agent(agent, val_days, seed=seed)
                train_reward, train_sharpe_ratio = self.evaluate_agent(agent, [self.start_day_train, self.end_day_train], seed=seed)
                
                print(f"seed: {seed}, reward: {reward}, train_reward: {train_reward}, sharpe_ratio: {sharpe_ratio}, train_sharpe_ratio: {train_sharpe_ratio}")
                rewards.append(reward)
                sharpe_ratios.append(sharpe_ratio)
                rewards_train.append(train_reward)
                sharpe_ratios_train.append(train_sharpe_ratio)
            
            if trial.number > 1:
                self._plot_results(trial)
            
            trial.set_user_attr("rewards", rewards)
            trial.set_user_attr("sharpe_ratios", sharpe_ratios)
            trial.set_user_attr("rewards_train", rewards_train)
            trial.set_user_attr("sharpe_ratios_train", sharpe_ratios_train)
            
            print(rewards)
            # print(sharpe_ratios)
            rewards = np.array(rewards)
            mean_rewards, median_rewards, iqm_rewards = np.mean(rewards), np.median(rewards), interquartile_mean(rewards)
            print(f"Mean: {mean_rewards}, Median: {median_rewards}, IQM: {iqm_rewards}")
            
            
            # sharpe_ratios = [x for x in sharpe_ratios if x != np.inf] # Ignoring inf values   
            # sharpe_ratios = np.array(sharpe_ratios) if len(sharpe_ratios) > 0 else np.array([0])
            
            # mean_sr, median_sr, iqm_sr = np.mean(sharpe_ratios), np.median(sharpe_ratios), interquartile_mean(sharpe_ratios)
            # print(f"Mean: {mean_sr}, Median: {median_sr}, IQM: {iqm_sr}")
            
            return mean_rewards
        
        print(f"Optimizing {self.agent_class.__name__} hyperparameters")
        print(f'Using seeds: {self.seeds}')
        self.study.optimize(objective, n_trials=self.n_trials)
    
    def _plot_results(self, trial):
        plots = [
            ("ParamsOptHistory.png", optuna.visualization.plot_optimization_history(self.study)),
            ("ParamsImportance.png", optuna.visualization.plot_param_importances(self.study)),
            ("ParamsContour.png", optuna.visualization.plot_contour(self.study)),
            ("ParamsSlice.png", optuna.visualization.plot_slice(self.study))
        ]
        for filename, fig in plots:
            fig.write_image(f"{self.plot_dir}/{filename}")
    
    def run(self):
        self.optimize_hyperparameters()

if __name__ == "__main__":
    args = get_cli_args()
    
    if args.agent == "PPO":
        agent_class = PPO
    elif args.agent == "DQN":
        agent_class = DQN
    elif args.agent == "A2C":
        agent_class = A2C
    else:
        raise ValueError()
    
    print(f'Tuning agent: {agent_class.__name__}')
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    device = torch.device("cpu")
    out_dir = f'{args.out_dir}_{timestamp}'
    plot_dir = f'{out_dir}/plots'
    storage = f'sqlite:///{out_dir}/optuna_study.db'

    # num_ignore_step = 60
    # step_gap = 2
    # slippage = 7e-7
    # max_step = (4800 - num_ignore_step) // step_gap
    max_step = 480
    eval_max_step = 480
    
    optimizer = TradeSimulatorOptimizer(
        agent_class=agent_class,
        device=device,
        gpu_id=-1,
        out_dir=out_dir,
        plot_dir=plot_dir,
        start_day_train=args.start_day_train,
        end_day_train=args.end_day_train,
        start_day_val=args.start_day_val,
        end_day_val=args.end_day_val,
        max_step=max_step,
        eval_max_step=eval_max_step,
        n_seeds=args.n_seeds,
        n_trials=args.n_trials,
        storage=storage,
        show_progress=args.progress,
        n_episodes=500,
        num_eval_sims=50,
        n_envs=4,
    )
    optimizer.create_study()
    optimizer.run()