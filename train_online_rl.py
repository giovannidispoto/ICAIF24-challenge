import argparse
import os
import pickle
from typing import Optional
from matplotlib.path import Path
import numpy as np
import optuna
import torch
from datetime import datetime
from sbx import DQN, PPO
from optuna import load_study

from config import EXP_DIR
from erl_config import build_env
from metrics import max_drawdown, return_over_max_drawdown, sharpe_ratio
from task1_tune_online_rl import SAMPLER
from trade_simulator import EvalTradeSimulator, TradeSimulator
import matplotlib.pyplot as plt
import seaborn as sns



def get_cli_args():
    """Create CLI parser and return parsed arguments"""
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--agent',
        type=str,
        default='DQN'
    )
    
    parser.add_argument(
        '--window',
        type=int,
        default=0,
        help="starting window"
    )
    parser.add_argument(
        '--n_seeds',
        type=int,
        default=5
    )
    parser.add_argument(
        '--storage',
        type=str,
        default=None
    )
    parser.add_argument(
        '--out_dir',
        type=str,
        default=None
    )

    return parser.parse_args()

def to_python_number(x):
    if isinstance(x, torch.Tensor):
        return x.cpu().item()
    else:
        return x
    
def plot_heatmap(results: np.ndarray, training_days: list[list[tuple[int, int]]], title="Daily Heatmap", 
                 xticklabels: list[str]=None, yticklabels: list[str]=None,
                 decimal_places: int = 2, use_e_notation: bool = False,
                 save_path:str = None, force_show: bool = False):
    assert results.ndim == 2
    assert results.shape[0] == len(training_days)
    
    results = np.array(results)
    num_rows, num_cols = results.shape
    plt.figure(figsize=(num_cols*1.5, num_rows*1.5))
    cmap = sns.light_palette("blue", as_cmap=True)
    
    fmt = f".{decimal_places}{'e' if use_e_notation else 'f'}"
    
    ax = sns.heatmap(results, annot=True, fmt=fmt, cmap=cmap, cbar=True, linewidths=0.5, linecolor='black', 
                     xticklabels=[f'Day {i+1}' for i in range(num_cols)] if xticklabels is None else xticklabels, 
                     yticklabels=[f'Results {i+1}' for i in range(num_rows)] if yticklabels is None else yticklabels)
    
    for row, train_days in enumerate(training_days):
        for start, end in train_days:
            ax.add_patch(plt.Rectangle((start, row), end - start + 1, 1, fill=False, edgecolor='red', lw=3))
    
    plt.title(title)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        if force_show:
            plt.show()
    else:
        plt.show()
    plt.close()
    

    # def evaluate_agent(self, agent, day, seed=None):
    #     eval_env_args = self.env_args.copy()
    #     eval_env_args.update({
    #         "eval": True,
    #         "days": [day, day],
    #         "env_class": EvalTradeSimulator,
    #         "seed": seed
    #     })
    #     eval_env = build_env(EvalTradeSimulator, eval_env_args, gpu_id=self.gpu_id)
    #     state, _ = eval_env.reset(seed=seed)        
    #     total_reward = 0
        
    #     for _ in range(eval_env.max_step):
    #         tensor_state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
    #         action, _ = agent.predict(tensor_state, deterministic=self.deteministic_eval)
    #         state, reward, terminated, truncated, _ = eval_env.step(action=action)
    #         total_reward += reward.item()
    #         if terminated or truncated:
    #             break
        
    #     return total_reward
    


class TradeSimulatorTrainer:
    def __init__(self, agent_class, device, start_day, end_day, out_dir, 
                 max_step, params,
                 eval_max_step=None,
                 n_episodes=50,
                 gpu_id=-1, 
                 deterministic_eval=True,
                 seeds: Optional[list[float]]=None, n_seeds: int=1):
        self.agent_class = agent_class
        self.device = device
        self.start_day = start_day
        self.end_day = end_day
        self.out_dir = out_dir
        self.max_step = max_step
        self.eval_max_step = max_step if eval_max_step is None else eval_max_step
        self.params = params
        self.gpu_id = gpu_id
        self.n_episodes = n_episodes
        self.deterministic_eval = deterministic_eval
        self.val_days = list(range(7, 17))
        
        if seeds is None:
            np.random.seed()
            self.seeds = [np.random.randint(2**32 - 1, dtype="int64").item() for i in range(n_seeds)]
            self.n_seeds = n_seeds
        else:
            self.seeds = seeds
            self.n_seeds = len(self.seeds)
        
        print(f'Using seeds: {self.seeds}')
            
        self.env_args = self._initialize_env_args()
        os.makedirs(self.out_dir, exist_ok=True)
    
    def _initialize_env_args(self):
        return {
            "env_name": "TradeSimulator-v0",
            "num_envs": 1,
            "state_dim": 10,
            "action_dim": 3,
            "if_discrete": True,
            "max_position": 1,
            "slippage": 7e-7,
            "num_sims": 1,
            "step_gap": 2,
            "env_class": TradeSimulator,
            "max_step": self.max_step,
            "days": [self.start_day, self.end_day]
        }
    
    def train_agent_with_seed(self, seed=None, learn_params={}):
        train_env_args = self.env_args.copy()
        train_env_args["seed"] = seed
        
        env = build_env(TradeSimulator, train_env_args, gpu_id=self.gpu_id)
        agent = self.agent_class("MlpPolicy", env, verbose=0, seed=seed, **self.params)
        agent.learn(total_timesteps=self.max_step * self.n_episodes, progress_bar=True, **learn_params)
        return agent
    
    def train_and_evaluate(self, save_path: Optional[str]=None):
        returns = []
        sharpe_ratios = []
        for seed in self.seeds:
            agent = self.train_agent_with_seed(seed)
            if save_path is not None:
                agent.save(os.path.join(save_path, f"seed_{seed}"))
            val_returns = []
            val_sharpe_ratios = []
            for val_day in self.val_days:
                final_return, final_sharpe_ratio = self.evaluate_agent(agent, val_day, seed)
                val_returns.append(final_return)
                val_sharpe_ratios.append(final_sharpe_ratio)
            returns.append(np.array(val_returns))  
            sharpe_ratios.append(np.array(val_sharpe_ratios))    
        
        return np.array(returns), np.array(sharpe_ratios)

    
    def evaluate_agent(self, agent, day, seed=None):
        eval_env_args = self.env_args.copy()
        eval_env_args.update({
            "eval": True,
            "days": [day, day],
            "max_step": self.eval_max_step,
            "env_class": EvalTradeSimulator,
            "seed": seed
        })
        eval_env = build_env(EvalTradeSimulator, eval_env_args, gpu_id=self.gpu_id)
        
        last_price = 0        
        current_btc = 0
        starting_cash = 1e6

        cash = [starting_cash]
        btc_assets = [0]
        net_assets = [starting_cash]

        positions = []
        action_ints = []
        correct_pred = []
        current_btcs = [current_btc]
        
        state, _ = eval_env.reset(seed=seed)
                
        total_reward = 0
        for _ in range(eval_env.max_step):
            tensor_state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
            action, _ = agent.predict(tensor_state, deterministic=self.deterministic_eval)
            action_int = action.item() - 1        
            
            
            state, reward, terminated, truncated, _ = eval_env.step(action=action)
            reward_int = reward.item()
            action_ints.append(action_int)
            positions.append(eval_env.position)
            
            # Manually compute cumulative returns
            mid_price = eval_env.price_ary[eval_env.step_i, 2].to(self.device)

            new_cash = cash[-1]

            if action_int > 0 and cash[-1] > mid_price:  # Buy
                last_cash = cash[-1]
                new_cash = last_cash - mid_price
                current_btc += 1
            elif action_int < 0 and current_btc > 0:  # Sell
                last_cash = cash[-1]
                new_cash = last_cash + mid_price
                current_btc -= 1

            cash.append(new_cash)
            btc_assets.append((current_btc * mid_price).item())
            net_assets.append((to_python_number(btc_assets[-1]) + to_python_number(new_cash)))

            # Log win rate
            if action_int == 1:
                correct_pred.append(1 if last_price < mid_price else -1 if last_price > mid_price else 0)
            elif action_int == -1:
                correct_pred.append(-1 if last_price < mid_price else 1 if last_price > mid_price else 0)
            else:
                correct_pred.append(0)

            last_price = mid_price
            current_btcs.append(current_btc)
            
            
            total_reward += reward_int
            if terminated or truncated:
                break
        
        returns = np.diff(net_assets) / net_assets[:-1]
        final_sharpe_ratio = sharpe_ratio(returns)
        # final_max_drawdown = max_drawdown(returns)
        # final_roma = return_over_max_drawdown(returns)
                
        return total_reward, final_sharpe_ratio

def main():
    args = get_cli_args()
    device = torch.device("cpu")
    # agent_class = DQN  # PPO, DQN
    agent_class = DQN if args.agent == 'DQN' else PPO
    window = args.window
    
    first_day = 7
    start_day = window + first_day
    end_day = start_day
    
    print(f"Training {agent_class.__name__} with window {window}")
    
    exp_name_dir = f"{agent_class.__name__}_window_{window}"
    
    storage = f"sqlite:///{EXP_DIR}/tuning/completed/{exp_name_dir}/optuna_study.db"
        
    if storage is not None:
        study = optuna.load_study(study_name=None, storage=storage)
        best_trial = study.best_trial
        model_params = SAMPLER[agent_class.__name__](best_trial, n_actions=3, n_envs=1, additional_args={})
        print(f'Trial number: {best_trial.number}, params: {model_params}, value: {best_trial.value}')
    else:
        model_params = {}

    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f'experiments/train/{agent_class.__name__}_window_{window}_{timestamp}' if args.out_dir is None else args.out_dir
    
    # num_ignore_step = 60
    # step_gap = 2
    # max_step = (4800 - num_ignore_step) // step_gap
    max_step = 480
    eval_max_step = 480
    
    
    trainer = TradeSimulatorTrainer(
        agent_class=agent_class,
        device=device,
        start_day=start_day,
        end_day=end_day,
        out_dir=out_dir,
        max_step=max_step,
        eval_max_step=eval_max_step,
        params=model_params,
        deterministic_eval=True,
        n_episodes=200,
        n_seeds=args.n_seeds,
    )
    
    tuning_results_dir = EXP_DIR / "tuning" / "completed" / "results"
    os.makedirs(tuning_results_dir, exist_ok=True)
    
    saved_agents_dir = tuning_results_dir / "saved_agents" / f'{agent_class.__name__}_window_{window}'
    os.makedirs(saved_agents_dir, exist_ok=True)
    
    returns, sharpe_ratios = trainer.train_and_evaluate(save_path=saved_agents_dir)
    
    plot_dir = f'{out_dir}/plots'
    os.makedirs(plot_dir, exist_ok=True)
    
    
    
    results_dict = {'returns': returns, 'sharpe_ratios': sharpe_ratios}
    for key, results in results_dict.items():
        decimal_places, use_e_notation = (5, False) if key in ['sharpe_ratios'] else (2, False)
        
        train_day_idx = start_day - first_day
        training_days = [[(train_day_idx, train_day_idx)] for _ in range(results.shape[0])]   
        xticklabels = [f'Day {i+first_day}' for i in range(results.shape[1])] 
        yticklabels = [f'Seed {trainer.seeds[i]}' for i in range(results.shape[0])]
        plot_heatmap(results, training_days,
                     title=f'{key} heatmap',
                     xticklabels=xticklabels, yticklabels=yticklabels, 
                     decimal_places=decimal_places, use_e_notation=use_e_notation,
                     save_path=f'{plot_dir}/heatmap_{key}_seeds.png')

        results_mean_seeds = np.mean(results, axis=0)
        results_std_seeds = np.std(results, axis=0)
        single_train_day = [[(train_day_idx, train_day_idx)]]
        plot_heatmap(results_mean_seeds.reshape(1, -1), single_train_day,
                    title=f'Mean {key} heatmap',
                     xticklabels=xticklabels, yticklabels=['Mean Seed'],
                     decimal_places=decimal_places, use_e_notation=use_e_notation,
                     save_path=f'{plot_dir}/heatmap_{key}_mean.png')
        plot_heatmap(results_std_seeds.reshape(1, -1), single_train_day, 
                     xticklabels=xticklabels, yticklabels=['Std Seed'],
                     title=f'Std {key} heatmap',
                     decimal_places=decimal_places, use_e_notation=False,
                     save_path=f'{plot_dir}/heatmap_{key}_std.png')

        res = {
            "all": results,
            "mean": results_mean_seeds,
            "std": results_std_seeds
        }
        with open(f'{out_dir}/{key}.pkl', 'wb') as f:
            pickle.dump(res, f)
        
        
        with open(tuning_results_dir / f"{key}_{exp_name_dir}.pkl", "wb") as f:
            pickle.dump(res, f)
    
    

if __name__ == "__main__":
    main()
    
    
