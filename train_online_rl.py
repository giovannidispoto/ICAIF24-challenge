import argparse
import os
from pathlib import Path
import pickle
from typing import Optional
import numpy as np
import optuna
import torch as th
from datetime import datetime
from sbx import DQN, PPO
from optuna import load_study
from stable_baselines3.common.vec_env import DummyVecEnv
from config import EXP_DIR, ROOT_DIR
from erl_config import build_env
from metrics import max_drawdown, return_over_max_drawdown, sharpe_ratio, sharpe_ratio_ms
from sample_online_rl import ONLINE_RL_NAME_TO_CLASS_DICT
from task1_eval import to_python_number, trade, trade_ms, winloss, winloss_ms
from trade_simulator import EvalTradeSimulator, TradeSimulator
import matplotlib.pyplot as plt
import seaborn as sns
from tune_online_rl import SAMPLER
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from stable_baselines3.common.env_util import make_vec_env

FIRST_DAY = 7
def get_cli_args():
    """Create CLI parser and return parsed arguments"""
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--agent',
        type=str,
        default='DQN'
    )
    
    parser.add_argument(
        '--start_train_day',
        type=int,
        default=7,
        help="start train day"
    )
    parser.add_argument(
        '--end_train_day',
        type=int,
        default=7,
        help="end train day"
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
    parser.add_argument(
        '--progress',
        action='store_true',
        help='Enable progress output',
        default=False
    )
    parser.add_argument(
        '--force_default',
        action='store_true',
        help='Enable progress output',
        default=False
    )

    return parser.parse_args()

def find_all_directories(path: str | Path) -> list[str]:
    if isinstance(path, str):
        path = Path(path)
    directories = [str(p) for p in path.rglob('*') if p.is_dir()]
    return directories


def save_tensorboard_plots(log_dir: str, output_dir: str):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the tensorboard logs
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()
    
    # Get all the scalar tags (metrics names)
    tags = event_acc.Tags()['scalars']
    
    for tag in tags:
        # Retrieve the scalar data for each tag
        events = event_acc.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        
        # Plot the data
        plt.figure(figsize=(10, 6))
        plt.plot(steps, values, label=tag)
        plt.title(f'Training Plot for {tag}')
        plt.xlabel('Steps')
        plt.ylabel(tag)
        plt.grid(True)
        plt.legend()
        
        # Save the figure
        output_path = os.path.join(output_dir, f'{tag.replace("/", "_")}.png')
        plt.savefig(output_path)
        plt.close()

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


class TradeSimulatorTrainer:
    def __init__(self, agent_class, device, 
                 start_day, end_day,
                 out_dir, 
                 max_step, params,
                 num_eval_sims,
                 n_envs,
                 eval_seq,
                show_progress=False,
                 tb_log_path=None,
                 eval_max_step=None,
                 load_model_path: Optional[Path] = None,
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
        self.tb_log_path = tb_log_path
        self.gpu_id = gpu_id
        self.n_episodes = n_episodes
        self.n_envs = n_envs
        self.num_eval_sims=num_eval_sims
        self.show_progress = show_progress
        self.deterministic_eval = deterministic_eval
        self.eval_seq = eval_seq
        self.load_model_path = load_model_path
        self.val_days = list(range(7, 17))
        # self.val_days = [7, 8]
        
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
            "num_sims": 1,
            "state_dim": 10,
            "action_dim": 3,
            "if_discrete": True,
            "max_position": 1,
            "slippage": 7e-7,
            "step_gap": 2,
            "eval_sequential": False,
            "env_class": TradeSimulator,
            "max_step": self.max_step,
            "days": [self.start_day, self.end_day]
        }
    
    def train_agent_with_seed(self, seed=None, learn_params={}):
        curr_tb_log_path = None if self.tb_log_path is None else os.path.join(self.tb_log_path, f"seed_{seed}")        
        env = make_vec_env(
            lambda: build_env(TradeSimulator, {**self.env_args, "seed": seed}, gpu_id=self.gpu_id),
            n_envs=self.n_envs,
            seed=seed
        )
        if self.load_model_path is not None:
            print(f'Loading model from {self.load_model_path}')
            agent = self.agent_class.load(self.load_model_path)
        else:
            print(f'Training with seed: {seed} on days: [{self.start_day}, {self.end_day}]')
            agent = self.agent_class("MlpPolicy", env, verbose=0, seed=seed, tensorboard_log=curr_tb_log_path, **self.params)
            agent.learn(total_timesteps=self.max_step * self.n_episodes, progress_bar=self.show_progress, **learn_params)
            
            # Plot tb plots
            if curr_tb_log_path is not None:
                tb_dirs = find_all_directories(curr_tb_log_path)
                tb_plot_dir = os.path.join(curr_tb_log_path, "plots")
                os.makedirs(tb_plot_dir, exist_ok=True)
                for tb_dir in tb_dirs:
                    tb_curr_plot_dir = os.path.join(tb_plot_dir, tb_dir.split('/')[-1])
                    os.makedirs(tb_curr_plot_dir, exist_ok=True)
                    save_tensorboard_plots(tb_dir, tb_curr_plot_dir)
        
        return agent
    
    def train_and_evaluate(self, save_path: Optional[str]=None):
        returns = []
        sharpe_ratios = []
        returns_std = []
        for seed in self.seeds:
            curr_path = None if save_path is None else os.path.join(save_path, f"seed_{seed}")
            agent = self.train_agent_with_seed(seed=seed)
            if curr_path is not None:
                agent.save(curr_path)
            val_returns = []
            val_sharpe_ratios = []
            val_std_returns = []
            for val_day in self.val_days:
                final_return, final_sharpe_ratio, final_returns_std = self.evaluate_agent(agent, [val_day, val_day], seed=seed)
                val_returns.append(final_return)
                val_sharpe_ratios.append(final_sharpe_ratio)
                val_std_returns.append(final_returns_std)
            returns.append(np.array(val_returns))  
            sharpe_ratios.append(np.array(val_sharpe_ratios))  
            returns_std.append(np.array(val_std_returns)) 
             
        
        return np.array(returns), np.array(sharpe_ratios), np.array(returns_std)
    
    
    #TODO: Check multienv sharpe ratio calculation, don't know if it's correct
    def evaluate_agent(self, agent, days, seed=None):
        print(f'Evaluation on: {days}')
        eval_env_args = self.env_args.copy()
        eval_env_args.update({
            "eval": True,
            "days": days,
            "num_sims": self.num_eval_sims,
            "num_envs": 1,
            "max_step": self.eval_max_step,
            "env_class": EvalTradeSimulator,
            "seed": seed,
            "eval_sequential": self.eval_seq
        })
        eval_env = build_env(EvalTradeSimulator, eval_env_args, gpu_id=self.gpu_id)
        
        last_price = 0   
        current_btc = th.zeros(self.num_eval_sims, dtype=th.float32, device=self.device)   
        starting_cash = 1e6

        cash = th.full((1, self.num_eval_sims), starting_cash, dtype=th.float32, device=self.device)
        btc_assets = th.zeros((1, self.num_eval_sims), dtype=th.float32, device=self.device)
        net_assets = th.full((1, self.num_eval_sims), starting_cash, dtype=th.float32, device=self.device)


        positions = th.empty((0, self.num_eval_sims), dtype=th.long, device=self.device)
        correct_pred = th.empty((0, self.num_eval_sims), dtype=th.long, device=self.device)
        current_btcs = th.zeros((1, self.num_eval_sims), dtype=th.long, device=self.device)
        actions = th.empty((0, self.num_eval_sims), dtype=th.long, device=self.device)

        state, _ = eval_env.reset(seed=seed, eval_sequential=self.eval_seq)
        
        total_reward = th.zeros(self.num_eval_sims, dtype=th.float32, device=self.device)
        rewards = th.empty((0, self.num_eval_sims), dtype=th.float32, device=self.device)
        
        
        print('Max step: ', eval_env.max_step)
        for i in range(eval_env.max_step):
            tensor_state = th.as_tensor(state, dtype=th.float32, device=self.device)
            
            # action= eval_env.action_space.sample()
            # action = np.random.choice(3, size=self.num_eval_sims)
            
            # action = np.full((self.num_eval_sims,), 2) # Long
            # action = np.full((self.num_eval_sims,), 0) # Short
            # action = np.full((self.num_eval_sims,), 1) # Hold
            
            action, _ = agent.predict(tensor_state, deterministic=self.deterministic_eval)
            
            action = th.from_numpy(action).to(self.device)            
            state, reward, terminated, truncated, _ = eval_env.step(action=action)
            
            rewards = th.cat((rewards, reward.unsqueeze(0)), dim=0)
                            
            price: float = eval_env.price_ary[eval_env.step_i, 2].to(self.device) # scalar
            
            new_cash, current_btc = trade_ms(
                action, price, cash[-1, :], current_btc
            )
                        
            cash = th.cat((cash, new_cash.unsqueeze(0)), dim=0)
            btc_assets = th.cat((btc_assets, (current_btc * price).unsqueeze(0)), dim=0)
            net_assets = th.cat((net_assets, (btc_assets[-1, :] + new_cash).unsqueeze(0)), dim=0)

            # # Upadting trading history
            position = eval_env.position
            positions = th.cat((positions, position.unsqueeze(0)), dim=0)
            actions = th.cat((actions, action.unsqueeze(0)), dim=0)
            current_btcs = th.cat((current_btcs, current_btc.unsqueeze(0)), dim=0)
            correct_pred = th.cat((correct_pred, winloss_ms(action, last_price, price).unsqueeze(0)), dim=0)
            # # Updating last state and price
            last_price = price
            total_reward += reward
            
            if terminated.any() or truncated:
                break
        
        print(f'Steps: {i}')
        # print(total_reward)
        
        mean_total_reward = total_reward.mean().item()
        std_total_reward = total_reward.std().item() if self.num_eval_sims > 1 else 0.
        print(f'Sims mean: {mean_total_reward} Sims std: {std_total_reward}')
        
        mean_std_reward = rewards.std(dim=0).mean().item()
        
        # returns = (net_assets[1:, :] - net_assets[:-1, :]) / net_assets[:-1, :]
        # final_sharpe_ratio = sharpe_ratio_ms(returns.numpy())
        # final_sharpe_ratio = np.array([r for r in final_sharpe_ratio if r != np.inf])
        # mean_final_sharpe_ratio = final_sharpe_ratio.mean() if len(final_sharpe_ratio) > 0 else 0
        mean_final_sharpe_ratio=0
        
        # # Action distribution in each simulation
        # action_counts_matrix = th.zeros(3, actions.shape[1], dtype=th.long)
        # for i in range(actions.shape[1]):
        #     action_counts_matrix[:, i] = th.bincount(actions[:, i], minlength=3)
        
        return mean_total_reward, mean_final_sharpe_ratio, mean_std_reward



def main():
    args = get_cli_args()
    device = th.device("cpu")
    # agent_class = DQN  # PPO, DQN, A2C
    agent_class = ONLINE_RL_NAME_TO_CLASS_DICT[args.agent.lower()]
    
    start_train_day = args.start_train_day
    end_train_day = args.end_train_day
    window=f'{start_train_day}_{end_train_day}'
    
    print(f"Training {agent_class.__name__} with window {window}")
    
    exp_name_dir = f"{agent_class.__name__}_window_{window}"
    
    storage = EXP_DIR / f"tuning/completed/{exp_name_dir}/optuna_study.db"
    # storage = None
    if storage is not None and os.path.exists(storage) and not args.force_default:
        print(f'Loading best params from {storage}')
        study = optuna.load_study(study_name=None, storage=f"sqlite:///{storage}")
        
        # best_trial = study.trials[0] # First trial
        best_trial = study.best_trial
        
        model_params = SAMPLER[agent_class.__name__](best_trial, n_actions=3, n_envs=1, additional_args={})
        print(f'Trial number: {best_trial.number}, params: {model_params}, value: {best_trial.value}')
    else:
        print(f'Loading default params')
        model_params = {}

    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f'experiments/train/{agent_class.__name__}_window_{window}_{timestamp}' if args.out_dir is None else args.out_dir
    tb_log_path = f'{out_dir}/tb_logs'
    
    max_step=480
    eval_max_step=480
    
    # load_model_path = ROOT_DIR / "agents" / "ppo" / "new_tuning" / f"PPO_window_{start_train_day}_{end_train_day}"
    # if not os.path.exists(load_model_path):
    #     raise FileNotFoundError(f'Model path {load_model_path} does not exist')
    # seeds_dir = [seed_dir for seed_dir in os.listdir(load_model_path)]
    # assert len(seeds_dir) == 1
    # load_model_path = os.path.join(load_model_path, seeds_dir[0])
    
    load_model_path = None
    
    trainer = TradeSimulatorTrainer(
        agent_class=agent_class,
        device=device,
        start_day=start_train_day,
        end_day=end_train_day,
        out_dir=out_dir,
        max_step=max_step,
        eval_max_step=eval_max_step,
        tb_log_path=tb_log_path,
        params=model_params,
        show_progress=args.progress,
        deterministic_eval=False,
        load_model_path= load_model_path,
        n_episodes=700,
        num_eval_sims=50,
        eval_seq = False,
        n_envs=4,
        n_seeds=args.n_seeds,
    )
    
    tuning_results_dir = EXP_DIR / "tuning" / "completed" / "results"
    os.makedirs(tuning_results_dir, exist_ok=True)
    
    saved_agents_dir = tuning_results_dir / "saved_agents" / f'{agent_class.__name__}_window_{window}'
    os.makedirs(saved_agents_dir, exist_ok=True)
    
    returns, sharpe_ratios, returns_std = trainer.train_and_evaluate(save_path=saved_agents_dir)
    
    plot_dir = f'{out_dir}/plots'
    os.makedirs(plot_dir, exist_ok=True)
    
    # results_dict = {'returns': returns, 'sharpe_ratios': sharpe_ratios, 'returns_std': returns_std}
    results_dict = {'returns': returns, 'step_std': returns_std}

    for key, results in results_dict.items():
        decimal_places, use_e_notation = (5, False) if key in ['sharpe_ratios', 'returns_std'] else (2, False)
        
        start_train_day_idx = start_train_day - FIRST_DAY
        end_train_day_idx = end_train_day - FIRST_DAY
        training_days = [[(start_train_day_idx, end_train_day_idx)] for _ in range(results.shape[0])]   
        xticklabels = [f'Day {i+FIRST_DAY}' for i in range(results.shape[1])] 
        yticklabels = [f'Seed {trainer.seeds[i]}' for i in range(results.shape[0])]
        plot_heatmap(results, training_days,
                     title=f'{key} heatmap',
                     xticklabels=xticklabels, yticklabels=yticklabels, 
                     decimal_places=decimal_places, use_e_notation=use_e_notation,
                     save_path=f'{plot_dir}/heatmap_{key}_seeds.png')

        results_mean_seeds = np.mean(results, axis=0)
        results_std_seeds = np.std(results, axis=0)
        single_train_day = [[(start_train_day_idx, end_train_day_idx)]]
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
    
    