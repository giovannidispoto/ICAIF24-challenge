import argparse
import glob
import os
import numpy as np
import optuna

from trade_simulator import EvalTradeSimulator
from task1_eval import run_evaluation


def get_cli_args():
    """Create CLI parser and return parsed arguments"""
    parser = argparse.ArgumentParser()
    # Example-specific args.
    parser.add_argument(
        '--day_eval',
        type=int,
        default=15,
        help="day to evaluate "
    )
    parser.add_argument(
        '--n_trials',
        type=int,
        default=2,
        help="number of iterations of optuna"
    )
    parser.add_argument(
        '--n_seeds',
        type=int,
        default=1
    )
    parser.add_argument(
        '--agent_dir',
        type=str,
        default="agents"
    )
    parser.add_argument(
        '--out_dir',
        type=str,
        default="experiments/oamp/"
    )

    return parser.parse_args()

def tune(exp_name=None, agents_list=[]):
    args = get_cli_args()
    env_args = {
        "env_name": "TradeSimulator-v0",
        "num_envs": 1,
        "max_step": 3600,
        "state_dim": 8 + 2,  # factor_dim + (position, holding)
        "action_dim": 3,  # long, 0, short
        "if_discrete": True,
        "max_position": 1,
        "slippage": 7e-7,
        "num_sims": 1,
        "step_gap": 2,
        "eval_sequential": True,
        "env_class": EvalTradeSimulator,
        "days": [args.day_eval, args.day_eval],
    }
    np.random.seed(0)

    agents_info = {}
    agents_count = 0
    agents_dir_names = (
        os.listdir(args.agent_dir)
        if len(agents_list) == 0
        else list(set(os.listdir(args.agent_dir)).intersection(set(agents_list)))
    )
    for agent_dir_name in agents_dir_names:
        agent_class = agent_dir_name.split("_")[0].lower()
        if agent_class == 'fqi':
            policy_list = glob.glob(f'{args.agent_dir}/{agent_dir_name}/Policy_iter*.pkl')
            policy = None
            max_iteration = -1
            for policy_path in policy_list:
                iteration = int(policy_path[-5])
                if iteration > max_iteration:
                    max_iteration = iteration
                    policy = policy_path.split(args.agent_dir+'/')[1]
            if policy is not None:
                print(f"Using Expert:{policy}")
                agents_info[f"agent_{agents_count}"] = {
                    "type": agent_class,
                    "file": policy,
                }
                agents_count += 1
        elif agent_class in ['dqn', 'ppo']:
            agents_info[f"agent_{agents_count}"] = {
                "type": agent_class,
                "file": agent_dir_name,
            }
        agents_count += 1
    
    if exp_name is None:
        exp_name = f'/day_{args.day_eval}_experts_{agents_count}'
    out_dir = args.out_dir + exp_name
    plot_dir = out_dir + "/plots"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    def objective(trial):
        run_name = f"oamp/{exp_name}/trial_{trial.number}"

        agents_weights_upd_freq = trial.suggest_int("agents_weights_upd_freq", low=1, high=100, step=5)
        loss_fn_window = trial.suggest_int("loss_fn_window", low=1, high=100, step=5)
        action_thresh = 0.5 # trial.suggest_float("action_thresh", low=0.1, high=0.9)
        oamp_params = {
            "agents_weights_upd_freq": agents_weights_upd_freq,
            "loss_fn_window": loss_fn_window,
            "action_thresh": action_thresh
        }

        _, return_ = run_evaluation(run_name, agents_info, oamp_params, env_args)
        if trial.number > 1:
            fig = optuna.visualization.plot_optimization_history(study)
            fig.write_image(plot_dir + '/ParamsOptHistory.png')
            fig = optuna.visualization.plot_param_importances(study)
            fig.write_image(plot_dir + '/ParamsImportance.png')
            fig = optuna.visualization.plot_contour(study)
            fig.write_image(plot_dir + '/ParamsContour.png', width=3000, height=1750)
            fig = optuna.visualization.plot_slice(study)
            fig.write_image(plot_dir + '/ParamsSlice.png')
        return return_
    
    study = optuna.create_study(direction="maximize", storage= f'sqlite:///{out_dir}/optuna_study.db')
    study.optimize(objective, n_trials=args.n_trials)

if __name__ == "__main__":
    exp_name = 'FQI_all'
    agents_list = [
        'FQI_window_0_v2',
        'FQI_window_1_v2',
        'FQI_window_2_v2', 
        'FQI_window_3_v2', 
        'FQI_window_4_v2', 
        'FQI_window_5_v2', 
        'FQI_window_6_v2', 
        'FQI_window_7_v2',
        # 'DQN_window_0',
        # 'DQN_window_1',
        # 'DQN_window_2',
        # 'DQN_window_3',
        # 'DQN_window_4',
        # 'DQN_window_5',
        # 'DQN_window_6',
        # 'DQN_window_7',
        # 'PPO_window_0',
        # 'PPO_window_1',
        # 'PPO_window_2',
        # 'PPO_window_3',
        # 'PPO_window_4',
        # 'PPO_window_5',
        # 'PPO_window_6',
        # 'PPO_window_7',
    ]
    tune(exp_name, agents_list)