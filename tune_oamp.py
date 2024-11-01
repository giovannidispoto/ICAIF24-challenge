import numpy as np
from erl_config import build_env
from trade_simulator import EvalTradeSimulator
import optuna
import argparse
import os
from task1_eval import run_evaluation
import glob


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
        default=50,
        help="number of iterations of optuna"
    )
    parser.add_argument(
        '--n_experts',
        type=int,
        default=5,
        help="max nr of experts"
    )
    parser.add_argument(
        '--n_seeds',
        type=int,
        default=1
    )
    parser.add_argument(
        '--agent_dir',
        type=str,
        default="agents/"
    )
    parser.add_argument(
        '--out_dir',
        type=str,
        default="."
    )

    return parser.parse_args()

def tune():
    args = get_cli_args()
    out_dir = args.out_dir
    plot_dir = out_dir + "/plots"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    day_eval = args.day_eval
    max_steps = args.max_steps
    def evaluation(algorithm, eval_env):
        reward = 0
        s, _ = eval_env.reset(eval=True)
        done = truncated = False
        while not (done or truncated):
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
        "eval_sequential": True,
        "env_class": EvalTradeSimulator
    }
    actions_values = [0, 1, 2]
    np.random.seed()
    study = optuna.create_study(direction="maximize", storage= f'sqlite:///{out_dir}/optuna_study.db')

    def objective(trial):

        agents_weights_upd_freq = trial.suggest_int("agents_weights_upd_freq", low=1, high=1000, step=5)
        loss_fn_window = trial.suggest_int("loss_fn_window", low=50, high=1000, step=10)
        action_thresh = trial.suggest_float("action_thresh", low=0.1, high=0.9)
        oamp_params = {
            "agents_weights_upd_freq": agents_weights_upd_freq,
            "loss_fn_window": loss_fn_window,
            "action_thresh": action_thresh
        }
        rewards = []
        env_args["days"] = [args.day_eval]
        eval_env = build_env(EvalTradeSimulator, env_args, -1)
        agents_info = {}
        n_experts = 0
        for i in range(args.n_experts):
            agent_window = args.day_eval - 7 - (i + 1)
            agent_dir = args.agent_dir + f"trial_{agent_window}_window_step_gap_2/"
            policy_list = glob.glob(agent_dir + f'PolicyIter*.pkl')
            policy = None
            max_iteration = -1
            for j, policy_path in policy_list:
                iteration = int(policy_path[-5])
                if iteration > max_iteration:
                    max_iteration = iteration
                    policy = policy_path
            if policy is not None:

                agents_info[f"agent_{n_experts}"] = {"type":"fqi",
                                                     "file":policy}
                n_experts += 1
        run_name = f"day_{args.day_eval}_{n_experts}_experts"
        _, return_ = run_evaluation(run_name, agents_info, oamp_params)
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

    study.optimize(objective, n_trials=args.n_trials)

if __name__ == "__main__":
    tune()