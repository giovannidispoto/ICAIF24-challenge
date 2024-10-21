
from erl_config import build_env
from trade_simulator import TradeSimulator, EvalTradeSimulator
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.results_plotter import load_results, ts2xy
import numpy as np
import os

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """

    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), "timesteps")
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(
                        f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}"
                    )

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print(f"Saving new best model to {self.save_path}.zip")
                    self.model.save(self.save_path)

        return True

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
        "step_gap": 2,
        "env_class": TradeSimulator
    }

log_dir = "/tmp/gym/"
env = build_env(TradeSimulator, env_args, -1)
env = Monitor(env, log_dir)
env_args["eval"] = True
eval_env = build_env(TradeSimulator, env_args, -1)
eval_env = Monitor(eval_env, log_dir)
callback = SaveOnBestTrainingRewardCallback(check_freq=max_steps*100, log_dir=log_dir)
eval_callback = EvalCallback(eval_env,
                             log_path="./logs_eval/", eval_freq=max_steps*100, n_eval_episodes=100,
                             deterministic=True, render=False, )
# set up logger
model = PPO("MlpPolicy", env, verbose=0, tensorboard_log="./ppo_tensorboard/")
model.learn(total_timesteps=max_steps*1000, callback=[callback, eval_callback], progress_bar=True)
model.save("PPO_Train")
env.close()

