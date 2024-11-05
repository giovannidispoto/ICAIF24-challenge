import os
import matplotlib.pyplot as plt
import numpy as np

from collections import deque

from oamp.oamp_config import ConfigOAMP
from oamp.oamp_utils import (
    get_m,
    get_p,
    get_r,
    upd_n,
    upd_w,
)


class OAMP:
    def __init__(
        self,
        agents_count: int,
        args: ConfigOAMP,
    ):
        # Initializing agents
        self.agents_count = agents_count
        self.agents_rewards = self.init_agents_rewards(args.loss_fn_window)
        self.agents_weights_upd_freq = args.agents_weights_upd_freq
        self.action_thresh = args.action_thresh
        # Initializing OAMP
        self.t = 0
        self.l_tm1 = np.zeros(agents_count)
        self.n_tm1 = np.ones(agents_count) * 0.25
        self.w_tm1 = np.ones(agents_count) / agents_count
        self.p_tm1 = np.ones(agents_count) / agents_count
        self.cum_err = np.zeros(agents_count)
        # Initializing OAMP stats
        self.stats = {
            "losses": [],
            "rewards": [],
            "weights": [],
        }

    def init_agents_rewards(
        self,
        loss_fn_window: int,
    ):
        return deque(maxlen=loss_fn_window)

    def step(
        self,
        agents_rewards: np.ndarray,
        agents_actions: np.ndarray,
    ):
        # Updating agents' rewards
        self.agents_rewards.append(agents_rewards)
        self.stats['rewards'].append(agents_rewards)
        # Updating agents' weights
        if self.t % self.agents_weights_upd_freq == 0:
            agents_weights = self.update_agents_weights()
        else:
            agents_weights = self.p_tm1
        # Updating timestep
        self.t += 1
        return self.compute_action(agents_actions, agents_weights)

    def update_agents_weights(
        self,
    ):
        if self.t % self.agents_weights_upd_freq == 0:
            # Computing agents' losses
            l_t = self.compute_agents_losses()
            # Computing agents' regrets estimates
            m_t = get_m(
                self.l_tm1,
                self.n_tm1,
                self.w_tm1,
                self.agents_count,
            )
            # Computing agents' selection probabilites
            p_t = get_p(m_t, self.w_tm1, self.n_tm1)
            # Computing agents' regrets
            r_t = get_r(l_t, p_t)
            # Computing agents' regrets estimatation error
            self.cum_err += (r_t - m_t) ** 2
            # Updating agents' learning rates
            n_t = upd_n(self.cum_err, self.agents_count)
            # Updating agents' weights
            w_t = upd_w(
                self.w_tm1,
                self.n_tm1,
                n_t,
                r_t,
                m_t,
                self.agents_count,
            )
            self.l_tm1 = l_t
            self.n_tm1 = n_t
            self.w_tm1 = w_t
            self.p_tm1 = p_t
            self.stats["losses"].append(l_t)
            self.stats["weights"].append(self.p_tm1)
        else:
            self.stats["losses"].append(np.zeros(len(self.agents_count)))
            self.stats["weights"].append(self.p_tm1)
        return self.p_tm1

    def compute_agents_losses(
        self,
    ) -> np.ndarray:
        # Computing agents' losses
        agents_losses: np.ndarray = -np.sum(self.agents_rewards, axis=0)
        # Normalizing agents' losses
        agents_losses_min = agents_losses.min()
        agents_losses_max = agents_losses.max()
        if agents_losses_min != agents_losses_max:
            agents_losses = (agents_losses - agents_losses_min) / (
                agents_losses_max - agents_losses_min
            )
        return agents_losses

    def compute_action(
        self,
        agents_actions: np.ndarray,
        agents_weights: np.ndarray,
    ) -> np.ndarray:
        action = np.dot(agents_actions, agents_weights)
        action_int = action - 1
        if np.abs(action_int) < self.action_thresh:
            action = 1
        else:
            action = np.sign(action_int) + 1
        return action

    def plot_stats(
        self,
        save_path: str,
    ):
        agents = [f"Agent {n}" for n in range(self.agents_count)]
        agents_rewards = np.array(self.stats["rewards"])
        agents_losses = np.array(self.stats["losses"])
        agents_weights = np.array(self.stats["weights"])
        fig, axs = plt.subplots(3, 1, figsize=(10, 15))
        axs[0].plot(agents_rewards.cumsum(axis=0))
        axs[0].set_title("Agents' Rewards")
        axs[0].grid()
        axs[1].plot(agents_losses.cumsum(axis=0))
        axs[1].set_title("Agents' Losses")
        axs[1].grid()
        axs[2].stackplot(np.arange(len(agents_weights)), np.transpose(agents_weights))
        axs[2].grid()
        axs[2].set_title("Agents' Weights")
        fig.legend(labels=agents, loc="center left", bbox_to_anchor=(0.95, 0.5))
        fig.savefig(os.path.join(save_path, "oamp_stats.png"), bbox_inches="tight")
        plt.close(fig)
