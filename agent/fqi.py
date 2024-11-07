import os
import numpy as np
import pickle
from trlib.algorithms.reinforcement.fqi import FQI
from trlib.policies.qfunction import ZeroQ
from trlib.policies.valuebased import EpsilonGreedy
from sklearn.ensemble import ExtraTreesRegressor
from agent.base import AgentBase
from agent.baselines import ShortOnlyBaseline, LongOnlyBaseline, RandomBaseline, FlatOnlyBaseline
from trade_simulator import TradeSimulator
from erl_config import build_env

policies = {
    'random_policy': RandomBaseline(),
    'long_only_policy': LongOnlyBaseline(),
    'short_only_policy': ShortOnlyBaseline(),
    'flat_only_policy': FlatOnlyBaseline()
}

class AgentFQI(AgentBase):
    def __init__(
            self,
            policy_path: str = None,
    ):
        if policy_path is not None:
            self.load(policy_path)

    def action(
            self,
            state: np.ndarray,
    ):
        q_values, max_actions = self.policy.Q.max(state)
        return np.array(max_actions)

    def read_dataset(self, sample_days, policies_to_read=None, data_dir='./data/'):
        policies_unread = []
        state_actions = []
        rewards = []
        absorbing_state = []
        next_states = []

        if policies_to_read is None:
            policies_to_read = policies.keys()
        for p in policies_to_read:
            try:
                path_name = f"{data_dir}/{p}_{sample_days}.pkl"
                data = pickle.load(open(path_name, "rb"))
                state_actions.append(data["state_actions"])
                rewards.append(data["rewards"])
                absorbing_state.append(data["absorbing_state"])
                next_states.append(data["next_states"])
            except:
                policies_unread.append(p)
        if len(state_actions) > 0:
            state_actions = np.concatenate(state_actions)
            rewards = np.concatenate(rewards)
            next_states = np.concatenate(next_states)
            absorbing_state = np.concatenate(absorbing_state)
        return state_actions, rewards, next_states, absorbing_state, policies_unread

    def generate_experience(self, env_args, days_to_sample, policy, max_steps=360, episodes=1000, save=True, data_dir='./data/'):
        pi = policies[policy]
        env = build_env(TradeSimulator, env_args, -1)
        states = []
        actions = []
        rewards = []
        absorbing_state = []
        next_states = []
        s, _ = env.reset()
        for step in range(max_steps):
            states.append(s.numpy())
            a = pi(s)
            s, r, done, truncated, info = env.step(a)
            actions.append(a)
            rewards.append(r.numpy())
            next_states.append(s.numpy())
            absorbing_state.append(done.numpy())
            if done.any():
                break
        states = np.concatenate(states)
        actions = np.concatenate(actions)[:, None]
        state_actions = np.concatenate([states, actions], axis=1)
        rewards = np.concatenate(rewards)
        next_states = np.concatenate(next_states)
        absorbing_state = np.concatenate(absorbing_state)

        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        if save:
            data = {
                "state_actions": state_actions,
                "rewards": rewards,
                "next_states": next_states,
                "absorbing_state": absorbing_state
            }
            file_name = f'{data_dir}/{policy}_{days_to_sample}.pkl'
            with open(file_name, 'wb+') as f:
                pickle.dump(data, f)
        return state_actions, rewards, next_states, absorbing_state

    def train(self, state_actions, rewards, next_states, absorbing, env, args):
        actions_values = [0, 1, 2]
        pi = EpsilonGreedy(actions_values, ZeroQ(), epsilon=0)
        max_iterations = args.get('iterations', 3)
        n_estimators = args.get('n_estimators', 100)
        max_depth = args.get('max_depth', 20)
        min_split = args.get('min_samples_split', np.random.randint(low=10000, high=100000))
        n_jobs = args.get('n_jobs', 10)
        seed = args.get('seed', np.random.randint(10000))

        self.algorithm = FQI(mdp=env, policy=pi, actions=actions_values, batch_size=5, max_iterations=max_iterations,
                             regressor_type=ExtraTreesRegressor, random_state=seed, n_estimators=n_estimators,
                             n_jobs=n_jobs,
                             max_depth=max_depth, min_samples_split=min_split)

        for i in range(max_iterations):
            self.algorithm._iter(
                state_actions,
                rewards,
                next_states,
                absorbing,
            )
            self.policy = self.algorithm._policy
        # prepare for faster eval
        for i in range(3):
            self.policy.Q._regressors[i].n_jobs = 1

    def load(self, policy_path):
        self.policy = pickle.load(open(policy_path, "rb"))
        # print(self.policy)
        for i in range(3):
            self.policy.Q._regressors[i].n_jobs = 1

    def save(self, out_dir, name=""):
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        model_name = out_dir + f'/fqi_{name}.pkl'
        with open(model_name, 'wb+') as f:
            pickle.dump(self.policy, f)
