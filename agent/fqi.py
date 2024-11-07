import os
import numpy as np
import pickle
from trlib.algorithms.reinforcement.fqi import FQI
from trlib.policies.qfunction import ZeroQ
from trlib.policies.valuebased import EpsilonGreedy
from sklearn.ensemble import ExtraTreesRegressor
from agent.base import AgentBase


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
        q_values = self.policy._q_values(state)
        return np.argmax(q_values).item()

    def train(self, state_actions, rewards, next_states, absorbing, env, args):
        actions_values = [0, 1, 2]
        pi = EpsilonGreedy(actions_values, ZeroQ(), epsilon=0)
        max_iterations = args.get('iterations', 3)
        n_estimators = args.get('n_estimators', 100)
        max_depth = args.get('max_depth', 20)
        min_split = args.get('min_samples_split', np.random.randint(low=10000, high=100000))
        n_jobs = args.get('n_jobs', 10)

        seed = args['seed']

        self.algorithm = FQI(mdp=env, policy=pi, actions=actions_values, batch_size=5, max_iterations=max_iterations,
                             regressor_type=ExtraTreesRegressor, random_state=seed, n_estimators=n_estimators,
                             n_jobs=n_jobs,
                             max_depth=max_depth, min_samples_split=min_split)

        for i in range(max_iterations):
            self.algorithm._iter(
                state_actions.to_numpy(dtype=np.float32),
                rewards.to_numpy(dtype=np.float32),
                next_states.to_numpy(dtype=np.float32),
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
        model_name = out_dir + f'fqi_{name}.pkl'
        with open(model_name, 'wb+') as f:
            pickle.dump(self.policy, f)
