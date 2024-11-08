import numpy as np
from trlib.algorithms.algorithm import Algorithm
from gymnasium import spaces
from trlib.policies.qfunction import FittedQ, DiscreteFittedQ, DiscreteFittedQSaveMemory, DiscreteBoostedFittedQSaveMemory, DiscreteNeuralQ
from trlib.policies.policy import Uniform
from trlib.utilities.interaction import generate_episodes, split_data


class FQI(Algorithm):
    """
    Fitted Q-Iteration
    
    References
    ----------
      - Ernst, Damien, Pierre Geurts, and Louis Wehenkel
        Tree-based batch mode reinforcement learning
        Journal of Machine Learning Research 6.Apr (2005): 503-556
    """
    
    def __init__(self, mdp, policy, actions, batch_size, max_iterations, regressor_type, 
                init_policy=None, verbose=False, neural_q=False, save_memory=False,
                double_Q=False, swap_Q=False, tau=None, double_Q_strategy = 'mean', **regressor_params):
        
        super().__init__("FQI", mdp, policy, verbose)
        
        self._actions = actions
        self._batch_size = batch_size
        self._max_iterations = max_iterations
        self._regressor_type = regressor_type
        self.double_Q = double_Q

        self.swap_Q = swap_Q
        self.tau = tau
        if self.swap_Q:
            self.tau=None

        if init_policy is None:
            self._init_policy = Uniform(actions)
        else:
            self._init_policy = init_policy

        if neural_q:
            self._policy.Q = DiscreteNeuralQ(regressor_type, mdp.state_dim, actions, **regressor_params)
        elif save_memory:
            if regressor_type == xgboost.sklearn.XGBRegressor:
                self._policy.Q = DiscreteBoostedFittedQSaveMemory(regressor_type, mdp.state_dim, actions, double_Q=double_Q, tau=self.tau, **regressor_params)
            else:
                self._policy.Q = DiscreteFittedQSaveMemory(regressor_type, mdp.state_dim, actions, double_Q=double_Q, tau=self.tau, **regressor_params)
        elif isinstance(mdp.action_space, spaces.Discrete):
            self._policy.Q = DiscreteFittedQ(regressor_type, mdp.state_dim, actions, double_Q=double_Q, swap_Q=swap_Q, tau=self.tau, join_type= double_Q_strategy, **regressor_params)
        else:
            self._policy.Q = FittedQ(regressor_type, mdp.state_dim, mdp.action_dim, **regressor_params)
        
        self.reset()
        
    def _iter(self, sa, r, s_prime, absorbing, return_loss_per_sample: bool= False, **fit_params):

        self.display("Iteration {0}".format(self._iteration))

        if self._iteration == 0:
            y = r
        else:
            if self.swap_Q:
                maxq, _ = self._policy.Q.max(s_prime, self._actions, absorbing, double_mask=fit_params['double_mask'])
                y = r.ravel() + self._mdp.gamma * maxq
            else:
                maxq, _ = self._policy.Q.max(s_prime, self._actions, absorbing)
                y = r.ravel() + self._mdp.gamma * maxq

        if return_loss_per_sample:
            regressors1_loss, regressors2_loss = self._policy.Q.fit(sa, y.ravel(), return_loss_per_sample, **fit_params)
        else:
            self._policy.Q.fit(sa, y.ravel(), return_loss_per_sample, **fit_params)

        self._iteration += 1

        if return_loss_per_sample:
            if self.double_Q:
                return regressors1_loss, regressors2_loss
            else:
                return regressors1_loss, None
        
    def _step_core(self, **kwargs):
        
        policy = self._policy if self._step > 1 else self._init_policy
        self._data.append(generate_episodes(self._mdp, policy, self._batch_size))
        self.n_episodes += self._batch_size
        data = np.concatenate(self._data)
        self._iteration = 0
        
        _, _, _, r, s_prime, absorbing, sa = split_data(data, self._mdp.state_dim, self._mdp.action_dim)
        
        for _ in range(self._max_iterations):
            self._iter(sa, r, s_prime, absorbing, **kwargs)
            
        self._result.update_step(n_episodes=self.n_episodes, n_samples=data.shape[0])
    
    def reset(self):
        
        super().reset()
        
        self._data = []
        self._iteration = 0
        
        self._result.add_fields(batch_size=self._batch_size, max_iterations=self._max_iterations,
                                regressor_type=str(self._regressor_type.__name__), policy = str(self._policy.__class__.__name__))
