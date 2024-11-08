import numpy as np
from numpy import matlib


class QFunction:
    """
    Base class for all Q-functions
    """

    def __call__(self, state, action):
        """
        Computes the value of the given state-action couple

        Parameters
        ----------
        state: an S-dimensional vector
        action: an A-dimensional vector

        Returns
        -------
        The value of (state,action).
        """
        raise NotImplementedError

    def values(self, sa, get_two_estimations=False):
        """
        Computes the values of all state-action vectors in sa

        Parameters
        ----------
        sa: an Nx(S+A) matrix

        Returns
        -------
        An N-dimensional vector with the value of each state-action row vector in sa
        """
        raise NotImplementedError

    def max(self, states, actions=None, absorbing=None):
        """
        Computes the action among actions achieving the maximum value for each state in states

        Parameters:
        -----------
        states: an NxS matrix
        actions: a list of A-dimensional vectors
        absorbing: an N-dimensional vector specifying whether each state is absorbing

        Returns:
        --------
        An NxA matrix with the maximizing actions and an N-dimensional vector with their values
        """
        raise NotImplementedError


class ZeroQ(QFunction):
    """
    A QFunction that is zero for every state-action couple.
    """

    def __call__(self, state, action):
        return 0

    def values(self, sa):
        return np.zeros(np.shape(sa)[0])


class FittedQ(QFunction):
    """
    A FittedQ is a Q-function represented by an underlying regressor that has been fitted on some data.
    The regressor receives SA-dimensional vectors and predicts their scalar value.
    This should be preferred for continuous action-spaces. For discrete action-spaces use DiscreteFittedQ instead.
    """

    def __init__(self, regressor_type, state_dim, action_dim, **regressor_params):
        self._regressor = regressor_type(**regressor_params)
        self._state_dim = state_dim
        self._action_dim = action_dim

    def __call__(self, state, action):
        return self.values(np.concatenate((state,action),0)[np.newaxis,:])

    def values(self, sa):
        if not np.shape(sa)[1] == self._state_dim + self._action_dim:
            raise AttributeError("An Nx(S+A) matrix must be provided")
        return self._regressor.predict(sa)

    def max(self, states, actions=None, absorbing=None):
        if not np.shape(states)[1] == self._state_dim:
            raise AttributeError("Wrong dimensions of the input matrices")
        if actions is None:
            raise AttributeError("Actions must be provided")

        n_actions = len(actions)
        n_states = np.shape(states)[0]
        actions = np.array(actions).reshape((n_actions,self._action_dim))
        sa = np.empty((n_states * n_actions, self._state_dim + self._action_dim))
        for i in range(n_states):
            sa[i*n_actions:(i+1)*n_actions,0:self._state_dim] = matlib.repmat(states[i,:], n_actions, 1)
            sa[i*n_actions:(i+1)*n_actions,self._state_dim:] = actions

        vals = self.values(sa)
        if absorbing is not None:
            absorbing = matlib.repmat(absorbing,n_actions,1).T.flatten()
            vals[absorbing == 1] = 0

        max_vals = np.empty(n_states)
        max_actions = np.empty((n_states,self._action_dim))
        for i in range(n_states):
            val = vals[i*n_actions:(i+1)*n_actions]
            a = np.argmax(val)
            max_vals[i] = val[a]
            max_actions[i,:] = actions[a,:]

        return (max_vals,max_actions)

    def fit(self, sa, q, **fit_params):
        self._regressor.fit(sa, q, **fit_params)


class DiscreteFittedQ(QFunction):
    """
    A DiscreteFittedQ is a Q-function represented by a set of underlying regressors, one for each discrete action.
    This is only for discrete action-spaces.
    """

    def __init__(self, regressor_type, state_dim, actions, double_Q=False, swap_Q=False, tau=None, join_type='mean', **regressor_params):
        self._n_actions = len(actions)
        self._state_dim = state_dim
        self._actions = actions
        self.double_Q = double_Q
        self.swap_Q = swap_Q
        self.tau = tau
        self.join_type = join_type

        if self.double_Q:
            self._regressors1 = {}
            self._regressors2 = {}
        else:
            self._regressors = {}

        if regressor_type.__name__=="MLP_regressor_jax":
          keys = regressor_params['keys']
          del regressor_params['keys']
          for a_i, a in enumerate(actions):
            if double_Q:
                self._regressors1[a] = regressor_type(key=keys[a_i], **regressor_params)
                self._regressors2[a] = regressor_type(key=keys[a_i], **regressor_params)
            else:
                self._regressors[a] = regressor_type(key=keys[a_i], **regressor_params)
        else:
          for a in actions:
                if self.double_Q:
                    self._regressors1[a] = regressor_type(**regressor_params)
                    self._regressors2[a] = regressor_type(**regressor_params)
                else:
                    self._regressors[a] = regressor_type(**regressor_params)

    def __call__(self, state, action):
        if not np.shape(state)[0] == self._state_dim:
            raise AttributeError("State is not of the right shape")
        if not action in self._actions:
            raise AttributeError("Action does not exist")
        return self._action_values(state[np.newaxis,:], action)

    def _action_values(self, states, action):
        if self.double_Q:
            return self._regressors1[action].predict(states), self._regressors2[action].predict(states)
        else:
            return self._regressors[action].predict(states)

    def _action_values_stds(self, states, action):
        if self.double_Q:
            preds1 = [self._regressors1[action].estimators_[j].predict(states) for j in range(self._regressors1[action].n_estimators)]
            preds2 = [self._regressors2[action].estimators_[j].predict(states) for j in range(self._regressors2[action].n_estimators)]
            return np.mean(preds1,axis=0), np.mean(preds2,axis=0), np.std(preds1, axis=0, ddof=1), np.std(preds2, axis=0, ddof=1)
        else:
            preds = [self._regressors[action].estimators_[j].predict(states) for j in range(self._regressors[action].n_estimators)]
            return np.mean(preds,axis=0), np.std(preds, axis=0, ddof=1)

    def values(self, sa, get_std=False, get_two_estimations=False):
        # print(sa.shape) #############################
        if not np.shape(sa)[1] == self._state_dim + 1:
            print(sa.shape) #############################
            raise AttributeError("An Nx(S+1) matrix must be provided")

        vals = np.zeros(np.shape(sa)[0])
        if get_std:
            stds = np.zeros(np.shape(sa)[0])
        if self.double_Q:
            split_vals = np.zeros((np.shape(sa)[0],2))
            stds = np.zeros((np.shape(sa)[0],2))
        check_mask = np.zeros(np.shape(sa)[0])

        for a in self._actions:
            mask = sa[:, -1] == a
            check_mask = np.logical_or(mask, check_mask)
            if self.double_Q:
                if get_std:
                    split_vals[mask,0], split_vals[mask,1], stds[mask,0], stds[mask,1] = self._action_values_stds(sa[mask, 0:-1], a)
                else:
                    split_vals[mask,0], split_vals[mask,1] = self._action_values(sa[mask, 0:-1], a)
                if get_two_estimations is False:
                    vals = get_join_vals(split_vals, self.join_type, self.tau)
                else:
                    vals = (np.min(split_vals, axis=-1), np.max(split_vals, axis=-1))

            else:
                if get_std:
                    vals[mask], stds[mask] = self._action_values_stds(sa[mask, 0:-1], a)
                else:
                    vals[mask] = self._action_values(sa[mask, 0:-1], a)

        if not np.all(check_mask):
            raise AttributeError("Some action in sa does not exist")

        if get_std:
            return vals, stds
            #TODO: compute stds for double Q case.
        else:
            return vals

    def max(self, states, actions=None, absorbing=None, double_mask=None):
        if not np.shape(states)[1] == self._state_dim:
            raise AttributeError("Wrong dimensions of the input matrices")
        n_states = np.shape(states)[0]
        if self.double_Q:
            split_vals = np.empty((n_states,self._n_actions,2))
            for a in range(self._n_actions):
                split_vals[:,a,0], split_vals[:,a,1] = self._action_values(states, self._actions[a])
            if absorbing is not None:
                split_vals[absorbing == 1] = 0
            if double_mask is None:
                vals = get_join_vals(split_vals, self.join_type, self.tau)
            else:
                vals = get_crossed_vals(split_vals, double_mask)

        else:
            vals = np.empty((n_states,self._n_actions))
            for a in range(self._n_actions):
                vals[:,a] = self._action_values(states, self._actions[a])
            if absorbing is not None:
                vals[absorbing == 1, :] = 0

        max_actions = np.argmax(vals,1)
        idx = np.ogrid[:n_states]
        max_vals = vals[idx,max_actions]
        max_actions = [self._actions[a] for a in list(max_actions)]

        return max_vals, max_actions

    def fit(self, sa, q, return_loss_per_sample=False, **fit_params):
        regressor1_loss = {}
        regressor2_loss = {}
        regressor_loss = {}
        if self.double_Q:
            double_mask = fit_params['double_mask']
            del (fit_params['double_mask'])
        params = dict(fit_params)
        for a in self._actions:
            if self.double_Q:
                mask1 = (sa[:, -1] == a) & (double_mask)
                mask2 = (sa[:, -1] == a) & (double_mask == 0)
                params2 = params.copy()
                if "sample_weight" in fit_params:
                    w = fit_params["sample_weight"]
                    w1 = w[mask1]
                    w2 = w[mask2]
                    params["sample_weight"] = w1
                    params2["sample_weight"] = w2
                self._regressors1[a].fit(sa[mask1, 0:-1], q[mask1], **params)
                self._regressors2[a].fit(sa[mask2, 0:-1], q[mask2], **params2)
                if return_loss_per_sample:
                    regressor1_loss[a] = (self._regressors1[a].predict(sa[mask1, 0:-1]) - q[mask1]) ** 2
                    regressor2_loss[a] = (self._regressors2[a].predict(sa[mask2, 0:-1]) - q[mask2]) ** 2
            else:
                mask = sa[:, -1] == a
                if "sample_weight" in fit_params:
                    w = fit_params["sample_weight"]
                    w = w[mask]
                    params["sample_weight"] = w
                self._regressors[a].fit(sa[mask, 0:-1], q[mask], **params)
                if return_loss_per_sample:
                    regressor_loss[a] = (self._regressors[a].predict(sa[mask, 0:-1]) - q[mask]) ** 2

        if return_loss_per_sample:
            if self.double_Q:
                return regressor1_loss, regressor2_loss
            else:
                return regressor_loss, None

    def set_regressor_params(self, **params):
        for a in self._actions:
            if self.double_Q:
                self._regressors1[a].set_params(**params)
                self._regressors2[a].set_params(**params)
            else:
                self._regressors[a].set_params(**params)


class DiscreteFittedQSaveMemory(QFunction):
    """
    A DiscreteFittedQ is a Q-function represented by a set of underlying regressors, one for each discrete action.
    This is only for discrete action-spaces.
    """

    def __init__(self, regressor_type, state_dim, actions, double_Q=False, swap_Q=False, tau=None, **regressor_params):

        self._n_actions = len(actions)
        self._state_dim = state_dim
        self._actions = actions
        self.double_Q = double_Q
        self.swap_Q = swap_Q
        self.tau = tau
        self.dim_remove = 2

        if self.double_Q:
            self._regressors1 = {}
            self._regressors2 = {}
        else:
            self._regressors = {}

        for a in actions:
            if self.double_Q:
                self._regressors1[a] = regressor_type(**regressor_params)
                self._regressors2[a] = regressor_type(**regressor_params)
            else:
                self._regressors[a] = regressor_type(**regressor_params)

    def __call__(self, state, action):
        if not np.shape(state)[0] == self._state_dim:
            raise AttributeError("State is not of the right shape")
        if not action in self._actions:
            raise AttributeError("Action does not exist")
        return self._action_values(state[np.newaxis,:], action)

    def _action_values(self, states, action):
        if self.double_Q:
            return self._regressors1[action].predict(states), self._regressors2[action].predict(states)
        else:
            return self._regressors[action].predict(states)

    def _action_values_stds(self, states, action):
        if self.double_Q:
            preds1 = [self._regressors1[action].estimators_[j].predict(states) for j in range(self._regressors1[action].n_estimators)]
            preds2 = [self._regressors2[action].estimators_[j].predict(states) for j in range(self._regressors2[action].n_estimators)]
            return np.mean(preds1,axis=0), np.mean(preds2,axis=0), np.std(preds1, axis=0, ddof=1), np.std(preds2, axis=0, ddof=1)
        else:
            preds = [self._regressors[action].estimators_[j].predict(states) for j in range(self._regressors[action].n_estimators)]
            return np.mean(preds,axis=0), np.std(preds, axis=0, ddof=1)

    def values(self, sa, get_std=False):
        if not np.shape(sa)[1] == self._state_dim+1:
            print(sa.shape) #############################
            raise AttributeError("An Nx(S+1) matrix must be provided")

        # get rows with portfolio and action already given and remove portfolio and action for others
        idx_eval = np.isnan(sa[:,-1])
        n_eval = sum(idx_eval)
        n_non_eval = len(sa)-n_eval

        # Add portfolio and action row for which portfolio and action are not given
        s_eval = sa[n_non_eval:,:-self.dim_remove]
        sa = sa[:n_non_eval]
        for p in self._actions:
            portfolio = np.full((n_eval, 1), p)
            for a in self._actions:
                actions = np.full((n_eval, 1), a)
                sa = np.vstack((sa, np.hstack((s_eval, portfolio, actions))))
        if self.double_Q:
            split_vals = np.zeros((np.shape(sa)[0],2))
            vals = np.zeros(np.shape(sa)[0])
            if get_std:
                stds = np.zeros((np.shape(sa)[0],2))
        else:
            vals = np.zeros(np.shape(sa)[0])
            if get_std:
                stds = np.zeros(np.shape(sa)[0])
        check_mask = np.zeros(np.shape(sa)[0])

        for a in self._actions:
            mask = sa[:,-1] == a
            check_mask = np.logical_or(mask,check_mask)

            if self.double_Q:
                if get_std:
                    split_vals[mask,0], split_vals[mask,1], stds[mask,0], stds[mask,1] = self._action_values_stds(sa[mask, 0:-1], a)
                else:
                    split_vals[mask,0], split_vals[mask,1] = self._action_values(sa[mask, 0:-1], a)
                vals = get_join_vals(split_vals, self.tau)
            else:
                if get_std:
                    vals[mask], stds[mask] = self._action_values_stds(sa[mask, 0:-1], a)
                else:
                    vals[mask] = self._action_values(sa[mask, 0:-1], a)

        if not np.all(check_mask):
            raise AttributeError("Some action in sa does not exist")

        if get_std:
            return vals, stds
            #TODO: compute std ascombination of the two predictors.
        else:
            return vals

    def max(self, states, actions=None, absorbing=None, double_mask=None):
        if not np.shape(states)[1] == self._state_dim:
            raise AttributeError("Wrong dimensions of the input matrices")
        # get rows with portfolio and action already given and remove portfolio and action for others
        idx_eval = np.isnan(states[:,-1])
        n_eval = sum(idx_eval)
        n_non_eval = len(states)-n_eval

        # Add portfolio and action row for which portfolio and action are not given
        s_eval = states[n_non_eval:,:-self.dim_remove+1]
        states = states[:n_non_eval]
        if self.double_Q and double_mask is not None:
            double_mask_eval = double_mask[n_non_eval:]
            double_mask = double_mask[:n_non_eval]

        for _ in self._actions:
            for p in self._actions:
                portfolio = np.full((n_eval, 1), p)
                states = np.vstack((states, np.hstack((s_eval, portfolio))))
                if double_mask is not None:
                    double_mask = np.hstack((double_mask, double_mask_eval))
        n_states = states.shape[0]
        if self.double_Q:
            split_vals = np.empty((n_states, self._n_actions,2))
            for a in range(self._n_actions):
                split_vals[:,a,0], split_vals[:,a,1] = self._action_values(states, self._actions[a])
        else:
            vals = np.empty((n_states, self._n_actions))
            for a in range(self._n_actions):
                vals[:,a] = self._action_values(states, self._actions[a])

        if absorbing is not None:
            absorbing = np.hstack([absorbing[:n_non_eval]]+ [absorbing[n_non_eval:]]*self._n_actions**2)
            if self.double_Q:
                split_vals[absorbing == 1, :] = 0
            else:
                vals[absorbing == 1, :] = 0
        if self.double_Q:
            if double_mask is None:
                vals = get_join_vals(split_vals, self.tau)
            else:
                vals = get_crossed_vals(split_vals, double_mask)

        max_actions = np.argmax(vals,1)
        idx = np.ogrid[:n_states]
        max_vals = vals[idx,max_actions]
        max_actions = [self._actions[a] for a in list(max_actions)]

        return max_vals, max_actions

    def fit(self, sa, q, **fit_params):
        # get rows with portfolio and action already given and remove portfolio and action for others
        idx_eval = np.isnan(sa[:,-1])
        n_eval = sum(idx_eval)
        n_non_eval = len(sa)-n_eval

        if self.double_Q:
            double_mask = fit_params['double_mask']
            del(fit_params['double_mask'])
        params = dict(fit_params)

        # Add portfolio and action row for which portfolio and action are not given
        s_eval = sa[n_non_eval:,:-self.dim_remove]
        q_eval = q[n_non_eval:]

        sa = sa[:n_non_eval]
        if self.double_Q:
            double_mask_eval = double_mask[n_non_eval:]
            double_mask = double_mask[:n_non_eval]

        q = q[:n_non_eval]
        for p_i, p in enumerate(self._actions):
            portfolio = np.full((n_eval, 1), p)
            for a_i, a in enumerate(self._actions):
                idx = np.arange(n_eval*(p_i*self._n_actions+a_i),n_eval*(p_i*self._n_actions+(a_i+1)))
                actions = np.full((n_eval, 1), a)
                sa = np.vstack((sa, np.hstack((s_eval, portfolio, actions))))
                if self.double_Q:
                    double_mask = np.hstack((double_mask, double_mask_eval))
                q = np.concatenate((q, q_eval[idx]))

        for a in self._actions:
            if self.double_Q:
                mask1 = (sa[:,-1] == a) & double_mask
                mask2 = (sa[:,-1] == a) & (double_mask == 0)
                params2 = params.copy()
                if "sample_weight" in fit_params:
                    w = fit_params["sample_weight"]
                    w1 = w[mask1]
                    w2 = w[mask2]
                    params["sample_weight"] = w1
                    params2["sample_weight"] = w2
                self._regressors1[a].fit(sa[mask1, 0:-1], q[mask1], **params)
                self._regressors2[a].fit(sa[mask2, 0:-1], q[mask2], **params2)
            else:
                mask = sa[:,-1] == a
                if "sample_weight" in fit_params:
                    w = fit_params["sample_weight"]
                    w = w[mask]
                    params["sample_weight"] = w
                self._regressors[a].fit(sa[mask, 0:-1], q[mask], **params)

    def set_regressor_params(self, **params):
        for a in self._actions:
            if self.double_Q:
                self._regressors1[a].set_params(**params)
                self._regressors2[a].set_params(**params)
            else:
                self._regressors[a].set_params(**params)


class DiscreteBoostedFittedQSaveMemory(QFunction):
    """
    A DiscreteFittedQ is a Q-function represented by a set of underlying regressors, one for each discrete action.
    This is only for discrete action-spaces.
    """

    def __init__(self, regressor_type, state_dim, actions, double_Q=False, tau=None, swap_Q=False, **regressor_params):
        self._n_actions = len(actions)
        self._state_dim = state_dim
        self._actions = actions
        self.tau = tau
        self.double_Q = double_Q
        self.swap_Q = swap_Q
        self.dim_remove = 2

        if self.double_Q:
            self._regressors1 = {}
            self._regressors2 = {}
        else:
            self._regressors = {}

        for a in actions:
            if self.double_Q:
                self._regressors1[a] = regressor_type(**regressor_params)
                self._regressors2[a] = regressor_type(**regressor_params)
            else:
                self._regressors[a] = regressor_type(**regressor_params)

    def __call__(self, state, action):
        if not np.shape(state)[0] == self._state_dim:
            raise AttributeError("State is not of the right shape")
        if not action in self._actions:
            raise AttributeError("Action does not exist")
        return self._action_values(state[np.newaxis,:], action)

    def _action_values(self, states, action):
        if self.double_Q:
            return self._regressors1[action].predict(states), self._regressors2[action].predict(states)
        else:
            return self._regressors[action].predict(states)

    def _action_values_stds(self, states, action):
        if self.double_Q:
            preds1 = [self._regressors1[action].estimators_[j].predict(states) for j in range(self._regressors1[action].n_estimators)]
            preds2 = [self._regressors2[action].estimators_[j].predict(states) for j in range(self._regressors2[action].n_estimators)]
            return np.mean(preds1,axis=0), np.mean(preds2,axis=0), np.std(preds1, axis=0, ddof=1), np.std(preds2, axis=0, ddof=1)
        else:
            preds = [self._regressors[action].estimators_[j].predict(states) for j in range(self._regressors[action].n_estimators)]
            return np.mean(preds,axis=0), np.std(preds, axis=0, ddof=1)

    def values(self, sa, get_std=False):
        if not np.shape(sa)[1] == self._state_dim+1:
            print(sa.shape) #############################
            raise AttributeError("An Nx(S+1) matrix must be provided")

        # get rows with portfolio and action already given and remove portfolio and action for others
        idx_eval = np.isnan(sa[:,-1])
        n_eval = sum(idx_eval)
        n_non_eval = len(sa)-n_eval

        # Add portfolio and action row for which portfolio and action are not given
        s_eval = sa[n_non_eval:,:-self.dim_remove]
        sa = sa[:n_non_eval]
        for p in self._actions:
            portfolio = np.full((n_eval, 1), p)
            for a in self._actions:
                actions = np.full((n_eval, 1), a)
                sa = np.vstack((sa, np.hstack((s_eval, portfolio, actions))))
        if self.double_Q:
            split_vals = np.zeros((np.shape(sa)[0],2))
            vals = np.zeros(np.shape(sa)[0])
            if get_std:
                stds = np.zeros((np.shape(sa)[0],2))
        else:
            vals = np.zeros(np.shape(sa)[0])
            if get_std:
                stds = np.zeros(np.shape(sa)[0])
        check_mask = np.zeros(np.shape(sa)[0])

        for a in self._actions:
            mask = sa[:,-1] == a
            check_mask = np.logical_or(mask,check_mask)

            if self.double_Q:
                if get_std:
                    split_vals[mask,0], split_vals[mask,1], stds[mask,0], stds[mask,1] = self._action_values_stds(sa[mask, 0:-1], a)
                else:
                    split_vals[mask,0], split_vals[mask,1] = self._action_values(sa[mask, 0:-1], a)
                vals = get_join_vals(split_vals, self.tau)
            else:
                if get_std:
                    vals[mask], stds[mask] = self._action_values_stds(sa[mask, 0:-1], a)
                else:
                    vals[mask] = self._action_values(sa[mask, 0:-1], a)

        if not np.all(check_mask):
            raise AttributeError("Some action in sa does not exist")

        if get_std:
            return vals, stds
            #TODO: compute std ascombination of the two predictors.
        else:
            return vals

    def max(self, states, actions=None, absorbing=None, double_mask=None):
        if not np.shape(states)[1] == self._state_dim:
            raise AttributeError("Wrong dimensions of the input matrices")
        # get rows with portfolio and action already given and remove portfolio and action for others
        idx_eval = np.isnan(states[:,-1])
        n_eval = sum(idx_eval)
        n_non_eval = len(states)-n_eval

        # Add portfolio and action row for which portfolio and action are not given
        s_eval = states[n_non_eval:,:-self.dim_remove+1]
        states = states[:n_non_eval]
        if self.double_Q and double_mask is not None:
            double_mask_eval = double_mask[n_non_eval:]
            double_mask = double_mask[:n_non_eval]

        for _ in self._actions:
            for p in self._actions:
                portfolio = np.full((n_eval, 1), p)
                states = np.vstack((states, np.hstack((s_eval, portfolio))))
                if double_mask is not None:
                    double_mask = np.hstack((double_mask, double_mask_eval))
        n_states = states.shape[0]
        if self.double_Q:
            split_vals = np.empty((n_states, self._n_actions, 2))
            for a in range(self._n_actions):
                split_vals[:, a, 0], split_vals[:, a, 1] = self._action_values(states, self._actions[a])
        else:
            vals = np.empty((n_states, self._n_actions))
            for a in range(self._n_actions):
                vals[:, a] = self._action_values(states, self._actions[a])

        if absorbing is not None:
            absorbing = np.hstack([absorbing[:n_non_eval]]+ [absorbing[n_non_eval:]]*self._n_actions**2)
            if self.double_Q:
                split_vals[absorbing == 1, :] = 0
            else:
                vals[absorbing == 1, :] = 0
        if self.double_Q:
            if double_mask is None:
                vals = get_join_vals(split_vals, self.tau)
            else:
                vals = get_crossed_vals(split_vals, double_mask)

        max_actions = np.argmax(vals,1)
        idx = np.ogrid[:n_states]
        max_vals = vals[idx,max_actions]
        max_actions = [self._actions[a] for a in list(max_actions)]

        return max_vals, max_actions

    def fit(self, sa, q, **fit_params):
        # if self.double_Q:
            # raise NotImplementedError("Creation of mask on specific rows still to be implemented")
        # get rows with portfolio and action already given and remove portfolio and action for others
        idx_eval = np.isnan(sa[:,-1])
        n_eval = sum(idx_eval)
        n_non_eval = len(sa)-n_eval

        if self.double_Q:
            double_mask = fit_params['double_mask']
            del(fit_params['double_mask'])
        params = dict(fit_params)

        # Add portfolio and action row for which portfolio and action are not given
        s_eval = sa[n_non_eval:,:-self.dim_remove]
        q_eval = q[n_non_eval:]

        sa = sa[:n_non_eval]
        if self.double_Q:
            double_mask_eval = double_mask[n_non_eval:]
            double_mask = double_mask[:n_non_eval]

        q = q[:n_non_eval]
        for p_i, p in enumerate(self._actions):
            portfolio = np.full((n_eval, 1), p)
            for a_i, a in enumerate(self._actions):
                idx = np.arange(n_eval*(p_i*self._n_actions+a_i),n_eval*(p_i*self._n_actions+(a_i+1)))
                actions = np.full((n_eval, 1), a)
                sa = np.vstack((sa, np.hstack((s_eval, portfolio, actions))))
                if self.double_Q:
                    double_mask = np.hstack((double_mask, double_mask_eval))
                q = np.concatenate((q, q_eval[idx]))

        for a in self._actions:
            if self.double_Q:
                mask1 = (sa[:,-1] == a) & double_mask
                mask2 = (sa[:,-1] == a) & (double_mask == 0)
                params2 = params.copy()
                if "sample_weight" in fit_params:
                    w = fit_params["sample_weight"]
                    w1 = w[mask1]
                    w2 = w[mask2]
                    params["sample_weight"] = w1
                    params2["sample_weight"] = w2
                self._regressors1[a].fit(sa[mask1, 0:-1], q[mask1], **params)
                self._regressors2[a].fit(sa[mask2, 0:-1], q[mask2], **params2)
            else:
                mask = sa[:,-1] == a
                if "sample_weight" in fit_params:
                    w = fit_params["sample_weight"]
                    w = w[mask]
                    params["sample_weight"] = w
                self._regressors[a].fit(sa[mask, 0:-1], q[mask], **params)

    def set_regressor_params(self, **params):
        for a in self._actions:
            if self.double_Q:
                self._regressors1[a].set_params(**params)
                self._regressors2[a].set_params(**params)
            else:
                self._regressors[a].set_params(**params)


class DiscreteNeuralQ(QFunction):
    """
    A DiscreteNeuralQ is a Q-function represented by a set of underlying neural networks, one for each discrete action.
    They share part of their structure. This is only for discrete action-spaces.
    """

    def __init__(self, regressor_type, state_dim, actions, **regressor_params):
        self._n_actions = len(actions)
        self._state_dim = state_dim
        self._actions = actions

        self._regressors = {}

        self._regressors = regressor_type(actions=actions, **regressor_params)

    def __call__(self, state, action):
        if not np.shape(state)[0] == self._state_dim:
            raise AttributeError("State is not of the right shape")
        if not action in self._actions:
            raise AttributeError("Action does not exist")
        return self._action_values(state[np.newaxis,:], action)

    def _action_values(self, states, action):
        return self._regressors.predict(states, action)

    # def _action_values_stds(self, states, action):
    #     preds = [self._regressors[action].estimators_[j].predict(states) for j in range(self._regressors[action].n_estimators)]
    #     return np.mean(preds,axis=0), np.std(preds, axis=0, ddof=1)

    def values(self, sa, get_std=False):
        #print(sa.shape) #############################
        if not np.shape(sa)[1] == self._state_dim + 1:
            print(sa.shape) #############################
            raise AttributeError("An Nx(S+1) matrix must be provided")

        vals = np.zeros(np.shape(sa)[0])
        if get_std:
            stds = np.zeros(np.shape(sa)[0])
        check_mask = np.zeros(np.shape(sa)[0])

        for a in self._actions:
            mask = sa[:,-1] == a
            check_mask = np.logical_or(mask,check_mask)
            vals[mask] = self._action_values(sa[mask, 0:-1], a)

        if not np.all(check_mask):
            raise AttributeError("Some action in sa does not exist")

        return vals

    def max(self, states, actions=None, absorbing=None):
        if not np.shape(states)[1] == self._state_dim:
            raise AttributeError("Wrong dimensions of the input matrices")

        n_states = np.shape(states)[0]
        vals = np.empty((n_states,self._n_actions))

        for a in range(self._n_actions):
            vals[:,a] = self._action_values(states, self._actions[a])

        if absorbing is not None:
            vals[absorbing == 1, :] = 0

        max_actions = np.argmax(vals,1)
        idx = np.ogrid[:n_states]
        max_vals = vals[idx,max_actions]
        max_actions = [self._actions[a] for a in list(max_actions)]

        return max_vals, max_actions

    def fit(self, sa, q, **fit_params):
        params = dict(fit_params)
        self._regressors.fit(sa, q[:], **params)

    def set_regressor_params(self, **params):
        for a in self._actions:
            self._regressors.set_params(a, **params)


def convert_portfolio_single(x):
    return [x]


def get_join_vals(split_vals, join_type, tau=None):
    if tau is not None:
        min = np.min(split_vals, axis=-1)
        max = np.max(split_vals, axis=-1)
        vals = np.sum([tau * min, (1-tau)*max], axis=0)
    else:
        #assert q_vals_min == True
        if join_type == 'min':
            vals = np.min(split_vals, axis=-1)
            # print("min")
        elif join_type == 'mean':
            vals = np.mean(split_vals, axis=-1)
            # print("mean")
        elif join_type == 'mean+2std':
            vals = np.mean(split_vals, axis=-1) + 2 * np.std(split_vals, axis=-1)
            # print(join_type)
        elif join_type == 'max':
            vals = np.min(split_vals, axis=-1)
            # print("max")
        else:
            raise AttributeError(f"Join type {join_type} not availbale")
    return vals


def get_crossed_vals(split_vals, double_mask=None):
    vals = np.zeros(np.shape(split_vals)[:-1])
    vals[double_mask] = split_vals[double_mask, :, 1]
    vals[double_mask==0] = split_vals[double_mask==0, :, 0]
    return vals
