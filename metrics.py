import numpy as np
import pandas as pd
import empyrical as ep
import torch

def cumulative_returns(returns_pct):
    """
    returns_pct: percentage change, i.e. (r_(t+1) - r_t)/r_t

    return a pd.Series
    """
    return ep.cum_returns(returns_pct)


def sharpe_ratio(returns_pct, risk_free=0):
    """
    returns_pct: percentage change, i.e. (r_(t+1) - r_t)/r_t

    return float
    """
    returns = np.array(returns_pct)
    if returns.std() == 0:
        sharpe_ratio = np.inf
    else:
        sharpe_ratio = (returns.mean() - risk_free) / returns.std()
    return sharpe_ratio


def sharpe_ratio_ms(returns_pct: np.ndarray, risk_free=0) -> np.ndarray:
    mean_returns = returns_pct.mean(axis=0)
    std_returns = returns_pct.std(axis=0)
    sharpe_ratios = (mean_returns - risk_free) / std_returns
    sharpe_ratios[std_returns == 0] = float('inf')
    return sharpe_ratios


def max_drawdown(returns_pct):
    """
    returns_pct: percentage change, i.e. (r_(t+1) - r_t)/r_t

    return: float
    """
    return ep.max_drawdown(returns_pct)


def return_over_max_drawdown(returns_pct):
    """
    returns_pct: percentage change, i.e. (r_(t+1) - r_t)/r_t

    return: float
    """
    mdd = abs(max_drawdown(returns_pct))
    returns = cumulative_returns(returns_pct)[len(returns_pct) - 1]
    if mdd == 0:
        return np.inf
    return returns / mdd
