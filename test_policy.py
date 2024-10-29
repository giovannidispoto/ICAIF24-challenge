import pickle
from erl_config import build_env
from trade_simulator import TradeSimulator
model_path = "logs_tune/trial_0_window/Policy_iter9.pkl"
start_day_train = 7
end_day_train = 7
n_validation_days = 1
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
        "step_gap": 1,
        "env_class": TradeSimulator,
        "eval": True,
        "days": [end_day_train + 1, end_day_train + n_validation_days]
    }
eval_env = build_env(TradeSimulator, env_args, -1)

policy = pickle.load(open(model_path, "rb"))
reward = 0
s, _ = eval_env.reset()
for st in range(max_steps):
    a = policy.sample_action(s)
    sp, r, done, truncated, _ = eval_env.step(a)
    reward = reward + r
    s = sp
    if done or truncated:
        break

print(f"Return:{reward}; steps:{st}")