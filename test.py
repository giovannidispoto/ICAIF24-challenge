
from erl_config import build_env
from trade_simulator import TradeSimulator, EvalTradeSimulator
import torch as th
env_args = {
        "env_name": "TradeSimulator-v0",
        "num_envs": 1,
        "max_step": 1000,
        "state_dim": 8 + 2,  # factor_dim + (position, holding)
        "action_dim": 3,  # long, 0, short
        "if_discrete": True,
        "max_position": 1,
        "slippage": 7e-7,
        "num_sims": 1,
        "step_gap": 1,
        "env_class":TradeSimulator
    }

env = build_env(TradeSimulator, env_args, -1)
device = th.device("cpu")
for i in range(10):
    env.reset()
    print("Episode " + str(i))
    for j in range(100):
        a = int(input())
        a = th.zeros((1, 1), dtype=th.float32, device=device) + a
        s, r, done, info = env.step(a)
        print(s)
        print(r)
        print(done)
        if done:
            break
