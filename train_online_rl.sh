#!/bin/sh
window=3
agent="PPO"
n_seeds=3

python train_online_rl.py --window "$window" --agent "$agent" --n_seeds "$n_seeds"


#!/bin/sh
agent="PPO"
n_seeds=1
for window in $(seq 0 7)
do
  python train_online_rl.py --window "$window" --agent "$agent" --n_seeds "$n_seeds"
done