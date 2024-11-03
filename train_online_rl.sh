# #!/bin/sh
# window=6
# agent="DQN"
# n_seeds=5

# python train_online_rl.py --window "$window" --agent "$agent" --n_seeds "$n_seeds"


#!/bin/sh
agent="PPO"
n_seeds=5
for window in $(seq 0 0)
do
  echo "Training agent $agent with window size: $window"
  python train_online_rl.py --window "$window" --agent "$agent" --n_seeds "$n_seeds"
done