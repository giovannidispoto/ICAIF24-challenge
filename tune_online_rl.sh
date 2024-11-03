#!/bin/sh
start_day=7
n_train_days=1
n_windows=8 #8
n_seeds=5
n_trials=100
agent="A2C" #PPO, DQN, A2C

for ((i=0; i<n_windows; i++)); do
    python3 tune_online_rl.py --start_day_train $((start_day + i)) \
                                    --end_day_train $((start_day + i + n_train_days - 1)) \
                                    --n_seeds $n_seeds \
                                    --n_trials $n_trials \
                                    --agent $agent \
                                    --out_dir "experiments/tuning/${agent}_window_${i}"
done

wait  # Wait for all background processes to complete