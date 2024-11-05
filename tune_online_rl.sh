#!/bin/sh
start_day=7
n_train_days=1
n_windows=8 #8
n_seeds=5
n_trials=50
agent="PPO" #PPO, DQN, A2C

for ((i=5; i<n_windows; i++)); do
    python3 tune_online_rl.py --start_day_train $((start_day + i)) \
                                    --end_day_train $((start_day + i + n_train_days - 1)) \
                                    --start_day_val $((start_day + i + n_train_days)) \
                                    --end_day_val $((start_day + i + n_train_days)) \
                                    --n_seeds $n_seeds \
                                    --n_trials $n_trials \
                                    --agent $agent \
                                    --out_dir "experiments/tuning/${agent}_window_${i}" &
done

wait  # Wait for all background processes to complete



#!/bin/sh
start_day_train=9
end_day_train=15
start_day_val=16
end_day_val=16
n_seeds=5
n_trials=50
agent="PPO" #PPO, DQN, A2C

python3 tune_online_rl.py --start_day_train $start_day_train \
                                --end_day_train $end_day_train \
                                --start_day_val $start_day_val \
                                --end_day_val $end_day_val \
                                --n_seeds $n_seeds \
                                --n_trials $n_trials \
                                --agent $agent \
                                --progress \
                                --out_dir "experiments/tuning/${agent}_${start_day_train}-${end_day_train}_${start_day_val}-${end_day_val}"