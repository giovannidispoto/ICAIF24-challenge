#!/bin/sh
start_day=7
n_train_days=1
n_validation_days=1
n_windows=8
n_seeds=5
n_trials=50

for ((i=6; i<n_windows; i++)); do
    python3 task1_tune_online_rl.py --start_day_train $((start_day + i)) \
                                    --end_day_train $((start_day + i + n_train_days - 1)) \
                                    --n_seeds $n_seeds \
                                    --n_trials $n_trials \
                                    --out_dir "experiments/2/window_${i}_step_gap_2"
done

wait  # Wait for all background processes to complete