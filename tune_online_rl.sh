#!/bin/sh
start_day=7
n_train_days=1
n_validation_days=1
n_windows=1
n_seeds=10
n_trials=50

i=0
while [ "$i" -lt "$n_windows" ]
do
    python task1_tune_online_rl.py --start_day_train $(($start_day+i)) \
                                   --end_day_train $(($start_day+i+n_train_days-1)) \
                                   --n_seeds $n_seeds \
                                   --n_trials $n_trials \
                                   --out_dir "experiments/trial_${i}_window_stap_gap_2" &
    i=$((i+1))
done
wait


# i=0

# python task1_tune_online_rl.py --start_day_train $(($start_day+i)) \
#                                 --end_day_train $(($start_day+i+n_train_days-1)) \
#                                 --n_seeds $n_seeds \
#                                 --n_trials $n_trials \
#                                 --out_dir "experiments/trial_${i}_window_stap_gap_2"