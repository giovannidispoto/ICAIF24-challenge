start_day=7
n_train_days=1
n_validation_days=1
n_windows=1

for ((i=0;i<n_windows;i+=1))
  do
      python task1_tune_online_rl.py --start_day_train $((start_day+i))  --end_day_train $((start_day+i+n_train_days-1)) --out_dir  experiments/trial_"$i"_window_stap_gap_2 &
done
# i=0

# python task1_tune_online_rl.py --start_day_train $((start_day+i))  --end_day_train $((start_day+i+n_train_days-1)) --out_dir  experiments/trial_"$i"_window_stap_gap_2