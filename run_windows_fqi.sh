
start_day=7
n_train_days=1
n_validation_days=1
n_windows=8

for ((i=0;i<n_windows;i+=1))
  do
      python3 train_fqi.py --start_day_train $((start_day+i))  --end_day_train $((start_day+i+n_train_days-1)) --num_days_validation $n_validation_days --out_dir  experiments/trial_"$i"_window_stap_gap_2 &
done
