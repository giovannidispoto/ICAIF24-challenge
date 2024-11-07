#!/bin/sh
start_train_day=9
end_train_day=9
agent="PPO"
n_seeds=1
python train_online_rl.py \
  --start_train_day $start_train_day \
  --end_train_day $end_train_day \
  --agent $agent \
  --n_seeds $n_seeds \
  --progress

#!/bin/sh
start_train_day=14
end_train_day=14
agent="PPO"
n_seeds=3
python train_online_rl.py \
  --start_train_day $start_train_day \
  --end_train_day $end_train_day \
  --agent $agent \
  --n_seeds $n_seeds \
  --progress \
  --force_default


#!/bin/sh
agent="PPO"
n_seeds=3
start_day=7
for window in $(seq 2 8)
do
  python train_online_rl.py \
    --start_train_day $((window+start_day)) \
    --end_train_day $((window+start_day)) \
    --agent $agent \
    --n_seeds $n_seeds \
    --force_default \
    --progress
done

wait