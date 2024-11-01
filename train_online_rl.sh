#!/bin/sh
start_day=8
end_day=8

python3 train_online_rl.py --start_day_train $start_day \
                                --end_day_train $end_day
