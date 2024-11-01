
start_day=11
n_windows=5

for ((i=0;i<n_windows;i+=1))
  do
      python3 tune_oamp.py --day_eval $((start_day+i))  --agent_dir ~/challenge_icaif/markov_data_2/ --out_dir  trial_"$i"_window_oamp &
done
