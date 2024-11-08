# RL3 FinRL Task 1 Submission
In this repository, we present the submission of the RL3 team to the FinRL2024 Competition.

We propose a two-layer trading architecture, where an Online Adaptation method based on OAMP [1] works as ensemble method on top of a set of RL agents. The OAMP algorithm allows performing an online, performance-driven, validation of the agents. In this way it is possible to overcome the limitations of traditional model selection procedures which are usually based on backtesting model on historical data.

To train the experts of the ensemble, we split the training dataset into 10 subsets, where each subset corresponds to a calendar day. The first two days are kept for validating the OAMP algorithm. The next 7 days are used as training subsets.
For each training subset, we train 3 different RL agents on the whole day, one DQN [2] agent, one PPO [3] agent and one FQI [4] agent. For DQN and PPO we employ the implementations found in stable_baselines3 [5], whereas for FQI we employ our own implementation. We evaluate each algorithm on the next day to perform hyperparameter tuning. In total, we train 21 agents, 3 for each day from 9th to 15th April. Out of these 21 agents, we choose 7 final agents to be employed in the ensemble, choosing the best out of DQN, PPO or FQI based on their respective validation scores for that day. We opt to use the first two days for validating the ensemble since the last two show a market trend reversal, this way we have a diverse set of experts that have seen diverse market regimes. 

For training and evaluating DQN and PPO agents, we employ an episodic setting, where in each episode we pick a random starting state in the training day and run our agents for 480 steps of 2 seconds. Therefore, our agents optimize the expected return of trading intervals of 16 minutes. We chose these values out of some preliminary testing.

For training FQI, we employ the same episodic setting. FQI needs an offline dataset of interactions as input.
For each training day, we generate a dataset composed of 1000 episodes of 4 baselines polices:
 - Random Agent
 - Short Only Agent
 - Long Only Agent
 - Flat Only Agent


The repository is organized as follows:
```
├── data # Folder containing the input datasets to trade_simulator
├── agents # Folder containing the pretrained models used in the Ensemble
│ ├── agents_args.json # json file containing the args to train each agent
│ ├── agents_best.json # json file containing the best agent for each training day
│ ├── agents_policies files
├── agent # Folder containing the agent classes, used to train and load the agents
│ ├── base.py # Base agent interface
│ ├── baselines.py # Baseline agents used to generate FQI datasets
│ ├── factory.py # Agent factory used for loading and utilities
│ ├── fqi.py # Implementation of the FQI agents
│ ├── online_rl.py # Implementation of the DQN and PPO agents
├── experiments # Folder containing the experiments results
├── oamp # Folder containing our implementation of the OAMP method 
│ ├── oamp.py # file containing our implementation of the OAMP method 
│ ├── oamp_config.py # file containing oamp args
│ ├── oamp_utils.py # file containing oamp utils fn
├── trlib # Folder containing our implementation of the FQI algorithm 
├── data_config.py # configuration file for market data path
├── trade_simulator.py # A slightly modified version of the simulator needed for OAMP
├── task_1_ensemble.py # Script for training, model selection and saving the Ensemble agents
├── task_1_eval.py: # Script for running the evaluation of the Ensemble with OAMP
├── requirements.txt: # Requirements needed to perform training and validation
├── README.md: # this readme file
```


To execute our ensemble method, you need to:
0. create a virtual env with python 3.10 and install the required packages (see requirements.py)
1. unzip the agents folder which includes:
    - the files of the policies of the 21 agents
    - the file agents_args.json, which in turn contains the model args to retrain the agent
    - the file agents_best.json, which in turn contains the results of the agent selection procedures
2. upload the data files (BTC_1sec.csv and BTC_1sec_predict.npy) relative to the evaluation period inside the data directory (check that the paths of these files is consistent with those inside data_config.py)
3. run task_1_eval.py

Note that:
- we updated trade_simulator.py in order to allow sequential evaluation (see row 105 in trade_simulator.py)
- we set the number of evaluation steps to the len of the market data dataframe (see row 104 in task_1_eval.py).

To rebuild the set of agents, for sake of reproducibility, you need to:
0. create a virtual env with python 3.10 and install the required packages (see requirements.py)
1. remove all the files of the policies of the 21 agents from the agents folder (otherwise it will load the model)
2. upload the data files (BTC_1sec.csv and BTC_1sec_predict.npy) relative to the training period inside the data director (check that the paths of these files is consistent with those inside data_config.py)
3. run task_1_ensemble.py.


[1] Antonio Riva, Lorenzo Bisi, Pierre Liotet, Luca Sabbioni, Edoardo Vittori, Marco Pinciroli, Michele Trapletti, and Marcello Restelli. Addressing non-stationarity in fx trading with online model selection of offline rl experts. In Proceedings of the Third ACM International Conference on AI in Finance, ICAIF ’22,

[2] Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A. Rusu, Joel Veness,Marc G. Bellemare, Alex Graves, Martin Riedmiller, Andreas K. Fidjeland, Georg Ostrovski, Stig Petersen, Charles Beattie, Amir Sadik, Ioannis Antonoglou, Helen King, Dharshan Kumaran, Daan Wierstra, Shane Legg, and Demis Hassabis.Human-level control through deep reinforcement learning. Nature,

[3] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. 

[4] Damien Ernst, Pierre Geurts, and Louis Wehenkel. Tree-based batch mode reinforcement learning. J. Mach. Learn. Res., 6

[5] Antonin Raffin, Ashley Hill, Adam Gleave, Anssi Kanervisto, Maximilian Ernestus, and Noah Dormann. Stable-baselines3: Reliable reinforcement learning implementations. Journal of Machine Learning Research,