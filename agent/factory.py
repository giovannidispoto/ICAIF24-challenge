import os

import numpy as np

from agent.fqi import AgentFQI
from agent.online_rl import AgentOnlineRl
from agent.baselines import ShortOnlyBaseline, LongOnlyBaseline, RandomBaseline
from erl_config import build_env
from sample_online_rl import ONLINE_RL_NAME_TO_CLASS_DICT
from trade_simulator import TradeSimulator

PROJECT_FOLDER = "./"
DATA_DIR = os.path.join(PROJECT_FOLDER, "data")
AGENTS_FOLDER = os.path.join(PROJECT_FOLDER, "agents")
EPISODES_FQI = 300
os.makedirs(AGENTS_FOLDER, exist_ok=True)


class AgentsFactory:
    @staticmethod
    def load_agent(agent_info):
        if agent_info['type'] == 'fqi':
            policy_path = os.path.join(AGENTS_FOLDER, agent_info['file'])
            return AgentFQI(policy_path)
        elif agent_info['type'] in ['dqn', 'ppo']:
            agent_class = ONLINE_RL_NAME_TO_CLASS_DICT[agent_info['type']]
            model_path = os.path.join(AGENTS_FOLDER, agent_info['file'])
            return AgentOnlineRl(agent_class, model_path)
        elif agent_info['type'] in ['lo']:
            return LongOnlyBaseline()
        elif agent_info['type'] in ['sho']:
            return ShortOnlyBaseline()
        elif agent_info['type'] in ['random']:
            return RandomBaseline()
        else:
            raise NotImplementedError


    @staticmethod
    def train(agent_info, env_args):
        if agent_info['type'] == 'fqi':
            if "fqi_"+agent_info['file']+".pkl" in os.listdir(os.path.join(AGENTS_FOLDER)):
                print("Loading...")
                policy_path = os.path.join(AGENTS_FOLDER, "fqi_"+agent_info['file']+".pkl")
                return AgentFQI(policy_path)
            else: #train the agent
                eval_env = build_env(TradeSimulator, env_args, -1)
                agent = AgentFQI()
                state_actions, rewards, next_states, absorbing, policies_unread = agent.read_dataset(env_args["days"],
                                                                                                     data_dir=DATA_DIR)
                if len(policies_unread) > 0:
                    for policy in policies_unread:
                        sa, r, ns, a = agent.generate_experience(days_to_sample=env_args["day"],
                                                                 env_args=env_args,
                                                                 episodes=EPISODES_FQI,
                                                                 policy=policy,
                                                                 data_dir=DATA_DIR)
                        if len(state_actions) > 0:
                            state_actions = np.concatenate([state_actions, sa], axis=0)
                            rewards = np.concatenate([rewards, r], axis=0)
                            next_states = np.concatenate([next_states, ns], axis=0)
                            absorbing = np.concatenate([absorbing, a], axis=0)
                        else:
                            state_actions = sa
                            rewards = r
                            next_states = ns
                            absorbing = a

                agent.train(state_actions=state_actions, rewards=rewards, next_states=next_states, absorbing=absorbing,
                            env=eval_env, args=agent_info['model_args'])

                agent.save(AGENTS_FOLDER, name=agent_info['file'])

                return agent

        elif agent_info['type'] in ['dqn', 'ppo']:
            agent_class = ONLINE_RL_NAME_TO_CLASS_DICT[agent_info['type']]
            if agent_info['type']+"_"+agent_info['file']+".pkl" in os.listdir(os.path.join(AGENTS_FOLDER)):
                print("Loading... ", agent_info['type'])
                model_path = os.path.join(AGENTS_FOLDER, agent_info['type']+"_"+agent_info['file']+".pkl")
                return AgentOnlineRl(agent_class, model_path)
            else: #train the agent
                model_path = os.path.join(AGENTS_FOLDER, agent_info['type']+"_"+agent_info['file'] + ".pkl")
                agent = AgentOnlineRl(agent_class)
                agent = agent.train(env_args=env_args, model_args=agent_info['model_args'], save_path=model_path )

                return agent
        else:
            raise NotImplementedError