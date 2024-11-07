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
            agent = AgentFQI()
            try:
                agent.load(agent_info['file'])
            except:
                agent.train(env_args=env_args, args=agent_info['model_args'])
                agent.save(agent_info['file'])

        elif agent_info['type'] in ['dqn', 'ppo']:
            agent_class = ONLINE_RL_NAME_TO_CLASS_DICT[agent_info['type']]
            agent = AgentOnlineRl(agent_class)
            try:
                agent.load(agent_info['file'])
            except:
                agent.train(env_args=env_args, model_args=agent_info['model_args'])
                agent.save(agent_info['file'])

        else:
            raise NotImplementedError