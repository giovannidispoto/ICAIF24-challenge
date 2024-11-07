import os

from agent.fqi import AgentFQI
from agent.online_rl import AgentOnlineRl
from agent.baselines import ShortOnlyBaseline, LongOnlyBaseline, RandomBaseline

PROJECT_FOLDER = "./"
DATA_DIR = os.path.join(PROJECT_FOLDER, "data")
AGENTS_FOLDER = os.path.join(PROJECT_FOLDER, "agents")
EPISODES_FQI = 300
os.makedirs(AGENTS_FOLDER, exist_ok=True)


class AgentsFactory:
    @staticmethod
    def load_agent(agent_info):
        agent_type = agent_info['type']
        agent_file = os.path.join(AGENTS_FOLDER, agent_info['file'])
        if agent_type == 'fqi':
            agent = AgentFQI()
            agent.load(agent_file)
            return agent
        elif agent_type in ['dqn', 'ppo']:
            agent = AgentOnlineRl(agent_type)
            agent.load(agent_file)
            return agent
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
            agent = AgentOnlineRl(agent_info['type'])
            try:
                agent.load(agent_info['file'])
            except:
                agent.train(env_args=env_args, model_args=agent_info['model_args'])
                agent.save(agent_info['file'])
        else:
            raise NotImplementedError