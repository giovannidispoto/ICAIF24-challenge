import os
from agent.fqi import AgentFQI
from agent.online_rl import AgentOnlineRl
from agent.baselines import ShortOnlyBaseline, LongOnlyBaseline, RandomBaseline
from sample_online_rl import ONLINE_RL_NAME_TO_CLASS_DICT

PROJECT_FOLDER = "./"

AGENTS_FOLDER = os.path.join(PROJECT_FOLDER, "agents")
os.makedirs(AGENTS_FOLDER, exist_ok=True)


class AgentsFactory:
    @staticmethod
    def load_agent(agent_info):
        if agent_info['type'] == 'fqi':
            policy_path = os.path.join(AGENTS_FOLDER, agent_info['file'])
            return AgentFQI(policy_path)
        elif agent_info['type'] in ['dqn', 'ppo', 'a2c']:
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