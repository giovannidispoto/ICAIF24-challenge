import os
from agent.fqi import AgentFQI
PROJECT_FOLDER = "./"

AGENTS_FOLDER = os.path.join(PROJECT_FOLDER, "agents")
os.makedirs(AGENTS_FOLDER, exist_ok=True)


class AgentsFactory:
    @staticmethod
    def load_agent(agent_info):
        if agent_info['type'] == 'fqi':
            policy_path = os.path.join(AGENTS_FOLDER, agent_info['file'])
            return AgentFQI(policy_path)
        else:
            raise NotImplementedError