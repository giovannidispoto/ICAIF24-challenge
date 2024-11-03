import os
from agent.dqn import AgentDQN
from agent.fqi import AgentFQI
from agent.ppo import AgentPPO
PROJECT_FOLDER = "./"

AGENTS_FOLDER = os.path.join(PROJECT_FOLDER, "agents")
os.makedirs(AGENTS_FOLDER, exist_ok=True)


class AgentsFactory:
    @staticmethod
    def load_agent(agent_info):
        if agent_info['type'] == 'fqi':
            policy_path = os.path.join(AGENTS_FOLDER, agent_info['file'])
            return AgentFQI(policy_path)
        elif agent_info['type'] in ['dqn', 'ppo']:
            model_path = os.path.join(AGENTS_FOLDER, agent_info['file'])
            if agent_info['type'] == 'dqn':
                return AgentDQN(model_path)
            elif agent_info['type'] == 'ppo':
                return AgentPPO(model_path)
        else:
            raise NotImplementedError