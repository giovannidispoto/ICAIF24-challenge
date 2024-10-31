import numpy as np
import pickle

from agent.base import AgentBase


class AgentFQI(AgentBase):
    def __init__(
        self,
        policy_path: str,
    ):
        self.policy = pickle.load(open(policy_path, "rb"))

    def action(
        self,
        state: np.ndarray,
    ):
        q_values = self.policy._q_values(state)
        return np.argmax(q_values).item()
