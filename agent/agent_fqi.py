import numpy as np
import pickle


class AgentFQI:
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
