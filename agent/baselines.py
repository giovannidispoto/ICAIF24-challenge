import numpy as np
from agent.base import AgentBase

class LongOnlyBaseline(AgentBase):
    def __init__(
        self,
    ):
        pass
    def action(
        self, state
    ):
        return np.ones(shape=(state.shape[0], )) * 2

class ShortOnlyBaseline(AgentBase):
    def __init__(
        self,
    ):
        pass
    def action(
        self,
        state
    ):
        return np.zeros(shape=(state.shape[0], ))

class FlatOnlyBaseline(AgentBase):
    def __init__(
        self,
    ):
        pass
    def action(
        self,
        state
    ):
        return np.ones(shape=(state.shape[0], ))

class RandomBaseline(AgentBase):
    def __init__(
        self,
    ):
        pass
    def action(
        self,
        state
    ):
        return np.random.choice(3, size=state.shape[0])