from abc import ABC, abstractmethod


class AgentBase(ABC):
    @abstractmethod
    def action(self, state):
        pass
    def __call__(self, state):
        return self.action(state)