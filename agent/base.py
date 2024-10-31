from abc import ABC, abstractmethod


class AgentBase(ABC):
    @abstractmethod
    def action(self):
        pass