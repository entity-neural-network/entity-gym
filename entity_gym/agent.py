from abc import ABC, abstractmethod
from typing import Dict, Tuple

from entity_gym.environment.environment import Action, Observation


class Agent(ABC):
    @abstractmethod
    def act(self, obs: Observation) -> Tuple[Dict[str, Action], float]:
        pass
