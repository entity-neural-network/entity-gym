from dataclasses import dataclass, field
import numpy as np
import random
from typing import Dict, List

from entity_gym.environment import (
    DenseSelectEntityActionMask,
    Entity,
    Environment,
    SelectEntityAction,
    Type,
    SelectEntityActionSpace,
    ActionSpace,
    ObsFilter,
    Observation,
    ActionMask,
    Action,
)


@dataclass
class CherryPick(Environment):
    """
    The CherryPick environment is initalized with a list of 32 cherries of random quality.
    On each timestep, the player can pick up one of the cherries.
    The player receives a reward of the quality of the cherry picked.
    The environment ends after 16 steps.
    The quality of the top 16 cherries is normalized so that the maximum total achievable reward is 1.0.
    """

    cherries: List[float] = field(default_factory=list)
    last_reward: float = 0.0
    step: int = 0

    @classmethod
    def state_space(cls) -> List[Entity]:
        return [
            Entity(
                name="Cherry",
                features=["quality"],
            ),
            Entity(
                name="Player",
                features=[],
            ),
        ]

    @classmethod
    def action_space(cls) -> List[ActionSpace]:
        return [SelectEntityActionSpace("Pick Cherry")]

    def _reset(self) -> Observation:
        cherries = [np.random.normal() for _ in range(32)]
        # Normalize so that the sum of the top 16 is 1.0
        top_16 = sorted(cherries, reverse=True)[:16]
        sum_top_16 = sum(top_16)
        normalized_cherries = [c / sum_top_16 for c in cherries]
        self.cherries = normalized_cherries
        self.last_reward = 0.0
        return self.observe()

    def observe(self) -> Observation:
        return Observation(
            entities=[
                ("Cherry", np.array(self.cherries).reshape(-1, 1)),
                ("Player", np.zeros([1, 0])),
            ],
            ids=np.arange(len(self.cherries) + 1),
            action_masks=[
                (
                    "Pick Cherry",
                    DenseSelectEntityActionMask(
                        actors=[len(self.cherries)],
                        mask=(
                            np.arange(len(self.cherries) + 1) < len(self.cherries)
                        ).astype(np.float32),
                    ),
                )
            ],
            reward=self.last_reward,
            done=self.step == 16,
        )

    def _act(self, action: Dict[str, Action]) -> Observation:
        for action_name, a in action.items():
            assert isinstance(a, SelectEntityAction)
            assert action_name == "Pick Cherry"
            self.last_reward = self.cherries.pop(a.actions[0][1])
        self.step += 1
        return self.observe()
