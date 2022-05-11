from dataclasses import dataclass, field
from typing import Dict, List, Mapping

import numpy as np

from entity_gym.env import (
    Action,
    ActionSpace,
    Entity,
    EntityID,
    Environment,
    Observation,
    ObsSpace,
    SelectEntityAction,
    SelectEntityActionMask,
    SelectEntityActionSpace,
)
from entity_gym.env.environment import EntityName


@dataclass
class CherryPick(Environment):
    """
    The CherryPick environment is initialized with a list of 32 cherries of random quality.
    On each timestep, the player can pick up one of the cherries.
    The player receives a reward of the quality of the cherry picked.
    The environment ends after 16 steps.
    The quality of the top 16 cherries is normalized so that the maximum total achievable reward is 1.0.
    """

    num_cherries: int = 32
    cherries: List[float] = field(default_factory=list)
    last_reward: float = 0.0
    step: int = 0

    def obs_space(self) -> ObsSpace:
        return ObsSpace(
            entities={
                "Cherry": Entity(["quality"]),
                "Player": Entity([]),
            }
        )

    def action_space(self) -> Dict[str, ActionSpace]:
        return {"Pick Cherry": SelectEntityActionSpace()}

    def reset(self) -> Observation:
        cherries = [np.random.normal() for _ in range(self.num_cherries)]
        # Normalize so that the sum of the top half is 1.0
        top_half = sorted(cherries, reverse=True)[: self.num_cherries // 2]
        sum_top_half = sum(top_half)
        add = 2.0 * (1.0 - sum_top_half) / self.num_cherries
        self.cherries = [c + add for c in cherries]
        self.last_reward = 0.0
        self.step = 0
        self.total_reward = 0.0
        return self.observe()

    def observe(self) -> Observation:
        done = self.step == self.num_cherries // 2
        ids: Dict[EntityName, List[EntityID]] = {
            "Cherry": [("Cherry", a) for a in range(len(self.cherries))],
            "Player": ["Player"],
        }
        return Observation(
            features={
                "Cherry": np.array(self.cherries, dtype=np.float32).reshape(-1, 1),
                "Player": np.zeros([1, 0], dtype=np.float32),
            },
            ids=ids,
            actions={
                "Pick Cherry": SelectEntityActionMask(
                    actor_ids=["Player"],
                    actee_ids=ids["Cherry"],
                ),
            },
            reward=self.last_reward,
            done=done,
        )

    def act(self, actions: Mapping[str, Action]) -> Observation:
        assert len(actions) == 1, actions
        a = actions["Pick Cherry"]
        assert isinstance(a, SelectEntityAction)
        _, chosen_cherry_idx = a.actees[0]
        self.last_reward = self.cherries.pop(chosen_cherry_idx)
        self.total_reward += self.last_reward
        self.step += 1
        return self.observe()
