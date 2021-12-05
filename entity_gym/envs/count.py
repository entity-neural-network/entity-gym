from dataclasses import dataclass, field
import numpy as np
import random
from typing import Dict, List, Mapping, Tuple

from entity_gym.environment import (
    CategoricalAction,
    DenseCategoricalActionMask,
    Entity,
    Environment,
    CategoricalActionSpace,
    ActionSpace,
    EpisodeStats,
    ObsSpace,
    Observation,
    Action,
)
from entity_gym.dataclass_utils import obs_space_from_dataclasses, extract_features


@dataclass
class Player:
    pass


@dataclass
class Bean:
    pass


class Count(Environment):
    """
    There are between 0 and 10 "Bean" entities.
    The "Player" entity gets 1 reward for counting the correct number of beans and 0 otherwise.
    """

    @classmethod
    def obs_space(cls) -> ObsSpace:
        return obs_space_from_dataclasses(Player, Bean)

    @classmethod
    def action_space(cls) -> Dict[str, ActionSpace]:
        return {
            "count": CategoricalActionSpace(
                ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
            )
        }

    def reset(self, obs_space: ObsSpace) -> Observation:
        self.count = random.randint(0, 1)
        return self.observe(obs_space)

    def _reset(self) -> Observation:
        return self.reset(Count.obs_space())

    def act(self, action: Mapping[str, Action], obs_filter: ObsSpace) -> Observation:
        reward = 0.0
        assert len(action) == 1
        a = action["count"]
        assert len(a.actions) == 1
        assert isinstance(a, CategoricalAction)
        choice = a.actions[0][1]
        if choice == self.count:
            reward = 1.0
        return self.observe(obs_filter, done=True, reward=reward)

    def _act(self, action: Mapping[str, Action]) -> Observation:
        return self.act(
            action,
            Count.obs_space(),
        )

    def observe(
        self, obs_filter: ObsSpace, done: bool = False, reward: float = 0.0
    ) -> Observation:
        return Observation(
            entities=extract_features(
                {
                    "Player": [Player()],
                    "Bean": [Bean()] * self.count,
                },
                obs_filter,
            ),
            action_masks={
                "count": DenseCategoricalActionMask(actors=np.array([0]), mask=None),
            },
            ids=list(range(1 + self.count)),
            reward=reward,
            done=done,
            end_of_episode_info=EpisodeStats(1, reward) if done else None,
        )
