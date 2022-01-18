from dataclasses import dataclass, field
import numpy as np
import numpy.typing as npt
import random
from typing import Dict, List, Mapping, Tuple, Optional

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

    This environment also randomly masks off some of the incorrect answers.

    Masking by default allows all actions, which is equivalent to disabling masking.
    """

    def __init__(self, masked_choices: int = 10):
        assert (
            masked_choices >= 1 and masked_choices <= 10
        ), "masked_choices must be between 1 and 10"
        self.masked_choices = masked_choices

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
        self.count = random.randint(0, self.masked_choices - 1)
        possible_counts = {
            self.count,
            *random.sample(
                range(0, self.masked_choices), random.randint(0, self.masked_choices)
            ),
        }
        mask = np.zeros((10), dtype=np.bool_)
        mask[list(possible_counts)] = True
        return self.observe(obs_space, mask)

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
        return self.observe(obs_filter, None, done=True, reward=reward)

    def _act(self, action: Mapping[str, Action]) -> Observation:
        return self.act(
            action,
            Count.obs_space(),
        )

    def observe(
        self,
        obs_filter: ObsSpace,
        mask: Optional[npt.NDArray[np.int64]],
        done: bool = False,
        reward: float = 0.0,
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
                "count": DenseCategoricalActionMask(actors=np.array([0]), mask=mask),
            },
            ids=["Player"] + [f"Bean{i}" for i in range(1, self.count + 1)],
            reward=reward,
            done=done,
            end_of_episode_info=EpisodeStats(1, reward) if done else None,
        )
