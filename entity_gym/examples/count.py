import random
from dataclasses import dataclass
from typing import Dict, Mapping, Optional

import numpy as np
import numpy.typing as npt

from entity_gym.dataclass_utils import extract_features, obs_space_from_dataclasses
from entity_gym.env import (
    Action,
    ActionSpace,
    CategoricalAction,
    CategoricalActionMask,
    CategoricalActionSpace,
    Environment,
    Observation,
    ObsSpace,
)


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

    def obs_space(self) -> ObsSpace:
        return obs_space_from_dataclasses(Player, Bean)

    def action_space(self) -> Dict[str, ActionSpace]:
        return {
            "count": CategoricalActionSpace(
                ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
            )
        }

    def reset_filter(self, obs_space: ObsSpace) -> Observation:
        self.count = random.randint(0, self.masked_choices - 1)
        possible_counts = {
            self.count,
            *random.sample(
                range(0, self.masked_choices), random.randint(0, self.masked_choices)
            ),
        }
        mask = np.zeros((1, 10), dtype=np.bool_)
        mask[:, list(possible_counts)] = True
        return self.observe(obs_space, mask)

    def reset(self) -> Observation:
        return self.reset_filter(self.obs_space())

    def act_filter(
        self, action: Mapping[str, Action], obs_filter: ObsSpace
    ) -> Observation:
        reward = 0.0
        assert len(action) == 1
        a = action["count"]
        assert isinstance(a, CategoricalAction)
        assert len(a.indices) == 1
        choice = a.indices[0]
        if choice == self.count:
            reward = 1.0
        return self.observe(obs_filter, None, done=True, reward=reward)

    def act(self, actions: Mapping[str, Action]) -> Observation:
        return self.act_filter(
            actions,
            self.obs_space(),
        )

    def observe(
        self,
        obs_filter: ObsSpace,
        mask: Optional[npt.NDArray[np.bool_]],
        done: bool = False,
        reward: float = 0.0,
    ) -> Observation:
        return Observation(
            features=extract_features(
                {
                    "Player": [Player()],
                    "Bean": [Bean()] * self.count,
                },
                obs_filter,
            ),
            actions={
                "count": CategoricalActionMask(actor_ids=["Player"], mask=mask),
            },
            ids={
                "Player": ["Player"],
                "Bean": [f"Bean{i}" for i in range(1, self.count + 1)],
            },
            reward=reward,
            done=done,
        )
