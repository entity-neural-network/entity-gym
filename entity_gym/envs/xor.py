from dataclasses import dataclass
import numpy as np
import random
from typing import Dict, Mapping

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
class Bit1:
    is_set: float


@dataclass
class Bit2:
    is_set: float


@dataclass
class Output:
    pass


class Xor(Environment):
    """
    There are three entities types, each with one instance on each timstep.
    The Bit1 and Bit2 entities are randomly set to 0 or 1.
    The Ouput entity has one action that should be set to the output of the XOR between the two bits.
    """

    @classmethod
    def obs_space(cls) -> ObsSpace:
        return obs_space_from_dataclasses(Output, Bit1, Bit2)

    @classmethod
    def action_space(cls) -> Dict[str, ActionSpace]:
        return {"output": CategoricalActionSpace(["0", "1"])}

    def reset(self, obs_space: ObsSpace) -> Observation:
        self.bit1 = Bit1(random.choice([0.0, 1.0]))
        self.bit2 = Bit2(random.choice([0.0, 1.0]))
        return self.observe(obs_space)

    def _reset(self) -> Observation:
        return self.reset(Xor.obs_space())

    def act(self, action: Mapping[str, Action], obs_filter: ObsSpace) -> Observation:
        reward = 0.0
        for action_name, a in action.items():
            assert isinstance(a, CategoricalAction)
            if action_name == "output":
                if a.actions[0][1] == 0 and self.bit1.is_set == self.bit2.is_set:
                    reward = 1.0
                elif a.actions[0][1] == 1 and self.bit1.is_set != self.bit2.is_set:
                    reward = 1.0

        return self.observe(obs_filter, done=True, reward=reward)

    def _act(self, action: Mapping[str, Action]) -> Observation:
        return self.act(
            action,
            Xor.obs_space(),
        )

    def observe(
        self, obs_filter: ObsSpace, done: bool = False, reward: float = 0.0
    ) -> Observation:
        return Observation(
            entities=extract_features(
                {
                    "Output": [Output()],
                    "Bit1": [self.bit1],
                    "Bit2": [self.bit2],
                },
                obs_filter,
            ),
            action_masks={
                "output": DenseCategoricalActionMask(actors=np.array([0]), mask=None),
            },
            ids=list(range(3)),
            reward=reward,
            done=done,
            end_of_episode_info=EpisodeStats(1, reward) if done else None,
        )
