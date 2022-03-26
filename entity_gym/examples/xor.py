import random
from dataclasses import dataclass
from typing import Dict, Mapping

from entity_gym.dataclass_utils import extract_features, obs_space_from_dataclasses
from entity_gym.environment import (
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
    The Output entity has one action that should be set to the output of the XOR between the two bits.
    """

    @classmethod
    def obs_space(cls) -> ObsSpace:
        return obs_space_from_dataclasses(Output, Bit1, Bit2)

    @classmethod
    def action_space(cls) -> Dict[str, ActionSpace]:
        return {"output": CategoricalActionSpace(["0", "1"])}

    def reset_filter(self, obs_space: ObsSpace) -> Observation:
        self.bit1 = Bit1(random.choice([0.0, 1.0]))
        self.bit2 = Bit2(random.choice([0.0, 1.0]))
        return self.observe(obs_space)

    def reset(self) -> Observation:
        return self.reset_filter(Xor.obs_space())

    def act_filter(
        self, action: Mapping[str, Action], obs_filter: ObsSpace
    ) -> Observation:
        reward = 0.0
        for action_name, a in action.items():
            assert isinstance(a, CategoricalAction)
            if action_name == "output":
                if a.actions[0] == 0 and self.bit1.is_set == self.bit2.is_set:
                    reward = 1.0
                elif a.actions[0] == 1 and self.bit1.is_set != self.bit2.is_set:
                    reward = 1.0

        return self.observe(obs_filter, done=True, reward=reward)

    def act(self, action: Mapping[str, Action]) -> Observation:
        return self.act_filter(
            action,
            Xor.obs_space(),
        )

    def observe(
        self, obs_filter: ObsSpace, done: bool = False, reward: float = 0.0
    ) -> Observation:
        return Observation(
            features=extract_features(
                {
                    "Output": [Output()],
                    "Bit1": [self.bit1],
                    "Bit2": [self.bit2],
                },
                obs_filter,
            ),
            actions={
                "output": CategoricalActionMask(actor_ids=[0]),
            },
            ids={"Output": [0], "Bit1": [1], "Bit2": [2]},
            reward=reward,
            done=done,
        )
