import random
from dataclasses import dataclass
from typing import Dict, Mapping

from entity_gym.env import Action, ActionSpace, Environment, Observation, ObsSpace
from entity_gym.env.environment import (
    Entity,
    GlobalCategoricalAction,
    GlobalCategoricalActionMask,
    GlobalCategoricalActionSpace,
)


@dataclass
class Input:
    is_set: float


class Xor(Environment):
    """
    There are three entities types, each with one instance on each timstep.
    The Bit1 and Bit2 entities are randomly set to 0 or 1.
    The Output entity has one action that should be set to the output of the XOR between the two bits.
    """

    def obs_space(self) -> ObsSpace:
        return ObsSpace(
            global_features=["negate"],
            entities={"Input": Entity(["is_set"])},
        )

    def action_space(self) -> Dict[str, ActionSpace]:
        return {"output": GlobalCategoricalActionSpace(["0", "1"])}

    def reset_filter(self, obs_space: ObsSpace) -> Observation:
        self.bit1 = random.choice([0.0, 1.0])
        self.bit2 = random.choice([0.0, 1.0])
        self.negate = random.choice([0.0, 1.0])
        return self.observe(obs_space)

    def reset(self) -> Observation:
        return self.reset_filter(self.obs_space())

    def act_filter(
        self, action: Mapping[str, Action], obs_filter: ObsSpace
    ) -> Observation:
        reward = 0.0
        a = action["output"]
        assert isinstance(a, GlobalCategoricalAction)
        if a.index == self.negate and self.bit1 == self.bit2:
            reward = 1.0
        elif a.index == 1.0 - self.negate and self.bit1 != self.bit2:
            reward = 1.0

        return self.observe(obs_filter, done=True, reward=reward)

    def act(self, actions: Mapping[str, Action]) -> Observation:
        return self.act_filter(
            actions,
            self.obs_space(),
        )

    def observe(
        self, obs_filter: ObsSpace, done: bool = False, reward: float = 0.0
    ) -> Observation:
        return Observation(
            reward=reward,
            done=done,
            features={"Input": [[self.bit1], [self.bit2]]},
            global_features=[self.negate],
            actions={
                "output": GlobalCategoricalActionMask(),
            },
        )
