import random
from dataclasses import dataclass
from typing import Dict, Mapping

import numpy as np

from entity_gym.env import (
    Action,
    ActionSpace,
    CategoricalAction,
    CategoricalActionMask,
    CategoricalActionSpace,
    Entity,
    Environment,
    Observation,
    ObsSpace,
)


@dataclass
class Slider(Environment):
    """
    On each timestep, there is either a generic "Object" entity with a `is_hotdog` property, or a "Hotdog" object.
    The "Player" entity is always present, and has an action to classify the other entity as hotdog or not hotdog.
    """

    def obs_space(self) -> ObsSpace:
        return ObsSpace(
            entities={
                "Player": Entity(["step"]),
                "Slider": Entity(["value"]),
            }
        )

    def action_space(self) -> Dict[str, ActionSpace]:
        return {
            "classify": CategoricalActionSpace([str(i / 1000) for i in range(1000)]),
        }

    def reset(self) -> Observation:
        self.step = 0
        self.slider = random.random()
        return self.observe()

    def act(self, actions: Mapping[str, Action]) -> Observation:
        self.step += 1

        a = actions["classify"]
        assert isinstance(a, CategoricalAction), f"{a} is not a CategoricalAction"
        val = float(a.index_to_label[a.indices[0]])
        reward = 1 - abs(val - self.slider)
        done = True

        return self.observe(done, reward)

    def observe(self, done: bool = False, reward: float = 0) -> Observation:
        return Observation(
            features={
                "Player": np.array(
                    [
                        [
                            self.step,
                        ]
                    ],
                    dtype=np.float32,
                ),
                "Slider": np.array(
                    [
                        [
                            self.slider,
                        ]
                    ],
                    dtype=np.float32,
                ),
            },
            actions={
                "classify": CategoricalActionMask(actor_ids=[0]),
            },
            ids={"Player": [0]},
            reward=reward,
            done=done,
        )
