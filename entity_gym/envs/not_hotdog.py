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


@dataclass
class NotHotdog(Environment):
    """
    Classify the object as hotdog or not.
    """

    multi_object: bool = True

    @classmethod
    def obs_space(cls) -> ObsSpace:
        return ObsSpace(
            {
                "Player": Entity(["step"]),
                "Object": Entity(["is_hotdog"]),
                "Hotdog": Entity([]),
            }
        )

    @classmethod
    def action_space(cls) -> Dict[str, ActionSpace]:
        return {
            "classify": CategoricalActionSpace(["hotdog", "not_hotdog"]),
        }

    def _reset(self) -> Observation:
        self.step = 0
        self.is_hotdog = random.randint(0, 1)
        return self.observe()

    def _act(self, action: Mapping[str, Action]) -> Observation:
        self.step += 1

        a = action["classify"]
        assert isinstance(a, CategoricalAction), f"{a} is not a CategoricalAction"
        if a.actions[0][1] == self.is_hotdog:
            reward = 1
        else:
            reward = 0
        done = True
        return self.observe(done, reward)

    def observe(self, done: bool = False, reward: float = 0) -> Observation:
        return Observation(
            entities={
                "Player": np.array(
                    [
                        [
                            self.step,
                        ]
                    ],
                    dtype=np.float32,
                ),
                "Object": np.array(
                    [
                        [
                            self.is_hotdog,
                        ]
                    ],
                    dtype=np.float32,
                )
                if (self.multi_object and self.is_hotdog == 0) or not self.multi_object
                else np.zeros((0, 1), dtype=np.float32).reshape(0, 1),
                "Hotdog": np.zeros((1, 0), dtype=np.float32)
                if self.multi_object and self.is_hotdog == 1
                else np.zeros((0, 0), dtype=np.float32),
            },
            action_masks={
                "classify": DenseCategoricalActionMask(actors=np.array([0]), mask=None),
            },
            ids=[0],
            reward=reward,
            done=done,
            end_of_episode_info=EpisodeStats(self.step, reward) if done else None,
        )
