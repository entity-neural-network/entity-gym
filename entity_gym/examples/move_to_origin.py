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
class MoveToOrigin(Environment):
    """
    Task with a single Spaceship that is rewarded for moving as close to the origin as possible.
    The Spaceship has two actions for accelerating the Spaceship in the x and y directions.
    """

    x_pos: float = 0.0
    y_pos: float = 0.0
    x_velocity: float = 0.0
    y_velocity: float = 0.0
    last_x_pos = 0.0
    last_y_pos = 0.0
    step: int = 0

    def obs_space(cls) -> ObsSpace:
        return ObsSpace(
            entities={
                "Spaceship": Entity(
                    ["x_pos", "y_pos", "x_velocity", "y_velocity", "step"]
                ),
            }
        )

    def action_space(cls) -> Dict[str, ActionSpace]:
        return {
            "horizontal_thruster": CategoricalActionSpace(
                [
                    "100% right",
                    "10% right",
                    "hold",
                    "10% left",
                    "100% left",
                ],
            ),
            "vertical_thruster": CategoricalActionSpace(
                ["100% up", "10% up", "hold", "10% down", "100% down"],
            ),
        }

    def reset(self) -> Observation:
        angle = random.uniform(0, 2 * np.pi)
        self.x_pos = np.cos(angle)
        self.y_pos = np.sin(angle)
        self.last_x_pos = self.x_pos
        self.last_y_pos = self.y_pos
        self.x_velocity = 0
        self.y_velocity = 0
        self.step = 0
        return self.observe()

    def act(self, actions: Mapping[str, Action]) -> Observation:
        self.step += 1

        for action_name, a in actions.items():
            assert isinstance(a, CategoricalAction), f"{a} is not a CategoricalAction"
            if action_name == "horizontal_thruster":
                for label in a.labels:
                    if label == "100% right":
                        self.x_velocity += 0.01
                    elif label == "10% right":
                        self.x_velocity += 0.001
                    elif label == "hold":
                        pass
                    elif label == "10% left":
                        self.x_velocity -= 0.001
                    elif label == "100% left":
                        self.x_velocity -= 0.01
                    else:
                        raise ValueError(f"Invalid choice id {label}")
            elif action_name == "vertical_thruster":
                for label in a.labels:
                    if label == "100% up":
                        self.y_velocity += 0.01
                    elif label == "10% up":
                        self.y_velocity += 0.001
                    elif label == "hold":
                        pass
                    elif label == "10% down":
                        self.y_velocity -= 0.001
                    elif label == "100% down":
                        self.y_velocity -= 0.01
                    else:
                        raise ValueError(f"Invalid choice id {label}")
            else:
                raise ValueError(f"Unknown action type {action_name}")

        self.last_x_pos = self.x_pos
        self.last_y_pos = self.y_pos

        self.x_pos += self.x_velocity
        self.y_pos += self.y_velocity

        done = self.step >= 32
        return self.observe(done)

    def observe(self, done: bool = False) -> Observation:
        return Observation(
            ids={
                "Spaceship": [0],
            },
            features={
                "Spaceship": np.array(
                    [
                        [
                            self.x_pos,
                            self.y_pos,
                            self.x_velocity,
                            self.y_velocity,
                            self.step,
                        ]
                    ],
                    dtype=np.float32,
                ),
            },
            actions={
                "horizontal_thruster": CategoricalActionMask(),
                "vertical_thruster": CategoricalActionMask(),
            },
            reward=(self.last_x_pos**2 + self.last_y_pos**2) ** 0.5
            - (self.x_pos**2 + self.y_pos**2) ** 0.5,
            done=done,
        )
