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
    Observation,
    Action,
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

    @classmethod
    def state_space(cls) -> Dict[str, Entity]:
        return {
            "Spaceship": Entity(["x_pos", "y_pos", "x_velocity", "y_velocity", "step"]),
        }

    @classmethod
    def action_space(cls) -> Dict[str, ActionSpace]:
        return {
            "horizontal_thruster": CategoricalActionSpace(
                ["100% right", "10% right", "hold", "10% left", "100% left",],
            ),
            "vertical_thruster": CategoricalActionSpace(
                ["100% up", "10% up", "hold", "10% down", "100% down"],
            ),
        }

    def _reset(self) -> Observation:
        angle = random.uniform(0, 2 * np.pi)
        self.x_pos = np.cos(angle)
        self.y_pos = np.sin(angle)
        self.last_x_pos = self.x_pos
        self.last_y_pos = self.y_pos
        self.x_velocity = 0
        self.y_velocity = 0
        self.step = 0
        return self.observe()

    def _act(self, action: Mapping[str, Action]) -> Observation:
        self.step += 1

        for action_name, a in action.items():
            assert isinstance(a, CategoricalAction), f"{a} is not a CategoricalAction"
            if action_name == "horizontal_thruster":
                for actor_id, choice_id in a.actions:
                    if choice_id == 0:
                        self.x_velocity += 0.01
                    elif choice_id == 1:
                        self.x_velocity += 0.001
                    elif choice_id == 2:
                        pass
                    elif choice_id == 3:
                        self.x_velocity -= 0.001
                    elif choice_id == 4:
                        self.x_velocity -= 0.01
                    else:
                        raise ValueError(f"Invalid choice id {choice_id}")
            elif action_name == "vertical_thruster":
                for actor_id, choice_id in a.actions:
                    if choice_id == 0:
                        self.y_velocity += 0.01
                    elif choice_id == 1:
                        self.y_velocity += 0.001
                    elif choice_id == 2:
                        pass
                    elif choice_id == 3:
                        self.y_velocity -= 0.001
                    elif choice_id == 4:
                        self.y_velocity -= 0.01
                    else:
                        raise ValueError(f"Invalid choice id {choice_id}")
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
            entities={
                "Spaceship": np.array(
                    [
                        [
                            self.x_pos,
                            self.y_pos,
                            self.x_velocity,
                            self.y_velocity,
                            self.step,
                        ]
                    ]
                ),
            },
            action_masks={
                "horizontal_thruster": DenseCategoricalActionMask(
                    actors=[0], mask=None
                ),
                "vertical_thruster": DenseCategoricalActionMask(actors=[0], mask=None),
            },
            ids=[0],
            reward=(self.last_x_pos ** 2 + self.last_y_pos ** 2) ** 0.5
            - (self.x_pos ** 2 + self.y_pos ** 2) ** 0.5,
            done=done,
        )
