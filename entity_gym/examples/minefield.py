import random
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Tuple

import numpy as np

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
class Vehicle:
    x_pos: float = 0.0
    y_pos: float = 0.0
    direction: float = 0.0
    step: int = 0


@dataclass
class Target:
    x_pos: float = 0.0
    y_pos: float = 0.0


@dataclass
class Mine:
    x_pos: float = 0.0
    y_pos: float = 0.0


@dataclass
class Minefield(Environment):
    """
    Task with a Vehicle entity that has to reach a target point, receiving a reward of 1.
    If the vehicle collides with any of the randomly placed mines, the episode ends without reward.
    The available actions either turn the vehicle left, right, or go straight.
    """

    vehicle: Vehicle = field(default_factory=Vehicle)
    target: Target = field(default_factory=Target)
    mine: Mine = field(default_factory=Mine)
    max_mines: int = 10
    max_steps: int = 200
    translate: bool = False
    width: float = 200.0

    def obs_space(self) -> ObsSpace:
        return obs_space_from_dataclasses(Vehicle, Mine, Target)

    def action_space(self) -> Dict[str, ActionSpace]:
        return {
            "move": CategoricalActionSpace(
                ["turn left", "move forward", "turn right"],
            )
        }

    def reset_filter(self, obs_space: ObsSpace) -> Observation:
        def randpos() -> Tuple[float, float]:
            return (
                random.uniform(-self.width / 2, self.width / 2),
                random.uniform(-self.width / 2, self.width / 2),
            )

        self.vehicle.x_pos, self.vehicle.y_pos = randpos()
        self.target.x_pos, self.target.y_pos = randpos()
        mines: List[Mine] = []
        for _ in range(self.max_mines):
            x, y = randpos()
            # Check that the mine is not too close to the vehicle, target, or any other mine
            pos = [(m.x_pos, m.y_pos) for m in mines] + [
                (self.vehicle.x_pos, self.vehicle.y_pos),
                (self.target.x_pos, self.target.y_pos),
            ]
            if any(map(lambda p: (x - p[0]) ** 2 + (y - p[1]) ** 2 < 15 * 15, pos)):
                continue
            mines.append(Mine(x, y))
        self.vehicle.direction = random.uniform(0, 2 * np.pi)
        self.step = 0
        self.mines = mines
        return self.observe(obs_space)

    def reset(self) -> Observation:
        return self.reset_filter(self.obs_space())

    def act_filter(
        self, action: Mapping[str, Action], obs_filter: ObsSpace
    ) -> Observation:
        for action_name, a in action.items():
            assert isinstance(a, CategoricalAction)
            if action_name == "move":
                move = a.indices[0]
                if move == 0:
                    self.vehicle.direction -= np.pi / 8
                elif move == 1:
                    self.vehicle.x_pos += 3 * np.cos(self.vehicle.direction)
                    self.vehicle.y_pos += 3 * np.sin(self.vehicle.direction)
                elif move == 2:
                    self.vehicle.direction += np.pi / 8
                else:
                    raise ValueError(
                        f"Invalid action {move} for action space {action_name}"
                    )
                self.vehicle.direction %= 2 * np.pi
            else:
                raise ValueError(f"Unknown action type {action_name}")

        self.step += 1
        self.vehicle.step = self.step

        return self.observe(obs_filter)

    def act(self, actions: Mapping[str, Action]) -> Observation:
        return self.act_filter(
            actions,
            self.obs_space(),
        )

    def observe(self, obs_filter: ObsSpace, done: bool = False) -> Observation:
        if (self.target.x_pos - self.vehicle.x_pos) ** 2 + (
            self.target.y_pos - self.vehicle.y_pos
        ) ** 2 < 5 * 5:
            done = True
            reward = 1
        elif (
            any(
                map(
                    lambda m: (self.vehicle.x_pos - m.x_pos) ** 2
                    + (self.vehicle.y_pos - m.y_pos) ** 2
                    < 5 * 5,
                    self.mines,
                )
            )
            or self.step >= self.max_steps
        ):
            done = True
            reward = 0
        else:
            done = False
            reward = 0

        if self.translate:
            ox = self.vehicle.x_pos
            oy = self.vehicle.y_pos
        else:
            ox = oy = 0
        return Observation(
            features=extract_features(
                {
                    "Mine": [Mine(m.x_pos - ox, m.y_pos - oy) for m in self.mines],
                    "Vehicle": [self.vehicle],
                    "Target": [Target(self.target.x_pos - ox, self.target.y_pos - oy)],
                },
                obs_filter,
            ),
            actions={
                "move": CategoricalActionMask(actor_types=["Vehicle"]),
            },
            ids={"Vehicle": ["Vehicle"]},
            reward=reward,
            done=done,
        )
