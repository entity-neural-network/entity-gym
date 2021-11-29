from dataclasses import dataclass, field
import numpy as np
import random
from typing import Dict, List, Mapping, Tuple

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

    @classmethod
    def obs_space(cls) -> ObsSpace:
        return obs_space_from_dataclasses(Vehicle, Mine, Target)

    @classmethod
    def action_space(cls) -> Dict[str, ActionSpace]:
        return {
            "move": CategoricalActionSpace(["turn left", "move forward", "turn right"],)
        }

    def reset(self, obs_space: ObsSpace) -> Observation:
        self.vehicle.x_pos, self.vehicle.y_pos = (
            random.uniform(-100, 100),
            random.uniform(-100, 100),
        )
        self.target.x_pos, self.target.y_pos = (
            random.uniform(-100, 100),
            random.uniform(-100, 100),
        )
        mines: List[Mine] = []
        for _ in range(10):
            x, y = (random.uniform(-100, 100), random.uniform(-100, 100))
            # Check that the mine is not too close to the vehicle, target, or any other mine
            pos = [(m.x_pos, m.y_pos) for m in mines] + [
                (self.vehicle.x_pos, self.vehicle.y_pos),
                (self.target.x_pos, self.target.y_pos),
            ]
            if any(map(lambda p: (x - p[0]) ** 2 + (y - p[1]) ** 2 < 15 * 15, pos)):
                continue
            mines.append(Mine(x, y))
        self.direction = random.uniform(0, 2 * np.pi)
        self.step = 0
        self.mines = mines
        return self.observe(obs_space)

    def _reset(self) -> Observation:
        return self.reset(Minefield.obs_space())

    def act(self, action: Mapping[str, Action], obs_filter: ObsSpace) -> Observation:
        for action_name, a in action.items():
            assert isinstance(a, CategoricalAction)
            if action_name == "move":
                move = a.actions[0][1]
                if move == 0:
                    self.direction -= np.pi / 8
                elif move == 1:
                    self.vehicle.x_pos += 3 * np.cos(self.direction)
                    self.vehicle.y_pos += 3 * np.sin(self.direction)
                elif move == 2:
                    self.direction += np.pi / 8
                else:
                    raise ValueError(
                        f"Invalid action {move} for action space {action_name}"
                    )
                self.direction %= 2 * np.pi
            else:
                raise ValueError(f"Unknown action type {action_name}")

        self.step += 1

        return self.observe(obs_filter)

    def _act(self, action: Mapping[str, Action]) -> Observation:
        return self.act(action, Minefield.obs_space(),)

    def observe(self, obs_filter: ObsSpace, done: bool = False) -> Observation:
        if (self.target.x_pos - self.vehicle.x_pos) ** 2 + (
            self.target.y_pos - self.vehicle.y_pos
        ) ** 2 < 5 * 5:
            done = True
            reward = 1
        elif any(
            map(
                lambda m: (self.vehicle.x_pos - m.x_pos) ** 2
                + (self.vehicle.y_pos - m.y_pos) ** 2
                < 5 * 5,
                self.mines,
            )
        ):
            done = True
            reward = 0
        else:
            done = False
            reward = 0

        return Observation(
            entities=extract_features(
                {
                    "Mine": self.mines,
                    "Vehicle": [self.vehicle],
                    "Target": [self.target],
                },
                obs_filter,
            ),
            action_masks={
                "move": DenseCategoricalActionMask(actors=np.array([0]), mask=None),
            },
            ids=list(range(len(self.mines) + 2)),
            reward=reward,
            done=done,
            end_of_episode_info=EpisodeStats(self.step, reward) if done else None,
        )
