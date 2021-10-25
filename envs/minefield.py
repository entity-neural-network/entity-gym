from dataclasses import dataclass, field
import numpy as np
import random
from typing import List, Tuple

from entity_gym.environment import ActionMask, Environment, Type, CategoricalActionSpace, ActionSpace, ObsFilter, Observation, Action


@dataclass
class Minefield(Environment):
    """
    Task with a Vehicle entity that has to reach a target point, receiving a reward of 1.
    If the vehicle collides with any of the randomly placed mines, the episode ends without reward.
    The available actions either turn the vehicle left, right, or go straight.
    """
    x_pos: float = 0.0
    y_pos: float = 0.0
    direction: float = 0.0
    x_pos_target: float = 0.0
    y_pos_target: float = 0.0
    mines: List[Tuple[float, float]] = field(default_factory=list)
    step: int = 0

    @classmethod
    def state_space(cls) -> List[Type]:
        return [
            Type(
                name="Vehicle",
                features=["x_pos", "y_pos", "direction", "step"],
            ),
            Type(
                name="Mine",
                features=["x_pos", "y_pos"],
            ),
            Type(
                name="Target",
                features=["x_pos", "y_pos"],
            )
        ]

    @classmethod
    def action_space(cls) -> List[ActionSpace]:
        return [
            CategoricalActionSpace(
                name="move",
                n=3,
                choice_labels=["turn left", "move forward", "turn right"],
            )
        ]

    def reset(self, obs_config: ObsFilter) -> Observation:
        self.x_pos, self.y_pos = (
            random.uniform(-100, 100), random.uniform(-100, 100))
        self.x_pos_target, self.y_pos_target = (
            random.uniform(-100, 100), random.uniform(-100, 100))
        mines = []
        for _ in range(10):
            x, y = (random.uniform(-100, 100), random.uniform(-100, 100))
            # Check that the mine is not too close to the vehicle, target, or any other mine
            pos = list(mines) + [(self.x_pos, self.y_pos),
                                 (self.x_pos_target, self.y_pos_target)]
            if any(map(lambda p: (x - p[0]) ** 2 + (y - p[1]) ** 2 < 15 * 15, pos)):
                continue
            mines.append((x, y))
        self.direction = random.uniform(0, 2 * np.pi)
        self.step = 0
        self.mines = mines
        return self.observe(obs_config)

    def act(self, action: Action, obs_config: ObsFilter) -> Observation:
        for action_name, chosen_actions in action.chosen_actions.items():
            if action_name == "move":
                action = chosen_actions[0][1]
                if action == 0:
                    self.direction -= np.pi / 8
                elif action == 1:
                    self.x_pos += 3 * np.cos(self.direction)
                    self.y_pos += 3 * np.sin(self.direction)
                elif action == 2:
                    self.direction += np.pi / 8
                else:
                    raise ValueError(
                        f"Invalid action {action} for action space {action_name}")
                self.direction %= 2 * np.pi
            else:
                raise ValueError(f"Unknown action type {action_name}")

        self.step += 1

        return self.observe(obs_config)

    def observe(self, obs_config: ObsFilter, done: bool = False) -> Observation:
        if (self.x_pos_target - self.x_pos) ** 2 + (
                self.y_pos_target - self.y_pos) ** 2 < 5 * 5:
            done = True
            reward = 1
        elif any(map(lambda m: (self.x_pos - m[0]) ** 2 + (self.y_pos - m[1]) ** 2 < 5 * 5, self.mines)):
            done = True
            reward = 0
        else:
            done = False
            reward = 0

        return self.filter_obs(Observation(
            entities=[
                (
                    "Vehicle",
                    np.array(
                        [[self.x_pos, self.y_pos, self.direction, self.step]])
                ),
                (
                    "Mine",
                    np.array([[x, y] for x, y in self.mines]) if len(
                        self.mines) > 0 else np.zeros([0, 2])
                ),
                (
                    "Target",
                    np.array(
                        [[self.x_pos_target, self.y_pos_target]])
                )
            ],
            action_masks=[
                ("move", ActionMask(actors=[0], mask=None)),
            ],
            ids=list(range(len(self.mines) + 2)),
            reward=reward,
            done=done,
        ), obs_config)
