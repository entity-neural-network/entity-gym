import random
from typing import Dict, List, Mapping, Tuple

from entity_gym.env import *


class MineSweeper(Environment):
    """
    The MineSweeper environment contains two types of objects, mines and robots.
    The player controls all robots in the environment.
    On every step, each robot may move in one of four cardinal directions, or stay in place and defuse all adjacent mines.
    If a robot defuses a mine, it is removed from the environment.
    If a robot steps on a mine, it is removed from the environment and the player loses the game.
    The player wins the game when all mines are defused.
    """

    def __init__(
        self,
        width: int = 6,
        height: int = 6,
        nmines: int = 5,
        nrobots: int = 2,
        orbital_cannon: bool = False,
        cooldown_period: int = 5,
    ):
        self.width = width
        self.height = height
        self.nmines = nmines
        self.nrobots = nrobots
        self.orbital_cannon = orbital_cannon
        self.cooldown_period = cooldown_period
        self.orbital_cannon_cooldown = cooldown_period
        # Positions of robots and mines
        self.robots: List[Tuple[int, int]] = []
        self.mines: List[Tuple[int, int]] = []

    def obs_space(cls) -> ObsSpace:
        return ObsSpace(
            entities={
                "Mine": Entity(features=["x", "y"]),
                "Robot": Entity(features=["x", "y"]),
                "Orbital Cannon": Entity(["cooldown"]),
            }
        )

    def action_space(cls) -> Dict[ActionName, ActionSpace]:
        return {
            "Move": CategoricalActionSpace(
                ["Up", "Down", "Left", "Right", "Defuse Mines"],
            ),
            "Fire Orbital Cannon": SelectEntityActionSpace(),
        }

    def reset(self) -> Observation:
        positions = random.sample(
            [(x, y) for x in range(self.width) for y in range(self.height)],
            self.nmines + self.nrobots,
        )
        self.mines = positions[: self.nmines]
        self.robots = positions[self.nmines :]
        self.orbital_cannon_cooldown = self.cooldown_period
        return self.observe()

    def observe(self) -> Observation:
        done = len(self.mines) == 0 or len(self.robots) == 0
        reward = 1.0 if len(self.mines) == 0 else 0.0
        return Observation(
            entities={
                "Mine": (
                    self.mines,
                    [("Mine", i) for i in range(len(self.mines))],
                ),
                "Robot": (
                    self.robots,
                    [("Robot", i) for i in range(len(self.robots))],
                ),
                "Orbital Cannon": (
                    [(self.orbital_cannon_cooldown,)],
                    [("Orbital Cannon", 0)],
                )
                if self.orbital_cannon
                else None,
            },
            actions={
                "Move": CategoricalActionMask(
                    # Allow all robots to move
                    actor_types=["Robot"],
                    mask=[self.valid_moves(x, y) for x, y in self.robots],
                ),
                "Fire Orbital Cannon": SelectEntityActionMask(
                    # Only the Orbital Cannon can fire, but not if cooldown > 0
                    actor_types=["Orbital Cannon"]
                    if self.orbital_cannon_cooldown == 0
                    else [],
                    # Both mines and robots can be fired at
                    actee_types=["Mine", "Robot"],
                ),
            },
            # The game is done once there are no more mines or robots
            done=done,
            # Give reward of 1.0 for defusing all mines
            reward=reward,
        )

    def act(self, actions: Mapping[ActionName, Action]) -> Observation:
        fire = actions["Fire Orbital Cannon"]
        assert isinstance(fire, SelectEntityAction)
        remove_robot = None
        for (entity_type, i) in fire.actees:
            if entity_type == "Mine":
                self.mines.remove(self.mines[i])
            elif entity_type == "Robot":
                # Don't remove yet to keep indices valid
                remove_robot = i

        move = actions["Move"]
        assert isinstance(move, CategoricalAction)
        for (_, i), choice in zip(move.actors, move.indices):
            if self.robots[i] is None:
                continue
            # Action space is ["Up", "Down", "Left", "Right", "Defuse Mines"],
            x, y = self.robots[i]
            if choice == 0 and y < self.height - 1:
                self.robots[i] = (x, y + 1)
            elif choice == 1 and y > 0:
                self.robots[i] = (x, y - 1)
            elif choice == 2 and x > 0:
                self.robots[i] = (x - 1, y)
            elif choice == 3 and x < self.width - 1:
                self.robots[i] = (x + 1, y)
            elif choice == 4:
                # Remove all mines adjacent to this robot
                rx, ry = self.robots[i]
                self.mines = [
                    (x, y) for (x, y) in self.mines if abs(x - rx) + abs(y - ry) > 1
                ]

        if remove_robot is not None:
            self.robots.pop(remove_robot)
        # Remove all robots that stepped on a mine
        self.robots = [r for r in self.robots if r not in self.mines]

        return self.observe()

    def valid_moves(self, x: int, y: int) -> List[bool]:
        return [
            x < self.width - 1,
            x > 0,
            y < self.height - 1,
            y > 0,
            # Always allow staying in place and defusing mines
            True,
        ]
