from dataclasses import dataclass
import dataclasses
from typing import Dict, List, Tuple
import random
import numpy as np

from entity_gym.environment import ActionMask, DenseCategoricalActionMask, Entity, Environment, Type, CategoricalActionSpace, ActionSpace, ObsFilter, Observation, Action, VecEnv


@dataclass
class Snake:
    color: int
    segments: List[Tuple[int, int]]


@dataclass
class Food:
    color: int
    position: Tuple[int, int]


class MultiSnake(Environment):
    """
    Turn-based version of Snake with multiple snakes.
    Each snake has a different color.
    For each snake, Food of that color is placed randomly on the board.
    Snakes can only eat Food of their color.
    When a snake eats Food of the same color, it grows by one unit.
    When a snake grows and it's length was less than 11, the player receives a reward of 0.1 / num_snakes.
    The game ends when a snake collides with another snake, runs into a wall, eats Food of another color, or all snakes reach a length of 11.
    """

    def __init__(self, board_size: int = 10, num_snakes: int = 2):
        """
        :param num_players: number of players
        :param board_size: size of the board
        :param num_snakes: number of snakes per player
        """
        assert num_snakes < 10, f"num_snakes must be less than 10, got {num_snakes}"
        self.board_size = board_size
        self.num_snakes = num_snakes
        self.snakes = []
        self.Food = []

    @classmethod
    def state_space(cls) -> List[Entity]:
        return [
            Entity(
                name="SnakeHead",
                features=["x", "y", "color"],
            ),
            Entity(
                name="SnakeBody",
                features=["x", "y", "color"],
            ),
            Entity(
                name="Food",
                features=["x", "y", "color"],
            )
        ]

    @classmethod
    def action_space(cls) -> List[ActionSpace]:
        return [
            CategoricalActionSpace(
                name="move",
                n=4,
                choice_labels=["up", "down", "left", "right"],
            ),
        ]

    def _spawn_snake(self, color: int):
        while True:
            x = random.randint(0, self.board_size - 1)
            y = random.randint(0, self.board_size - 1)
            if any(
                (x, y) == (sx, sy) for snake in self.snakes for sx, sy in snake.segments
            ):
                continue
            self.snakes.append(Snake(color, [(x, y)]))
            break

    def _spawn_Food(self, color: int):
        while True:
            x = random.randint(0, self.board_size - 1)
            y = random.randint(0, self.board_size - 1)
            if any(
                (x, y) == (f.position[0], f.position[1]) for f in self.Food
            ) or any(
                (x, y) == (sx, sy) for snake in self.snakes for sx, sy in snake.segments
            ):
                continue
            self.Food.append(Food(color, (x, y)))
            break

    def _reset(self) -> Observation:
        self.snakes = []
        self.Food = []
        for i in range(self.num_snakes):
            self._spawn_snake(i)
        for i in range(self.num_snakes):
            self._spawn_Food(i)
        return self._observe()

    def _act(self, action: Dict[str, Action]) -> Observation:
        game_over = False
        reward = 0
        for id, move in action["move"].actions:
            snake = self.snakes[id]
            x, y = snake.segments[-1]
            if move == 0:
                y += 1
            elif move == 1:
                y -= 1
            elif move == 2:
                x -= 1
            elif move == 3:
                x += 1
            if x < 0 or x >= self.board_size or y < 0 or y >= self.board_size:
                game_over = True
            if any(
                (x, y) == (sx, sy) for s in self.snakes for sx, sy in s.segments
            ):
                game_over = True
            ate_Food = False
            for i in range(len(self.Food)):
                if self.Food[i].position == (x, y):
                    if self.Food[i].color != snake.color:
                        game_over = True
                    elif len(snake.segments) < 11:
                        ate_Food = True
                        reward += 0.1 / self.num_snakes
                    self.Food.pop(i)
                    self._spawn_Food(snake.color)
                    break
            snake.segments.append((x, y))
            if not ate_Food:
                snake.segments = snake.segments[1:]
        if all(len(s.segments) >= 11 for s in self.snakes):
            game_over = True
        return self._observe(done=game_over, reward=reward)

    def _observe(self, done: bool = False, reward: float = 0) -> Observation:
        return Observation(
            entities=[
                (
                    "SnakeHead",
                    np.array(
                        [
                            [s.segments[0][0], s.segments[0][1], s.color]
                            for s in self.snakes
                        ]
                    ),
                ),
                (
                    "SnakeBody",
                    np.array(
                        [
                            [sx, sy, snake.color]
                            for snake in self.snakes
                            for sx, sy in snake.segments[1:]
                        ]
                    ).reshape(-1, 3)
                ),
                (
                    "Food",
                    np.array(
                        [
                            [f.position[0], f.position[1], f.color]
                            for f in self.Food
                        ]
                    )
                ),
            ],
            ids=list(range(sum([len(s.segments)
                                for s in self.snakes]) + len(self.Food))),
            action_masks=[
                ("move", DenseCategoricalActionMask(
                    actors=list(range(self.num_snakes)),
                ))
            ],
            reward=reward,
            done=done,
        )


# Implements VecEnv directly to allow for multiple players without requiring proper multi-agent support.
