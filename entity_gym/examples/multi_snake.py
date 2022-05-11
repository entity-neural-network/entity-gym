import random
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Mapping, Tuple

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

    def __init__(
        self,
        board_size: int = 10,
        num_snakes: int = 2,
        num_players: int = 1,
        max_snake_length: int = 11,
        max_steps: int = 180,
    ):
        """
        :param num_players: number of players
        :param board_size: size of the board
        :param num_snakes: number of snakes per player
        """
        assert num_snakes < 10, f"num_snakes must be less than 10, got {num_snakes}"
        self.board_size = board_size
        self.num_snakes = num_snakes
        self.num_players = num_players
        self.max_snake_length = max_snake_length
        self.snakes: List[Snake] = []
        self.food: List[Food] = []
        self.game_over = False
        self.last_scores = [0] * self.num_players
        self.scores = [0] * self.num_players
        self.step = 0
        self.max_steps = max_steps

    def obs_space(cls) -> ObsSpace:
        return ObsSpace(
            global_features=["step"],
            entities={
                "SnakeHead": Entity(["x", "y", "color"]),
                "SnakeBody": Entity(["x", "y", "color"]),
                "Food": Entity(["x", "y", "color"]),
            },
        )

    def action_space(cls) -> Dict[str, ActionSpace]:
        return {
            "move": CategoricalActionSpace(["up", "down", "left", "right"]),
        }

    def _spawn_snake(self, color: int) -> None:
        while True:
            x = random.randint(0, self.board_size - 1)
            y = random.randint(0, self.board_size - 1)
            if any(
                (x, y) == (sx, sy) for snake in self.snakes for sx, sy in snake.segments
            ):
                continue
            self.snakes.append(Snake(color, [(x, y)]))
            break

    def _spawn_food(self, color: int) -> None:
        while True:
            x = random.randint(0, self.board_size - 1)
            y = random.randint(0, self.board_size - 1)
            if any((x, y) == (f.position[0], f.position[1]) for f in self.food) or any(
                (x, y) == (sx, sy) for snake in self.snakes for sx, sy in snake.segments
            ):
                continue
            self.food.append(Food(color, (x, y)))
            break

    def reset(self) -> Observation:
        self.snakes = []
        self.food = []
        self.game_over = False
        self.last_scores = [0] * self.num_players
        self.scores = [0] * self.num_players
        self.step = 0
        for i in range(self.num_snakes):
            self._spawn_snake(i)
        for i in range(self.num_snakes):
            self._spawn_food(i)
        return self._observe()

    def act(self, actions: Mapping[str, Action]) -> Observation:
        game_over = False
        self.step += 1
        move_action = actions["move"]
        self.last_scores = deepcopy(self.scores)
        food_to_spawn = []
        assert isinstance(move_action, CategoricalAction)
        for id, move in zip(move_action.actors, move_action.indices):
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
            if any((x, y) == (sx, sy) for s in self.snakes for sx, sy in s.segments):
                game_over = True
            ate_food = False
            snake.segments.append((x, y))
            for i in range(len(self.food)):
                if self.food[i].position == (x, y):
                    if self.food[i].color != snake.color:
                        game_over = True
                    elif len(snake.segments) <= self.max_snake_length:
                        ate_food = True
                        self.scores[id // self.num_snakes] += (
                            1.0 / (self.max_snake_length - 1) / self.num_snakes
                        )
                    self.food.pop(i)
                    # Don't spawn food immediately since it might spawn in front of another snake that hasn't moved yet
                    food_to_spawn.append(snake.color)
                    break
            if not ate_food:
                snake.segments = snake.segments[1:]
        for player in range(self.num_players):
            snakes_per_player = self.num_snakes // self.num_players
            if all(
                len(s.segments) >= self.max_snake_length
                for s in self.snakes[
                    player * snakes_per_player : (player + 1) * snakes_per_player
                ]
            ):
                game_over = True
        for color in food_to_spawn:
            self._spawn_food(color)
        if self.step >= self.max_steps:
            game_over = True
        return self._observe(done=game_over)

    def _observe(self, done: bool = False, player: int = 0) -> Observation:
        color_offset = player * (self.num_snakes // self.num_players)

        def cycle_color(color: int) -> int:
            return (color - color_offset) % self.num_snakes

        return Observation(
            global_features=[self.step],
            features={
                "SnakeHead": np.array(
                    [
                        [
                            s.segments[-1][0],
                            s.segments[-1][1],
                            cycle_color(s.color),
                        ]
                        for s in self.snakes
                    ],
                    dtype=np.float32,
                ),
                "SnakeBody": np.array(
                    [
                        [sx, sy, cycle_color(snake.color)]
                        for snake in self.snakes
                        for sx, sy in snake.segments[:-1]
                    ],
                    dtype=np.float32,
                ).reshape(-1, 3),
                "Food": np.array(
                    [
                        [
                            f.position[0],
                            f.position[1],
                            cycle_color(f.color),
                        ]
                        for f in self.food
                    ],
                    dtype=np.float32,
                ),
            },
            ids={
                "SnakeHead": list(range(self.num_snakes)),
            },
            actions={
                "move": CategoricalActionMask(
                    actor_types=["SnakeHead"],
                ),
            },
            reward=self.scores[player] - self.last_scores[player],
            done=done,
        )
