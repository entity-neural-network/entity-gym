from typing import Dict, Type

from entity_gym.env import Environment
from entity_gym.examples.cherry_pick import CherryPick
from entity_gym.examples.count import Count
from entity_gym.examples.floor_is_lava import FloorIsLava
from entity_gym.examples.minefield import Minefield
from entity_gym.examples.minesweeper import MineSweeper
from entity_gym.examples.move_to_origin import MoveToOrigin
from entity_gym.examples.multi_armed_bandit import MultiArmedBandit
from entity_gym.examples.multi_snake import MultiSnake
from entity_gym.examples.not_hotdog import NotHotdog
from entity_gym.examples.pick_matching_balls import PickMatchingBalls
from entity_gym.examples.rock_paper_scissors import RockPaperScissors
from entity_gym.examples.tutorial import TreasureHunt
from entity_gym.examples.xor import Xor

ENV_REGISTRY: Dict[str, Type[Environment]] = {
    "MoveToOrigin": MoveToOrigin,
    "CherryPick": CherryPick,
    "PickMatchingBalls": PickMatchingBalls,
    "Minefield": Minefield,
    "MultiSnake": MultiSnake,
    "MultiArmedBandit": MultiArmedBandit,
    "NotHotdog": NotHotdog,
    "Xor": Xor,
    "Count": Count,
    "FloorIsLava": FloorIsLava,
    "MineSweeper": MineSweeper,
    "RockPaperScissors": RockPaperScissors,
    "TreasureHunt": TreasureHunt,
}

__all__ = [
    "MoveToOrigin",
    "CherryPick",
    "PickMatchingBalls",
    "Minefield",
    "MultiSnake",
    "MultiArmedBandit",
    "NotHotdog",
    "Xor",
    "Count",
    "FloorIsLava",
    "MineSweeper",
    "RockPaperScissors",
    "TreasureHunt",
]
