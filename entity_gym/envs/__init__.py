from typing import Dict, Type
from entity_gym.environment import Environment

from entity_gym.envs.move_to_origin import MoveToOrigin
from entity_gym.envs.cherry_pick import CherryPick
from entity_gym.envs.pick_matching_balls import PickMatchingBalls
from entity_gym.envs.minefield import Minefield
from entity_gym.envs.multi_snake import MultiSnake
from entity_gym.envs.multi_armed_bandit import MultiArmedBandit
from entity_gym.envs.not_hotdog import NotHotdog
from entity_gym.envs.xor import Xor
from entity_gym.envs.count import Count

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
}
