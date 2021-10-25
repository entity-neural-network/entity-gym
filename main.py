from collections import defaultdict
import argparse
from typing import Dict, Type

from entity_gym.envs.move_to_origin import MoveToOrigin
from entity_gym.envs.cherry_pick import CherryPick
from entity_gym.envs.pick_matching_balls import PickMatchingBalls
from entity_gym.envs.minefield import Minefield
from entity_gym.envs.multi_snake import MultiSnake
from entity_gym.environment import (
    CategoricalAction,
    Environment,
    ObsFilter,
    CategoricalActionSpace,
    Action,
    Observation,
    SelectEntityAction,
    SelectEntityActionSpace,
)


def print_obs(obs: Observation, total_reward: float, obs_filter: ObsFilter):
    total_reward += obs.reward
    print(f"Reward: {obs.reward}")
    print(f"Total reward: {total_reward}")
    entity_index = 0
    for entity_type, features in obs.entities:
        for entity in range(features.shape[0]):
            print(
                f"{obs.ids[entity_index]}: {entity_type}({', '.join(map(lambda nv: nv[0] + '=' + str(nv[1]), zip(obs_filter.entity_to_feats[entity_type], features[entity, :])))})"
            )
            entity_index += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="MoveToOrigin")
    args = parser.parse_args()

    envs: Dict[str, Type[Environment]] = {
        "MoveToOrigin": MoveToOrigin,
        "CherryPick": CherryPick,
        "PickMatchingBalls": PickMatchingBalls,
        "Minefield": Minefield,
        "MultiSnake": MultiSnake,
    }
    if args.env not in envs:
        raise ValueError(
            f"Unknown environment {args.env}\nValid environments are {list(envs.keys())}"
        )
    else:
        env_cls = envs[args.env]

    print(env_cls)
    env = env_cls()
    obs_filter = ObsFilter(
        entity_to_feats={
            entity.name: entity.features for entity in env_cls.state_space()
        },
    )

    actions = env_cls.action_space_dict()
    obs = env.reset(obs_filter)
    total_reward = obs.reward
    while not obs.done:
        print_obs(obs, total_reward, obs_filter)
        action = {}
        for action_name, action_mask in obs.action_masks:
            action_def = actions[action_name]
            for actor_id in action_mask.actors:
                if isinstance(action_def, CategoricalActionSpace):
                    # Prompt user for action
                    print(f"Choose {action_name} for {actor_id}")
                    for i in range(action_def.n):
                        print(
                            f"{i}) {action_def.choice_labels[i] if action_def.choice_labels is not None else ''}"
                        )
                    choice_id = int(input())
                    if action_name not in action:
                        action[action_name] = CategoricalAction([])
                    action[action_name].actions.append((actor_id, choice_id))
                elif isinstance(action_def, SelectEntityActionSpace):
                    # Prompt user for entity
                    print(f"{action_name}")
                    print(
                        f"Selectable entities: {', '.join([str(obs.ids[i]) for i, selectable in enumerate(action_mask.mask) if selectable])}"
                    )
                    entity_id = int(input())
                    if action_name not in action:
                        action[action_name] = SelectEntityAction([])
                    action[action_name].actions.append((actor_id, entity_id))
                else:
                    raise ValueError(f"Unknown action type {action_def}")
        obs = env.act(action, obs_filter)
        total_reward += obs.reward

    print_obs(obs, total_reward, obs_filter)
    print("Episode finished")
