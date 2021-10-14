from collections import defaultdict
import argparse

from entity_gym.envs.move_to_origin import MoveToOrigin
from entity_gym.envs.cherry_pick import CherryPick
from entity_gym.envs.pick_matching_balls import PickMatchingBalls
from entity_gym.envs.minefield import Minefield
from entity_gym.environment import ObsConfig, Categorical, Action, Observation, SelectEntity


def print_obs(obs: Observation, total_reward: float):
    total_reward += obs.reward
    print(f"Reward: {obs.reward}")
    print(f"Total reward: {total_reward}")
    entity_index = 0
    for entity_type, features in obs.entities:
        for entity in range(features.shape[0]):
            print(
                f"{obs.ids[entity_index]}: {entity_type}({', '.join(map(lambda nv: nv[0] + '=' + str(nv[1]), zip(obs_config.entities[entity_type], features[entity, :])))})")
            entity_index += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="MoveToOrigin")
    args = parser.parse_args()

    if args.env == "MoveToOrigin":
        env_cls = MoveToOrigin
    elif args.env == "CherryPick":
        env_cls = CherryPick
    elif args.env == "PickMatchingBalls":
        env_cls = PickMatchingBalls
    elif args.env == "Minefield":
        env_cls = Minefield
    else:
        raise ValueError("Unknown environment: {}".format(args.env))

    print(env_cls)
    env = env_cls()
    obs_config = ObsConfig(
        entities={
            entity.name: entity.features for entity in env_cls.entities()},
    )

    total_reward = 0
    actions = env_cls.action_space_dict()
    obs = env.reset(obs_config)
    while not obs.done:
        print_obs(obs, total_reward)
        chosen_actions = defaultdict(list)
        for action_name, action_mask in obs.action_masks:
            action_def = actions[action_name]
            for actor_id in action_mask.actors:
                if isinstance(action_def, Categorical):
                    # Prompt user for action
                    print(f"Choose {action_name}")
                    for i in range(action_def.n):
                        print(f"{i}: {action_def.choice_labels[i]}")
                    choice_id = int(input())
                    chosen_actions[action_name].append((actor_id, choice_id))
                elif isinstance(action_def, SelectEntity):
                    # Prompt user for entity
                    print(f"{action_name}")
                    print(
                        f"Selectable entities: {', '.join([str(obs.ids[i]) for i, selectable in enumerate(action_mask.mask) if selectable])}")
                    entity_id = int(input())
                    chosen_actions[action_name].append((actor_id, entity_id))
                else:
                    raise ValueError(f"Unknown action type {action_def}")
        obs = env.act(Action(chosen_actions), obs_config)

    print_obs(obs, total_reward)
    print("Episode finished")
