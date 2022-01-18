import argparse
from typing import Dict, Type

from entity_gym.environment import (
    CategoricalAction,
    DenseSelectEntityActionMask,
    Environment,
    ObsSpace,
    CategoricalActionSpace,
    Action,
    Observation,
    SelectEntityAction,
    SelectEntityActionSpace,
)
from entity_gym.examples import ENV_REGISTRY


def print_obs(obs: Observation, total_reward: float, obs_filter: ObsSpace) -> None:
    print(f"Reward: {obs.reward}")
    print(f"Total reward: {total_reward}")
    entity_index = 0
    for entity_type, features in obs.entities.items():
        for entity in range(features.shape[0]):
            print(
                f"{obs.ids[entity_index]}: {entity_type}({', '.join(map(lambda nv: nv[0] + '=' + str(nv[1]), zip(obs_filter.entities[entity_type].features, features[entity, :])))})"
            )
            entity_index += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="MoveToOrigin")
    args = parser.parse_args()

    envs = ENV_REGISTRY
    if args.env not in envs:
        raise ValueError(
            f"Unknown environment {args.env}\nValid environments are {list(envs.keys())}"
        )
    else:
        env_cls = envs[args.env]

    print(env_cls)
    env = env_cls()
    obs_space = env_cls.obs_space()
    actions = env_cls.action_space()
    obs = env.reset(obs_space)
    total_reward = obs.reward
    while not obs.done:
        print_obs(obs, total_reward, obs_space)
        action: Dict[str, Action] = {}
        for action_name, action_mask in obs.action_masks.items():
            action_def = actions[action_name]
            for actor_id in action_mask.actors:
                if isinstance(action_def, CategoricalActionSpace):
                    # Prompt user for action
                    print(f"Choose {action_name} for {actor_id}")
                    for i, label in enumerate(action_def.choices):
                        print(f"{i}) {label}")
                    choice_id = int(input())
                    if action_name not in action:
                        action[action_name] = CategoricalAction([])
                    action[action_name].actions.append((actor_id, choice_id))
                elif isinstance(action_def, SelectEntityActionSpace):
                    assert isinstance(action_mask, DenseSelectEntityActionMask)
                    # Prompt user for entity
                    print(f"{action_name}")
                    if action_mask.mask is not None:
                        print(
                            f"Selectable entities: {', '.join([str(obs.ids[i]) for i, selectable in enumerate(action_mask.mask) if selectable])}"
                        )
                    else:
                        print("Selectable entities: all")
                    entity_id = int(input())
                    if action_name not in action:
                        action[action_name] = SelectEntityAction([])
                    action[action_name].actions.append((actor_id, entity_id))
                else:
                    raise ValueError(f"Unknown action type {action_def}")
        obs = env.act(action, obs_space)
        total_reward += obs.reward

    print_obs(obs, total_reward, obs_space)
    print("Episode finished")
