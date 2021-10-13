from collections import defaultdict
from entity_gym.envs.move_to_origin import MoveToOrigin
from entity_gym.environment import ObsConfig, Categorical, Action


if __name__ == "__main__":
    env = MoveToOrigin()
    obs_config = ObsConfig(
        entities={
            entity.name: entity.features for entity in MoveToOrigin.entities()},
    )

    total_reward = 0
    actions = MoveToOrigin.action_space_dict()
    obs = env.reset(obs_config)
    while not obs.done:
        total_reward += obs.reward
        print(f"Reward: {obs.reward}")
        print(f"Total reward: {total_reward}")
        entity_index = 0
        for entity_type, features in obs.entities:
            for entity in range(features.shape[0]):
                print(
                    f"{obs.ids[entity_index]}: {entity_type}({', '.join(map(lambda nv: nv[0] + '=' + str(nv[1]), zip(obs_config.entities[entity_type], features[entity, :])))})")
                entity_index += 1

        chosen_actions = defaultdict(list)
        for action_name, actors in obs.actions:
            action_def = actions[action_name]
            for actor_id in actors:
                if isinstance(action_def, Categorical):
                    # Prompt user for action
                    print(f"Choose {action_name}")
                    for i in range(action_def.n):
                        print(f"{i}: {action_def.choice_labels[i]}")
                    choice_id = int(input())
                    chosen_actions[action_name].append((actor_id, choice_id))
                else:
                    raise ValueError(f"Unknown action type {action_def}")
        obs = env.act(Action(chosen_actions), obs_config)
000
