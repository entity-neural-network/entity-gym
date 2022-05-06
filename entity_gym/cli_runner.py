from typing import Dict

import click
import numpy as np

from entity_gym.environment import (
    Action,
    CategoricalAction,
    CategoricalActionSpace,
    Environment,
    Observation,
    ObsSpace,
    SelectEntityAction,
    SelectEntityActionMask,
    SelectEntityActionSpace,
)
from entity_gym.environment.environment import (
    GlobalCategoricalAction,
    GlobalCategoricalActionMask,
    GlobalCategoricalActionSpace,
)
from entity_gym.environment.validator import ValidatingEnv


class CliRunner:
    def __init__(self, env: Environment) -> None:
        self.env = ValidatingEnv(env)

    def run(self) -> None:
        print_env(self.env)
        print()

        obs_space = self.env.obs_space()
        actions = self.env.action_space()
        obs = self.env.reset_filter(obs_space)
        total_reward = obs.reward
        step = 0
        while not obs.done:
            print_obs(step, obs, total_reward, obs_space)
            action: Dict[str, Action] = {}
            received_action = False
            for action_name, action_mask in obs.actions.items():
                action_def = actions[action_name]
                if isinstance(action_mask, GlobalCategoricalActionMask):
                    assert isinstance(action_def, GlobalCategoricalActionSpace)
                    click.echo(
                        f"Choose "
                        + click.style(f"{action_name}", fg="green")
                        + " ("
                        + " ".join(
                            f"{i}/{label}" for i, label in enumerate(action_def.choices)
                        )
                        + ")"
                    )
                    try:
                        choice_id = int(input())
                        received_action = True
                    except KeyboardInterrupt:
                        print()
                        print("Exiting")
                        return
                    action[action_name] = GlobalCategoricalAction(
                        index=choice_id,
                        choice=action_def.choices[choice_id],
                    )
                    continue
                elif action_mask.actor_ids is not None:
                    actor_ids = action_mask.actor_ids
                elif action_mask.actor_types is not None:
                    actor_ids = [
                        id for atype in action_mask.actor_types for id in obs.ids[atype]
                    ]
                else:
                    actor_ids = obs.index_to_id(obs_space)

                print()
                for actor_id in actor_ids:
                    if isinstance(action_def, CategoricalActionSpace):
                        # Prompt user for action
                        click.echo(
                            f"Choose "
                            + click.style(f"{action_name}", fg="green")
                            + f" for actor {actor_id}"
                            + " ("
                            + " ".join(
                                f"{i}/{label}"
                                for i, label in enumerate(action_def.choices)
                            )
                            + ")"
                        )
                        try:
                            choice_id = int(input())
                            received_action = True
                        except KeyboardInterrupt:
                            print()
                            print("Exiting")
                            return
                        if action_name not in action:
                            action[action_name] = CategoricalAction(
                                actions=np.zeros(
                                    (0, len(action_def.choices)), dtype=np.int64
                                ),
                                actors=[],
                            )
                        a = action[action_name]
                        assert isinstance(a, CategoricalAction)
                        a.actions = np.array(list(a.actions) + [choice_id])
                        a.actors = list(a.actors) + [actor_id]
                    elif isinstance(action_def, SelectEntityActionSpace):
                        assert isinstance(action_mask, SelectEntityActionMask)
                        # Prompt user for entity
                        click.echo(
                            f"Choose "
                            + click.style(f"{action_name}", fg="green")
                            + f" for actor {actor_id}"
                        )
                        if action_mask.actee_ids is not None:
                            print(
                                f"Selectable entities: {', '.join([str(id) for id in action_mask.actee_ids])}"
                            )
                        elif action_mask.actee_types is not None:
                            print(
                                f"Selectable entity types: {', '.join([str(id) for id in action_mask.actee_types])}"
                            )
                        else:
                            print("Selectable entities: all")

                        try:
                            entity_id = int(input())
                        except KeyboardInterrupt:
                            print()
                            print("Exiting")
                            return
                        received_action = True
                        if action_name not in action:
                            action[action_name] = SelectEntityAction([], [])
                        a = action[action_name]
                        assert isinstance(a, SelectEntityAction)
                        a.actors = list(a.actors) + [actor_id]
                        a.actees = list(a.actees) + [entity_id]
                    else:
                        raise ValueError(f"Unknown action type {action_def}")
            if not received_action:
                try:
                    input("Press ENTER to continue, CTRL-C to exit")
                except KeyboardInterrupt:
                    print()
                    print("Exiting")
                    return
            obs = self.env.act_filter(action, obs_space)
            total_reward += obs.reward
            step += 1

        print_obs(step, obs, total_reward, obs_space)
        click.secho("Episode finished", fg="green")


def print_env(env: ValidatingEnv) -> None:
    click.secho(f"Environment: {env.env.__class__.__name__}", fg="white", bold=True)
    obs = env.obs_space()
    if len(obs.global_features) > 0:
        click.echo(
            click.style("Global features: ", fg="cyan") + ", ".join(obs.global_features)
        )
    for label, entity in obs.entities.items():
        click.echo(
            click.style(f"Entity ", fg="cyan")
            + click.style(f"{label}", fg="green")
            + click.style(f": " if len(entity.features) > 0 else "", fg="cyan")
            + ", ".join(entity.features)
        )
    acts = env.action_space()
    for label, action in acts.items():
        if isinstance(action, CategoricalActionSpace) or isinstance(
            action, GlobalCategoricalActionSpace
        ):
            click.echo(
                click.style(f"Categorical", fg="cyan")
                + click.style(f" {label}", fg="green")
                + click.style(f": ", fg="cyan")
                + ", ".join(action.choices)
            )
        elif isinstance(action, SelectEntityActionSpace):
            click.echo(
                click.style(f"Select entity", fg="cyan")
                + click.style(f" {label}", fg="green")
            )
        else:
            raise ValueError(f"Unknown action type {action}")


def print_obs(
    step: int, obs: Observation, total_reward: float, obs_filter: ObsSpace
) -> None:
    click.secho(f"Step {step}", fg="white", bold=True)
    click.echo(click.style("Reward: ", fg="cyan") + f"{obs.reward}")
    click.echo(click.style("Total: ", fg="cyan") + f"{total_reward}")
    if len(obs_filter.global_features) > 0:
        click.echo(
            click.style("Global features: ", fg="cyan")
            + ", ".join(
                f"{label}={value}"
                for label, value in zip(obs_filter.global_features, obs.global_features)
            )
        )
    if len(obs_filter.entities) > 0:
        click.echo(click.style("Entities", fg="cyan"))
    entity_index = 0
    for entity_type, features in obs.features.items():
        for entity in range(len(features)):
            if entity_type in obs.ids:
                id = f" (id={obs.ids[entity_type][entity]})"
            else:
                id = ""
            rendered = (
                click.style(entity_type, fg="green")
                + "("
                + ", ".join(
                    map(
                        lambda nv: nv[0] + "=" + str(nv[1]),
                        zip(
                            obs_filter.entities[entity_type].features,
                            features[entity],
                        ),
                    )
                )
                + ")"
            )
            print(f"{entity_index} {rendered}{id}")
            entity_index += 1
