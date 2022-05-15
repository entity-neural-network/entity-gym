from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import click
import numpy as np

from entity_gym.env import (
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
from entity_gym.env.environment import (
    Action,
    GlobalCategoricalAction,
    GlobalCategoricalActionMask,
    GlobalCategoricalActionSpace,
    Observation,
)
from entity_gym.env.validator import ValidatingEnv


class Agent(ABC):
    """Interface for an agent that receives observations and outputs actions."""

    @abstractmethod
    def act(self, obs: Observation) -> Tuple[Dict[str, Action], float]:
        pass


class CliRunner:
    """
    Interactively run any entity gym environment in a CLI.

    Example:

    .. code-block:: pycon

        >>> from entity_gym.runner import CliRunner
        >>> from entity_gym.examples import TreasureHunt
        >>> CliRunner(TreasureHunt()).run()
    """

    def __init__(self, env: Environment, agent: Optional[Agent] = None) -> None:
        self.env = ValidatingEnv(env)
        self.agent = agent

    def run(self, restart: bool = False) -> None:
        print_env(self.env)
        print()

        obs_space = self.env.obs_space()
        actions = self.env.action_space()
        obs = self.env.reset_filter(obs_space)
        total_reward = obs.reward
        step = 0
        while True:
            if obs.done:
                if restart:
                    obs = self.env.reset_filter(obs_space)
                    total_reward = obs.reward
                    step = 0
                else:
                    break
            if self.agent is None:
                agent_action: Optional[Dict[str, Action]] = None
                agent_prediction: Optional[float] = None
            else:
                agent_action, agent_prediction = self.agent.act(obs)
            print_obs(step, obs, total_reward, obs_space, agent_prediction)
            action: Dict[str, Action] = {}
            received_action = False
            for action_name, action_mask in obs.actions.items():
                action_def = actions[action_name]
                if isinstance(action_mask, GlobalCategoricalActionMask):
                    assert isinstance(action_def, GlobalCategoricalActionSpace)
                    if agent_action is None:
                        choices = " ".join(
                            f"{i}/{label}"
                            for i, label in enumerate(action_def.index_to_label)
                        )
                    else:
                        probs = agent_action[action_name].probs
                        assert probs is not None
                        choice_id = agent_action[action_name].index  # type: ignore
                        choices = " ".join(
                            click.style(
                                f"{i}/{label} ",
                                fg="yellow" if i == choice_id else None,
                                bold=i == choice_id,
                            )
                            + click.style(f"{100 * prob:.1f}%", fg="yellow")
                            for i, (label, prob) in enumerate(
                                zip(action_def.index_to_label, probs)
                            )
                        )

                    click.echo(
                        f"Choose "
                        + click.style(f"{action_name}", fg="green")
                        + f" ({choices})"
                    )
                    try:
                        inp = input()
                        if inp == "" and agent_action is not None:
                            choice_id = agent_action[action_name].index  # type: ignore
                        else:
                            choice_id = int(inp)
                        received_action = True
                    except KeyboardInterrupt:
                        print()
                        print("Exiting")
                        return
                    action[action_name] = GlobalCategoricalAction(
                        index=choice_id,
                        label=action_def.index_to_label[choice_id],
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
                        if agent_action is None:
                            choices = " ".join(
                                f"{i}/{label}"
                                for i, label in enumerate(action_def.index_to_label)
                            )
                        else:
                            aa = agent_action[action_name]
                            assert isinstance(aa, CategoricalAction)
                            actor_index = aa.actors.index(actor_id)
                            probs = aa.probs[actor_index]  # type: ignore
                            assert probs is not None
                            choice_id = aa.indices[actor_index]
                            choices = " ".join(
                                click.style(
                                    f"{i}/{label} ",
                                    fg="yellow" if i == choice_id else None,
                                    bold=i == choice_id,
                                )
                                + click.style(f"{100 * prob:.1f}%", fg="yellow")
                                for i, (label, prob) in enumerate(
                                    zip(action_def.index_to_label, probs)
                                )
                            )
                        click.echo(
                            f"Choose "
                            + click.style(f"{action_name}", fg="green")
                            + f" for actor {actor_id}"
                            + f" ({choices})"
                        )

                        try:
                            inp = input()
                            if inp == "" and agent_action is not None:
                                aa = agent_action[action_name]
                                assert isinstance(aa, CategoricalAction)
                                choice_id = aa.indices[actor_index]
                            else:
                                choice_id = int(inp)
                            received_action = True
                        except KeyboardInterrupt:
                            print()
                            print("Exiting")
                            return
                        if action_name not in action:
                            action[action_name] = CategoricalAction(
                                indices=np.zeros(
                                    (0, len(action_def.index_to_label)), dtype=np.int64
                                ),
                                index_to_label=action_def.index_to_label,
                                actors=[],
                            )
                        a = action[action_name]
                        assert isinstance(a, CategoricalAction)
                        a.indices = np.array(list(a.indices) + [choice_id])
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
                + ", ".join(action.index_to_label)
            )
        elif isinstance(action, SelectEntityActionSpace):
            click.echo(
                click.style(f"Select entity", fg="cyan")
                + click.style(f" {label}", fg="green")
            )
        else:
            raise ValueError(f"Unknown action type {action}")


def print_obs(
    step: int,
    obs: Observation,
    total_reward: float,
    obs_filter: ObsSpace,
    predicted_return: Optional[float] = None,
) -> None:
    click.secho(f"Step {step}", fg="white", bold=True)
    click.echo(click.style("Reward: ", fg="cyan") + f"{obs.reward}")
    click.echo(click.style("Total: ", fg="cyan") + f"{total_reward}")
    if predicted_return is not None:
        click.echo(
            click.style("Predicted return: ", fg="cyan")
            + click.style(f"{predicted_return:.3e}", fg="yellow")
        )
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


__all__ = ["CliRunner", "Agent"]
