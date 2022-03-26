from typing import Mapping, Type

import numpy as np

from entity_gym.environment.environment import (
    Action,
    ActionType,
    CategoricalActionMask,
    CategoricalActionSpace,
    Environment,
    Observation,
    SelectEntityActionMask,
    SelectEntityActionSpace,
)


def validated_env(env: Type[Environment]) -> Type[Environment]:
    obs_space = env.obs_space()
    action_space = env.action_space()

    class ValidatedEnv(env):  # type: ignore
        def act(self, action: Mapping[ActionType, Action]) -> Observation:
            obs = super().act(action)
            try:
                self.validate(obs)
            except AssertionError as e:
                print(f"Invalid observation:\n{e}")
                raise e
            return obs  # type: ignore

        def reset(self) -> Observation:
            obs = super().reset()
            try:
                self.validate(obs)
            except AssertionError as e:
                print(f"Invalid observation:\n{e}")
                raise e
            return obs  # type: ignore

        def validate(self, obs: Observation) -> None:
            assert isinstance(
                obs, Observation
            ), f"Observation has invalid type: {type(obs)}"

            # Validate features
            for entity_type, entity_features in obs.features.items():
                assert (
                    entity_type in obs_space.entities
                ), f"Features contain entity of type '{entity_type}' which is not in observation space: {list(obs_space.entities.keys())}"
                if isinstance(entity_features, np.ndarray):
                    assert (
                        entity_features.dtype == np.float32
                    ), f"Features of entity of type '{entity_type}' have invalid dtype: {entity_features.dtype}. Expected: {np.float32}"
                    shape = entity_features.shape
                    assert len(shape) == 2 and shape[1] == len(
                        obs_space.entities[entity_type].features
                    ), f"Features of entity of type '{entity_type}' have invalid shape: {shape}. Expected: (n, {len(obs_space.entities[entity_type].features)})"
                else:
                    for i, entity in enumerate(entity_features):
                        assert len(entity) == len(
                            obs_space.entities[entity_type].features
                        ), f"Features of {i}-th entity of type '{entity_type}' have invalid length: {len(entity)}. Expected: {len(obs_space.entities[entity_type].features)}"

                if entity_type in obs.ids:
                    assert len(obs.ids[entity_type]) == len(
                        entity_features
                    ), f"Length of ids of entity of type '{entity_type}' does not match length of features: {len(obs.ids[entity_type])} != {len(entity_features)}"

            # Validate ids
            previous_ids = set()
            for entity_type, entity_ids in obs.ids.items():
                assert (
                    entity_type in obs_space.entities
                ), f"IDs contain entity of type '{entity_type}' which is not in observation space: {list(obs_space.entities.keys())}"
                for id in entity_ids:
                    assert (
                        id not in previous_ids
                    ), f"Observation has duplicate id '{id}'"
                    previous_ids.add(id)

            # Validate actions
            ids = obs.id_to_index(obs_space)
            for action_type, action_mask in obs.actions.items():
                assert (
                    action_type in action_space
                ), f"Actions contain action of type '{action_type}' which is not in action space: {list(action_space.keys())}"
                space = action_space[action_type]
                if isinstance(space, CategoricalActionSpace):
                    assert isinstance(
                        action_mask, CategoricalActionMask
                    ), f"Action of type '{action_type}' has invalid type: {type(action_mask)}. Expected: CategoricalActionMask"
                    if action_mask.actor_ids is not None:
                        for id in action_mask.actor_ids:
                            assert (
                                id in ids
                            ), f"Action of type '{action_type}' contains invalid actor id {id} which is not in ids: {obs.ids}"
                    if action_mask.actor_types is not None:
                        for actor_type in action_mask.actor_types:
                            assert (
                                actor_type in obs.ids
                            ), f"Action of type '{action_type}' contains invalid actor type {actor_type} which is not in ids: {obs.ids.keys()}"
                    mask = action_mask.mask
                    if isinstance(mask, np.ndarray):
                        assert (
                            mask.dtype == np.bool_
                        ), f"Action of type '{action_type}' has invalid dtype: {mask.dtype}. Expected: {np.bool_}"
                        shape = mask.shape
                        actor_indices = obs._actor_indices(action_type, obs_space)
                        assert shape == (
                            len(actor_indices),
                            len(space.choices),
                        ), f"Action of type '{action_type}' has invalid shape: {shape}. Expected: ({len(actor_indices), len(space.choices)})"
                        unmasked_count = mask.sum(axis=1)
                        for i in range(len(unmasked_count)):
                            assert (
                                unmasked_count[i] > 0
                            ), f"Action of type '{action_type}' contains invalid mask for {i}-th actor: {mask[i]}. Expected at least one possible action"
                    elif mask is not None:
                        assert len(mask) == len(
                            actor_indices
                        ), f"Action of type '{action_type}' has invalid length: {len(mask)}. Expected: {len(actor_indices)}"
                        for i in range(len(mask)):
                            assert len(mask[i]) == len(
                                space.choices
                            ), f"Action of type '{action_type}' has invalid length of mask for {i}-th actor: {len(mask[i])}. Expected: {len(space.choices)}"
                            assert any(
                                mask[i]
                            ), f"Action of type '{action_type}' contains invalid mask for {i}-th actor: {mask[i]}. Expected at least one possible action"

                elif isinstance(action_space[action_type], SelectEntityActionSpace):
                    assert isinstance(
                        action_mask, SelectEntityActionMask
                    ), f"Action of type '{action_type}' has invalid type: {type(action_mask)}. Expected: SelectEntityActionMask"

    return ValidatedEnv
