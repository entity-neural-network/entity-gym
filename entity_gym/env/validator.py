from typing import Any, Dict, Mapping

import numpy as np
import numpy.typing as npt

from entity_gym.env.environment import (
    Action,
    ActionName,
    ActionSpace,
    CategoricalActionMask,
    CategoricalActionSpace,
    Environment,
    Observation,
    ObsSpace,
    SelectEntityActionMask,
    SelectEntityActionSpace,
)


class ValidatingEnv(Environment):
    def __init__(self, env: Environment) -> None:
        self.env = env
        self._obs_space = env.obs_space()
        self._action_space = env.action_space()

    def act(self, actions: Mapping[ActionName, Action]) -> Observation:
        obs = self.env.act(actions)
        try:
            self._validate(obs)
        except AssertionError as e:
            print(f"Invalid observation:\n{e}")
            raise e
        return obs

    def reset(self) -> Observation:
        obs = self.env.reset()
        try:
            self._validate(obs)
        except AssertionError as e:
            print(f"Invalid observation:\n{e}")
            raise e
        return obs

    def render(self, **kwargs: Any) -> npt.NDArray[np.uint8]:
        return self.env.render(**kwargs)

    def obs_space(self) -> ObsSpace:
        return self._obs_space

    def action_space(self) -> Dict[str, ActionSpace]:
        return self._action_space

    def _validate(self, obs: Observation) -> None:
        assert isinstance(
            obs, Observation
        ), f"Observation has invalid type: {type(obs)}"

        # Validate features
        for entity_type, entity_features in obs.features.items():
            assert (
                entity_type in self._obs_space.entities
            ), f"Features contain entity of type '{entity_type}' which is not in observation space: {list(self._obs_space.entities.keys())}"
            if isinstance(entity_features, np.ndarray):
                assert (
                    entity_features.dtype == np.float32
                ), f"Features of entity of type '{entity_type}' have invalid dtype: {entity_features.dtype}. Expected: {np.float32}"
                shape = entity_features.shape
                assert len(shape) == 2 and shape[1] == len(
                    self._obs_space.entities[entity_type].features
                ), f"Features of entity of type '{entity_type}' have invalid shape: {shape}. Expected: (n, {len(self._obs_space.entities[entity_type].features)})"
            else:
                for i, entity in enumerate(entity_features):
                    assert len(entity) == len(
                        self._obs_space.entities[entity_type].features
                    ), f"Features of {i}-th entity of type '{entity_type}' have invalid length: {len(entity)}. Expected: {len(self._obs_space.entities[entity_type].features)}"

            if entity_type in obs.ids:
                assert len(obs.ids[entity_type]) == len(
                    entity_features
                ), f"Length of ids of entity of type '{entity_type}' does not match length of features: {len(obs.ids[entity_type])} != {len(entity_features)}"

        # Validate global features
        if len(obs.global_features) != len(self._obs_space.global_features):
            raise AssertionError(
                f"Length of global features does not match length of global features in observation space: {len(obs.global_features)} != {len(self._obs_space.global_features)}"
            )

        # Validate ids
        previous_ids = set()
        for entity_type, entity_ids in obs.ids.items():
            assert (
                entity_type in self._obs_space.entities
            ), f"IDs contain entity of type '{entity_type}' which is not in observation space: {list(self._obs_space.entities.keys())}"
            for id in entity_ids:
                assert id not in previous_ids, f"Observation has duplicate id '{id}'"
                previous_ids.add(id)

        # Validate actions
        ids = obs.id_to_index(self._obs_space)
        for action_type, action_mask in obs.actions.items():
            assert (
                action_type in self._action_space
            ), f"Actions contain action of type '{action_type}' which is not in action space: {list(self._action_space.keys())}"
            space = self._action_space[action_type]
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
                actor_indices = obs._actor_indices(action_type, self._obs_space)
                if isinstance(mask, np.ndarray):
                    assert (
                        mask.dtype == np.bool_
                    ), f"Action of type '{action_type}' has invalid dtype: {mask.dtype}. Expected: {np.bool_}"
                    shape = mask.shape
                    if shape[0] != 0:
                        assert shape == (
                            len(actor_indices),
                            len(space.index_to_label),
                        ), f"Action of type '{action_type}' has invalid shape: {shape}. Expected: ({len(actor_indices), len(space.index_to_label)})"
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
                            space.index_to_label
                        ), f"Action of type '{action_type}' has invalid length of mask for {i}-th actor: {len(mask[i])}. Expected: {len(space.index_to_label)}"
                        assert any(
                            mask[i]
                        ), f"Action of type '{action_type}' contains invalid mask for {i}-th actor: {mask[i]}. Expected at least one possible action"

            elif isinstance(self._action_space[action_type], SelectEntityActionSpace):
                assert isinstance(
                    action_mask, SelectEntityActionMask
                ), f"Action of type '{action_type}' has invalid type: {type(action_mask)}. Expected: SelectEntityActionMask"
