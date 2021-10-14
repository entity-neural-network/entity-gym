from abc import ABC
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np


@dataclass
class ActionSpace:
    name: str


@dataclass
class Categorical(ActionSpace):
    n: int
    choice_labels: Optional[List[str]] = None


@dataclass
class SelectEntity(ActionSpace):
    pass


@dataclass
class ObsConfig:
    entities: Dict[str, List[str]]


@dataclass
class ActionMask(ABC):
    # Indices of entities which can perform the action on this time step.
    actors: Sequence[int]
    # Action mask with dimensions (len(actors), n_choices) if this is a categorical action,
    # or (len(actors), len(entities)) if this is a select object action.
    mask: Optional[np.ndarray]
    # TODO: also support sparse action masks for select object actions that specify the actees.


@dataclass
class Observation:
    # Maps each entity type to an array with features for each entity of that type.
    entities: Sequence[Tuple[str, np.ndarray]]
    # Maps each entity index to an identifier for the entity.
    ids: Sequence[int]
    # Maps each action type to an action mask.
    action_masks: Sequence[Tuple[str, ActionMask]]
    reward: float
    done: bool


@dataclass
class Entity:
    name: str
    features: List[str]


@dataclass
class Action:
    # Maps each action type to a list of tuples.
    # The first element of each tuple is the id of the entity that performs the action.
    # If this is a categorical action, the second element is the index of the action.
    # If this is a select object action, the second element is the id of the selected object.
    chosen_actions: Dict[str, Sequence[Tuple[int, int]]]


class Environment(ABC):
    @classmethod
    def entities(cls) -> List[Entity]:
        raise NotImplementedError

    @classmethod
    def action_space(cls) -> List[ActionSpace]:
        raise NotImplementedError

    @classmethod
    def entity_dict(cls) -> Dict[str, Entity]:
        return {e.name: e for e in cls.entities()}

    @classmethod
    def action_space_dict(cls) -> Dict[str, ActionSpace]:
        return {a.name: a for a in cls.action_space()}

    # TODO: cache this
    @classmethod
    def _compile_feature_selection(cls, obs_config: ObsConfig) -> Dict[str, np.ndarray]:
        entity_dict = cls.entity_dict()
        feature_selection = {}
        for entity_name, entity_features in obs_config.entities.items():
            entity = entity_dict[entity_name]
            feature_selection[entity_name] = np.array(
                [entity.features.index(f) for f in entity_features],
                dtype=np.int32
            )
        return feature_selection

    # TODO: generic way of filtering selected entities/features so environments don't have to implement this (but still have the option to do so for potentially better efficiency).
    def filter_obs(self, obs: Observation, obs_config: ObsConfig) -> Observation:
        entities = []
        selectors = self.__class__._compile_feature_selection(obs_config)
        for entity_name, entity_features in obs.entities:
            entities.append(
                (entity_name, entity_features[:, selectors[entity_name]]))
        return Observation(entities, obs.ids, obs.action_masks, obs.reward, obs.done)

    def simple_reset(self) -> Observation:
        raise NotImplementedError

    def reset(self, obs_config: ObsConfig) -> Observation:
        raise NotImplementedError

    def act(self, action: Action, obs_config: ObsConfig) -> Observation:
        raise NotImplementedError
