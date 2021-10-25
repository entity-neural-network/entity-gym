from abc import ABC, abstractclassmethod, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type
import numpy as np


@dataclass
class ActionSpace(ABC):
    name: str


@dataclass
class CategoricalActionSpace(ActionSpace):
    # TODO: get rid of n, just have a `choices` list?
    n: int
    choice_labels: Optional[List[str]] = None


@dataclass
class SelectEntityActionSpace(ActionSpace):
    pass


@dataclass
class ActionMask(ABC):
    # Indices of entities which can perform the action on this time step.
    actors: Sequence[int]


@dataclass
class DenseCategoricalActionMask(ActionMask):
    # Action mask with dimensions (len(actors), n_choices). Each row is a
    # binary vector indicating which choices are allowed for the corresponding
    # actor.
    mask: Optional[np.ndarray] = None


@dataclass
class DenseSelectEntityActionMask(ActionMask):
    # Action mask with dimensions (len(actors), len(entities)). Each row is a
    # binary vector indicating which entities can be selected by the corresponding
    # actor.
    mask: Optional[np.ndarray]


EntityID = Any


@dataclass
class Observation:
    # Maps each entity type to an array with the features for each observed entity of that type.
    entities: Sequence[Tuple[str, np.ndarray]]
    # Maps each entity index to an opaque identifier used by the environment to
    # identify that entity.
    ids: Sequence[EntityID]
    # Maps each action type to an action mask.
    action_masks: Sequence[Tuple[str, ActionMask]]
    reward: float
    done: bool
    info: Any = None


@dataclass
class Entity:
    name: str
    features: List[str]


@dataclass
class ObsFilter:
    entity_to_feats: Dict[str, List[str]]


class Action(ABC):
    pass


@dataclass
class CategoricalAction(Action):
    # Maps each actor to the index of the chosen action.
    actions: Sequence[Tuple[EntityID, int]]


@dataclass
class SelectEntityAction(Action):
    # Maps each actor to the index of the selected entity.
    actions: Sequence[Tuple[EntityID, EntityID]]


class Environment(ABC):
    """
    Abstraction over reinforcement learning environments with observations based on structured lists of entities.

    As a simple hack to support basic multi-agent environments with parallel observations and actions,
    methods may return lists of observations and accept lists of actions.
    This should be replaced by a more general multi-agent environment interface in the future.
    """

    @abstractclassmethod
    def state_space(cls) -> List[Entity]:
        """
        Returns a list of entity types that can be observed in the environment.
        """
        raise NotImplementedError

    @abstractclassmethod
    def action_space(cls) -> List[ActionSpace]:
        """
        Returns a list of actions that can be performed in the environment.
        """
        raise NotImplementedError

    @abstractmethod
    def _reset(self) -> Observation:
        """
        Resets the environment and returns the initial observation.
        """
        raise NotImplementedError

    @abstractmethod
    def _act(self, action: Dict[str, Action]) -> Observation:
        """
        Performs the given action and returns the resulting observation.

        Args:
            action: Maps the name of each action type to the action to perform.
        """
        raise NotImplementedError

    @classmethod
    def state_space_dict(cls) -> Dict[str, Entity]:
        return {e.name: e for e in cls.state_space()}

    @classmethod
    def action_space_dict(cls) -> Dict[str, ActionSpace]:
        return {a.name: a for a in cls.action_space()}

    def reset(self, obs_filter: ObsFilter) -> Observation:
        return self.__class__.filter_obs(self._reset(), obs_filter)

    def act(self, action: Action, obs_filter: ObsFilter) -> Observation:
        return self.__class__.filter_obs(self._act(action), obs_filter)

    @classmethod
    def filter_obs(cls, obs: Observation, obs_filter: ObsFilter) -> Observation:
        selectors = cls._compile_feature_filter(obs_filter)
        entities = []
        for entity_name, entity_features in obs.entities:
            entities.append(
                (entity_name, entity_features[:, selectors[entity_name]]))
        return Observation(entities, obs.ids, obs.action_masks, obs.reward, obs.done)

    @classmethod
    def _compile_feature_filter(cls, obs_filter: ObsFilter) -> Dict[str, np.ndarray]:
        entity_dict = cls.state_space_dict()
        feature_selection = {}
        for entity_name, entity_features in obs_filter.entity_to_feats.items():
            entity = entity_dict[entity_name]
            feature_selection[entity_name] = np.array(
                [entity.features.index(f) for f in entity_features],
                dtype=np.int32
            )
        return feature_selection


class VecEnv(ABC):
    @abstractclassmethod
    def env_cls(cls) -> Type[Environment]:
        """
        Returns the class of the underlying environment.
        """
        raise NotImplementedError

    @abstractmethod
    def _reset(self) -> List[Observation]:
        raise NotImplementedError

    @abstractmethod
    def _act(self, actions: List[Dict[str, Action]]) -> List[Observation]:
        raise NotImplementedError

    def reset(self, obs_config: ObsFilter) -> List[Observation]:
        obs = self._reset()
        return [self.__class__.env_cls().filter_obs(o, obs_config) for o in obs]

    def act(self, actions: List[Action], obs_filter: ObsFilter) -> List[Observation]:
        obs = self._act(actions)
        return [self.__class__.env_cls().filter_obs(o, obs_filter) for o in obs]


class VecEnvWrapper(VecEnv):
    def __init__(self, envs: List[Environment]):
        self.envs = envs
        self.cls = self.envs[0].__class__

    @classmethod
    def env_cls(cls) -> Type[Environment]:
        return cls.cls

    def _reset(self) -> List[Observation]:
        return [e._reset() for e in self.envs]

    def _act(self, actions: List[Dict[str, Action]]) -> List[Observation]:
        return [e._act(a) for e, a in zip(self.envs, actions)]
