from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Type, Union
import numpy as np


@dataclass
class CategoricalActionSpace:
    choices: List[str]


@dataclass
class SelectEntityActionSpace:
    pass


ActionSpace = Union[CategoricalActionSpace, SelectEntityActionSpace]


@dataclass
class ActionMask(ABC):
    """
    Base class for action masks that specify what agents can perform a particular action.
    """

    actors: Sequence[int]
    """A list of agents that can perform the action."""


@dataclass
class DenseCategoricalActionMask(ActionMask):
    """
    Action mask for categorical action that specifies which agents can perform the action,
    and includes a dense mask that further contraints the choices available to each agent.
    """

    mask: Optional[np.ndarray] = None
    """
    A boolean array of shape (len(actors), len(choices)). If mask[i, j] is True, then
    agent i can perform action j.
    """


@dataclass
class DenseSelectEntityActionMask(ActionMask):
    """
    Action mask for select entity action that specifies which agents can perform the action,
    and includes a dense mask that further contraints what other entities can be selected by
    each actor.
    """

    mask: Optional[np.ndarray]
    """
    An boolean array of shape (len(actors), len(entities)). If mask[i, j] is True, then
    agent i can select entity j.
    """


EntityID = Any


@dataclass
class Observation:
    entities: Dict[str, np.ndarray]
    """Maps each entity type to an array with the features for each observed entity of that type."""
    ids: Sequence[EntityID]
    """
    Maps each entity index to an opaque identifier used by the environment to
    identify that entity.
    """
    action_masks: Mapping[str, ActionMask]
    """Maps each action to an action mask."""
    reward: float
    done: bool
    info: Any = None


@dataclass
class Entity:
    features: List[str]


@dataclass
class ObsFilter:
    """
    Allows filtering observations to only include a subset of entities and features.
    """

    entity_to_feats: Dict[str, List[str]]


@dataclass
class CategoricalAction:
    actions: List[Tuple[EntityID, int]]
    """Maps each actor to the index of the chosen action."""


@dataclass
class SelectEntityAction:
    actions: List[Tuple[EntityID, EntityID]]
    """Maps each actor to the entity they selected."""


Action = Union[CategoricalAction, SelectEntityAction]


class Environment(ABC):
    """
    Abstraction over reinforcement learning environments with observations based on structured lists of entities.

    As a simple hack to support basic multi-agent environments with parallel observations and actions,
    methods may return lists of observations and accept lists of actions.
    This should be replaced by a more general multi-agent environment interface in the future.
    """

    @classmethod
    @abstractmethod
    def state_space(cls) -> Dict[str, Entity]:
        """
        Returns a dictionary mapping the name of observable entities to their type.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def action_space(cls) -> Dict[str, ActionSpace]:
        """
        Returns a dictionary mapping the name of actions to their action space.
        """
        raise NotImplementedError

    @abstractmethod
    def _reset(self) -> Observation:
        """
        Resets the environment and returns the initial observation.
        """
        raise NotImplementedError

    @abstractmethod
    def _act(self, action: Mapping[str, Action]) -> Observation:
        """
        Performs the given action and returns the resulting observation.

        Args:
            action: Maps the name of each action type to the action to perform.
        """
        raise NotImplementedError

    def reset(self, obs_filter: ObsFilter) -> Observation:
        return self.__class__.filter_obs(self._reset(), obs_filter)

    def act(self, action: Mapping[str, Action], obs_filter: ObsFilter) -> Observation:
        return self.__class__.filter_obs(self._act(action), obs_filter)

    @classmethod
    def filter_obs(cls, obs: Observation, obs_filter: ObsFilter) -> Observation:
        selectors = cls._compile_feature_filter(obs_filter)
        entities = {
            entity_name: entity_features[:, selectors[entity_name]]
            for entity_name, entity_features in obs.entities.items()
        }
        return Observation(entities, obs.ids, obs.action_masks, obs.reward, obs.done)

    @classmethod
    def _compile_feature_filter(cls, obs_filter: ObsFilter) -> Dict[str, np.ndarray]:
        entity_dict = cls.state_space()
        feature_selection = {}
        for entity_name, entity_features in obs_filter.entity_to_feats.items():
            entity = entity_dict[entity_name]
            feature_selection[entity_name] = np.array(
                [entity.features.index(f) for f in entity_features], dtype=np.int32
            )
        return feature_selection


class VecEnv(ABC):
    @classmethod
    @abstractmethod
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

    def act(
        self, actions: List[Dict[str, Action]], obs_filter: ObsFilter
    ) -> List[Observation]:
        obs = self._act(actions)
        return [self.__class__.env_cls().filter_obs(o, obs_filter) for o in obs]


class EnvList(VecEnv):
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
