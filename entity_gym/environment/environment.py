from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)
import numpy as np
import numpy.typing as npt


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

    actors: npt.NDArray[np.int64]
    """
    The indices of the entities that can perform the action.
    """


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

    actees: npt.NDArray[np.int64]
    mask: Optional[np.ndarray] = None
    """
    An boolean array of shape (len(actors), len(entities)). If mask[i, j] is True, then
    agent i can select entity j.
    """


EntityID = Any


@dataclass
class EpisodeStats:
    length: int
    total_reward: float


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
    end_of_episode_info: Optional[EpisodeStats] = None


@dataclass
class Entity:
    features: List[str]


@dataclass
class ObsSpace:
    entities: Dict[str, Entity]


@dataclass
class CategoricalAction:
    # TODO: figure out best representation
    actions: List[Tuple[EntityID, int]]
    # actions: np.ndarray
    """
    Maps each actor to the index of the chosen action.
    Given `Observation` obs and `ActionMask` mask, the `EntityID`s of the corresponding
    actors are given as `obs.ids[mask.actors]`.
    """


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
    def obs_space(cls) -> ObsSpace:
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

    def reset(self, obs_filter: ObsSpace) -> Observation:
        return self.__class__.filter_obs(self._reset(), obs_filter)

    def act(self, action: Mapping[str, Action], obs_filter: ObsSpace) -> Observation:
        return self.__class__.filter_obs(self._act(action), obs_filter)

    def close(self) -> None:
        pass

    @classmethod
    def filter_obs(cls, obs: Observation, obs_filter: ObsSpace) -> Observation:
        selectors = cls._compile_feature_filter(obs_filter)
        entities = {
            entity_name: entity_features[:, selectors[entity_name]].reshape(
                entity_features.shape[0], len(selectors[entity_name])
            )
            for entity_name, entity_features in obs.entities.items()
        }
        return Observation(
            entities,
            obs.ids,
            obs.action_masks,
            obs.reward,
            obs.done,
            obs.end_of_episode_info,
        )

    @classmethod
    def _compile_feature_filter(cls, obs_space: ObsSpace) -> Dict[str, np.ndarray]:
        obs_space = cls.obs_space()
        feature_selection = {}
        for entity_name, entity in obs_space.entities.items():
            feature_selection[entity_name] = np.array(
                [entity.features.index(f) for f in entity.features], dtype=np.int32
            )
        return feature_selection

    def env_cls(self) -> Type["Environment"]:
        return self.__class__
