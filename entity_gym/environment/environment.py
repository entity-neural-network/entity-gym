from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    Generator,
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

EntityID = Any
EntityType = str
ActionType = str


@dataclass
class CategoricalActionSpace:
    choices: List[str]


@dataclass
class SelectEntityActionSpace:
    pass


ActionSpace = Union[CategoricalActionSpace, SelectEntityActionSpace]


@dataclass
class CategoricalActionMask:
    """
    Action mask for categorical action that specifies which agents can perform the action,
    and includes a dense mask that further constraints the choices available to each agent.
    """

    actor_ids: Optional[Sequence[EntityID]] = None
    """
    The ids of the entities that can perform the action.
    If None, all entities can perform the action.
    """

    actor_types: Optional[Sequence[EntityType]] = None
    """
    The types of the entities that can perform the action.
    If None, all entities can perform the action.
    """

    mask: Union[Sequence[Sequence[bool]], np.ndarray, None] = None
    """
    A boolean array of shape (len(actor_ids), len(choices)). If mask[i, j] is True, then
    agent with id actor_ids[i] can perform action j.
    """

    def __post_init__(self) -> None:
        assert (
            self.actor_ids is None or self.actor_types is None
        ), "Only one of actor_ids or actor_types can be specified"


@dataclass
class SelectEntityActionMask:
    """
    Action mask for select entity action that specifies which agents can perform the action,
    and includes a dense mask that further constraints what other entities can be selected by
    each actor.
    """

    actor_ids: Optional[Sequence[EntityID]] = None
    """
    The ids of the entities that can perform the action.
    If None, all entities can perform the action.
    """

    actor_types: Optional[Sequence[EntityType]] = None
    """
    The types of the entities that can perform the action.
    If None, all entities can perform the action.
    """

    actee_types: Optional[Sequence[EntityType]] = None
    """
    The types of entities that can be selected by each actor.
    If None, all entities types can be selected by each actor.
    """

    actee_ids: Optional[Sequence[EntityID]] = None
    """
    The ids of the entities of each type that can be selected by each actor.
    If None, all entities can be selected by each actor.
    """

    mask: Optional[npt.NDArray[np.bool_]] = None
    """
    An boolean array of shape (len(actor_ids), len(actee_ids)). If mask[i, j] is True, then
    the agent with id actor_ids[i] can select entity with id actee_ids[j].
    (NOT CURRENTLY IMPLEMENTED)
    """

    def __post_init__(self) -> None:
        assert (
            self.actor_ids is None or self.actor_types is None
        ), "Only one of actor_ids or actor_types can be specified"
        assert (
            self.actee_types is None or self.actee_ids is None
        ), "Either actee_entity_types or actees can be specified, but not both."


ActionMask = Union[CategoricalActionMask, SelectEntityActionMask]


@dataclass
class Entity:
    features: List[str]


@dataclass
class ObsSpace:
    entities: Dict[EntityType, Entity]


@dataclass
class EntityObs:
    features: Union[npt.NDArray[np.float32], Sequence[Sequence[float]]]
    ids: Optional[Sequence[EntityID]] = None


@dataclass
class Observation:
    """
    Observation returned by the environment on one timestep.

    Attributes:
        features: Maps each entity type to a list of features for the entities of that type.
        actions: Maps each action type to an ActionMask specifying which entities can perform
            the action.
        reward: Reward received on this timestep.
        done: Whether the episode has ended.
        ids: Maps each entity type to a list of entity ids for the entities of that type.
        visible: Optional mask for each entity type that prevents the policy but not the
            value function from observing certain entities.
    """

    features: Mapping[
        EntityType, Union[npt.NDArray[np.float32], Sequence[Sequence[float]]]
    ]
    actions: Mapping[ActionType, ActionMask]
    done: bool
    reward: float
    ids: Mapping[EntityType, Sequence[EntityID]] = field(default_factory=dict)
    visible: Mapping[EntityType, Union[npt.NDArray[np.bool_], Sequence[bool]]] = field(
        default_factory=dict
    )
    metrics: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._id_to_index: Optional[Dict[EntityID, int]] = None
        self._index_to_id: Optional[List[EntityID]] = None

    @classmethod
    def from_entity_obs(
        cls,
        entities: Mapping[EntityType, Optional[EntityObs]],
        actions: Mapping[ActionType, ActionMask],
        done: bool,
        reward: float,
        metrics: Optional[Dict[str, float]] = None,
    ) -> "Observation":
        return cls(
            features={
                etype: entity.features
                for etype, entity in entities.items()
                if entity is not None
            },
            actions=actions,
            done=done,
            reward=reward,
            ids={
                etype: entity.ids
                for etype, entity in entities.items()
                if entity is not None and entity.ids is not None
            },
            metrics=metrics or {},
        )

    def _actor_indices(
        self, atype: ActionType, obs_space: ObsSpace
    ) -> npt.NDArray[np.int64]:
        action = self.actions[atype]
        if action.actor_ids is not None:
            id_to_index = self.id_to_index(obs_space)
            return np.array(
                [id_to_index[id] for id in action.actor_ids], dtype=np.int64
            )
        elif action.actor_types is not None:
            ids: List[int] = []
            id_to_index = self.id_to_index(obs_space)
            for etype in action.actor_types:
                ids.extend(id_to_index[id] for id in self.ids[etype])
            return np.array(ids, dtype=np.int64)
        else:
            return np.arange(
                sum(len(self.ids[etype]) for etype in obs_space.entities),
                dtype=np.int64,
            )

    def _actee_indices(
        self, atype: ActionType, obs_space: ObsSpace
    ) -> npt.NDArray[np.int64]:
        action = self.actions[atype]
        assert isinstance(action, SelectEntityActionMask)
        if action.actee_ids is not None:
            id_to_index = self.id_to_index(obs_space)
            return np.array(
                [id_to_index[id] for id in action.actee_ids], dtype=np.int64
            )
        elif action.actee_types is not None:
            ids: List[int] = []
            id_to_index = self.id_to_index(obs_space)
            for etype in action.actee_types:
                ids.extend(id_to_index[id] for id in self.ids[etype])
            return np.array(ids, dtype=np.int64)
        else:
            return np.arange(
                sum(len(self.ids[etype]) for etype in obs_space.entities),
                dtype=np.int64,
            )

    def id_to_index(self, obs_space: ObsSpace) -> Dict[EntityID, int]:
        offset = 0
        if self._id_to_index is None:
            self._id_to_index = {}
            for etype in obs_space.entities.keys():
                ids = self.ids.get(etype)
                if ids is None:
                    continue
                for i, id in enumerate(ids):
                    self._id_to_index[id] = i + offset
                offset += len(ids)
        return self._id_to_index

    def index_to_id(self, obs_space: ObsSpace) -> List[EntityID]:
        if self._index_to_id is None:
            self._index_to_id = []
            for etype in obs_space.entities.keys():
                ids = self.ids.get(etype)
                if ids is None:
                    ids = [None] * self.num_entities(etype)
                self._index_to_id.extend(ids)
        return self._index_to_id

    def num_entities(self, entity: EntityType) -> int:
        feats = self.features[entity]
        if isinstance(feats, np.ndarray):
            return feats.shape[0]
        else:
            return len(feats)


@dataclass
class CategoricalAction:
    actors: Sequence[EntityID]
    actions: npt.NDArray[np.int64]

    def items(self) -> Generator[Tuple[EntityID, int], None, None]:
        yield from zip(self.actors, self.actions)


@dataclass
class SelectEntityAction:
    actors: Sequence[EntityID]
    actees: Sequence[EntityID]

    def items(self) -> Generator[Tuple[EntityID, EntityID], None, None]:
        yield from zip(self.actors, self.actees)


Action = Union[CategoricalAction, SelectEntityAction]


class Environment(ABC):
    """
    Abstraction over reinforcement learning environments with observations based on structured lists of entities.
    """

    @abstractmethod
    def obs_space(self) -> ObsSpace:
        """
        Returns a dictionary mapping the name of observable entities to their type.
        """
        raise NotImplementedError

    @abstractmethod
    def action_space(self) -> Dict[str, ActionSpace]:
        """
        Returns a dictionary mapping the name of actions to their action space.
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> Observation:
        """
        Resets the environment and returns the initial observation.
        """
        raise NotImplementedError

    @abstractmethod
    def act(self, action: Mapping[ActionType, Action]) -> Observation:
        """
        Performs the given action and returns the resulting observation.

        Args:
            action: Maps the name of each action type to the action to perform.
        """
        raise NotImplementedError

    def reset_filter(self, obs_filter: ObsSpace) -> Observation:
        return self.filter_obs(self.reset(), obs_filter)

    def render(self, **kwargs: Any) -> npt.NDArray[np.uint8]:
        """
        Renders the environment

        Args:
            **kwargs: a dictionary of arguments to send to the rendering process
        """
        raise NotImplementedError

    def act_filter(
        self, action: Mapping[ActionType, Action], obs_filter: ObsSpace
    ) -> Observation:
        return self.filter_obs(self.act(action), obs_filter)

    def close(self) -> None:
        pass

    def filter_obs(self, obs: Observation, obs_filter: ObsSpace) -> Observation:
        selectors = self._compile_feature_filter(obs_filter)
        features: Dict[
            EntityType, Union[npt.NDArray[np.float32], Sequence[Sequence[float]]]
        ] = {}
        for etype, feats in obs.features.items():
            selector = selectors[etype]
            if isinstance(feats, np.ndarray):
                features[etype] = feats[:, selector].reshape(
                    feats.shape[0], len(selector)
                )
            else:
                features[etype] = [[entity[i] for i in selector] for entity in feats]
        return Observation(
            features=features,
            actions=obs.actions,
            done=obs.done,
            reward=obs.reward,
            ids=obs.ids,
            metrics=obs.metrics,
            visible=obs.visible,
        )

    def _compile_feature_filter(self, obs_space: ObsSpace) -> Dict[str, np.ndarray]:
        obs_space = self.obs_space()
        feature_selection = {}
        for entity_name, entity in obs_space.entities.items():
            feature_selection[entity_name] = np.array(
                [entity.features.index(f) for f in entity.features], dtype=np.int32
            )
        return feature_selection

    def env_cls(self) -> Type["Environment"]:
        return self.__class__
