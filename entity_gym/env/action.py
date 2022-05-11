from dataclasses import dataclass
from typing import List, Optional, Sequence, Union

import numpy as np
import numpy.typing as npt

from .common import EntityID, EntityName


@dataclass
class CategoricalActionSpace:
    """
    Defines a discrete set of actions that can be taken by multiple entities.
    """

    index_to_label: List[str]
    """list of human-readable labels for each action"""

    def __len__(self) -> int:
        return len(self.index_to_label)


@dataclass
class GlobalCategoricalActionSpace:
    """
    Defines a discrete set of actions that can be taken on each timestep.

    For example, the following actions space allows the agent to choose between four actions "up", "down", "left", and "right":

    .. code-block:: python

        GlobalCategoricalActionSpace(["up", "down", "left", "right"])
    """

    index_to_label: List[str]
    """list of human-readable labels for each action"""

    def __len__(self) -> int:
        return len(self.index_to_label)


@dataclass
class SelectEntityActionSpace:
    """
    Allows multiple entities to each select another entity.
    """


ActionSpace = Union[
    CategoricalActionSpace, SelectEntityActionSpace, GlobalCategoricalActionSpace
]


@dataclass
class CategoricalActionMask:
    """
    Action mask for categorical action that specifies which agents can perform the action,
    and includes a dense mask that further constraints the choices available to each agent.
    """

    actor_ids: Optional[Sequence[EntityID]] = None
    """
    The ids of the entities that can perform the action.
    If ``None``, all entities can perform the action.
    Mutually exclusive with ``actor_types``.
    """

    actor_types: Optional[Sequence[EntityName]] = None
    """
    The types of the entities that can perform the action.
    If ``None``, all entities can perform the action.
    Mutually exclusive with ``actor_ids``.
    """

    mask: Union[Sequence[Sequence[bool]], np.ndarray, None] = None
    """
    A boolean array of shape ``(len(actor_ids), len(choices))`` that prevents specific actions from being available to certain entities.
    If ``mask[i, j]`` is ``True``, then the entity with id ``actor_ids[i]`` can perform action ``j``.
    """

    def __post_init__(self) -> None:
        assert (
            self.actor_ids is None or self.actor_types is None
        ), "Only one of actor_ids or actor_types can be specified"


@dataclass
class GlobalCategoricalActionMask:
    """
    Action mask for global categorical action.
    """

    mask: Union[Sequence[Sequence[bool]], np.ndarray, None] = None
    """
    An optional boolean array of shape (len(choices),). If mask[i] is True, then
    action choice i can be performed.
    """


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

    actor_types: Optional[Sequence[EntityName]] = None
    """
    The types of the entities that can perform the action.
    If None, all entities can perform the action.
    """

    actee_types: Optional[Sequence[EntityName]] = None
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


ActionMask = Union[
    CategoricalActionMask, SelectEntityActionMask, GlobalCategoricalActionMask
]


@dataclass
class CategoricalAction:
    """
    Outcome of a categorical action.
    """

    actors: Sequence[EntityID]
    """the ids of the entities that chose the actions"""

    indices: npt.NDArray[np.int64]
    """the indices of the actions that were chosen"""

    index_to_label: List[str]
    """mapping from action indices to human readable labels"""

    probs: Optional[npt.NDArray[np.float32]] = None
    """the probablity assigned to each action by each agent"""

    @property
    def labels(self) -> List[str]:
        """the human readable labels of the actions that were performed"""
        return [self.index_to_label[i] for i in self.indices]


@dataclass
class SelectEntityAction:
    """
    Outcome of a select entity action.
    """

    actors: Sequence[EntityID]
    """the ids of the entities that chose the action"""
    actees: Sequence[EntityID]
    """the ids of the entities that were selected by the actors"""
    probs: Optional[npt.NDArray[np.float32]] = None
    """the probablity assigned to each selection by each agent"""


@dataclass
class GlobalCategoricalAction:
    """Outcome of a global categorical action."""

    index: int
    """the index of the action that was chosen"""
    label: str
    """the human readable label of the action that was chosen"""
    probs: Optional[npt.NDArray[np.float32]] = None
    """the probablity assigned to the action by each agent"""


Action = Union[CategoricalAction, SelectEntityAction, GlobalCategoricalAction]
