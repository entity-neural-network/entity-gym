from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Type,
    Union,
    overload,
)
from entity_gym.environment.environment import (
    Action,
    ActionSpace,
    CategoricalActionSpace,
    DenseSelectEntityActionMask,
    EntityID,
    Environment,
    EpisodeStats,
    DenseCategoricalActionMask,
    ObsSpace,
    Observation,
    SelectEntityActionSpace,
)
import numpy as np
import numpy.typing as npt
from ragged_buffer import RaggedBufferF32, RaggedBufferI64, RaggedBufferBool


@dataclass
class CategoricalActionMaskBatch:
    actors: RaggedBufferI64
    masks: Optional[RaggedBufferBool]

    def push(self, mask: Any) -> None:
        assert isinstance(mask, DenseCategoricalActionMask)
        self.actors.push(mask.actors.reshape(-1, 1))
        if self.masks is not None and mask.mask is not None:
            self.masks.push(mask.mask.reshape(-1, self.masks.size2()))

    def __getitem__(
        self, i: Union[int, npt.NDArray[np.int64]]
    ) -> "CategoricalActionMaskBatch":
        if self.masks is not None and self.masks.size0() > 0:
            return CategoricalActionMaskBatch(self.actors[i], self.masks[i])
        else:
            return CategoricalActionMaskBatch(self.actors[i], None)

    def extend(self, other: Any) -> None:
        assert isinstance(
            other, CategoricalActionMaskBatch
        ), f"Expected CategoricalActionMaskBatch, got {type(other)}"
        self.actors.extend(other.actors)
        if self.masks is not None and other.masks is not None:
            self.masks.extend(other.masks)

    def clear(self) -> None:
        self.actors.clear()
        if self.masks is not None:
            self.masks.clear()


@dataclass
class SelectEntityActionMaskBatch:
    actors: RaggedBufferI64
    actees: RaggedBufferI64

    def push(self, mask: Any) -> None:
        assert isinstance(
            mask, DenseSelectEntityActionMask
        ), f"Expected DenseSelectEntityActionMask, got {type(mask)}"
        self.actors.push(mask.actors.reshape(-1, 1))
        self.actees.push(mask.actees.reshape(-1, 1))

    @overload
    def __getitem__(self, i: int) -> RaggedBufferI64:
        ...

    @overload
    def __getitem__(self, i: npt.NDArray[np.int64]) -> "SelectEntityActionMaskBatch":
        ...

    def __getitem__(
        self, i: Union[int, npt.NDArray[np.int64]]
    ) -> Union["SelectEntityActionMaskBatch", RaggedBufferI64]:
        if isinstance(i, int):
            return self.actors[i]
        else:
            return SelectEntityActionMaskBatch(self.actors[i], self.actees[i])

    def extend(self, other: Any) -> None:
        assert isinstance(
            other, SelectEntityActionMaskBatch
        ), f"Expected SelectEntityActionMaskBatch, got {type(other)}"
        self.actors.extend(other.actors)
        self.actees.extend(other.actees)

    def clear(self) -> None:
        self.actors.clear()
        self.actees.clear()


ActionMaskBatch = Union[CategoricalActionMaskBatch, SelectEntityActionMaskBatch]


@dataclass
class ObsBatch:
    entities: Dict[str, RaggedBufferF32]
    ids: List[Sequence[EntityID]]
    action_masks: Dict[str, ActionMaskBatch]
    reward: npt.NDArray[np.float32]
    done: npt.NDArray[np.bool_]
    end_of_episode_info: Dict[int, EpisodeStats]

    def merge_obs(self, b: "ObsBatch") -> None:
        """
        Merges ObsBatch b into this batch
        """
        envs = len(self.ids)

        # merge entities
        for k in b.entities.keys():
            self.entities[k].extend(b.entities[k])

        # merge ids
        self.ids.extend(b.ids)

        # merge masks
        for k in b.action_masks.keys():
            self.action_masks[k].extend(b.action_masks[k])

        self.reward = np.concatenate((self.reward, b.reward))
        self.done = np.concatenate((self.done, b.done))

        for i, stats in b.end_of_episode_info.items():
            self.end_of_episode_info[i + envs] = stats


class VecEnv(ABC):
    @abstractmethod
    def env_cls(cls) -> Type[Environment]:
        """
        Returns the class of the underlying environment.
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self, obs_config: ObsSpace) -> ObsBatch:
        raise NotImplementedError

    @abstractmethod
    def act(
        self, actions: Sequence[Mapping[str, Action]], obs_filter: ObsSpace
    ) -> ObsBatch:
        raise NotImplementedError

    @abstractmethod
    def render(self, **kwargs: Any) -> npt.NDArray[np.uint8]:
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    def close(self) -> None:
        pass


def batch_obs(
    obs: List[Observation], obs_space: ObsSpace, action_space: Dict[str, ActionSpace]
) -> ObsBatch:
    """
    Converts a list of observations into a batch of observations.
    """
    entities = {}
    ids = []
    action_masks: Dict[str, ActionMaskBatch] = {}
    reward = []
    done = []
    end_of_episode_info = {}

    # Initialize the entire batch with all entities and actions
    for entity_name, entity in obs_space.entities.items():
        feature_size = len(entity.features)
        entities[entity_name] = RaggedBufferF32(feature_size)

    for action_name, space in action_space.items():
        if isinstance(space, CategoricalActionSpace):
            mask_size = len(space.choices)
            action_masks[action_name] = CategoricalActionMaskBatch(
                RaggedBufferI64(1), RaggedBufferBool(mask_size)
            )
        elif isinstance(space, SelectEntityActionSpace):
            action_masks[action_name] = SelectEntityActionMaskBatch(
                RaggedBufferI64(1), RaggedBufferI64(1)
            )

    for o in obs:

        # Append the IDs
        ids.append(o.ids)

        # Append the entities
        for entity_name in obs_space.entities.keys():
            if entity_name not in o.entities:
                feature_size = len(obs_space.entities[entity_name].features)
                entities[entity_name].push(np.array([], dtype=np.float32))
            else:
                entities[entity_name].push(o.entities[entity_name])

        # Append the action masks
        for action_name, space in action_space.items():
            if isinstance(space, CategoricalActionSpace):
                if action_name not in o.action_masks:
                    mask_size = len(space.choices)
                    action_masks[action_name].push(
                        DenseCategoricalActionMask(
                            np.array([], dtype=int), np.array([], dtype=bool)
                        )
                    )
                else:
                    action_masks[action_name].push(o.action_masks[action_name])
            elif isinstance(space, SelectEntityActionSpace):
                if action_name not in o.action_masks:
                    action_masks[action_name].push(
                        DenseSelectEntityActionMask(
                            np.array([], dtype=int), np.array([], dtype=int)
                        )
                    )
                else:
                    action_masks[action_name].push(o.action_masks[action_name])

        # Append the rewards
        reward.append(o.reward)

        # Append the dones
        done.append(o.done)

        # Append the episode infos
        if o.end_of_episode_info:
            end_of_episode_info[len(ids) - 1] = o.end_of_episode_info

    return ObsBatch(
        entities,
        ids,
        action_masks,
        np.array(reward),
        np.array(done),
        end_of_episode_info,
    )
