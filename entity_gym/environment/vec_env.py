from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Type,
    Union,
    overload,
)
from entity_gym.environment.environment import (
    ActionSpace,
    CategoricalActionSpace,
    CategoricalActionMask,
    Environment,
    EpisodeStats,
    ObsSpace,
    Observation,
    SelectEntityActionSpace,
    EntityType,
    ActionType,
)
import numpy as np
import numpy.typing as npt
from ragged_buffer import RaggedBufferF32, RaggedBufferI64, RaggedBufferBool


@dataclass
class VecSelectEntityActionMask:
    actors: RaggedBufferI64
    actees: RaggedBufferI64

    @overload
    def __getitem__(self, i: int) -> RaggedBufferI64:
        ...

    @overload
    def __getitem__(self, i: npt.NDArray[np.int64]) -> "VecSelectEntityActionMask":
        ...

    def __getitem__(
        self, i: Union[int, npt.NDArray[np.int64]]
    ) -> Union["VecSelectEntityActionMask", RaggedBufferI64]:
        if isinstance(i, int):
            return self.actors[i]
        else:
            return VecSelectEntityActionMask(self.actors[i], self.actees[i])

    def extend(self, other: Any) -> None:
        assert isinstance(
            other, VecSelectEntityActionMask
        ), f"Expected VecSelectEntityActionMask, got {type(other)}"
        self.actors.extend(other.actors)
        self.actees.extend(other.actees)

    def clear(self) -> None:
        self.actors.clear()
        self.actees.clear()


@dataclass
class VecCategoricalActionMask:
    actors: RaggedBufferI64
    mask: Optional[RaggedBufferBool]

    def __getitem__(
        self, i: Union[int, npt.NDArray[np.int64]]
    ) -> "VecCategoricalActionMask":
        if self.mask is not None and self.mask.size0() > 0:
            return VecCategoricalActionMask(self.actors[i], self.mask[i])
        else:
            return VecCategoricalActionMask(self.actors[i], None)

    def extend(self, other: Any) -> None:
        assert isinstance(
            other, VecCategoricalActionMask
        ), f"Expected CategoricalActionMaskBatch, got {type(other)}"
        if self.mask is not None and other.mask is not None:
            self.mask.extend(other.mask)
        elif self.mask is None and other.mask is None:
            pass
        elif self.mask is not None:
            self.mask.extend(
                RaggedBufferBool.from_flattened(
                    flattened=np.ones(
                        shape=(other.actors.items(), self.mask.size2()),
                        dtype=np.bool_,
                    ),
                    lengths=other.actors.size1(),
                )
            )
        elif other.mask is not None:
            self.mask = RaggedBufferBool.from_flattened(
                flattened=np.ones(
                    shape=(self.actors.items(), other.mask.size2()),
                    dtype=np.bool_,
                ),
                lengths=self.actors.size1(),
            )
            self.mask.extend(other.mask)
        else:
            raise Exception("Impossible!")
        self.actors.extend(other.actors)

    def clear(self) -> None:
        self.actors.clear()
        if self.mask is not None:
            self.mask.clear()


VecActionMask = Union[VecCategoricalActionMask, VecSelectEntityActionMask]


@dataclass
class VecObs:
    features: Dict[EntityType, RaggedBufferF32]
    action_masks: Dict[ActionType, VecActionMask]
    reward: npt.NDArray[np.float32]
    done: npt.NDArray[np.bool_]
    end_of_episode_info: Dict[int, EpisodeStats]

    def extend(self, b: "VecObs") -> None:
        num_envs = len(self.reward)
        for etype, feats in b.features.items():
            if etype not in self.features:
                self.features[etype] = empty_ragged_f32(
                    feats=feats.size2(), sequences=num_envs
                )
            self.features[etype].extend(feats)
        for atype, amask in b.action_masks.items():
            if atype not in self.action_masks:
                raise NotImplementedError()
            else:
                self.action_masks[atype].extend(amask)
        self.reward = np.concatenate((self.reward, b.reward))
        self.done = np.concatenate((self.done, b.done))
        num_envs = len(self.reward)
        for i, stats in b.end_of_episode_info.items():
            self.end_of_episode_info[i + num_envs] = stats


class VecEnv(ABC):
    @abstractmethod
    def env_cls(cls) -> Type[Environment]:
        """
        Returns the class of the underlying environment.
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self, obs_config: ObsSpace) -> VecObs:
        raise NotImplementedError

    @abstractmethod
    def act(
        self, actions: Mapping[ActionType, RaggedBufferI64], obs_filter: ObsSpace
    ) -> VecObs:
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
) -> VecObs:
    """
    Converts a list of observations into a batch of observations.
    """
    features: Dict[EntityType, RaggedBufferF32] = {}
    action_masks: Dict[ActionType, VecActionMask] = {}
    reward = []
    done = []
    end_of_episode_info = {}

    # Initialize the entire batch with all entities and actions
    for entity_name, entity in obs_space.entities.items():
        feature_size = len(entity.features)
        features[entity_name] = RaggedBufferF32(feature_size)

    for action_name, space in action_space.items():
        if isinstance(space, CategoricalActionSpace):
            action_masks[action_name] = VecCategoricalActionMask(
                RaggedBufferI64(1),
                None,
            )
        elif isinstance(space, SelectEntityActionSpace):
            action_masks[action_name] = VecSelectEntityActionMask(
                RaggedBufferI64(1), RaggedBufferI64(1)
            )

    for i, o in enumerate(obs):
        for entity_type, entity in obs_space.entities.items():
            if entity_type not in features:
                features[entity_type] = RaggedBufferF32.from_flattened(
                    np.zeros((0, len(entity.features)), dtype=np.float32),
                    lengths=np.zeros(i, dtype=np.int64),
                )
            if entity_type in o.features:
                ofeats = o.features[entity_type]
                if not isinstance(ofeats, np.ndarray):
                    ofeats = np.array(ofeats, dtype=np.float32).reshape(
                        len(ofeats), len(obs_space.entities[entity_type].features)
                    )
                features[entity_type].push(ofeats)
            else:
                features[entity_type].push(
                    np.zeros((0, len(entity.features)), dtype=np.float32)
                )

        for atype, space in action_space.items():
            if atype not in o.actions:
                if atype in action_masks:
                    if isinstance(space, CategoricalActionSpace):
                        vec_action = action_masks[atype]
                        assert isinstance(vec_action, VecCategoricalActionMask)
                        vec_action.actors.push(np.zeros((0, 1), dtype=np.int64))
                        if vec_action.mask is not None:
                            vec_action.mask.push(
                                np.zeros((0, len(space.choices)), dtype=np.bool_)
                            )
                    elif isinstance(space, SelectEntityActionSpace):
                        vec_action = action_masks[atype]
                        assert isinstance(vec_action, VecSelectEntityActionMask)
                        vec_action.actors.push(np.zeros((0, 1), dtype=np.int64))
                        vec_action.actees.push(np.zeros((0, 1), dtype=np.int64))
                continue
            action = o.actions[atype]
            if atype not in action_masks:
                if isinstance(space, CategoricalActionSpace):
                    action_masks[atype] = VecCategoricalActionMask(
                        empty_ragged_i64(1, i), None
                    )
                elif isinstance(space, SelectEntityActionSpace):
                    action_masks[atype] = VecSelectEntityActionMask(
                        empty_ragged_i64(1, i), empty_ragged_i64(1, i)
                    )
            if isinstance(space, CategoricalActionSpace):
                vec_action = action_masks[atype]
                assert isinstance(vec_action, VecCategoricalActionMask)
                actor_indices = o._actor_indices(atype, obs_space)
                vec_action.actors.push(actor_indices.reshape(-1, 1))
                if action.mask is not None:
                    if vec_action.mask is None:
                        vec_action.mask = RaggedBufferBool.from_flattened(
                            np.ones((0, len(space.choices)), dtype=np.bool_),
                            np.zeros(i, dtype=np.int64),
                        )
                    amask = action.mask
                    if not isinstance(amask, np.ndarray):
                        amask = np.array(amask, dtype=np.bool_)
                    vec_action.mask.push(amask)
                elif vec_action.mask is not None:
                    vec_action.mask.push(
                        np.ones(
                            (len(actor_indices), len(space.choices)), dtype=np.bool_
                        )
                    )
            elif isinstance(space, SelectEntityActionSpace):
                vec_action = action_masks[atype]
                assert isinstance(vec_action, VecSelectEntityActionMask)
                actors = o._actor_indices(atype, obs_space).reshape(-1, 1)
                vec_action.actors.push(actors)
                if len(actors) > 0:
                    vec_action.actees.push(
                        o._actee_indices(atype, obs_space).reshape(-1, 1)
                    )
                else:
                    vec_action.actees.push(np.zeros((0, 1), dtype=np.int64))
            else:
                raise NotImplementedError()

        reward.append(o.reward)
        done.append(o.done)
        if o.end_of_episode_info:
            end_of_episode_info[len(reward) - 1] = o.end_of_episode_info

    return VecObs(
        features,
        action_masks,
        np.array(reward, dtype=np.float32),
        np.array(done, dtype=np.bool_),
        end_of_episode_info,
    )


def empty_ragged_f32(feats: int, sequences: int) -> RaggedBufferF32:
    return RaggedBufferF32.from_flattened(
        np.zeros((0, feats), dtype=np.float32),
        lengths=np.array([0] * sequences, dtype=np.int64),
    )


def empty_ragged_i64(feats: int, sequences: int) -> RaggedBufferI64:
    return RaggedBufferI64.from_flattened(
        np.zeros((0, feats), dtype=np.int64),
        lengths=np.array([0] * sequences, dtype=np.int64),
    )
