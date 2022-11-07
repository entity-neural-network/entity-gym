import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Union, overload

import numpy as np
import numpy.typing as npt
from ragged_buffer import RaggedBufferBool, RaggedBufferF32, RaggedBufferI64

from entity_gym.env.environment import (
    ActionName,
    ActionSpace,
    CategoricalActionSpace,
    EntityName,
    GlobalCategoricalActionSpace,
    Observation,
    ObsSpace,
    SelectEntityActionSpace,
)


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
class Metric:
    count: int = 0
    sum: float = 0.0
    min: float = float("inf")
    max: float = float("-inf")

    def push(self, value: float) -> None:
        self.count += 1
        self.sum += value
        self.min = min(self.min, value)
        self.max = max(self.max, value)

    def __iadd__(self, m: "Metric") -> "Metric":
        self.count += m.count
        self.sum += m.sum
        self.min = min(self.min, m.min)
        self.max = max(self.max, m.max)
        return self

    def __add__(self, m: "Metric") -> "Metric":
        return Metric(
            count=self.count + m.count,
            sum=self.sum + m.sum,
            min=min(self.min, m.min),
            max=max(self.max, m.max),
        )

    @property
    def mean(self) -> float:
        if self.count == 0:
            return 0.0
        else:
            return self.sum / self.count


@dataclass
class VecObs:
    """
    A batch of observations from a vectorized environment.
    """

    features: Dict[EntityName, RaggedBufferF32]
    # Optional mask to hide specific entities from the policy but not the value function
    visible: Dict[EntityName, RaggedBufferBool]
    action_masks: Dict[ActionName, VecActionMask]
    reward: npt.NDArray[np.float32]
    done: npt.NDArray[np.bool_]
    metrics: Dict[str, Metric]

    def extend(self, b: "VecObs") -> None:
        num_envs = len(self.reward)
        # Extend visible (must happen before features in case of backfill)
        for etype in self.features.keys():
            if etype in b.visible:
                if etype not in self.visible:
                    self.visible[etype] = RaggedBufferBool.from_flattened(
                        flattened=np.ones(
                            shape=(self.features[etype].items(), 1), dtype=np.bool_
                        ),
                        lengths=self.features[etype].size1(),
                    )
                self.visible[etype].extend(b.visible[etype])
            elif etype in self.visible:
                self.visible[etype].extend(
                    RaggedBufferBool.from_flattened(
                        flattened=np.ones(
                            shape=(b.features[etype].items(), 1), dtype=np.bool_
                        ),
                        lengths=b.features[etype].size1(),
                    )
                )
        for etype, feats in b.features.items():
            if etype not in self.features:
                self.features[etype] = empty_ragged_f32(
                    feats=feats.size2(), sequences=num_envs
                )
            self.features[etype].extend(feats)
        for etype, feats in self.features.items():
            if etype not in b.features:
                feats.extend(empty_ragged_f32(feats.size2(), len(b.reward)))
        for atype, amask in b.action_masks.items():
            if atype not in self.action_masks:
                raise NotImplementedError()
            else:
                self.action_masks[atype].extend(amask)
        self.reward = np.concatenate((self.reward, b.reward))
        self.done = np.concatenate((self.done, b.done))
        num_envs = len(self.reward)
        for name, stats in b.metrics.items():
            if name in self.metrics:
                self.metrics[name].count += stats.count
                self.metrics[name].sum += stats.sum
                self.metrics[name].min = min(self.metrics[name].min, stats.min)
                self.metrics[name].max = max(self.metrics[name].max, stats.max)
            else:
                self.metrics[name] = copy.copy(stats)


class VecEnv(ABC):
    """
    Interface for vectorized environments. The main goal of VecEnv is to allow
    for maximally efficient environment implementations.
    """

    @abstractmethod
    def obs_space(self) -> ObsSpace:
        """
        Returns a dictionary mapping the name of observable entities to their type.
        """
        raise NotImplementedError

    @abstractmethod
    def action_space(self) -> Dict[ActionName, ActionSpace]:
        """
        Returns a dictionary mapping the name of actions to their action space.
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self, obs_config: ObsSpace) -> VecObs:
        """
        Resets all environments and returns the initial observations.
        """
        raise NotImplementedError

    @abstractmethod
    def act(
        self, actions: Mapping[ActionName, RaggedBufferI64], obs_filter: ObsSpace
    ) -> VecObs:
        """
        Performs the given actions on the underlying environments and returns the resulting observations.
        Any environment that reaches the end of its episode is reset and returns the initial observation of the next episode.
        """
        raise NotImplementedError

    @abstractmethod
    def render(self, **kwargs: Any) -> npt.NDArray[np.uint8]:
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    def close(self) -> None:
        pass

    def has_global_entity(self) -> bool:
        return len(self.obs_space().global_features) > 0 or any(
            isinstance(space, GlobalCategoricalActionSpace)
            for space in self.action_space().values()
        )


def batch_obs(
    obs: List[Observation], obs_space: ObsSpace, action_space: Dict[str, ActionSpace]
) -> VecObs:
    """
    Converts a list of observations into a batch of observations.
    """
    features: Dict[EntityName, RaggedBufferF32] = {}
    visible: Dict[EntityName, RaggedBufferBool] = {}
    action_masks: Dict[ActionName, VecActionMask] = {}
    reward = []
    done = []
    metrics = {}

    # Initialize the entire batch with all entities and actions
    for entity_name, entity in obs_space.entities.items():
        nfeat = len(entity.features)
        features[entity_name] = RaggedBufferF32(nfeat)
    global_entity = len(obs_space.global_features) > 0
    for action_name, space in action_space.items():
        if isinstance(space, CategoricalActionSpace):
            action_masks[action_name] = VecCategoricalActionMask(
                RaggedBufferI64(1),
                None,
            )
        elif isinstance(space, GlobalCategoricalActionSpace):
            action_masks[action_name] = VecCategoricalActionMask(
                RaggedBufferI64(1),
                None,
            )
            global_entity = True
        elif isinstance(space, SelectEntityActionSpace):
            action_masks[action_name] = VecSelectEntityActionMask(
                RaggedBufferI64(1), RaggedBufferI64(1)
            )
        else:
            raise NotImplementedError(f"Action space {space} not supported")
    if global_entity:
        nfeat = len(obs_space.global_features)
        features["__global__"] = RaggedBufferF32(nfeat)

    for i, o in enumerate(obs):
        # Merge entity features
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
        if global_entity:
            gfeats = o.global_features
            if not isinstance(gfeats, np.ndarray):
                gfeats = np.array(gfeats, dtype=np.float32)
            features["__global__"].push(
                gfeats.reshape(1, len(obs_space.global_features))
            )

        # Merge visibilities
        for etype, vis in o.visible.items():
            if etype not in visible:
                lengths = []
                for j in range(i):
                    if etype in obs[j].features:
                        lengths.append(len(obs[j].features[etype]))
                    else:
                        lengths.append(0)
                visible[etype] = RaggedBufferBool.from_flattened(
                    np.ones((sum(lengths), 1), dtype=np.bool_),
                    lengths=np.array(lengths, dtype=np.int64),
                )
            if not isinstance(vis, np.ndarray):
                vis = np.array(vis, dtype=np.bool_)
            visible[etype].push(vis.reshape(-1, 1))

        # Merge action masks
        for atype, space in action_space.items():
            if atype not in o.actions:
                if atype in action_masks:
                    if isinstance(space, CategoricalActionSpace):
                        vec_action = action_masks[atype]
                        assert isinstance(vec_action, VecCategoricalActionMask)
                        vec_action.actors.push(np.zeros((0, 1), dtype=np.int64))
                        if vec_action.mask is not None:
                            vec_action.mask.push(
                                np.zeros((0, len(space.index_to_label)), dtype=np.bool_)
                            )
                    elif isinstance(space, SelectEntityActionSpace):
                        vec_action = action_masks[atype]
                        assert isinstance(vec_action, VecSelectEntityActionMask)
                        vec_action.actors.push(np.zeros((0, 1), dtype=np.int64))
                        vec_action.actees.push(np.zeros((0, 1), dtype=np.int64))
                    else:
                        raise ValueError(
                            f"Unsupported action space type: {type(space)}"
                        )
                continue
            action = o.actions[atype]
            if atype not in action_masks:
                if isinstance(space, CategoricalActionSpace) or isinstance(
                    space, GlobalCategoricalActionSpace
                ):
                    action_masks[atype] = VecCategoricalActionMask(
                        empty_ragged_i64(1, i), None
                    )
                elif isinstance(space, SelectEntityActionSpace):
                    action_masks[atype] = VecSelectEntityActionMask(
                        empty_ragged_i64(1, i), empty_ragged_i64(1, i)
                    )
                else:
                    raise ValueError(f"Unknown action space type: {space}")
            if isinstance(space, CategoricalActionSpace) or isinstance(
                space, GlobalCategoricalActionSpace
            ):
                vec_action = action_masks[atype]
                assert isinstance(vec_action, VecCategoricalActionMask)
                actor_indices = o._actor_indices(atype, obs_space)
                vec_action.actors.push(actor_indices.reshape(-1, 1))
                if action.mask is not None:
                    if vec_action.mask is None:
                        vec_action.mask = RaggedBufferBool.from_flattened(
                            np.ones((0, len(space.index_to_label)), dtype=np.bool_),
                            np.zeros(i, dtype=np.int64),
                        )
                    amask = action.mask
                    if not isinstance(amask, np.ndarray):
                        amask = np.array(amask, dtype=np.bool_)
                    vec_action.mask.push(amask)
                elif vec_action.mask is not None:
                    vec_action.mask.push(
                        np.ones(
                            (len(actor_indices), len(space.index_to_label)),
                            dtype=np.bool_,
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
        for name, value in o.metrics.items():
            if name not in metrics:
                metrics[name] = Metric()
            metrics[name].push(value)

    return VecObs(
        features,
        visible,
        action_masks,
        np.array(reward, dtype=np.float32),
        np.array(done, dtype=np.bool_),
        metrics,
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
