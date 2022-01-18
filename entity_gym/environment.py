import functools
import multiprocessing as mp
import multiprocessing.connection as conn
from multiprocessing.connection import Connection
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
    Iterable,
    Callable,
    Generator,
    overload,
)
import cloudpickle
import msgpack
import msgpack_numpy
import numpy as np
import numpy.typing as npt
from ragged_buffer import RaggedBufferF32, RaggedBufferI64, RaggedBufferBool


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
class Entity:
    features: List[str]


@dataclass
class ObsSpace:
    entities: Dict[str, Entity]


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

        self.end_of_episode_info.update(b.end_of_episode_info)


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
    def __len__(self) -> int:
        raise NotImplementedError

    def close(self) -> None:
        pass


class EnvList(VecEnv):
    def __init__(
        self, env_cls: Type[Environment], env_kwargs: Dict[str, Any], num_envs: int
    ):
        self.envs = [env_cls(**env_kwargs) for _ in range(num_envs)]  # type: ignore
        self.cls = env_cls

    def env_cls(cls) -> Type[Environment]:
        return cls.cls

    def reset(self, obs_space: ObsSpace) -> ObsBatch:
        return batch_obs(
            [e.reset(obs_space) for e in self.envs],
            self.cls.obs_space(),
            self.cls.action_space(),
        )

    def close(self) -> None:
        for env in self.envs:
            env.close()

    def act(
        self, actions: Sequence[Mapping[str, Action]], obs_space: ObsSpace
    ) -> ObsBatch:
        observations = []
        for e, a in zip(self.envs, actions):
            obs = e.act(a, obs_space)
            if obs.done:
                # TODO: something is wrong with the interface here
                new_obs = e.reset(obs_space)
                new_obs.done = True
                new_obs.reward = obs.reward
                new_obs.end_of_episode_info = obs.end_of_episode_info
                observations.append(new_obs)
            else:
                observations.append(obs)
        return batch_obs(observations, self.cls.obs_space(), self.cls.action_space())

    def __len__(self) -> int:
        return len(self.envs)


class CloudpickleWrapper:
    def __init__(self, var: Any):
        self.var = var

    def __getstate__(self) -> Any:
        return cloudpickle.dumps(self.var)

    def __setstate__(self, var: Any) -> None:
        self.var = cloudpickle.loads(var)


class MsgpackConnectionWrapper(object):
    """
    Use msgpack instead of pickle to send and recieve data from workers.
    """

    def __init__(self, conn: Connection) -> None:
        self._conn = conn

    def close(self) -> None:
        self._conn.close()

    def send(self, data: Any) -> None:
        s = msgpack_numpy.dumps(data, default=ragged_buffer_encode)
        self._conn.send_bytes(s)

    def recv(self) -> Any:
        data_bytes = self._conn.recv_bytes()
        return msgpack_numpy.loads(
            data_bytes,
            object_hook=ragged_buffer_decode,
            strict_map_key=False,
        )


# For security reasons we don't want to deserialize classes that are not in this list.
WHITELIST = [
    "ObsSpace",
    "ObsBatch",
    "CategoricalActionMaskBatch",
    "SelectEntityActionMaskBatch",
    "SelectEntityAction",
    "CategoricalAction",
    "Entity",
    "EpisodeStats",
]


def ragged_buffer_encode(obj: Any) -> Any:
    if isinstance(obj, RaggedBufferF32) or isinstance(obj, RaggedBufferI64) or isinstance(obj, RaggedBufferBool):  # type: ignore
        flattened = obj.as_array()
        lengths = obj.size1()
        return {
            "__flattened__": msgpack_numpy.encode(flattened),
            "__lengths__": msgpack_numpy.encode(lengths),
        }
    elif hasattr(obj, "__dict__"):
        return {"__classname__": obj.__class__.__name__, "data": vars(obj)}
    else:
        return obj


def ragged_buffer_decode(obj: Any) -> Any:
    if "__flattened__" in obj:
        flattened = msgpack_numpy.decode(obj["__flattened__"])
        lengths = msgpack_numpy.decode(obj["__lengths__"])

        dtype = flattened.dtype

        if dtype == np.float32:
            return RaggedBufferF32.from_flattened(flattened, lengths)
        elif dtype == int:
            return RaggedBufferI64.from_flattened(flattened, lengths)
    elif "__classname__" in obj:
        classname = obj["__classname__"]
        if classname in WHITELIST:
            cls_name = globals()[classname]
            return cls_name(**obj["data"])
        else:
            raise RuntimeError(
                f"Attempt to deserialize class {classname} outside whitelist."
            )
    else:
        return obj


def _worker(
    remote: conn.Connection,
    parent_remote: conn.Connection,
    env_list_config: CloudpickleWrapper,
) -> None:
    parent_remote.close()
    env_args = env_list_config.var
    envs = EnvList(*env_args)
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == "act":
                observation = envs.act(data[0], data[1])
                remote.send(observation)
            elif cmd == "reset":
                observation = envs.reset(data)
                remote.send(observation)
            elif cmd == "close":
                envs.close()
                remote.close()
                break
            else:
                raise NotImplementedError(f"`{cmd}` is not implemented in the worker")
        except EOFError:
            break


class ParallelEnvList(VecEnv):
    """
    We fork the subprocessing from the stable-baselines implementation, but use RaggedBuffers for collecting batches

    Citation here: https://github.com/DLR-RM/stable-baselines3/blob/master/CITATION.bib
    """

    def __init__(
        self,
        env_cls: Type[Environment],
        env_kwargs: Dict[str, Any],
        num_envs: int,
        num_processes: int,
        start_method: Optional[str] = None,
    ):

        if start_method is None:
            # Fork is not a thread safe method (see issue #217)
            # but is more user friendly (does not require to wrap the code in
            # a `if __name__ == "__main__":`)
            forkserver_available = "forkserver" in mp.get_all_start_methods()
            start_method = "forkserver" if forkserver_available else "spawn"
        ctx = mp.get_context(start_method)

        assert (
            num_envs % num_processes == 0
        ), "The required number of environments can not be equally split into the number of specified processes."

        self.num_processes = num_processes
        self.num_envs = num_envs
        self.envs_per_process = int(num_envs / num_processes)

        env_list_configs = [
            (env_cls, env_kwargs, self.envs_per_process)
            for _ in range(self.num_processes)
        ]

        self.remotes = []
        self.work_remotes = []
        for i in range(self.num_processes):
            pipe = ctx.Pipe()
            self.remotes.append(MsgpackConnectionWrapper(pipe[0]))
            self.work_remotes.append(MsgpackConnectionWrapper(pipe[1]))

        self.processes = []
        for work_remote, remote, env_list_config in zip(
            self.work_remotes, self.remotes, env_list_configs
        ):
            # Have to use cloudpickle wrapper here to serialize the ABCMeta class reference
            # TODO: Can this be achieved with custom msgpack somehow?
            args = (work_remote, remote, CloudpickleWrapper(env_list_config))
            # daemon=True: if the main process crashes, we should not cause things to hang
            process = ctx.Process(
                target=_worker, args=args, daemon=True
            )  # pytype:disable=attribute-error
            process.start()
            self.processes.append(process)
            work_remote.close()

        self.cls = env_cls

    def env_cls(cls) -> Type[Environment]:
        return cls.cls

    def reset(self, obs_space: ObsSpace) -> ObsBatch:
        for remote in self.remotes:
            remote.send(("reset", obs_space))

        # Empty initialized observation batch
        observations = batch_obs([], self.cls.obs_space(), self.cls.action_space())

        for remote in self.remotes:
            remote_obs_batch = remote.recv()
            observations.merge_obs(remote_obs_batch)

        assert isinstance(observations, ObsBatch)
        return observations

    def close(self) -> None:
        for remote in self.remotes:
            remote.send(("close", None))
        for process in self.processes:
            process.join()

    def _chunk_actions(
        self, actions: Sequence[Mapping[str, Action]]
    ) -> Generator[Sequence[Mapping[str, Action]], List[Observation], None]:
        for i in range(0, len(actions), self.envs_per_process):
            yield actions[i : i + self.envs_per_process]

    def act(
        self, actions: Sequence[Mapping[str, Action]], obs_space: ObsSpace
    ) -> ObsBatch:
        remote_actions = self._chunk_actions(actions)
        for remote, action in zip(self.remotes, remote_actions):
            remote.send(("act", (action, obs_space)))

        # Empty initialized observation batch
        observations = batch_obs([], self.cls.obs_space(), self.cls.action_space())

        for remote in self.remotes:
            remote_obs_batch = remote.recv()
            observations.merge_obs(remote_obs_batch)
        return observations

    def __len__(self) -> int:
        return self.num_envs
