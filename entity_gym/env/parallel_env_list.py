import multiprocessing as mp
import multiprocessing.connection as conn
from multiprocessing.connection import Connection
from typing import Any, Callable, Dict, Generator, List, Mapping, Optional

import cloudpickle
import msgpack_numpy
import numpy as np
import numpy.typing as npt
from ragged_buffer import RaggedBufferI64

from entity_gym.env.env_list import EnvList
from entity_gym.env.environment import (
    ActionName,
    ActionSpace,
    Environment,
    Observation,
    ObsSpace,
)
from entity_gym.env.vec_env import VecEnv, VecObs, batch_obs
from entity_gym.serialization.msgpack_ragged import (
    ragged_buffer_decode,
    ragged_buffer_encode,
)


class CloudpickleWrapper:
    def __init__(self, var: Any):
        self.var = var

    def __getstate__(self) -> Any:
        return cloudpickle.dumps(self.var)

    def __setstate__(self, var: Any) -> None:
        self.var = cloudpickle.loads(var)


class MsgpackConnectionWrapper:
    """
    Use msgpack instead of pickle to send and receive data from workers.
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
            elif cmd == "render":
                rgb_pixels = envs.render(**data)
                remote.send(rgb_pixels)
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
        create_env: Callable[[], Environment],
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
            (create_env, self.envs_per_process) for _ in range(self.num_processes)
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

        env = create_env()
        self._obs_space = env.obs_space()
        self._action_space = env.action_space()

    def reset(self, obs_space: ObsSpace) -> VecObs:
        for remote in self.remotes:
            remote.send(("reset", obs_space))

        # Empty initialized observation batch
        observations = batch_obs([], self.obs_space(), self.action_space())

        for remote in self.remotes:
            remote_obs_batch = remote.recv()
            observations.extend(remote_obs_batch)

        assert isinstance(observations, VecObs)
        return observations

    def render(self, **kwargs: Any) -> npt.NDArray[np.uint8]:
        rgb_arrays = []
        for remote in self.remotes:
            remote.send(("render", kwargs))
            rgb_arrays.append(remote.recv())

        np_rgb_arrays = np.concatenate(rgb_arrays)
        assert isinstance(np_rgb_arrays, np.ndarray)
        return np_rgb_arrays

    def close(self) -> None:
        for remote in self.remotes:
            remote.send(("close", None))
        for process in self.processes:
            process.join()

    def _chunk_actions(
        self, actions: Mapping[str, RaggedBufferI64]
    ) -> Generator[Mapping[str, RaggedBufferI64], List[Observation], None]:
        for i in range(0, self.num_envs, self.envs_per_process):
            yield {
                atype: a[i : i + self.envs_per_process, :, :]
                for atype, a in actions.items()
            }

    def act(
        self, actions: Mapping[str, RaggedBufferI64], obs_space: ObsSpace
    ) -> VecObs:
        remote_actions = self._chunk_actions(actions)
        for remote, action in zip(self.remotes, remote_actions):
            remote.send(("act", (action, obs_space)))

        # Empty initialized observation batch
        observations = batch_obs([], self.obs_space(), self.action_space())

        for remote in self.remotes:
            remote_obs_batch = remote.recv()
            observations.extend(remote_obs_batch)
        return observations

    def __len__(self) -> int:
        return self.num_envs

    def obs_space(self) -> ObsSpace:
        return self._obs_space

    def action_space(self) -> Dict[ActionName, ActionSpace]:
        return self._action_space
