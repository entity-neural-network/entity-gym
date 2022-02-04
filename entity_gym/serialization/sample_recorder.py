from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Type, Any

from ragged_buffer import RaggedBufferF32, RaggedBufferI64
import numpy as np
import msgpack_numpy

from entity_gym.environment import (
    Action,
    ActionSpace,
    Environment,
    VecObs,
    ObsSpace,
    VecEnv,
)
from entity_gym.serialization.msgpack_ragged import (
    ragged_buffer_decode,
    ragged_buffer_encode,
)


@dataclass
class Sample:
    obs: VecObs
    step: List[int]
    episode: List[int]
    actions: Mapping[str, RaggedBufferI64]
    probs: Dict[str, RaggedBufferF32]
    logits: Optional[Dict[str, RaggedBufferF32]]

    def serialize(self) -> bytes:
        return msgpack_numpy.dumps(  # type: ignore
            {
                "obs": self.obs,
                "step": self.step,
                "episode": self.episode,
                "actions": self.actions,
                "probs": self.probs,
                "logits": self.logits,
            },
            default=ragged_buffer_encode,
        )

    @classmethod
    def deserialize(cls, data: bytes) -> "Sample":
        return Sample(
            **msgpack_numpy.loads(
                data, object_hook=ragged_buffer_decode, strict_map_key=False
            )
        )


class SampleRecorder:
    """
    Writes samples to disk.
    """

    def __init__(
        self, path: str, act_space: Dict[str, ActionSpace], obs_space: ObsSpace
    ) -> None:
        self.path = path
        self.file = open(path, "wb")
        # TODO: write header and obs space and act space

    def record(
        self,
        sample: Sample,
    ) -> None:
        bytes = sample.serialize()
        # Write 8 bytes unsigned int for the size of the serialized sample
        self.file.write(np.uint64(len(bytes)).tobytes())
        self.file.write(bytes)

    def close(self) -> None:
        self.file.close()


class SampleRecordingVecEnv(VecEnv):
    def __init__(self, inner: VecEnv, out_path: str) -> None:
        self.inner = inner
        self.out_path = out_path
        self.sample_recorder = SampleRecorder(
            out_path, inner.env_cls().action_space(), inner.env_cls().obs_space()
        )
        self.last_obs: Optional[VecObs] = None
        self.episodes = list(range(len(inner)))
        self.curr_step = [0] * len(inner)
        self.next_episode = len(inner)

    def reset(self, obs_config: ObsSpace) -> VecObs:
        self.curr_step = [0] * len(self)
        self.last_obs = self.record_obs(self.inner.reset(obs_config))
        return self.last_obs

    def record_obs(self, obs: VecObs) -> VecObs:
        for i, done in enumerate(obs.done):
            if done:
                self.episodes[i] = self.next_episode
                self.next_episode += 1
                self.curr_step[i] = 0
            else:
                self.curr_step[i] += 1
        self.last_obs = obs
        return obs

    def act(
        self,
        actions: Mapping[str, RaggedBufferI64],
        obs_filter: ObsSpace,
        probs: Optional[Dict[str, RaggedBufferF32]] = None,
        logits: Optional[Dict[str, RaggedBufferF32]] = None,
    ) -> VecObs:
        if probs is None:
            probs = {}
        # with tracer.span("record_samples"):
        assert self.last_obs is not None
        self.sample_recorder.record(
            Sample(
                self.last_obs,
                step=list(self.curr_step),
                episode=list(self.episodes),
                actions=actions,
                probs=probs,
                logits=logits,
            )
        )
        return self.record_obs(self.inner.act(actions, obs_filter))

    def render(self, **kwargs: Any) -> np.ndarray:
        return self.inner.render(**kwargs)

    def env_cls(cls) -> Type[Environment]:
        return super().env_cls()

    def __len__(self) -> int:
        return len(self.inner)

    def close(self) -> None:
        self.sample_recorder.close()
        print("Recorded samples to: ", self.sample_recorder.path)
        self.inner.close()
