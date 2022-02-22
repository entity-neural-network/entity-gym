from dataclasses import dataclass
from typing import Dict, List, Optional
from entity_gym.environment.environment import ActionSpace
from entity_gym.environment.vec_env import VecActionMask, VecCategoricalActionMask

import msgpack_numpy
from ragged_buffer import RaggedBufferF32, RaggedBufferI64
import tqdm
import numpy as np

from entity_gym.environment import (
    Action,
    ObsSpace,
)
from entity_gym.serialization.sample_recorder import Sample
from entity_gym.serialization.msgpack_ragged import ragged_buffer_decode


@dataclass
class Episode:
    number: int
    steps: int
    entities: Dict[str, RaggedBufferF32]
    actions: Dict[str, RaggedBufferI64]
    masks: Dict[str, VecActionMask]
    logprobs: Dict[str, RaggedBufferF32]
    logits: Dict[str, RaggedBufferF32]
    total_reward: float
    complete: bool = False


@dataclass
class Trace:
    action_space: Dict[str, ActionSpace]
    obs_space: ObsSpace
    samples: List[Sample]

    @classmethod
    def deserialize(cls, data: bytes, progress_bar: bool = False) -> "Trace":
        samples: List[Sample] = []
        if progress_bar:
            pbar = tqdm.tqdm(total=len(data))

        offset = 0
        # Read version
        version = int(np.frombuffer(data[:8], dtype=np.uint64)[0])
        assert version == 0
        header_len = int(np.frombuffer(data[8:16], dtype=np.uint64)[0])
        header = msgpack_numpy.loads(
            data[16 : 16 + header_len],
            object_hook=ragged_buffer_decode,
            strict_map_key=False,
        )
        action_space = header["act_space"]
        obs_space = header["obs_space"]

        offset = 16 + header_len
        while offset < len(data):
            size = int(np.frombuffer(data[offset : offset + 8], dtype=np.uint64)[0])
            offset += 8
            samples.append(Sample.deserialize(data[offset : offset + size]))
            offset += size
            if progress_bar:
                pbar.update(size + 8)
        return Trace(action_space, obs_space, samples)

    def episodes(
        self, include_incomplete: bool = False, progress_bar: bool = False
    ) -> List[Episode]:
        episodes = {}
        prev_episodes: Optional[List[int]] = None
        if progress_bar:
            samples = tqdm.tqdm(self.samples)
        else:
            samples = self.samples
        for sample in samples:
            for i, e in enumerate(sample.episode):
                if e not in episodes:
                    episodes[e] = Episode(
                        e,
                        0,
                        {},
                        {},
                        {},
                        {},
                        {},
                        0.0,
                    )

                episodes[e].steps += 1
                episodes[e].total_reward += sample.obs.reward[i]
                if sample.obs.done[i] and prev_episodes is not None:
                    episodes[prev_episodes[i]].complete = True

                for name, feats in sample.obs.features.items():
                    if name not in episodes[e].entities:
                        episodes[e].entities[name] = feats[i]
                    else:
                        episodes[e].entities[name].extend(feats[i])
                for name, acts in sample.actions.items():
                    if name not in episodes[e].actions:
                        episodes[e].actions[name] = acts[i]
                    else:
                        episodes[e].actions[name].extend(acts[i])
                for name, mask in sample.obs.action_masks.items():
                    if name not in episodes[e].masks:
                        episodes[e].masks[name] = mask[i]
                    else:
                        episodes[e].masks[name].extend(mask[i])
                for name, logprobs in sample.probs.items():
                    if name not in episodes[e].logprobs:
                        episodes[e].logprobs[name] = logprobs[i]
                    else:
                        episodes[e].logprobs[name].extend(logprobs[i])
                if sample.logits is not None:
                    for name, logits in sample.logits.items():
                        if name not in episodes[e].logits:
                            episodes[e].logits[name] = logits[i]
                        else:
                            episodes[e].logits[name].extend(logits[i])
            prev_episodes = sample.episode
        return sorted(
            [e for e in episodes.values() if e.complete or include_incomplete],
            key=lambda e: e.number,
        )
