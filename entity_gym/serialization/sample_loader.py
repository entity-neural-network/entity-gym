from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import msgpack_numpy
import numpy as np
import tqdm
from ragged_buffer import RaggedBufferBool, RaggedBufferF32, RaggedBufferI64

from entity_gym.env import ObsSpace
from entity_gym.env.environment import ActionSpace
from entity_gym.env.vec_env import VecActionMask
from entity_gym.ragged_dict import RaggedActionDict, RaggedBatchDict
from entity_gym.serialization.msgpack_ragged import ragged_buffer_decode
from entity_gym.serialization.sample_recorder import Sample


@dataclass
class Episode:
    number: int
    steps: int
    entities: Dict[str, RaggedBufferF32]
    visible: Dict[str, RaggedBufferBool]
    actions: Dict[str, RaggedBufferI64]
    masks: Dict[str, VecActionMask]
    logprobs: Dict[str, RaggedBufferF32]
    logits: Dict[str, RaggedBufferF32]
    total_reward: float
    complete: bool = False


@dataclass
class MergedSamples:
    entities: RaggedBatchDict[np.float32]
    visible: RaggedBatchDict[np.bool_]
    actions: RaggedBatchDict[np.int64]
    logprobs: RaggedBatchDict[np.float32]
    masks: RaggedActionDict
    logits: Optional[RaggedBatchDict[np.float32]]
    frames: int

    @classmethod
    def empty(clz) -> "MergedSamples":
        return MergedSamples(
            entities=RaggedBatchDict(RaggedBufferF32),
            visible=RaggedBatchDict(RaggedBufferBool),
            actions=RaggedBatchDict(RaggedBufferI64),
            logprobs=RaggedBatchDict(RaggedBufferF32),
            logits=None,
            masks=RaggedActionDict(),
            frames=0,
        )

    def push_sample(self, sample: Sample) -> None:
        self.entities.extend(sample.obs.features)
        self.visible.extend(sample.obs.visible)
        self.actions.extend(sample.actions)
        self.logprobs.extend(sample.probs)
        if sample.logits is not None:
            if self.logits is None:
                self.logits = RaggedBatchDict(RaggedBufferF32)
            self.logits.extend(sample.logits)
        self.masks.extend(sample.obs.action_masks)
        self.frames += len(sample.episode)


@dataclass
class Trace:
    action_space: Dict[str, ActionSpace]
    obs_space: ObsSpace
    samples: List[Sample]
    subsample: int = 1

    @classmethod
    def load(cls, path: str, progress_bar: bool = False) -> "Trace":
        with open(path, "rb") as f:
            return cls.deserialize(f.read(), progress_bar=progress_bar)

    @classmethod
    def deserialize(cls, data: bytes, progress_bar: bool = False) -> "Trace":
        samples: List[Sample] = []
        if progress_bar:
            pbar = tqdm.tqdm(total=len(data))

        offset = 0
        # Read version
        version = int(np.frombuffer(data[:8], dtype=np.uint64)[0])
        assert version == 0 or version == 1
        header_len = int(np.frombuffer(data[8:16], dtype=np.uint64)[0])
        header = msgpack_numpy.loads(
            data[16 : 16 + header_len],
            object_hook=ragged_buffer_decode,
            strict_map_key=False,
        )
        action_space = header["act_space"]
        obs_space = header["obs_space"]
        subsample = header.get("subsample", 1)

        offset = 16 + header_len
        while offset < len(data):
            size = int(np.frombuffer(data[offset : offset + 8], dtype=np.uint64)[0])
            offset += 8
            samples.append(Sample.deserialize(data[offset : offset + size]))
            offset += size
            if progress_bar:
                pbar.update(size + 8)
        return Trace(action_space, obs_space, samples, subsample=subsample)

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
                for name, vis in sample.obs.visible.items():
                    if name not in episodes[e].visible:
                        episodes[e].visible[name] = vis[i]
                    else:
                        episodes[e].visible[name].extend(vis[i])
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
            (e for e in episodes.values() if e.complete or include_incomplete),
            key=lambda e: e.number,
        )

    def train_test_split(
        self, test_frac: float = 0.1, progress_bar: bool = False
    ) -> Tuple[MergedSamples, MergedSamples]:
        if self.subsample == 1:
            total_frames = len(self.samples) * len(self.samples[0].episode)
        else:
            total_frames = sum(len(s.episode) for s in self.samples)
        if progress_bar:
            pbar = tqdm.tqdm(total=len(self.samples))

        test = MergedSamples.empty()
        test_episodes: Set[int] = set()
        i = 0
        while test.frames < total_frames * test_frac:
            sample = self.samples[i]
            test_episodes.update(sample.episode)
            test.push_sample(sample)
            i += 1
            if progress_bar:
                pbar.update(1)

        train = MergedSamples.empty()
        for sample in self.samples[i:]:
            # TODO: could be more efficient
            if any(e in test_episodes for e in sample.episode):
                continue
            train.push_sample(sample)
            if progress_bar:
                i += 1
                pbar.update(1)

        return train, test
