from dataclasses import dataclass
from typing import Dict, List, Optional

from ragged_buffer import RaggedBufferF32
import tqdm
import numpy as np

from entity_gym.environment import (
    Action,
    ObsSpace,
)
from entity_gym.serialization.sample_recorder import Sample


@dataclass
class Episode:
    number: int
    steps: int
    entities: Dict[str, RaggedBufferF32]
    actions: Dict[str, List[Action]]
    logprobs: Dict[str, RaggedBufferF32]
    logits: Dict[str, RaggedBufferF32]
    total_reward: float
    complete: bool = False


@dataclass
class Trace:
    action_space: Dict[str, int]
    obs_space: ObsSpace
    samples: List[Sample]

    @classmethod
    def deserialize(cls, data: bytes, progress_bar: bool = False) -> "Trace":
        samples: List[Sample] = []
        if progress_bar:
            pbar = tqdm.tqdm(total=len(data))

        offset = 0
        while offset < len(data):
            size = int(np.frombuffer(data[offset : offset + 8], dtype=np.uint64)[0])
            offset += 8
            samples.append(Sample.deserialize(data[offset : offset + size]))
            offset += size
            if progress_bar:
                pbar.update(size + 8)
        return Trace(None, None, samples)  # type: ignore

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
                    episodes[e] = Episode(e, 0, {}, {}, {}, {}, 0.0)

                episodes[e].steps += 1
                episodes[e].total_reward += sample.obs.reward[i]
                if sample.obs.done[i] and prev_episodes is not None:
                    episodes[prev_episodes[i]].complete = True

                for name, feats in sample.obs.features.items():
                    if name not in episodes[e].entities:
                        episodes[e].entities[name] = feats[i]
                    else:
                        episodes[e].entities[name].extend(feats[i])
                for name, acts in sample.actions[i].items():
                    if name not in episodes[e].actions:
                        episodes[e].actions[name] = [acts]
                    else:
                        episodes[e].actions[name].append(acts)
                for name, logprobs in sample.probs.items():
                    if name not in episodes[e].logprobs:
                        episodes[e].logprobs[name] = logprobs[i]
                    else:
                        episodes[e].logprobs[name].extend(logprobs[i])
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
