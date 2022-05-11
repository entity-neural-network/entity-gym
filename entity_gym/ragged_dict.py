from dataclasses import dataclass, field
from typing import Dict, Generic, Mapping, Type, TypeVar

import numpy as np
import numpy.typing as npt
from ragged_buffer import RaggedBuffer

from .env.vec_env import VecActionMask

ScalarType = TypeVar("ScalarType", bound=np.generic, covariant=True)


@dataclass
class RaggedBatchDict(Generic[ScalarType]):
    rb_cls: Type[RaggedBuffer[ScalarType]]
    buffers: Dict[str, RaggedBuffer[ScalarType]] = field(default_factory=dict)

    def extend(self, batch: Mapping[str, RaggedBuffer[ScalarType]]) -> None:
        for k, v in batch.items():
            if k not in self.buffers:
                self.buffers[k] = v
            else:
                self.buffers[k].extend(v)

    def clear(self) -> None:
        for buffer in self.buffers.values():
            buffer.clear()

    def __getitem__(
        self, index: npt.NDArray[np.int64]
    ) -> Dict[str, RaggedBuffer[ScalarType]]:
        return {k: v[index] for k, v in self.buffers.items()}


@dataclass
class RaggedActionDict:
    buffers: Dict[str, VecActionMask] = field(default_factory=dict)

    def extend(self, batch: Mapping[str, VecActionMask]) -> None:
        for k, v in batch.items():
            if k not in self.buffers:
                self.buffers[k] = v
            else:
                self.buffers[k].extend(v)

    def clear(self) -> None:
        for buffer in self.buffers.values():
            buffer.clear()

    def __getitem__(self, index: npt.NDArray[np.int64]) -> Dict[str, VecActionMask]:
        return {k: v[index] for k, v in self.buffers.items()}
