from typing import Any, Sequence, Union

import numpy as np
import numpy.typing as npt

Features = Union[npt.NDArray[np.float32], Sequence[Sequence[float]]]
EntityID = Any
EntityName = str
ActionName = str
