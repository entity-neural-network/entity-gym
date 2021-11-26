from typing import Any, List, Dict, Type
from dataclasses import is_dataclass
import numpy as np

from .environment import (
    Entity,
    ObsFilter,
)


def state_space_from_dataclasses(*dss: Type) -> Dict[str, Entity]:
    state_space = {}
    for ds in dss:
        if not is_dataclass(ds):
            raise ValueError(f"{ds} is not a dataclass")
        # TODO: check field types are valid
        state_space[ds.__name__] = Entity(
            features=list(
                [
                    key
                    for key in ds.__dataclass_fields__.keys()
                    if not key.startswith("_")
                ]
            ),
        )
    return state_space


def extract_features(
    entities: Dict[str, List[Any]], obs_filter: ObsFilter
) -> Dict[str, np.ndarray]:
    selectors = {}
    for entity_name, features in obs_filter.entity_to_feats.items():
        selectors[entity_name] = np.array(
            [[getattr(e, f) for f in features] for e in entities[entity_name]],
        ).reshape(-1, len(features))
    return selectors
