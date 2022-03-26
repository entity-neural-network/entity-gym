from typing import Any, Dict, List, Mapping, Type

import numpy as np
import numpy.typing as npt
from ragged_buffer import RaggedBufferI64

from entity_gym.environment.environment import (
    Action,
    ActionType,
    CategoricalAction,
    CategoricalActionSpace,
    EntityID,
    Environment,
    Observation,
    ObsSpace,
    SelectEntityAction,
    SelectEntityActionMask,
    SelectEntityActionSpace,
)
from entity_gym.environment.vec_env import VecEnv, VecObs, batch_obs


class EnvList(VecEnv):
    def __init__(
        self, env_cls: Type[Environment], env_kwargs: Dict[str, Any], num_envs: int
    ):
        self.envs = [env_cls(**env_kwargs) for _ in range(num_envs)]  # type: ignore
        self.cls = env_cls
        self.last_obs: List[Observation] = []

    def env_cls(cls) -> Type[Environment]:
        return cls.cls

    def reset(self, obs_space: ObsSpace) -> VecObs:
        batch = self._batch_obs([e.reset_filter(obs_space) for e in self.envs])
        return batch

    def render(self, **kwargs: Any) -> npt.NDArray[np.uint8]:
        return np.stack([e.render(**kwargs) for e in self.envs])

    def close(self) -> None:
        for env in self.envs:
            env.close()

    def act(
        self, actions: Mapping[str, RaggedBufferI64], obs_space: ObsSpace
    ) -> VecObs:
        observations = []
        action_space = self.cls.action_space()
        for i, env in enumerate(self.envs):
            _actions: Dict[ActionType, Action] = {}
            for atype, action in actions.items():
                mask = self.last_obs[i].actions[atype]
                if mask.actor_ids is not None:
                    actors = mask.actor_ids
                elif mask.actor_types is not None:
                    actors = []
                    for etype in mask.actor_types:
                        actors.extend(self.last_obs[i].ids[etype])
                else:
                    actors = []
                    for ids in self.last_obs[i].ids.values():
                        actors.extend(ids)
                if isinstance(action_space[atype], CategoricalActionSpace):
                    _actions[atype] = CategoricalAction(
                        actors=actors,
                        actions=action[i].as_array().reshape(-1),
                    )
                elif isinstance(action_space[atype], SelectEntityActionSpace):
                    assert isinstance(mask, SelectEntityActionMask)
                    if mask.actee_types is not None:
                        index_to_actee: List[EntityID] = []
                        for etype in mask.actee_types:
                            index_to_actee.extend(self.last_obs[i].ids[etype])
                        actees = [
                            index_to_actee[a] for a in action[i].as_array().reshape(-1)
                        ]
                    elif mask.actee_ids is not None:
                        actees = [
                            mask.actee_ids[e] for e in action[i].as_array().reshape(-1)
                        ]
                    else:
                        index_to_id = self.last_obs[i].index_to_id(obs_space)
                        actees = [
                            index_to_id[e] for e in action[i].as_array().reshape(-1)
                        ]
                    _actions[atype] = SelectEntityAction(
                        actors=actors,
                        actees=actees,
                    )
                else:
                    raise NotImplementedError(
                        f"Action space type {type(action_space[atype])} not supported"
                    )

            obs = env.act_filter(_actions, obs_space)
            if obs.done:
                # TODO: something is wrong with the interface here
                new_obs = env.reset_filter(obs_space)
                new_obs.done = True
                new_obs.reward = obs.reward
                new_obs.end_of_episode_info = obs.end_of_episode_info
                observations.append(new_obs)
            else:
                observations.append(obs)
        return self._batch_obs(observations)

    def _batch_obs(self, obs: List[Observation]) -> VecObs:
        self.last_obs = obs
        return batch_obs(obs, self.cls.obs_space(), self.cls.action_space())

    def __len__(self) -> int:
        return len(self.envs)
