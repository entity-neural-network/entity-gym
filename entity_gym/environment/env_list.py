from typing import (
    Any,
    Dict,
    Mapping,
    Sequence,
    Type,
)
from entity_gym.environment.environment import (
    Action,
    Environment,
    ObsSpace,
)
from entity_gym.environment.vec_env import (
    ObsBatch,
    VecEnv,
    batch_obs,
)


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
