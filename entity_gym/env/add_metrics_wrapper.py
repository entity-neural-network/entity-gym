from typing import Any, Dict, Mapping, Optional

import numpy as np
import numpy.typing as npt
from ragged_buffer import RaggedBufferI64

from entity_gym.env.environment import ActionName, ActionSpace, ObsSpace
from entity_gym.env.vec_env import Metric, VecEnv, VecObs


class AddMetricsWrapper(VecEnv):
    def __init__(
        self, env: VecEnv, filter: Optional[npt.NDArray[np.bool8]] = None
    ) -> None:
        """
        Wrap a VecEnv to track and add metrics for (episodic) rewards and episode lengths.

        Args:
            env: The VecEnv to wrap.
            filter: A boolean array of length len(env) indicating which environments to
                track metrics for. If filter[i] is True, then the metrics for the i-th
                environment will be tracked.
        """
        self.env = env
        self.entity_types = list(env.obs_space().entities.keys())
        if env.has_global_entity():
            self.entity_types.append("__global__")
        self.total_reward = np.zeros(len(env), dtype=np.float32)
        self.total_steps = np.zeros(len(env), dtype=np.int64)
        self.filter = np.ones(len(env), dtype=np.bool8) if filter is None else filter

    def reset(self, obs_config: ObsSpace) -> VecObs:
        return self.track_metrics(self.env.reset(obs_config))

    def act(
        self, actions: Mapping[ActionName, RaggedBufferI64], obs_filter: ObsSpace
    ) -> VecObs:
        return self.track_metrics(self.env.act(actions, obs_filter))

    def render(self, **kwargs: Any) -> npt.NDArray[np.uint8]:
        return self.env.render(**kwargs)

    def __len__(self) -> int:
        return len(self.env)

    def close(self) -> None:
        self.env.close()

    def track_metrics(self, obs: VecObs) -> VecObs:
        self.total_reward += obs.reward
        self.total_steps += 1
        episodic_reward = Metric()
        episodic_length = Metric()
        count = len(self.total_steps)
        obs.metrics["step"] = Metric(
            sum=self.total_steps.sum(),
            count=count,
            min=self.total_steps.min(),
            max=self.total_steps.max(),
        )

        for entity in self.entity_types:
            if entity in obs.features:
                _sum = obs.features[entity].items()
                counts = obs.features[entity].size1()
                _min = counts.min()
                _max = counts.max()
            else:
                _sum = 0
                _min = 0
                _max = 0
            obs.metrics[f"entity_count/{entity}"] = Metric(
                sum=_sum, count=count, min=_min, max=_max
            )
        if len(obs.features) > 0:
            combined_counts: Any = sum(
                features.size1() for features in obs.features.values()
            )
        else:
            combined_counts = np.zeros(count, dtype=np.int64)
        obs.metrics["entity_count"] = Metric(
            sum=combined_counts.sum(),
            count=count,
            min=combined_counts.min(),
            max=combined_counts.max(),
        )

        for i in np.arange(len(self))[obs.done & self.filter]:
            episodic_reward.push(self.total_reward[i])
            episodic_length.push(self.total_steps[i])
            self.total_reward[i] = 0.0
            self.total_steps[i] = 0
        obs.metrics["episodic_reward"] = episodic_reward
        obs.metrics["episode_length"] = episodic_length
        obs.metrics["reward"] = Metric(
            sum=obs.reward.sum(),
            count=obs.reward.size,
            min=obs.reward.min(),
            max=obs.reward.max(),
        )
        return obs

    def action_space(self) -> Dict[ActionName, ActionSpace]:
        return self.env.action_space()

    def obs_space(self) -> ObsSpace:
        return self.env.obs_space()
