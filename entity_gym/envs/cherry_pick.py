from dataclasses import dataclass, field
import numpy as np
from typing import Dict, List, Mapping

from entity_gym.environment import (
    DenseSelectEntityActionMask,
    Entity,
    Environment,
    EpisodeStats,
    ObsSpace,
    SelectEntityAction,
    SelectEntityActionSpace,
    ActionSpace,
    Observation,
    Action,
)


@dataclass
class CherryPick(Environment):
    """
    The CherryPick environment is initalized with a list of 32 cherries of random quality.
    On each timestep, the player can pick up one of the cherries.
    The player receives a reward of the quality of the cherry picked.
    The environment ends after 16 steps.
    The quality of the top 16 cherries is normalized so that the maximum total achievable reward is 1.0.
    """

    num_cherries: int = 32
    cherries: List[float] = field(default_factory=list)
    last_reward: float = 0.0
    step: int = 0

    @classmethod
    def obs_space(cls) -> ObsSpace:
        return ObsSpace(
            {
                "Cherry": Entity(["quality"]),
                "Player": Entity([]),
            }
        )

    @classmethod
    def action_space(cls) -> Dict[str, ActionSpace]:
        return {"Pick Cherry": SelectEntityActionSpace()}

    def _reset(self) -> Observation:
        cherries = [np.random.normal() for _ in range(self.num_cherries)]
        # Normalize so that the sum of the top half is 1.0
        top_half = sorted(cherries, reverse=True)[: self.num_cherries // 2]
        sum_top_half = sum(top_half)
        add = 2.0 * (1.0 - sum_top_half) / self.num_cherries
        self.cherries = [c + add for c in cherries]
        self.last_reward = 0.0
        self.step = 0
        self.total_reward = 0.0
        return self.observe()

    def observe(self) -> Observation:
        done = self.step == self.num_cherries // 2
        return Observation(
            entities={
                "Cherry": np.array(self.cherries, dtype=np.float32).reshape(-1, 1),
                "Player": np.zeros([1, 0], dtype=np.float32),
            },
            ids=np.arange(len(self.cherries) + 1),
            action_masks={
                "Pick Cherry": DenseSelectEntityActionMask(
                    actors=np.array([len(self.cherries)]),
                    actees=np.arange(len(self.cherries)),
                    mask=(
                        np.arange(len(self.cherries) + 1) < len(self.cherries)
                    ).astype(np.float32),
                ),
            },
            reward=self.last_reward,
            done=done,
            end_of_episode_info=EpisodeStats(self.step, self.total_reward)
            if done
            else None,
        )

    def _act(self, action: Mapping[str, Action]) -> Observation:
        assert len(action) == 1
        a = action["Pick Cherry"]
        assert isinstance(a, SelectEntityAction)
        self.last_reward = self.cherries.pop(a.actions[0][1])
        self.total_reward += self.last_reward
        self.step += 1
        return self.observe()
