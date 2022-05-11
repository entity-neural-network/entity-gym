from dataclasses import dataclass
from typing import Dict, Mapping

from entity_gym.env import Action, ActionSpace, Environment, Observation, ObsSpace
from entity_gym.env.environment import (
    GlobalCategoricalAction,
    GlobalCategoricalActionMask,
    GlobalCategoricalActionSpace,
)


@dataclass
class MultiArmedBandit(Environment):
    """
    Task with single cateorical action with 5 choices which gives a reward of 1 for choosing action 0 and reward of 0 otherwise.
    """

    def obs_space(cls) -> ObsSpace:
        return ObsSpace(global_features=["step"])

    def action_space(cls) -> Dict[str, ActionSpace]:
        return {
            "pull": GlobalCategoricalActionSpace(["A", "B", "C", "D", "E"]),
        }

    def reset(self) -> Observation:
        self.step = 0
        self._total_reward = 0.0
        return self.observe()

    def act(self, actions: Mapping[str, Action]) -> Observation:
        self.step += 1

        a = actions["pull"]
        assert isinstance(
            a, GlobalCategoricalAction
        ), f"{a} is not a GlobalCategoricalAction"
        if a.label == "A":
            reward = 1 / 32.0
        else:
            reward = 0
        done = self.step >= 32
        self._total_reward += reward
        return self.observe(done, reward)

    def observe(self, done: bool = False, reward: float = 0) -> Observation:
        return Observation(
            global_features=[self.step],
            actions={
                "pull": GlobalCategoricalActionMask(),
            },
            reward=reward,
            done=done,
        )
