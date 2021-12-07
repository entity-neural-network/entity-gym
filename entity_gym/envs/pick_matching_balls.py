from dataclasses import dataclass, field
import numpy as np
import random
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
class Ball:
    color: int
    selected: bool = False


@dataclass
class PickMatchingBalls(Environment):
    """
    The PickMatchingBalls environment is initalized with a list of 32 balls of different colors.
    On each timestamp, the player can pick up one of the balls.
    The episode ends when the player picks up a ball of a different color from the last one.
    The player receives a reward equal to the number of balls picked up divided by the maximum number of balls of the same color.
    """

    num_balls: int = 32
    balls: List[Ball] = field(default_factory=list)

    @classmethod
    def obs_space(cls) -> ObsSpace:
        return ObsSpace(
            {
                "Ball": Entity(
                    # TODO: better support for categorical features
                    ["color", "selected"],
                ),
                "Player": Entity([]),
            }
        )

    @classmethod
    def action_space(cls) -> Dict[str, ActionSpace]:
        return {"Pick Ball": SelectEntityActionSpace()}

    def _reset(self) -> Observation:
        self.balls = [Ball(color=random.randint(0, 5)) for _ in range(self.num_balls)]
        self.step = 0
        return self.observe()

    def observe(self) -> Observation:
        done = len({b.color for b in self.balls if b.selected}) > 1 or all(
            b.selected for b in self.balls
        )
        if done:
            if self.step == self.num_balls:
                reward = 1.0
            else:
                reward = (sum(b.selected for b in self.balls) - 1) / max(
                    [
                        len([b for b in self.balls if b.color == color])
                        for color in range(6)
                    ]
                )
        else:
            reward = 0.0

        return Observation(
            entities={
                "Ball": np.array(
                    [[float(b.color), float(b.selected)] for b in self.balls],
                    dtype=np.float32,
                ),
                "Player": np.zeros([1, 0], dtype=np.float32),
            },
            ids=np.arange(len(self.balls) + 1),
            action_masks={
                "Pick Ball": DenseSelectEntityActionMask(
                    actors=np.array([len(self.balls)]),
                    actees=np.array(
                        [i for i, b in enumerate(self.balls) if not b.selected]
                    ),
                ),
            },
            reward=reward,
            done=done,
            end_of_episode_info=EpisodeStats(self.step, reward) if done else None,
        )

    def _act(self, actions: Mapping[str, Action]) -> Observation:
        action = actions["Pick Ball"]
        assert isinstance(action, SelectEntityAction)
        for _, selected_ball in action.actions:
            assert not self.balls[selected_ball].selected
            self.balls[selected_ball].selected = True
        self.step += 1
        return self.observe()
