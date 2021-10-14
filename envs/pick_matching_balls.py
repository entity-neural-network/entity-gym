from dataclasses import dataclass, field
import numpy as np
import random
from typing import List

from entity_gym.environment import Environment, Entity, SelectEntity, ActionSpace, ObsConfig, Observation, ActionMask, Action


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
    balls: List[Ball] = field(default_factory=list)

    @classmethod
    def entities(cls) -> List[Entity]:
        return [
            Entity(
                name="Ball",
                # TODO: better support for categorical features
                features=["color", "selected"],
            ),
            Entity(
                name="Player",
                features=[],
            ),
        ]

    @classmethod
    def action_space(cls) -> List[ActionSpace]:
        return [SelectEntity("Pick Ball")]

    def reset(self, obs_config: ObsConfig) -> Observation:
        self.balls = [
            Ball(color=random.randint(0, 5)) for _ in range(32)
        ]
        return self.filter_obs(self.observe(), obs_config)

    def observe(self) -> Observation:
        done = len({b.color for b in self.balls if b.selected}
                   ) > 1 or all(b.selected for b in self.balls)
        if done:
            reward = (sum(b.selected for b in self.balls) - 1) / max(
                [len([b for b in self.balls if b.color == color]) for color in range(6)])
        else:
            reward = 0.0

        return Observation(
            entities=[
                ("Ball", np.array(
                    [[float(b.color), float(b.selected)] for b in self.balls])),
                ("Player", np.zeros([1, 0])),
            ],
            ids=np.arange(len(self.balls) + 1),
            action_masks=[
                ("Pick Ball",
                 ActionMask(
                     actors=[len(self.balls)],
                     mask=np.array(
                         [not b.selected for b in self.balls] + [False]).astype(np.float32),
                 )
                 ),
            ],
            reward=reward,
            done=done,
        )

    def act(self, actions: Action, obs_config: ObsConfig) -> Observation:
        for action_name, action_choices in actions.chosen_actions.items():
            assert not self.balls[action_choices[0][1]].selected
            self.last_reward = self.balls[action_choices[0][1]].selected = True
        return self.filter_obs(self.observe(), obs_config)
