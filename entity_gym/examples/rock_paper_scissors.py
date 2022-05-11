import random
from dataclasses import dataclass
from typing import Dict, Mapping

from entity_gym.dataclass_utils import extract_features, obs_space_from_dataclasses
from entity_gym.env import (
    Action,
    ActionSpace,
    CategoricalAction,
    CategoricalActionMask,
    CategoricalActionSpace,
    Environment,
    Observation,
    ObsSpace,
)


@dataclass
class Player:
    pass


@dataclass
class Opponent:
    rock: float = 0.0
    paper: float = 0.0
    scissors: float = 0.0


class RockPaperScissors(Environment):
    """
    This environment tests giving additional information to the value function
    which can not be observed by the policy.

    On each timestep, the opponent randomly chooses rock, paper or scissors with
    probability of 50%, 30% and 20% respectively. The value function can observe
    the opponent's choice, but the policy can not.
    The agent must choose either rock, paper or scissors. If the agent beats the
    opponent, the agent receives a reward of 2.0, otherwise it receives a reward of 0.0.
    The optimal strategy is to always choose paper for an average reward of 1.0.
    Since the value function can observe the opponent's choice, it can perfectly
    predict reward.
    """

    def __init__(self, cheat: bool = False) -> None:
        self.cheat = cheat
        self.reset()

    def obs_space(self) -> ObsSpace:
        return obs_space_from_dataclasses(Player, Opponent)

    def action_space(self) -> Dict[str, ActionSpace]:
        return {"throw": CategoricalActionSpace(["rock", "paper", "scissors"])}

    def reset_filter(self, obs_space: ObsSpace) -> Observation:
        rand = random.random()
        if rand < 0.5:
            self.opponent = Opponent(rock=1.0)
        elif rand < 0.8:
            self.opponent = Opponent(paper=1.0)
        else:
            self.opponent = Opponent(scissors=1.0)
        return self.observe(obs_space)

    def reset(self) -> Observation:
        return self.reset_filter(self.obs_space())

    def act_filter(
        self, action: Mapping[str, Action], obs_filter: ObsSpace
    ) -> Observation:
        reward = 0.0
        for action_name, a in action.items():
            assert isinstance(a, CategoricalAction)
            if action_name == "throw":
                if (
                    (a.indices[0] == 0 and self.opponent.scissors == 1.0)
                    or (a.indices[0] == 1 and self.opponent.rock == 1.0)
                    or (a.indices[0] == 2 and self.opponent.paper == 1.0)
                ):
                    reward = 2.0
        return self.observe(obs_filter, done=True, reward=reward)

    def act(self, actions: Mapping[str, Action]) -> Observation:
        return self.act_filter(
            actions,
            self.obs_space(),
        )

    def observe(
        self, obs_filter: ObsSpace, done: bool = False, reward: float = 0.0
    ) -> Observation:
        return Observation(
            features=extract_features(
                {
                    "Player": [Player()],
                    "Opponent": [self.opponent],
                },
                obs_filter,
            ),
            actions={
                "throw": CategoricalActionMask(actor_ids=[0]),
            },
            ids={"Player": [0]},
            visible={"Opponent": [self.cheat]},
            reward=reward,
            done=done,
        )
