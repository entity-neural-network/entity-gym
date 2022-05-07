import random
from typing import Dict, Mapping, Tuple

from entity_gym.cli_runner import CliRunner
from entity_gym.environment import *


class TreasureHunt(Environment):
    def __init__(self) -> None:
        self.x_pos = 0
        self.y_pos = 0

    def obs_space(self) -> ObsSpace:
        return ObsSpace(
            global_features=["x_pos", "y_pos"],
            entities={
                "Trap": Entity(features=["x_pos", "y_pos"]),
                "Treasure": Entity(features=["x_pos", "y_pos"]),
            },
        )

    def action_space(self) -> Dict[str, ActionSpace]:
        # The `GlobalCategoricalActionSpace` allows the agent to choose from set of discrete actions.
        return {
            "move": GlobalCategoricalActionSpace(
                choices=["up", "down", "left", "right"]
            )
        }

    def _random_empty_pos(self) -> Tuple[int, int]:
        # Generate a random position on the grid that is not occupied by a trap, treasure, or player.
        while True:
            x = random.randint(-5, 5)
            y = random.randint(-5, 5)
            if (x, y) not in (self.traps + self.treasure + [(self.x_pos, self.y_pos)]):
                return x, y

    def reset(self) -> Observation:
        self.x_pos = 0
        self.y_pos = 0
        self.game_over = False
        self.traps = []
        self.treasure = []
        for _ in range(5):
            self.traps.append(self._random_empty_pos())
        for _ in range(5):
            self.treasure.append(self._random_empty_pos())
        return self._observe()

    def act(self, actions: Mapping[ActionName, Action]) -> Observation:
        action = actions["move"]
        assert isinstance(action, GlobalCategoricalAction)
        if action.choice == "up" and self.y_pos < 10:
            self.y_pos += 1
        elif action.choice == "down" and self.y_pos > -10:
            self.y_pos -= 1
        elif action.choice == "left" and self.x_pos > -10:
            self.x_pos -= 1
        elif action.choice == "right" and self.x_pos < 10:
            self.x_pos += 1

        reward = 0.0
        if (self.x_pos, self.y_pos) in self.treasure:
            reward = 1.0
            self.treasure.remove((self.x_pos, self.y_pos))
        if (self.x_pos, self.y_pos) in self.traps or len(self.treasure) == 0:
            self.game_over = True

        return self._observe(reward)

    def _observe(self, reward: float = 0.0) -> Observation:
        return Observation(
            global_features=[self.x_pos, self.y_pos],
            features={
                "Trap": self.traps,
                "Treasure": self.treasure,
            },
            done=self.game_over,
            reward=reward,
            actions={"move": GlobalCategoricalActionMask()},
        )


if __name__ == "__main__":
    env = TreasureHunt()
    CliRunner(env).run()
