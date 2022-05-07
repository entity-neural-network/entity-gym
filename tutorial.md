# Quick Start Guide

This tutorial will guide you through the process of creating a simple entity gym environment.

## 1. Installation

First, install the entity_gym package:

```bash
pip install entity_gym
```

## 2. The `Environment` class

Create a new file `treasure_hunt.py` with the following contents:

```python
from typing import Dict, Mapping
from entity_gym.cli_runner import CliRunner
from entity_gym.environment import *

# The `Environment` class defines the interface that all entity gym environments must implement.
class TreasureHunt(Environment):
    # The `obs_space` specifies the shape of observations returned by the environment.
    def obs_space(self) -> ObsSpace:
        return ObsSpace()

    # The `action_space` specifies what actions that can be performed by the agent.
    def action_space(self) -> Dict[str, ActionSpace]:
        return {}

    # `reset` should initialize the environment and return the initial observation.
    def reset(self) -> Observation:
        return Observation.empty()

    # `act` performs the chosen actions and returns the new observation.
    def act(self, actions: Mapping[ActionType, Action]) -> Observation:
        return Observation.empty()


if __name__ == "__main__":
    env = TreasureHunt()
    # The `CliRunner` can run any environment with a command line interface.
    CliRunner(env).run()
```

Try it out by running the following command:

```bash
python treasure_hunt.py
```

Since we haven't implemented any functionality for our environment, this won't do much yet.
However, you should still see something like the following output:

```
Environment: TreasureHunt

Step 0
Reward: 0.0
Total: 0.0
Press ENTER to continue, CTRL-C to exit
```

## 3. Keeping track of Player position

Let's add some logic to keep track of the player's position and expose it in observations:

```python
class TreasureHunt(Environment):
    def __init__(self) -> None:
        self.x_pos = 0
        self.y_pos = 0

    def obs_space(self) -> ObsSpace:
        # `global_features` adds a fixed-length vector of features to the observation.
        return ObsSpace(global_features=["x_pos", "y_pos"])

    def reset(self) -> Observation:
        self.x_pos = 0
        self.y_pos = 0
        return self._observe()

    def _observe(self) -> Observation:
        return Observation(
            global_features=[self.x_pos, self.y_pos], done=False, reward=0
        )

    def act(self, actions: Mapping[ActionType, Action]) -> Observation:
        return self._observe()

    def action_space(self) -> Dict[str, ActionSpace]:
        return {}
```

If you run the environment again, you should now see it print out the player's position:

```
Environment: TreasureHunt
Global features: x_pos, y_pos

Step 0
Reward: 0
Total: 0
Global features: x_pos=0, y_pos=0
Press ENTER to continue, CTRL-C to exit
```

## 4. Adding a "move" action

Now that the player has a position, we can add an action that moves the player.
Change the `action_space` and `act` methods to the following:

```python
class TreasureHunt(Environment):
    ...

    def action_space(self) -> Dict[str, ActionSpace]:
        # The `GlobalCategoricalActionSpace` allows the agent to choose from set of discrete actions.
        return {
            "move": GlobalCategoricalActionSpace(
                choices=["up", "down", "left", "right"]
            )
        }

    def act(self, actions: Mapping[ActionType, Action]) -> Observation:
        # Adjust the player's position according to the chosen action.
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
        return self._observe()

    def _observe(self) -> Observation:
        return Observation(
            global_features=[self.x_pos, self.y_pos],
            done=False,
            reward=0,
            # Each `Observation` must specify which actions are available on the current step.
            actions={"move": GlobalCategoricalActionMask()},
        )
```

You can now move the player:

```
Environment: TreasureHunt
Global features: x_pos, y_pos
Categorical move: up, down, left, right

Step 0
Reward: 0
Total: 0
Global features: x_pos=0, y_pos=0
Choose move (0/up 1/down 2/left 3/right)
0
Step 1
Reward: 0
Total: 0
Global features: x_pos=0, y_pos=1
Choose move (0/up 1/down 2/left 3/right)
3
Step 2
Reward: 0
Total: 0
Global features: x_pos=1, y_pos=1
Choose move (0/up 1/down 2/left 3/right)
```

## 5. Treasure and trap entities

Now, we are going to place additional entities in the environment.
- _Treasure_ can be collected by the player and increases the player's score by 1.0. Once all treasures are collected, the game is won.
- Moving onto a _trap_ immediately ends the game.

```python
import random
from typing import Mapping, Tuple, Dict

class TreasureHunt(Environment):
    def __init__(self):
        self.x_pos = 0
        self.y_pos = 0
        self.traps = []
        self.treasure = []

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

    def obs_space(self) -> ObsSpace:
        return ObsSpace(
            global_features=["x_pos", "y_pos"],
            # An observation space can have several entities with different features.
            # On any given step, an observation may include any number of the defined entities.
            entities={
                "Trap": Entity(features=["x_pos", "y_pos"]),
                "Treasure": Entity(features=["x_pos", "y_pos"]),
            }
        )

    def act(self, actions: Mapping[ActionType, Action]) -> Observation:
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
```

If you run the environment again, you will now see and be able to interact with all the entities:

```
Environment: TreasureHunt
Global features: x_pos, y_pos
Entity Trap: x_pos, y_pos
Entity Treasure: x_pos, y_pos
Categorical move: up, down, left, right

Step 0
Reward: 0.0
Total: 0.0
Global features: x_pos=0, y_pos=0
Entities
0 Trap(x_pos=-2, y_pos=5)
1 Trap(x_pos=-1, y_pos=-4)
2 Trap(x_pos=0, y_pos=2)
3 Trap(x_pos=-5, y_pos=-3)
4 Trap(x_pos=4, y_pos=3)
5 Treasure(x_pos=-3, y_pos=3)
6 Treasure(x_pos=3, y_pos=4)
7 Treasure(x_pos=5, y_pos=5)
8 Treasure(x_pos=-1, y_pos=-5)
9 Treasure(x_pos=5, y_pos=3)
Choose move (0/up 1/down 2/left 3/right)
```

TODO: links to other docs