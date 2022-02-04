
# Quick Start Guide

This tutorial will walk you through implementing a simple grid-world environment.
The complete implementation can be found in [entity_gym/examples/minesweeper.py]().

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Overview](#overview)
- [Environment](#environment)
- [Observation Space and Action Space](#observation-space-and-action-space)
- [Reset](#reset)
- [Observation](#observation)
- [Actions](#actions)
- [Bonus Topic: Action Masks](#bonus-topic-action-masks)
- [Bonus Topic: Select Entity Actions](#bonus-topic-select-entity-actions)

## Overview

The environment we will implement contains two types of objects, mines and robots.

![Minesweeper](https://user-images.githubusercontent.com/12845088/151688370-4ab0dd31-2dd9-4d25-9a4e-531c24b99865.png)

The player controls all robots in the environment.
On every step, each robot may move in one of four cardinal directions, or stay in place and defuse all adjacent mines.
If a robot defuses a mine, it is removed from the environment.
If a robot steps on a mine, it is removed from the environment and the player loses the game.
The player wins the game when all mines are defused.

## Environment

To define a new environment, start by creating a class that inherits from the [`entity_gym.environment.Environment`]() class:

```python
from typing import List, Tuple
from entity_gym.environment import *

class MineSweeper(Environment):
    def __init__(
        self,
        width: int = 6,
        height: int = 6,
        nmines: int = 5,
        nrobots: int = 2,
    ):
        self.width = width
        self.height = height
        self.nmines = nmines
        self.nrobots = nrobots
        # Positions of robots and mines
        self.robots: List[Tuple[int, int]] = []
        self.mines: List[Tuple[int, int]] = []
    
    ...
```

## Observation Space and Action Space

Next, we define the observation space and action space for our methods.
The observation space has two different types of entities, mines and robots, both of which have an x and y coordinate.
The action space has a single categorical action with five possible choices.

```python
from typing import List, Tuple, Dict
from entity_gym.environment import *

class MineSweeper(Environment):
    ...

    @classmethod
    def obs_space(cls) -> ObsSpace:
        return ObsSpace({
            "Mine": Entity(features=["x", "y"]),
            "Robot": Entity(features=["x", "y"]),
        })
    
    @classmethod
    def action_space(cls) -> Dict[ActionType, ActionSpace]:
        return {
            "Move": CategoricalActionSpace(
                ["Up", "Down", "Left", "Right", "Defuse Mines"],
            ),
        }

```

## Reset

The reset method is called when the environment is first created.
It is used to initialize the environment and reset the state of the environment.
It also returns the initial observation.

```python
import random
from entity_gym.environment import *


class MineSweeper(Environment):
    ...

    def reset(self) -> Observation:
        positions = random.sample(
            [(x, y) for x in range(self.width) for y in range(self.height)],
            self.nmines + self.nrobots,
        )
        self.mines = positions[:self.nmines]
        self.robots = positions[self.nmines:]
        return self.observe()
```

## Observation

The [`Observation`](todo link to docs) class is used to represent the current state of the environment.
The observation class has several fields:
- The `features` attribute is a dictionary of entities, where the key is the name of the entity and the value is a numpy array of float32 values with a shape of (number_entities, number_features) that represent current features of each entity.
- The `ids` attribute is a dictionary of entities, where the key is the name of the entity, and the value is a list of objects that identify all entities of that type. This will become useful later for knowing which entity performed a particular action.
- The `actions` attribute is a dictionary of actions, where the key is the name of the action and the value is a [`ActionMask`](todo link to docs) object that specifies what entities can perform this action.
- The `done` attribute is a boolean that indicates whether the game is over.

```python
import random
from entity_gym.environment import *


class MineSweeper(Environment):
    ...

    def observe(self) -> Observation:
        return Observation.from_entity_obs(
            entities={
                "Robot": EntityObs(
                    features=self.robots,
                    # Identifiers for each Robot
                    ids=[("Robot", i) for i in range(len(self.robots))],
                ),
                # We don't need identifiers for mines since they are not 
                # directly referenced by any actions.
                "Mine": EntityObs(features=self.mines),
            },
            actions={
                "Move": CategoricalActionMask(
                    # Allow all robots to move
                    actor_types=["Robot"],
                ),
            },
            # The game is done once there are no more mines or robots
            done=len(self.mines) == 0 or len(self.robots) == 0,
            # Give reward of 1.0 for defusing all mines
            reward=1.0 if len(self.mines) == 0 else 0,
        )
```

## Actions

Finally, we implement the `act` method that takes an action and returns the next observation.

```python
from entity_gym.environment import *

class MineSweeper(Environment):
    ...

    def act(self, actions: Mapping[ActionType, Action]) -> Observation:
        move = actions["Move"]
        assert isinstance(move, CategoricalAction)
        for (_, i), choice in move.items():
            # Action space is ["Up", "Down", "Left", "Right", "Defuse Mines"],
            x, y = self.robots[i]
            if choice == 0 and y < self.height - 1:
                self.robots[i] = (x, y + 1)
            elif choice == 1 and y > 0:
                self.robots[i] = (x, y - 1)
            elif choice == 2 and x > 0:
                self.robots[i] = (x - 1, y)
            elif choice == 3 and x < self.width - 1:
                self.robots[i] = (x + 1, y)
            elif choice == 4:
                # Remove all mines adjacent to this robot
                rx, ry = self.robots[i]
                self.mines = [
                    (x, y)
                    for (x, y) in self.mines
                    if abs(x - rx) + abs(y - ry) > 1
                ]

        # Remove all robots that stepped on a mine
        self.robots = [
            (x, y)
            for (x, y) in self.robots
            if (x, y) not in self.mines
        ]

        return self.observe() 
```

## Bonus Topic: Action Masks

Currently, robots may move in any direction, but any movement that would take a robot outside the grid will be ignored.
We may want to restrict the robots choices so that they cannot move outside the grid.
We can do this by setting the `mask` attribute of the [`ActionMask`](todo link to docs) object to a boolean array of shape (number_entities, number_actions) that specifies which actions are allowed.

```python

import random
from entity_gym.environment import *


class MineSweeper(Environment):
    ...

    def valid_moves(self, x: int, y: int) -> List[bool]:
        return [
            x < self.width - 1,
            x > 0,
            y < self.height - 1,
            y > 0,
            # Always allow staying in place and defusing mines
            True,
        ]

    def observe(self) -> Observation:
        return Observation(
            actions={
                "Move": CategoricalActionMask(
                    # Allow all robots to move
                    actor_types=["Robot"],
                    mask=[
                        self.valid_moves(x, y)
                        for (x, y) in self.robots
                    ],
                ),
            },
            ...
        )
```

## Bonus Topic: Select Entity Actions

Suppose we want to add a new _Orbital Cannon_ entity to the game that can fire a laser at any mine or robot every 5 steps.
Since the number of mines and robots is unknown, we cannot use a normal categorical action for the Orbital Cannon.
Instead, we will use a [`SelectEntityAction`](todo link to docs), which allows us to select one entity from a list of entities.


```python
from entity_gym.environment import *

class MineSweeper(Environment):
    ...

    @classmethod
    def obs_space(cls) -> ObsSpace:
        return ObsSpace({
            "Mine": Entity(features=["x", "y"]),
            "Robot": Entity(features=["x", "y"]),
            # The Orbital Cannon entity
            "Orbital Cannon": Entity(["cooldown"]),
        })
    
    @classmethod
    def action_space(cls) -> ActionSpace:
        return ActionSpace({
            "Move": CategoricalAction(
                ["Up", "Down", "Left", "Right", "Defuse Mines"]
            ),
            # New action for firing laser
            "Fire Orbital Cannon": SelectEntityActionSpace(),
        })
    
    

    def reset(self) -> Observation:
        ...
        # Set orbital cannon cooldown to 5
        self.orbital_cannon_cooldown = 5
        return self.observe()
    
    def observe(self) -> Observation:
        return Observation.from_entity_obs(
            entities={
                "Mine": EntityObs(
                    features=self.mines,
                    ids=[("Mine", i) for i in range(len(self.mines))],
                ),
                "Robot": EntityObs(
                    features=self.robots,
                    ids=[("Robot", i) for i in range(len(self.robots))],
                ),
                "Orbital Cannon": EntityObs(
                    features=[(self.orbital_cannon_cooldown,)],
                    ids=[("Orbital Cannon", 0)],
                )
            },
            actions={
                "Move": DenseCategoricalActionMask(
                    actor_types=["Robot"],
                ),
                "Fire Orbital Cannon": SelectEntityActionMask(
                    # Only the Orbital Cannon can fire, but not if cooldown > 0
                    actor_types=["Orbital Cannon"] if self.orbital_cannon_cooldown == 0 else [],
                    # Both mines and robots can be fired at
                    actee_types=["Mine", "Robot"],
                ),
            },
            done=len(self.mines) == 0 or len(self.robots) == 0,
            reward=1.0 if len(self.mines) == 0 else 0,
        )
    
    def act(self, actions: Mapping[ActionType, Action]) -> Observation:
        fire = actions["Fire Orbital Cannon"]
        assert isinstance(fire, SelectEntityAction)
        remove_robot = None
        for (entity_type, i) in fire.actees:
            if entity_type == "Mine":
                self.mines.remove(self.mines[i])
            elif entity_type == "Robot":
                # Don't remove yet to keep indices valid
                remove_robot = i

        move = actions["Move"]
        ...

        if remove_robot is not None:
            self.robots.pop(remove_robot)
        # Remove all robots that stepped on a mine
        self.robots = [r for r in self.robots if r not in self.mines]

        return self.observe

```