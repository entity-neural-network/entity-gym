=====================
Complex Action Spaces
=====================

This tutorial walks you through implementing a grid-world environment in which the player controls multiple entities at the same time.
You will learn how to use the ``CategoricalActionSpace`` to allow multiple entities perform an action, use action masks to limit the set of available action choices, and use the ``SelectEntiyActionSpace`` to implement an action that allows entities to select other entities.

An extended version of the environment implemented in this tutorial can be found in `entity_gym/examples/minesweeper.py <https://github.com/entity-neural-network/entity-gym/blob/main/entity_gym/examples/minesweeper.py>`_.

.. toctree::

Overview
========

The environment we will implement contains two types of objects, mines and robots.

.. image:: https://user-images.githubusercontent.com/12845088/151688370-4ab0dd31-2dd9-4d25-9a4e-531c24b99865.png

The player controls all robots in the environment.
On every step, each robot may move in one of four cardinal directions, or stay in place and defuse all adjacent mines.
If a robot defuses a mine, the mine is removed from the environment.
If a robot steps on a mine, the robot is removed from the environment.
If there are no more robots, the player loses.
The player wins the game when all mines are defused.

Environment
===========

We start off by defining the initial state, observation space, and action space of the environment.
The observation space has two different types of entities, mines and robots, both of which have an x and y coordinate.
The action space has a single categorical action with five possible choices, which will be used to move the robots.

.. code-block:: python

    from typing import List, Tuple, Dict
    from entity_gym.env import *

    class MineSweeper(Environment):
        def reset(self) -> Observation:
            positions = random.sample(
                [(x, y) for x in range(6) for y in range(6)],
                7,
            )
            self.mines = positions[:5]
            self.robots = positions[5:]
            return self.observe()

        @classmethod
        def obs_space(cls) -> ObsSpace:
            return ObsSpace({
                "Mine": Entity(features=["x", "y"]),
                "Robot": Entity(features=["x", "y"]),
            })
        
        @classmethod
        def action_space(cls) -> Dict[ActionName, ActionSpace]:
            return {
                "Move": CategoricalActionSpace(
                    ["Up", "Down", "Left", "Right", "Defuse Mines"],
                ),
            }
        
        def observe(self) -> Observation:
            raise NotImplementedError
        
        def act(self, actions: Action) -> Observation:
            raise NotImplementedError

Observation
===========

Next, we implement the ``observe`` method, which returns an `Observation </entity_gym/entity_gym.env.html#entity_gym.env.Observation>`_ representing the current state of the environment.

The ``entities`` dictionary contains the current state of the environment.
For the "Mine" entities, we need to specify only the features for each entity.
Because the "Robot" entities will be performing an action, we have to additionally supply a list of IDs for the "Robot" entities.
The IDs will later be used to determine which "Robot" entity performed which action.

On every step, we make the "Move" action available by specifying a ``CategoricalActionMask``.
The ``actor_types`` parameter specifies the types of entities that can perform the action.
In this case, we only allow "Robot" entities to perform the action (and not "Mine" entities).
As an alternative to ``actor_types``, ``CategoricalActionMask`` can also be supplied with an ``actor_ids`` list with the IDs of the entities that can perform the action.

The game is ``done`` once there are no more mines or robots, and we award a ``reward`` of 1.0 if all mines are defused.

.. code-block:: python

    def observe(self) -> Observation:
        return Observation(
            actions={
                "Move": CategoricalActionMask(
                    # Allow all robots to move
                    actor_types=["Robot"],
                ),
            },
            entities={
                "Robot": (
                    self.robots,
                    # Unique identifiers for all "Robot" entities
                    [("Robot", i) for i in range(len(self.robots))],
                ),
                # We don't need identifiers for mines since they are not 
                # directly referenced by any actions.
                "Mine": self.mines,
            },
            # The game is done once there are no more mines or robots
            done=len(self.mines) == 0 or len(self.robots) == 0,
            # Give reward of 1.0 for defusing all mines
            reward=1.0 if len(self.mines) == 0 else 0,
        )

Actions
=======

Finally, we implement the `act` method that takes an action and returns the next observation.

.. code-block:: python

    def act(self, actions: Mapping[ActionName, Action]) -> Observation:
        move = actions["Move"]
        assert isinstance(move, CategoricalAction)
        for (_, i), action in zip(move.actors, move.actions):
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

Action Masks
============

Currently, robots may move in any direction, but any movement that would take a robot outside the grid will be ignored.
We may want to restrict the robots choices so that they cannot move outside the grid.
We can do this by setting the `mask` attribute of the [`ActionMask`](todo link to docs) object to a boolean array of shape (number_entities, number_actions) that specifies which actions are allowed.

.. code-block:: python

    import random
    from entity_gym.env import *


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

SelectEntityAction
==================

Suppose we want to add a new *Orbital Cannon* entity to the game that can fire a laser at any mine or robot every 5 steps.
Since the number of mines and robots is unknown, we cannot use a normal categorical action for our Orbital Cannon.
Instead, we will use a `SelectEntityAction </entity_gym/entity_gym.env.html#entity_gym.env.SelectEntityAction>`_, which allows us to select one entity from a list of entities.


.. code-block:: python

    from entity_gym.env import *

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
            return Observation(
                entities={
                    "Mine": (
                        self.mines,
                        [("Mine", i) for i in range(len(self.mines))],
                    ),
                    "Robot": (
                        self.robots,
                        [("Robot", i) for i in range(len(self.robots))],
                    ),
                    "Orbital Cannon": (
                        [(self.orbital_cannon_cooldown,)],
                        [("Orbital Cannon", 0)],
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
        
        def act(self, actions: Mapping[ActionName, Action]) -> Observation:
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
