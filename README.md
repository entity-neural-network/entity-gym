# entity-gym

Entity Gym is a highly flexible abstraction for reinforcement learning environments that allows observations to be represented by lists of structured objects.
The expressivity of Entity Gym's API makes it very easy to integrate complex environements that expose efficient state-based observation and action spaces.

The interface is defined in [entity_gym/environment.py](https://github.com/entity-neural-network/incubator/blob/main/entity_gym/entity_gym/environment/environment.py).

A number of simple example environments can be found in [entity_gym/examples](https://github.com/entity-neural-network/incubator/tree/main/entity_gym/entity_gym/examples).

After installing entity-gym with `pip install -e .`, the example environments can be run with `python entity_gym/main.py --env=MoveToOrigin|CherryPick|PickMatchingBalls|Minefield|MultiSnake`.

The [Quick Start Guide](TUTORIAL.md) walks you through the process of creating a new environment.
