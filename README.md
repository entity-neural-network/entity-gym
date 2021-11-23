# entity-gym

Prototype for an abstraction over reinforcement learning environments that represent observations as lists of structured objects.

The interface is defined in [entity_gym/environment.py](https://github.com/cswinter/entity-gym/blob/main/entity_gym/environment.py).

A number of simple example environments can be found in [entity_gym/envs](https://github.com/cswinter/entity-gym/tree/main/entity_gym/envs).

After installing entity-gym with `pip install -e .`, the example environments can be run with `python entity_gym/main.py --env=MoveToOrigin|CherryPick|PickMatchingBalls|Minefield|MultiSnake`.
