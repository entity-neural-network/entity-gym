# Entity Gym

[![Actions Status](https://github.com/entity-neural-network/entity-gym/workflows/Checks/badge.svg)](https://github.com/entity-neural-network/entity-gym/actions)
[![PyPI](https://img.shields.io/pypi/v/entity-gym.svg?style=flat-square)](https://pypi.org/project/entity-gym/)
[![Documentation Status](https://readthedocs.org/projects/entity-gym/badge/?version=latest&style=flat-square)](https://entity-gym.readthedocs.io/en/latest/?badge=latest)
[![Discord](https://img.shields.io/discord/913497968701747270?style=flat-square)](https://discord.gg/SjVqhSW4Qf)


Entity Gym is an open source Python library that defines an _entity based_ API for reinforcement learning environments.
Entity Gym extends the standard paradigm of fixed-size observation spaces by allowing observations to contain dynamically-sized lists of entities.
This enables a seamless and highly efficient interface with simulators, games, and other complex environments whose state can be naturally expressed as a collection of entities.

The [enn-trainer library](https://github.com/entity-neural-network/enn-trainer) can be used to train agents for Entity Gym environments.

## Installation

```
pip install entity-gym
```

## Usage

You can find tutorials, guides, and an API reference on the [Entity Gym documentation website](https://entity-gym.readthedocs.io/en/latest/index.html).

## Examples

A number of simple example environments can be found in [entity_gym/examples](https://github.com/entity-neural-network/entity-gym/tree/main/entity_gym/examples). More complex examples can be found in the [ENN-Zoo](https://github.com/entity-neural-network/incubator/tree/main/enn_zoo/enn_zoo) project, which contains Entity Gym bindings for [Procgen](https://github.com/openai/procgen), [Griddly](https://github.com/Bam4d/Griddly), [MicroRTS](https://github.com/santiontanon/microrts), [VizDoom](https://github.com/mwydmuch/ViZDoom), and [CodeCraft](https://github.com/cswinter/DeepCodeCraft).
