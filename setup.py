#!/usr/bin/env python

from distutils.core import setup

setup(
    name="entity-gym",
    version="1.0",
    description="Entity Gym",
    author="Clemens Winter",
    author_email="clemenswinter1@gmail.com",
    packages=["entity_gym"],
    package_data={"entity_gym": ["py.typed"]},
    install_requires=["ragged-buffer==0.2.0"],
)
