#!/usr/bin/env python

from setuptools import setup

setup(
    name="qspace_direction",
    version="0.0.1",
    description="A package for generating and optimizing sampling scheme in qspace",
    author="Si-Miao Zhang",
    author_email="zhangsimiao@buaa.edu,cn",
    url="",
    packages=["qspace_direction", "qspace_direction.sampling", "qspace_direction.lib"],
    package_dir={
        "qspace_direction": "src/qspace_direction",
        "qspace_direction.sampling": "src/qspace_direction/sampling",
        "qspace_direction.io_util": "src/qspace_direction/lib",
    },
    install_requires=[
        "numpy >= 1.19.5",
        "scipy >= 1.10.0",
        "gurobipy == 10.0.3",
        "docopt == 0.6.2",
    ],
)
