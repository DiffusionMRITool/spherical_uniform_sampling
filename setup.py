#!/usr/bin/env python

from setuptools import setup

setup(
    name="spherical_uniform_sampling",
    version="0.0.1",
    description="A package for generating and optimizing sampling scheme in qspace",
    author="Si-Miao Zhang",
    author_email="zhangsimiao@buaa.edu,cn",
    url="",
    packages=["spherical_uniform_sampling", "spherical_uniform_sampling.sampling", "spherical_uniform_sampling.lib"],
    package_dir={
        "spherical_uniform_sampling": "src/spherical_uniform_sampling",
        "spherical_uniform_sampling.sampling": "src/spherical_uniform_sampling/sampling",
        "spherical_uniform_sampling.io_util": "src/spherical_uniform_sampling/lib",
    },
    scripts=[
        "src/spherical_uniform_sampling/scripts/combine_bvec_bval.py",
        "src/spherical_uniform_sampling/scripts/direction_flip.py",
        "src/spherical_uniform_sampling/scripts/direction_generation.py",
        "src/spherical_uniform_sampling/scripts/direction_subsampling.py",
        "src/spherical_uniform_sampling/scripts/direction_continous_optimization.py",
        "src/spherical_uniform_sampling/scripts/direction_geem.py",
        "src/spherical_uniform_sampling/scripts/direction_order.py",
        "src/spherical_uniform_sampling/scripts/direction_statistics.py",
    ],
    install_requires=[
        "numpy >= 1.19.5",
        "scipy >= 1.10.0",
        "gurobipy == 10.0.3",
        "docopt == 0.6.2",
    ],
)
