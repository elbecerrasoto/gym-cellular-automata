import os
import sys

from setuptools import setup

from version import VERSION

setup(
    name="gym_cellular_automata",
    packages=["gym_cellular_automata"],
    version=VERSION,
    description="Cellular Automata Environments for Reinforcement Learning following the OpenAI Gym API",
    url="https://github.com/elbecerrasoto/gym-cellular-automata",
    author="Emanuel Becerra Soto",
    author_email="elbecerrasoto@gmail.com",
    license="MIT",
    install_requires=[
        "gym",
        "numpy",
        "matplotlib",
        "scipy",
        "seaborn",
        "pyyaml",
        "svgpath2mpl",
    ],
    tests_require=["pytest", "pytest-cov", "pytest-repeat", "pytest-randomly"],
    python_requires=">=3.6",
)
