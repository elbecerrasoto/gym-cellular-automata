import sys
import os
from setuptools import setup

# Don't import gym_cellular_automata module here, since deps may not be installed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "gym_cellular_automata"))
from version import VERSION

setup(
    name="gym_cellular_automata",
    packages=["gym_cellular_automata"],
    version=VERSION,
    description="Cellular Automata Environments for Reinforcement Learning",
    url="https://github.com/elbecerrasoto/gym-cellular-automata",
    author="Emanuel Becerra Soto",
    author_email="elbecerrasoto@gmail.com",
    license="MIT",
    install_requires=[
        "gymnasium",
        "numpy",
        "matplotlib",
        "scipy",
        "svgpath2mpl",
    ],
    tests_require=["pytest", "pytest-cov", "pytest-repeat", "pytest-randomly"],
    python_requires=">=3.9",
)
