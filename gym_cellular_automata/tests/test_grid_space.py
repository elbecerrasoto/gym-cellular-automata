from copy import copy

import pytest
import numpy as np

from gym_cellular_automata.grid_space import Grid


@pytest.mark.parametrize(
    "space",
    [
        Grid(2, shape=(2, 2, 2)),
        Grid(5, shape=(5,)),
        Grid(values=[1, 2, 2], shape=(42,)),
        Grid(values=np.arange(10), shape=(1, 1, 1)),
        Grid(n=3, shape=(2, 2)),
    ],
)
def test_space_contains_its_samples(space):
    assert space.contains(space.sample())


@pytest.mark.parametrize(
    "space1, space2",
    [
        (Grid(2, shape=(2, 2, 2)), Grid(2, shape=(2, 2, 2))),
        (Grid(5, shape=(5,)), Grid(5, shape=(5,))),
        (Grid(values=[1, 2, 2, 2, 2], shape=(5,)), Grid(values=[1, 2, 1], shape=(5,))),
        (
            Grid(values=np.arange(10), shape=(1, 1, 1)),
            Grid(values=np.arange(9, -1, -1), shape=(1, 1, 1)),
        ),
        (Grid(n=3, shape=(2, 2)), Grid(values=np.arange(3), shape=(2, 2))),
    ],
)
def test_grid_equality(space1, space2):
    assert space1 == space2


@pytest.mark.parametrize(
    "space",
    [
        Grid(2, shape=(2, 2, 2)),
        Grid(5, shape=(5,)),
        Grid(values=[1, 2, 2], shape=(42,)),
        Grid(values=np.arange(10), shape=(1, 1, 1)),
        Grid(n=3, shape=(2, 2)),
    ],
)
def test_copy(space):
    space1 = space
    space2 = copy(space)
    assert space1 == space2, "Expected {} to equal {}".format(space1, space)
