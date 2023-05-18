from copy import copy

import numpy as np
import pytest

from gym_cellular_automata import GridSpace


@pytest.mark.parametrize(
    "space",
    [
        GridSpace(2, shape=(2, 2, 2)),
        GridSpace(5, shape=(5,)),
        GridSpace(values=[1, 2, 2], shape=(42,)),
        GridSpace(values=np.arange(10), shape=(1, 1, 1)),
        GridSpace(n=3, shape=(2, 2)),
    ],
)
def test_space_contains_its_samples(space):
    assert space.contains(space.sample())


@pytest.mark.parametrize(
    "space1, space2",
    [
        (GridSpace(2, shape=(2, 2, 2)), GridSpace(2, shape=(2, 2, 2))),
        (GridSpace(5, shape=(5,)), GridSpace(5, shape=(5,))),
        (
            GridSpace(values=[1, 2, 2, 2, 2], shape=(5,)),
            GridSpace(values=[1, 2, 1], shape=(5,)),
        ),
        (
            GridSpace(values=np.arange(10), shape=(1, 1, 1)),
            GridSpace(values=np.arange(9, -1, -1), shape=(1, 1, 1)),
        ),
        (GridSpace(n=3, shape=(2, 2)), GridSpace(values=np.arange(3), shape=(2, 2))),
    ],
)
def test_grid_equality(space1, space2):
    assert space1 == space2


@pytest.mark.parametrize(
    "space",
    [
        GridSpace(2, shape=(2, 2, 2)),
        GridSpace(5, shape=(5,)),
        GridSpace(values=[1, 2, 2], shape=(42,)),
        GridSpace(values=np.arange(10), shape=(1, 1, 1)),
        GridSpace(n=3, shape=(2, 2)),
    ],
)
def test_copy(space):
    space1 = space
    space2 = copy(space)
    assert space1 == space2, f"Expected {space1} to equal {space2}"


SEEDS = 4
@pytest.mark.repeat(SEEDS)
def test_seed():
    SEED = np.random.choice(2048)
    space1 = GridSpace(12, shape=(6, 6, 6), seed=SEED)
    space2 = GridSpace(12, shape=(6, 6, 6), seed=SEED)

    grid1 = space1.sample()
    grid2 = space2.sample()

    assert np.all(grid1 == grid2), f"Not equal with SEED {SEED}"
