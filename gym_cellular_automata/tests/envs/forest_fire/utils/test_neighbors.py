from icecream import ic
from copy import copy

import numpy as np
import pytest
from gym import spaces

from gym_cellular_automata.envs.forest_fire.utils.neighbors import (
    moore_n,
    neighborhood_at,
)
from gym_cellular_automata.grid_space import Grid

ROW = 8
COL = 8

CELL_STATES = 3

INVARIANT = 0


@pytest.fixture
def grid_space():
    return Grid(n=CELL_STATES, shape=(ROW, COL))


@pytest.fixture
def position_space():
    return spaces.MultiDiscrete([ROW, COL])


def test_moore_n(grid_space, position_space):
    grid1 = grid_space.sample()
    grid2 = copy(grid1)

    rmid = ROW // 2
    cmid = COL // 2

    N = 4

    g1 = moore_n(grid1, (rmid, cmid), n=N)
    g2 = moore_n(grid2, (rmid, cmid), n=N)

    ic(g1)
    ic(g2)
    ic(g1.shape)
    assert g1.shape == (2 * N + 1, 2 * N + 1)

    assert np.all(g1 == g2)


# @pytest.mark.skip(reason="Debugging")
def test_neighborhood_at(grid_space):
    empty = 0
    tree = 1
    fire = 2

    singleton_2d = Grid(values=[fire], shape=(1, 1)).sample()
    neighbors = neighborhood_at(grid=singleton_2d, pos=(0, 0), invariant=empty)

    neighbors2 = neighborhood_at2(singleton_2d, (0, 0), empty)
    assert neighbors == neighbors2

    assert neighbors.up_left == empty
    assert neighbors.up == empty
    assert neighbors.up_right == empty

    assert neighbors.left == empty
    assert neighbors.self == fire
    assert neighbors.right == empty

    assert neighbors.down_left == empty
    assert neighbors.down == empty
    assert neighbors.down_right == empty

    fire_x_grid = np.array([[fire, tree, fire], [tree, fire, tree], [fire, tree, fire]])
    neighbors = neighborhood_at(grid=fire_x_grid, pos=(2, 0), invariant=empty)

    neighbors2 = neighborhood_at2(fire_x_grid, (2, 0), empty)
    assert neighbors == neighbors2

    assert neighbors.up_left == empty
    assert neighbors.up == tree
    assert neighbors.up_right == fire

    assert neighbors.left == empty
    assert neighbors.self == fire
    assert neighbors.right == tree

    assert neighbors.down_left == empty
    assert neighbors.down == empty
    assert neighbors.down_right == empty


# Orthogonal Test
def neighborhood_at2(grid, pos, invariant=0):
    from collections import namedtuple

    """
    Calculates the Moore's neighborgood of cell at target position 'pos'.
    The boundary conditions are invariant and set to 'empty'.
    Returns a named tuple with the values of the nighborhood cells in the following
    order: up_left, up, up_right,
            left, middle, right,
            down_left, down, down_right
    """

    Neighbors = namedtuple(
        "Neighbors",
        [
            "up_left",
            "up",
            "up_right",
            "left",
            "self",
            "right",
            "down_left",
            "down",
            "down_right",
        ],
    )

    def neighbor_value(roffset, coffset):
        """Easier to Ask for Forgiveness than Permission."""
        trow, tcol = row + roffset, col + coffset

        try:

            if trow < 0 or tcol < 0:
                raise IndexError

            return grid[trow, tcol]

        except IndexError:

            return invariant

    row, col = pos

    # fmt: off
    up_left    = neighbor_value(-1, -1)
    up         = neighbor_value(-1, 0)
    up_right   = neighbor_value(-1, +1)

    left       = neighbor_value(0, -1)
    self       = neighbor_value(0, 0)
    right      = neighbor_value(0, +1)

    down_left  = neighbor_value(+1, -1)
    down       = neighbor_value(+1, 0)
    down_right = neighbor_value(+1, +1)
    # fmt: on

    return Neighbors(
        up_left, up, up_right, left, self, right, down_left, down, down_right
    )
