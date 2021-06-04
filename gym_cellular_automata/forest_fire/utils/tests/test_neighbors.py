import numpy as np
import pytest
from gym import spaces

from gym_cellular_automata import GridSpace
from gym_cellular_automata.forest_fire.utils.neighbors import moore_n, neighborhood_at

ROW = 4
COL = 3
CELL_STATES = 3
INVARIANT = 0
MAX_N = 3
REPEATS = 24


@pytest.fixture
def grid_space():
    return GridSpace(values=[55, 66, 77, 88, 99], shape=(ROW, COL))


@pytest.fixture
def position_space():
    return spaces.MultiDiscrete([ROW, COL])


@pytest.fixture
def n_space():
    return spaces.Discrete(MAX_N + 1)


@pytest.mark.repeat(REPEATS)
def test_moore_n(grid_space, position_space, n_space):
    def enclose_grid(grid):
        r, c = grid.shape
        nrows = 2 * MAX_N + r
        ncols = 2 * MAX_N + c
        eg = np.zeros((nrows, ncols), dtype=grid.dtype)
        eg[MAX_N : MAX_N + r, MAX_N : MAX_N + c] = grid[:]
        return eg

    def n_moore_certain(n, position, grid):
        r, c = position
        r, c = r + MAX_N, c + MAX_N
        return grid[r - n : r + n + 1, c - n : c + n + 1]

    n = n_space.sample()

    position = position_space.sample()
    row, col = position

    grid = grid_space.sample()
    gridc = grid.copy()

    g = moore_n(n, position, grid)
    gc = moore_n(n, position, gridc)

    # Copy
    assert np.all(g == gc)

    l = lambda n: 2 * n + 1
    # Size
    assert g.shape == (l(n), l(n))

    # Center
    mid = l(n) // 2
    assert g[mid, mid] == grid[row, col]

    # Compare against a non-out-of-borders big grid
    enclosed = enclose_grid(grid)
    expected = n_moore_certain(n, position, enclosed)

    assert np.all(g == expected)


def test_neighborhood_at(grid_space):
    empty, tree, fire = range(3)

    singleton_2d = GridSpace(values=[fire], shape=(1, 1)).sample()
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
