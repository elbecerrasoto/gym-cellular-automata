from copy import copy

import numpy as np
import pytest
from gym import spaces

from gym_cellular_automata.envs.forest_fire.utils.neighbors import moore_n
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

    g1 = moore_n(grid1, (rmid, cmid), n=24)
    g2 = moore_n(grid2, (rmid, cmid), n=24)

    print(g1)
    print(g2)

    assert np.all(g1 == g2)
