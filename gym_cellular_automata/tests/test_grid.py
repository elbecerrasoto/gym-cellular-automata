import pytest

import numpy as np
from gym import spaces

from gym_cellular_automata import Grid

SHAPE = (2, 2)
CELL_STATES = 4
DATA = [[0,1], [2,3]]

def test_Grid_API_specifications(grid = Grid(data = DATA,
                                             cell_states = CELL_STATES,
                                             shape = SHAPE)):
    # Strong typing of grid attributes
    assert isinstance(grid.data, np.ndarray)
    assert isinstance(grid.shape, tuple)
    assert isinstance(grid.cell_states, int)
    assert isinstance(grid.cell_type, type)
    assert isinstance(grid.grid_space, spaces.Space)
    assert hasattr(grid, '__getitem__'),'Must support indexing'
    assert hasattr(grid, '__setitem__'), 'Must support index assignation'
    with pytest.raises(Exception):
        Grid() # Cannot be initialized from defaults
    with pytest.raises(Exception):
        Grid(data = DATA, shape = SHAPE) # Cell States are necessary 
