import numpy as np
from gym import spaces

from gym_automata.interface import Grid, State
from gym_automata.interface import Modifier, State, Grid

N = 7

SHAPE = (2, 2)
CELL_STATES = 2
GRID_DATA = np.eye(2, 2, dtype=np.uint16)

RANDOM_SHAPE = tuple([np.random.randint(1, N) for i in range(n)])
RANDOM_CELL_STATES = np.random.randint(2, N)

def generate_random_grid(shape=(2, 2), cell_states=2):
    return np.random.randint(cell_states, size=shape)

def generate_random_state(shape=(2, 2)):
    return np.random.normal(size=shape)

def helper_test_Grid_API(grid):
    assert isinstance(grid.data, np.ndarray)
    assert isinstance(grid.shape, tuple)
    assert isinstance(grid.cell_states, int)
    assert isinstance(grid.cell_type, type)
    assert isinstance(grid.grid_space, spaces.Space)
    assert hasattr(grid, '_getitem_'),'Must support indexing'
    assert hasattr(grid, '_setitem_'), 'Must support index assignation'

def helper_test_State_API(state):
    assert isinstance(state.data, np.ndarray) or isinstance(grid.data, tuple)
    assert isinstance(state.state_space, spaces.Space)
    assert hasattr(state, '_getitem_'),'Must support indexing'
    assert hasattr(state, '_setitem_'), 'Must support index assignation'

def test_Grid_functionality_with_constant_data():
    pass

def test_Grid_functionality_with_random_data():
    pass

def test_Grid_functionality_with_edge_cases():
    pass

# with pytest.raises(Exception):
#     Grid()
