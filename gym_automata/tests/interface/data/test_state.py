import pytest

import numpy as np
from gym import spaces

from gym_automata.interface.data.state import State
 
SHAPE = (8, 8)
DATA = np.random.normal(size=SHAPE)
STATE_SPACE = spaces.Box(low=float('-inf'), high=float('inf'), shape=SHAPE)

def test_State_API_specifications(state = State(data = DATA,
                                                state_space = STATE_SPACE)):
    assert hasattr(state, 'data')
    assert isinstance(state.state_space, spaces.Space)
    # Default initialization must raise an UserWarning and set everything to None 
    with pytest.warns(UserWarning):
        state_obj_defaults = State()
    assert state_obj_defaults.data is None
    assert state_obj_defaults.state_space is None
