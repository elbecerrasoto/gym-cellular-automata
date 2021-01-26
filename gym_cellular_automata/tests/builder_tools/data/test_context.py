import pytest

import numpy as np
from gym import spaces

from gym_automata.builder_tools.data import Context
 
SHAPE = (8, 8)
DATA = np.random.normal(size=SHAPE)
CONTEXT_SPACE = spaces.Box(low=float('-inf'), high=float('inf'), shape=SHAPE)

def test_Context_API_specifications(context = Context(data = DATA,
                                                context_space = CONTEXT_SPACE)):
    assert hasattr(context, 'data')
    assert isinstance(context.context_space, spaces.Space)
    # Default initialization must raise an UserWarning and set everything to None 
    with pytest.warns(UserWarning):
        context_obj_defaults = Context()
    assert context_obj_defaults.data is None
    assert context_obj_defaults.context_space is None
