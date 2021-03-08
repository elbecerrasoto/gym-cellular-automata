import numpy as np
from gym.spaces import Space


CELL_TYPE = np.int16


class Grid(Space):
    r"""A discrete space in :math:`\{ 0, 1, \\dots, n-1 \}`. 

    Example::

        >>> Discrete(2)

    """
    def __init__(self, n=None, values=None, shape=None, probs=None, dtype=np.int16):
        
        self.=
        super(Grid, self).__init__(shape, dtype)

    def sample(self):
        
        return self.np_random.randint(self.n)

    def contains(self, x):
        
        if isinstance(x, int):
            as_int = x
        elif isinstance(x, (np.generic, np.ndarray)) and (x.dtype.char in np.typecodes['AllInteger'] and x.shape == ()):
            as_int = int(x)
        else:
            return False
        return as_int >= 0 and as_int < self.n

    def __repr__(self):
        return "Grid(%d)" % self.n

    def __eq__(self, other):
        return isinstance(other, Discrete) and self.n == other.n



ca_params_space = spaces.Box(0.0, 1.0, shape=(2,))
pos_space = spaces.MultiDiscrete([ROW, COL])
freeze_space = spaces.Discrete(MAX_FREEZE + 1)

context_space = spaces.Tuple((ca_params_space, pos_space, freeze_space))
grid_space = spaces.Box(0, CELL_STATES - 1, shape=(ROW, COL), dtype=CELL_TYPE)

action_space = spaces.Box(ACTION_MIN, ACTION_MAX, shape=tuple(), dtype=ACTION_TYPE)
observation_space = spaces.Tuple((grid_space, context_space))

def random_grid():
    shape = (TEST_ROW, TEST_COL)
    size = reduce(mul, shape)
    probs = [P_EMPTY, P_TREE, P_FIRE]
    cell_values = np.array([ca.EMPTY, ca.TREE, ca.FIRE], dtype=ca.CELL_TYPE)

    return np.random.choice(cell_values, size, probs).reshape(shape)
