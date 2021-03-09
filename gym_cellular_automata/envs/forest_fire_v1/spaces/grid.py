from operator import mul
from functools import reduce

import numpy as np
from gym.spaces import Space


CELL_TYPE = np.int16


class Grid(Space):
    r"""
    A Space for Cellular Automata Lattices.
    Arbitrary integers can be used as cell states.

    Example::

        >>> Grid(n=3, shape=(2, 2))
        >>> Grid(values=[-1, 0, 1], shape=(2,2))

    """
    def __init__(self, n=None, values=None, shape=tuple(), probs=None, dtype=CELL_TYPE):
        
        assert shape is not tuple(), "'shape' must be a non-empty tuple."
        assert n is not None or values is not None, "'n' or 'values' must be provided."
     
        if values is not None:
            
            self._from_values = True
            
            self.values = np.array(values, dtype=dtype)
            self.n = len(self.values) 

        else:
            
            assert n > 0, "'n' must be a positive integer."
            self._from_values = False

            self.values = np.arange(self.n, dtype=dtype)
        
        self.size = reduce(mul, self.shape)
        
        uniform = np.repeat(1.0, self.n) / self.n
        self.probs = uniform if probs is None else probs

        super(Grid, self).__init__(shape, dtype)

    def sample(self):        
        
        return np.random.choice(self.values, self.size, self.probs).reshape(self.shape)

    def contains(self, x):
        
        if isinstance(x, list):
            x = np.array(x, dtype=self.dtype)

        return {np.unique(x)}.issubset({self.values}) and self.shape == x.shape

    def __repr__(self):
        if self._from_values:

            return f"Grid(values={self.values}, shape={self.shape})"

        else:
            
            return f"Grid(n={self.n}, shape={self.shape})"
