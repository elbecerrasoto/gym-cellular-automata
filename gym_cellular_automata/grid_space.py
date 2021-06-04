from functools import reduce
from operator import mul
from typing import Optional, Sequence

import numpy as np
from gym.spaces import Space


class GridSpace(Space):
    r"""
    A Space for Cellular Automata Lattices.
    Arbitrary integers can be used as cell states.

    Example::

        >>> GridSpace(n=3, shape=(2, 2))
        >>> GridSpace(values=[-1, 0, 1], shape=(2,2))

    """

    def __init__(
        self,
        n: Optional[int] = None,
        values: Optional[Sequence[int]] = None,
        shape: tuple = tuple(),
        probs: Optional[Sequence[float]] = None,
        dtype: np.intc = np.int32,
    ):

        super().__init__(shape, dtype)

        assert shape, "Shape must be a non-empty tuple."

        if values is not None:

            self._from_values = True

            self.values = np.unique(np.array(values, dtype=dtype))
            self.n = len(self.values)

        elif n is not None:

            self._from_values = False

            assert n is not None and n > 0, "'n' must be a positive integer."
            self.n = n

            self.values = np.arange(self.n, dtype=dtype)

        else:

            raise ValueError("'n' or 'values' must be provided.")

        uniform = np.repeat(1.0, self.n) / self.n
        self.probs = uniform if probs is None else probs

        assert len(self.values) == len(
            self.probs
        ), "Unique values do NOT MATCH with assigned probabilities."

        self.size = reduce(mul, self.shape)

    def sample(self) -> np.ndarray:

        return np.random.choice(a=self.values, size=self.size, p=self.probs).reshape(
            self.shape
        )

    def contains(self, x) -> bool:

        if isinstance(x, list):
            x = np.array(x, dtype=self.dtype)

        return set(np.unique(x)).issubset(set(self.values)) and self.shape == x.shape

    def __repr__(self):
        if self._from_values:

            return f"GridSpace(values={self.values}, shape={self.shape})"

        else:

            return f"GridSpace(n={self.n}, shape={self.shape})"

    def __eq__(self, other):
        return (
            isinstance(other, GridSpace)
            and (self.shape == other.shape)
            and np.all(self.values == other.values)
        )
