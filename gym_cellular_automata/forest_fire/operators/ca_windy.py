from collections import namedtuple

import numpy as np
from gym import spaces
from scipy.signal import convolve2d

from gym_cellular_automata import Operator


class WindyForestFire(Operator):

    # Convolution Weights, magic variables
    _identity = 2 ** 11
    _propagation = 2 ** 3

    # Kernel Size
    _row_k = 3
    _col_k = 3

    def __init__(self, empty=0, burned=1, tree=3, fire=25, *args, **kwargs):

        super().__init__(*args, **kwargs)

        # Cell Values
        self._empty = empty
        self._burned = burned
        self._tree = tree
        self._fire = fire

        self._assert_correctness()

        if self.context_space is None:
            self.context_space = spaces.Box(0.0, 1.0, shape=(3, 3))

    def update(self, grid, action, wind):

        # Sample which FIREs fail to propagate this update
        fail_to_propagate = self._get_failed_propagations_mask(wind)

        kernel = self._get_kernel(fail_to_propagate)

        grid_signal = self._convolve(grid, kernel)

        new_grid = self._translate_analogic_to_discrete(grid_signal, self._get_breaks())

        return new_grid, wind

    def _get_failed_propagations_mask(self, wind):
        """
        Here goes the only sampling of the step.
        """
        uniform_space = spaces.Box(low=0.0, high=1.0, shape=(self._row_k, self._col_k))
        uniform_roll = uniform_space.sample()

        failed_propagations = np.repeat(False, self._row_k * self._col_k).reshape(
            self._row_k, self._col_k
        )
        failed_propagations = wind <= uniform_roll

        return failed_propagations

    def _get_kernel(self, failed_propagations):
        kernel = np.repeat(self._propagation, self._row_k * self._col_k).reshape(
            self._row_k, self._col_k
        )

        kernel[failed_propagations] = self._empty
        kernel[1, 1] = self._identity

        return kernel

    def _convolve(self, grid, kernel):
        return convolve2d(
            grid, kernel, mode="same", boundary="fill", fillvalue=self._empty
        )

    def _get_breaks(self):
        """
        4 breaks needed for 5 rules.
        """

        # "Unborn / Dead"
        dead_break = self._identity * self._burned

        # "Dead / Keep"
        keep_break = self._identity * self._tree

        # "Keep / Propagate"
        propagate_break = self._identity * self._tree + self._propagation * self._fire

        # "Propagate / Consume"
        consume_break = self._identity * self._fire

        Breaks = namedtuple("Breaks", ["dead", "keep", "propagate", "consume"])

        return Breaks(dead_break, keep_break, propagate_break, consume_break)

    def _translate_analogic_to_discrete(self, grid, breaks):
        row, col = grid.shape

        # Init on empty by default
        empty = np.array(self._empty)
        new_grid = np.repeat(empty, row * col).reshape(row, col)

        # 5 Rules to carry out:

        # 1. Unborn
        # EMPTY -> EMPTY
        # Implicitly defined by default values

        less_keep = grid < breaks.keep
        less_propagate = grid < breaks.propagate
        less_consume = grid < breaks.consume

        # 2. Dead
        # BURNED -> BURNED
        dead_mask = np.logical_and(grid >= breaks.dead, less_keep)
        new_grid[dead_mask] = self._burned

        # 3. Keep
        # TREE -> TREE
        keep_mask = np.logical_and(np.logical_not(less_keep), less_propagate)
        new_grid[keep_mask] = self._tree

        # 4. Propagate
        # TREE -> FIRE
        propagate_mask = np.logical_and(np.logical_not(less_propagate), less_consume)
        new_grid[propagate_mask] = self._fire

        # 4. Consume
        # FIRE -> BURNED
        propagate_mask = np.logical_not(less_consume)
        new_grid[propagate_mask] = self._burned

        return new_grid

    def _assert_correctness(self):

        assert self._row_k == 3, "Only Moore's neighborhood"
        assert self._col_k == 3, "Only Moore's neighborhood"

        # Neighbors without including the current cell
        n = 8

        # Weights
        i = self._identity
        p = self._propagation

        # Values
        E = self._empty
        B = self._burned
        T = self._tree
        F = self._fire

        # Ordering of cell values
        assert E < B
        assert B < T
        assert T < F

        # Ordering of Weights
        assert p < i

        # Test the boundaries of the 5 intervals
        # Interval Names:
        # Unborn, Dead, Keep, Propagate, Consume

        worst = n * p * F

        assert i * E + worst < i * B, "Unborn / Dead"
        assert i * B + worst < i * T, "Dead / Keep"

        # Key Assert, a TREE cell is subject to two different rules
        assert i * T + n * p * T < i * T + p * F, "Keep / Propagate"

        assert i * T + worst < i * F, "Propagate / Consume"
