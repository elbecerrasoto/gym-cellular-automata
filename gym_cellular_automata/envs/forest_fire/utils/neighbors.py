from collections import namedtuple

import numpy as np


def are_my_neighbors_a_boundary(grid, pos):
    """
    Check if the neighbors of target position are a boundary.
    Return a tuple of Bools informing which neighbor is a boundary.
    It checks the up, down, left, and right neighbors.
    """
    row, col = pos
    n_row, n_col = grid.shape

    up_offset, down_offset = row + np.array([-1, 1])
    left_offset, right_offset = col + np.array([-1, 1])

    up = bool(up_offset < 0)
    down = bool(down_offset > n_row - 1)
    left = bool(left_offset < 0)
    right = bool(right_offset > n_col - 1)

    Boundaries = namedtuple("Boundaries", ["up", "down", "left", "right"])

    return Boundaries(up, down, left, right)


def neighborhood_at(grid, pos, invariant=0):
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


def moore_n(grid, pos, n=1, invariant=0):
    row, col = pos

    left_up = row - n, col - n
    right_up = row - n, col + n

    left_down = row + n, col - n
    right_down = row + n, col + n

    for corner in [left_up, right_up, left_down, right_down]:
        i_am_boundary = are_my_neighbors_a_boundary(grid, corner)

        if any(i_am_boundary):
            # Early return if any boundary
            return np.array(
                neighborhood_at(grid, pos, invariant), dtype=grid.dtype
            ).reshape(3, 3)

    return grid[left_up[0] : left_down[0] + 1, left_up[1] : right_up[1] + 1]
