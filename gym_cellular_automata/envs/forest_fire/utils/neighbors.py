from collections import namedtuple

import numpy as np


def moore_n(grid, position, n=1, invariant=0):
    row, col = position
    nrow, ncol = grid.shape

    # Target Positions on Original Grid
    tax0 = row + np.array([-n, +n])
    tax1 = col + np.array([-n, +n])
    targets = tup, tdown, tleft, tright = np.concatenate((tax0, tax1))

    try:

        # Down and Right targets already are IndexError.
        if tup < 0 or tleft < 0:
            raise IndexError

        return grid[tup : tdown + 1, tleft : tright + 1]

    except IndexError:

        borders = 0, (nrow - 1), 0, (ncol - 1)
        origins = row, row, col, col

        d = lambda a, b: abs(b - a)

        distances_bt = [d(b, t) for b, t in zip(borders, targets)]
        distances_ob = [d(o, b) for o, b in zip(origins, borders)]

        distances_rows = distances_bt[:2] + distances_ob[:2]
        distances_cols = distances_bt[2:] + distances_ob[2:]

        rows, cols = map(lambda l: sum(l) + 1, (distances_rows, distances_cols))

        invariant = np.array(invariant, dtype=grid.dtype)
        extended_grid = np.repeat(invariant, rows * cols).reshape(rows, cols)

        # fmt: off
        a0 =        distances_bt[0]
        e0 = rows - distances_bt[1]
        a1 =        distances_bt[2]
        e1 = cols - distances_bt[3]
        # fmt: on

        extended_grid[a0:e0, a1:e1] = grid[:, :]
        return extended_grid


# Depracated
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


# Depracated
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
