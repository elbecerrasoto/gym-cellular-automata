import numpy as np


# def moore_n(grid, position, n=1, invariant=0):

#     invariant = np.array(invariant, dtype=grid.dtype)
#     row, col = position

#     # Target Positions on Grid.
#     tup, tdown = row + np.array([-n, n + 1])
#     tleft, tright = col + np.array([-n, n + 1])

#     try:

#         # Wrong Down and Right targets already raise IndexError.
#         if tup < 0 or tleft < 0:
#             raise IndexError

#         # Enough Grid, just return the requested values.
#         return grid[tup:tdown, tleft:tright]

#     except IndexError:

#         # Recursive handling of the error.
#         # Add Invariant Rows and Cols when Grid would be out of bounds.
#         return rmoore_n(n, position, grid, invariant)


def moore_n(grid, position, n=1, invariant=0):
    return rmoore_n(n, position, grid, invariant)


def rmoore_n(n, POSITION, GRID, INVARIANT=0):

    # Grid lenght at step N.
    l = lambda n: 2 * n + 1

    # Grow matrix 1 step.
    def grow_matrix(M: np.ndarray, c: int = 0) -> np.ndarray:

        c = np.array(c, dtype=M.dtype)
        nrows, ncols = M.shape

        row = np.repeat(c, ncols).reshape(1, ncols)
        col = np.repeat(c, nrows + 2).reshape(nrows + 2, 1)

        M = np.concatenate((row, M, row), axis=0)
        M = np.concatenate((col, M, col), axis=1)

        return M

    row, col = POSITION
    nrows, ncols = GRID.shape
    INVARIANT = np.array(INVARIANT, dtype=GRID.dtype)

    # Offsets from POSITION
    oup, odo = row + np.array([-n, n + 1])
    ole, ori = col + np.array([-n, n + 1])

    # Base Case
    if n == 0:
        return GRID[oup:odo, ole:ori]

    # Defaults when out of borders.
    defaultC = np.repeat(INVARIANT, l(n - 1)).reshape(l(n - 1), 1)
    defaultR = np.repeat(INVARIANT, l(n - 1)).reshape(1, l(n - 1))

    grid = rmoore_n(n - 1, POSITION, GRID, INVARIANT)  # Recursive call.

    grid = grow_matrix(grid, INVARIANT)

    # fmt: off
    # Select Uppermost and Downmost Rows with correct size.
    up    = defaultR if oup < 0     else GRID[ oup:oup+1, ole+1:ori-1 ]
    down  = defaultR if odo > nrows else GRID[ odo-1:odo, ole+1:ori-1 ]

    # Select Last Leftmost and Rightmost Cols with correct size.
    left  = defaultC if ole < 0     else GRID[ oup+1:odo-1, ole:ole+1 ]
    right = defaultC if ori > ncols else GRID[ oup+1:odo-1, ori-1:ori ]

    grid[ 0   , 1:-1 ] = up   [0,:]
    grid[ -1  , 1:-1 ] = down [0,:]
    grid[ 1:-1, 0    ] = left [:,0]
    grid[ 1:-1, -1   ] = right[:,0]
    # fmt: on

    # Get the corners right
    if oup >= 0 and ole >= 0:
        grid[0, 0] = GRID[oup, ole]

    if oup >= 0 and ori <= ncols:
        grid[0, l(n) - 1] = GRID[oup, ori - 1]

    if odo <= nrows and ole >= 0:
        grid[l(n) - 1, 0] = GRID[odo - 1, ole]

    if odo <= nrows and ori <= ncols:
        grid[l(n) - 1, l(n) - 1] = GRID[odo - 1, ori - 1]

    return grid


# Depracated: Still used as interface for CAs.
# Superseded by Moore N function.
def neighborhood_at(grid, pos, invariant=0):
    """
    Calculates the Moore's neighborhood of cell at target position 'pos'.
    The boundary conditions are invariant and set to 'empty'.
    Returns a named tuple with the values of the nighborhood cells in the following
    order: up_left, up, up_right,
            left, self, right,
            down_left, down, down_right
    """
    from collections import namedtuple

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

    N = 1
    neighborhood = moore_n(grid, pos, N, invariant).flatten().tolist()
    return Neighbors(*neighborhood)
