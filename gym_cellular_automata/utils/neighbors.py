import numpy as np
from collections import namedtuple

def are_neighbors_a_boundary(grid, pos):
    """
    Check if the neighbors of target position are a boundary.
    Return a tuple of Bools informing which neighbor is a boundary.
    It checks the up, down, left, and right neighbors.
    """
    row, col = pos 
    n_row, n_col = grid.data.shape
    
    up_offset, down_offset = row + np.array([-1, 1])
    left_offset, right_offset = col + np.array([-1, 1])

    up = up_offset >= 0
    down = down_offset <= n_row-1
    left = left_offset >= 0
    right = right_offset <= n_col-1
    
    LegalBounds = namedtuple('Bounds', ['up', 'down', 'left', 'right'])
    return LegalBounds(up, down, left, right)

def neighborhood_at(grid, pos, invariant=0):
    """
    Calculates the Moore's neighborgood of cell at target position `pos`.
    The boundary conditions are invariant and set to `empty`.
    Returns a tuple with the values of the nighborhood cells in the following
    order: up_left, up_center, up_right,
            middle_left, middle, middle_right,
            down_left, down_center, down_right
    """        
    row, col = pos

    legality = are_neighbors_a_boundary(grid, pos)   

    up_left = grid[row-1, col-1] if legality.up and legality.left else invariant
    up_center = grid[row-1, col] if legality.up else invariant       
    up_right = grid[row-1, col+1] if legality.up and legality.right else invariant

    middle_left = grid[row, col-1] if legality.left else invariant
    middle = grid[row, col]
    middle_right = grid[row, col+1] if legality.right else invariant
    
    down_left = grid[row+1, col-1] if legality.down and legality.left else invariant
    down_center = grid[row+1, col] if legality.down else invariant
    down_right = grid[row+1, col+1] if legality.down and legality.right else invariant

    return up_left, up_center, up_right, middle_left, middle, middle_right, down_left, down_center, down_right
