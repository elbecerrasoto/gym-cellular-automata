import numpy as np
from gym import spaces

class Grid:
    """
    DATA class gym_automata
    Grid(data, shape, cell_states, cell_type)
    
    It represents the grid (spatial disposition of cells) of a Cellular Automaton.
    It labels its `n` cell states from `0` to `n-1`.
    
    It can be initialized by providing an array-like data or shape & cell_states or both.
    If only data is provided, shape and cell_states would be inferred from it.

    Parameters
    ----------
    data : numeric array
        n-dimensional array representing the state of a Cellular Automaton, usually 2-D.
        It assumes that the `n` cell states are label from `0` to `n-1`.

    shape : tuple
        Shape of each array dimension, from outer most to inner most dimensions.
    
    cell_states : int
        Number of different cell states.
        If only data is provided it would be inferred as the max number on the data.
    
    cell_type : type, default=np.uint16
        Type of the data entries.

    Attributes
    ----------
    data : numeric array
    
    shape : tuple
    
    cell_states : int
    
    cell_type : type
    
    grid_space : gym.spaces.Space
        A gym space that the data belongs to. For a grid is defined as
        `spaces.Box(low=0, high=self.cell_states-1, shape=self.shape, dtype=self.cell_type)`

    Examples
    --------
    >>> # Grid of 8x8 with 2 cell states and random initialization
    >>> grid = Grid(shape=(8,8), cell_states=2)
    
    >>> # Grid of 8x8 with 2 cell states and custom initialization
    >>> import numpy as np
    >>> grid_ones = Grid(data=np.ones((8,8)), shape=(8,8), cell_states=2))
    
    >>> # Shape and cell states can be inferred from the data
    >>> grid_ones = Grid(np.ones((8,8)))
    
    See Also
    --------
    
    Notes
    -----
    When building your own CA environments usually this class is used directly.
    """

    def __init__(self, data=None, cell_states=None, shape=None, cell_type=np.uint16):
        
        def infer_data_shape(data):
            return tuple(data.shape)
        
        def generate_grid_space(cell_states, shape, cell_type):
            return spaces.Box(low=0, high=cell_states-1, shape=shape, dtype=cell_type)
        
        self.cell_type = cell_type
        
        assert cell_states is not None, 'A number of possible cell_states must be provided'.
        assert cell_states > 0, 'cell_states must be a positive integer'.
        
        self.cell_states = int(cell_states)
        
        if data is None:
            assert shape is not None, 'If no data is provided, shape and cell_states must be provided to generate random data.'
            self.shape = tuple(shape)
            self.grid_space = generate_grid_space(self.cell_states, self.shape, self.cell_type)
            self.data = None # samples random data
        else:
            self.shape = infer_data_shape(data)
            self.grid_space = generate_grid_space(self.cell_states, self.shape, self.cell_type)
            self.data = data

    @property
    def data(self):
        return self._data
    
    @data.setter
    def data(self, data):
        # Forces data into np.ndarray
        # Allows to piggyback on methods and to use a low-level cell_type
        if data is None:
            self._data = np.array(self.grid_space.sample(), dtype=self.cell_type)
        else:
            data = np.array(data, dtype=self.cell_type)
            assert self.grid_space.contains(data), f'data does not belong to the space {self.grid_space}'
            self._data = data

    def __getitem__(self, index):
        return self.data[index]

    def __setitem__(self, index, value):
        self.data[index] = value
        assert self.grid_space.contains(self.data), f'data does not belong to the space {self.grid_space}'

    def __repr__(self):
        return f"Grid(\n{self.data},\nshape={self.shape}, cell_states={self.cell_states})"
