import numpy as np
from gym import spaces
from gym.utils import colorize
import warnings

gym_automata_doc = \
    """
    gym-automata API
    ----------------
    DATA classes
    1. Grid(data, shape, cell_states, cell_type)
    2. State(data, state_space)
    
    OPERATOR classes
    1. Automaton
    2. Modifier
    3. Organizer
    
    operator.update(grid : Grid, action, aust : State, most : State, orst : State)
    returns: Grid
    """

# ---------------- Data Classes

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
    __doc__ += gym_automata_doc

    def __init__(self, data=None, shape=None, cell_states=None, cell_type=np.uint16):
        def infer_data_shape(data):
            # Piggybacking on numpy methods
            data = np.array(data)
            return data.shape
        def infer_data_cell_states(data):   
            data = np.array(data)
            # Infer the cell states to be the ceil of the max
            return int(data.max()) + 1
        
        self.cell_type = cell_type
        if data is None:
            assert shape is not None and cell_states is not None, 'If no data is provided, shape and cell_states must be provided to generate random data.'
            self.shape = shape
            self.cell_states = cell_states
            self.grid_space = spaces.Box(low=0, high=self.cell_states-1, shape=self.shape, dtype=self.cell_type)
            self.data = None
        else:
            self.shape = infer_data_shape(data) if shape is None else tuple(shape)
            self.cell_states = infer_data_cell_states(data) if cell_states is None else int(cell_states)
            self.grid_space = spaces.Box(low=0, high=self.cell_states-1, shape=self.shape, dtype=self.cell_type)
            self.data = data

    @property
    def data(self):
        return self._data
    
    @data.setter
    def data(self, data):
        if data is None: # If no data, sample a grid
            self._data = self.grid_space.sample()
        else:
            assert self.grid_space.contains(data), f'data does not belong to the space {self.grid_space}'
            self._data = data

    def __getitem__(self, index):
        return self.data[index]

    def __setitem__(self, index, value):
        self.data[index] = value
        assert self.grid_space.contains(self.data), f'data does not belong to the space {self.grid_space}'

    def __repr__(self):
        return f"Grid(\n{self.data},\nshape={self.shape}, cell_states={self.cell_states})"

class State:
    """
    DATA class gym_automata
    State(data, state_space)
    
    It represents the current state of the Modifier or the Organizer.
    Used to store information need by them to update the grid
    and posibly by the environment to calculate reward and termination.
    
    It can be initialized by providing an array-like data or a tuple with array-like data and a gym.spaces.Space.
    If only data is provided, the space would be inferred from the data.
    Also it can be initialized by providing a gym.spaces.Space in which case a random sample from the space is taken.

    Parameters
    ----------
    data : numeric array or tuple of numeric arrays 
        n-dimensional array or tuple of arrays representing the state of the Modifier.

    state_space : gym.spaces.Space
        A Tuple space if a tuple is provided, a Box otherwise.
        If only data is provided it is inferred to be, per entry:
            `spaces.Box(-float('inf'), float('inf'), shape=data.shape)`

    Attributes
    ----------
    data : numeric array or tuple of numeric arrays

    state_space : gym.spaces.Space

    Examples
    --------
    
    See Also
    --------
    
    Notes
    -----
    When building your own CA environments usually this class is used directly.
    """
    __doc__ += gym_automata_doc

    def __init__(self, data=None, state_space=None):
        def infer_data_space(data):
            subspaces = []
            if isinstance(data, tuple): # Assume it is a composition of several spaces.
                 for subdata in data:
                     subdata = np.array(subdata)
                     subspace = spaces.Box(-float('inf'), float('inf'), shape=subdata.shape)
                     subspaces.append(subspace)
                 return spaces.Tuple(subspaces)
            else: # Assume a single space.
                data = np.array(data)
                return spaces.Box(-float('inf'), float('inf'), shape=data.shape)
        
        if data is None and state_space is None:
            self.state_space = None
            self.data = None
            msg = 'State object initialized just for gym-automata library consistency.\n\
Its data and state_space attributes are set to `None`.'
            warnings.warn(colorize(msg, 'yellow'), UserWarning)
        elif state_space is not None and data is None:
            self.state_space = state_space
            self.data = self.state_space.sample()
        else:
            self.state_space = infer_data_space(data) if state_space is None else state_space
            self.data = data

    @property
    def data(self):
        return self._data
    
    @data.setter
    def data(self, data):
        if data is None:
            self._data = None
        else:
            assert self.state_space.contains(data), f'data does not belong to the space {self.state_space}'
            self._data = data
            
    @property
    def state_space(self):
        return self._state_space

    @state_space.setter    
    def state_space(self, state_space):
        if state_space is None:
            self._state_space = None
        else:
            assert isinstance(state_space, spaces.Space), f'state_space must be an instance of gym.spaces.Space and currently is a {type(state_space)}'
            self._state_space = state_space

    def __getitem__(self, index):
        return self.data[index]

    def __setitem__(self, index, value):
        self.data[index] = value
        assert self.state_space.contains(self.data), f'data does not belong to the space {self.state_space}'

    def __repr__(self):
        if self.data is None and self.state_space is None:
            return "State(\nNone, state_space=None)" 
        else:
            return f"State(\n{self.data},\nstate_space={self.state_space})"    
  
# ---------------- Operator Classes

class Automaton:
    """
    OPERATOR class gym_automata
    
    It operates over a grid,
    representing a 1-step computation of a Cellular Automaton.
    
    Its main method update performs
    a 1-step update of a given grid by the following of the Automaton rules.

    Attributes
    ----------
    grid_space : gym.spaces.Space
        Grid Space.
    
    action_space : gym.spaces.Space
        Action Space.
        
    aust_space : gym.spaces.Space
        Automaton Space.
        
    most_space : gym.spaces.Space
        It is not used, set to `None` for API internal consistency.
    
    orst_space : gym.spaces.Space
        It is not used, set to `None` for API internal consistency.
                
    Methods
    ----------
    Operation over a grid. Main method of gym_automata.
    
    Args:
        grid : Grid
            A grid provided by the environment.
        action : object
            An action provided by the agent.
        aust : State
            An automaton state, if any, provided by the environment.
        most : State
            It is not used, set to `None` for API internal consistency.
        orst : State
            It is not used, set to `None` for API internal consistency.
        
    Returns:
        grid : Grid
            A 1-step updated grid by the CA rules.
    
    Examples
    --------
    
    See Also
    --------
    
    Notes
    -----
    Customize this class when building your own CA environments.
    """
    __doc__ += gym_automata_doc

    # Set these in ALL subclasses
    grid_space = None
    action_space = None
    aust = None

    # Not used, set to `None` for API internal consistency
    most_space = None
    orst_space = None

    def update(self, grid, action, aust=None, most=None, orst=None):
        """
        Operation over a grid. Main method of gym_automata.
        
        Args:
            grid : Grid
                A grid provided by the environment.
            action : object
                An action provided by the agent.
            aust : State
                An automaton state, if any, provided by the environment.
            most : State
                It is not used, set to `None` for API internal consistency.
            orst : State
                It is not used, set to `None` for API internal consistency.
            
        Returns:
            grid : Grid
                A 1-step updated grid by the CA rules.
        """
        raise NotImplementedError

class Modifier:
    """
    OPERATOR class gym_automata
    
    It operates over a grid,
    represeting some sort of control task over it.
    
    Its main method is update, which receives a grid, and action and a modifier state
    and returns a grid.

    Attributes
    ----------
    grid_space : gym.spaces.Space
        Grid Space.
    
    action_space : gym.spaces.Space
        Action Space.
        
    aust_space : gym.spaces.Space
        It is not used, set to `None` for API internal consistency.
        
    most_space : gym.spaces.Space
        Modifier State Space.
    
    orst_space : gym.spaces.Space
        It is not used, set to `None` for API internal consistency.
    
    Methods
    ----------
    Operation over a grid. Main method of gym_automata.
    
    Args:
        grid : Grid
            A grid provided by the environment.
        action : object
            An action provided by the agent.
        aust : State
            It is not used, set to `None` for API internal consistency.
        most : State
            A modifier state, if any, provided by the environment.
        orst : State
            It is not used, set to `None` for API internal consistency.
        
    Returns:
        grid : Grid
            Modified grid, with exchanged cells.
    
    Examples
    --------
    
    See Also
    --------
    
    Notes
    -----
    Customize this class when building your own CA environments.
    """
    __doc__ += gym_automata_doc

    # Set these in ALL subclasses
    grid_space = None
    action_space = None
    most_space = None
    
    # Not used, set to `None` for API internal consistency
    aust_space = None
    orst_space = None

    def update(self, grid, action, aust=None, most=None, orst=None):
        """
        Operation over a grid. Main method of gym_automata.
        
        Args:
            grid : Grid
                A grid provided by the environment.
            action : object
                An action provided by the agent.
            aust : State
                It is not used, set to `None` for API internal consistency.
            most : State
                A modifier state, if any, provided by the environment.
            orst : State
                It is not used, set to `None` for API internal consistency.
            
        Returns:
            grid : Grid
                Modified grid, with exchanged cells. 
        """
        raise NotImplementedError

class Organizer:
    """
    OPERATOR class gym_automata
    
    Provides the logic and the operation order of Automaton and Modifier updates.
    coherent OpenAI Gym Environment.

    Attributes
    ----------
    automaton : Automaton
        An operator that computes CA rules.
    
    modifier : Modifier
        An operator that exchange cells.
    
    grid_space : gym.spaces.Space
        Grid Space.
    
    action_space : gym.spaces.Space
        Action Space.
        
    aust_space : gym.spaces.Space
        Automaton State Space.
        
    most_space : gym.spaces.Space
        Modifier State Space.
    
    orst_space : gym.spaces.Space
        Organizer State Space.    
    
    Methods
    ----------
    Operation over a grid. Main method of gym_automata.
    
    Args:
        grid : Grid
            A grid provided by the environment.
        action : object
            An action provided by the agent.
        aust : State
            Automaton State, if any, provided by the environment.
        most : State
            Modifier State, if any, provided by the environment. 
        orst : State
            Organizer State, if any, provided by the environment.
        
    Returns:
        grid : Grid
            A new grid produced by a series of automaton and modifier updates.
    
    Examples
    --------
    
    See Also
    --------
    
    Notes
    -----
    Customize this class when building your own CA environments.
    """
    __doc__ += gym_automata_doc

    # Set these in ALL subclasses
    automaton = None
    modifier = None
    
    grid_space = None
    action_space = None
    aust_space = None
    most_space = None
    orst_space = None
    
    def update(self, grid, action, aust=None, most=None, orst=None):
        """
        Operation over a grid. Main method of gym_automata.
        
        Args:
            grid : Grid
                A grid provided by the environment.
            action : object
                An action provided by the agent.
            aust : State
                Automaton State, if any, provided by the environment.
            most : State
                Modifier State, if any, provided by the environment. 
            orst : State
                Organizer State, if any, provided by the environment.
            
        Returns:
            grid : Grid
                A new grid produced by a series of automaton and modifier updates.
        """
        raise NotImplementedError