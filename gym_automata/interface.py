import numpy as np
from gym import spaces

gym_automata_doc = \
    """
    gym-automata API
    ----------------
    DATA objects
    1. Grid
    2. MoState
    
    OPERATOR objects
    1. Automaton
    2. Modifier
    operator.update(grid, action, state)
    returns: grid
    
    ORGANIZER objects
    1. CAEnv  
    """
    
# ---------------- Data Classes
class Grid:
    """DATA object gym_automata
    Grid(shape, cell_states, data, cell_type)
        
    It represents the grid (spatial disposition of cells) of a Cellular Automaton.
    Access the data as a numpy array.
    
    When building your own CA environments usually this Class is used directly,
    without extra customization.
    
    A Grid data object needs four pieces of information
    1. shape (tuple) multidimensional shape, usually 2-D
    2. cell_states (int) n number of cell states, they will be labeled from 0 to n-1
    3. data (array like) optional, random sampled if not provided
    4. cell_type (type) optional, default=np.int32
    
    e.g. grid of 8x8 with 2 cell states and random initialization
    grid = Grid(shape=(8,8), cell_states=2)
    
    e.g. grid of 8x8 with 2 cell states and custom initialization
    grid_ones = Grid(shape=(8,8), cell_states=2, data=np.ones((8,8)))
    
    Access the data as a numpy array
    grid[:]
    grid_ones[:]
    
    Check if both Grids lie on the same space
    grid.grid_space.contains(grid_ones[:])
    grid_ones.grid_space.contains(grid[:])
    """
    __doc__ += gym_automata_doc

    def __init__(self, shape=None, cell_states=None, data=None, cell_type=np.int32):
        assert not(shape is None or cell_states is None), 'shape and cell_states must be explicitly provided.'
        self.grid_space = spaces.Box(low=0, high=cell_states-1, shape=shape, dtype=cell_type)
        if data is None:
            self.data = self.grid_space.sample()
        elif self.grid_space.contains(data):
            self.data = data
        else:
            raise ValueError('Invalid grid data.')

    def __getitem__(self, index):
        return self.data[index]
    
    def __str__(self):
        return self.data.__str__()
    
    def __repr__(self):
        return self.data.__repr__()

class MoState:
    """DATA object gym_automata
    MoState(data, mostate_space)
    
    It represents the current state of the modifier.
    Used to store information need by
    the Modifier to update the grid and posibly by
    the environment to calculate reward and termination.
    
    When building your own CA environments usually this Class is used directly,
    without extra customization.
    
    A MoState data object needs two pieces of information
    1. data
    2. mostate_space (a Gym Space type)
    
    Declare a mospace
    e.g. A discrete space from 0 to 7
    mostate_space = spaces.Discrete(8)
    
    MoState data must be explicitly provided
    So let's create a random sample
    mostate_data = mostate_space.sample()
    
    Create a MoState data object
    mostate = MoState(data=mostate_data, mostate_space=mostate_space)
    
    Access the data by its __call__() method
    mostate()
    """
    __doc__ += gym_automata_doc

    def __init__(self, data=None, mostate_space=None):
        if data is None and mostate_space is None:
            self.data = None
            self.state_space = None
            print('MoState object initialized just for library consistency.')
            print('Its data and state_space attributes are set to None.')
        else:
            try:
                self.mostate_space = mostate_space
                if self.mostate_space.contains(data):
                    self.data = data
                else:
                    raise ValueError('Invalid data, it is not a member of the provided space.')
            except AttributeError:
                print('A Gym Space must be provided as a mostate_space')

    def __call__(self):
        return self.data
    
    def __str__(self):
        return self.data.__str__()
    
    def __repr__(self):
        return self.data.__repr__()

# ---------------- Operator Classes
class Automaton:
    """OPERATOR object gym_automata
    
    It operates over a grid,
    representing a 1-step computation of a Cellular Automaton.
    
    When building your own CA environments this Class must be customized, usually by
    inheritance, to build your own Cellular Automaton.

    Its main method update performs
    a 1-step update of the grid by following the Automaton rules and neighborhood.
    """
    __doc__ += gym_automata_doc

    # Set these in ALL subclasses
    grid_space = None

    def update(self, grid, action=None, mostate=None):
        """Operation over a grid.
        
        Args:
            grid (grid): a grid provided by the environment
            action (object): It is not used, set to None for API internal consistency
            mostate (mostate): It is not used, set to None for API internal consistency
        Returns:
            grid (grid): modified grid
        """
        raise NotImplementedError

class Modifier:
    """OPERATOR object gym_automata
    
    It operates over a grid,
    represeting some sort of control task over it.
    
    When building your own CA environments this Class must be customized, usually by
    inheritance, to build your own Modifier, that controls the dynamics of a CA
    by changing its grid according to the taken action and its current state.
    
    Its main method is update, which receives a grid, and action and a modifier state
    and returns a grid.
    """
    __doc__ += gym_automata_doc

    # Set these in ALL subclasses
    grid_space = None
    action_space = None
    state_space = None

    def update(self, grid, action, mostate=None):
        """Operation over a grid.
        
        Args:
            grid (grid): a grid provided by the environment
            action (object): an action provided by the agent
            mostate (mostate): modifier state, if any
        Returns:
            grid (grid): modified grid
        """
        raise NotImplementedError

# ---------------- Wrapper Classes
class CAEnv:
    """WRAPPER object gym_automata
    Wraps the data objects and the operator objects of gym-automata library into a
    coherent OpenAI Gym Environment.
    
    When building your own CA environments this Class is usually inherited by your
    final Environment Class, together with the Gym.Env Class.
    """
    __doc__ += gym_automata_doc

    # Set these in ALL subclasses
    # Data
    grid = None
    mostate = None
    
    # Services
    modifier = None
    automaton = None
    
    # Data Spaces
    grid_space = None
    mostate_space = None
    
    # RL Spaces
    observation_space = None # spaces.Tuple((grid_space, state_space))
    action_space = None
