import numpy as np
from gym import spaces

gym_automata_doc = \
    """
    gym-automata API
    ----------------
    DATA objects
    1. Grid
    2. State
    
    OPERATOR objects
    1. Automaton
    2. Modifier
    
    operator.update(grid, action, state)
    
    WRAPPER objects
    1. CAEnv  
    """

# ---------------- Data Classes
class Grid:
    """DATA object 
    
    It represents the grid (spatial disposition of cells) of a Cellular Automaton.
    
    A Grid data object needs four pieces of information
    1. data (optional, random sampled if not provided)
    2. shape (multidimensional shape, usually 2-D)
    3. cell_states (n number of cell states, they will be labeled from 0 to n-1)
    4. cell_type (optional, default=np.int32)
    
    e.g. grid of 8x8 with 2 cell states and random initialization
    grid = Grid(shape=(8,8), cell_states=2)
    
    e.g. grid of 8x8 with 2 cell states and custom initialization
    grid_ones = Grid(data=np.ones((8,8)), shape=(8,8), cell_states=2)
    
    Access the data as a numpy array
    grid[:]
    grid_ones[:]
    
    Check if both Grids lie on the same space
    grid.grid_space.contains(grid_ones[:])
    grid_ones.grid_space.contains(grid[:])
    """
    def __init__(self, data=None, shape=None, cell_states=None, cell_type=np.int32):
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

class State:
    """DATA object
    
    It represents the hidden state of the modifier.
    Used to store information need by
    the Modifier to update the grid and
    the CAEnv to calculate Reward and Termination
    
    A State data object needs two pieces of information
    1. data
    2. space (a Gym Space type)
    
    Declare a space
    e.g. A discrete space from 0 to 7
    state_space = spaces.Discrete(8)
    
    State data must be explicitly provided
    So let's create a random sample
    state_data = state_space.sample()
    
    Create a ModifierState data object
    state = State(data=state_data, state_space=state_space)
    
    Access the data by its __call__() method
    state()
    """
    def __init__(self, data=None, state_space=None):
        if data is None and state_space is None:
            self.data = None
            self.state_space = None
            print('State object initialized just for library consistency.')
            print('Its data and state_space attributes are set to None.')
        else:
            try:
                self.state_space = state_space
                if self.state_space.contains(data):
                    self.data = data
                else:
                    raise ValueError('Invalid data, it is not a member of the provided space.')
            except AttributeError:
                print('A Gym Space must be provided as a state_space')

    def __call__(self):
        return self.data
    
    def __str__(self):
        return self.data.__str__()
    
    def __repr__(self):
        return self.data.__repr__()

# ---------------- Operator Classes
class Automaton:
    """OPERATOR object
    
    It operates over a grid,
    representing a 1-step computation of a Cellular Automaton.
    
    Its main method update performs
    a 1-step update of the grid by following the Automaton rules and neighborhood.
    """
    # Set these in ALL subclasses
    grid_space = None

    def update(self, grid, action=None, state=None):
        """Operation over a grid.
        
        Args:
            grid (grid): a grid provided by the environment
            action (object): It is not used, set to None for API internal consistency
            state (state): It is not used, set to None for API internal consistency
        Returns:
            grid (grid): modified grid
        """
        raise NotImplementedError

class Modifier:
    """OPERATOR object    
    It operates over a grid,
    represeting some sort of control task over it.
    
    Its main method is update, which receives a grid, and action and a current mofifier state
    and returns a grid.
    """
    # Set these in ALL subclasses
    grid_space = None
    action_space = None
    state_space = None

    def update(self, grid, action, state=None):
        """Operation over a grid.
        
        Args:
            grid (grid): a grid provided by the environment
            action (object): an action provided by the agent
            state (state): modifier internal state, if any
        Returns:
            grid (grid): modified grid
        """
        raise NotImplementedError

# ---------------- Wrapper Classes
class CAEnv:
    """WRAPPER object
    Wraps the data objects and the operator objects of gym-automata library into a
    coherent OpenAI Gym Environment.
    """
    # Set these in ALL subclasses
    # Data
    grid = None
    state = None
    
    # Services
    modifier = None
    automaton = None
    
    # Data Spaces
    grid_space = None
    state_space = None
    
    # RL Spaces
    observation_space = None # spaces.Tuple((grid_space, state_space))
    action_space = None
