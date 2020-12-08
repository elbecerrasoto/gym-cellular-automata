import numpy as np
from gym import spaces

# ---------------- Data Classes
class Grid:
    """
    Doc
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
    """
    Doc
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

# ---------------- Service Classes
class Automaton:
    """
    It operates over a grid,
    performing an update by following the Automaton rules and neighborhood.
    """
    # Set these in ALL subclasses
    grid_space = None

    def update(self, grid, action=None, state=None):
        """Operation over a grid.
        
        Args:
            grid (object): a grid provided by the env
        Returns:
            grid (object): modified grid
        """
        raise NotImplementedError

class Modifier:
    """    
    It operates over a grid,
    represeting some sort of control over it.
    
    Its main method is update, which receives an action and a grid and returns
    a grid and a hidden state, both are used by the CAenv class to define an
    environment with the semantics for an Reinforcement Learning task.
    """
    # Set these in ALL subclasses
    grid_space = None
    action_space = None
    state_space = None

    def update(self, grid, action, state):
        """Operation over a grid. It represents the control task
        to perform.
        
        Args:
            action (object): an action provided by the agent
            grid (object): a grid provided by the env
        Returns:
            grid (object): modified grid
        """
        raise NotImplementedError

class CAEnv:
    """   
    It operates over a grid,
    represeting some sort of control over it.
    
    Its main method is update, which receives an action and a grid and returns
    a grid and a hidden state, both are used by the CAenv class to define an
    environment with the semantics for an Reinforcement Learning task.
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
