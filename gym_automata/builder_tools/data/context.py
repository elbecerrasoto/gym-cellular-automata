import numpy as np
from gym import spaces
from gym.utils import colorize
import warnings

class Context:
    """
    DATA class gym_automata
    State(data, context_space)
    
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

    context_space : gym.spaces.Space
        A Tuple space if a tuple is provided, a Box otherwise.
        If only data is provided it is inferred to be, per entry:
            `spaces.Box(-float('inf'), float('inf'), shape=data.shape)`

    Attributes
    ----------
    data : numeric array or tuple of numeric arrays

    context_space : gym.spaces.Space

    Examples
    --------
    
    See Also
    --------
    
    Notes
    -----
    When building your own CA environments usually this class is used directly.
    """

    def __init__(self, data=None, context_space=None):
        
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
        
        if data is None and context_space is None:
            self.context_space = None # Raises a warning
            self.data = None
        
        elif context_space is not None and data is None:
            self.context_space = context_space
            self.data = self.context_space.sample()
        
        else:
            self.context_space = infer_data_space(data) if context_space is None else context_space
            self.data = data

    @property
    def data(self):
        return self._data
    
    @data.setter
    def data(self, data):
        if data is None:
            assert self.context_space is None, f'data cannot be None while existing a space {self.context_space}'
            self._data = None
        else:
            assert self.context_space.contains(data), f'data does not belong to the space {self.context_space}'
            self._data = data
            
    @property
    def context_space(self):
        return self._context_space

    @context_space.setter    
    def context_space(self, context_space):
        if context_space is None:
            msg = 'State object initialized just for gym-automata library consistency.\n\
Its data and context_space attributes are set to None.'
            warnings.warn(colorize(msg, 'yellow'), UserWarning)
            self._context_space = None
        else:
            assert isinstance(context_space, spaces.Space), f'context_space must be an instance of gym.spaces.Space and currently is a {type(context_space)}'
            self._context_space = context_space

    def __getitem__(self, index):
        return self.data[index]

    def __setitem__(self, index, value):
        self.data[index] = value
        assert self.context_space.contains(self.data), f'data does not belong to the space {self.context_space}'

    def __repr__(self):
        if self.data is None and self.context_space is None:
            return "State(\nNone, context_space=None)" 
        else:
            return f"State(\n{self.data},\ncontext_space={self.context_space})"
