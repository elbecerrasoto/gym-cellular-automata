import numpy as np
from matplotlib import pyplot as plt
from collections import Counter

import gym
from gym import spaces
from gym.utils import seeding

from .operators import ForestFireCellularAutomaton, ForestFireModifier, ForestFireCoordinator
from .utils.config import get_forest_fire_config_dict
from .utils.render_ca import plot_grid, add_helicopter_cross

CONFIG = get_forest_fire_config_dict()

CELL_STATES = CONFIG['cell_states']

ROW = CONFIG['grid_shape']['n_row']
COL = CONFIG['grid_shape']['n_row']

P_FIRE = CONFIG['ca_params']['p_fire']
P_TREE = CONFIG['ca_params']['p_tree']

EFFECTS = CONFIG['effects']

MAX_FREEZE = CONFIG['max_freeze']

# spaces.Box requires typing for discrete values
CELL_TYPE   = CONFIG['cell_type']
ACTION_TYPE = CONFIG['action_type']

# ------------ Forest Fire Environment

class ForestFireEnv(gym.Env):
    metadata = {'render.modes': ['human']}
   
    empty = CONFIG['cell_symbols']['empty']
    tree  = CONFIG['cell_symbols']['tree']
    fire  = CONFIG['cell_symbols']['fire']
    
    ca_params_space  = spaces.Box(0.0, 1.0, shape = (2,))
    pos_space        = spaces.MultiDiscrete([ROW, COL])
    freeze_space     = spaces.Discrete(MAX_FREEZE + 1)  
    
    context_space = spaces.Tuple((ca_params_space,
                                  pos_space,
                                  freeze_space))
    grid_space = spaces.Box(0, CELL_STATES - 1, shape = (ROW, COL), dtype = CELL_TYPE)
        
    action_space = spaces.Box(1, 9, shape = tuple(), dtype = ACTION_TYPE)
    observation_space = spaces.Tuple((grid_space,
                                      context_space))

    def __init__(self):
        
        self.cellular_automaton = ForestFireCellularAutomaton(grid_space    = self.grid_space,
                                                              action_space  = self.action_space,
                                                              context_space = self.ca_params_space)  

        self.modifier = ForestFireModifier(EFFECTS,
                                           grid_space    = self.grid_space,
                                           action_space  = self.action_space,
                                           context_space = self.pos_space)

        self.coordinator = ForestFireCoordinator(self.cellular_automaton,
                                                 self.modifier,
                                                 max_freeze = MAX_FREEZE)

        self.reward_per_empty = CONFIG['rewards']['per_empty']
        self.reward_per_tree  = CONFIG['rewards']['per_tree']
        self.reward_per_fire  = CONFIG['rewards']['per_fire']

    def reset(self):
        self.grid = self.grid_space.sample()
        
        ca_params = np.array([P_FIRE, P_TREE])
        pos = np.array([ROW // 2, COL // 2])
        freeze = np.array(MAX_FREEZE)

        self.context = ca_params, pos, freeze
        
        obs = self.grid, self.context
        
        return obs

    def step(self, action):
        done = self._is_done()
        
        if not done:

            new_grid, new_context = self.coordinator(self.grid, action, self.context)
            
            obs    = new_grid, new_context
            reward = self._award()
            info   = self._report()
            
            self.grid = new_grid
            self.context = new_context
            
            return obs, reward, done, info 

    def _award(self):
        dict_counts = Counter(self.grid.flatten().tolist())
        
        cell_counts = np.array([dict_counts[self.empty],
                                dict_counts[self.tree],
                                dict_counts[self.fire]])

        reward_weights = np.array([self.reward_per_empty,
                                    self.reward_per_tree,
                                    self.reward_per_fire])

        return np.dot(reward_weights, cell_counts)
    
    def _is_done(self):
        return False
    
    def _report(self):
        return {'hit': self.modifier.hit}
  
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode='human'):
        ca_params, pos, freeze = self.context
        
        figure = add_helicopter_cross( plot_grid( self.grid ), pos )
        plt.show()

        return figure

