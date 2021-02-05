import numpy as np
from collections import Counter

import gym
from gym import spaces
from gym.utils import seeding

from gym_cellular_automata.envs.forest_fire import ForestFireCellularAutomaton, ForestFireModifier, ForestFireCoordinator
from gym_cellular_automata.utils.config import get_forest_fire_config_dict

CONFIG = get_forest_fire_config_dict()

CELL_STATES = CONFIG['cell_states']

ROW = CONFIG['grid_shape']['n_row']
COL = CONFIG['grid_shape']['n_row']

P_FIRE = CONFIG['ca_params']['p_fire']
P_TREE = CONFIG['ca_params']['p_tree']

EFFECTS = CONFIG['effects']

MAX_FREEZE = CONFIG['max_freeze']

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
    grid_space = spaces.Box(0, CELL_STATES - 1, shape = (ROW, COL), dtype = np.uint8)
        
    action_space = spaces.Box(1, 9, shape = tuple(), dtype = np.uint8)
    observation_space = spaces.Tuple((grid_space,
                                      ca_params_space,
                                      pos_space,
                                      freeze_space))

    def init(self):
        
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
        
        ca_params = P_FIRE, P_TREE
        pos = ROW // 2, COL // 2
        freeze = MAX_FREEZE

        self.context = ca_params, pos, freeze
        
        obs = self.grid, self.context
        reward = self._award()
        done = self._is_done()
        info = self._report()
        
        return obs, reward, done, info

    def step(self, action):
        done = self._is_done()
        
        if not done:

            new_grid, new_context = self.coordinator(self.grid, action, self.context)
            
            obs    = new_grid[:], new_context
            reward = self._award()
            info   = self._report()
            
            self.grid = new_grid
            self.context = new_context
            
            return obs, reward, done, info 

    def _award(self):
        cell_counts = Counter(self.grid.flatten().tolist())

        weights = np.array([self.reward_per_empty,
                            self.reward_per_tree,
                            self.reward_per_fire])
        
        counts = np.array([cell_counts[self.empty],
                           cell_counts[self.tree],
                           cell_counts[self.fire]])

        return np.dot(weights, counts)
    
    def _is_done(self):
        return False
    
    def _report(self):
        return {}
  
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode='human'):
        pass
