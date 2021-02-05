import gym
from gym import spaces, logger
from gym.utils import seeding

from .forest_fire_cellular_automaton import ForestFireCellularAutomaton
from .forest_fire_modifier import ForestFireModifier
from .forest_fire_coordinator import ForestFireCoordinator

from gym_cellular_automata.utils.config import get_forest_fire_config_dict
CONFIG = get_forest_fire_config_dict()

def instantiatiate_coordinator():
    pass

class ForestFireEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def init(self, coordinator):
        self.coordinator = coordinator

    def reset(self):
        self.grid = None
        self.context = None
    
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
        
        else:
            logger.warn(
                            "You are calling 'step()' even though this "
                            "environment has already returned done = True. You "
                            "should always call 'reset()' once you receive 'done = "
                            "True' -- any further steps are undefined behavior."
                       )
            
            last_obs = self.new_grid[:], self.new_context
            info   = self._report()

            return last_obs, 0.0, True, info

    def _award(self):
        return 0.0
    
    def _is_done(self):
        return False
    
    def _report(self):
        return {}

    def render(self, mode='human'):
        ...
  
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
