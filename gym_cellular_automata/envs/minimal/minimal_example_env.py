import numpy as np

import gym
from gym import spaces
from gym.utils import seeding

from gym_automata.interface.data import Grid, State
from .minimal_synchronizer import MinimalCAEnvSynchronizer

class MinimalCAEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
		
        self.CELL_STATES = CELL_STATES

        self.synchronizer = SYNCHRONIZER
        
        self.grid_space = SYNCHRONIZER.grid_space

        self.action_space = SYNCHRONIZER.action_space
        self.observation_space = spaces.Tuple(SYNCHRONIZER.grid_space, SYNCHRONIZER.state_space)
        self.initial_obs_space = self.observation_space

        self.current_grid = None
        self.current_sync_state = None

    def reset(self):
        obs = self.initial_obs_space.sample()
        grid_data = obs[0]
        sync_data = obs[1]
        self.current_grid = Grid(data=grid_data, cell_states=self.CELL_STATES)
        self.current_sync_state = State(data=sync_data, state_space=self.synchronizer.state_space)
        
        # For a MDP
        # If not change signature to something like f(obs, env_state)
        # Or directly extract it from the object state (self)
	 	reward = _award(obs) 
		done = _is_done(obs)
		
        info = _report()
		
        return obs, reward, done, info 
    
    def step(self, action):
        done = self._is_done()
        if not done:
            self.current_grid = self.synchronizer(self.current_grid, action, current_sync_state)
            self.current_sync_state = self._track_sync_state()

            obs = (current_grid.data, current_sync_state)
            reward = self._award(obs)
            info = self._report()

            return (self.current_grid.data, self.current.data), 0.0, done, info

        else:
            logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )
            
            return (self.current_grid.data, self.current_sync_state.data), 0.0, True, {}

    
    def render(self, mode='human'):
        print('Nothing to see here!')
    
    def close(self):
        print('Hasta la vista baby!')
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def _award(self, obs): # For a MDP
        return 1.0
    
    def _is_done(self): # For a MDP the calling could be explicit
        return np.random.choice((True, False))
    
    def _report(self):
        return {}

    def _track_sync_state(self):
        return self.synchronizer.state_space.sample()
