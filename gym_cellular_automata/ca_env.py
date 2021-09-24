from abc import ABC, abstractmethod

import gym
from gym import logger, spaces
from gym.utils import seeding

from gym_cellular_automata.grid_space import GridSpace
from gym_cellular_automata.operator import Identity


class CAEnv(ABC, gym.Env):
    @property
    @abstractmethod
    def MDP(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def initial_state(self):
        self._resample_initial = False

    def __init__(self, nrows, ncols, *args, **kwargs):
        # Get default parameters

        # Scale free parameters default dictionary
        self._defaults_free = self._get_defaults_free(*args, **kwargs)

        # Scale dependant parameters default dictionary
        self._defaults_scale = self._get_defaults_scale(nrows, ncols)

        # Parameters Default Dictionary
        self._defaults = {**self._defaults_free, **self._defaults_scale}

        # Gym spec method
        self.seed()

    def step(self, action):

        if not self.done:

            # MDP Transition
            self.state = self.grid, self.context = self.MDP(
                self.grid, action, self.context
            )

            # Check for termination
            self._is_done()

            # Gym API Formatting
            obs = self.state
            reward = self._award()
            done = self.done
            info = self._report()

            # Status method
            self.steps_elapsed += 1
            self.reward_accumulated += reward

            return obs, reward, done, info

        else:

            if self.steps_beyond_done == 0:

                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )

            self.steps_beyond_done += 1

            # Graceful after termination
            return self.state, 0.0, True, self._report()

    def reset(self):

        self.done = False
        self.steps_elapsed = 0
        self.reward_accumulated = 0.0
        self.steps_beyond_done = 0
        self._resample_initial = True
        obs = self.state = self.grid, self.context = self.initial_state

        return obs

    def status(self):
        return {
            "steps_elapsed": self.steps_elapsed,
            "reward_accumulated": self.reward_accumulated,
        }

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    @abstractmethod
    def _get_defaults_free(self, *args, **kwargs) -> dict:
        raise NotImplementedError

    @abstractmethod
    def _get_defaults_scale(self, nrows, ncols) -> dict:
        raise NotImplementedError

    @abstractmethod
    def _award(self):
        raise NotImplementedError

    @abstractmethod
    def _is_done(self):
        raise NotImplementedError

    @abstractmethod
    def _report(self):
        raise NotImplementedError

    def count_cells(self, grid=None):
        """Returns dict of cell counts"""
        from collections import Counter

        grid = self.grid if grid is None else grid
        return Counter(grid.flatten().tolist())


class MockCAEnv(CAEnv):
    _nrows = 8
    _ncols = 8
    _states = 8

    def _get_defaults_free(self, *args, **kwargs):
        """
        place holder
        """
        return {}

    def _get_defaults_scale(self, nrows, ncols):
        """
        place holder
        """
        return {}

    def __init__(self, nrows=_nrows, ncols=_ncols, *args, **kwargs):

        super().__init__(nrows, ncols, *args, **kwargs)

        self._set_spaces()

        # Composite Operators
        self._MDP = Identity(**self.MDP_space)

    @property
    def MDP(self):
        return self._MDP

    @property
    def initial_state(self):

        if self._resample_initial:

            self.grid = self.grid_space.sample()
            self.context = self.context_space.sample()

            self._initial_state = self.grid, self.context
            self._resample_initial = False

        return self._initial_state

    def _award(self):
        return 0.0

    def _is_done(self):
        return False

    def _report(self):
        return {}

    def _set_spaces(self):
        self.grid_space = GridSpace(n=self._states, shape=(self._nrows, self._ncols))
        self.action_space = spaces.Discrete(self._states)
        self.context_space = spaces.Discrete(self._states)

        self.observation_space = spaces.Tuple((self.grid_space, self.context_space))

        self.MDP_space = {
            "grid_space": self.grid_space,
            "action_space": self.action_space,
            "context_space": self.context_space,
        }
