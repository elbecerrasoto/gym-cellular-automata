from abc import ABC, abstractmethod

import gym
from gym import logger
from gym.utils import seeding


class CAEnv(ABC, gym.Env):
    @property
    @abstractmethod
    def MDP(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def initial_state(self):
        self._resample_initial = False

    def __init__(self, nrows, ncols, **kwargs):
        self.nrows, self.ncols = nrows, ncols  # nrows & ncols is API

        # Set default dict and create atts per key
        self._set_defaults(nrows, ncols, **kwargs)

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

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def status(self):
        return {
            "steps_elapsed": self.steps_elapsed,
            "reward_accumulated": self.reward_accumulated,
        }

    @abstractmethod
    def _award(self):
        raise NotImplementedError

    @abstractmethod
    def _is_done(self):
        raise NotImplementedError

    @abstractmethod
    def _report(self):
        raise NotImplementedError

    @abstractmethod
    def _get_defaults_free(self, **kwargs) -> dict:
        return {}

    @abstractmethod
    def _get_defaults_scale(self, nrows, ncols) -> dict:
        return {}

    def _set_defaults(self, nrows, ncols, **kwargs):

        # Scale free parameters default dictionary
        self._defaults_free = self._get_defaults_free(**kwargs)

        # Scale dependant parameters default dictionary
        self._defaults_scale = self._get_defaults_scale(nrows, ncols)

        # Parameters Default Dictionary
        self._defaults = {**self._defaults_free, **self._defaults_scale}

    def _get_kwarg(self, arg, kwargs):
        try:
            return kwargs[arg]
        except KeyError:
            return self._defaults[arg]

    def count_cells(self, grid=None):
        """Returns dict of cell counts"""
        from collections import Counter

        grid = self.grid if grid is None else grid
        return Counter(grid.flatten().tolist())
