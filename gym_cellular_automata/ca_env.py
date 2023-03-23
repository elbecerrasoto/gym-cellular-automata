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

    def __init__(self, nrows, ncols, debug=False, **kwargs):
        self.nrows, self.ncols = nrows, ncols  # nrows & ncols is API

        # Gym spec method
        self.seed()

        self._debug = debug
        if self._debug:
            print("Perhaps you forgot to do env.reset()")

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

    def count_cells(self, grid=None):
        """Returns dict of cell counts"""
        from collections import Counter

        grid = self.grid if grid is None else grid
        return Counter(grid.flatten().tolist())
