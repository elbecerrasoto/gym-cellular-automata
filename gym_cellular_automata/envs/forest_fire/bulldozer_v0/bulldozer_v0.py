from collections import Counter

import gym
import numpy as np
from gym import logger, spaces
from gym.utils import seeding

from gym_cellular_automata.envs.forest_fire.v1.operators import (
    Bulldozer,
    Coordinator,
    WindyForestFireB,
)
from gym_cellular_automata.envs.forest_fire.v1.utils.config import CONFIG
from gym_cellular_automata.envs.forest_fire.v1.utils.render import env_visualization
from gym_cellular_automata.grid_space import Grid

# ------------ Forest Fire Environment


class ForestFireEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    # fmt: off
    _max_freeze      = CONFIG["max_freeze"]

    _n_moves         = len(CONFIG["actions"]["movement"])
    _n_shoots        = len(CONFIG["actions"]["shooting"])

    _reward_per_tree = CONFIG["rewards"]["per_tree"]

    _row             = CONFIG["grid_shape"]["n_row"]
    _col             = CONFIG["grid_shape"]["n_col"]

    _empty           = CONFIG["cell_symbols"]["empty"]
    _burned          = CONFIG["cell_symbols"]["burned"]
    _tree            = CONFIG["cell_symbols"]["tree"]
    _fire            = CONFIG["cell_symbols"]["fire"]

    _p_tree          = CONFIG["p_tree"]
    _p_empty         = CONFIG["p_empty"]

    _wind            = CONFIG["wind"]
    # fmt: on

    def __init__(self):

        self._set_spaces()

        self.cellular_automaton = WindyForestFireB(**self._ca_kwargs)

        self.modifier = Bulldozer(**self._mod_kwargs)

        self.coordinator = Coordinator(
            self.cellular_automaton,
            self.modifier,
            max_freeze=self._max_freeze,
            **self._coord_kwargs,
        )

        self.seed()

    def reset(self):

        self.done = False
        self.steps_beyond_done = 0

        self.grid = self._initial_grid_distribution()
        self.context = self._initial_context_distribution()

        return self.grid, self.context

    def step(self, action):

        if not self.done:

            # MDP Transition
            new_grid, new_context = self.coordinator(self.grid, action, self.context)

            # New State
            self.grid = new_grid
            self.context = new_context

            # Termination as a function of New State
            self._is_done()

            # API Formatting
            # Necessary condition for MDP, its New State is public
            obs = new_grid, new_context
            # Reward as a function of New State
            reward = self._award()
            info = self._report()

            return obs, reward, self.done, info

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
            return (self.grid, self.context), 0.0, True, {}

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode="human"):
        if mode == "human":

            wind, pos, freeze = self.context
            env_visualization(self.grid, pos, self._fire_seed)

        else:

            logger.warn(
                f"Undefined mode.\nAvailable modes {self.metadata['render.modes']}"
            )

    def _award(self):

        if not self.done:

            return 0.0

        else:

            return self._reward_per_tree * self._count_cells()[self._tree]

    def _is_done(self):
        self.done = not bool(np.any(self.grid == self._fire))

    def _report(self):
        return {}

    def _initial_grid_distribution(self):
        # fmt: off
        grid_space = Grid(
            values = [  self._empty,  self._burned,   self._tree,  self._fire],
            probs  = [self._p_empty,           0.0, self._p_tree,         0.0],
            shape=(self._row, self._col),
        )
        # fmt: on

        grid = grid_space.sample()

        row, col = self._fire_seed = self.mod_params_space.sample()

        grid[row, col] = self._fire

        return grid

    def _initial_context_distribution(self):
        position = self.mod_params_space.sample()
        freeze = np.array(self._max_freeze)

        return self._wind, position, freeze

    def _count_cells(self):
        """Returns dict of cell counts"""
        return Counter(self.grid.flatten().tolist())

    def _set_spaces(self):

        self.grid_space = Grid(
            values=[self._empty, self._burned, self._tree, self._fire],
            shape=(self._row, self._col),
        )

        self.ca_params_space = spaces.Box(0.0, 1.0, shape=(3, 3))
        self.mod_params_space = spaces.MultiDiscrete([self._row, self._col])
        self.coord_params_space = spaces.Discrete(self._max_freeze + 1)

        self.context_space = spaces.Tuple(
            (self.ca_params_space, self.mod_params_space, self.ca_params_space)
        )

        # RL Spaces
        self.observation_space = spaces.Tuple((self.grid_space, self.context_space))
        self.action_space = spaces.MultiDiscrete([self._n_moves, self._n_shoots])

        # Operator Spaces
        self._ca_kwargs = {
            "grid_space": self.grid_space,
            "action_space": self.action_space,
            "context_space": self.ca_params_space,
        }
        self._mod_kwargs = {
            "grid_space": self.grid_space,
            "action_space": self.action_space,
            "context_space": self.mod_params_space,
        }
        self._coord_kwargs = {
            "grid_space": self.grid_space,
            "action_space": self.action_space,
            "context_space": self.coord_params_space,
        }

        self.ca_space = spaces.Dict(spaces=self._ca_kwargs)
        self.mod_space = spaces.Dict(spaces=self._mod_kwargs)
        self.coord_space = spaces.Dict(spaces=self._coord_kwargs)
