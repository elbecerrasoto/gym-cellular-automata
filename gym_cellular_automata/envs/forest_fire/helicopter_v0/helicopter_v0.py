from collections import Counter

import gym
import numpy as np
from gym import spaces
from gym.utils import seeding
from matplotlib import pyplot as plt

from gym_cellular_automata.envs.forest_fire.operators.cella_drossel_schwabl import (
    ForestFireCellularAutomaton,
)
from gym_cellular_automata.envs.forest_fire.operators.coord_freezer import (
    Freezer as ForestFireCoordinator,
)
from gym_cellular_automata.grid_space import Grid

from .operators import ForestFireModifier
from .utils.config import CONFIG
from .utils.render import add_helicopter, plot_grid

# ------------ Forest Fire Environment


class ForestFireEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    # fmt: off
    _empty           = CONFIG["cell_symbols"]["empty"]
    _tree            = CONFIG["cell_symbols"]["tree"]
    _fire            = CONFIG["cell_symbols"]["fire"]


    _row              = CONFIG["grid_shape"]["n_row"]
    _col              = CONFIG["grid_shape"]["n_col"]

    _p_fire           = CONFIG["ca_params"]["p_fire"]
    _p_tree           = CONFIG["ca_params"]["p_tree"]

    _effects          = CONFIG["effects"]

    _max_freeze       = CONFIG["max_freeze"]

    _n_actions    = len(CONFIG["actions"])

    _reward_per_empty = CONFIG["rewards"]["per_empty"]
    _reward_per_tree  = CONFIG["rewards"]["per_tree"]
    _reward_per_fire  = CONFIG["rewards"]["per_fire"]
    # fmt: on

    def _set_spaces(self):
        self.ca_params_space = spaces.Box(0.0, 1.0, shape=(2,))
        self.pos_space = spaces.MultiDiscrete([self._row, self._col])
        self.freeze_space = spaces.Discrete(self._max_freeze + 1)

        self.context_space = spaces.Tuple(
            (self.ca_params_space, self.pos_space, self.freeze_space)
        )

        self.grid_space = Grid(
            values=[self._empty, self._tree, self._fire],
            shape=(self._row, self._col),
        )

        self.action_space = spaces.Discrete(self._n_actions)
        self.observation_space = spaces.Tuple((self.grid_space, self.context_space))

    def __init__(self):

        self._set_spaces()

        self.cellular_automaton = ForestFireCellularAutomaton(
            self._empty, self._tree, self._fire
        )

        self.modifier = ForestFireModifier(
            self._effects,
            grid_space=self.grid_space,
            action_space=self.action_space,
            context_space=self.pos_space,
        )

        self.coordinator = ForestFireCoordinator(
            self.cellular_automaton, self.modifier, max_freeze=self._max_freeze
        )

    def reset(self):
        self.grid = self.grid_space.sample()

        ca_params = np.array([self._p_fire, self._p_tree])
        pos = np.array([self._row // 2, self._col // 2])
        freeze = np.array(self._max_freeze)

        self.context = ca_params, pos, freeze

        obs = self.grid, self.context

        return obs

    def step(self, action):
        done = self._is_done()

        if not done:

            new_grid, new_context = self.coordinator(self.grid, action, self.context)

            obs = new_grid, new_context
            reward = self._award()
            info = self._report()

            self.grid = new_grid
            self.context = new_context

            return obs, reward, done, info

    def _award(self):
        dict_counts = Counter(self.grid.flatten().tolist())

        cell_counts = np.array(
            [dict_counts[self._empty], dict_counts[self._tree], dict_counts[self._fire]]
        )

        reward_weights = np.array(
            [self._reward_per_empty, self._reward_per_tree, self._reward_per_fire]
        )

        return np.dot(reward_weights, cell_counts)

    def _is_done(self):
        return False

    def _report(self):
        return {"hit": self.modifier.hit}

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode="human"):
        ca_params, pos, freeze = self.context

        figure = add_helicopter(plot_grid(self.grid), pos)
        plt.show()

        return figure
