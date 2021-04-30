from collections import Counter

import gym
import matplotlib.pyplot as plt
import numpy as np
from gym import logger, spaces
from gym.utils import seeding

from gym_cellular_automata.envs.forest_fire.operators.ca_DrosselSchwabl import (
    ForestFire,
)
from gym_cellular_automata.envs.forest_fire.operators.coordinate import Coordinate
from gym_cellular_automata.envs.forest_fire.operators.modify import Modify
from gym_cellular_automata.envs.forest_fire.operators.move import Move
from gym_cellular_automata.grid_space import Grid

from .utils.config import CONFIG
from .utils.render import add_helicopter, plot_grid


class ForestFireEnvHelicopterV0(gym.Env):
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

    def __init__(self):

        self._set_spaces()

        self.cellular_automaton = ForestFire(self._empty, self._tree, self._fire)

        self.move = Move(CONFIG["actions_sets"])

        self.modify = Modify(self._effects)

        self.coordinate = Coordinate(
            self.cellular_automaton, self.move, self.modify, self._max_freeze
        )

        # Gym spec method
        self.seed()

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

        # Process the action to reuse shared Operator Machinery
        action = self._action_processing(action)

        if not done:

            # Pre-Process the context to reuse shared Operator Machinery
            context = self._context_preprocessing(self.context)

            # MDP Transition
            new_grid, new_context = self.coordinate(self.grid, action, context)

            # Post-Process the context shown to the user
            new_context = self._context_postprocessing(new_context)

            # Gym API Formatting
            # Necessary condition for MDP, its New State is public
            obs = new_grid, new_context
            # Reward as a function of New State, dependence NOT necessary for MDP
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
        return {"hit": self.modify.hit}

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode="human"):

        if mode == "human":

            ca_params, pos, freeze = self.context

            figure = add_helicopter(plot_grid(self.grid), pos)
            plt.show()

            # Returning figure for convenience, formally render mode=human returns None
            return figure

        else:

            logger.warn(
                f"Undefined mode.\nAvailable modes {self.metadata['render.modes']}"
            )

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

    def _action_processing(self, action):
        # Correct size and Modify is always active
        ca_action = None
        move_action = action
        modify_action = True
        coordinate_action = None

        return ca_action, move_action, modify_action, coordinate_action

    def _context_preprocessing(self, context):
        ca_context, move_context, coordinate_context = context
        # Only repeats the Move Context
        return ca_context, move_context, move_context, coordinate_context

    def _context_postprocessing(self, new_context):
        # Only un-repeats the Move Context
        ca_context, move_context, modify_context, coordinate_context = new_context
        return ca_context, move_context, coordinate_context
