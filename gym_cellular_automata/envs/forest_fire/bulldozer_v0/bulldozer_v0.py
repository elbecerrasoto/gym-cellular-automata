from collections import Counter

import gym
import numpy as np
from gym import logger, spaces
from gym.utils import seeding

from gym_cellular_automata.envs.forest_fire.bulldozer_v0.utils.config import CONFIG
from gym_cellular_automata.envs.forest_fire.bulldozer_v0.utils.render import (
    env_visualization,
)
from gym_cellular_automata.envs.forest_fire.operators.ca_windy import WindyForestFire
from gym_cellular_automata.envs.forest_fire.operators.coordinate import Coordinate
from gym_cellular_automata.envs.forest_fire.operators.modify import Modify
from gym_cellular_automata.envs.forest_fire.operators.move import Move
from gym_cellular_automata.grid_space import Grid


class ForestFireEnvBulldozerV0(gym.Env):
    metadata = {"render.modes": ["human"]}

    # fmt: off
    _max_freeze      = CONFIG["max_freeze"]

    _n_moves         = len(CONFIG["actions"]["movement"])
    _n_shoots        = len(CONFIG["actions"]["shooting"])
    _action_sets     = CONFIG["actions"]["sets"]

    _row             = CONFIG["grid_shape"]["n_row"]
    _col             = CONFIG["grid_shape"]["n_col"]

    _empty           = CONFIG["cell_symbols"]["empty"]
    _burned          = CONFIG["cell_symbols"]["burned"]
    _tree            = CONFIG["cell_symbols"]["tree"]
    _fire            = CONFIG["cell_symbols"]["fire"]

    _p_tree          = CONFIG["p_tree"]
    _p_empty         = CONFIG["p_empty"]

    _wind            = CONFIG["wind"]
    _effects         = CONFIG["effects"]
    # fmt: on

    def __init__(self, rows=None, cols=None):

        self._row = self._row if rows is None else rows
        self._col = self._col if cols is None else cols

        self._set_spaces()

        self.cellular_automaton = WindyForestFire()

        self.move = Move(self._action_sets)

        self.modify = Modify(self._effects)

        self.coordinate = Coordinate(
            self.cellular_automaton, self.move, self.modify, self._max_freeze
        )

        # Gym spec method
        self.seed()

    def reset(self):

        self.done = False
        self.steps_beyond_done = 0

        self.grid = self._initial_grid_distribution()
        self.context = self._initial_context_distribution()

        return self.grid, self.context

    def step(self, action):

        # Process the action to reuse shared Operator Machinery
        action = self._action_processing(action)

        if not self.done:

            # Pre-Process the context to reuse shared Operator Machinery
            context = self._context_preprocessing(self.context)

            # MDP Transition
            new_grid, new_context = self.coordinate(self.grid, action, context)

            # Post-Process the context shown to the user
            new_context = self._context_postprocessing(new_context)

            # New State
            self.grid = new_grid
            self.context = new_context

            # Termination as a function of New State
            self._is_done()

            # Gym API Formatting
            # Necessary condition for MDP, its New State is public
            obs = new_grid, new_context
            # Reward as a function of New State, dependence NOT necessary for MDP
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

            # Returning figure for convenience, formally render mode=human returns None
            return env_visualization(self.grid, pos, self._fire_seed)

        else:

            logger.warn(
                f"Undefined mode.\nAvailable modes {self.metadata['render.modes']}"
            )

    def _award(self):

        # Negative Ratio of Fire and Trees
        # Reasons for using this Reward function:
        # 1. Easy to interpret
        # 2. Communicates the desire to terminate as fast as possible
        # 3. Internalizes the cost of Bulldozer actions
        counts = self._count_cells()
        return -(counts[self._fire] / (counts[self._fire] + counts[self._tree]))

    def _is_done(self):
        self.done = not bool(np.any(self.grid == self._fire))

    def _report(self):
        return {"hit": self.modify.hit}

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
            (self.ca_params_space, self.mod_params_space, self.coord_params_space)
        )

        # RL Spaces
        self.observation_space = spaces.Tuple((self.grid_space, self.context_space))
        self.action_space = spaces.MultiDiscrete([self._n_moves, self._n_shoots])

    def _action_processing(self, action):
        # Correct size and separates the sub-actions
        ca_action = None
        move_action = action[0]
        modify_action = action[1]
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
