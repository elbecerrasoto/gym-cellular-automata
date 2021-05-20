import numpy as np
from gym import logger, spaces

from gym_cellular_automata import Operator
from gym_cellular_automata.ca_env import CAEnv
from gym_cellular_automata.envs.forest_fire.bulldozer_v0.utils.config import CONFIG
from gym_cellular_automata.envs.forest_fire.bulldozer_v0.utils.render import (
    env_visualization,
)
from gym_cellular_automata.envs.forest_fire.operators.ca_windy import WindyForestFire
from gym_cellular_automata.envs.forest_fire.operators.modify import Modify
from gym_cellular_automata.envs.forest_fire.operators.move import Move
from gym_cellular_automata.grid_space import Grid


class ForestFireEnvBulldozerV0(CAEnv):
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

        self.seed()

        self._row = self._row if rows is None else rows
        self._col = self._col if cols is None else cols

        self._set_spaces()

        self.cellular_automaton = WindyForestFire()

        self.move = Move(self._action_sets)

        self.modify = Modify(self._effects)

        self._MDP = MDP(
            self.cellular_automaton, self.move, self.modify, self._max_freeze
        )

    @property
    def MDP(self):
        return self._MDP

    @property
    def initial_state(self):

        if self._resample_initial:

            self.grid = self._initial_grid_distribution()
            self.context = self._initial_context_distribution()

            self._initial_state = self.grid, self.context

        self._resample_initial = False

        return self._initial_state

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
        counts = self.count_cells(self.grid)
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


class MDP(Operator):
    from collections import namedtuple

    Suboperators = namedtuple("Suboperators", ["cellular_automaton", "move", "modify"])

    grid_dependant = True
    action_dependant = True
    context_dependant = True

    def __init__(self, cellular_automaton, move, modify, max_freeze, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.suboperators = self.Suboperators(cellular_automaton, move, modify)

        self.max_freeze = max_freeze
        self.freeze_space = spaces.Discrete(max_freeze + 1)

    def update(self, grid, action, context):

        ca_params, position, freeze = context
        move_action, modify_action = action

        def move_then_modify(grid, move_action, modify_action, position):

            grid, position = self.suboperators.move(grid, move_action, position)
            grid, position = self.suboperators.modify(grid, modify_action, position)

            return grid, position

        if freeze == 0:

            grid, ca_params = self.suboperators.cellular_automaton(
                grid, None, ca_params
            )

            grid, position = move_then_modify(
                grid, move_action, modify_action, position
            )

            freeze = np.array(self.max_freeze)

        else:

            grid, position = move_then_modify(
                grid, move_action, modify_action, position
            )

            freeze = np.array(freeze - 1)

        context = ca_params, position, freeze

        return grid, context
