import numpy as np
from gym import logger, spaces

from gym_cellular_automata import CAEnv, GridSpace, Operator
from gym_cellular_automata.forest_fire.operators import (
    ForestFire,
    Modify,
    Move,
    MoveModify,
)

from .utils.config import CONFIG
from .utils.render import add_helicopter, plot_grid


class ForestFireEnvHelicopterV0(CAEnv):
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

    _n_actions        = len(CONFIG["actions"])
    _action_sets      = CONFIG["actions_sets"]

    _reward_per_empty = CONFIG["rewards"]["per_empty"]
    _reward_per_tree  = CONFIG["rewards"]["per_tree"]
    _reward_per_fire  = CONFIG["rewards"]["per_fire"]
    # fmt: on

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self._set_spaces()

        self.cellular_automaton = ForestFire(
            self._empty, self._tree, self._fire, **self.ca_space
        )

        self.move = Move(self._action_sets, **self.move_space)
        self.modify = Modify(self._effects, **self.modify_space)

        self.move_modify = MoveModify(self.move, self.modify, **self.move_modify_space)

        # Composite Operators
        self._MDP = MDP(
            self.cellular_automaton,
            self.move_modify,
            self._max_freeze,
            **self.MDP_space,
        )

    @property
    def MDP(self):
        return self._MDP

    @property
    def initial_state(self):

        if self._resample_initial:

            self.grid = self.grid_space.sample()

            ca_params = np.array([self._p_fire, self._p_tree])
            pos = np.array([self._row // 2, self._col // 2])
            freeze = np.array(self._max_freeze)
            self.context = ca_params, pos, freeze

            self._initial_state = self.grid, self.context

        self._resample_initial = False

        return self._initial_state

    def _award(self):
        dict_counts = self.count_cells(self.grid)

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

    def render(self, mode="human"):

        if mode == "human":

            ca_params, pos, freeze = self.context

            return add_helicopter(plot_grid(self.grid), pos)

        else:

            logger.warn(
                f"Undefined mode.\nAvailable modes {self.metadata['render.modes']}"
            )

    def _set_spaces(self):
        self.ca_params_space = spaces.Box(0.0, 1.0, shape=(2,))
        self.position_space = spaces.MultiDiscrete([self._row, self._col])
        self.freeze_space = spaces.Discrete(self._max_freeze + 1)

        self.context_space = spaces.Tuple(
            (self.ca_params_space, self.position_space, self.freeze_space)
        )

        self.grid_space = GridSpace(
            values=[self._empty, self._tree, self._fire],
            shape=(self._row, self._col),
        )

        # RL spaces

        self.action_space = spaces.Discrete(self._n_actions)
        self.observation_space = spaces.Tuple((self.grid_space, self.context_space))

        # Suboperators Spaces

        self.ca_space = {
            "grid_space": self.grid_space,
            "action_space": self.action_space,
            "context_space": self.ca_params_space,
        }

        self.move_space = {
            "grid_space": self.grid_space,
            "action_space": self.action_space,
            "context_space": self.position_space,
        }

        self.modify_space = {
            "grid_space": self.grid_space,
            "action_space": spaces.Discrete(2),
            "context_space": self.position_space,
        }

        self.move_modify_space = {
            "grid_space": self.grid_space,
            "action_space": spaces.Tuple((self.action_space, spaces.Discrete(2))),
            "context_space": self.position_space,
        }

        self.MDP_space = {
            "grid_space": self.grid_space,
            "action_space": self.action_space,
            "context_space": self.context_space,
        }


class MDP(Operator):
    from collections import namedtuple

    Suboperators = namedtuple("Suboperators", ["cellular_automaton", "move_modify"])

    grid_dependant = True
    action_dependant = True
    context_dependant = True

    deterministic = False

    def __init__(self, cellular_automaton, move_modify, max_freeze, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.move_modify = move_modify
        self.ca = cellular_automaton

        self.suboperators = self.Suboperators(cellular_automaton, move_modify)

        self.max_freeze = max_freeze
        self.freeze_space = spaces.Discrete(max_freeze + 1)

    def update(self, grid, action, context):

        ca_params, position, freeze = context

        if freeze == 0:

            grid, ca_params = self.ca(grid, None, ca_params)
            grid, position = self.move_modify(grid, (action, True), position)

            freeze = np.array(self.max_freeze)

        else:

            grid, position = self.move_modify(grid, (action, True), position)

            freeze = np.array(freeze - 1)

        context = ca_params, position, freeze

        return grid, context
