import numpy as np
from gym import spaces

from gym_cellular_automata import CAEnv, GridSpace, Operator
from gym_cellular_automata.forest_fire.bulldozer.utils.config import CONFIG
from gym_cellular_automata.forest_fire.operators import (
    Modify,
    Move,
    MoveModify,
    RepeatCA,
    WindyForestFire,
)


class ForestFireEnvBulldozerV1(CAEnv):
    metadata = {"render.modes": ["human"]}

    # fmt: off

    # Actions
    _moves           = CONFIG["actions"]["movement"]
    _shoots          = CONFIG["actions"]["shooting"]
    _action_sets     = CONFIG["actions"]["sets"]

    # Time parameters on CA updates units.
    _t_act_none      = CONFIG["time"]["ta_none"]
    _t_act_move      = CONFIG["time"]["ta_move"]
    _t_act_shoot     = CONFIG["time"]["ta_shoot"]
    _t_env_any       = CONFIG["time"]["te_any"]

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

    def __init__(self, rows=None, cols=None, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self._row = self._row if rows is None else rows
        self._col = self._col if cols is None else cols

        self._set_spaces()
        self._init_time_mappings()

        self.ca = WindyForestFire(
            self._empty, self._burned, self._tree, self._fire, **self.ca_space
        )

        self.move = Move(self._action_sets, **self.move_space)
        self.modify = Modify(self._effects, **self.modify_space)

        self.move_modify = MoveModify(self.move, self.modify, **self.move_modify_space)

        # Composite Operators
        self.repeater = RepeatCA(
            self.ca, self.time_per_action, self.time_per_state, **self.repeater_space
        )
        self._MDP = MDP(self.repeater, self.move_modify, **self.MDP_space)

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

    def _award(self):
        """Reward Function

        Negative Ratio of Burning Area per Total Flammable Area

        -[f / (t + f + b)]
        Where:
            t: tree cell counts
            f: fire cell counts
            b: burned cell counts

        Objective:
        Keep as much forest as possible.

        Advantages:
        1. Easy to interpret.
            + Percent of the forest lost at each step.
        2. Terminate ASAP.
            + As the reward is negative.
        3. Built-in cost of action.
            + The agent removes trees, this decreases the reward.
        4. Shaped reward.
            + Reward is given at each step.

        Disadvantages:
        1. Lack of experimental results.
        2. Is it equivalent with Sparse Reward?
        """
        counts = self.count_cells(self.grid)
        t = counts[self._tree]
        f = counts[self._fire]
        b = counts[self._burned]
        return -(f / (t + f + b))

    def _is_done(self):
        self.done = not bool(np.any(self.grid == self._fire))

    def _report(self):
        return {"hit": self.modify.hit}

    def render(self, mode="human"):
        from warnings import catch_warnings

        with catch_warnings():
            from gym_cellular_automata.forest_fire.bulldozer.utils.render import render

            return render(self)

    def _set_spaces(self):
        self.grid_space = GridSpace(
            values=[self._empty, self._burned, self._tree, self._fire],
            shape=(self._row, self._col),
        )

        self.ca_params_space = spaces.Box(0.0, 1.0, shape=(3, 3))
        self.position_space = spaces.MultiDiscrete([self._row, self._col])
        self.time_space = spaces.Box(0.0, float("inf"), shape=tuple())

        self.context_space = spaces.Tuple(
            (self.ca_params_space, self.position_space, self.time_space)
        )

        # RL spaces

        m, n = len(self._moves), len(self._shoots)
        self.action_space = spaces.MultiDiscrete([m, n])
        self.observation_space = spaces.Tuple((self.grid_space, self.context_space))

        # Suboperators Spaces

        self.ca_space = {
            "grid_space": self.grid_space,
            "action_space": self.action_space,
            "context_space": self.ca_params_space,
        }

        self.move_space = {
            "grid_space": self.grid_space,
            "action_space": spaces.Discrete(m),
            "context_space": self.position_space,
        }

        self.modify_space = {
            "grid_space": self.grid_space,
            "action_space": spaces.Discrete(n),
            "context_space": self.position_space,
        }

        self.move_modify_space = {
            "grid_space": self.grid_space,
            "action_space": self.action_space,
            "context_space": self.position_space,
        }

        self.repeater_space = {
            "grid_space": self.grid_space,
            "action_space": self.action_space,
            "context_space": spaces.Tuple((self.ca_params_space, self.time_space)),
        }

        self.MDP_space = {
            "grid_space": self.grid_space,
            "action_space": self.action_space,
            "context_space": self.context_space,
        }

    def _noise(self):
        l = ((self._row + self._col) // 2) // 12
        return int(np.random.choice(range(l), size=1))

    def _initial_grid_distribution(self):
        # fmt: off
        grid_space = GridSpace(
            values = [  self._empty,  self._burned,   self._tree,  self._fire],
            probs  = [self._p_empty,           0.0, self._p_tree,         0.0],
            shape=(self._row, self._col),
        )
        # fmt: on

        grid = grid_space.sample()

        r, c = (3 * self._row // 4), (1 * self._col // 4)
        row, col = self._fire_seed = r + self._noise(), c + self._noise()

        grid[row, col] = self._fire

        return grid

    def _initial_context_distribution(self):
        init_time = np.array(0.0)

        r, c = (1 * self._row // 4), (3 * self._col // 4)

        init_row = r + self._noise()
        init_col = c + self._noise()

        init_position = np.array([init_row, init_col])

        init_context = self._wind, init_position, init_time

        return init_context

    def _init_time_mappings(self):

        self._movement_timings = {
            move: self._t_act_move for move in self._moves.values()
        }
        self._shooting_timings = {
            shoot: self._t_act_shoot for shoot in self._shoots.values()
        }

        self._movement_timings[self._moves["not_move"]] = self._t_act_none
        self._shooting_timings[self._shoots["none"]] = self._t_act_none

        def time_per_action(action):
            move, shoot = action

            time_on_move = self._movement_timings[int(move)]
            time_on_shoot = self._shooting_timings[int(shoot)]

            return time_on_move + time_on_shoot

        self.time_per_action = time_per_action
        self.time_per_state = lambda s: self._t_env_any


class MDP(Operator):

    grid_dependant = True
    action_dependant = True
    context_dependant = True

    deterministic = False

    def __init__(self, repeat_ca, move_modify, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.repeat_ca = repeat_ca
        self.move_modify = move_modify

        self.suboperators = self.repeat_ca, self.move_modify

    def update(self, grid, action, context):

        amove, ashoot = action
        ca_params, position, time = context

        grid, (ca_params, time) = self.repeat_ca(grid, action, (ca_params, time))
        grid, position = self.move_modify(grid, action, position)

        return grid, (ca_params, position, time)
