import numpy as np
from gym import spaces

from gym_cellular_automata._config import TYPE_BOX
from gym_cellular_automata.ca_env import CAEnv
from gym_cellular_automata.forest_fire.operators import (
    Modify,
    Move,
    MoveModify,
    RepeatCA,
    WindyForestFire,
)
from gym_cellular_automata.grid_space import GridSpace
from gym_cellular_automata.operator import Operator

from .utils.render import render


class ForestFireBulldozerEnv(CAEnv):
    metadata = {"render.modes": ["human"]}

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

    def __init__(
        self,
        nrows,
        ncols,
        p_tree=0.90,
        p_empty=0.10,
        wind={
            "up_left": 0.48,
            "up": 0.64,
            "up_right": 0.98,
            "left": 0.12,
            "right": 0.64,
            "down_left": 0.06,
            "down": 0.12,
            "down_right": 0.48,
        },
        **kwargs
    ):

        super().__init__(nrows, ncols, **kwargs)

        self.title = "ForestFireBulldozer" + str(nrows) + "x" + str(ncols)

        # Variables, scale free
        (
            up_left,
            up,
            up_right,
            left,
            not_move,
            right,
            down_left,
            down,
            down_right,
        ) = range(9)

        self._shoots = {"shoot": 1, "none": 0}

        self._empty = 0
        self._tree = 3
        self._fire = 25

        self._p_tree = p_tree
        self._p_empty = p_empty

        self._wind = self.parse_wind(wind)

        self._effects = {self._tree: self._empty}

        self._t_env_any = 0.001
        self._t_act_none = 0.0

        # Variables, scale dependant variables
        self._t_act_move = 0.04
        self._t_act_shoot = 0.12

        self._moves = {
            "up_left": up_left,
            "up": up,
            "up_right": up_right,
            "left": left,
            "not_move": not_move,
            "right": right,
            "down_left": down_left,
            "down": down,
            "down_right": down_right,
        }

        self._action_sets = {
            "up": {up_left, up, up_right},
            "down": {down_left, down, down_right},
            "left": {up_left, left, down_left},
            "right": {up_right, right, down_right},
            "not_move": {not_move},
        }

        self._set_spaces()
        self._init_time_mappings()

        self.ca = WindyForestFire(self._empty, self._tree, self._fire, **self.ca_space)

        self.move = Move(self._action_sets, **self.move_space)
        self.modify = Modify(self._effects, **self.modify_space)

        # Composite Operators
        self.move_modify = MoveModify(self.move, self.modify, **self.move_modify_space)
        self.repeater = RepeatCA(
            self.ca, self.time_per_action, self.time_per_state, **self.repeater_space
        )
        self._MDP = MDP(self.repeater, self.move_modify, **self.MDP_space)

    # Gym API
    # step, reset & seed methods inherited from parent class

    def render(self, mode="human"):
        return render(self)

    def _award(self):
        """Reward Function

        Negative Ratio of Burning Area per Total Flammable Area

        -(f / (t + f))
        Where:
            t: tree cell counts
            f: fire cell counts

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

        The sparse reward is alive trees at epidose's end:
        t / (e + t + f)
        """
        counts = self.count_cells(self.grid)
        t = counts[self._tree]
        f = counts[self._fire]
        return -(f / (t + f))

    def _is_done(self):
        self.done = not bool(np.any(self.grid == self._fire))

    def _report(self):
        return {"hit": self.modify.hit}

    def _noise(self):
        """
        Noise to initial conditions. A circular deviation of 1/12 of the grid size.
        """
        l = ((self.nrows + self.ncols) // 2) // 12
        return int(self.np_random.choice(range(l), size=1))

    def _initial_grid_distribution(self):
        # fmt: off
        grid_space = GridSpace(
            values = [  self._empty,   self._tree,   self._fire],
            probs  = [self._p_empty, self._p_tree,          0.0],
            shape=(self.nrows, self.ncols),
        )
        # fmt: on

        grid = grid_space.sample()

        # Around the lower left quadrant
        r, c = (3 * self.nrows // 4), (1 * self.ncols // 4)
        row, col = self._fire_seed = r + self._noise(), c + self._noise()

        grid[row, col] = self._fire

        return grid

    def _initial_context_distribution(self):
        init_time = np.array(0.0)

        # Around the upper right quadrant
        r, c = (1 * self.nrows // 4), (3 * self.ncols // 4)

        initnrows = r + self._noise()
        initncols = c + self._noise()

        init_position = np.array([initnrows, initncols])

        init_context = self._wind, init_position, np.array(init_time, dtype=TYPE_BOX)

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

    def parse_wind(self, windD: dict) -> np.ndarray:
        from gym import spaces

        # fmt: off
        wind = np.array(
            [
                [ windD["up_left"]  , windD["up"]  , windD["up_right"]   ],
                [ windD["left"]     ,    0.0       , windD["right"]      ],
                [ windD["down_left"], windD["down"], windD["down_right"] ],
            ], dtype=TYPE_BOX
        )

        # fmt: on
        wind_space = spaces.Box(0.0, 1.0, shape=(3, 3))

        assert wind_space.contains(wind), "Bad Wind Data, check ranges [0.0, 1.0]"

        return wind

    def _set_spaces(self):
        self.grid_space = GridSpace(
            values=[self._empty, self._tree, self._fire],
            shape=(self.nrows, self.ncols),
        )

        self.ca_params_space = spaces.Box(0.0, 1.0, shape=(3, 3))
        self.position_space = spaces.MultiDiscrete([self.nrows, self.ncols])
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
