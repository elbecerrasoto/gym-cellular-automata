import gym
import numpy as np
from gym import logger, spaces
from gym.utils import seeding

from gym_cellular_automata.envs.forest_fire.bulldozer_v0.utils.render import (
    env_visualization,
)
from gym_cellular_automata.envs.forest_fire.bulldozer_v1.config import CONFIG
from gym_cellular_automata.envs.forest_fire.bulldozer_v1.operators.ca_repeat import (
    RepeatCA,
)
from gym_cellular_automata.envs.forest_fire.bulldozer_v1.operators.mdp import MDP
from gym_cellular_automata.envs.forest_fire.operators import (
    Modify,
    Move,
    WindyForestFire,
)
from gym_cellular_automata.grid_space import Grid


class ForestFireEnvBulldozerV1(gym.Env):
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

    def __init__(self, rows=None, cols=None):

        self._row = self._row if rows is None else rows
        self._col = self._col if cols is None else cols

        self._set_spaces()
        self._init_time_mappings()

        self.ca = WindyForestFire(
            self._empty, self._burned, self._tree, self._fire, **self.ca_space
        )
        self.move = Move(self._action_sets, **self.move_space)
        self.modify = Modify(self._effects, **self.modify_space)

        # Composite Operators
        self.repeater = RepeatCA(
            self.ca, self.time_per_action, self.time_per_state, **self.repeater_space
        )
        self.MDP = MDP(self.repeater, self.move, self.modify, **self.MDP_space)

        # Gym spec method
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
            self.grid, self.context = self.MDP(self.grid, action, self.context)

            # Check for termination
            self._is_done()

            # Gym API Formatting
            obs = self.grid, self.context
            reward = self._award()
            done = self.done
            info = self._report()

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
            return (self.grid, self.context), 0.0, True, self._report()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode="human"):
        if mode == "human":

            wind, time, position = self.context

            # Returning figure for convenience, formally render mode=human returns None
            return env_visualization(self.grid, position, self._fire_seed)

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

    def _set_spaces(self):
        self.grid_space = Grid(
            values=[self._empty, self._burned, self._tree, self._fire],
            shape=(self._row, self._col),
        )

        self.ca_params_space = spaces.Box(0.0, 1.0, shape=(3, 3))
        self.time_space = spaces.Box(0.0, float("inf"), shape=tuple())
        self.position_space = spaces.MultiDiscrete([self._row, self._col])

        self.context_space = spaces.Tuple(
            (self.ca_params_space, self.time_space, self.position_space)
        )

        # RL spaces

        m, n = len(self._moves), len(self._shoots)
        self.action_space = spaces.MultiDiscrete([m, n])
        self.observation_space = spaces.Tuple((self.grid_space, self.context_space))

        # Suboperators Spaces

        self.ca_space = {
            "grid_space": self.grid_space,
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

        self.repeater_space = {
            "grid_space": self.grid_space,
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
        grid_space = Grid(
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

        # assert self.ca_params_space.contains(self._wind)
        # assert self.time_space.contains(init_time)
        # assert self.position_space.contains(init_position)

        init_context = self._wind, init_time, init_position

        # assert self.context_space.contains(init_context)

        return init_context

    def _count_cells(self):
        """Returns dict of cell counts"""
        from collections import Counter

        return Counter(self.grid.flatten().tolist())

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
