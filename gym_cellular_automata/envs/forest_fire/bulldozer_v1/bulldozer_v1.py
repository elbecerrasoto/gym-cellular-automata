import gym
import numpy as np
from gym import logger, spaces
from gym.utils import seeding

from gym_cellular_automata.envs.forest_fire.bulldozer_v0.utils.render import (
    env_visualization,
)
from gym_cellular_automata.envs.forest_fire.bulldozer_v1.config import CONFIG
from gym_cellular_automata.envs.forest_fire.bulldozer_v1.operators.ca_repeat import (
    CAThenOps,
    RepeatCA,
    SinglePass,
)
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

        self.cellular_automaton = WindyForestFire(
            self._empty, self._burned, self._tree, self._fire
        )
        self.move = Move(self._action_sets)
        self.modify = Modify(self._effects)

        # Composite Operators
        self.single_pass = SinglePass((self.move, self.modify))
        self.repeat_ca = RepeatCA(
            self.cellular_automaton, self.time_per_action, self.time_per_state
        )
        self.MDP = CAThenOps(self.repeat_ca, self.single_pass)

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

    def _set_spaces(self):
        self.grid_space = Grid(
            values=[self._empty, self._burned, self._tree, self._fire],
            shape=(self._row, self._col),
        )

        ca_params_space = spaces.Box(0.0, 1.0, shape=(3, 3))
        accu_space = spaces.Box(0.0, float("inf"), shape=tuple())

        rep_space = spaces.Tuple((ca_params_space, accu_space))

        pos_space = spaces.MultiDiscrete([self._row, self._col])
        logic_space = spaces.Discrete(2)

        single_space = spaces.Tuple((pos_space, logic_space))

        self.context_space = spaces.Tuple((rep_space, single_space))

        self.observation_space = spaces.Tuple((self.grid_space, self.context_space))

        dummy_space = spaces.Discrete(1)

        m, n = len(self._moves), len(self._shoots)
        move_shoot_space = spaces.MultiDiscrete([m, n])

        self.action_space = spaces.Tuple((move_shoot_space, move_shoot_space))

    def _initial_grid_distribution(self):
        # fmt: off
        grid_space = Grid(
            values = [  self._empty,  self._burned,   self._tree,  self._fire],
            probs  = [self._p_empty,           0.0, self._p_tree,         0.0],
            shape=(self._row, self._col),
        )
        # fmt: on

        grid = grid_space.sample()

        row, col = self._fire_seed = (3 * self._row // 4), (1 * self._col // 4)

        grid[row, col] = self._fire

        return grid

    def _initial_context_distribution(self):
        # Wind, Initial Accumulated Time
        rep_contexts = self._wind, 0.0
        # Initial Position
        init_row = 1 * self._row // 4
        init_col = 3 * self._col // 4

        # Position and Active effects 0 = False
        single_contexts = (init_row, init_col), 0

        return rep_contexts, single_contexts

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
