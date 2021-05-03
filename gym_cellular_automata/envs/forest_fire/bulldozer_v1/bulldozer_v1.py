import gym
import numpy as np
from gym import logger, spaces
from gym.utils import seeding

from gym_cellular_automata.envs.forest_fire.bulldozer_v1.utils.config import CONFIG
from gym_cellular_automata.envs.forest_fire.operators import (
    Modify,
    Move,
    Sequence,
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

        self._init_action_time_mappings()

        self.cellular_automaton = WindyForestFire(
            self._empty, self._burned, self._tree, self._fire, **self._ca_spaces
        )
        self.move = Move(self._action_sets, **self._move_spaces)
        self.modify = Modify(self._effects, **self._modify_spaces)
        self.sequence = Sequence(
            (self.cellular_automaton, self.move, self.modify), **self._seq_spaces
        )

        # Gym spec method
        self.seed()

    def reset(self):

        self.done = False
        self.steps_beyond_done = 0

        self.grid = self._initial_grid_distribution()
        self.context = self._initial_context_distribution()

        self.accumulated_time = 0.0

        return self.grid, self.context

    def step(self, action):

        if not self.done:

            # Action processing
            actions = None, action[0], action[1]

            # Context preprocessing
            operation_flow = self._get_operation_flow(action)
            context = self.context[0], operation_flow

            # MDP Transition
            self.grid, self.context = self.sequence(self.grid, actions, context)

            # Check for termination based on New State
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

    def _get_operation_flow(self, action):
        import math

        movement, shooting = action

        # Mapping of actions ---> to time (on units of CA updates)
        time_move = self._movement_timings[movement]
        time_shoot = self._shooting_timings[shooting]
        time_environment = self._t_env_any

        # The time taken on a step is the time taken doing the actions
        # plus some enviromental (internal) time.
        time_taken = time_move + time_shoot + time_environment

        self.accumulated_time += time_taken

        # Decimal and Integer parts
        self.accumulated_time, ca_repeats = math.modf(self.accumulated_time)

        ica, imove, imodify = range(3)

        # Operartors order is: CA, Modifier
        operation_flow = int(ca_repeats) * [ica] + [imove] + [imodify]

        return operation_flow

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode="human"):
        pass

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

        row, col = self._fire_seed = 100, 100

        grid[row, col] = self._fire

        return grid

    def _initial_context_distribution(self):
        return self._wind, 42, 42, [0]

    def _count_cells(self):
        """Returns dict of cell counts"""
        from collections import Counter

        return Counter(self.grid.flatten().tolist())

    def _set_spaces(self):

        self.grid_space = Grid(
            values=[self._empty, self._burned, self._tree, self._fire],
            shape=(self._row, self._col),
        )

        operation_flow_space = Grid(values=[1], shape=(3,))

        self.context_space = spaces.Tuple((operation_flow_space,))

        # RL Spaces
        self.observation_space = spaces.Tuple((self.grid_space, self.context_space))
        self.action_space = spaces.MultiDiscrete([len(self._moves), len(self._shoots)])

        # Operator spaces
        self._ca_spaces = {
            "grid_space": None,
            "action_space": None,
            "context_space": None,
        }
        self._move_spaces = {
            "grid_space": None,
            "action_space": None,
            "context_space": None,
        }
        self._modify_spaces = {
            "grid_space": None,
            "action_space": None,
            "context_space": None,
        }
        self._seq_spaces = {
            "grid_space": None,
            "action_space": None,
            "context_space": None,
        }

    def _init_action_time_mappings(self):

        self._movement_timings = {
            move: self._t_act_move for move in self._moves.values()
        }
        self._shooting_timings = {
            shoot: self._t_act_shoot for shoot in self._shoots.values()
        }

        self._movement_timings[self._moves["not_move"]] = self._t_act_none
        self._shooting_timings[self._shoots["none"]] = self._t_act_none
