from collections import Counter
from math import isclose
import numpy as np

import gym
from gym import spaces
from gym.utils import seeding

from gym_cellular_automata.envs.forest_fire_v1.operators import (
    WindyForestFire,
    Bulldozer,
    Coordinator,
)
from gym_cellular_automata.envs.forest_fire_v1.utils.grid import Grid
from gym_cellular_automata.envs.forest_fire_v1.utils.config import CONFIG

ROW = CONFIG["grid_shape"]["n_row"]
COL = CONFIG["grid_shape"]["n_col"]

MAX_FREEZE = CONFIG["max_freeze"]

EMPTY = CONFIG["cell_symbols"]["empty"]
TREE = CONFIG["cell_symbols"]["tree"]
FIRE = CONFIG["cell_symbols"]["fire"]

WIND = CONFIG["wind"]

# Initial Cell Probabilities
P_TREE = CONFIG["p_tree"]
P_EMPTY = CONFIG["p_empty"]

assert isclose(1.0, P_TREE + P_EMPTY)

EPSILON_FIRE = 8e-6

def set_grid_space(epsilon):
    return Grid(values=[EMPTY, TREE, FIRE],
                      shape=(ROW, COL),
                      probs=[P_TREE-epsilon, P_EMPTY, epsilon])

GRID_SPACE = set_grid_space(EPSILON_FIRE)
POSITION_SPACE = spaces.MultiDiscrete([ROW, COL])

# Sample and plant fire seed
def initial_grid_distribution():
    grid_space = Grid(values=[EMPTY, TREE, FIRE],
                      shape=(ROW, COL),
                      probs=[P_TREE, P_EMPTY, 0.0])
    
    grid = grid_space.sample()

    row, col = POSITION_SPACE.sample()

    grid[row, col] = FIRE

    return grid    


def initial_context_distribution():
    wind = WIND
    position = POSITION_SPACE.sample()
    freeze = np.array(MAX_FREEZE)
   
    return wind, position, freeze



# ------------ Forest Fire Environment


WIND_TYPE = np.float64

class ForestFireEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    empty = EMPTY
    tree = TREE
    fire = FIRE

    ca_params_space = spaces.Box(
        0.0, 1.0, shape=(3, 3), dtype=WIND_TYPE)
    pos_space = POSITION_SPACE
    freeze_space = spaces.Discrete(MAX_FREEZE + 1)

    context_space = spaces.Tuple((ca_params_space, pos_space, freeze_space))
    grid_space = GRID_SPACE

    action_space = action_space = spaces.MultiDiscrete(
        [len(CONFIG["actions"]["movement"]), len(CONFIG["actions"]["shooting"])]
    )
    observation_space = spaces.Tuple((grid_space, context_space))

    def __init__(self):

        self.cellular_automaton = WindyForestFire(
            grid_space=self.grid_space,
            action_space=self.action_space,
            context_space=self.ca_params_space,
        )

        self.modifier = Bulldozer(
            grid_space=self.grid_space,
            action_space=self.action_space,
            context_space=self.pos_space,
        )

        self.coordinator = Coordinator(
            self.cellular_automaton, self.modifier, max_freeze=MAX_FREEZE
        )

        self.reward_per_empty = CONFIG["rewards"]["per_empty"]
        self.reward_per_tree = CONFIG["rewards"]["per_tree"]
        self.reward_per_fire = CONFIG["rewards"]["per_fire"]

    def reset(self):
        self.done = False
        
        self.grid = initial_grid_distribution()
        self.context = initial_context_distribution()

        return self.grid, self.context

    def step(self, action):

        if not self.done:

            new_grid, new_context = self.coordinator(self.grid, action, self.context)

            self.done = self._is_done()

            obs = new_grid, new_context
            reward = self._award()
            info = self._report()

            self.grid = new_grid
            self.context = new_context

            return obs, reward, self.done, info

        else:

            raise Exception("Task is Done")

        # else:
        #     if self.steps_beyond_done == 0:
        #         logger.warn(
        #             "You are calling 'step()' even though this "
        #             "environment has already returned done = True. You "
        #             "should always call 'reset()' once you receive 'done = "
        #             "True' -- any further steps are undefined behavior."
        #         )
        #     self.steps_beyond_done += 1
        #     reward = 0.0

        # return np.array(self.state), reward, done, {}


    def _award(self):
        dict_counts = Counter(self.grid.flatten().tolist())

        cell_counts = np.array(
            [dict_counts[self.empty], dict_counts[self.tree], dict_counts[self.fire]]
        )

        reward_weights = np.array(
            [self.reward_per_empty, self.reward_per_tree, self.reward_per_fire]
        )

        return np.dot(reward_weights, cell_counts)

    def _is_done(self):
        return not bool(np.any(self.grid == self.fire))

    def _report(self):
        return {}

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode="human"):
        pass
