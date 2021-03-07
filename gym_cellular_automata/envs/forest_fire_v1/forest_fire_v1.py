# !!!!!!!!! BUG on CA update

# Layout is key

# Env Layout

# Moving the globals a little
from operator import mul
from functools import reduce

# Logic?
# Already done I think

# Add init pos
# Add termination
# Render and Close to pass
import numpy as np
from collections import Counter

import gym
from gym import spaces
from gym.utils import seeding

from gym_cellular_automata.envs.forest_fire_v1.operators import WindyForestFire, Bulldozer, Coordinator
from gym_cellular_automata.envs.forest_fire_v1.utils.config import CONFIG

ROW = CONFIG["grid_shape"]["n_row"]
COL = CONFIG["grid_shape"]["n_col"]

EFFECTS = CONFIG["effects"]

# Max freeze gets determined by size
# NUMERATOR = 1
# DENOMINATOR = 4  
# MAX_FREEZE = NUMERATOR * ROW * COL // 2 * DENOMINATOR

MAX_FREEZE = 64


# spaces.Box requires typing for discrete values
CELL_TYPE = CONFIG["cell_type"]
ACTION_TYPE = CONFIG["action_type"]

WIND_TYPE = np.float64


EMPTY = CONFIG["cell_symbols"]["empty"]
TREE = CONFIG["cell_symbols"]["tree"]
FIRE = CONFIG["cell_symbols"]["fire"]


def load_wind():
    UP_LEFT    = CONFIG["wind"]["up_left"]
    UP         = CONFIG["wind"]["up"]
    UP_RIGHT   = CONFIG["wind"]["up_right"]
    LEFT       = CONFIG["wind"]["left"]
    SELF       = CONFIG["wind"]["self"]
    RIGHT      = CONFIG["wind"]["right"]
    DOWN_LEFT  = CONFIG["wind"]["down_left"]
    DOWN       = CONFIG["wind"]["down"]
    DOWN_RIGHT = CONFIG["wind"]["down_right"]
    
    WIND = [[UP_LEFT,   UP,   UP_RIGHT  ],
            [LEFT,      SELF, RIGHT     ],
            [DOWN_LEFT, DOWN, DOWN_RIGHT]]
    		
    return np.array(WIND, dtype=WIND_TYPE)


WIND = load_wind()


def random_grid(shape=(ROW, COL), probs=[0.20, 0.80, 0.00]):
    size = reduce(mul, shape)
    cell_values = np.array([EMPTY, TREE, FIRE], dtype=CELL_TYPE)

    return np.random.choice(cell_values, size, probs).reshape(shape)



# ------------ Forest Fire Environment


class ForestFireEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    empty = EMPTY
    tree = TREE
    fire = FIRE

    ca_params_space = context_space = spaces.Box(0.0, 1.0, shape=(3, 3), dtype=WIND_TYPE)
    pos_space = spaces.MultiDiscrete([ROW, COL])
    freeze_space = spaces.Discrete(MAX_FREEZE + 1)

    context_space = spaces.Tuple((ca_params_space, pos_space, freeze_space))
    grid_space = spaces.Box(0, 10, shape=(ROW, COL), dtype=CELL_TYPE)

    action_space = action_space = spaces.MultiDiscrete([
                len(CONFIG["actions"]["movement"]),
                len(CONFIG["actions"]["shooting"])
            ])
    observation_space = spaces.Tuple((grid_space, context_space))

    def __init__(self):

        self.cellular_automaton = WindyForestFire(
            grid_space=self.grid_space,
            action_space=self.action_space,
            context_space=self.ca_params_space,
        )

        self.modifier = Bulldozer(
            EFFECTS,
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
        self.grid = random_grid()
        
        fseed_row, fseed_col = self.pos_space.sample()

        self.grid[fseed_row, fseed_col] == FIRE

        ca_params = WIND
        pos = np.array([ROW // 2, COL // 2])
        freeze = np.array(MAX_FREEZE)

        self.context = ca_params, pos, freeze

        obs = self.grid, self.context

        return obs

    def step(self, action):
        done = self._is_done()

        if not done:

            new_grid, new_context = self.coordinator(self.grid, action, self.context)

            obs = new_grid, new_context
            reward = self._award()
            info = self._report()

            self.grid = new_grid
            self.context = new_context

            return obs, reward, done, info
        
        else:
            
            raise Exception("Task is Done")

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
        return not bool( np.any(self.grid == self.fire) )

    def _report(self):
        return {}

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode="human"):
        pass
