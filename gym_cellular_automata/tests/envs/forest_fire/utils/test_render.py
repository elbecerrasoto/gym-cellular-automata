import numpy as np
import matplotlib.pyplot as plt

from gym import spaces

from gym_cellular_automata.envs.forest_fire.utils.render import (
    plot_grid,
    add_helicopter_cross,
)
from gym_cellular_automata.envs.forest_fire.utils.config import (
    get_forest_fire_config_dict,
)

CONFIG = get_forest_fire_config_dict()

CELL_STATES = CONFIG["cell_states"]

ROW = CONFIG["grid_shape"]["n_row"]
COL = CONFIG["grid_shape"]["n_col"]

CELL_TYPE = np.uint8

TITLE = "My Awesome Plot"

EMPTY = CONFIG["cell_symbols"]["empty"]
TREE = CONFIG["cell_symbols"]["tree"]
FIRE = CONFIG["cell_symbols"]["fire"]
SYMBOLS = EMPTY, TREE, FIRE

COLOR_EMPTY = CONFIG["cell_colors"]["empty"]
COLOR_TREE = CONFIG["cell_colors"]["tree"]
COLOR_FIRE = CONFIG["cell_colors"]["fire"]


def test_color_ordering():
    grid_space = spaces.Box(0, CELL_STATES - 1, shape=(ROW, COL), dtype=CELL_TYPE)
    grid = grid_space.sample()

    figure = plot_grid(grid, TITLE)

    colors_forest = figure.axes[0].images[0].cmap.colors

    symbols_with_colors = {EMPTY: COLOR_EMPTY, TREE: COLOR_TREE, FIRE: COLOR_FIRE}

    sorted_keys = sorted(symbols_with_colors)

    assert symbols_with_colors[sorted_keys[0]] == colors_forest[0]
    assert symbols_with_colors[sorted_keys[1]] == colors_forest[1]
    assert symbols_with_colors[sorted_keys[2]] == colors_forest[2]


def visualize_cross():
    grid_space = spaces.Box(0, CELL_STATES - 1, shape=(ROW, COL), dtype=CELL_TYPE)
    grid = grid_space.sample()

    figure = plot_grid(grid, TITLE)
    my_new_figure = add_helicopter_cross(figure, (3, 3))

    plt.show(my_new_figure)
