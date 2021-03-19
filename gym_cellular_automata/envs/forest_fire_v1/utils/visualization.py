"""
Visualization Prototype
"""

from gym_cellular_automata.envs.forest_fire_v1 import ForestFireEnv
from gym_cellular_automata.envs.forest_fire_v1.utils.config import CONFIG
from gym_cellular_automata.envs.forest_fire_v1.utils.grid import Grid

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

# plt.style.use('fivethirtyeight')
plt.style.use('fast')
# plt.style.use("dark_background")
# plt.style.use('default')

WIND = CONFIG["wind"]

EMPTY = CONFIG["cell_symbols"]["empty"]
BURNED = CONFIG["cell_symbols"]["burned"]
TREE = CONFIG["cell_symbols"]["tree"]
FIRE = CONFIG["cell_symbols"]["fire"]


# COLOR_EMPTY = CONFIG["plot"]["cell_colors"]["empty"]
# COLOR_TREE = CONFIG["plot"]["cell_colors"]["tree"]
# COLOR_FIRE = CONFIG["plot"]["cell_colors"]["fire"]


COLOR_EMPTY = "C7"
COLOR_BURNED = "DarkRed"
COLOR_TREE = "C2"
COLOR_FIRE = "C3"

colors     = [COLOR_EMPTY, COLOR_BURNED, COLOR_TREE, COLOR_FIRE]
boundaries = [      EMPTY,       BURNED,       TREE,       FIRE]

cmap = ListedColormap(colors)
norm = BoundaryNorm(boundaries, len(colors), extend="max")

grid_space = Grid(values=[EMPTY, BURNED, TREE, FIRE], shape=(256, 256))
# grid = grid_space.sample()

env = ForestFireEnv()
obs = env.reset()
grid, context = obs


# Main Plot
plt.imshow(grid, interpolation="none", cmap=cmap, norm=norm)

plt.title(label="Forest Fire", size="32", color="0.7", fontname="Lato")
plt.colorbar()
plt.show()
