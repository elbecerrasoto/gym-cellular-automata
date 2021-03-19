"""
Visualization Prototype
"""


from gym_cellular_automata.envs.forest_fire_v1 import ForestFireEnv
from gym_cellular_automata.envs.forest_fire_v1.utils.config import CONFIG
from gym_cellular_automata.envs.forest_fire_v1.utils.grid import Grid
from gym_cellular_automata.envs.forest_fire_v1.utils.neighbors import (
    neighborhood_at,
    moore_n,
)


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm


sns.set_style("whitegrid")
# plt.style.use('fivethirtyeight')
# plt.style.use('fast')
# plt.style.use("dark_background")
# plt.style.use('default')

WIND = CONFIG["wind"]

EMPTY = CONFIG["cell_symbols"]["empty"]
BURNED = CONFIG["cell_symbols"]["burned"]
TREE = CONFIG["cell_symbols"]["tree"]
FIRE = CONFIG["cell_symbols"]["fire"]


COLOR_EMPTY = "#DDD1D3"  # Gray
COLOR_BURNED = "#DFA4A0"  # Red
COLOR_TREE = "#A9C499"  # Green
COLOR_FIRE = "IndianRed"


def prot1(grid, color_mapping):
    """
    Values must be ascending ordered.
    """

    values, colors = zip(*color_mapping)

    norm = BoundaryNorm(values, len(colors), extend="max")
    cmap = ListedColormap(colors)

    plt.imshow(grid, interpolation="none", cmap=cmap, norm=norm)

    plt.title(label="Forest Fire", size="32", color="0.7", fontname="Lato")
    plt.colorbar()
    plt.show()


def prot2():
    plt.imshow(grid, interpolation="none", cmap=cmap, norm=norm)

    plt.title(label="Forest Fire", size="32", color="0.7", fontname="Lato")

    ax = plt.gca()

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid([])
    ax.grid([])
    plt.show()


grid_space = Grid(values=[EMPTY, BURNED, TREE, FIRE], shape=(256, 256))
grid = grid_space.sample()

colors = (COLOR_EMPTY, COLOR_BURNED, COLOR_TREE, COLOR_FIRE)
values = (EMPTY, BURNED, TREE, FIRE)
prot1(grid, tuple(zip(values, colors)))


# ------------------


env = ForestFireEnv()
obs = env.reset()
grid, context = obs

wind, pos, freeze = context
env._fire_seed


# ------------------

colors = (COLOR_EMPTY, COLOR_BURNED, COLOR_TREE, COLOR_FIRE)
values = (EMPTY, BURNED, TREE, FIRE)
norm = BoundaryNorm(values, len(colors), extend="max")
cmap = ListedColormap(colors)


fig, axs = plt.subplots(1, 2)

ax_grid, ax_hood = axs

ax_grid.imshow(grid, interpolation="none", cmap=cmap, norm=norm)
ax_grid.set_xticklabels([])
ax_grid.set_yticklabels([])
ax_grid.grid([])
ax_grid.grid([])


# ------------------


ax_hood = plt.subplot()

bd_hood = moore_n(grid, pos, n=4, invariant=EMPTY)

bd_hood

n_col, n_row = bd_hood.shape


ax_hood.imshow(bd_hood, interpolation="none", cmap=cmap, norm=norm)
# ax_hood.plot(pos[1], pos[0])

# NO Labels for ticks
ax_hood.set_xticklabels([])
ax_hood.set_yticklabels([])

# Major ticks
ax_hood.set_xticks(np.arange(0, n_col, 1))
ax_hood.set_yticks(np.arange(0, n_row, 1))

# Minor ticks
ax_hood.set_xticks(np.arange(-0.5, n_col, 1), minor=True)
ax_hood.set_yticks(np.arange(-0.5, n_row, 1), minor=True)

# Gridlines based on minor ticks
ax_hood.grid(which="minor", color="whitesmoke", linestyle="-", linewidth=2)
ax_hood.grid(which="major", color="w", linestyle="-", linewidth=0)
ax_hood.tick_params(axis="both", which="both", length=0)

plt.show()
