"""
Visualization Prototype
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import BoundaryNorm, ListedColormap

from gym_cellular_automata.forest_fire.bulldozer.utils.config import CONFIG
from gym_cellular_automata.forest_fire.utils.neighbors import moore_n

sns.set_style("whitegrid")


EMPTY = CONFIG["cell_symbols"]["empty"]
BURNED = CONFIG["cell_symbols"]["burned"]
TREE = CONFIG["cell_symbols"]["tree"]
FIRE = CONFIG["cell_symbols"]["fire"]


COLOR_EMPTY = "#DDD1D3"  # Gray
COLOR_BURNED = "#DFA4A0"  # Red
COLOR_TREE = "#A9C499"  # Green
COLOR_FIRE = "IndianRed"


def env_visualization(grid, pos, fire_seed):
    """
    Local-Global Approach
    """

    colors = (COLOR_EMPTY, COLOR_BURNED, COLOR_TREE, COLOR_FIRE)
    values = (EMPTY, BURNED, TREE, FIRE)
    norm = BoundaryNorm(values, len(colors), extend="max")
    cmap = ListedColormap(colors)

    fig, axs = plt.subplots(1, 2)

    # left, right
    ax_hood, ax_grid = axs

    # -------- Global

    ax_grid.imshow(grid, interpolation="none", cmap=cmap, norm=norm)
    ax_grid.set_xticklabels([])
    ax_grid.set_yticklabels([])
    ax_grid.grid([])
    ax_grid.grid([])

    ax_grid.spines["right"].set_visible(False)
    ax_grid.spines["top"].set_visible(False)
    ax_grid.spines["left"].set_visible(False)
    ax_grid.spines["bottom"].set_visible(False)

    # Fire Seed
    ax_grid.plot(fire_seed[1], fire_seed[0], marker="o", markersize=6, color=COLOR_FIRE)

    # Bulldozer
    ax_grid.plot(pos[1], pos[0], marker="$B$", markersize=6, color="1.0")

    # -------- Local
    bd_hood = moore_n(4, pos, grid, EMPTY)

    n_row, n_col = bd_hood.shape
    mid_row, mid_col = n_row // 2, n_row // 2

    ax_hood.imshow(bd_hood, interpolation="none", cmap=cmap, norm=norm)

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

    ax_hood.spines["right"].set_visible(False)
    ax_hood.spines["top"].set_visible(False)
    ax_hood.spines["left"].set_visible(False)
    ax_hood.spines["bottom"].set_visible(False)

    ax_hood.plot(mid_col, mid_row, marker="$B$", markersize=12, color="1.0")

    plt.show()
    return plt.gcf()
