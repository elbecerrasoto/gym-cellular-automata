from operator import itemgetter

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import colors

from gym_cellular_automata.forest_fire.utils.render import TITLEFONT, parse_svg_into_mpl

from .config import CONFIG
from .helicopter_shape import SVG_PATH

EMPTY = CONFIG["cell_symbols"]["empty"]
TREE = CONFIG["cell_symbols"]["tree"]
FIRE = CONFIG["cell_symbols"]["fire"]


DEFAULT_KWARGS = {
    "color_empty": "#DDD1D3",  # Gray
    "color_tree": "#A9C499",  # Green
    "color_fire": "#DFA4A0",  # Redo
    "title": "Save the Forest!",
    "title_size": 64,
    "title_color": "#B3B3B3",  # Gray 70%
    "helicopter_size": 96,
    "helicopter_color": "#FFFFFF",  # White
}


def plot_grid(grid, **kwargs):

    kwargs = {**DEFAULT_KWARGS, **kwargs}

    color_mapping = cell_colors_to_cmap_and_norm(
        (kwargs["color_empty"], kwargs["color_tree"], kwargs["color_fire"])
    )

    # Plot style
    sns.set_style("whitegrid")

    fig, ax = plt.subplots(figsize=(15, 12))

    # Main Plot
    ax.imshow(
        grid, aspect="equal", cmap=color_mapping["cmap"], norm=color_mapping["norm"]
    )

    # Title
    fig.suptitle(
        "Forest Fire",
        color=kwargs["title_color"],
        font=TITLEFONT,
        fontsize=64,
        ha="center",
    )

    # Modify Ticks by Axes methods
    grid_ticks_settings(plt.gca(), nrows=grid.shape[0], ncols=grid.shape[1])

    return fig


def add_helicopter(fig, pos, **kwargs):
    import matplotlib.patheffects as path_effects

    helicopter = parse_svg_into_mpl(SVG_PATH)
    pe = [path_effects.Stroke(linewidth=3, foreground="white"), path_effects.Normal()]
    kwargs = {**DEFAULT_KWARGS, **kwargs}

    ax = fig.get_axes()[0]
    row, col = pos

    ax.plot(
        col,
        row,
        marker=helicopter,
        markersize=kwargs["helicopter_size"],
        color=kwargs["helicopter_color"],
        fillstyle="none",
        path_effects=pe,
    )

    return fig


def cell_colors_to_cmap_and_norm(colors_forest):
    """
    Mappings from Color to Cell Symbols.
    """
    symbols = EMPTY, TREE, FIRE

    symbols_with_colors = zip(symbols, colors_forest)
    symbols_with_colors_sorted_by_symbol = sorted(
        symbols_with_colors, key=itemgetter(0)
    )
    symbols, colors_forest = zip(*symbols_with_colors_sorted_by_symbol)

    return {
        "cmap": colors.ListedColormap(colors_forest),
        "norm": colors.BoundaryNorm([0, 1, 2, 3], 3),
    }


def grid_ticks_settings(ax, nrows, ncols):

    # NO Labels for ticks
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # Major ticks
    ax.set_xticks(np.arange(0, ncols, 1))
    ax.set_yticks(np.arange(0, nrows, 1))

    # Minor ticks
    ax.set_xticks(np.arange(-0.5, ncols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, nrows, 1), minor=True)

    # Gridlines based on minor ticks
    ax.grid(which="minor", color="whitesmoke", linestyle="-", linewidth=2)
    ax.grid(which="major", color="w", linestyle="-", linewidth=0)
    ax.tick_params(axis="both", which="both", length=0)

    return ax
