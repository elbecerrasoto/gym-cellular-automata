from operator import itemgetter

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
from svgpath2mpl import parse_path

from .helicopter_shape import SVG_PATH
from .config import get_forest_fire_config_dict

CONFIG = get_forest_fire_config_dict()

EMPTY = CONFIG["cell_symbols"]["empty"]
TREE = CONFIG["cell_symbols"]["tree"]
FIRE = CONFIG["cell_symbols"]["fire"]

print(f"CONFIG {CONFIG}")

DEFAULT_KWARGS = {
    "color_empty": CONFIG["plot"]["cell_colors"]["empty"],
    "color_tree": CONFIG["plot"]["cell_colors"]["tree"],
    "color_fire": CONFIG["plot"]["cell_colors"]["fire"],
    "title_size": CONFIG["plot"]["title_size"],
    "title_color": CONFIG["plot"]["title_color"],
    "helicopter_size": CONFIG["plot"]["helicopter_size"],
    "helicopter_color": CONFIG["plot"]["helicopter_color"],
    "fontname": CONFIG["plot"]["font"],
}


def plot_grid(grid, title=CONFIG["plot"]["title"], **kwargs):

    kwargs = {**DEFAULT_KWARGS, **kwargs}

    color_mapping = cell_colors_to_cmap_and_norm(
        (kwargs["color_empty"], kwargs["color_tree"], kwargs["color_fire"])
    )

    # Plot style
    sns.set_style("whitegrid")

    # Main Plot
    plt.imshow(
        grid, aspect="equal", cmap=color_mapping["cmap"], norm=color_mapping["norm"]
    )

    # Title
    plt.title(
        title,
        size=kwargs["title_size"],
        color=kwargs["title_color"],
        fontname=kwargs["fontname"],
    )

    # Modify Ticks by Axes methods
    grid_ticks_settings(plt.gca(), n_row=grid.shape[0], n_col=grid.shape[1])

    fig = plt.gcf()
    return fig


def add_helicopter(fig, pos, **kwargs):
    helicopter = parse_svg_into_mpl(SVG_PATH)
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


def grid_ticks_settings(ax, n_row, n_col):

    # NO Labels for ticks
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # Major ticks
    ax.set_xticks(np.arange(0, n_col, 1))
    ax.set_yticks(np.arange(0, n_row, 1))

    # Minor ticks
    ax.set_xticks(np.arange(-0.5, n_col, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_row, 1), minor=True)

    # Gridlines based on minor ticks
    ax.grid(which="minor", color="whitesmoke", linestyle="-", linewidth=2)
    ax.grid(which="major", color="w", linestyle="-", linewidth=0)
    ax.tick_params(axis="both", which="both", length=0)

    return ax


def parse_svg_into_mpl(svg_path):

    mpl_path = parse_path(svg_path)

    def center(mpl_path):
        mpl_path.vertices -= mpl_path.vertices.mean(axis=0)
        return mpl_path

    def upsidedown(mpl_path):
        mpl_path.vertices[:, 1] *= -1
        return mpl_path

    return upsidedown(center(mpl_path))
