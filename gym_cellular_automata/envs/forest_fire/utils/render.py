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
SYMBOLS = EMPTY, TREE, FIRE


def plot_grid(grid, title=CONFIG["plot_title"], **kwargs):
    n_row, n_col = grid.shape

    default_kwargs = {
        "title_font": {"fontname": CONFIG["plot_font"]},
        "color_empty": CONFIG["cell_colors"]["empty"],
        "color_tree": CONFIG["cell_colors"]["tree"],
        "color_fire": CONFIG["cell_colors"]["fire"],
    }

    kwargs = {**default_kwargs, **kwargs}

    title_font = kwargs["title_font"]

    color_empty = kwargs["color_empty"]
    color_tree = kwargs["color_tree"]
    color_fire = kwargs["color_fire"]
    colors_forest = color_empty, color_tree, color_fire

    symbols_with_colors = zip(SYMBOLS, colors_forest)
    symbols_with_colors_sorted_by_symbol = sorted(
        symbols_with_colors, key=itemgetter(0)
    )
    symbols, colors_forest = zip(*symbols_with_colors_sorted_by_symbol)

    # Mapping from Color to Cell Symbols
    cmap_colors = colors.ListedColormap(colors_forest)
    norm_symbols = colors.BoundaryNorm([0, 1, 2, 3], 3)

    # Plot style
    sns.set_style("whitegrid")

    # Main Plot
    plt.imshow(grid, aspect="equal", cmap=cmap_colors, norm=norm_symbols)

    # Title showing Reward
    plt.title(title, **title_font, color="0.7")

    # Modify Axes
    ax = plt.gca()

    # Major ticks
    ax.set_xticks(np.arange(0, n_col, 1))
    ax.set_yticks(np.arange(0, n_row, 1))

    # Labels for major ticks
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # Minor ticks
    ax.set_xticks(np.arange(-0.5, n_col, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_row, 1), minor=True)

    # Gridlines based on minor ticks
    ax.grid(which="minor", color="whitesmoke", linestyle="-", linewidth=2)
    ax.grid(which="major", color="w", linestyle="-", linewidth=0)
    ax.tick_params(axis=u"both", which=u"both", length=0)

    fig = plt.gcf()
    return fig


def parse_svg_into_mpl(svg_path):

    mpl_path = parse_path(svg_path)

    def center(mpl_path):
        mpl_path.vertices -= mpl_path.vertices.mean(axis=0)
        return mpl_path

    def upsidedown(mpl_path):
        mpl_path.vertices[:, 1] *= -1
        return mpl_path

    return upsidedown(center(mpl_path))


def add_helicopter(fig, pos):
    helicopter = parse_svg_into_mpl(SVG_PATH)

    ax = fig.get_axes()[0]
    row, col = pos

    ax.plot(col, row, marker=helicopter, markersize=39, color="1", fillstyle="none")

    return fig
