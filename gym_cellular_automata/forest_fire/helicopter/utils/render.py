import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt

from gym_cellular_automata.forest_fire.utils.render import (
    TITLEFONT,
    get_norm_cmap,
    parse_svg_into_mpl,
    plot_grid,
)

from .config import CONFIG
from .helicopter_shape import SVG_PATH

EMPTY = CONFIG["cell_symbols"]["empty"]
TREE = CONFIG["cell_symbols"]["tree"]
FIRE = CONFIG["cell_symbols"]["fire"]

COLOR_EMPTY = "#DDD1D3"  # Gray
COLOR_TREE = "#A9C499"  # Green
COLOR_FIRE = "#DFA4A0"  # Red
TITLE = "Forest Fire"
TITLE_SIZE = 64
TITLE_COLOR = "#B3B3B3"  # Gray 70%
HELICOPTER_SIZE = 96
HELICOPTER_COLOR = "#FFFFFF"  # White

CELLS = [EMPTY, TREE, FIRE]
COLORS = [COLOR_EMPTY, COLOR_TREE, COLOR_FIRE]


def render(env):
    grid = env.grid
    __, pos, __ = env.context
    row, col = pos

    plt.style.use("seaborn-whitegrid")
    fig, ax = plt.subplots(figsize=(15, 12))

    # Title
    fig.suptitle(
        TITLE,
        color=TITLE_COLOR,
        font=TITLEFONT,
        fontsize=TITLE_SIZE,
        ha="center",
    )

    # Main Plot
    norm, cmap = get_norm_cmap(CELLS, COLORS)
    plot_grid(ax, grid, aspect="equal", norm=norm, cmap=cmap)

    # Helicopter Mark
    helicopter_mark = parse_svg_into_mpl(SVG_PATH)
    pe = [path_effects.Stroke(linewidth=3, foreground="white"), path_effects.Normal()]
    ax.plot(
        col,
        row,
        marker=helicopter_mark,
        markersize=HELICOPTER_SIZE,
        color=HELICOPTER_COLOR,
        fillstyle="none",
        path_effects=pe,
    )

    return fig
