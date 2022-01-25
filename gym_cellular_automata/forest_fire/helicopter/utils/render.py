import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt

from gym_cellular_automata.forest_fire.utils.render import (
    TITLEFONT,
    get_norm_cmap,
    parse_svg_into_mpl,
    plot_grid,
)

from .helicopter_shape import SVG_PATH

COLOR_EMPTY = "#DDD1D3"  # Gray
COLOR_TREE = "#A9C499"  # Green
COLOR_FIRE = "#DFA4A0"  # Red

TITLE_SIZE = 42
TITLE_COLOR = "#B3B3B3"  # Gray 70%
HELICOPTER_SIZE = 96
HELICOPTER_COLOR = "#FFFFFF"  # White


def render(env):
    grid = env.grid
    __, pos, __ = env.context
    row, col = pos

    plt.style.use("seaborn-whitegrid")
    fig, ax = plt.subplots(figsize=(15, 12))

    TITLE = env.spec.id

    EMPTY = env._empty
    TREE = env._tree
    FIRE = env._fire

    CELLS = [EMPTY, TREE, FIRE]
    COLORS = [COLOR_EMPTY, COLOR_TREE, COLOR_FIRE]

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
