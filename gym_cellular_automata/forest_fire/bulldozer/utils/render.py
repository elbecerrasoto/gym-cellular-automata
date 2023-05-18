"""
The render of the bulldozer consists of four subplots:
1. Local Grid
    + Grid centered at current position, visualizes agent's micromanagment
2. Global Grid
    + Whole grid view, visualizes agent's strategy
3. Gauge
    + Shows time until next CA update
4. Counts
    + Shows Forest vs No Forest cell counts. Translates on how well the agent is doing.
"""
from warnings import filterwarnings

import matplotlib.pyplot as plt
import numpy as np

from gym_cellular_automata.forest_fire.utils.neighbors import moore_n
from gym_cellular_automata.forest_fire.utils.render import (
    EMOJIFONT,
    TITLEFONT,
    align_marker,
    clear_ax,
    get_norm_cmap,
    parse_svg_into_mpl,
    plot_grid,
)

from . import svg_paths

# Figure Globals
FIGSIZE = (15, 12)
FIGSTYLE = "seaborn-v0_8-whitegrid"

TITLE_SIZE = 42
TITLE_POS = {"x": 0.121, "y": 0.96}
TITLE_ALIGN = "left"

COLOR_EMPTY = "#DDD1D3"  # Gray
COLOR_TREE = "#A9C499"  # Green
COLOR_FIRE = "#E68181"  # Salmon-Red

# Local Grid
N_LOCAL = 3  # n x n local grid size
MARKBULL_SIZE = 52

# Global Grid
MARKFSEED_SIZE = 62
MARKLOCATION_SIZE = 62

# Gauge
COLOR_GAUGE = "#D4CCDB"  # "Gray-Purple"
CYCLE_SYMBOL = "\U0001f504"
CYCLE_SIZE = 32

# Counts
TREE_SYMBOL = "\U0001f332"
BURNED_SYMBOL = "\ue08a"


# Ignore warnings trigger by Bulldozer Render
# EmojiFont raises RuntimeWarning
filterwarnings("ignore", message="Glyph 108")
filterwarnings("ignore", message="Glyph 112")


def render(env):
    NROWS = env.nrows
    NCOLS = env.ncols

    EMPTY = env._empty
    TREE = env._tree
    FIRE = env._fire

    # Assumes that cells values are in ascending order and paired with its colors
    COLORS = [COLOR_EMPTY, COLOR_TREE, COLOR_FIRE]
    CELLS = [EMPTY, TREE, FIRE]
    NORM, CMAP = get_norm_cmap(CELLS, COLORS)

    grid = env.grid
    ca_params, pos, time = env.context

    local_grid = moore_n(N_LOCAL, pos, grid, EMPTY)
    pos_fseed = env._pos_fire

    # Why two titles?
    # The env was registered (benchmark) or
    # The env was directly created (prototype)
    TITLE = env.spec.id if env.spec is not None else env.title

    def main():
        plt.style.use(FIGSTYLE)
        fig_shape = (12, 14)
        fig = plt.figure(figsize=FIGSIZE)
        fig.suptitle(
            TITLE,
            font=TITLEFONT,
            fontsize=TITLE_SIZE,
            **TITLE_POS,
            color="0.6",
            ha=TITLE_ALIGN
        )

        ax_lgrid = plt.subplot2grid(fig_shape, (0, 0), colspan=8, rowspan=10)
        ax_ggrid = plt.subplot2grid(fig_shape, (0, 8), colspan=6, rowspan=6)
        ax_gauge = plt.subplot2grid(fig_shape, (10, 0), colspan=8, rowspan=2)
        ax_counts = plt.subplot2grid(fig_shape, (6, 8), colspan=6, rowspan=6)

        plot_local(ax_lgrid, local_grid)

        plot_global(ax_ggrid, grid, pos, pos_fseed)

        plot_gauge(ax_gauge, time)

        d = env.count_cells()
        counts = d[EMPTY], d[TREE], d[FIRE]
        plot_counts(ax_counts, *counts)

        return plt.gcf()

    def plot_local(ax, grid):
        nrows, ncols = grid.shape
        mid_row, mid_col = nrows // 2, nrows // 2

        plot_grid(ax, grid, interpolation="none", cmap=CMAP, norm=NORM)

        markbull = parse_svg_into_mpl(svg_paths.BULLDOZER)
        ax.plot(
            mid_col, mid_row, marker=markbull, markersize=MARKBULL_SIZE, color="1.0"
        )

    def plot_global(ax, grid, pos, pos_fseed):
        ax.imshow(grid, interpolation="none", cmap=CMAP, norm=NORM)

        # Fire Seed
        markfire = align_marker(parse_svg_into_mpl(svg_paths.FIRE), valign="bottom")

        ax.plot(
            pos_fseed[1],
            pos_fseed[0],
            marker=markfire,
            markersize=MARKFSEED_SIZE,
            color=COLOR_FIRE,
        )

        # Bulldozer Location
        marklocation = align_marker(
            parse_svg_into_mpl(svg_paths.LOCATION), valign="bottom"
        )

        ax.plot(
            pos[1],
            pos[0],
            marker=marklocation,
            markersize=MARKLOCATION_SIZE,
            color="1.0",
        )
        clear_ax(ax)

    def plot_gauge(ax, time):
        HEIGHT_GAUGE = 0.1
        ax.barh(0.0, time, height=HEIGHT_GAUGE, color=COLOR_GAUGE, edgecolor="None")

        ax.barh(
            0.0,
            1.0,
            height=0.15,
            color="None",
            edgecolor="0.86",
        )

        # Mess with x,y limits for aethetics reasons
        INCREASE_LIMS = True

        if INCREASE_LIMS:
            ax.set_xlim(0 - 0.03, 1 + 0.1)  # Breathing room
            ax.set_ylim(-0.4, 0.4)  # Center the bar

        ax.set_xticks([0.0, 1.0])  # Start Time and End Time x ticks

        # Set the CA update symbol
        ax.set_yticks([0])  # Set symbol position
        ax.set_yticklabels(CYCLE_SYMBOL, font=EMOJIFONT, size=CYCLE_SIZE)
        ax.get_yticklabels()[0].set_color("0.74")  # Light gray

        clear_ax(ax, yticks=False)

    def plot_counts(ax, counts_empty, counts_tree, counts_fire):
        counts_total = sum((counts_empty, counts_tree, counts_fire))

        commons = {"x": [0, 1], "width": 0.1}
        pc = "1.0"  # placeholder color

        lv1y = [counts_tree, 0]
        lv1c = [COLOR_TREE, pc]

        lv2y = [0, counts_empty]  # level 2 y axis
        lv2c = [pc, COLOR_EMPTY]  # level 2 colors
        lv2b = lv1y  # level 2 bottom

        lv3y = [0, counts_fire]
        lv3c = [pc, COLOR_FIRE]
        lv3b = [lv1y[i] + lv2y[i] for i in range(len(lv1y))]

        # First Level Bars
        ax.bar(height=lv1y, color=lv1c, **commons)

        # Second Level Bars
        ax.bar(height=lv2y, color=lv2c, bottom=lv2b, **commons)

        # Third Level Bars
        ax.bar(height=lv3y, color=lv3c, bottom=lv3b, **commons)

        # Bar Symbols Settings
        ax.set_xticks(np.arange(2))
        ax.set_xticklabels([TREE_SYMBOL, BURNED_SYMBOL], font=EMOJIFONT, size=34)
        # Same colors as bars
        for label, color in zip(ax.get_xticklabels(), [COLOR_TREE, COLOR_FIRE]):
            label.set_color(color)

        # Mess with x,y limits for aethetics reasons
        INCREASE_LIMS = True
        INCREASE_FACTORS = [0.1, 0.3]  # Y axis down, up

        if INCREASE_LIMS:
            # Makes the bars look long & tall, also centers them
            offdown, offup = (
                counts_total * INCREASE_FACTORS[i] for i in range(len(INCREASE_FACTORS))
            )
            ax.set_ylim(
                0 - offdown, counts_total + offup
            )  # It gives breathing room for bars
            ax.set_xlim(-1, 2)  # It centers the bars

        # Grid Settings and Tick settings
        # Show marks each quarter
        ax.set_yticks(np.linspace(0, counts_total, 3, dtype=int))
        # Remove clutter
        clear_ax(ax, xticks=False)
        # Add back y marks each quarter
        ax.grid(axis="y", color="0.94")  # Dim gray

    return main()
