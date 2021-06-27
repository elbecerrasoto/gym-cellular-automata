import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import gym_cellular_automata.forest_fire.bulldozer.utils.render.svg_paths as svg_paths
from gym_cellular_automata.forest_fire.utils.render import parse_svg_into_mpl


def get_font(ttfpath):
    from pathlib import Path

    return Path(mpl.get_data_path(), ttfpath)


def init_env():
    from gym_cellular_automata.forest_fire.bulldozer import ForestFireEnvBulldozerV1

    env = ForestFireEnvBulldozerV1()
    env.reset()
    return env


def get_norm_cmap(values, colors):
    from matplotlib.colors import BoundaryNorm, ListedColormap

    norm = BoundaryNorm(values, len(values), extend="max")
    cmap = ListedColormap(colors)
    return norm, cmap


plt.style.use("seaborn-whitegrid")

COLOR_EMPTY = "#DDD1D3"  # Gray
COLOR_BURNED = "#DFA4A0"  # Light-Red
COLOR_TREE = "#A9C499"  # Green
COLOR_FIRE = "#E68181"  # Salmon-Red
COLOR_OLDGAUGE = "#D1B2EA"  # Light-Purple
COLOR_NEWGAUGE = "#B991D9"  # Purple

EMOJIFONT = get_font("/home/ebecerra/.fonts/OpenMoji-Black.ttf")
TITLEFONT = get_font(
    "/home/ebecerra/.fonts/frederika_the_great/FrederickatheGreat-Regular.ttf"
)


ENV = init_env()

EMPTY = ENV._empty
BURNED = ENV._burned
TREE = ENV._tree
FIRE = ENV._fire

COLORS = [COLOR_EMPTY, COLOR_BURNED, COLOR_TREE, COLOR_FIRE]
CELLS = [EMPTY, BURNED, TREE, FIRE]
norm, cmap = get_norm_cmap(CELLS, COLORS)


def clear_ax(ax, xticks=True, yticks=True):
    ax.grid([])

    if xticks:
        ax.set_xticklabels([])
    if yticks:
        ax.set_yticklabels([])

    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.spines.left.set_visible(False)
    ax.spines.bottom.set_visible(False)


def plot_gauge(ax, env):

    old_gauge = np.array([0.4])
    new_gauge = np.array([0.1])
    y = np.array([0])

    # Old Gauge
    ax.barh(y, old_gauge, height=0.04, color=COLOR_OLDGAUGE, edgecolor="None")
    # New Gauge
    ax.barh(
        y,
        new_gauge,
        height=0.04,
        color=COLOR_NEWGAUGE,
        left=old_gauge,
        edgecolor="None",
    )

    ax.set_yticks([0])
    ax.set_xlim(0 - 0.1, 1 + 0.1)
    ax.set_ylim(-0.4, 0.4)

    ucycle = "\U0001f504"
    ax.set_yticklabels(ucycle, font=EMOJIFONT, size=34)

    ax.get_yticklabels()[0].set_color("0.74")

    ax.set_xticks([0.0, 1.0])

    clear_ax(ax, yticks=False)

    ax.grid(axis="x", color="0.86")


def plot_counts(ax, env):
    utree = "\U0001f332"
    uburned = "\ue08a"

    dcounts = env.count_cells()
    tree = env._tree
    burned = env._burned
    fire = env._fire
    empty = env._empty

    # Rocks & Burned
    ax.bar(
        [0, 1],
        [dcounts[empty], dcounts[burned]],
        width=0.1,
        color=[COLOR_EMPTY, COLOR_BURNED],
    )
    # Tree & Fire
    ax.bar(
        [0, 1],
        [dcounts[tree], dcounts[fire]],
        width=0.1,
        color=[COLOR_TREE, COLOR_FIRE],
        bottom=[dcounts[empty], dcounts[burned]],
    )

    ax.set_xticks(np.arange(2))
    ax.set_xticklabels([utree, uburned], font=EMOJIFONT, size=34)
    ax.set_yticks(np.linspace(0, env._row * env._col, 3, dtype=int))

    for label, color in zip(ax.get_xticklabels(), [COLOR_TREE, COLOR_BURNED]):
        label.set_color(color)

    # Off set to plot grid lines
    goff = 2048
    ax.set_ylim(0 - goff, env._row * env._col + goff)
    ax.set_xlim(-1, 2)
    clear_ax(ax, xticks=False)
    ax.grid(axis="y", color="0.94")


def plot_global_grid(ax, env):
    size = 17
    from gym_cellular_automata.forest_fire.bulldozer.utils import svg_paths

    grid = env.grid
    __, pos, __ = env.context

    ax.imshow(grid, interpolation="none", cmap=cmap, norm=norm)

    # Fire Seed
    svg_fire = parse_svg_into_mpl(svg_paths.FIRE)
    fire_seed = env._fire_seed
    offset = 10

    if fire_seed[0] - offset >= 0:
        # Position the Fire svg for better visualization
        ax.plot(
            fire_seed[1],
            fire_seed[0] - offset,
            marker=svg_fire,
            markersize=size,
            color=COLOR_FIRE,
        )
    else:
        # If offset just put a point
        ax.plot(
            fire_seed[1], fire_seed[0], marker=".", markersize=size, color=COLOR_FIRE
        )

    # Bulldozer Location
    svg_location = parse_svg_into_mpl(svg_paths.LOCATION)

    # Off set
    offset = 15

    if pos[0] - offset >= 0:
        ax.plot(
            pos[1], pos[0] - offset, marker=svg_location, markersize=size, color="1.0"
        )
    else:
        ax.plot(pos[1], pos[0], marker=".", markersize=size, color="1.0")

    clear_ax(ax)


def plot_local_grid(ax, env):
    grid = env.grid
    __, pos, __ = env.context

    from gym_cellular_automata.forest_fire.utils.neighbors import moore_n

    bd_hood = moore_n(3, pos, grid, EMPTY)
    n_row, n_col = bd_hood.shape
    mid_row, mid_col = n_row // 2, n_row // 2

    ax.imshow(bd_hood, interpolation="none", cmap=cmap, norm=norm)

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

    svg_bulldozer = parse_svg_into_mpl(svg_paths.BULLDOZER)
    ax.plot(mid_col, mid_row, marker=svg_bulldozer, markersize=17, color="1.0")
    clear_ax(ax)


def render(env):
    fig_shape = (12, 14)
    fig = plt.figure(figsize=(8, 5))
    ax_gauge = plt.subplot2grid(fig_shape, (0, 0), colspan=8, rowspan=2)
    ax_lgrid = plt.subplot2grid(fig_shape, (2, 0), colspan=8, rowspan=10)
    ax_ggrid = plt.subplot2grid(fig_shape, (0, 8), colspan=6, rowspan=6)
    ax_counts = plt.subplot2grid(fig_shape, (6, 8), colspan=6, rowspan=6)
    plot_gauge(ax_gauge, env)
    plot_local_grid(ax_lgrid, env)
    plot_global_grid(ax_ggrid, env)
    plot_counts(ax_counts, env)


# Things to do with this
# 1. Convert it into a function DONE
# 2. Interface with Bulldozer
# 3. Improve it.


def main():
    import gym

    env = gym.make("gym_cellular_automata:ForestFireBulldozer-v0")
    obs = env.reset()
    total_reward = 0.0
    done = False
    step = 0
    threshold = 12

    render(env)

    # Random Policy for at most "threshold" steps
    while not done and step < threshold:
        action = env.action_space.sample()  # Your agent goes here!
        obs, reward, done, info = env.step(action)
        render(env)
        total_reward += reward
        step += 1

    print(f"Total Steps: {step}")
    print(f"Total Reward: {total_reward}")


main()
