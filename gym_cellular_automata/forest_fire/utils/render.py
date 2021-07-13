from gym_cellular_automata import PROJECT_PATH

EMOJIFONT = PROJECT_PATH / "fonts/OpenMoji-Black.ttf"
TITLEFONT = PROJECT_PATH / "fonts/FrederickatheGreat-Regular.ttf"


def parse_svg_into_mpl(svg_path):
    from svgpath2mpl import parse_path

    mpl_path = parse_path(svg_path)

    def center(mpl_path):
        mpl_path.vertices -= mpl_path.vertices.mean(axis=0)
        return mpl_path

    def upsidedown(mpl_path):
        mpl_path.vertices[:, 1] *= -1
        return mpl_path

    return upsidedown(center(mpl_path))


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


def get_norm_cmap(values, colors):
    """Assumes ordering of values and Colors"""
    from matplotlib.colors import BoundaryNorm, ListedColormap

    norm = BoundaryNorm(values, len(values), extend="max")
    cmap = ListedColormap(colors)
    return norm, cmap


def plot_local_grid(ax, grid, cmap, norm):
    """
    Just the grid
    """

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

    clear_ax(ax)
    return ax.imshow(grid, interpolation="none", cmap=cmap, norm=norm)
