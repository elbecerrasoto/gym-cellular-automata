import numpy as np

from gym_cellular_automata._config import PROJECT_PATH

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
    """Assumes ordering of values (ascending) and Colors"""
    from matplotlib.colors import BoundaryNorm, ListedColormap

    norm = BoundaryNorm(values, len(values), extend="max")
    cmap = ListedColormap(colors)
    return norm, cmap


def plot_grid(ax, grid, **kwargs):
    nrows, ncols = grid.shape

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
    return ax.imshow(grid, **kwargs)


def align_marker(
    marker,
    halign="center",
    valign="middle",
):
    """
    Code from:
    https://python.tutorialink.com/align-matplotlib-scatter-marker-left-and-or-right/
    create markers with specified alignment.

    Parameters
    ----------

    marker : a valid marker specification.
      See mpl.markers

    halign : string, float {'left', 'center', 'right'}
      Specifies the horizontal alignment of the marker. *float* values
      specify the alignment in units of the markersize/2 (0 is 'center',
      -1 is 'right', 1 is 'left').

    valign : string, float {'top', 'middle', 'bottom'}
      Specifies the vertical alignment of the marker. *float* values
      specify the alignment in units of the markersize/2 (0 is 'middle',
      -1 is 'top', 1 is 'bottom').

    Returns
    -------

    marker_array : numpy.ndarray
      A Nx2 array that specifies the marker path relative to the
      plot target point at (0, 0).

    Notes
    -----
    The mark_array can be passed directly to ax.plot and ax.scatter, e.g.::

        ax.plot(1, 1, marker=align_marker('>', 'left'))

    """

    from matplotlib import markers
    from matplotlib.path import Path

    if isinstance(halign, str):
        halign = {
            "right": -1.0,
            "middle": 0.0,
            "center": 0.0,
            "left": 1.0,
        }[halign]

    if isinstance(valign, str):
        valign = {
            "top": -1.0,
            "middle": 0.0,
            "center": 0.0,
            "bottom": 1.0,
        }[valign]

    # Define the base marker
    bm = markers.MarkerStyle(marker)

    # Get the marker path and apply the marker transform to get the
    # actual marker vertices (they should all be in a unit-square
    # centered at (0, 0))
    m_arr = bm.get_path().transformed(bm.get_transform()).vertices

    # Shift the marker vertices for the specified alignment.
    m_arr[:, 0] += halign / 2
    m_arr[:, 1] += valign / 2

    return Path(m_arr, bm.get_path().codes)
