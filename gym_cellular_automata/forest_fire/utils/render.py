from svgpath2mpl import parse_path

from gym_cellular_automata import PROJECT_PATH


def get_font(ttfpath):
    from pathlib import Path

    import matplotlib as mpl

    return Path(mpl.get_data_path(), ttfpath)


EMOJIFONT = get_font(PROJECT_PATH / "fonts/OpenMoji-Black.ttf")
TITLEFONT = get_font(PROJECT_PATH / "fonts/FrederickatheGreat-Regular.ttf")


def parse_svg_into_mpl(svg_path):

    mpl_path = parse_path(svg_path)

    def center(mpl_path):
        mpl_path.vertices -= mpl_path.vertices.mean(axis=0)
        return mpl_path

    def upsidedown(mpl_path):
        mpl_path.vertices[:, 1] *= -1
        return mpl_path

    return upsidedown(center(mpl_path))
