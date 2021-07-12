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
