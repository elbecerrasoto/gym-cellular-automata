from typing import Any

from gymnasium.envs.registration import register
from gymnasium.spaces import flatten
from numpy.typing import NDArray

from gym_cellular_automata.forest_fire.bulldozer import ForestFireBulldozerEnv
from gym_cellular_automata.forest_fire.helicopter import ForestFireHelicopterEnv
from gym_cellular_automata.grid_space import GridSpace

FFDIR = "gym_cellular_automata.forest_fire"


LIBRARY = "gym_cellular_automata"

prototypes = (ForestFireHelicopterEnv, ForestFireBulldozerEnv)


HELR, HELC = 5, 5
BULR, BULC = 256, 256

REGISTERED_CA_ENVS = {
    "ForestFireHelicopter"
    + str(HELR)
    + "x"
    + str(HELC)
    + "-v1": {
        "kwargs": {"nrows": HELR, "ncols": HELC},
        "entry_point": FFDIR + ".helicopter:ForestFireHelicopterEnv",
    },
    "ForestFireBulldozer"
    + str(BULR)
    + "x"
    + str(BULC)
    + "-v3": {
        "kwargs": {"nrows": BULR, "ncols": BULC},
        "entry_point": FFDIR + ".bulldozer:ForestFireBulldozerEnv",
    },
}

GYM_MAKE = tuple(LIBRARY + ":" + ca_env for ca_env in REGISTERED_CA_ENVS)


def _register_caenvs():
    for ca_env in REGISTERED_CA_ENVS:
        register(
            ca_env,
            kwargs=REGISTERED_CA_ENVS[ca_env]["kwargs"],
            entry_point=REGISTERED_CA_ENVS[ca_env]["entry_point"],
        )


@flatten.register(GridSpace)
def _flatten_grid_space(space: GridSpace, x: NDArray[Any]) -> NDArray[Any]:
    return np.asarray(x, dtype=space.dtype).flatten()
