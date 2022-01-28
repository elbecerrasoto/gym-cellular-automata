from gym.envs.registration import register

from gym_cellular_automata.forest_fire.bulldozer import ForestFireBulldozerEnv
from gym_cellular_automata.forest_fire.helicopter import ForestFireHelicopterEnv

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


def register_caenvs():
    for ca_env in REGISTERED_CA_ENVS:
        register(
            ca_env,
            kwargs=REGISTERED_CA_ENVS[ca_env]["kwargs"],
            entry_point=REGISTERED_CA_ENVS[ca_env]["entry_point"],
        )
