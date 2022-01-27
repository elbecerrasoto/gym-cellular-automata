from gym.envs.registration import register

FFDIR = "gym_cellular_automata.forest_fire"

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


LIBRARY = "gym_cellular_automata"
GYM_MAKE = [LIBRARY + ":" + ca_env for ca_env in REGISTERED_CA_ENVS]


def register_caenvs():
    for ca_env in REGISTERED_CA_ENVS:
        register(
            ca_env,
            kwargs=REGISTERED_CA_ENVS[ca_env]["kwargs"],
            entry_point=REGISTERED_CA_ENVS[ca_env]["entry_point"],
        )
