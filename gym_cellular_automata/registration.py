from gym.envs.registration import register

FFDIR = "gym_cellular_automata.forest_fire"

REGISTERED_CA_ENVS = {
    "ForestFireHelicopter5x5-v1": {
        "kwargs": {"nrows": 5, "ncols": 5},
        "entry_point": FFDIR + ".helicopter:ForestFireEnvHelicopterV0",
    },
    "ForestFireBulldozer256x256-v2": {
        "kwargs": {"nrows": 256, "ncols": 256},
        "entry_point": FFDIR + ".bulldozer:ForestFireEnvBulldozerV1",
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
