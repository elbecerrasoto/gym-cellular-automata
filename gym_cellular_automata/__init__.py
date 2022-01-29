"""
Gym Cellular Automata Enviroments
==================================

Gym Cellular Automata is a collection of
Reinforcement Learning Environments (RLEs)
that follow the OpenAI Gym API.

The available RLEs are based on Cellular Automata (CAs).
On them an Agent interacts with a CA,
by changing its cell states,
in a attempt to drive the emergent properties
of its grid.

```python
import gym
import gym_cellular_automata as gymca

# benchmark mode
env_id = gymca.envs[0]
env = gym.make(env_id)

# prototype mode
ProtoEnv = gymca.prototypes[0]
env = ProtoEnv(nrows=42, ncols=42)
```

See https://github.com/elbecerrasoto/gym-cellular-automata for documentation.
"""
from gym.error import Error as GymError

from gym_cellular_automata.ca_env import CAEnv
from gym_cellular_automata.grid_space import GridSpace
from gym_cellular_automata.operator import Operator
from gym_cellular_automata.registration import GYM_MAKE as envs
from gym_cellular_automata.registration import _register_caenvs, prototypes
from gym_cellular_automata.version import VERSION as __version__

# Exports for user code
# do NOT import from here into gymca
# as it would trigger a circular ImportError

# Avoids annoying error when working interactively
try:
    _register_caenvs()
except GymError:
    pass


__all__ = ["envs", "prototypes", "CAEnv", "GridSpace", "Operator"]
