from gymnasium import spaces

from gym_cellular_automata._config import TYPE_INT
from gym_cellular_automata.ca_env import CAEnv
from gym_cellular_automata.grid_space import GridSpace
from gym_cellular_automata.tests import Identity


class MockCAEnv(CAEnv):
    _nrows = 8
    _ncols = 8
    _states = 8

    def __init__(self, nrows=_nrows, ncols=_ncols, **kwargs):
        super().__init__(nrows, ncols, **kwargs)

        self._set_spaces()

        # Composite Operators
        self._MDP = Identity(**self.MDP_space)

    @property
    def MDP(self):
        return self._MDP

    @property
    def initial_state(self):
        if self._resample_initial:
            self.grid = self.grid_space.sample()
            self.context = self.context_space.sample()

            self._initial_state = self.grid, self.context
            self._resample_initial = False

        return self._initial_state

    def _award(self):
        return 0.0

    def _is_done(self):
        return False

    def _report(self):
        return {}

    def _set_spaces(self):
        self.grid_space = GridSpace(
            n=self._states, shape=(self._nrows, self._ncols), dtype=TYPE_INT
        )
        self.action_space = spaces.Discrete(self._states)
        self.context_space = spaces.Discrete(self._states)

        self.observation_space = spaces.Tuple((self.grid_space, self.context_space))

        self.MDP_space = {
            "grid_space": self.grid_space,
            "action_space": self.action_space,
            "context_space": self.context_space,
        }

    def _get_defaults_free(self, **kwargs):
        """
        place holder
        """
        return {}

    def _get_defaults_scale(self, nrows, ncols):
        """
        place holder
        """
        return {}
