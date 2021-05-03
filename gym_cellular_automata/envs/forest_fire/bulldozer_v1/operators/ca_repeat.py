def _get_operation_flow(self, action):
    import math

    movement, shooting = action

    # Mapping of actions ---> to time (on units of CA updates)
    time_move = self._movement_timings[movement]
    time_shoot = self._shooting_timings[shooting]
    time_environment = self._t_env_any

    # The time taken on a step is the time taken doing the actions
    # plus some enviromental (internal) time.
    time_taken = time_move + time_shoot + time_environment

    self.accumulated_time += time_taken

    # Decimal and Integer parts
    self.accumulated_time, ca_repeats = math.modf(self.accumulated_time)

    ica, imove, imodify = range(3)

    # Operartors order is: CA, Modifier
    operation_flow = int(ca_repeats) * [ica] + [imove] + [imodify]

    return operation_flow
