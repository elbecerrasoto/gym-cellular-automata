import pytest


@pytest.mark.skip(reason="Working on other stuff")
def test_gym_if_done_behave_gracefully(env, grid_space, context_space):
    env.reset()
    env.done = True

    action = env.action_space.sample()
    with pytest.warns(UserWarning):
        obs, reward, done, info = env.step(action)

    assert done
    assert reward == 0.0


@pytest.mark.skip(reason="Working on other stuff")
def test_counts(env):
    obs = env.reset()
    grid, context = obs

    def get_dict_of_counts(grid):
        values, counts = np.unique(grid, return_counts=True)
        return dict(zip(values, counts))

    observed_counts = env.count_cells(grid)
    expected_counts = get_dict_of_counts(grid)

    assert all(
        [observed_counts[cell] == expected_counts[cell] for cell in expected_counts]
    )
