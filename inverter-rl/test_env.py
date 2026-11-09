from env import CHARGE, DISCHARGE, HOLD, InverterEnv, battery_step, grid_on_prob, period


def make_env(battery, hour, grid):
    env = InverterEnv()
    env.reset(seed=0)
    env.battery, env.hour, env.grid = battery, hour, grid
    return env


def test_actions_with_grid_on():
    assert battery_step(5, True, CHARGE) == (6, 2, False)
    assert battery_step(5, True, HOLD) == (5, 1, False)
    assert battery_step(5, True, DISCHARGE) == (4, 0, False)


def test_battery_limits_act_as_hold():
    assert battery_step(10, True, CHARGE) == (10, 1, False)
    assert battery_step(0, True, DISCHARGE) == (0, 1, False)


def test_grid_off_ignores_action():
    for action in (CHARGE, HOLD, DISCHARGE):
        assert battery_step(5, False, action) == (4, 0, False)


def test_blackout_when_grid_off_and_battery_empty():
    assert battery_step(0, False, CHARGE) == (0, 0, True)


def test_periods():
    assert [period(h) for h in (5, 6, 17, 18, 21, 22)] == [
        "offpeak", "normal", "normal", "peak", "peak", "offpeak"]


def test_grid_probabilities():
    assert grid_on_prob(19, True, "normal") == 0.90
    assert grid_on_prob(19, True, "summer") == 0.80
    assert grid_on_prob(19, False, "normal") == 0.5


def test_reward_uses_price_of_current_hour():
    _, reward, *_ = make_env(5, 19, 1).step(CHARGE)  # peak, 2 units
    assert reward == -20
    _, reward, *_ = make_env(5, 2, 1).step(HOLD)  # off-peak, 1 unit
    assert reward == -4


def test_blackout_penalty():
    env = make_env(0, 12, 0)
    env.penalty = 100
    _, reward, _, _, info = env.step(HOLD)
    assert reward == -100 and info["blackout"]


def test_hour_wraps():
    (_, hour, _), *_ = make_env(5, 23, 1).step(HOLD)
    assert hour == 0


def test_episode_truncates_after_168_steps():
    env = InverterEnv()
    env.reset(seed=1)
    for t in range(168):
        _, _, terminated, truncated, _ = env.step(HOLD)
        assert not terminated
        assert truncated == (t == 167)
