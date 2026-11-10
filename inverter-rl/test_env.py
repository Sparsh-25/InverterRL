import pytest

from env import CHARGE, DISCHARGE, HOLD, InverterEnv, battery_step, change_prob, next_off, period


def make_env(battery, hour, off):
    env = InverterEnv()
    env.reset(seed=0)
    env.battery, env.hour, env.off = battery, hour, off
    return env


def test_actions_with_grid_on():
    assert battery_step(5, 0, CHARGE) == (6, 2, False)
    assert battery_step(5, 0, HOLD) == (5, 1, False)
    assert battery_step(5, 0, DISCHARGE) == (4, 0, False)


def test_battery_limits_act_as_hold():
    assert battery_step(10, 0, CHARGE) == (10, 1, False)
    assert battery_step(0, 0, DISCHARGE) == (0, 1, False)


def test_grid_off_ignores_action():
    for off in (1, 2, 3):
        for action in (CHARGE, HOLD, DISCHARGE):
            assert battery_step(5, off, action) == (4, 0, False)


def test_blackout_when_grid_off_and_battery_empty():
    assert battery_step(0, 1, CHARGE) == (0, 0, True)


def test_periods():
    assert [period(h) for h in (9, 10, 17, 18, 21, 22)] == [
        "normal", "solar", "solar", "peak", "peak", "normal"]


def test_grid_probabilities():
    assert change_prob(19, 0, "normal") == 0.013  # cut starts at peak
    assert change_prob(12, 0, "summer") == 0.064  # cut starts in solar hours
    assert change_prob(19, 1, "normal") == 0.573  # back after 1 hour off
    assert change_prob(19, 3, "summer") == 0.206  # back after 3+ hours off


def test_hours_off_count_up_to_3():
    assert [next_off(0, True), next_off(0, False)] == [1, 0]
    assert [next_off(1, False), next_off(2, False), next_off(3, False)] == [2, 3, 3]
    assert next_off(3, True) == 0


def test_reward_uses_price_of_current_hour():
    _, reward, *_ = make_env(5, 19, 0).step(CHARGE)  # peak, 2 units
    assert reward == pytest.approx(-14.4)
    _, reward, *_ = make_env(5, 12, 0).step(HOLD)  # solar hours, 1 unit
    assert reward == pytest.approx(-4.8)


def test_blackout_penalty():
    env = make_env(0, 12, 2)
    env.penalty = 100
    _, reward, _, _, info = env.step(HOLD)
    assert reward == -100 and info["blackout"]


def test_hour_wraps():
    (_, hour, _), *_ = make_env(5, 23, 0).step(HOLD)
    assert hour == 0


def test_episode_truncates_after_168_steps():
    env = InverterEnv()
    env.reset(seed=1)
    for t in range(168):
        _, _, terminated, truncated, _ = env.step(HOLD)
        assert not terminated
        assert truncated == (t == 167)
