from dp import value_iteration
from env import InverterEnv


def average_daily_score(policy, pattern, days=5000):
    env = InverterEnv(pattern=pattern, episode_hours=24 * days)
    obs, _ = env.reset(seed=0)
    total = 0
    for _ in range(24 * days):
        obs, reward, *_ = env.step(policy[obs])
        total += reward
    return total / days


def test_dp_matches_project_numbers():
    # Long-run daily scores from project.md, section 7
    _, normal = value_iteration("normal")
    _, summer = value_iteration("summer")
    assert abs(average_daily_score(normal, "normal") - (-119.3)) < 1
    assert abs(average_daily_score(summer, "summer") - (-121.5)) < 1
    assert abs(average_daily_score(normal, "summer") - (-126.2)) < 1
