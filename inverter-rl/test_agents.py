import numpy as np
import pytest

from agents import q_learning, sarsa
from agents.common import linear_epsilon, new_q_table
from env import CHARGE, InverterEnv


def test_q_table_starts_at_value_of_always_holding():
    assert new_q_table() == pytest.approx(np.full((11, 24, 2, 3), -600))


def test_linear_epsilon():
    assert linear_epsilon(0, 100) == 1.0
    assert linear_epsilon(50, 100) == pytest.approx(0.525)
    assert linear_epsilon(500, 100) == pytest.approx(0.05)


@pytest.mark.parametrize("agent", [q_learning, sarsa])
def test_one_step_td_update(agent):
    # One step from battery 5 at 19:00 (peak), grid on
    env = InverterEnv(episode_hours=1)
    env.reset(seed=0)
    env.battery, env.hour, env.grid = 5, 19, 1
    env.reset = lambda: (env._obs(), {})  # start from this state
    Q = new_q_table()

    agent.train_episode(Q, env, epsilon=0, rng=np.random.default_rng(0))

    # All Q equal, so greedy picks Charge: reward -20 (2 units at ₹10)
    # target = -20 + 0.99 * -600 = -614, new Q = -600 + 0.1 * (-614 + 600)
    assert Q[5, 19, 1, CHARGE] == pytest.approx(-601.4)
