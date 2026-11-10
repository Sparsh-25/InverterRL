import numpy as np
import pytest

from agents import q_learning, sarsa
from agents.common import linear_epsilon, new_q_table
from agents.dqn import DQN, encode
from env import CHARGE, InverterEnv


def test_q_table_starts_at_value_of_always_holding():
    assert new_q_table() == pytest.approx(np.full((11, 24, 4, 3), -580))


def test_linear_epsilon():
    assert linear_epsilon(0, 100) == 1.0
    assert linear_epsilon(50, 100) == pytest.approx(0.525)
    assert linear_epsilon(500, 100) == pytest.approx(0.05)


@pytest.mark.parametrize("agent", [q_learning, sarsa])
def test_one_step_td_update(agent):
    # One step from battery 5 at 19:00 (peak), grid on (off = 0)
    env = InverterEnv(episode_hours=1)
    env.reset(seed=0)
    env.battery, env.hour, env.off = 5, 19, 0
    env.reset = lambda: (env._obs(), {})  # start from this state
    Q = new_q_table()

    agent.train_episode(Q, env, epsilon=0, rng=np.random.default_rng(0))

    # All Q equal, so greedy picks Charge: reward -14.4 (2 units at ₹7.20)
    # target = -14.4 + 0.99 * -580 = -588.6, new Q = -580 + 0.05 * (-588.6 + 580)
    assert Q[5, 19, 0, CHARGE] == pytest.approx(-580.43)


def test_dqn_encoding_puts_hour_23_next_to_hour_0():
    battery, sin_0, cos_0, off = encode((5, 0, 3))
    _, sin_23, cos_23, _ = encode((5, 23, 3))
    assert (battery, off) == (0.5, 1)
    assert abs(sin_0 - sin_23) < 0.3 and abs(cos_0 - cos_23) < 0.05


def test_dqn_trains_and_gives_a_policy_table():
    env = InverterEnv(episode_hours=100)  # more steps than one batch, so it learns
    env.reset(seed=0)
    agent = DQN(seed=0)
    agent.train_episode(env, epsilon=1.0, rng=np.random.default_rng(0))
    policy = agent.policy_table()
    assert agent.steps == 100
    assert policy.shape == (11, 24, 4) and set(policy.flat) <= {0, 1, 2}
