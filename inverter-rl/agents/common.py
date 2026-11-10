"""Helpers shared by the tabular agents."""

import numpy as np

from env import MAX_BATTERY, MAX_OFF, PRICE, period


def initial_q_value(gamma=0.99):
    """Value of always holding: the average price (₹5.80) every hour, -580 at gamma 0.99.

    Starting near the true scale makes learning much faster than starting at 0
    (see decisions.md).
    """
    average_price = np.mean([PRICE[period(hour)] for hour in range(24)])
    return -average_price / (1 - gamma)


def new_q_table(gamma=0.99):
    """Q[battery, hour, off, action], all set to initial_q_value."""
    return np.full((MAX_BATTERY + 1, 24, MAX_OFF + 1, 3), initial_q_value(gamma))


def epsilon_greedy(Q, obs, epsilon, rng):
    if rng.random() < epsilon:
        return int(rng.integers(3))
    return int(Q[obs].argmax())


def linear_epsilon(episode, decay_episodes, start=1.0, end=0.05):
    """Goes from start to end over decay_episodes, then stays at end."""
    fraction = min(episode / decay_episodes, 1)
    return start + fraction * (end - start)
