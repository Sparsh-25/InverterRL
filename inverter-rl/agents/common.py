"""Helpers shared by the tabular agents."""

import numpy as np

from env import MAX_BATTERY, PRICE, period


def new_q_table(gamma=0.99):
    """Q[battery, hour, grid, action], all set to the value of always holding.

    Always holding costs the average price (₹6) every hour, worth 6 / (1 - gamma)
    = 600 at gamma 0.99. Starting near the true scale makes learning much faster
    than starting at 0 (see decisions.md).
    """
    average_price = np.mean([PRICE[period(hour)] for hour in range(24)])
    return np.full((MAX_BATTERY + 1, 24, 2, 3), -average_price / (1 - gamma))


def epsilon_greedy(Q, obs, epsilon, rng):
    if rng.random() < epsilon:
        return int(rng.integers(3))
    return int(Q[obs].argmax())


def linear_epsilon(episode, decay_episodes, start=1.0, end=0.05):
    """Goes from start to end over decay_episodes, then stays at end."""
    fraction = min(episode / decay_episodes, 1)
    return start + fraction * (end - start)
