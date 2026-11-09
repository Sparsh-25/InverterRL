"""Value iteration: the exact optimal policy, used as the benchmark."""

import numpy as np

from env import MAX_BATTERY, PRICE, battery_step, grid_on_prob, period


def value_iteration(pattern="normal", penalty=50, gamma=0.99, tol=1e-6):
    """Returns V and policy, both indexed as [battery, hour, grid]."""
    V = np.zeros((MAX_BATTERY + 1, 24, 2))
    while True:
        Q = q_values(V, pattern, penalty, gamma)
        V_new = Q.max(axis=3)
        change = np.abs(V_new - V).max()
        V = V_new
        if change < tol:
            return V, Q.argmax(axis=3)


def q_values(V, pattern, penalty, gamma):
    Q = np.zeros((MAX_BATTERY + 1, 24, 2, 3))
    for battery in range(MAX_BATTERY + 1):
        for hour in range(24):
            for grid in (0, 1):
                p_on = grid_on_prob(hour, grid, pattern)
                next_hour = (hour + 1) % 24
                for action in range(3):
                    new_battery, units, blackout = battery_step(battery, grid, action)
                    reward = -PRICE[period(hour)] * units - penalty * blackout
                    next_value = (p_on * V[new_battery, next_hour, 1]
                                  + (1 - p_on) * V[new_battery, next_hour, 0])
                    Q[battery, hour, grid, action] = reward + gamma * next_value
    return Q
