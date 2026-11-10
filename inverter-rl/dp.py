"""Value iteration: the exact optimal policy, used as the benchmark."""

import numpy as np

from env import MAX_BATTERY, MAX_OFF, PRICE, battery_step, change_prob, next_off, period


def value_iteration(pattern="normal", penalty=50, gamma=0.99, tol=1e-6):
    """Returns V and policy, both indexed as [battery, hour, off]."""
    V = np.zeros((MAX_BATTERY + 1, 24, MAX_OFF + 1))
    while True:
        Q = q_values(V, pattern, penalty, gamma)
        V_new = Q.max(axis=3)
        change = np.abs(V_new - V).max()
        V = V_new
        if change < tol:
            return V, Q.argmax(axis=3)


def q_values(V, pattern, penalty, gamma):
    Q = np.zeros((MAX_BATTERY + 1, 24, MAX_OFF + 1, 3))
    for battery in range(MAX_BATTERY + 1):
        for hour in range(24):
            for off in range(MAX_OFF + 1):
                p = change_prob(hour, off, pattern)
                next_hour = (hour + 1) % 24
                for action in range(3):
                    new_battery, units, blackout = battery_step(battery, off, action)
                    reward = -PRICE[period(hour)] * units - penalty * blackout
                    next_value = (p * V[new_battery, next_hour, next_off(off, True)]
                                  + (1 - p) * V[new_battery, next_hour, next_off(off, False)])
                    Q[battery, hour, off, action] = reward + gamma * next_value
    return Q
