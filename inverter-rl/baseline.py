"""Hand-written rule, used as a sanity check."""

import numpy as np

from env import CHARGE, DISCHARGE, HOLD, MAX_BATTERY


def rule_action(battery, hour):
    if hour >= 22 or hour < 6:  # off-peak
        return CHARGE
    if 18 <= hour <= 21:  # peak
        return DISCHARGE if battery > 3 else HOLD
    return HOLD


def rule_policy():
    """Same shape as the DP policy: [battery, hour, grid]."""
    policy = np.zeros((MAX_BATTERY + 1, 24, 2), dtype=int)
    for battery in range(MAX_BATTERY + 1):
        for hour in range(24):
            policy[battery, hour, :] = rule_action(battery, hour)
    return policy
