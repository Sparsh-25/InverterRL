"""Hand-written rule, used as a sanity check."""

import numpy as np

from env import CHARGE, DISCHARGE, HOLD, MAX_BATTERY, MAX_OFF, period


def rule_action(battery, hour):
    if period(hour) == "solar":  # cheapest hours
        return CHARGE
    if period(hour) == "peak":
        return DISCHARGE if battery > 3 else HOLD
    return HOLD


def rule_policy():
    """Same shape as the DP policy: [battery, hour, off]."""
    policy = np.zeros((MAX_BATTERY + 1, 24, MAX_OFF + 1), dtype=int)
    for battery in range(MAX_BATTERY + 1):
        for hour in range(24):
            policy[battery, hour, :] = rule_action(battery, hour)
    return policy
