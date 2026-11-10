"""Home inverter battery environment. See project.md, section 3, and decisions.md.

Prices: national ToD rule with Karnataka's evening peak, normal rate ₹6.
Power cuts: estimated by estimate_cuts.py from ESMI 2018 data for Guddiangadi.
"""

import gymnasium as gym
from gymnasium import spaces

MAX_BATTERY = 10
MAX_OFF = 3  # grid status: 0 = on, 1/2/3 = off for 1, 2, or 3+ hours
CHARGE, HOLD, DISCHARGE = 0, 1, 2

PRICE = {"solar": 4.8, "normal": 6.0, "peak": 7.2}

# Chance a cut starts next hour, if the grid is on now
CUT_PROB = {
    "normal": {"solar": 0.026, "normal": 0.018, "peak": 0.013},  # Jan-Mar, Jul-Dec
    "summer": {"solar": 0.064, "normal": 0.035, "peak": 0.029},  # Apr-Jun
}
# Chance the grid is back next hour, after 1, 2, or 3+ hours off
RESTORE_PROB = {
    "normal": [0.573, 0.415, 0.258],
    "summer": [0.558, 0.447, 0.206],
}


def period(hour):
    if 18 <= hour <= 21:
        return "peak"
    if 10 <= hour <= 17:
        return "solar"
    return "normal"


def battery_step(battery, off, action):
    """Returns (new_battery, units_bought, blackout)."""
    if off:
        if battery == 0:
            return 0, 0, True
        return battery - 1, 0, False
    if action == CHARGE and battery < MAX_BATTERY:
        return battery + 1, 2, False
    if action == DISCHARGE and battery > 0:
        return battery - 1, 0, False
    return battery, 1, False  # hold (also charge at full, discharge at empty)


def change_prob(hour, off, pattern):
    """Chance the grid status flips next hour (a cut starts, or power comes back)."""
    if off == 0:
        return CUT_PROB[pattern][period(hour)]
    return RESTORE_PROB[pattern][off - 1]


def next_off(off, changed):
    """Next grid status after a flip (changed=True) or no flip."""
    if off == 0:
        return 1 if changed else 0
    return 0 if changed else min(off + 1, MAX_OFF)


class InverterEnv(gym.Env):
    """Observation is (battery, hour, off)."""

    def __init__(self, pattern="normal", penalty=50, episode_hours=168):
        self.pattern = pattern  # can be changed mid-training to "summer"
        self.penalty = penalty
        self.episode_hours = episode_hours
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.MultiDiscrete([MAX_BATTERY + 1, 24, MAX_OFF + 1])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.battery = int(self.np_random.integers(MAX_BATTERY + 1))
        self.hour = int(self.np_random.integers(24))
        self.off = int(self.np_random.integers(MAX_OFF + 1))
        self.t = 0
        return self._obs(), {}

    def step(self, action):
        self.battery, units, blackout = battery_step(self.battery, self.off, action)
        bill = PRICE[period(self.hour)] * units
        reward = -bill - self.penalty * blackout

        changed = self.np_random.random() < change_prob(self.hour, self.off, self.pattern)
        self.off = next_off(self.off, changed)
        self.hour = (self.hour + 1) % 24
        self.t += 1

        truncated = self.t >= self.episode_hours  # continuing task, never terminated
        return self._obs(), reward, False, truncated, {"bill": bill, "blackout": blackout}

    def _obs(self):
        return (self.battery, self.hour, self.off)
