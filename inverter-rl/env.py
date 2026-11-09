"""Home inverter battery environment. See project.md, section 3."""

import gymnasium as gym
from gymnasium import spaces

MAX_BATTERY = 10
CHARGE, HOLD, DISCHARGE = 0, 1, 2

PRICE = {"offpeak": 4, "normal": 6, "peak": 10}
CUT_PROB = {
    "normal": {"offpeak": 0.02, "normal": 0.05, "peak": 0.10},
    "summer": {"offpeak": 0.04, "normal": 0.10, "peak": 0.20},
}
RESTORE_PROB = 0.5


def period(hour):
    if 18 <= hour <= 21:
        return "peak"
    if 6 <= hour <= 17:
        return "normal"
    return "offpeak"


def battery_step(battery, grid_on, action):
    """Returns (new_battery, units_bought, blackout)."""
    if not grid_on:
        if battery == 0:
            return 0, 0, True
        return battery - 1, 0, False
    if action == CHARGE and battery < MAX_BATTERY:
        return battery + 1, 2, False
    if action == DISCHARGE and battery > 0:
        return battery - 1, 0, False
    return battery, 1, False  # hold (also charge at full, discharge at empty)


def grid_on_prob(hour, grid_on, pattern):
    """Probability that the grid is on next hour."""
    if grid_on:
        return 1 - CUT_PROB[pattern][period(hour)]
    return RESTORE_PROB


class InverterEnv(gym.Env):
    """Observation is (battery, hour, grid), with grid 1 = on, 0 = off."""

    def __init__(self, pattern="normal", penalty=50, episode_hours=168):
        self.pattern = pattern  # can be changed mid-training to "summer"
        self.penalty = penalty
        self.episode_hours = episode_hours
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.MultiDiscrete([MAX_BATTERY + 1, 24, 2])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.battery = int(self.np_random.integers(MAX_BATTERY + 1))
        self.hour = int(self.np_random.integers(24))
        self.grid = int(self.np_random.integers(2))
        self.t = 0
        return self._obs(), {}

    def step(self, action):
        self.battery, units, blackout = battery_step(self.battery, self.grid, action)
        bill = PRICE[period(self.hour)] * units
        reward = -bill - self.penalty * blackout

        p_on = grid_on_prob(self.hour, self.grid, self.pattern)
        self.grid = int(self.np_random.random() < p_on)
        self.hour = (self.hour + 1) % 24
        self.t += 1

        truncated = self.t >= self.episode_hours  # continuing task, never terminated
        return self._obs(), reward, False, truncated, {"bill": bill, "blackout": blackout}

    def _obs(self):
        return (self.battery, self.hour, self.grid)
