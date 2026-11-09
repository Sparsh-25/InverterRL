"""Evaluate a fixed policy (no exploration) on simulated days."""

from env import InverterEnv


def evaluate(policy, pattern="normal", penalty=50, days=2000, seed=0):
    """policy is an array indexed [battery, hour, grid].

    Returns per-day averages: (score, bill, blackout_hours).
    The env draws one random number per step whatever the action,
    so the same seed gives every policy the same power cuts.
    """
    env = InverterEnv(pattern=pattern, penalty=penalty, episode_hours=24 * days)
    obs, _ = env.reset(seed=seed)
    score = bill = blackouts = 0
    for _ in range(24 * days):
        obs, reward, _, _, info = env.step(policy[obs])
        score += reward
        bill += info["bill"]
        blackouts += info["blackout"]
    return score / days, bill / days, blackouts / days
