"""Monte Carlo control: every-visit, constant step size.

Returns are cut off where the episode ends. With gamma 0.99 that cut-off
distorts the values badly (see decisions.md), so each episode runs extra
hours past the 7 days and only the first 7 days are learned from.
"""

from agents.common import epsilon_greedy

UPDATE_HOURS = 168  # learn from the first 7 days
TAIL_HOURS = 460  # extra hours so every return is nearly complete (0.99 ** 460 ≈ 0.01)


def train_episode(Q, env, epsilon, rng, alpha=0.01, gamma=0.99):
    """Runs one episode, then updates Q in place from the returns."""
    obs, _ = env.reset()
    steps = []
    for _ in range(UPDATE_HOURS + TAIL_HOURS):  # runs past the env's 168-hour limit on purpose
        action = epsilon_greedy(Q, obs, epsilon, rng)
        next_obs, reward, *_ = env.step(action)
        steps.append((obs, action, reward))
        obs = next_obs

    G = 0  # return
    for t in reversed(range(len(steps))):
        obs, action, reward = steps[t]
        G = reward + gamma * G
        if t < UPDATE_HOURS:
            Q[obs][action] += alpha * (G - Q[obs][action])
