"""Q-learning: off-policy TD."""

from agents.common import epsilon_greedy


def train_episode(Q, env, epsilon, rng, alpha=0.05, gamma=0.99):
    """Runs one episode and updates Q in place."""
    obs, _ = env.reset()
    done = False
    while not done:
        action = epsilon_greedy(Q, obs, epsilon, rng)
        next_obs, reward, _, done, _ = env.step(action)  # done = truncated
        target = reward + gamma * Q[next_obs].max()  # bootstrap even at truncation
        Q[obs][action] += alpha * (target - Q[obs][action])
        obs = next_obs
