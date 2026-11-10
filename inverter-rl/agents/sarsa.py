"""SARSA: on-policy TD."""

from agents.common import epsilon_greedy


def train_episode(Q, env, epsilon, rng, alpha=0.05, gamma=0.99):
    """Runs one episode and updates Q in place."""
    obs, _ = env.reset()
    action = epsilon_greedy(Q, obs, epsilon, rng)
    done = False
    while not done:
        next_obs, reward, _, done, _ = env.step(action)  # done = truncated
        next_action = epsilon_greedy(Q, next_obs, epsilon, rng)
        target = reward + gamma * Q[next_obs][next_action]  # bootstrap even at truncation
        Q[obs][action] += alpha * (target - Q[obs][action])
        obs, action = next_obs, next_action
