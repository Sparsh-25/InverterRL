"""DQN: Q-learning with a neural network, a replay buffer and a target network."""

import copy
import math

import numpy as np
import torch
from torch import nn

from agents.common import initial_q_value
from env import MAX_BATTERY, MAX_OFF

REWARD_SCALE = 100  # the network learns Q / 100, so values are about -6 instead of -600


def encode(obs):
    """[battery/10, sin(hour), cos(hour), off/3]; sin/cos put hour 23 next to hour 0."""
    battery, hour, off = obs
    angle = 2 * math.pi * hour / 24
    return [battery / MAX_BATTERY, math.sin(angle), math.cos(angle), off / MAX_OFF]


class DQN:
    def __init__(self, seed=0, gamma=0.99, lr=1e-3, buffer_size=50_000,
                 batch_size=64, target_update=500):
        torch.manual_seed(seed)
        self.net = nn.Sequential(nn.Linear(4, 64), nn.ReLU(),
                                 nn.Linear(64, 64), nn.ReLU(),
                                 nn.Linear(64, 3))
        nn.init.constant_(self.net[-1].bias, initial_q_value(gamma) / REWARD_SCALE)  # like the Q-table start
        self.target_net = copy.deepcopy(self.net)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.gamma, self.batch_size, self.target_update = gamma, batch_size, target_update

        # Replay buffer: the oldest transitions are overwritten when it is full
        self.states = np.zeros((buffer_size, 4), dtype=np.float32)
        self.actions = np.zeros(buffer_size, dtype=np.int64)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.next_states = np.zeros((buffer_size, 4), dtype=np.float32)
        self.steps = 0  # total environment steps so far

    def act(self, obs, epsilon, rng):
        if rng.random() < epsilon:
            return int(rng.integers(3))
        with torch.no_grad():
            return int(self.net(torch.tensor(encode(obs))).argmax())

    def train_episode(self, env, epsilon, rng):
        obs, _ = env.reset()
        done = False
        while not done:
            action = self.act(obs, epsilon, rng)
            next_obs, reward, _, done, _ = env.step(action)  # done = truncated
            self.remember(obs, action, reward, next_obs)
            self.learn(rng)
            obs = next_obs

    def remember(self, obs, action, reward, next_obs):
        i = self.steps % len(self.actions)
        self.states[i], self.actions[i] = encode(obs), action
        self.rewards[i], self.next_states[i] = reward / REWARD_SCALE, encode(next_obs)
        self.steps += 1

    def learn(self, rng):
        size = min(self.steps, len(self.actions))
        if size < self.batch_size:
            return
        i = rng.integers(size, size=self.batch_size)
        states, next_states = torch.from_numpy(self.states[i]), torch.from_numpy(self.next_states[i])
        actions, rewards = torch.from_numpy(self.actions[i]), torch.from_numpy(self.rewards[i])

        with torch.no_grad():  # bootstrap even at truncation, like tabular Q-learning
            target = rewards + self.gamma * self.target_net(next_states).max(dim=1).values
        q = self.net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = nn.functional.smooth_l1_loss(q, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.net.state_dict())

    def policy_table(self):
        """Greedy action for every state, shaped [battery, hour, off] like the DP policy."""
        obs = [(b, h, o) for b in range(MAX_BATTERY + 1) for h in range(24) for o in range(MAX_OFF + 1)]
        with torch.no_grad():
            actions = self.net(torch.tensor([encode(o) for o in obs])).argmax(dim=1)
        return actions.numpy().reshape(MAX_BATTERY + 1, 24, MAX_OFF + 1)
