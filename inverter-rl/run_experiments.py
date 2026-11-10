"""Experiments 1 to 3 (project.md, section 5, and decisions.md).

Every job is one method, penalty and seed. Each job trains two runs:
- ε decaying to 0.05 on the normal pattern, then split three ways:
  keep the normal pattern (Experiment 1), switch to summer (Experiment 2, "decayed"),
  switch to summer with ε reset to 0.5 (Experiment 2, "reset")
- ε constant at 0.10 for the whole run, normal then summer (Experiment 2, "constant")
Penalties 20 and 100 repeat only Experiment 1 (Experiment 3), to save time.
Regret curves and final policies go to results/<method>_penalty<penalty>_seed<seed>.npz.
"""

import copy
import os
from multiprocessing import Pool

import numpy as np
import torch

from agents import monte_carlo, q_learning, sarsa
from agents.common import linear_epsilon, new_q_table
from agents.dqn import DQN
from dp import value_iteration
from env import InverterEnv
from evaluate import evaluate

TABULAR = {"q_learning": q_learning, "sarsa": sarsa, "monte_carlo": monte_carlo}
METHODS = ["dqn", "monte_carlo", "q_learning", "sarsa"]  # slowest first
PENALTIES = [50, 20, 100]
SEEDS = range(5)  # 10 would take about 4 hours
# Episodes before (and after) the change, ε decay length, episodes between checkpoints
BUDGET = {"tabular": (50_000, 40_000, 2_000), "dqn": (2_000, 1_000, 100)}


def run(job):
    method, penalty, seed = job
    path = f"results/{method}_penalty{penalty}_seed{seed}.npz"
    if os.path.exists(path):  # lets a stopped run carry on
        return
    torch.set_num_threads(1)  # one core per job
    n, decay, every = BUDGET["dqn" if method == "dqn" else "tabular"]
    eval_seed = 1000 + seed
    best = {p: evaluate(value_iteration(p, penalty)[1], p, penalty, seed=eval_seed)[0]
            for p in ("normal", "summer")}

    def policy(agent):
        return agent.policy_table() if method == "dqn" else agent.argmax(axis=3)

    def start():
        env = InverterEnv(penalty=penalty)
        env.reset(seed=seed)
        agent = DQN(seed=seed) if method == "dqn" else new_q_table()
        return agent, env, np.random.default_rng(seed)

    def train(agent, env, rng, pattern, epsilon):
        """Trains n episodes; returns the regret % against DP at every checkpoint."""
        env.pattern = pattern
        curve = []
        for ep in range(n):
            if method == "dqn":
                agent.train_episode(env, epsilon(ep), rng)
            else:
                TABULAR[method].train_episode(agent, env, epsilon(ep), rng)
            if (ep + 1) % every == 0:
                score = evaluate(policy(agent), pattern, penalty, seed=eval_seed)[0]
                curve.append((best[pattern] - score) / -best[pattern] * 100)
        return curve

    results = {}
    decayed = start()
    results["before"] = train(*decayed, "normal", lambda ep: linear_epsilon(ep, decay))
    branches = [("normal", "normal", lambda ep: 0.05),
                ("decayed", "summer", lambda ep: 0.05),
                ("reset", "summer", lambda ep: linear_epsilon(ep, 0.2 * n, start=0.5))]
    if penalty != 50:  # Experiment 3 only repeats Experiment 1
        branches = branches[:1]
    for name, pattern, epsilon in branches:
        branch = copy.deepcopy(decayed)  # same agent, env and random state at the change
        results[name] = train(*branch, pattern, epsilon)
        results[name + "_policy"] = policy(branch[0])

    if penalty != 50:
        np.savez(path, **results)
        return
    constant = start()
    results["constant_before"] = train(*constant, "normal", lambda ep: 0.1)
    results["constant"] = train(*constant, "summer", lambda ep: 0.1)
    results["constant_policy"] = policy(constant[0])

    np.savez(path, **results)
    print("done", path, flush=True)


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    jobs = [(m, p, s) for m in METHODS for p in PENALTIES for s in SEEDS]  # longest jobs first
    with Pool() as pool:
        pool.map(run, jobs, chunksize=1)
