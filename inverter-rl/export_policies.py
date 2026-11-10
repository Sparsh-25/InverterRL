"""Writes ../simulator/policies.json for the browser simulator.

Each policy is a string of 1,056 digits (0 Charge, 1 Hold, 2 Discharge), indexed battery * 96 + hour * 4 + off.
Learned policies are the median seed by final regret, as in Fig. 4 of the report.
"""

import json
import os

import numpy as np

from baseline import rule_policy
from dp import value_iteration
from env import CUT_PROB, PRICE, RESTORE_PROB
from run_experiments import PENALTIES, SEEDS

METHODS = ["sarsa", "q_learning", "monte_carlo", "dqn"]


def flat(policy):
    return "".join(str(int(a)) for a in np.asarray(policy).reshape(-1))


def median_run(method, penalty):
    runs = [np.load(f"results/{method}_penalty{penalty}_seed{s}.npz") for s in SEEDS]
    final = [np.concatenate([r["before"], r["normal"]])[-1] for r in runs]
    seed = int(np.argsort(final)[len(final) // 2])
    return seed, runs[seed]


def main():
    out = {"price": PRICE, "cut": CUT_PROB, "restore": RESTORE_PROB, "rule": flat(rule_policy()),
           "dp": {}, "learned": {}}
    for k in PENALTIES:
        out["dp"][k] = {p: flat(value_iteration(p, k)[1]) for p in ("normal", "summer")}
        out["learned"][k] = {}
        for m in METHODS:
            seed, run = median_run(m, k)
            entry = {"seed": seed, "normal": flat(run["normal_policy"])}
            if k == 50:  # only penalty 50 has the summer switch (Experiment 2)
                entry["summer"] = flat(run["reset_policy"])
            out["learned"][k][m] = entry
    os.makedirs("../simulator", exist_ok=True)
    with open("../simulator/policies.json", "w") as f:
        json.dump(out, f)


if __name__ == "__main__":
    main()
