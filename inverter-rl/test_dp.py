import numpy as np

from baseline import rule_policy
from dp import value_iteration
from env import HOLD
from evaluate import evaluate


def test_dp_beats_simple_policies():
    # Same seed, so every policy faces the same power cuts
    _, normal = value_iteration("normal")
    _, summer = value_iteration("summer")
    always_hold = np.full_like(normal, HOLD)
    score = lambda policy, pattern: evaluate(policy, pattern, days=5000)[0]

    assert score(normal, "normal") > score(rule_policy(), "normal") > score(always_hold, "normal")
    assert score(summer, "summer") > score(rule_policy(), "summer")
    assert score(summer, "summer") > score(normal, "summer")  # adapting to summer pays
