import pytest

from baseline import rule_action, rule_policy
from env import CHARGE, DISCHARGE, HOLD
from evaluate import evaluate


def test_rule_actions():
    assert rule_action(5, 12) == CHARGE  # solar hours
    assert rule_action(4, 19) == DISCHARGE
    assert rule_action(3, 19) == HOLD
    assert rule_action(10, 23) == HOLD


def test_score_adds_up():
    score, bill, blackout_hours = evaluate(rule_policy(), "normal", days=500)
    assert score == pytest.approx(-bill - 50 * blackout_hours)
