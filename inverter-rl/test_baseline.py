from baseline import rule_action, rule_policy
from env import CHARGE, DISCHARGE, HOLD
from evaluate import evaluate


def test_rule_actions():
    assert rule_action(5, 23) == CHARGE
    assert rule_action(5, 5) == CHARGE
    assert rule_action(4, 19) == DISCHARGE
    assert rule_action(3, 19) == HOLD
    assert rule_action(10, 12) == HOLD


def test_rule_matches_project_numbers():
    # project.md, section 7: rule scores about -121.7 (normal) and -130.4 (summer)
    score, bill, blackout_hours = evaluate(rule_policy(), "normal", days=5000)
    assert abs(score - (-121.7)) < 1
    assert score == -bill - 50 * blackout_hours
    assert abs(evaluate(rule_policy(), "summer", days=5000)[0] - (-130.4)) < 1
