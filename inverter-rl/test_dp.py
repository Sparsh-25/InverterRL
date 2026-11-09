from dp import value_iteration
from evaluate import evaluate


def test_dp_matches_project_numbers():
    # Long-run daily scores from project.md, section 7
    _, normal = value_iteration("normal")
    _, summer = value_iteration("summer")
    assert abs(evaluate(normal, "normal", days=5000)[0] - (-119.3)) < 1
    assert abs(evaluate(summer, "summer", days=5000)[0] - (-121.5)) < 1
    assert abs(evaluate(normal, "summer", days=5000)[0] - (-126.2)) < 1
