from model import risk_level


def test_risk_level_high_at_and_above_threshold():
    assert risk_level(0.5) == "high"
    assert risk_level(0.9) == "high"


def test_risk_level_medium_band():
    assert risk_level(0.25) == "medium"
    assert risk_level(0.49) == "medium"


def test_risk_level_low_below_threshold():
    assert risk_level(0.24) == "low"
    assert risk_level(0.0) == "low"
