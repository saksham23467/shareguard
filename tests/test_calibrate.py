from shareguard.benchmarking.calibrate import clopper_pearson_upper, learn_then_test


def test_zero_events_rule_of_three():
    ub = clopper_pearson_upper(0, 3000)
    assert 0.0009 < ub < 0.0011  # ~3/n


def test_ltt_stops_in_order():
    confs = [0.95] * 5 + [0.5] * 50
    r = learn_then_test(confs, 3000, alpha=0.01)
    assert r["tau"] is not None and r["tau"] > 0.5
