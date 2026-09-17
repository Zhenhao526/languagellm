import numpy as np

from research_program.triadic_factorial_payoff_study import design


def test_support_and_splits():
    value = design.make_prepared()
    assert len(value['runs']) == 128
    assert value['split']['train_need_count'] == 2088
    assert value['split']['heldout_need_count'] == 3288
    assert value['partitions']['train']['world_count'] == 2088 * 18 * 6
    assert value['partitions']['heldout_resource']['world_count'] == 3288 * 18 * 6
    assert not value['need_response_cases']['heldout_resource']['empty_strata']
    assert all(all(value['factor_edge_groups']['heldout_resource'][key]) for key in ('heldout_changed_actor', 'seen_changed_actor'))


def test_payoff_transform():
    native = np.zeros((1, 24), dtype=np.float64); native[0, 0] = 1.; native[0, 1] = .5
    expected = native.copy(); expected[0, 1] = 0.
    assert np.array_equal(design.payoff_rewards(native, 'all_or_nothing'), expected)
    assert np.array_equal(design.payoff_rewards(native, 'partial'), native)


def test_condition_grid():
    parsed = {design.parse_condition(r['condition']) for r in design.make_prepared()['runs']}
    assert parsed == {(p, rule, live) for p in design.PAYOFFS for rule in design.RULES for live in design.LIVES}
