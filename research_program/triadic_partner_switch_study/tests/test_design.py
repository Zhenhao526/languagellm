from itertools import product
import numpy as np

from research_program.triadic_partner_switch_study import design


def test_support_and_pair_balance():
    prepared = design.make_prepared()
    assert prepared['pair_counts'] == [1792, 1792, 1792]
    assert len(prepared['runs']) == 32
    assert len(set(r['condition'] for r in prepared['runs'])) == 8
    assert sum(s['world_count'] for s in prepared['partitions'].values()) == 5376 * (18 + 6) * 6


def test_payoff_transform_preserves_full_support():
    native = np.zeros((1, 24), dtype=np.float64)
    native[0, 0] = 1.
    native[0, 1] = .5
    transformed = design.payoff_rewards(native, 'all_or_nothing')
    expected = np.zeros((1, 24), dtype=np.float64)
    expected[0, 0] = 1.
    assert np.array_equal(transformed, expected)
    assert np.array_equal(design.payoff_rewards(native, 'partial'), native)


def test_condition_grid():
    parsed = {design.parse_condition(r['condition']) for r in design.make_prepared()['runs']}
    assert parsed == {(p, rule, live) for p in design.PAYOFFS for rule in design.RULES for live in design.LIVES}
