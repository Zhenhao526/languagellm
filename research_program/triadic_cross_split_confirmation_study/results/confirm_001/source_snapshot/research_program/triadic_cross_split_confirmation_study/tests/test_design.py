from research_program.triadic_cross_split_confirmation_study import design


def test_bidirectional_edges():
    prepared = design.make_prepared()
    assert len(prepared['runs']) == 128
    assert prepared['seeds'][0] == 65101 and prepared['seeds'][-1] == 65116
    for direction in ('train_to_heldout', 'heldout_to_train'):
        split = prepared['cross_splits'][direction]
        assert len(split['edges']) == 576
        assert all(len(split['groups'][f'{actor}_{axis}']) == 96 for actor in range(3) for axis in (0, 1))
        assert all(len(split['groups'][f'{actor}_2']) == 0 for actor in range(3))


def test_payoff_and_condition_grid():
    import numpy as np
    native = np.zeros((1, 24), dtype=np.float64)
    native[0, 0] = 1.0
    native[0, 1] = 0.5
    expected = native.copy()
    expected[0, 1] = 0.0
    assert np.array_equal(design.payoff_rewards(native, 'all_or_nothing'), expected)
    assert {design.parse_condition(row['condition']) for row in design.make_prepared()['runs']} == {
        (p, rule, live) for p in design.PAYOFFS for rule in design.RULES for live in design.LIVES
    }
